//! A line search for the strong Wolfe conditions: bracketing followed by zoom with cubic interpolation (Nocedal &
//! Wright, *Numerical Optimization*, 2nd ed., Algorithms 3.5 and 3.6). It minimises φ(α) = f(x + α d) along a
//! descent direction d, and is independent of any particular optimizer.

use crate::error::Result;

/// The settings of the line search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LineSearch {
    /// The sufficient decrease (Armijo) constant: φ(α) ≤ φ(0) + c1 α φ'(0).
    pub(crate) c1: f64,
    /// The curvature constant: |φ'(α)| ≤ c2 |φ'(0)|.
    pub(crate) c2: f64,
    /// Steps, and brackets, shorter than this are given up on.
    pub(crate) min_step: f64,
    /// The maximum number of trial steps (evaluations of φ).
    pub(crate) max_iterations: usize,
}

impl Default for LineSearch {
    /// Stan's settings for L-BFGS: c1 = 1e-4, c2 = 0.9, at most 20 trial steps, none shorter than 1e-12.
    fn default() -> Self {
        Self {
            c1: 1e-4,
            c2: 0.9,
            min_step: 1e-12,
            max_iterations: 20,
        }
    }
}

/// A trial step: φ and φ' at α, with whatever else the caller computed there (such as the point and its
/// gradient).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Trial<T> {
    pub(crate) alpha: f64,
    pub(crate) value: f64,
    pub(crate) slope: f64,
    pub(crate) data: T,
}

/// How a line search ended.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Outcome<T> {
    /// A step satisfying the strong Wolfe conditions.
    Wolfe(Trial<T>),
    /// The trial steps ran out (or became too short) before the curvature condition was met: the best step found
    /// that satisfies sufficient decrease.
    Decrease(Trial<T>),
    /// No step found decreases φ sufficiently.
    Failed,
}

/// One end of a bracket. A trial step whose value or slope is not finite has an infinite value.
#[derive(Debug, Clone, Copy)]
struct End {
    alpha: f64,
    value: f64,
    slope: f64,
}

impl End {
    fn of<T>(t: &Trial<T>) -> Self {
        if t.value.is_finite() && t.slope.is_finite() {
            Self {
                alpha: t.alpha,
                value: t.value,
                slope: t.slope,
            }
        } else {
            Self {
                alpha: t.alpha,
                value: f64::INFINITY,
                slope: f64::NAN,
            }
        }
    }

    fn is_finite(&self) -> bool {
        self.value.is_finite()
    }
}

/// The minimiser of the cubic through (a, φ(a), φ'(a)) and (b, φ(b), φ'(b)) (Nocedal & Wright eq. 3.59); NaN if
/// the cubic has no minimiser.
fn cubic_minimiser(a: End, b: End) -> f64 {
    let d1 = a.slope + b.slope - 3.0 * (a.value - b.value) / (a.alpha - b.alpha);
    let discriminant = d1 * d1 - a.slope * b.slope;
    if discriminant < 0.0 {
        return f64::NAN;
    }
    let d2 = (b.alpha - a.alpha).signum() * discriminant.sqrt();
    b.alpha - (b.alpha - a.alpha) * (b.slope + d2 - d1) / (b.slope - a.slope + 2.0 * d2)
}

impl LineSearch {
    /// Searches along a descent direction given φ(0) = `value0` and φ'(0) = `slope0` < 0, starting with the step
    /// `initial`. `phi(α)` returns φ(α), φ'(α) and data to keep with the trial; a non-finite φ or φ' counts as
    /// too long a step, so the search shrinks it.
    pub(crate) fn search<T>(
        &self,
        value0: f64,
        slope0: f64,
        initial: f64,
        mut phi: impl FnMut(f64) -> Result<(f64, f64, T)>,
    ) -> Result<Outcome<T>> {
        debug_assert!(slope0 < 0.0, "not a descent direction");
        let mut evaluate = |alpha: f64| -> Result<Trial<T>> {
            let (value, slope, data) = phi(alpha)?;
            Ok(Trial {
                alpha,
                value,
                slope,
                data,
            })
        };
        let armijo = |t: &End| t.value <= value0 + self.c1 * t.alpha * slope0;
        let curvature = |t: &End| t.slope.abs() <= -self.c2 * slope0;

        // the previous step: always satisfies sufficient decrease, with its trial (none at α = 0)
        let mut previous = End {
            alpha: 0.0,
            value: value0,
            slope: slope0,
        };
        let mut previous_trial: Option<Trial<T>> = None;
        let mut alpha = initial;
        let mut iterations = 0;
        loop {
            if iterations >= self.max_iterations || alpha.is_nan() || alpha < self.min_step {
                return Ok(Self::finish(previous_trial));
            }
            let trial = evaluate(alpha)?;
            iterations += 1;
            let end = End::of(&trial);
            if !end.is_finite() || !armijo(&end) || (iterations > 1 && end.value >= previous.value)
            {
                return self.zoom(
                    previous,
                    previous_trial,
                    end,
                    value0,
                    slope0,
                    iterations,
                    &mut evaluate,
                );
            }
            if curvature(&end) {
                return Ok(Outcome::Wolfe(trial));
            }
            if end.slope >= 0.0 {
                return self.zoom(
                    end,
                    Some(trial),
                    previous,
                    value0,
                    slope0,
                    iterations,
                    &mut evaluate,
                );
            }
            // still descending: extrapolate, by cubic interpolation limited to 1.1–4 times the last increase
            let increase = alpha - previous.alpha;
            let (low, high) = (alpha + 1.1 * increase, alpha + 4.0 * increase);
            let next = cubic_minimiser(previous, end);
            let next = if next.is_nan() {
                alpha + 2.0 * increase
            } else {
                next.clamp(low, high)
            };
            previous = end;
            previous_trial = Some(trial);
            alpha = next;
        }
    }

    /// Algorithm 3.6: narrows the bracket between `lo` (satisfying sufficient decrease, with the lowest value so far)
    /// and `hi`, which may be on either side of it.
    #[allow(clippy::too_many_arguments)]
    fn zoom<T>(
        &self,
        mut lo: End,
        mut lo_trial: Option<Trial<T>>,
        mut hi: End,
        value0: f64,
        slope0: f64,
        mut iterations: usize,
        evaluate: &mut impl FnMut(f64) -> Result<Trial<T>>,
    ) -> Result<Outcome<T>> {
        loop {
            let width = hi.alpha - lo.alpha;
            if iterations >= self.max_iterations || width.is_nan() || width.abs() < self.min_step {
                return Ok(Self::finish(lo_trial));
            }
            // cubic interpolation, kept at least a tenth of the bracket away from its ends (so the bracket keeps
            // shrinking); bisection if hi is not finite or the cubic has no minimiser
            let middle = lo.alpha + 0.5 * width;
            let cubic = if hi.is_finite() {
                cubic_minimiser(lo, hi)
            } else {
                f64::NAN
            };
            let alpha = if cubic.is_nan() {
                middle
            } else {
                let (a, b) = (lo.alpha + 0.1 * width, hi.alpha - 0.1 * width);
                let (a, b) = if a <= b { (a, b) } else { (b, a) };
                cubic.clamp(a, b)
            };
            let trial = evaluate(alpha)?;
            iterations += 1;
            let end = End::of(&trial);
            if !end.is_finite()
                || end.value > value0 + self.c1 * alpha * slope0
                || end.value >= lo.value
            {
                hi = end;
            } else {
                if end.slope.abs() <= -self.c2 * slope0 {
                    return Ok(Outcome::Wolfe(trial));
                }
                if end.slope * (hi.alpha - lo.alpha) >= 0.0 {
                    hi = lo;
                }
                lo = end;
                lo_trial = Some(trial);
            }
        }
    }

    fn finish<T>(best: Option<Trial<T>>) -> Outcome<T> {
        best.map_or(Outcome::Failed, Outcome::Decrease)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// φ(α) = (α − m)² as a line search callback (φ(0) = m², φ'(0) = −2m), counting its evaluations.
    fn quadratic(m: f64, count: &mut usize) -> impl FnMut(f64) -> Result<(f64, f64, ())> + '_ {
        move |a| {
            *count += 1;
            Ok(((a - m).powi(2), 2.0 * (a - m), ()))
        }
    }

    fn accepted(outcome: Outcome<()>) -> (f64, bool) {
        match outcome {
            Outcome::Wolfe(t) => (t.alpha, true),
            Outcome::Decrease(t) => (t.alpha, false),
            Outcome::Failed => panic!("failed"),
        }
    }

    #[test]
    fn a_step_that_satisfies_the_wolfe_conditions_is_accepted_at_once() {
        let mut count = 0;
        let outcome = LineSearch::default()
            .search(1.0, -2.0, 1.0, quadratic(1.0, &mut count))
            .unwrap();
        assert_eq!(accepted(outcome), (1.0, true));
        assert_eq!(count, 1);
    }

    #[test]
    fn a_short_step_is_extrapolated() {
        let mut count = 0;
        let search = LineSearch::default();
        let outcome = search
            .search(100.0, -20.0, 1e-3, quadratic(10.0, &mut count))
            .unwrap();
        let (alpha, wolfe) = accepted(outcome);
        assert!(wolfe);
        // |φ'(α)| ≤ 0.9 × 20
        assert!((2.0 * (alpha - 10.0)).abs() <= 18.0, "{alpha}");
        assert!(count < 10, "{count}");
    }

    #[test]
    fn a_long_step_is_zoomed_by_cubic_interpolation() {
        let mut count = 0;
        let outcome = LineSearch::default()
            .search(1.0, -2.0, 100.0, quadratic(1.0, &mut count))
            .unwrap();
        let (alpha, wolfe) = accepted(outcome);
        assert!(wolfe);
        // the cubic through two points of a quadratic is the quadratic itself, so once the safeguard (a tenth of the
        // bracket from its ends) no longer binds, its minimiser is exact
        assert!((alpha - 1.0).abs() < 1e-9, "{alpha}");
        assert!(count <= 3, "{count}");
    }

    /// A non-finite φ (or φ') beyond α = 0.5 shrinks the step until it is finite.
    #[test]
    fn a_non_finite_trial_shrinks_the_step() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let outcome = LineSearch::default()
                .search(0.09, -0.6, 1.0, |a| {
                    Ok(if a < 0.5 {
                        ((a - 0.3).powi(2), 2.0 * (a - 0.3), ())
                    } else {
                        (bad, 1.0, ())
                    })
                })
                .unwrap();
            let (alpha, wolfe) = accepted(outcome);
            assert!(wolfe);
            assert!(alpha < 0.5, "{alpha}");
        }
        // a finite value with a NaN slope is also too far
        let outcome = LineSearch::default()
            .search(0.09, -0.6, 1.0, |a| {
                Ok(if a < 0.5 {
                    ((a - 0.3).powi(2), 2.0 * (a - 0.3), ())
                } else {
                    (0.0, f64::NAN, ())
                })
            })
            .unwrap();
        assert!(accepted(outcome).0 < 0.5);
    }

    #[test]
    fn no_decrease_fails() {
        // the slope claims descent but φ only increases: every step fails sufficient decrease
        let outcome = LineSearch::default()
            .search(0.0, -1.0, 1.0, |a| Ok((a, 1.0, ())))
            .unwrap();
        assert_eq!(outcome, Outcome::Failed);
        // and a function that is never finite away from 0 fails too, within the trial budget
        let mut count = 0;
        let outcome = LineSearch::default()
            .search(0.0, -1.0, 1.0, |_| {
                count += 1;
                Ok((f64::NAN, f64::NAN, ()))
            })
            .unwrap();
        assert_eq!(outcome, Outcome::Failed);
        assert!(count <= 20);
    }

    /// When the trial budget runs out after a step with sufficient decrease but not enough curvature, that step is
    /// returned.
    #[test]
    fn sufficient_decrease_without_curvature_is_returned() {
        let search = LineSearch {
            max_iterations: 1,
            ..LineSearch::default()
        };
        let mut count = 0;
        let outcome = search
            .search(100.0, -20.0, 1e-3, quadratic(10.0, &mut count))
            .unwrap();
        let (alpha, wolfe) = accepted(outcome);
        assert!(!wolfe);
        assert_eq!(alpha, 1e-3);
    }
}
