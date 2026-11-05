use super::Objective;
use crate::error::{AlgorithmError, Result};
use keanu_graph::State;

/// How the next search direction is combined with the previous one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UpdateFormula {
    /// β = g'(g - g_prev) / g_prev'g_prev, restarting when β < 0 (default).
    PolakRibiere,
    /// β = g'g / g_prev'g_prev.
    FletcherReeves,
}

/// Nonlinear conjugate gradient ascent with an exact (bracketing + Brent) line search. The line search gives up on
/// steps that can't change the fitness: when the fitness is flat (to within rounding) at its first trial step, when
/// shorter steps would no longer move the latents, or when its bracket becomes flat.
///
/// It stops when the fitness changes by no more than `relative_threshold` times its magnitude, or by no more than
/// `absolute_threshold`, between iterations, or when `max_evaluations` fitness and gradient evaluations are used
/// (counting any made by the initial-condition check). It also stops, without converging, at the best point found
/// so far if the fitness becomes impossible or the gradient NaN or infinite, and at the first point past the
/// boundary threshold of a bounded latent in unconstrained space (see the [module documentation](super)).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ConjugateGradient {
    /// The maximum number of fitness and gradient evaluations (default unlimited).
    pub max_evaluations: usize,
    /// Default 1e-8.
    pub relative_threshold: f64,
    /// Default 1e-8.
    pub absolute_threshold: f64,
    /// Default Polak–Ribière.
    pub formula: UpdateFormula,
}

/// Setters for each setting, for use as `ConjugateGradient::default().with_...(...)`.
impl ConjugateGradient {
    /// Sets [`max_evaluations`](Self::max_evaluations).
    #[must_use]
    pub fn with_max_evaluations(mut self, max_evaluations: usize) -> Self {
        self.max_evaluations = max_evaluations;
        self
    }

    /// Sets [`relative_threshold`](Self::relative_threshold).
    #[must_use]
    pub fn with_relative_threshold(mut self, relative_threshold: f64) -> Self {
        self.relative_threshold = relative_threshold;
        self
    }

    /// Sets [`absolute_threshold`](Self::absolute_threshold).
    #[must_use]
    pub fn with_absolute_threshold(mut self, absolute_threshold: f64) -> Self {
        self.absolute_threshold = absolute_threshold;
        self
    }

    /// Sets [`formula`](Self::formula).
    #[must_use]
    pub fn with_formula(mut self, formula: UpdateFormula) -> Self {
        self.formula = formula;
        self
    }
}

impl Default for ConjugateGradient {
    fn default() -> Self {
        Self {
            max_evaluations: usize::MAX,
            relative_threshold: 1e-8,
            absolute_threshold: 1e-8,
            formula: UpdateFormula::PolakRibiere,
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// x + α d.
fn step_along(x: &[f64], alpha: f64, d: &[f64]) -> Vec<f64> {
    x.iter().zip(d).map(|(xi, di)| xi + alpha * di).collect()
}

impl ConjugateGradient {
    fn validate(&self) -> Result<()> {
        if self.max_evaluations == 0
            || (self.relative_threshold.is_nan() || self.relative_threshold <= 0.0)
            || (self.absolute_threshold.is_nan() || self.absolute_threshold <= 0.0)
        {
            return Err(AlgorithmError::InvalidSetting(
                "max_evaluations and the thresholds must be positive".into(),
            ));
        }
        Ok(())
    }

    fn converged(&self, previous: f64, current: f64) -> bool {
        let difference = (previous - current).abs();
        difference <= self.relative_threshold * previous.abs().max(current.abs())
            || difference <= self.absolute_threshold
    }

    /// Returns the final position (flattened), the fitness there and whether it converged. `known_start`, if
    /// given, is the fitness and gradient at `start` (from the initial-condition check), so they aren't evaluated
    /// again.
    pub(crate) fn maximize(
        &self,
        objective: &Objective<'_>,
        state: &mut State,
        start: Vec<f64>,
        known_start: Option<(f64, Vec<f64>)>,
    ) -> Result<(Vec<f64>, f64, bool)> {
        self.validate()?;
        let n = start.len();

        // an impossible (or NaN) fitness counts as -infinity, so it is never preferred
        let fitness = |point: &[f64], state: &mut State| -> Result<f64> {
            let value = objective.value_flat(state, point)?;
            Ok(if value.is_nan() {
                f64::NEG_INFINITY
            } else {
                value
            })
        };
        let exhausted = || objective.evaluations() >= self.max_evaluations;
        let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());

        let mut x = start;
        let (mut f, mut g) = match known_start {
            Some((f, g)) => (if f.is_nan() { f64::NEG_INFINITY } else { f }, g),
            None => (fitness(&x, state)?, objective.gradient_flat(state, &x)?),
        };
        // there is nowhere to go from an impossible point or along a NaN or infinite gradient
        if !f.is_finite() || !finite(&g) {
            return Ok((x, f, false));
        }
        let mut d = g.clone();
        let mut iteration = 0usize;
        loop {
            if exhausted() {
                return Ok((x, f, false));
            }
            if dot(&g, &g) == 0.0 {
                return Ok((x, f, true));
            }

            // the smallest step along d that still changes x (relative to its size, or absolutely near zero)
            let min_step = f64::EPSILON * norm(&x).max(1.0) / norm(&d);
            // once the evaluation budget is used up the line search sees every further step as impossible
            // (without evaluating it), so it settles on the best step it has already evaluated
            let (step, f_step) = line_search(
                |alpha, s: &mut State| {
                    if exhausted() {
                        return Ok(f64::NEG_INFINITY);
                    }
                    fitness(&step_along(&x, alpha, &d), s)
                },
                f,
                min_step,
                state,
            )?;
            // the line search evaluated φ at exactly this point, so f_step is the fitness at `next`
            let next = step_along(&x, step, &d);
            if exhausted() || !f_step.is_finite() {
                // stop at the best point found so far: the line search's step if it improved on x
                return Ok(if f_step.is_finite() && f_step > f {
                    (next, f_step, false)
                } else {
                    (x, f, false)
                });
            }
            iteration += 1;
            if objective.boundary(&next).is_some() {
                // heading for the boundary of a support, where the optimum is at infinity: the caller reports it
                return Ok((next, f_step, false));
            }
            if self.converged(f, f_step) {
                return Ok((next, f_step, true));
            }
            let g_next = objective.gradient_flat(state, &next)?;
            if !finite(&g_next) {
                return Ok(if f_step > f {
                    (next, f_step, false)
                } else {
                    (x, f, false)
                });
            }
            x = next;

            let denominator = dot(&g, &g);
            let beta = match self.formula {
                UpdateFormula::FletcherReeves => dot(&g_next, &g_next) / denominator,
                UpdateFormula::PolakRibiere => {
                    let diff: Vec<f64> = g_next.iter().zip(&g).map(|(a, b)| a - b).collect();
                    dot(&g_next, &diff) / denominator
                }
            };
            // restart with steepest ascent every n iterations, when β < 0, or when the direction stops ascending
            let restart = iteration % n.max(1) == 0 || (beta.is_nan() || beta <= 0.0);
            d = if restart {
                g_next.clone()
            } else {
                g_next
                    .iter()
                    .zip(&d)
                    .map(|(gi, di)| gi + beta * di)
                    .collect()
            };
            if dot(&d, &g_next) <= 0.0 {
                d = g_next.clone();
            }
            f = f_step;
            g = g_next;
        }
    }
}

/// The golden ratio, (1 + √5) / 2. Spelled out rather than `std::f64::consts::GOLDEN_RATIO`, which needs Rust 1.94,
/// to keep the minimum supported Rust version lower.
const GOLDEN: f64 = 1.618_033_988_749_895;
/// Relative precision of the step found by the line search (that of a typical line search for nonlinear CG:
/// finer precision costs evaluations without improving the optimum the outer iteration converges to).
const LINE_SEARCH_TOLERANCE: f64 = 1e-8;
/// The first trial step of the line search.
const FIRST_STEP: f64 = 1e-8;

/// Finds the step α >= 0 that maximizes φ(α), given φ(0) = `phi0`: grows a bracket from a small initial step
/// (with parabolic extrapolation) until φ falls, then narrows it with Brent's method (parabolic interpolation,
/// falling back to golden section steps). Steps shorter than `min_step` are not distinguished from 0: they no
/// longer change the point being optimized.
///
/// When φ is flat at the first trial step (the point is stationary to within rounding), or every step down to
/// `min_step` makes φ worse, the step is 0.
/// Returns the step and φ at it (φ is evaluated at every returned step other than 0).
fn line_search(
    mut phi: impl FnMut(f64, &mut State) -> Result<f64>,
    phi0: f64,
    min_step: f64,
    state: &mut State,
) -> Result<(f64, f64)> {
    // work with h(α) = -φ(α), minimised
    let mut h = |alpha: f64, s: &mut State| -> Result<f64> { Ok(-phi(alpha, s)?) };

    // bracket: a < b < c with h(b) <= h(a) and h(b) <= h(c)
    let (mut a, mut ha) = (0.0, -phi0);
    let mut b = FIRST_STEP.max(16.0 * min_step);
    let mut hb = h(b, state)?;
    if hb > ha {
        // even a tiny step makes things worse: shrink towards zero, while the step still moves the point
        for shrinks in 0.. {
            b *= 0.1;
            if b < min_step || shrinks == 60 {
                return Ok((0.0, phi0));
            }
            hb = h(b, state)?;
            if hb <= ha {
                break;
            }
        }
    }
    if hb == ha {
        // flat to within rounding: no step improves on the current point
        return Ok((0.0, phi0));
    }
    let mut c = b + GOLDEN * (b - a);
    let mut hc = h(c, state)?;
    let mut expansions = 0;
    while hc < hb && expansions < 200 {
        // parabolic extrapolation through (a, b, c), limited to 100 × the current step
        let r = (b - a) * (hb - hc);
        let q = (b - c) * (hb - ha);
        let denominator = 2.0 * (q - r);
        let limit = b + 100.0 * (c - b);
        let mut u = if denominator.abs() > 1e-300 {
            b - ((b - c) * q - (b - a) * r) / denominator
        } else {
            c + GOLDEN * (c - b)
        };
        let mut hu;
        if (b - u) * (u - c) > 0.0 {
            // u between b and c
            hu = h(u, state)?;
            if hu < hc {
                a = b;
                ha = hb;
                b = u;
                hb = hu;
                break;
            } else if hu > hb {
                c = u;
                break;
            }
            u = c + GOLDEN * (c - b);
            hu = h(u, state)?;
        } else if (c - u) * (u - limit) > 0.0 {
            // u between c and the limit
            hu = h(u, state)?;
            if hu < hc {
                b = c;
                hb = hc;
                c = u;
                hc = hu;
                u = c + GOLDEN * (c - b);
                hu = h(u, state)?;
            }
        } else if (u - limit) * (limit - c) >= 0.0 {
            u = limit;
            hu = h(u, state)?;
        } else {
            u = c + GOLDEN * (c - b);
            hu = h(u, state)?;
        }
        a = b;
        ha = hb;
        b = c;
        hb = hc;
        c = u;
        hc = hu;
        expansions += 1;
    }
    let _ = ha;
    let (lo, hi) = if a < c { (a, c) } else { (c, a) };
    let (best, h_best) = brent_minimize(&mut h, state, lo, hi, b, hb, min_step)?;
    // never return a step worse than the bracket's middle point
    Ok(if h_best <= hb {
        (best, -h_best)
    } else {
        (b, -hb)
    })
}

/// Brent's method: minimises h on [lo, hi] given an interior point `x` with value `hx`, to a relative precision
/// of `LINE_SEARCH_TOLERANCE` plus an absolute precision of `min_step`. Returns the best point and its value.
fn brent_minimize(
    h: &mut impl FnMut(f64, &mut State) -> Result<f64>,
    state: &mut State,
    mut lo: f64,
    mut hi: f64,
    x: f64,
    hx: f64,
    min_step: f64,
) -> Result<(f64, f64)> {
    let absolute_tolerance = min_step.max(1e-300);
    let ratio = 2.0 - GOLDEN; // 0.381966...
    let (mut x, mut w, mut v) = (x, x, x);
    let (mut fx, mut fw, mut fv) = (hx, hx, hx);
    let mut d: f64 = 0.0;
    let mut e: f64 = 0.0;
    for _ in 0..200 {
        let middle = 0.5 * (lo + hi);
        let tol1 = LINE_SEARCH_TOLERANCE * x.abs() + absolute_tolerance;
        let tol2 = 2.0 * tol1;
        if (x - middle).abs() <= tol2 - 0.5 * (hi - lo) {
            break;
        }
        // three distinct points with the same value: h is flat to within rounding around the best point, so no
        // further step can measurably improve on it
        if fx == fw && fw == fv && x != w && w != v && x != v {
            break;
        }
        let mut golden = true;
        if e.abs() > tol1 {
            // try a parabola through x, w, v
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let previous_e = e;
            if p.abs() < (0.5 * q * previous_e).abs() && p > q * (lo - x) && p < q * (hi - x) {
                e = d;
                d = p / q;
                let u = x + d;
                if u - lo < tol2 || hi - u < tol2 {
                    d = if middle >= x { tol1 } else { -tol1 };
                }
                golden = false;
            }
        }
        if golden {
            e = if x >= middle { lo - x } else { hi - x };
            d = ratio * e;
        }
        let u = if d.abs() >= tol1 {
            x + d
        } else {
            x + tol1.copysign(d)
        };
        let fu = h(u, state)?;
        if fu <= fx {
            if u >= x {
                lo = x;
            } else {
                hi = x;
            }
            (v, fv) = (w, fw);
            (w, fw) = (x, fx);
            (x, fx) = (u, fu);
        } else {
            if u < x {
                lo = u;
            } else {
                hi = u;
            }
            if fu <= fw || w == x {
                (v, fv) = (w, fw);
                (w, fw) = (u, fu);
            } else if fu <= fv || v == x || v == w {
                (v, fv) = (u, fu);
            }
        }
    }
    Ok((x, fx))
}
