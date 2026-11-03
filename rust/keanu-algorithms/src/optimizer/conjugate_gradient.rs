use super::Objective;
use crate::error::{AlgorithmError, Result};
use keanu_graph::State;
use keanu_tensor::DoubleTensor;

/// How the next search direction is combined with the previous one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateFormula {
    /// β = g'(g - g_prev) / g_prev'g_prev, restarting when β < 0 (default).
    PolakRibiere,
    /// β = g'g / g_prev'g_prev.
    FletcherReeves,
}

/// Nonlinear conjugate gradient ascent with an exact (bracketing + golden section) line search.
///
/// It stops when the fitness changes by no more than `relative_threshold` times its magnitude, or by no more than
/// `absolute_threshold`, between iterations, or when `max_evaluations` fitness and gradient evaluations are used.
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// Flat views of a list of tensors as one vector, and back.
fn flatten(tensors: &[DoubleTensor]) -> Vec<f64> {
    tensors
        .iter()
        .flat_map(|t| t.as_slice().iter().copied())
        .collect()
}

fn unflatten(flat: &[f64], like: &[DoubleTensor]) -> Vec<DoubleTensor> {
    let mut offset = 0;
    like.iter()
        .map(|t| {
            let n = t.len();
            let tensor = DoubleTensor::new(t.shape(), flat[offset..offset + n].to_vec())
                .expect("same shape");
            offset += n;
            tensor
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
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

    /// Returns the final position and whether it converged.
    pub(crate) fn maximize(
        &self,
        objective: &Objective<'_>,
        state: &mut State,
        start: Vec<DoubleTensor>,
    ) -> Result<(Vec<DoubleTensor>, bool)> {
        self.validate()?;
        let shapes = start.clone();
        let mut x = flatten(&start);
        let n = x.len();

        let fitness = |point: &[f64], state: &mut State| -> Result<f64> {
            let value = objective.value(state, &unflatten(point, &shapes))?;
            Ok(if value.is_nan() {
                f64::NEG_INFINITY
            } else {
                value
            })
        };
        let gradient = |point: &[f64], state: &mut State| -> Result<Vec<f64>> {
            Ok(flatten(
                &objective.gradient(state, &unflatten(point, &shapes))?,
            ))
        };

        let mut f = fitness(&x, state)?;
        let mut g = gradient(&x, state)?;
        let mut d = g.clone();
        let mut iteration = 0usize;
        loop {
            if objective.evaluations() >= self.max_evaluations {
                return Ok((unflatten(&x, &start), false));
            }
            if dot(&g, &g) == 0.0 {
                return Ok((unflatten(&x, &start), true));
            }

            let step = line_search(
                |alpha, s: &mut State| {
                    let point: Vec<f64> =
                        x.iter().zip(&d).map(|(xi, di)| xi + alpha * di).collect();
                    fitness(&point, s)
                },
                state,
            )?;
            let next: Vec<f64> = x.iter().zip(&d).map(|(xi, di)| xi + step * di).collect();
            let f_next = fitness(&next, state)?;
            let g_next = gradient(&next, state)?;
            x = next;
            iteration += 1;

            if self.converged(f, f_next) {
                return Ok((unflatten(&x, &start), true));
            }

            let denominator = dot(&g, &g);
            let beta = match self.formula {
                UpdateFormula::FletcherReeves => dot(&g_next, &g_next) / denominator,
                UpdateFormula::PolakRibiere => {
                    let diff: Vec<f64> = g_next.iter().zip(&g).map(|(a, b)| a - b).collect();
                    dot(&g_next, &diff) / denominator
                }
            };
            // restart with steepest ascent every n iterations, when β < 0, or when the direction stops ascending
            let restart = iteration.is_multiple_of(n.max(1)) || (beta.is_nan() || beta <= 0.0);
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
            f = f_next;
            g = g_next;
        }
    }
}

const GOLDEN: f64 = std::f64::consts::GOLDEN_RATIO;
/// Relative precision of the step found by the line search (that of a typical line search for nonlinear CG:
/// finer precision costs evaluations without improving the optimum the outer iteration converges to).
const LINE_SEARCH_TOLERANCE: f64 = 1e-8;

/// Finds the step α >= 0 that maximizes φ(α): grows a bracket from a small initial step (with parabolic
/// extrapolation) until φ falls, then narrows it with Brent's method (parabolic interpolation, falling back to
/// golden section steps).
fn line_search(
    mut phi: impl FnMut(f64, &mut State) -> Result<f64>,
    state: &mut State,
) -> Result<f64> {
    // work with h(α) = -φ(α), minimised
    let mut h = |alpha: f64, s: &mut State| -> Result<f64> { Ok(-phi(alpha, s)?) };
    let h0 = h(0.0, state)?;

    // bracket: a < b < c with h(b) <= h(a) and h(b) <= h(c)
    let (mut a, mut ha) = (0.0, h0);
    let mut b = 1e-8;
    let mut hb = h(b, state)?;
    if hb > ha {
        // even a tiny step makes things worse: shrink towards zero
        for _ in 0..60 {
            b *= 0.1;
            hb = h(b, state)?;
            if hb <= ha {
                break;
            }
        }
        if hb > ha {
            return Ok(0.0);
        }
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
    let (best, h_best) = brent_minimize(&mut h, state, lo, hi, b, hb)?;
    // never return a step worse than the bracket's middle point
    Ok(if h_best <= hb { best } else { b })
}

/// Brent's method: minimises h on [lo, hi] given an interior point `x` with value `hx`, to a relative precision
/// of `LINE_SEARCH_TOLERANCE`. Returns the best point and its value.
fn brent_minimize(
    h: &mut impl FnMut(f64, &mut State) -> Result<f64>,
    state: &mut State,
    mut lo: f64,
    mut hi: f64,
    x: f64,
    hx: f64,
) -> Result<(f64, f64)> {
    let ratio = 2.0 - GOLDEN; // 0.381966...
    let (mut x, mut w, mut v) = (x, x, x);
    let (mut fx, mut fw, mut fv) = (hx, hx, hx);
    let mut d: f64 = 0.0;
    let mut e: f64 = 0.0;
    for _ in 0..200 {
        let middle = 0.5 * (lo + hi);
        let tol1 = LINE_SEARCH_TOLERANCE * x.abs() + 1e-300;
        let tol2 = 2.0 * tol1;
        if (x - middle).abs() <= tol2 - 0.5 * (hi - lo) {
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
