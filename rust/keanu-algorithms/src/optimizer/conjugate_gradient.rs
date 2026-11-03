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

/// Finds the step α >= 0 that maximizes φ(α): grows a bracket from a small initial step until φ falls, then
/// narrows it with golden section search.
fn line_search(
    mut phi: impl FnMut(f64, &mut State) -> Result<f64>,
    state: &mut State,
) -> Result<f64> {
    let f0 = phi(0.0, state)?;

    // bracket: find a < b < c with φ(b) >= φ(a) and φ(b) >= φ(c)
    let (mut a, mut fa) = (0.0, f0);
    let mut b = 1e-8;
    let mut fb = phi(b, state)?;
    if fb < fa {
        // even a tiny step decreases φ: shrink towards zero
        for _ in 0..60 {
            b *= 0.1;
            fb = phi(b, state)?;
            if fb >= fa {
                break;
            }
        }
        if fb < fa {
            return Ok(0.0);
        }
    }
    let mut c = b + GOLDEN * (b - a);
    let mut fc = phi(c, state)?;
    let mut expansions = 0;
    while fc > fb && expansions < 200 {
        a = b;
        fa = fb;
        b = c;
        fb = fc;
        c = b + GOLDEN * (b - a);
        fc = phi(c, state)?;
        expansions += 1;
    }
    let _ = fa;

    // golden section search on [a, c]
    let ratio = 2.0 - GOLDEN; // 0.381966...
    let (mut lo, mut hi) = (a, c);
    let mut x1 = lo + ratio * (hi - lo);
    let mut x2 = hi - ratio * (hi - lo);
    let mut f1 = phi(x1, state)?;
    let mut f2 = phi(x2, state)?;
    for _ in 0..200 {
        if (hi - lo).abs() <= 1e-10 * (x1.abs() + x2.abs()) + 1e-300 {
            break;
        }
        if f1 >= f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = lo + ratio * (hi - lo);
            f1 = phi(x1, state)?;
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = hi - ratio * (hi - lo);
            f2 = phi(x2, state)?;
        }
    }
    let best = if f1 >= f2 { x1 } else { x2 };
    // never return a step worse than the bracket's middle point
    Ok(if phi(best, state)? >= fb { best } else { b })
}
