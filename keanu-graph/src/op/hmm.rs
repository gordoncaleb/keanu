//! Hidden Markov models: observations of a hidden discrete Markov chain, each drawn from an emission distribution
//! whose parameters depend on the hidden state.
//!
//! A [`DistributionKind::HiddenMarkovModel`] is a kind built around another one, with the layout described in the
//! wrappers module and used by mixtures: the node's parents are the emission kind's parameters, in its order, each
//! with a trailing state axis (of length K, or 1, or no axes at all, to share one value between the states),
//! followed by the model's own parameters, the `initial` state probabilities `[K]` and the `transition`
//! probabilities [K, K] (row i the probabilities of moving from state i to each state). The value is the series of
//! observations, shape [T, ...]: its first axis is time, and the step t observation `x[t]` may itself have axes
//! (several observations per step, all emitted by the same state).
//!
//! With `log p(x_t | j)` the emission kind's log density at the step t observation with the j-th state's
//! parameters (summed over the observation's elements), the log density is the hidden Markov chain's marginal
//! likelihood (see [`keanu_distributions::HiddenMarkovChain`]): the log-space forward algorithm over T steps and K
//! states. With γ_t(j) the posterior probability of state j at step t, the gradient with respect to the emission
//! parameters (and the value, for continuous emissions) is Σ_j γ_t(j) ∂ log p(x_t | j), like a mixture's with the
//! posterior marginals as responsibilities, and the gradient with respect to the probabilities is the backward
//! algorithm's.
//!
//! A draw simulates the chain forward (ancestral sampling) and draws each step's observation from its state's
//! emission distribution. The support is the emission kind's, widened to hold every state's.

use super::distribution::{DistributionKind, LogProbGradientParts, ParameterGradients, Univariate};
use super::distribution_hvp::{LogProbHvpParts, elementwise, reduce, spread_tangent};
use super::mixture::{gather, reduce_components, with_component_axis};
use super::wrappers;
use super::{Support, broadcast, invalid_parameter};
use crate::autodiff::{batched_shape, reduce_to};
use crate::error::Result;
use crate::value::{DType, Value};
use keanu_distributions::{HiddenMarkovChain, KeanuRng};
use keanu_tensor::special::log_sum_exp;
use keanu_tensor::{DoubleTensor, Shape, broadcast_shape};
use smallvec::SmallVec;

/// The number of parameters a hidden Markov model adds after its emission's: `initial` and `transition`.
pub(crate) const OWN_PARAMETERS: usize = 2;

/// The value shape of a hidden Markov model node with `parameters` (the emission's, then the initial and
/// transition probabilities, all already checked to have the right types) and the requested shape, which is
/// required: [T, ...], with T > 0 steps. Every emission parameter's batch shape (its shape without the state axis)
/// must broadcast to it without enlarging it.
pub(crate) fn value_shape(
    requested: Option<&[usize]>,
    parameters: &[(DType, &[usize])],
) -> std::result::Result<Shape, String> {
    let Some(requested) = requested.filter(|r| r.first().is_some_and(|&t| t > 0)) else {
        return Err(format!(
            "needs a shape [T, ...] with T > 0 time steps for its observations, not {requested:?}"
        ));
    };
    let [emission @ .., (_, initial), (_, transition)] = parameters else {
        return Err("a hidden Markov model needs initial and transition probabilities".into());
    };
    let &[count] = *initial else {
        return Err(format!(
            "the initial probabilities must be a vector [K], not of shape {initial:?}"
        ));
    };
    if count == 0 {
        return Err("there must be at least one state".into());
    }
    if *transition != [count, count] {
        return Err(format!(
            "the transition probabilities must be [{count}, {count}] for {count} states, not of shape \
             {transition:?}"
        ));
    }
    for (i, (_, parameter)) in emission.iter().enumerate() {
        let Some((&last, batch)) = parameter.split_last() else {
            continue;
        };
        if last != count && last != 1 {
            return Err(format!(
                "emission parameter {i} has shape {parameter:?}, whose last axis should be the state axis, of \
                 length {count} (the number of states) or 1"
            ));
        }
        let broadcast = broadcast_shape(requested, batch).ok();
        if broadcast.as_ref().map(|b| b.as_slice()) != Some(requested) {
            return Err(format!(
                "emission parameter {i} has shape {parameter:?}, whose batch shape {batch:?} does not broadcast \
                 to the observations' shape {requested:?}"
            ));
        }
    }
    Ok(Shape::from_slice(requested))
}

/// A hidden Markov model's emission parameters and chain, with the chain's probabilities checked.
struct Parameters<'p, 'v> {
    emission: &'p [&'v Value],
    chain: HiddenMarkovChain<'v>,
    count: usize,
}

impl<'p, 'v> Parameters<'p, 'v> {
    /// The parameters, an error unless the chain's shapes and probabilities are valid.
    fn new(parameters: &'p [&'v Value]) -> Result<Self> {
        let (emission, own) = wrappers::split(parameters, OWN_PARAMETERS)?;
        let chain = HiddenMarkovChain::new(own[0].f64()?, own[1].f64()?);
        let count = chain.states()?;
        Ok(Self {
            emission,
            chain,
            count,
        })
    }

    /// The parameters, an error unless the chain's shapes are valid ([K] and [K, K], K > 0); None if its
    /// probabilities are outside their valid region (negative, or a set not summing to 1).
    fn with_valid_shapes(parameters: &'p [&'v Value]) -> Result<Option<Self>> {
        let (_, own) = wrappers::split(parameters, OWN_PARAMETERS)?;
        let (initial, transition) = (own[0].f64()?, own[1].f64()?);
        let valid_shapes = match *initial.shape() {
            [k] => k > 0 && transition.shape() == [k, k],
            _ => false,
        };
        if !valid_shapes {
            // the error, with its message
            return Self::new(parameters).map(Some);
        }
        // with the shapes valid, the chain's only remaining check is of the probabilities
        Ok(Self::new(parameters).ok())
    }
}

/// `shape` with a trailing state axis of length `count`.
fn with_states(shape: &[usize], count: usize) -> Vec<usize> {
    let mut full = shape.to_vec();
    full.push(count);
    full
}

/// The log probability of a hidden Markov model node at `value` and, with `gradient`, its gradient parts (None if
/// the log probability is impossible): with respect to the value (for continuous emissions), to each emission
/// parameter (shaped like the value with the state axis) and to the initial and transition probabilities.
///
/// Initial or transition probabilities off the simplex are outside the parameters' valid region: the log
/// probability is -infinity there (with no gradient), so that a sampler proposing them rejects the proposal, as
/// for any other impossible state. Sampling such a model is an error.
pub(crate) fn log_prob(
    emission: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
    gradient: bool,
) -> Result<(f64, Option<LogProbGradientParts>)> {
    let Some(p) = Parameters::with_valid_shapes(parameters)? else {
        return Ok((f64::NEG_INFINITY, None));
    };
    let count = p.count;
    let Some(&steps) = value.shape().first().filter(|&&t| t > 0) else {
        return invalid_parameter(format!(
            "the observations must have a time axis with at least one step, not shape {:?}",
            value.shape()
        ));
    };
    let x = with_component_axis(value)?;
    let (densities, parts) = emission.univariate(Univariate::Density, &x, p.emission, gradient)?;
    let full = with_states(value.shape(), count);
    let per_step = per_step(&densities, &full, steps, count)?;
    // the elements of each step's observation
    let inner = full.iter().product::<usize>() / (steps * count);
    if !gradient {
        return Ok((p.chain.log_likelihood(&per_step)?, None));
    }
    let (log_prob, chain_gradient) = p.chain.log_likelihood_and_gradient(&per_step)?;
    let (Some(chain_gradient), Some((dx, dparameters))) = (chain_gradient, parts) else {
        return Ok((log_prob, None));
    };
    let posterior = chain_gradient.emission.as_slice();
    // γ_t(j) times an emission derivative, 0 where γ_t(j) is (whatever the derivative is there)
    let weighted = |d: &DoubleTensor| -> Result<Vec<f64>> {
        Ok(broadcast(d, &full)?
            .into_iter()
            .enumerate()
            .map(|(i, d)| {
                let g = posterior[i / count / inner * count + i % count];
                if g == 0.0 { 0.0 } else { g * d }
            })
            .collect())
    };
    let dx = dx
        .map(|dx| -> Result<DoubleTensor> {
            let summed = weighted(&dx)?
                .chunks(count)
                .map(|row| row.iter().sum())
                .collect();
            Ok(DoubleTensor::new(value.shape(), summed)?)
        })
        .transpose()?;
    let mut gradients: ParameterGradients = dparameters
        .iter()
        .map(|d| {
            d.as_ref()
                .map(|d| Ok(DoubleTensor::new(&full, weighted(d)?)?))
                .transpose()
        })
        .collect::<Result<_>>()?;
    gradients.push(Some(chain_gradient.initial));
    gradients.push(Some(chain_gradient.transition));
    Ok((log_prob, Some((dx, gradients))))
}

/// Each step's log density under each state, [T, K], from the emission's element-wise log densities (which must
/// broadcast to `full`, the value's shape with the state axis), summed over the step's elements.
fn per_step(
    densities: &DoubleTensor,
    full: &[usize],
    steps: usize,
    count: usize,
) -> Result<DoubleTensor> {
    if broadcast_shape(densities.shape(), full)?.as_slice() != full {
        return invalid_parameter(format!(
            "the emission parameters don't broadcast to the observations' shape {:?} with {count} states",
            &full[..full.len() - 1]
        ));
    }
    let densities = broadcast(densities, full)?;
    let inner = densities.len() / (steps * count);
    let mut per_step = vec![0.0; steps * count];
    for (i, d) in densities.iter().enumerate() {
        let (row, j) = (i / count, i % count);
        per_step[row / inner * count + j] += d;
    }
    Ok(DoubleTensor::new(&[steps, count], per_step)?)
}

impl DistributionKind {
    /// For a [hidden Markov model](DistributionKind::HiddenMarkovModel) at `value` with the parameters' values
    /// (`parameters`, in the node's order): its chain, and the log probability of each step's observation under
    /// each state, [T, K] (the emission's log density with that state's parameters, summed over the step's
    /// elements). These are the inputs of [`HiddenMarkovChain`]'s decoding methods
    /// ([`viterbi`](HiddenMarkovChain::viterbi), [`posterior_marginals`](HiddenMarkovChain::posterior_marginals)
    /// and [`sample_posterior_states`](HiddenMarkovChain::sample_posterior_states)).
    ///
    /// # Errors
    ///
    /// [`GraphError::InvalidNode`](crate::GraphError::InvalidNode) for any other kind, and an error if the
    /// chain's probabilities or the shapes are invalid.
    pub fn hidden_markov_chain(
        &self,
        value: &Value,
        parameters: &[&Value],
    ) -> Result<(HiddenMarkovChain<'static>, DoubleTensor)> {
        let DistributionKind::HiddenMarkovModel(emission) = self else {
            return Err(crate::GraphError::InvalidNode {
                op: self.name(),
                message: "is not a hidden Markov model".into(),
            });
        };
        let p = Parameters::new(parameters)?;
        let Some(&steps) = value.shape().first().filter(|&&t| t > 0) else {
            return invalid_parameter(format!(
                "the observations must have a time axis with at least one step, not shape {:?}",
                value.shape()
            ));
        };
        let x = with_component_axis(value)?;
        let (densities, _) = emission.univariate(Univariate::Density, &x, p.emission, false)?;
        let full = with_states(value.shape(), p.count);
        let emission = per_step(&densities, &full, steps, p.count)?;
        let chain = HiddenMarkovChain::new(
            p.chain.initial.clone().into_owned(),
            p.chain.transition.clone().into_owned(),
        );
        Ok((chain, emission))
    }
}

/// The support of a hidden Markov model: the emission kind's support with each bound reduced over the state axis
/// (the smallest lower bound, the largest upper bound), so that it holds every state's support. The chain's
/// probabilities don't bound the support, so they aren't checked here (the log probability and sampling check
/// them): probabilities that are latent can be perturbed off the simplex without the support appearing to move.
pub(crate) fn support(emission: &DistributionKind, parameters: &[&Value]) -> Result<Support> {
    let (emission_parameters, _) = wrappers::split(parameters, OWN_PARAMETERS)?;
    let support = emission.support(emission_parameters)?;
    let Some((lower, upper)) = wrappers::support_bounds(support.clone()) else {
        return Ok(support);
    };
    Ok(wrappers::support_between(
        reduce_components(&lower, f64::min)?,
        reduce_components(&upper, f64::max)?,
    ))
}

/// Draws a series of `shape` ([T, ...]): a path of states by ancestral simulation of the chain, then each step's
/// observation from its state's emission distribution (from a draw of every state's at every element, of which
/// each step keeps its state's).
pub(crate) fn sample(
    emission: &DistributionKind,
    shape: &[usize],
    parameters: &[&Value],
    rng: &mut KeanuRng,
) -> Result<Value> {
    let p = Parameters::new(parameters)?;
    let count = p.count;
    let Some(&steps) = shape.first() else {
        return invalid_parameter("a hidden Markov model's observations need a time axis".into());
    };
    let full = with_states(shape, count);
    let draws = emission.sample(&full, p.emission, rng)?;
    if draws.shape() != full.as_slice() {
        return invalid_parameter(format!(
            "the emission parameters don't broadcast to the observations' shape {shape:?} with {count} states"
        ));
    }
    let states = p.chain.sample_states(steps, rng)?;
    let inner: usize = shape[1..].iter().product();
    let picks: Vec<usize> = (0..steps * inner)
        .map(|i| i * count + states[i / inner.max(1)])
        .collect();
    gather(&draws, shape, &picks)
}

/// `exp(log_weight)` for a term of a log-sum-exp, 0 where the term or the total is −∞ (so that a zero weight
/// never multiplies an infinite tangent).
fn weight(term: f64, total: f64) -> f64 {
    if term == f64::NEG_INFINITY || total == f64::NEG_INFINITY {
        0.0
    } else {
        (term - total).exp()
    }
}

/// `dp / p`, the tangent of `ln p`, taken as 0 where p is 0 (where every term it enters has zero weight).
fn log_tangent(dp: f64, p: f64) -> f64 {
    if p == 0.0 { 0.0 } else { dp / p }
}

/// The Hessian-vector products of a hidden Markov model's log density along `b` directions (see
/// [`DistributionKind::log_prob_hvp`]), by forward-mode differentiation of the forward-backward algorithm.
///
/// The log density is L(e, π, A) with e_tj the emission log density of step t under state j (summed over the
/// step's elements), and its gradient is (γ_tj, ∂L/∂π_j, ∂L/∂A_ij) with γ the posterior state marginals. Along a
/// direction, the forward variables move by α̇_0j = π̇_j/π_j + ė_0j and
/// α̇_tj = Σ_i s_tij (α̇_{t−1,i} + Ȧ_ij/A_ij) + ė_tj (s the softmax weights of the forward recursion), the backward
/// variables likewise, and L̇ = Σ_j softmax(α_{T−1})_j α̇_{T−1,j}; then γ̇ = γ (α̇ + β̇ − L̇), and the derivatives
/// with respect to π and A, each the exponential of a sum of forward, emission and backward terms minus L, move by
/// themselves times that sum's tangent. The emission parameters' (and the value's) products are, as for a mixture,
/// Σ_j γ_tj ∇²e_tj v + ∇e_tj γ̇_tj, with the emission's own element-wise rule.
pub(crate) fn log_prob_hvp(
    emission: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
    value_tangent: Option<&DoubleTensor>,
    parameter_tangents: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<LogProbHvpParts> {
    let p = Parameters::new(parameters)?;
    let k = p.count;
    let m = p.emission.len();
    let Some(&steps) = value.shape().first().filter(|&&t| t > 0) else {
        return invalid_parameter(format!(
            "the observations must have a time axis with at least one step, not shape {:?}",
            value.shape()
        ));
    };
    let x = with_component_axis(value)?;
    let full = with_states(value.shape(), k);
    let value_tangent = value_tangent
        .map(|t| -> Result<DoubleTensor> {
            let mut shape: Vec<isize> = t.shape().iter().map(|&d| d as isize).collect();
            shape.push(1);
            Ok(t.reshape(&shape)?)
        })
        .transpose()?;
    let tangents: Vec<Option<Vec<f64>>> = std::iter::once(value_tangent.as_ref())
        .chain(parameter_tangents[..m].iter().copied())
        .map(|t| spread_tangent(t, b, &full))
        .collect::<Result<_>>()?;
    let parts = elementwise(emission, &x, p.emission, &tangents, b, &full)?;
    let len = parts.values.len();
    let inner = len / (steps * k);
    // the (step, state) cell of each element
    let cell = |i: usize| (i / k) / inner * k + i % k;
    let mut e = vec![0.0; steps * k];
    for (i, v) in parts.values.iter().enumerate() {
        e[cell(i)] += v;
    }
    let pi = p.chain.initial.as_slice();
    let a = p.chain.transition.as_slice();
    let log_a: Vec<f64> = a.iter().map(|p| p.ln()).collect();
    let pi_tangent = spread_tangent(parameter_tangents[m], b, &[k])?;
    let a_tangent = spread_tangent(parameter_tangents[m + 1], b, &[k, k])?;

    // the forward and backward variables, and the log density
    let mut alpha = vec![0.0; steps * k];
    for j in 0..k {
        alpha[j] = pi[j].ln() + e[j];
    }
    for t in 1..steps {
        for j in 0..k {
            alpha[t * k + j] =
                log_sum_exp((0..k).map(|i| alpha[(t - 1) * k + i] + log_a[i * k + j]))
                    + e[t * k + j];
        }
    }
    let last = &alpha[(steps - 1) * k..];
    let total = log_sum_exp(last.iter().copied());
    if !total.is_finite() {
        // an impossible point, whose gradient is undefined too
        return Ok((None, smallvec::smallvec![None; parameters.len()]));
    }
    let mut beta = vec![0.0; steps * k];
    for t in (0..steps - 1).rev() {
        for i in 0..k {
            beta[t * k + i] = log_sum_exp(
                (0..k).map(|j| log_a[i * k + j] + e[(t + 1) * k + j] + beta[(t + 1) * k + j]),
            );
        }
    }
    let posterior: Vec<f64> = (0..steps * k)
        .map(|c| weight(alpha[c] + beta[c], total))
        .collect();

    let mut outputs: Vec<Option<Vec<f64>>> = parts
        .hvps
        .iter()
        .map(|h| h.as_ref().map(|_| vec![0.0; b * len]))
        .collect();
    let mut pi_output = vec![0.0; b * k];
    let mut a_output = vec![0.0; b * k * k];
    let (mut e_dot, mut alpha_dot, mut beta_dot) = (
        vec![0.0; steps * k],
        vec![0.0; steps * k],
        vec![0.0; steps * k],
    );
    for s in 0..b {
        let dpi = |j: usize| pi_tangent.as_ref().map_or(0.0, |t| t[s * k + j]);
        let da = |ij: usize| {
            a_tangent
                .as_ref()
                .map_or(0.0, |t| log_tangent(t[s * k * k + ij], a[ij]))
        };
        e_dot.iter_mut().for_each(|d| *d = 0.0);
        for i in 0..len {
            let mut d = 0.0;
            for (g, t) in parts.gradients.iter().zip(&tangents) {
                if let (Some(g), Some(t)) = (g, t) {
                    d += g[i] * t[s * len + i];
                }
            }
            e_dot[cell(i)] += d;
        }
        for j in 0..k {
            alpha_dot[j] = log_tangent(dpi(j), pi[j]) + e_dot[j];
        }
        for t in 1..steps {
            for j in 0..k {
                let lse = alpha[t * k + j] - e[t * k + j];
                let mut d = e_dot[t * k + j];
                for i in 0..k {
                    let w = weight(alpha[(t - 1) * k + i] + log_a[i * k + j], lse);
                    if w != 0.0 {
                        d += w * (alpha_dot[(t - 1) * k + i] + da(i * k + j));
                    }
                }
                alpha_dot[t * k + j] = d;
            }
        }
        let total_dot: f64 = (0..k)
            .map(|j| {
                let w = weight(last[j], total);
                if w == 0.0 {
                    0.0
                } else {
                    w * alpha_dot[(steps - 1) * k + j]
                }
            })
            .sum();
        beta_dot[(steps - 1) * k..]
            .iter_mut()
            .for_each(|d| *d = 0.0);
        for t in (0..steps - 1).rev() {
            for i in 0..k {
                let mut d = 0.0;
                for j in 0..k {
                    let next = (t + 1) * k + j;
                    let w = weight(log_a[i * k + j] + e[next] + beta[next], beta[t * k + i]);
                    if w != 0.0 {
                        d += w * (da(i * k + j) + e_dot[next] + beta_dot[next]);
                    }
                }
                beta_dot[t * k + i] = d;
            }
        }
        // ∂L/∂π_j = exp(e_0j + β_0j − L)
        for j in 0..k {
            let g = weight(e[j] + beta[j], total);
            if g != 0.0 {
                pi_output[s * k + j] = g * (e_dot[j] + beta_dot[j] - total_dot);
            }
        }
        // ∂L/∂A_ij = Σ_t exp(α_ti + e_{t+1,j} + β_{t+1,j} − L)
        for t in 0..steps - 1 {
            for i in 0..k {
                for j in 0..k {
                    let next = (t + 1) * k + j;
                    let xi = weight(alpha[t * k + i] + e[next] + beta[next], total);
                    if xi != 0.0 {
                        a_output[s * k * k + i * k + j] +=
                            xi * (alpha_dot[t * k + i] + e_dot[next] + beta_dot[next] - total_dot);
                    }
                }
            }
        }
        for i in 0..len {
            let c = cell(i);
            let gamma = posterior[c];
            if gamma == 0.0 {
                continue;
            }
            let gamma_dot = gamma * (alpha_dot[c] + beta_dot[c] - total_dot);
            for (u, out) in outputs.iter_mut().enumerate() {
                let (Some(out), Some(g), Some(h)) = (out, &parts.gradients[u], &parts.hvps[u])
                else {
                    continue;
                };
                let j = s * len + i;
                out[j] = gamma * h[j] + g[i] * gamma_dot;
            }
        }
    }
    let mut outputs = outputs.into_iter();
    // the value's products, summed over the state axis
    let value_hvp = outputs
        .next()
        .flatten()
        .map(|h| -> Result<DoubleTensor> {
            let summed: Vec<f64> = h.chunks(k).map(|row| row.iter().sum()).collect();
            reduce_to(
                DoubleTensor::new(&batched_shape(b, value.shape()), summed)?,
                value.shape(),
            )
        })
        .transpose()?;
    let mut hvps: SmallVec<[Option<DoubleTensor>; 4]> = SmallVec::new();
    for (h, parameter) in outputs.zip(p.emission) {
        hvps.push(reduce(h, b, &full, parameter.shape())?);
    }
    hvps.push(Some(DoubleTensor::new(&[b, k], pi_output)?));
    hvps.push(Some(DoubleTensor::new(&[b, k, k], a_output)?));
    Ok((value_hvp, hvps))
}
