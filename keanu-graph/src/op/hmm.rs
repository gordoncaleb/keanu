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

use super::Support;
use super::distribution::{DistributionKind, LogProbGradientParts, ParameterGradients, Univariate};
use super::mixture::{broadcast, gather, reduce_components, with_component_axis};
use super::wrappers;
use crate::error::{GraphError, Result};
use crate::value::{DType, Value};
use keanu_distributions::{DistributionError, HiddenMarkovChain, KeanuRng};
use keanu_tensor::{DoubleTensor, Shape, broadcast_shape};

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

fn invalid<T>(message: String) -> Result<T> {
    Err(GraphError::Distribution(
        DistributionError::InvalidParameter(message),
    ))
}

/// A hidden Markov model's emission parameters and chain, with the chain's probabilities checked.
struct Parameters<'p, 'v> {
    emission: &'p [&'v Value],
    chain: HiddenMarkovChain<'v>,
    count: usize,
}

impl<'p, 'v> Parameters<'p, 'v> {
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
pub(crate) fn log_prob(
    emission: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
    gradient: bool,
) -> Result<(f64, Option<LogProbGradientParts>)> {
    let p = Parameters::new(parameters)?;
    let count = p.count;
    let Some(&steps) = value.shape().first().filter(|&&t| t > 0) else {
        return invalid(format!(
            "the observations must have a time axis with at least one step, not shape {:?}",
            value.shape()
        ));
    };
    let x = with_component_axis(value)?;
    let (densities, parts) = emission.univariate(Univariate::Density, &x, p.emission, gradient)?;
    let full = with_states(value.shape(), count);
    if broadcast_shape(densities.shape(), &full)?.as_slice() != full.as_slice() {
        return invalid(format!(
            "the emission parameters don't broadcast to the observations' shape {:?} with {count} states",
            value.shape()
        ));
    }
    let densities = broadcast(&densities, &full)?;
    // each step's log density under each state, summed over the step's elements
    let inner = densities.len() / (steps * count);
    let mut per_step = vec![0.0; steps * count];
    for (i, d) in densities.iter().enumerate() {
        let (row, j) = (i / count, i % count);
        per_step[row / inner * count + j] += d;
    }
    let per_step = DoubleTensor::new(&[steps, count], per_step)?;
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

/// The support of a hidden Markov model: the emission kind's support with each bound reduced over the state axis
/// (the smallest lower bound, the largest upper bound), so that it holds every state's support.
pub(crate) fn support(emission: &DistributionKind, parameters: &[&Value]) -> Result<Support> {
    let p = Parameters::new(parameters)?;
    let support = emission.support(p.emission)?;
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
        return invalid("a hidden Markov model's observations need a time axis".into());
    };
    let full = with_states(shape, count);
    let draws = emission.sample(&full, p.emission, rng)?;
    if draws.shape() != full.as_slice() {
        return invalid(format!(
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
