//! Smoothing, decoding and forecasting of time-series nodes over posterior draws.
//!
//! A [hidden Markov model](keanu_graph::op::DistributionKind::HiddenMarkovModel) or
//! [linear-Gaussian state-space](keanu_graph::op::DistributionKind::LinearGaussianStateSpace) node's log density
//! sums or integrates its hidden states out, so samplers draw only its parameters. The hidden states, and the
//! future, are recovered afterwards from each draw: the node's parameters are evaluated at the draw (its
//! deterministic parents recomputed from the draw's latents), and the exact computations of
//! [`keanu_distributions::HiddenMarkovChain`] and [`keanu_distributions::LinearGaussianStateSpace`] run on them.
//! Repeating them over the draws gives the posterior with the parameters' uncertainty included.
//!
//! - [`StateSpacePosterior`]: the Kalman (Rauch–Tung–Striebel) smoother, forward-filtering backward-sampling
//!   draws of the states, and forecasts (moments, or simulated paths) for a time-invariant model or with future
//!   matrices given per draw.
//! - [`HiddenMarkovPosterior`]: Viterbi paths, posterior state marginals and posterior state paths.
//!
//! ```
//! use keanu_algorithms::NetworkSamples;
//! use keanu_algorithms::timeseries::StateSpacePosterior;
//! use keanu_graph::{GraphBuilder, Model};
//! use keanu_tensor::DoubleTensor;
//!
//! // a local level model with an unknown level noise scale
//! let b = GraphBuilder::new();
//! let one = || DoubleTensor::new(&[1, 1], vec![1.0]).unwrap();
//! let scale = b.half_gaussian(1.0);
//! let y = b.linear_gaussian_state_space(
//!     4,
//!     DoubleTensor::vector(vec![0.0]),
//!     DoubleTensor::new(&[1, 1], vec![10.0])?,
//!     one(),
//!     (scale * scale).reshape(&[1, 1]),
//!     one(),
//!     DoubleTensor::new(&[1, 1], vec![0.25])?,
//! );
//! b.observe(y, DoubleTensor::new(&[4, 1], vec![0.3, 0.9, 1.4, 1.1])?);
//! let (scale, y) = (scale.id(), y.id());
//! let (graph, mut state) = b.build()?;
//! let model = Model::new(graph, &state)?;
//!
//! // two "draws" of the scale
//! let mut samples = NetworkSamples::new(model.graph(), &[scale])?;
//! for value in [0.5, 1.0] {
//!     state.set_and_cascade(model.graph(), scale, DoubleTensor::scalar(value))?;
//!     samples.record(&state, model.log_prob(&state)?)?;
//! }
//! let posterior = StateSpacePosterior::new(&model, &state, y, &samples)?;
//! let smoothed = posterior.smooth()?;
//! assert_eq!(smoothed.len(), 2);
//! assert_eq!(smoothed[0].means.shape(), &[4, 1]);
//! let forecasts = posterior.forecast(3)?;
//! // the forecast variance grows by the level variance at each step
//! let v = forecasts[1].state_covariances.as_slice();
//! assert!((v[1] - v[0] - 1.0).abs() < 1e-12);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{AlgorithmError, Result};
use crate::samples::{ChainSamples, NetworkSamples};
use crate::support::{check_nodes, latent_ancestors};
use keanu_distributions::{
    FutureStep, HiddenMarkovChain, KalmanSmoother, KeanuRng, LinearGaussianStateSpace,
    StateSpaceForecast, StateSpacePath,
};
use keanu_graph::op::{DistributionKind, Op};
use keanu_graph::{Model, NodeId, State, Value};
use keanu_tensor::DoubleTensor;

/// The parameters' values and the value of `node` at each draw of each of `samples`, in order.
///
/// The draws are set in a copy of `state` and the node's parents evaluated from them, so every latent the
/// parameters depend on must be recorded (as must the node itself unless it is observed).
fn node_draws(
    model: &Model,
    state: &State,
    node: NodeId,
    samples: &[&NetworkSamples],
) -> Result<Vec<(Vec<Value>, Value)>> {
    let graph = model.graph();
    check_nodes(graph, &[node])?;
    let mut targets = graph[node].parents().to_vec();
    targets.push(node);
    Ok(
        values_at_draws(model, state, &targets, &graph.describe(node), samples)?
            .into_iter()
            .map(|mut values| {
                let value = values.pop().expect("the node is a target");
                (values, value)
            })
            .collect(),
    )
}

/// The values of `targets` at each draw of each of `samples`, in order: each draw is set in a copy of `state` and
/// the targets evaluated from it, so every latent they depend on (a latent target itself among them) must be
/// recorded; an error naming `what` and the missing latents otherwise.
pub(crate) fn values_at_draws(
    model: &Model,
    state: &State,
    targets: &[NodeId],
    what: &str,
    samples: &[&NetworkSamples],
) -> Result<Vec<Vec<Value>>> {
    let graph = model.graph();
    check_nodes(graph, targets)?;
    let mut sampled = vec![false; graph.len()];
    for s in samples {
        check_nodes(graph, s.nodes())?;
        for id in s.nodes() {
            sampled[id.index()] = true;
        }
    }
    let missing: Vec<String> = latent_ancestors(graph, state, targets)
        .into_iter()
        .filter(|id| !sampled[id.index()])
        .map(|id| graph.describe(id))
        .collect();
    if !missing.is_empty() {
        return Err(AlgorithmError::InvalidArgument(format!(
            "{what} depends on latents without samples: {}",
            missing.join(", ")
        )));
    }
    let mut work = state.clone();
    // every latent the targets need has a value from the samples, so evaluating them samples nothing
    let mut rng = KeanuRng::seed_from_u64(0);
    let mut draws = Vec::new();
    for s in samples {
        for sample in s.iter() {
            for &id in s.nodes() {
                if let Some(value) = sample.get(id) {
                    work.set(graph, id, value)?;
                }
            }
            work.eval(graph, targets, &mut rng)?;
            let values = targets
                .iter()
                .map(|&id| {
                    work.value(id).cloned().ok_or_else(|| {
                        AlgorithmError::InvalidArgument(format!(
                            "{} has no value",
                            graph.describe(id)
                        ))
                    })
                })
                .collect::<Result<Vec<Value>>>()?;
            draws.push(values);
        }
    }
    Ok(draws)
}

/// The node's distribution kind, an error naming `expected` unless it is one.
fn kind_of<'m>(
    model: &'m Model,
    node: NodeId,
    expected: &str,
    is: impl Fn(&DistributionKind) -> bool,
) -> Result<&'m DistributionKind> {
    let graph = model.graph();
    check_nodes(graph, &[node])?;
    match graph[node].op() {
        Op::Distribution(d) if is(&d.kind) => Ok(&d.kind),
        _ => Err(AlgorithmError::InvalidArgument(format!(
            "{} is not {expected}",
            graph.describe(node)
        ))),
    }
}

/// A value's f64 tensor, an error for another type.
pub(crate) fn f64_value(value: &Value) -> Result<&DoubleTensor> {
    value.as_f64().ok_or_else(|| {
        AlgorithmError::InvalidArgument(format!(
            "expected an f64 value, not a {:?} one",
            value.dtype()
        ))
    })
}

/// Every chain's samples, in order.
pub(crate) fn chains(samples: &ChainSamples) -> Vec<&NetworkSamples> {
    samples.chains().iter().collect()
}

/// A linear-Gaussian state-space node at each posterior draw: the model's parameters and the observations, from
/// which the smoother, posterior state draws and forecasts are computed per draw. See the
/// [module documentation](self).
#[derive(Debug, Clone, PartialEq)]
pub struct StateSpacePosterior {
    /// Each draw's six parameters, in the order of [`LinearGaussianStateSpace::new`], and observations.
    draws: Vec<([DoubleTensor; 6], DoubleTensor)>,
}

impl StateSpacePosterior {
    /// The model of the linear-Gaussian state-space `node` at each draw in `samples`: its parameters evaluated from
    /// the draw (set in a copy of `state`, a state of the model's graph), and its observations (its observed value,
    /// or its value in the draw).
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidArgument`] if `node` is not a linear-Gaussian state-space node or `samples` doesn't
    /// record every latent its parameters depend on (naming the missing ones), an unknown-node error if a sampled
    /// node isn't in the graph, and any error evaluating the parameters.
    pub fn new(
        model: &Model,
        state: &State,
        node: NodeId,
        samples: &NetworkSamples,
    ) -> Result<Self> {
        Self::from_samples(model, state, node, &[samples])
    }

    /// [`Self::new`] over every chain's draws, chain by chain.
    ///
    /// # Errors
    ///
    /// As for [`Self::new`].
    pub fn from_chains(
        model: &Model,
        state: &State,
        node: NodeId,
        samples: &ChainSamples,
    ) -> Result<Self> {
        Self::from_samples(model, state, node, &chains(samples))
    }

    fn from_samples(
        model: &Model,
        state: &State,
        node: NodeId,
        samples: &[&NetworkSamples],
    ) -> Result<Self> {
        kind_of(model, node, "a linear Gaussian state space model", |k| {
            matches!(k, DistributionKind::LinearGaussianStateSpace)
        })?;
        let draws = node_draws(model, state, node, samples)?
            .into_iter()
            .map(|(parameters, value)| {
                let tensors: Vec<DoubleTensor> = parameters
                    .iter()
                    .map(|p| f64_value(p).cloned())
                    .collect::<Result<_>>()?;
                let tensors: [DoubleTensor; 6] = tensors.try_into().map_err(|_| {
                    AlgorithmError::InvalidArgument(
                        "a linear Gaussian state space node needs six parameters".into(),
                    )
                })?;
                Ok((tensors, f64_value(&value)?.clone()))
            })
            .collect::<Result<_>>()?;
        Ok(Self { draws })
    }

    /// The number of draws.
    pub fn len(&self) -> usize {
        self.draws.len()
    }

    /// Whether there are no draws.
    pub fn is_empty(&self) -> bool {
        self.draws.is_empty()
    }

    /// The state-space model at draw `index` (None past the last draw).
    pub fn model(&self, index: usize) -> Option<LinearGaussianStateSpace<'_>> {
        self.draws.get(index).map(|(p, _)| model_of(p))
    }

    /// The observations at draw `index`, [T, m] (None past the last draw).
    pub fn observations(&self, index: usize) -> Option<&DoubleTensor> {
        self.draws.get(index).map(|(_, x)| x)
    }

    /// Runs `f` on each draw's model and observations.
    fn each<T>(
        &self,
        mut f: impl FnMut(usize, &LinearGaussianStateSpace<'_>, &DoubleTensor) -> Result<T>,
    ) -> Result<Vec<T>> {
        self.draws
            .iter()
            .enumerate()
            .map(|(i, (p, x))| f(i, &model_of(p), x))
            .collect()
    }

    /// The Kalman smoother at each draw ([`LinearGaussianStateSpace::smooth`]): the states' means and covariances
    /// given every observation and the draw's parameters.
    ///
    /// # Errors
    ///
    /// An error if a draw's observations are impossible under its parameters.
    pub fn smooth(&self) -> Result<Vec<KalmanSmoother>> {
        self.each(|_, model, x| Ok(model.smooth(x)?))
    }

    /// The posterior mean of the states, [T, d]: the smoothed means averaged over the draws (a Rao–Blackwellised
    /// estimate, with less Monte Carlo error than averaging sampled states).
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidArgument`] if there are no draws, and as for [`Self::smooth`].
    pub fn mean_states(&self) -> Result<DoubleTensor> {
        let means: Vec<DoubleTensor> = self.smooth()?.into_iter().map(|s| s.means).collect();
        average(&means)
    }

    /// One draw of the states, [T, d], for each posterior draw, by forward-filtering backward-sampling
    /// ([`LinearGaussianStateSpace::sample_posterior_states`]): together with the parameters' draws, draws from the
    /// joint posterior of states and parameters.
    ///
    /// # Errors
    ///
    /// As for [`Self::smooth`].
    pub fn sample_states(&self, rng: &mut KeanuRng) -> Result<Vec<DoubleTensor>> {
        self.each(|_, model, x| Ok(model.sample_posterior_states(x, rng)?))
    }

    /// The predictive distribution of the next `horizon` steps at each draw, for a time-invariant model
    /// ([`LinearGaussianStateSpace::forecast`]).
    ///
    /// # Errors
    ///
    /// As for [`Self::smooth`].
    pub fn forecast(&self, horizon: usize) -> Result<Vec<StateSpaceForecast>> {
        self.each(|_, model, x| Ok(model.forecast(x, horizon)?))
    }

    /// The predictive distribution of future steps at each draw, with the steps' matrices given by `future` from
    /// the draw's index and model ([`LinearGaussianStateSpace::forecast_with`]): for time-varying models, or future
    /// matrices that depend on the draw's parameters (a component's noise scale, say) as well as on known future
    /// inputs.
    ///
    /// # Errors
    ///
    /// Any error `future` returns, and as for [`LinearGaussianStateSpace::forecast_with`].
    pub fn forecast_with(
        &self,
        mut future: impl FnMut(usize, &LinearGaussianStateSpace<'_>) -> Result<Vec<FutureStep>>,
    ) -> Result<Vec<StateSpaceForecast>> {
        self.each(|i, model, x| Ok(model.forecast_with(x, &future(i, model)?)?))
    }

    /// One simulated path of the next `horizon` steps for each draw, for a time-invariant model
    /// ([`LinearGaussianStateSpace::sample_forecast`]): draws from the posterior predictive distribution of the
    /// future, jointly over its steps.
    ///
    /// # Errors
    ///
    /// As for [`Self::smooth`].
    pub fn sample_forecast(
        &self,
        horizon: usize,
        rng: &mut KeanuRng,
    ) -> Result<Vec<StateSpacePath>> {
        self.each(|_, model, x| Ok(model.sample_forecast(x, horizon, rng)?))
    }

    /// [`Self::sample_forecast`] with the future steps' matrices given by `future`, as for
    /// [`Self::forecast_with`].
    ///
    /// # Errors
    ///
    /// As for [`Self::forecast_with`].
    pub fn sample_forecast_with(
        &self,
        mut future: impl FnMut(usize, &LinearGaussianStateSpace<'_>) -> Result<Vec<FutureStep>>,
        rng: &mut KeanuRng,
    ) -> Result<Vec<StateSpacePath>> {
        self.each(|i, model, x| Ok(model.sample_forecast_with(x, &future(i, model)?, rng)?))
    }
}

/// The model of a draw's parameters, borrowed.
fn model_of(p: &[DoubleTensor; 6]) -> LinearGaussianStateSpace<'_> {
    LinearGaussianStateSpace::new(&p[0], &p[1], &p[2], &p[3], &p[4], &p[5])
}

/// The element-wise mean of tensors of one shape.
fn average(tensors: &[DoubleTensor]) -> Result<DoubleTensor> {
    let Some(first) = tensors.first() else {
        return Err(AlgorithmError::InvalidArgument(
            "there are no draws to average".into(),
        ));
    };
    let mut total = vec![0.0; first.len()];
    for t in tensors {
        for (s, x) in total.iter_mut().zip(t.as_slice()) {
            *s += x;
        }
    }
    let count = tensors.len() as f64;
    Ok(DoubleTensor::new(
        first.shape(),
        total.into_iter().map(|s| s / count).collect(),
    )?)
}

/// A hidden Markov model node at each posterior draw: the chain and each step's emission log probability under
/// each state, from which the Viterbi path, the posterior state marginals and posterior state paths are computed
/// per draw. See the [module documentation](self).
#[derive(Debug, Clone, PartialEq)]
pub struct HiddenMarkovPosterior {
    draws: Vec<(HiddenMarkovChain<'static>, DoubleTensor)>,
}

impl HiddenMarkovPosterior {
    /// The hidden Markov model `node` at each draw in `samples`: its chain and emission log probabilities ([T, K],
    /// see [`DistributionKind::hidden_markov_chain`]) at its parameters evaluated from the draw (set in a copy of
    /// `state`, a state of the model's graph) and its observations.
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidArgument`] if `node` is not a hidden Markov model or `samples` doesn't record every
    /// latent its parameters depend on (naming the missing ones), an unknown-node error if a sampled node isn't in
    /// the graph, and any error evaluating the parameters or the chain (probabilities off the simplex).
    pub fn new(
        model: &Model,
        state: &State,
        node: NodeId,
        samples: &NetworkSamples,
    ) -> Result<Self> {
        Self::from_samples(model, state, node, &[samples])
    }

    /// [`Self::new`] over every chain's draws, chain by chain.
    ///
    /// # Errors
    ///
    /// As for [`Self::new`].
    pub fn from_chains(
        model: &Model,
        state: &State,
        node: NodeId,
        samples: &ChainSamples,
    ) -> Result<Self> {
        Self::from_samples(model, state, node, &chains(samples))
    }

    fn from_samples(
        model: &Model,
        state: &State,
        node: NodeId,
        samples: &[&NetworkSamples],
    ) -> Result<Self> {
        let kind = kind_of(model, node, "a hidden Markov model", |k| {
            matches!(k, DistributionKind::HiddenMarkovModel(_))
        })?;
        let draws = node_draws(model, state, node, samples)?
            .into_iter()
            .map(|(parameters, value)| {
                let parameters: Vec<&Value> = parameters.iter().collect();
                Ok(kind.hidden_markov_chain(&value, &parameters)?)
            })
            .collect::<Result<_>>()?;
        Ok(Self { draws })
    }

    /// The number of draws.
    pub fn len(&self) -> usize {
        self.draws.len()
    }

    /// Whether there are no draws.
    pub fn is_empty(&self) -> bool {
        self.draws.is_empty()
    }

    /// The chain and the emission log probabilities ([T, K]) at draw `index` (None past the last draw).
    pub fn draw(&self, index: usize) -> Option<(&HiddenMarkovChain<'static>, &DoubleTensor)> {
        self.draws.get(index).map(|(c, e)| (c, e))
    }

    /// The most probable state path at each draw and its log joint probability
    /// ([`HiddenMarkovChain::viterbi`]).
    ///
    /// # Errors
    ///
    /// An error if a draw's observations are impossible.
    pub fn viterbi(&self) -> Result<Vec<(Vec<usize>, f64)>> {
        self.draws
            .iter()
            .map(|(chain, e)| Ok(chain.viterbi(e)?))
            .collect()
    }

    /// The posterior state marginals p(z_t = j | x, draw), [T, K], at each draw
    /// ([`HiddenMarkovChain::posterior_marginals`]).
    ///
    /// # Errors
    ///
    /// An error if a draw's observations are impossible.
    pub fn posterior_marginals(&self) -> Result<Vec<DoubleTensor>> {
        self.draws
            .iter()
            .map(|(chain, e)| Ok(chain.posterior_marginals(e)?))
            .collect()
    }

    /// The posterior probability of each state at each step, [T, K]: the marginals averaged over the draws (a
    /// Rao–Blackwellised estimate).
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidArgument`] if there are no draws, and as for [`Self::posterior_marginals`].
    pub fn mean_marginals(&self) -> Result<DoubleTensor> {
        average(&self.posterior_marginals()?)
    }

    /// One posterior state path for each draw, by forward-filtering backward-sampling
    /// ([`HiddenMarkovChain::sample_posterior_states`]): with the parameters' draws, draws from the joint
    /// posterior of the states and the parameters.
    ///
    /// # Errors
    ///
    /// An error if a draw's observations are impossible.
    pub fn sample_states(&self, rng: &mut KeanuRng) -> Result<Vec<Vec<usize>>> {
        self.draws
            .iter()
            .map(|(chain, e)| Ok(chain.sample_posterior_states(e, rng)?))
            .collect()
    }
}
