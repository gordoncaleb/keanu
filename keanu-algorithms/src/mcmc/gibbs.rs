//! Steps for discrete latents: [`Gibbs`] sampling by exact enumeration, for latents whose elements have finitely
//! many values, and [`DiscreteMetropolis`], a random walk for integer latents with infinitely many.
//!
//! Both update each element of each of their latents in turn, holding everything else fixed, and work out the
//! change in the joint log probability from the latent's Markov blanket alone: the latent's own log probability,
//! its children's, and the factors it reaches ([`Model::affected_by`] and [`Model::local_log_prob`]). Changing an
//! element of a vector latent recomputes the whole of each affected node, so a sweep over a latent of n elements
//! costs about n evaluations of those nodes per value tried.
//!
//! Either can sample a model of discrete latents on its own, or be a step of a
//! [`Compound`](crate::mcmc::compound::Compound) sampler that updates the continuous latents with NUTS between them.
//!
//! ```
//! use keanu_algorithms::mcmc::gibbs::Gibbs;
//! use keanu_graph::{GraphBuilder, Model};
//! use keanu_tensor::DoubleTensor;
//!
//! // z ~ Bernoulli(0.3), y ~ N(2 z, 1) observed at 1.5
//! let b = GraphBuilder::new();
//! let z = b.bernoulli(0.3);
//! let y = b.gaussian(b.where_(z, b.scalar(2.0), b.scalar(0.0)), 1.0);
//! b.observe(y, DoubleTensor::scalar(1.5));
//! let z = z.id();
//! let (graph, mut state) = b.build()?;
//! let model = Model::new(graph, &state)?;
//!
//! let mut rng = keanu_distributions::KeanuRng::seed_from_u64(1);
//! let samples = Gibbs::new(&[z]).sample(&model, &mut state, &[z], 4000, &mut rng)?;
//! let p = samples.bool(z).unwrap().probability(|z| z.as_slice()[0]);
//! // the exact posterior: 0.3 N(1.5; 2, 1) / (0.3 N(1.5; 2, 1) + 0.7 N(1.5; 0, 1)) = 0.538...
//! assert!((p - 0.538).abs() < 0.05);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{AlgorithmError, Result};
use crate::mcmc::{SamplingAlgorithm, SamplingOptions, generate, is_impossible};
use crate::samples::NetworkSamples;
use crate::support::{check_nodes, evaluate_all};
use keanu_distributions::KeanuRng;
use keanu_graph::op::{DistributionKind, Op};
use keanu_graph::{DType, Graph, GraphError, Model, NodeId, State, Value};
use keanu_tensor::{BoolTensor, IntTensor};
use std::ops::RangeInclusive;

/// Gibbs sampling of discrete latents by exact enumeration.
///
/// Each step sweeps over every element of every latent in [`latents`](Self::latents), in order. For each element it
/// evaluates the joint log probability (incrementally, over the latent's Markov blanket) at every value the element
/// can take with everything else held fixed, and draws the element's new value from those exact conditional
/// probabilities. Each draw is from the element's full conditional distribution, so nothing is rejected.
///
/// It handles latents whose elements each have finitely many values ([`DistributionKind::finite_support`]): the
/// Bernoulli, binomial, categorical, ordered logistic and discrete uniform, and a truncated binomial, Poisson or
/// negative binomial with a finite upper bound. The supports are read from the parameters' current values at every
/// step, so they may depend on other latents. An element with more than
/// [`max_support_size`](Self::max_support_size) values is an error: use [`DiscreteMetropolis`] for it instead.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Gibbs {
    /// The latents to update, in the order they are swept.
    pub latents: Vec<NodeId>,
    /// The most values an element may have (default [`Gibbs::DEFAULT_MAX_SUPPORT_SIZE`]).
    pub max_support_size: usize,
}

impl Gibbs {
    /// The default [`max_support_size`](Self::max_support_size).
    pub const DEFAULT_MAX_SUPPORT_SIZE: usize = 1000;

    /// Gibbs sampling of `latents`, with the default maximum support size.
    pub fn new(latents: &[NodeId]) -> Self {
        Self {
            latents: latents.to_vec(),
            max_support_size: Self::DEFAULT_MAX_SUPPORT_SIZE,
        }
    }

    /// Sets [`max_support_size`](Self::max_support_size).
    #[must_use]
    pub fn with_max_support_size(mut self, max_support_size: usize) -> Self {
        self.max_support_size = max_support_size;
        self
    }

    /// A chain starting from the current state, recording the values of `nodes`. Latents without a value are
    /// sampled from their priors first.
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidArgument`] if there are no latents or no nodes to record, a latent is not a discrete
    /// latent of the model or is given twice, or its elements don't have finitely many values each, or more than
    /// [`max_support_size`](Self::max_support_size) of them (the message names the latent); and
    /// [`AlgorithmError::InvalidSetting`] for a maximum support size of 0.
    pub fn sampler<'m>(
        &self,
        model: &'m Model,
        state: &mut State,
        nodes: &[NodeId],
        rng: &mut KeanuRng,
    ) -> Result<GibbsSampler<'m>> {
        if self.max_support_size == 0 {
            return Err(AlgorithmError::InvalidSetting(
                "the maximum support size must be at least 1".into(),
            ));
        }
        let latents = check_discrete_latents(model, nodes, &self.latents, "Gibbs")?;
        evaluate_all(model.graph(), state, rng)?;
        for latent in &latents {
            enumerable_support(model.graph(), state, latent.id, self.max_support_size)?;
        }
        Ok(GibbsSampler {
            model,
            latents,
            max_support_size: self.max_support_size,
            nodes: nodes.to_vec(),
            log_prob: model.log_prob(state)?,
        })
    }

    /// Runs a chain of `count` steps (sweeps) from the current state, recording every step's values of `nodes`.
    pub fn sample(
        &self,
        model: &Model,
        state: &mut State,
        nodes: &[NodeId],
        count: usize,
        rng: &mut KeanuRng,
    ) -> Result<NetworkSamples> {
        self.sample_with(model, state, nodes, count, SamplingOptions::default(), rng)
    }

    /// Like [`sample`](Self::sample), dropping and down-sampling as `options` says.
    pub fn sample_with(
        &self,
        model: &Model,
        state: &mut State,
        nodes: &[NodeId],
        count: usize,
        options: SamplingOptions,
        rng: &mut KeanuRng,
    ) -> Result<NetworkSamples> {
        let mut sampler = self.sampler(model, state, nodes, rng)?;
        generate(&mut sampler, state, count, options, rng)
    }
}

/// A latent a discrete step updates, and the nodes whose log probability depends on it.
#[derive(Debug, Clone)]
struct Target {
    id: NodeId,
    affected: Vec<NodeId>,
}

/// Checks the latents of a discrete step and the nodes it records, and finds each latent's Markov blanket.
fn check_discrete_latents(
    model: &Model,
    nodes: &[NodeId],
    latents: &[NodeId],
    step: &str,
) -> Result<Vec<Target>> {
    let graph = model.graph();
    if nodes.is_empty() {
        return Err(AlgorithmError::InvalidArgument("no nodes to sample".into()));
    }
    if latents.is_empty() {
        return Err(AlgorithmError::InvalidArgument(format!(
            "{step} needs at least one latent to update"
        )));
    }
    check_nodes(graph, nodes)?;
    check_nodes(graph, latents)?;
    let mut targets = Vec::with_capacity(latents.len());
    for (i, &id) in latents.iter().enumerate() {
        let describe = || graph.describe(id);
        if !model.latents().contains(&id) {
            return Err(AlgorithmError::InvalidArgument(format!(
                "{step} can only update latents, and {} is not one",
                describe()
            )));
        }
        if graph[id].dtype() == DType::F64 {
            return Err(AlgorithmError::InvalidArgument(format!(
                "{step} updates discrete latents, and {} is continuous",
                describe()
            )));
        }
        if latents[..i].contains(&id) {
            return Err(AlgorithmError::InvalidArgument(format!(
                "{} is given twice",
                describe()
            )));
        }
        targets.push(Target {
            id,
            affected: model.affected_by(&[id])?,
        });
    }
    Ok(targets)
}

/// The distribution of a probabilistic node.
fn kind_of(graph: &Graph, id: NodeId) -> Option<&DistributionKind> {
    match graph[id].op() {
        Op::Distribution(op) => Some(&op.kind),
        _ => None,
    }
}

/// The values each element of a latent can take, checked against the maximum support size.
fn enumerable_support(
    graph: &Graph,
    state: &State,
    id: NodeId,
    max_support_size: usize,
) -> Result<Vec<RangeInclusive<i64>>> {
    let describe = || graph.describe(id);
    let Some(kind) = kind_of(graph, id) else {
        return Err(AlgorithmError::InvalidArgument(format!(
            "Gibbs can't enumerate the values of {}",
            describe()
        )));
    };
    let parameters: Vec<&Value> = graph[id]
        .parents()
        .iter()
        .map(|&p| state.value(p).ok_or(GraphError::MissingValue(p)))
        .collect::<std::result::Result<_, _>>()?;
    let Some(support) = kind.finite_support(&parameters, graph[id].shape())? else {
        return Err(AlgorithmError::InvalidArgument(format!(
            "Gibbs needs finitely many values for each element, and the {} latent {} has infinitely many or can't \
             be updated one element at a time; use DiscreteMetropolis or MetropolisHastings for it",
            kind.name(),
            describe()
        )));
    };
    for range in &support {
        let size = support_size(range);
        if size == 0 {
            return Err(AlgorithmError::InvalidArgument(format!(
                "an element of {} has no possible values ({}..={})",
                describe(),
                range.start(),
                range.end()
            )));
        }
        if size > max_support_size as u128 {
            return Err(AlgorithmError::InvalidArgument(format!(
                "an element of {} has {size} possible values, more than Gibbs' maximum support size of \
                 {max_support_size}; use DiscreteMetropolis for it, or raise the maximum",
                describe()
            )));
        }
    }
    Ok(support)
}

fn support_size(range: &RangeInclusive<i64>) -> u128 {
    if range.is_empty() {
        0
    } else {
        (i128::from(*range.end()) - i128::from(*range.start()) + 1) as u128
    }
}

/// A value's elements as integers (false and true as 0 and 1), to change one at a time.
#[derive(Debug, Clone)]
enum Elements {
    I64(IntTensor),
    Bool(BoolTensor),
}

impl Elements {
    fn of(state: &State, id: NodeId) -> Result<Self> {
        match state.value(id) {
            Some(Value::I64(t)) => Ok(Self::I64(t.clone())),
            Some(Value::Bool(t)) => Ok(Self::Bool(t.clone())),
            None => Err(GraphError::MissingValue(id).into()),
            Some(_) => Err(AlgorithmError::InvalidArgument(format!(
                "node {id:?} is not discrete"
            ))),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::I64(t) => t.len(),
            Self::Bool(t) => t.len(),
        }
    }

    fn get(&self, i: usize) -> i64 {
        match self {
            Self::I64(t) => t.as_slice()[i],
            Self::Bool(t) => i64::from(t.as_slice()[i]),
        }
    }

    /// The value with element `i` set to `x`.
    fn with(&self, i: usize, x: i64) -> Result<Value> {
        Ok(match self {
            Self::I64(t) => {
                let mut data = t.as_slice().to_vec();
                data[i] = x;
                Value::I64(IntTensor::new(t.shape(), data)?)
            }
            Self::Bool(t) => {
                let mut data = t.as_slice().to_vec();
                data[i] = x != 0;
                Value::Bool(BoolTensor::new(t.shape(), data)?)
            }
        })
    }

    fn set(&mut self, i: usize, x: i64) -> Result<()> {
        *self = match self.with(i, x)? {
            Value::I64(t) => Self::I64(t),
            Value::Bool(t) => Self::Bool(t),
            _ => unreachable!("discrete values stay discrete"),
        };
        Ok(())
    }

    fn to_value(&self) -> Value {
        match self {
            Self::I64(t) => Value::I64(t.clone()),
            Self::Bool(t) => Value::Bool(t.clone()),
        }
    }
}

/// Sets a latent's value and recomputes its deterministic descendants. False if the cascade failed at an
/// impossible point (such as a failed ODE solve), where the value has zero probability.
fn set(model: &Model, state: &mut State, id: NodeId, value: Value) -> Result<bool> {
    match model.set_and_cascade(state, [(id, value)]) {
        Ok(()) => Ok(true),
        Err(e) if e.is_impossible_point() => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// The local log probability after setting a latent's value: −∞ where the cascade fails at an impossible point.
fn local_after(
    model: &Model,
    state: &mut State,
    id: NodeId,
    value: Value,
    affected: &[NodeId],
) -> Result<f64> {
    Ok(if set(model, state, id, value)? {
        model.local_log_prob(state, affected)?
    } else {
        f64::NEG_INFINITY
    })
}

/// The joint log probability after a change of `delta` in the local log probability: exact when both sides are
/// possible, and NaN (to be recomputed) otherwise.
fn shifted(log_prob: f64, before: f64, after: f64) -> f64 {
    if before.is_finite() && after.is_finite() && log_prob.is_finite() {
        log_prob + (after - before)
    } else {
        f64::NAN
    }
}

/// A running Gibbs chain.
#[derive(Debug)]
pub struct GibbsSampler<'m> {
    model: &'m Model,
    latents: Vec<Target>,
    max_support_size: usize,
    nodes: Vec<NodeId>,
    log_prob: f64,
}

impl GibbsSampler<'_> {
    /// The joint log probability of the chain's current position.
    pub fn log_prob(&self) -> f64 {
        self.log_prob
    }

    /// One sweep from a state whose joint log probability is `log_prob`; returns the new joint log probability.
    pub(crate) fn sweep(
        &self,
        state: &mut State,
        log_prob: f64,
        rng: &mut KeanuRng,
    ) -> Result<f64> {
        let mut log_prob = log_prob;
        let graph = self.model.graph();
        let mut weights: Vec<f64> = Vec::new();
        for target in &self.latents {
            let support = enumerable_support(graph, state, target.id, self.max_support_size)?;
            let mut elements = Elements::of(state, target.id)?;
            for (i, range) in support.iter().enumerate().take(elements.len()) {
                let current = elements.get(i);
                let local = self.model.local_log_prob(state, &target.affected)?;
                weights.clear();
                let mut in_state = current;
                for x in range.clone() {
                    let lp = if x == current {
                        local
                    } else {
                        in_state = x;
                        local_after(
                            self.model,
                            state,
                            target.id,
                            elements.with(i, x)?,
                            &target.affected,
                        )?
                    };
                    weights.push(lp);
                }
                let chosen = match draw(&weights, rng) {
                    Some(k) => range.start() + k as i64,
                    // every value is impossible: leave the element as it was
                    None => current,
                };
                if chosen != in_state {
                    set(self.model, state, target.id, elements.with(i, chosen)?)?;
                }
                if chosen != current {
                    let after = weights[(chosen - range.start()) as usize];
                    log_prob = shifted(log_prob, local, after);
                    elements.set(i, chosen)?;
                }
            }
        }
        if log_prob.is_nan() {
            log_prob = self.model.log_prob(state)?;
        }
        Ok(log_prob)
    }
}

/// Draws an index with probability proportional to `exp(log_weights)`, or `None` if every weight is zero.
fn draw(log_weights: &[f64], rng: &mut KeanuRng) -> Option<usize> {
    let max = log_weights
        .iter()
        .copied()
        .filter(|w| !w.is_nan())
        .fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return None;
    }
    let weight = |w: f64| if w.is_nan() { 0.0 } else { (w - max).exp() };
    let total: f64 = log_weights.iter().map(|&w| weight(w)).sum();
    let mut u = rng.next_f64() * total;
    let mut last = 0;
    for (k, &w) in log_weights.iter().enumerate() {
        let w = weight(w);
        if w > 0.0 {
            last = k;
            if u < w {
                return Some(k);
            }
            u -= w;
        }
    }
    // rounding left u at the top of the range
    Some(last)
}

impl SamplingAlgorithm for GibbsSampler<'_> {
    fn step(&mut self, state: &mut State, rng: &mut KeanuRng) -> Result<()> {
        self.log_prob = self.sweep(state, self.log_prob, rng)?;
        Ok(())
    }

    fn record(&self, state: &State, samples: &mut NetworkSamples) -> Result<()> {
        samples.record(state, self.log_prob)
    }

    fn graph(&self) -> &Graph {
        self.model.graph()
    }

    fn sampled_nodes(&self) -> &[NodeId] {
        &self.nodes
    }
}

/// Random-walk Metropolis–Hastings for integer latents, one element at a time.
///
/// Each step sweeps over every element of every latent in [`latents`](Self::latents), in order, proposing to move
/// the element by a step drawn uniformly from ±1, …, ±[`max_step`](Self::max_step) (±1 or ±2 by default) and
/// accepting the move with the Metropolis probability. The proposal is symmetric, so no correction is needed, and a
/// move out of the support has probability zero and is rejected. It suits counts with infinitely many values (the
/// Poisson, negative binomial and geometric, which [`Gibbs`] can't enumerate), and works for any integer latent
/// whose elements can be changed one at a time; boolean latents need [`Gibbs`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DiscreteMetropolis {
    /// The latents to update, in the order they are swept.
    pub latents: Vec<NodeId>,
    /// The largest step proposed (default 2).
    pub max_step: u32,
}

impl DiscreteMetropolis {
    /// Random-walk Metropolis–Hastings of `latents`, with steps of ±1 or ±2.
    pub fn new(latents: &[NodeId]) -> Self {
        Self {
            latents: latents.to_vec(),
            max_step: 2,
        }
    }

    /// Sets [`max_step`](Self::max_step).
    #[must_use]
    pub fn with_max_step(mut self, max_step: u32) -> Self {
        self.max_step = max_step;
        self
    }

    /// A chain starting from the current state, recording the values of `nodes`. Latents without a value are
    /// sampled from their priors first.
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidArgument`] if there are no latents or no nodes to record, or a latent is not an
    /// integer latent of the model, is given twice, or is a multinomial (whose elements must keep their sum); and
    /// [`AlgorithmError::InvalidSetting`] for a maximum step of 0.
    pub fn sampler<'m>(
        &self,
        model: &'m Model,
        state: &mut State,
        nodes: &[NodeId],
        rng: &mut KeanuRng,
    ) -> Result<DiscreteMetropolisSampler<'m>> {
        if self.max_step == 0 {
            return Err(AlgorithmError::InvalidSetting(
                "the maximum step must be at least 1".into(),
            ));
        }
        let latents = check_discrete_latents(model, nodes, &self.latents, "DiscreteMetropolis")?;
        let graph = model.graph();
        for latent in &latents {
            if graph[latent.id].dtype() != DType::I64 {
                return Err(AlgorithmError::InvalidArgument(format!(
                    "DiscreteMetropolis updates integer latents, and {} is boolean; use Gibbs for it",
                    graph.describe(latent.id)
                )));
            }
            if matches!(
                kind_of(graph, latent.id),
                Some(DistributionKind::Multinomial)
            ) {
                return Err(AlgorithmError::InvalidArgument(format!(
                    "the elements of the multinomial latent {} must keep their sum, so DiscreteMetropolis can't \
                     move them one at a time; use MetropolisHastings for it",
                    graph.describe(latent.id)
                )));
            }
        }
        evaluate_all(graph, state, rng)?;
        Ok(DiscreteMetropolisSampler {
            model,
            latents,
            max_step: i64::from(self.max_step),
            nodes: nodes.to_vec(),
            log_prob: model.log_prob(state)?,
            proposed: 0,
            accepted: 0,
        })
    }

    /// Runs a chain of `count` steps (sweeps) from the current state, recording every step's values of `nodes`.
    pub fn sample(
        &self,
        model: &Model,
        state: &mut State,
        nodes: &[NodeId],
        count: usize,
        rng: &mut KeanuRng,
    ) -> Result<NetworkSamples> {
        self.sample_with(model, state, nodes, count, SamplingOptions::default(), rng)
    }

    /// Like [`sample`](Self::sample), dropping and down-sampling as `options` says.
    pub fn sample_with(
        &self,
        model: &Model,
        state: &mut State,
        nodes: &[NodeId],
        count: usize,
        options: SamplingOptions,
        rng: &mut KeanuRng,
    ) -> Result<NetworkSamples> {
        let mut sampler = self.sampler(model, state, nodes, rng)?;
        generate(&mut sampler, state, count, options, rng)
    }
}

/// A running discrete random-walk chain.
#[derive(Debug)]
pub struct DiscreteMetropolisSampler<'m> {
    model: &'m Model,
    latents: Vec<Target>,
    max_step: i64,
    nodes: Vec<NodeId>,
    log_prob: f64,
    proposed: usize,
    accepted: usize,
}

impl DiscreteMetropolisSampler<'_> {
    /// The joint log probability of the chain's current position.
    pub fn log_prob(&self) -> f64 {
        self.log_prob
    }

    /// The fraction of proposals accepted so far (0 before the first).
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposed == 0 {
            0.0
        } else {
            self.accepted as f64 / self.proposed as f64
        }
    }

    /// A step drawn uniformly from ±1, …, ±max_step.
    fn random_step(&self, rng: &mut KeanuRng) -> i64 {
        let k = rng.next_below(2 * self.max_step as u64) as i64;
        if k < self.max_step {
            k - self.max_step
        } else {
            k - self.max_step + 1
        }
    }

    /// One sweep from a state whose joint log probability is `log_prob`; returns the new joint log probability.
    pub(crate) fn sweep(
        &mut self,
        state: &mut State,
        log_prob: f64,
        rng: &mut KeanuRng,
    ) -> Result<f64> {
        let mut log_prob = log_prob;
        for t in 0..self.latents.len() {
            let (id, affected) = (self.latents[t].id, &self.latents[t].affected);
            let mut elements = Elements::of(state, id)?;
            for i in 0..elements.len() {
                let current = elements.get(i);
                let step = self.random_step(rng);
                self.proposed += 1;
                let Some(proposed) = current.checked_add(step) else {
                    continue;
                };
                let before = self.model.local_log_prob(state, affected)?;
                let after =
                    local_after(self.model, state, id, elements.with(i, proposed)?, affected)?;
                let accept = !is_impossible(after)
                    && (is_impossible(before) || rng.next_f64().ln() < after - before);
                if accept {
                    self.accepted += 1;
                    elements.set(i, proposed)?;
                    log_prob = shifted(log_prob, before, after);
                } else {
                    set(self.model, state, id, elements.to_value())?;
                }
            }
        }
        if log_prob.is_nan() {
            log_prob = self.model.log_prob(state)?;
        }
        Ok(log_prob)
    }
}

impl SamplingAlgorithm for DiscreteMetropolisSampler<'_> {
    fn step(&mut self, state: &mut State, rng: &mut KeanuRng) -> Result<()> {
        self.log_prob = self.sweep(state, self.log_prob, rng)?;
        Ok(())
    }

    fn record(&self, state: &State, samples: &mut NetworkSamples) -> Result<()> {
        samples.record(state, self.log_prob)
    }

    fn graph(&self) -> &Graph {
        self.model.graph()
    }

    fn sampled_nodes(&self) -> &[NodeId] {
        &self.nodes
    }
}
