//! Compound sampling: a cycle of steps, each updating its own block of the latents with everything else held fixed,
//! as PyMC's `CompoundStep`. NUTS updates continuous latents ([`Nuts::block_sampler`]), [`Gibbs`] and
//! [`DiscreteMetropolis`] discrete ones, and Metropolis–Hastings anything else.
//!
//! Each step of a [`CompoundSampler`] runs every step once, in the order they were added, and records the draw at
//! the end of the cycle. Each step leaves the joint posterior invariant, so the cycle does too. A NUTS step adapts its
//! step size and mass matrix over its first [`Nuts::adapt_count`] steps, which are the compound sampler's first
//! `adapt_count` steps, whatever the other steps do in between; drop them (`SamplingOptions::with_drop`) from the
//! draws as for plain NUTS.
//!
//! [`Compound::for_model`] chooses the steps for a model, and is what [`Sampler::for_model`](super::select::Sampler)
//! uses for a model with both continuous and discrete latents.
//!
//! ```
//! use keanu_algorithms::mcmc::SamplingOptions;
//! use keanu_algorithms::mcmc::compound::Compound;
//! use keanu_algorithms::mcmc::nuts::Nuts;
//! use keanu_graph::{GraphBuilder, Model};
//! use keanu_tensor::DoubleTensor;
//!
//! // a two-component mixture written with an explicit assignment z per point
//! let b = GraphBuilder::new();
//! let mu = b.gaussian_with_shape(&[2], 0.0, 5.0);
//! let z = b.bernoulli_with_shape(&[4], 0.5);
//! let mean = b.where_(z, mu.take(&[1]), mu.take(&[0]));
//! let y = b.gaussian_with_shape(&[4], mean, 1.0);
//! b.observe(y, DoubleTensor::vector(vec![-2.1, -1.8, 2.2, 1.9]));
//! b.set_value(mu, DoubleTensor::vector(vec![-1.0, 1.0]));
//! let (mu, z) = (mu.id(), z.id());
//! let (graph, mut state) = b.build()?;
//! let model = Model::new(graph, &state)?;
//!
//! let compound = Compound::new()
//!     .gibbs(&[z])
//!     .nuts_with(Nuts::default().with_adapt_count(200), &[mu]);
//! let mut rng = keanu_distributions::KeanuRng::seed_from_u64(1);
//! let (samples, statistics) = compound.sample_with_statistics(
//!     &model,
//!     &mut state,
//!     &[mu, z],
//!     1200,
//!     SamplingOptions::default().with_drop(200),
//!     &mut rng,
//! )?;
//! assert_eq!(samples.len(), 1000);
//! assert_eq!(statistics.nuts.len(), 1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{AlgorithmError, Result};
use crate::mcmc::gibbs::{DiscreteMetropolis, DiscreteMetropolisSampler, Gibbs, GibbsSampler};
use crate::mcmc::metropolis::{MetropolisHastings, MetropolisHastingsStep};
use crate::mcmc::nuts::{Nuts, NutsSampler, NutsStatistics};
use crate::mcmc::select::is_differentiable_wrt;
use crate::mcmc::{SamplingAlgorithm, SamplingOptions, generate, is_impossible};
use crate::samples::NetworkSamples;
use crate::support::{check_nodes, evaluate_all};
use keanu_distributions::KeanuRng;
use keanu_graph::op::{DistributionKind, Op};
use keanu_graph::{DType, Graph, GraphError, Model, NodeId, Parameterisation, Space, State, Value};

/// One step of a [`Compound`] sampler and the latents it updates.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum CompoundStep {
    /// NUTS over a block of continuous latents ([`Nuts::block_sampler`]).
    Nuts {
        /// The NUTS settings.
        nuts: Nuts,
        /// The continuous latents it updates.
        latents: Vec<NodeId>,
    },
    /// Gibbs sampling of discrete latents with finitely many values per element.
    Gibbs(Gibbs),
    /// Random-walk Metropolis–Hastings of integer latents.
    DiscreteMetropolis(DiscreteMetropolis),
    /// Metropolis–Hastings of any latents, one latent at a time, each step proposing once for each of them in
    /// order (the [`MetropolisHastings::selector`] is not used).
    MetropolisHastings {
        /// The Metropolis–Hastings settings (its proposal distribution and rejection strategy).
        mh: MetropolisHastings,
        /// The latents it updates.
        latents: Vec<NodeId>,
    },
}

impl CompoundStep {
    /// The latents the step updates.
    pub fn latents(&self) -> &[NodeId] {
        match self {
            Self::Nuts { latents, .. } | Self::MetropolisHastings { latents, .. } => latents,
            Self::Gibbs(gibbs) => &gibbs.latents,
            Self::DiscreteMetropolis(walk) => &walk.latents,
        }
    }
}

/// A compound sampler's configuration: the steps of each cycle, in order. See the [module documentation](self).
///
/// Build one step by step with `Compound::new().gibbs(...).nuts(...)`, or let [`Compound::for_model`] choose. The
/// steps' latents must not overlap, and every latent of the model must be updated by one of them. Like the other
/// samplers' configurations it holds no state, so it can be shared by parallel chains
/// ([`sample_chains`](crate::mcmc::chains::sample_chains)).
#[derive(Debug, Clone, Default)]
pub struct Compound {
    steps: Vec<CompoundStep>,
}

impl Compound {
    /// A compound sampler with no steps yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// The steps for a model, as [`Sampler::for_model`](super::select::Sampler::for_model) chooses them:
    ///
    /// - a [`Gibbs`] step for the discrete latents whose elements each have finitely many values (at most
    ///   [`Gibbs::DEFAULT_MAX_SUPPORT_SIZE`], where the state holds the parameters' values to tell);
    /// - a [`DiscreteMetropolis`] step for the other integer latents, except multinomials;
    /// - a [`MetropolisHastings`] step (with prior proposals) for any discrete latent left, and for the continuous
    ///   latents if the log probability can't be differentiated with respect to them ([`is_differentiable_wrt`]) or
    ///   a support bound moves with a latent (see [`Parameterisation::for_latents`]);
    /// - otherwise a NUTS step, with the default settings, for every continuous latent, last.
    ///
    /// Steps with no latents are left out.
    pub fn for_model(model: &Model, state: &State) -> Self {
        let graph = model.graph();
        let (mut gibbs, mut walk, mut mh) = (Vec::new(), Vec::new(), Vec::new());
        for id in model.discrete_latents() {
            match discrete_step_for(graph, state, id) {
                DiscreteChoice::Gibbs => gibbs.push(id),
                DiscreteChoice::Walk => walk.push(id),
                DiscreteChoice::MetropolisHastings => mh.push(id),
            }
        }
        let continuous = model.continuous_latents();
        // a support bound that moves with a latent has no fixed transform for NUTS (a missing value can't tell)
        let transformable = || {
            !matches!(
                Parameterisation::for_latents(model, state, &continuous, Space::Unconstrained),
                Err(GraphError::Transform(_))
            )
        };
        let nuts = !continuous.is_empty()
            && is_differentiable_wrt(model, state, &continuous)
            && transformable();
        if !nuts {
            mh.extend(continuous.iter().copied());
        }
        let mut compound = Self::new();
        if !gibbs.is_empty() {
            compound = compound.gibbs(&gibbs);
        }
        if !walk.is_empty() {
            compound = compound.discrete_metropolis(&walk);
        }
        if !mh.is_empty() {
            mh.sort();
            compound = compound.metropolis_hastings(MetropolisHastings::default(), &mh);
        }
        if nuts {
            compound = compound.nuts(&continuous);
        }
        compound
    }

    /// Adds a step.
    #[must_use]
    pub fn step(mut self, step: CompoundStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Adds a NUTS step over the continuous `latents`, with the default settings.
    #[must_use]
    pub fn nuts(self, latents: &[NodeId]) -> Self {
        self.nuts_with(Nuts::default(), latents)
    }

    /// Adds a NUTS step over the continuous `latents`, with the given settings.
    #[must_use]
    pub fn nuts_with(self, nuts: Nuts, latents: &[NodeId]) -> Self {
        self.step(CompoundStep::Nuts {
            nuts,
            latents: latents.to_vec(),
        })
    }

    /// Adds a [`Gibbs`] step over `latents`, with the default settings.
    #[must_use]
    pub fn gibbs(self, latents: &[NodeId]) -> Self {
        self.step(CompoundStep::Gibbs(Gibbs::new(latents)))
    }

    /// Adds a [`Gibbs`] step.
    #[must_use]
    pub fn gibbs_with(self, gibbs: Gibbs) -> Self {
        self.step(CompoundStep::Gibbs(gibbs))
    }

    /// Adds a [`DiscreteMetropolis`] step over `latents`, with the default settings.
    #[must_use]
    pub fn discrete_metropolis(self, latents: &[NodeId]) -> Self {
        self.step(CompoundStep::DiscreteMetropolis(DiscreteMetropolis::new(
            latents,
        )))
    }

    /// Adds a [`DiscreteMetropolis`] step.
    #[must_use]
    pub fn discrete_metropolis_with(self, walk: DiscreteMetropolis) -> Self {
        self.step(CompoundStep::DiscreteMetropolis(walk))
    }

    /// Adds a Metropolis–Hastings step over `latents`.
    #[must_use]
    pub fn metropolis_hastings(self, mh: MetropolisHastings, latents: &[NodeId]) -> Self {
        self.step(CompoundStep::MetropolisHastings {
            mh,
            latents: latents.to_vec(),
        })
    }

    /// Replaces the settings of every NUTS step (for instance to change the adaptation steps of the steps
    /// [`for_model`](Self::for_model) chose), keeping their latents.
    #[must_use]
    pub fn with_nuts_settings(mut self, settings: Nuts) -> Self {
        for step in &mut self.steps {
            if let CompoundStep::Nuts { nuts, .. } = step {
                *nuts = settings;
            }
        }
        self
    }

    /// The steps, in the order each cycle runs them.
    pub fn steps(&self) -> &[CompoundStep] {
        &self.steps
    }

    /// Checks that the steps' latents are latents of the model, don't overlap and cover every latent.
    fn check_blocks(&self, model: &Model) -> Result<()> {
        let graph = model.graph();
        if self.steps.is_empty() {
            return Err(AlgorithmError::InvalidArgument(
                "a compound sampler needs at least one step".into(),
            ));
        }
        let mut owner: Vec<Option<usize>> = vec![None; graph.len()];
        for (s, step) in self.steps.iter().enumerate() {
            check_nodes(graph, step.latents())?;
            for &id in step.latents() {
                if !model.latents().contains(&id) {
                    return Err(AlgorithmError::InvalidArgument(format!(
                        "{} is not a latent of the model",
                        graph.describe(id)
                    )));
                }
                if let Some(other) = owner[id.index()] {
                    return Err(AlgorithmError::InvalidArgument(format!(
                        "{} is updated by both step {other} and step {s}; each latent belongs to one step",
                        graph.describe(id)
                    )));
                }
                owner[id.index()] = Some(s);
            }
        }
        if let Some(&id) = model
            .latents()
            .iter()
            .find(|id| owner[id.index()].is_none())
        {
            return Err(AlgorithmError::InvalidArgument(format!(
                "no step updates the latent {}; add it to a step, or observe it to hold it fixed",
                graph.describe(id)
            )));
        }
        Ok(())
    }

    /// A chain starting from the current state, recording the values of `nodes`. Latents without a value are
    /// sampled from their priors first, and each step is set up as its own sampler would be (NUTS finds its
    /// initial step size).
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidArgument`] if there are no steps or no nodes to record, a step's latent is not a
    /// latent of the model or belongs to two steps, or a latent belongs to no step;
    /// [`AlgorithmError::ImpossibleStart`] if the state has zero probability; and any error setting up a step
    /// ([`Nuts::block_sampler`], [`Gibbs::sampler`], [`DiscreteMetropolis::sampler`]).
    pub fn sampler<'m>(
        &self,
        model: &'m Model,
        state: &mut State,
        nodes: &[NodeId],
        rng: &mut KeanuRng,
    ) -> Result<CompoundSampler<'m>> {
        if nodes.is_empty() {
            return Err(AlgorithmError::InvalidArgument("no nodes to sample".into()));
        }
        check_nodes(model.graph(), nodes)?;
        self.check_blocks(model)?;
        evaluate_all(model.graph(), state, rng)?;
        if is_impossible(model.log_prob(state)?) {
            return Err(AlgorithmError::ImpossibleStart);
        }
        let mut running = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            running.push(match step {
                CompoundStep::Nuts { nuts, latents } => Running::Nuts(Box::new(
                    nuts.block_sampler(model, state, latents, nodes, rng)?,
                )),
                CompoundStep::Gibbs(gibbs) => {
                    Running::Gibbs(gibbs.sampler(model, state, nodes, rng)?)
                }
                CompoundStep::DiscreteMetropolis(walk) => {
                    Running::Walk(walk.sampler(model, state, nodes, rng)?)
                }
                CompoundStep::MetropolisHastings { mh, latents } => Running::MetropolisHastings {
                    step: MetropolisHastingsStep::new(model, mh.proposal.clone(), mh.rejection),
                    latents: latents.clone(),
                },
            });
        }
        Ok(CompoundSampler {
            model,
            running,
            nodes: nodes.to_vec(),
            log_prob: model.log_prob(state)?,
        })
    }

    /// Runs a chain of `count` steps (cycles) from the current state, recording every step's values of `nodes`.
    pub fn sample(
        &self,
        model: &Model,
        state: &mut State,
        nodes: &[NodeId],
        count: usize,
        rng: &mut KeanuRng,
    ) -> Result<NetworkSamples> {
        Ok(self
            .sample_with_statistics(model, state, nodes, count, SamplingOptions::default(), rng)?
            .0)
    }

    /// Runs a chain, dropping and down-sampling as `options` says, and returns what its steps recorded as well.
    pub fn sample_with_statistics(
        &self,
        model: &Model,
        state: &mut State,
        nodes: &[NodeId],
        count: usize,
        options: SamplingOptions,
        rng: &mut KeanuRng,
    ) -> Result<(NetworkSamples, CompoundStatistics)> {
        let mut sampler = self.sampler(model, state, nodes, rng)?;
        let samples = generate(&mut sampler, state, count, options, rng)?;
        Ok((samples, sampler.into_statistics()))
    }
}

/// Which kind of step suits a discrete latent.
enum DiscreteChoice {
    Gibbs,
    Walk,
    MetropolisHastings,
}

fn discrete_step_for(graph: &Graph, state: &State, id: NodeId) -> DiscreteChoice {
    let integer = graph[id].dtype() == DType::I64;
    let Op::Distribution(op) = graph[id].op() else {
        return DiscreteChoice::MetropolisHastings;
    };
    let walk_or_mh = || {
        if integer && !matches!(op.kind, DistributionKind::Multinomial) {
            DiscreteChoice::Walk
        } else {
            DiscreteChoice::MetropolisHastings
        }
    };
    let parameters: Option<Vec<&Value>> = graph[id]
        .parents()
        .iter()
        .map(|&p| state.value(p))
        .collect();
    let support = parameters.and_then(|p| op.kind.finite_support(&p, graph[id].shape()).ok());
    match support {
        Some(Some(ranges)) => {
            let enumerable = ranges.iter().all(|r| {
                !r.is_empty()
                    && (i128::from(*r.end()) - i128::from(*r.start()))
                        < Gibbs::DEFAULT_MAX_SUPPORT_SIZE as i128
            });
            if enumerable {
                DiscreteChoice::Gibbs
            } else {
                walk_or_mh()
            }
        }
        Some(None) => walk_or_mh(),
        // the parameters' values aren't known: go by the kind
        None if op.kind.has_finite_support() => DiscreteChoice::Gibbs,
        None => walk_or_mh(),
    }
}

/// A step of a running compound chain.
#[derive(Debug)]
enum Running<'m> {
    Nuts(Box<NutsSampler<'m>>),
    Gibbs(GibbsSampler<'m>),
    Walk(DiscreteMetropolisSampler<'m>),
    MetropolisHastings {
        step: MetropolisHastingsStep<'m>,
        latents: Vec<NodeId>,
    },
}

/// What a compound chain's steps recorded.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct CompoundStatistics {
    /// Each NUTS step's statistics, in step order (one per step of each cycle). With one NUTS step, as
    /// [`Compound::for_model`] gives, its divergences, energies and adaptation are the chain's, and it can be passed
    /// to [`CmdStanCsv::with_nuts`](crate::export::CmdStanCsv::with_nuts).
    pub nuts: Vec<NutsStatistics>,
    /// Each discrete Metropolis step's acceptance rate, in step order.
    pub discrete_acceptance_rate: Vec<f64>,
}

/// A running compound chain. See the [module documentation](self).
#[derive(Debug)]
pub struct CompoundSampler<'m> {
    model: &'m Model,
    running: Vec<Running<'m>>,
    nodes: Vec<NodeId>,
    log_prob: f64,
}

impl CompoundSampler<'_> {
    /// The joint log probability of the chain's current position.
    pub fn log_prob(&self) -> f64 {
        self.log_prob
    }

    /// What the steps have recorded so far.
    pub fn statistics(&self) -> CompoundStatistics {
        let mut statistics = CompoundStatistics::default();
        for running in &self.running {
            match running {
                Running::Nuts(nuts) => statistics.nuts.push(nuts.statistics().clone()),
                Running::Walk(walk) => statistics
                    .discrete_acceptance_rate
                    .push(walk.acceptance_rate()),
                _ => {}
            }
        }
        statistics
    }

    /// The statistics, taken out of the sampler.
    pub fn into_statistics(self) -> CompoundStatistics {
        let mut statistics = CompoundStatistics::default();
        for running in self.running {
            match running {
                Running::Nuts(nuts) => statistics.nuts.push(nuts.into_statistics()),
                Running::Walk(walk) => statistics
                    .discrete_acceptance_rate
                    .push(walk.acceptance_rate()),
                _ => {}
            }
        }
        statistics
    }
}

impl SamplingAlgorithm for CompoundSampler<'_> {
    fn step(&mut self, state: &mut State, rng: &mut KeanuRng) -> Result<()> {
        let mut log_prob = self.log_prob;
        for running in &mut self.running {
            log_prob = match running {
                Running::Nuts(nuts) => {
                    nuts.step(state, rng)?;
                    nuts.log_prob()
                }
                Running::Gibbs(gibbs) => gibbs.sweep(state, log_prob, rng)?,
                Running::Walk(walk) => walk.sweep(state, log_prob, rng)?,
                Running::MetropolisHastings { step, latents } => {
                    for &latent in latents.iter() {
                        log_prob = step
                            .step(state, &[latent], log_prob, 1.0, rng, None)?
                            .log_prob;
                    }
                    log_prob
                }
            };
        }
        self.log_prob = log_prob;
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
