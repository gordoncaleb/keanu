//! Tempered sequential Monte Carlo (SMC): a population of particles moved from the prior to the posterior through
//! a sequence of tempered targets, which also estimates the model's log marginal likelihood (its log evidence).
//!
//! The algorithm follows PyMC's `sample_smc` (Ching and Chen, 2007; Minson et al., 2013). The target at inverse
//! temperature β ∈ [0, 1] is
//!
//! ```text
//! π_β(x) ∝ p(x) · L(x)^β,    log π_β = log prior + β · log likelihood,
//! ```
//!
//! where the prior is the latents' log probability terms and the likelihood, by default ([`ModelLikelihood`]), is
//! the observed nodes' terms and the factors' (as [`Model::log_likelihood`]). One run:
//!
//! 1. draws [`particles`](Smc::particles) independent particles from the prior, sampling each latent from its
//!    distribution in dependency order (β = 0);
//! 2. chooses the next β by bisection, so that the normalised effective sample size of the incremental weights
//!    `w = L(x)^(β' − β)` is [`ess_threshold`](Smc::ess_threshold) (or β' = 1 if that already keeps it above);
//! 3. adds the log of the weights' mean to the log marginal likelihood (the product of the stages' mean
//!    incremental weights estimates the evidence without bias);
//! 4. resamples the particles by their weights (systematic resampling);
//! 5. mutates each particle by Markov chain steps that leave π_β invariant, and repeats from 2 until β = 1.
//!
//! The continuous latents are mutated in unconstrained space (each latent mapped to ℝⁿ by its support's bijector,
//! as NUTS does), by the [`SmcKernel`]: by default an independent Metropolis–Hastings sampler whose proposal is a
//! multivariate normal fitted to the weighted particles, or a random walk with that covariance and a per-particle
//! scale tuned by the acceptance rate. Each stage takes steps until the mean correlation between the
//! particles' coordinates before and after is at most [`correlation_threshold`](Smc::correlation_threshold), up to
//! [`max_steps`](Smc::max_steps). Discrete latents are mutated after each continuous step, at the stage's β, by
//! [`Gibbs`] (finitely many values per element) or
//! [`DiscreteMetropolis`] (integer latents with infinitely many).
//!
//! Points where the model is impossible (a log probability of −∞ or NaN, or an evaluation that fails at an
//! impossible point such as a failed ODE solve) have zero weight and are never accepted as moves.
//!
//! [`Smc::sample`] runs [`chains`](Smc::chains) independent runs (in parallel with the `parallel` feature), so the
//! draws can be checked with R-hat like MCMC chains, and each run's log marginal likelihood can be compared.
//!
//! ```
//! use keanu_algorithms::smc::Smc;
//! use keanu_graph::{GraphBuilder, Model};
//! use keanu_tensor::DoubleTensor;
//!
//! // mu ~ N(0, 1), y ~ N(mu, 1) observed at 1: the evidence is N(1; 0, 2)
//! let b = GraphBuilder::new();
//! let mu = b.gaussian(0.0, 1.0);
//! let y = b.gaussian(mu, 1.0);
//! b.observe(y, DoubleTensor::scalar(1.0));
//! let mu = mu.id();
//! let (graph, state) = b.build()?;
//! let model = Model::new(graph, &state)?;
//!
//! let result = Smc::default().with_particles(500).with_chains(2).sample(&model, &state, &[mu], 1)?;
//! let exact = -0.5 * (2.0 * std::f64::consts::PI * 2.0).ln() - 0.25;
//! assert!((result.combined_log_marginal_likelihood() - exact).abs() < 0.1);
//! let mean = result.samples.chains()[0].f64(mu).unwrap().mean()?.to_scalar().unwrap();
//! assert!((mean - 0.5).abs() < 0.15);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{AlgorithmError, Result};
use crate::mcmc::chains::run_chains;
use crate::mcmc::compound::{DiscreteChoice, discrete_step_for};
use crate::mcmc::gibbs::{DiscreteMetropolis, DiscreteMetropolisSampler, Gibbs, GibbsSampler};
use crate::mcmc::is_impossible;
use crate::samples::{ChainSamples, NetworkSamples};
use crate::support::{check_nodes, evaluate_all};
use keanu_distributions::KeanuRng;
use keanu_graph::{GraphError, Model, NodeId, Parameterisation, Space, State, Value};
use keanu_tensor::DoubleTensor;

/// The likelihood term SMC tempers: `log π_β = log prior + β · log likelihood`.
///
/// [`ModelLikelihood`], the default, is the model's own log likelihood. Other implementations can replace it, for
/// example by an approximate likelihood that compares data simulated at the particle with the observations
/// (approximate Bayesian computation). The prior is always the latents' log probability terms.
pub trait TemperedLikelihood: Sync {
    /// The log likelihood at `state`, which holds a particle's latent values with their deterministic descendants
    /// computed. It may draw from `rng` (a simulation based likelihood does). −∞ or NaN, or an error for which
    /// [`GraphError::is_impossible_point`] is true, means the particle is impossible.
    fn log_likelihood(&self, model: &Model, state: &mut State, rng: &mut KeanuRng) -> Result<f64>;

    /// The model's nodes whose log probability terms make up this likelihood, if it is such a sum. Discrete latents
    /// are mutated one element at a time over their Markov blankets, which needs these terms; a likelihood that
    /// returns `None` (the default) can only be used with models whose latents are all continuous.
    fn model_terms(&self, model: &Model) -> Option<Vec<NodeId>> {
        let _ = model;
        None
    }

    /// Checks that this likelihood can be used with the model, before a run starts. By default, that the model has
    /// no [simulator](keanu_graph::op::Simulator) nodes, whose likelihood can't be evaluated
    /// ([`GraphError::NoLikelihood`]); an approximate likelihood for them ([`Abc`](crate::abc::Abc)) overrides it.
    ///
    /// # Errors
    ///
    /// An [`AlgorithmError`] saying why the likelihood can't be used.
    fn check(&self, model: &Model) -> Result<()> {
        crate::support::check_likelihood(model)
    }
}

/// The model's log likelihood ([`Model::log_likelihood`]): the observed nodes' log probability terms and the
/// factors' terms (−∞ if an observed deterministic node contradicts its observation).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelLikelihood;

impl TemperedLikelihood for ModelLikelihood {
    fn log_likelihood(&self, model: &Model, state: &mut State, _: &mut KeanuRng) -> Result<f64> {
        Ok(model.log_likelihood(state)?)
    }

    fn model_terms(&self, model: &Model) -> Option<Vec<NodeId>> {
        Some(
            model
                .observed()
                .iter()
                .chain(model.factors())
                .copied()
                .collect(),
        )
    }
}

/// How SMC mutates the continuous latents at each stage, in unconstrained space.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SmcKernel {
    /// Independent Metropolis–Hastings: every particle proposes a draw from a multivariate normal whose mean and
    /// covariance are the weighted particles' (PyMC's `IMH`, the default). It moves particles between modes
    /// easily, and works best when the tempered targets are roughly Gaussian.
    #[default]
    IndependentMetropolis,
    /// Random-walk Metropolis–Hastings: every particle proposes a move from where it is, a multivariate normal step
    /// with the weighted particles' covariance times a per-particle scale. The scales start at 2.38/√d and after
    /// each stage move by `exp(acceptance rate − 0.234)`, then halfway towards their mean (PyMC's `MH`).
    RandomWalkMetropolis,
}

/// Tempered sequential Monte Carlo settings (see the [module documentation](self)).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Smc {
    /// The number of particles of each run, which is the number of draws it returns (default 2000).
    pub particles: usize,
    /// The number of independent runs (default 4).
    pub chains: usize,
    /// The normalised effective sample size each stage's β is chosen to give, in (0, 1) (default 0.5).
    pub ess_threshold: f64,
    /// The kernel that mutates the continuous latents (default [`SmcKernel::IndependentMetropolis`]).
    pub kernel: SmcKernel,
    /// A stage's mutation stops when the mean over coordinates of the correlation (across particles) between the
    /// particles' coordinates before and after it is at most this (default 0.01). The correlation is signed, as
    /// PyMC's: once the particles have decorrelated it fluctuates around 0 by about 1/√particles, so the mutation
    /// stops soon after.
    pub correlation_threshold: f64,
    /// The most mutation steps per stage (default 25).
    pub max_steps: usize,
    /// The most stages (tempering steps) of a run before it fails (default 1000).
    pub max_stages: usize,
}

impl Default for Smc {
    fn default() -> Self {
        Self {
            particles: 2000,
            chains: 4,
            ess_threshold: 0.5,
            kernel: SmcKernel::IndependentMetropolis,
            correlation_threshold: 0.01,
            max_steps: 25,
            max_stages: 1000,
        }
    }
}

/// One run of SMC: its particles at β = 1 and how it got there.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SmcRun {
    /// The final particles, one sample each, with the joint log probability (prior plus likelihood) recorded.
    pub samples: NetworkSamples,
    /// The estimate of the log marginal likelihood: the sum over stages of the log mean incremental weight.
    pub log_marginal_likelihood: f64,
    /// The inverse temperature of each stage, from 0 (the prior) to 1 (the posterior).
    pub betas: Vec<f64>,
    /// The fraction of the continuous kernel's proposals accepted at each stage after the first (NaN without
    /// continuous latents).
    pub acceptance_rates: Vec<f64>,
    /// The number of mutation steps taken at each stage after the first.
    pub steps: Vec<usize>,
}

/// The result of several independent SMC runs ([`Smc::sample`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SmcResult {
    /// Each run's final particles, as chains (for R-hat and the other diagnostics).
    pub samples: ChainSamples,
    /// Each run's estimate of the log marginal likelihood.
    pub log_marginal_likelihood: Vec<f64>,
    /// Each run's inverse temperatures, from 0 to 1.
    pub betas: Vec<Vec<f64>>,
    /// Each run's acceptance rate at each stage after the first (see [`SmcRun::acceptance_rates`]).
    pub acceptance_rates: Vec<Vec<f64>>,
    /// Each run's number of mutation steps at each stage after the first.
    pub steps: Vec<Vec<usize>>,
}

impl SmcResult {
    /// The runs' estimates of the log marginal likelihood combined: the log of the mean of their marginal
    /// likelihoods (each run's estimate of the marginal likelihood is unbiased, so their mean is too).
    pub fn combined_log_marginal_likelihood(&self) -> f64 {
        let n = self.log_marginal_likelihood.len() as f64;
        log_sum_exp(&self.log_marginal_likelihood) - n.ln()
    }

    fn from_runs(runs: Vec<SmcRun>) -> Result<Self> {
        let mut samples = Vec::with_capacity(runs.len());
        let mut log_marginal_likelihood = Vec::with_capacity(runs.len());
        let mut betas = Vec::with_capacity(runs.len());
        let mut acceptance_rates = Vec::with_capacity(runs.len());
        let mut steps = Vec::with_capacity(runs.len());
        for run in runs {
            samples.push(run.samples);
            log_marginal_likelihood.push(run.log_marginal_likelihood);
            betas.push(run.betas);
            acceptance_rates.push(run.acceptance_rates);
            steps.push(run.steps);
        }
        Ok(Self {
            samples: ChainSamples::new(samples)?,
            log_marginal_likelihood,
            betas,
            acceptance_rates,
            steps,
        })
    }
}

/// Setters for each setting, for use as `Smc::default().with_...(...)`.
impl Smc {
    /// Sets [`particles`](Self::particles).
    #[must_use]
    pub fn with_particles(mut self, particles: usize) -> Self {
        self.particles = particles;
        self
    }

    /// Sets [`chains`](Self::chains).
    #[must_use]
    pub fn with_chains(mut self, chains: usize) -> Self {
        self.chains = chains;
        self
    }

    /// Sets [`ess_threshold`](Self::ess_threshold).
    #[must_use]
    pub fn with_ess_threshold(mut self, ess_threshold: f64) -> Self {
        self.ess_threshold = ess_threshold;
        self
    }

    /// Sets [`kernel`](Self::kernel).
    #[must_use]
    pub fn with_kernel(mut self, kernel: SmcKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Sets [`correlation_threshold`](Self::correlation_threshold).
    #[must_use]
    pub fn with_correlation_threshold(mut self, correlation_threshold: f64) -> Self {
        self.correlation_threshold = correlation_threshold;
        self
    }

    /// Sets [`max_steps`](Self::max_steps).
    #[must_use]
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Sets [`max_stages`](Self::max_stages).
    #[must_use]
    pub fn with_max_stages(mut self, max_stages: usize) -> Self {
        self.max_stages = max_stages;
        self
    }
}

impl Smc {
    /// Runs [`chains`](Self::chains) independent runs on the model's own likelihood ([`ModelLikelihood`]), each
    /// from its own copy of `state` with a random number generator seeded from `seed` and the run's index (as
    /// [`run_chains`]), recording the values of `nodes` at each final particle.
    ///
    /// # Errors
    ///
    /// As [`run`](Self::run), and [`AlgorithmError::InvalidArgument`] for 0 chains.
    pub fn sample(
        &self,
        model: &Model,
        state: &State,
        nodes: &[NodeId],
        seed: u64,
    ) -> Result<SmcResult> {
        self.sample_with(model, state, nodes, &ModelLikelihood, seed)
    }

    /// [`sample`](Self::sample) with another likelihood term to temper.
    ///
    /// # Errors
    ///
    /// As [`sample`](Self::sample).
    pub fn sample_with<L: TemperedLikelihood + ?Sized>(
        &self,
        model: &Model,
        state: &State,
        nodes: &[NodeId],
        likelihood: &L,
        seed: u64,
    ) -> Result<SmcResult> {
        if self.chains == 0 {
            return Err(AlgorithmError::InvalidArgument(
                "SMC needs at least one chain".into(),
            ));
        }
        let runs = run_chains(self.chains, seed, state, |state, rng| {
            self.run(model, state, nodes, likelihood, rng)
        })?;
        SmcResult::from_runs(runs)
    }

    /// One run from `state` (whose observed values and fixed parameters it uses; latents are drawn from their
    /// priors), recording the values of `nodes` at each final particle. `state` is left at the last particle.
    ///
    /// # Errors
    ///
    /// - [`AlgorithmError::InvalidSetting`] for fewer than 2 particles, an ESS threshold outside (0, 1), a
    ///   correlation threshold outside [0, 1], or a maximum of 0 steps or stages;
    /// - [`AlgorithmError::NoLatents`] if the model has no latents;
    /// - [`AlgorithmError::InvalidArgument`] if there are no nodes to record, a discrete latent can be mutated by
    ///   neither [`Gibbs`] nor [`DiscreteMetropolis`] (a multinomial, a mixture or a custom distribution), the
    ///   model has discrete latents and `likelihood` has no [`model_terms`](TemperedLikelihood::model_terms), no
    ///   particle drawn from the prior where the model can be evaluated could be found in 1000 attempts, every
    ///   particle has zero likelihood, or β did not reach 1 in [`max_stages`](Self::max_stages) stages;
    /// - [`AlgorithmError::Graph`] if a node is not in the graph or the model can't be evaluated
    ///   ([`GraphError::NoLikelihood`] if it has a simulator node and `likelihood` is not [`Abc`](crate::abc::Abc));
    /// - any error from the likelihood's [`check`](TemperedLikelihood::check).
    pub fn run<L: TemperedLikelihood + ?Sized>(
        &self,
        model: &Model,
        state: &mut State,
        nodes: &[NodeId],
        likelihood: &L,
        rng: &mut KeanuRng,
    ) -> Result<SmcRun> {
        self.check_settings()?;
        likelihood.check(model)?;
        if model.latents().is_empty() {
            return Err(AlgorithmError::NoLatents);
        }
        if nodes.is_empty() {
            return Err(AlgorithmError::InvalidArgument("no nodes to sample".into()));
        }
        check_nodes(model.graph(), nodes)?;
        evaluate_all(model.graph(), state, rng)?;
        let mut target = Target::new(model, state, likelihood, rng)?;
        let n = self.particles;
        let mut particles = Particles::from_prior(&target, n, state, rng)?;

        let mut beta = 0.0;
        let mut betas = vec![0.0];
        let mut log_marginal_likelihood = 0.0;
        let mut acceptance_rates = Vec::new();
        let mut steps = Vec::new();
        let d = target.dimension();
        let mut scales = vec![(2.38 / (d.max(1) as f64).sqrt()).min(1.0); n];
        while beta < 1.0 {
            if betas.len() > self.max_stages {
                return Err(AlgorithmError::InvalidArgument(format!(
                    "SMC did not reach β = 1 in {} stages (β = {beta})",
                    self.max_stages
                )));
            }
            let next = next_beta(beta, &particles.log_likelihood, self.ess_threshold);
            let log_weights: Vec<f64> = particles
                .log_likelihood
                .iter()
                .map(|&l| incremental(next - beta, l))
                .collect();
            let total = log_sum_exp(&log_weights);
            if total == f64::NEG_INFINITY {
                return Err(AlgorithmError::InvalidArgument(
                    "every SMC particle has zero likelihood (the likelihood is −∞ or the model can't be \
                     evaluated at every particle drawn)"
                        .into(),
                ));
            }
            log_marginal_likelihood += total - (n as f64).ln();
            let weights: Vec<f64> = log_weights.iter().map(|&w| (w - total).exp()).collect();
            let proposal = if d > 0 {
                Some(Proposal::fit(&particles, &weights, d)?)
            } else {
                None
            };
            particles = particles.resample(&systematic(&weights, rng));
            beta = next;
            betas.push(beta);
            target.set_beta(beta);
            let (rate, taken) = self.mutate(
                &mut target,
                &mut particles,
                proposal.as_ref(),
                &mut scales,
                state,
                rng,
            )?;
            acceptance_rates.push(rate);
            steps.push(taken);
        }

        let mut samples = NetworkSamples::new(model.graph(), nodes)?;
        for i in 0..n {
            if !target.position(particles.z(i), &particles.discrete[i], state)? {
                return Err(AlgorithmError::InvalidArgument(
                    "an SMC particle at β = 1 can't be evaluated".into(),
                ));
            }
            samples.record(state, particles.log_prior[i] + particles.log_likelihood[i])?;
        }
        Ok(SmcRun {
            samples,
            log_marginal_likelihood,
            betas,
            acceptance_rates,
            steps,
        })
    }

    fn check_settings(&self) -> Result<()> {
        let invalid = |message: &str| Err(AlgorithmError::InvalidSetting(message.into()));
        if self.particles < 2 {
            return invalid("SMC needs at least 2 particles");
        }
        if !(self.ess_threshold > 0.0 && self.ess_threshold < 1.0) {
            return invalid("the SMC ESS threshold must be in (0, 1)");
        }
        if !(0.0..=1.0).contains(&self.correlation_threshold) {
            return invalid("the SMC correlation threshold must be in [0, 1]");
        }
        if self.max_steps == 0 {
            return invalid("SMC needs at least 1 mutation step per stage");
        }
        if self.max_stages == 0 {
            return invalid("SMC needs at least 1 stage");
        }
        Ok(())
    }

    /// Mutates every particle at the target's β until the correlation threshold or the step limit, returning the
    /// continuous kernel's acceptance rate and the number of steps.
    fn mutate<L: TemperedLikelihood + ?Sized>(
        &self,
        target: &mut Target<'_, L>,
        particles: &mut Particles,
        proposal: Option<&Proposal>,
        scales: &mut [f64],
        state: &mut State,
        rng: &mut KeanuRng,
    ) -> Result<(f64, usize)> {
        let n = particles.len();
        let d = target.dimension();
        let beta = target.beta;
        let start = particles.coordinates();
        let mut accepted = vec![0usize; n];
        // the log proposal density of each particle's position, for the independent kernel
        let mut log_q: Vec<f64> = match (proposal, self.kernel) {
            (Some(p), SmcKernel::IndependentMetropolis) => {
                (0..n).map(|i| p.log_density(particles.z(i))).collect()
            }
            _ => Vec::new(),
        };
        let mut taken = 0;
        let mut epsilon = vec![0.0; d];
        while taken < self.max_steps {
            taken += 1;
            for i in 0..n {
                if let Some(p) = proposal {
                    for e in &mut epsilon {
                        *e = rng.next_gaussian();
                    }
                    let independent = self.kernel == SmcKernel::IndependentMetropolis;
                    let proposed = if independent {
                        p.draw(&p.mean, &epsilon, 1.0)
                    } else {
                        p.draw(particles.z(i), &epsilon, scales[i])
                    };
                    let current = particles.tempered(i, beta);
                    let evaluated =
                        target.evaluate(&proposed, &particles.discrete[i], state, rng)?;
                    let new = evaluated.tempered(beta);
                    let correction = if independent {
                        log_q[i] - (-0.5 * crate::support::dot(&epsilon, &epsilon))
                    } else {
                        0.0
                    };
                    let accept = !is_impossible(new)
                        && (is_impossible(current)
                            || rng.next_f64().ln() < new - current + correction);
                    if accept {
                        accepted[i] += 1;
                        particles.set(i, &proposed, evaluated);
                        if independent {
                            log_q[i] = -0.5 * crate::support::dot(&epsilon, &epsilon);
                        }
                    }
                }
                if target.has_discrete() {
                    target.sweep(particles, i, state, rng)?;
                }
            }
            if mean_correlation(&start, &particles.coordinates()) <= self.correlation_threshold {
                break;
            }
        }
        if proposal.is_none() {
            return Ok((f64::NAN, taken));
        }
        let rates: Vec<f64> = accepted.iter().map(|&a| a as f64 / taken as f64).collect();
        if self.kernel == SmcKernel::RandomWalkMetropolis {
            for (s, rate) in scales.iter_mut().zip(&rates) {
                *s = (s.ln() + (rate - 0.234)).exp();
            }
            let mean = scales.iter().sum::<f64>() / n as f64;
            for s in scales.iter_mut() {
                *s = 0.5 * (*s + mean);
            }
        }
        Ok((rates.iter().sum::<f64>() / n as f64, taken))
    }
}

/// The tempered target and the steps that mutate it: the model's parameterisation of its continuous latents, the
/// likelihood, and the discrete steps.
struct Target<'m, L: TemperedLikelihood + ?Sized> {
    model: &'m Model,
    param: Option<Parameterisation<'m>>,
    discrete: Vec<NodeId>,
    likelihood: &'m L,
    gibbs: Option<GibbsSampler<'m>>,
    walk: Option<DiscreteMetropolisSampler<'m>>,
    beta: f64,
}

/// A particle's log prior (constrained, without the Jacobian), log |det J| and log likelihood.
#[derive(Debug, Clone, Copy)]
struct Evaluated {
    log_prior: f64,
    log_det: f64,
    log_likelihood: f64,
}

impl Evaluated {
    const IMPOSSIBLE: Self = Self {
        log_prior: f64::NEG_INFINITY,
        log_det: 0.0,
        log_likelihood: f64::NEG_INFINITY,
    };

    /// The tempered log density in unconstrained space.
    fn tempered(&self, beta: f64) -> f64 {
        tempered(self.log_prior + self.log_det, beta, self.log_likelihood)
    }
}

/// `prior + β · likelihood`, −∞ if either is impossible.
fn tempered(prior: f64, beta: f64, likelihood: f64) -> f64 {
    if is_impossible(prior) || is_impossible(likelihood) {
        f64::NEG_INFINITY
    } else {
        prior + beta * likelihood
    }
}

/// The incremental log weight `Δβ · log likelihood`, −∞ for an impossible particle.
fn incremental(delta: f64, log_likelihood: f64) -> f64 {
    if is_impossible(log_likelihood) {
        f64::NEG_INFINITY
    } else {
        delta * log_likelihood
    }
}

/// −∞ for an evaluation that failed at an impossible point.
fn possible(result: Result<f64>) -> Result<f64> {
    match result {
        Err(AlgorithmError::Graph(e)) if e.is_impossible_point() => Ok(f64::NEG_INFINITY),
        Ok(x) if !x.is_finite() => Ok(f64::NEG_INFINITY),
        result => result,
    }
}

impl<'m, L: TemperedLikelihood + ?Sized> Target<'m, L> {
    fn new(
        model: &'m Model,
        state: &mut State,
        likelihood: &'m L,
        rng: &mut KeanuRng,
    ) -> Result<Self> {
        let graph = model.graph();
        let continuous = model.continuous_latents();
        let param = if continuous.is_empty() {
            None
        } else {
            // a support that moves with a latent has no fixed transform; such models are mutated in constrained
            // space, where moves out of the support are rejected
            Some(
                match Parameterisation::for_latents(model, state, &continuous, Space::Unconstrained)
                {
                    Err(GraphError::Transform(_)) => Parameterisation::for_latents(
                        model,
                        state,
                        &continuous,
                        Space::Constrained,
                    )?,
                    param => param?,
                },
            )
        };
        let discrete = model.discrete_latents();
        let (mut gibbs, mut walk) = (Vec::new(), Vec::new());
        for &id in &discrete {
            match discrete_step_for(graph, state, id) {
                DiscreteChoice::Gibbs => gibbs.push(id),
                DiscreteChoice::Walk => walk.push(id),
                DiscreteChoice::MetropolisHastings => {
                    return Err(AlgorithmError::InvalidArgument(format!(
                        "SMC mutates discrete latents with Gibbs or DiscreteMetropolis steps, and neither can \
                         update {}",
                        graph.describe(id)
                    )));
                }
            }
        }
        let terms = if discrete.is_empty() {
            Vec::new()
        } else {
            likelihood.model_terms(model).ok_or_else(|| {
                AlgorithmError::InvalidArgument(
                    "SMC mutates discrete latents over their Markov blankets, which needs a likelihood made of \
                     the model's terms (TemperedLikelihood::model_terms)"
                        .into(),
                )
            })?
        };
        let gibbs = if gibbs.is_empty() {
            None
        } else {
            let mut sampler = Gibbs::new(&gibbs).sampler(model, state, &gibbs, rng)?;
            sampler.temper(&terms);
            Some(sampler)
        };
        let walk = if walk.is_empty() {
            None
        } else {
            let mut sampler = DiscreteMetropolis::new(&walk).sampler(model, state, &walk, rng)?;
            sampler.temper(&terms);
            Some(sampler)
        };
        Ok(Self {
            model,
            param,
            discrete,
            likelihood,
            gibbs,
            walk,
            beta: 0.0,
        })
    }

    /// The number of unconstrained coordinates of the continuous latents.
    fn dimension(&self) -> usize {
        self.param.as_ref().map_or(0, Parameterisation::dimension)
    }

    fn has_discrete(&self) -> bool {
        !self.discrete.is_empty()
    }

    fn set_beta(&mut self, beta: f64) {
        self.beta = beta;
        if let Some(gibbs) = &mut self.gibbs {
            gibbs.set_beta(beta);
        }
        if let Some(walk) = &mut self.walk {
            walk.set_beta(beta);
        }
    }

    /// Sets the latents to a particle's values and recomputes their descendants; false where that fails at an
    /// impossible point.
    fn position(&self, z: &[f64], discrete: &[Value], state: &mut State) -> Result<bool> {
        let mut values: Vec<(NodeId, Value)> = self
            .discrete
            .iter()
            .copied()
            .zip(discrete.iter().cloned())
            .collect();
        if let Some(param) = &self.param {
            values.extend(
                param
                    .latents()
                    .iter()
                    .copied()
                    .zip(param.to_constrained(z)?.into_iter().map(Value::F64)),
            );
        }
        match self.model.set_and_cascade(state, values) {
            Ok(()) => Ok(true),
            Err(e) if e.is_impossible_point() => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    /// The prior, Jacobian and likelihood terms at the state's current position, whose unconstrained coordinates
    /// are `z`.
    fn terms(&self, z: &[f64], state: &mut State, rng: &mut KeanuRng) -> Result<Evaluated> {
        let log_prior = possible(
            self.model
                .local_log_prob(state, self.model.latents())
                .map_err(Into::into),
        )?;
        if is_impossible(log_prior) {
            return Ok(Evaluated::IMPOSSIBLE);
        }
        let log_det = match &self.param {
            Some(param) => param.log_det_jacobian(z)?,
            None => 0.0,
        };
        let log_likelihood = possible(self.likelihood.log_likelihood(self.model, state, rng))?;
        Ok(Evaluated {
            log_prior,
            log_det,
            log_likelihood,
        })
    }

    /// Moves the state to a particle and evaluates it.
    fn evaluate(
        &self,
        z: &[f64],
        discrete: &[Value],
        state: &mut State,
        rng: &mut KeanuRng,
    ) -> Result<Evaluated> {
        if self.position(z, discrete, state)? {
            self.terms(z, state, rng)
        } else {
            Ok(Evaluated::IMPOSSIBLE)
        }
    }

    /// Updates particle `i`'s discrete latents by one sweep of each discrete step at the current β.
    fn sweep(
        &mut self,
        particles: &mut Particles,
        i: usize,
        state: &mut State,
        rng: &mut KeanuRng,
    ) -> Result<()> {
        if !self.position(particles.z(i), &particles.discrete[i], state)? {
            return Ok(());
        }
        if let Some(gibbs) = &self.gibbs {
            gibbs.sweep(state, f64::NAN, rng)?;
        }
        if let Some(walk) = &mut self.walk {
            walk.sweep(state, f64::NAN, rng)?;
        }
        let discrete: Vec<Value> = self
            .discrete
            .iter()
            .map(|&id| {
                state
                    .value(id)
                    .cloned()
                    .ok_or(AlgorithmError::Graph(GraphError::MissingValue(id)))
            })
            .collect::<Result<_>>()?;
        let z = particles.z(i).to_vec();
        let evaluated = self.terms(&z, state, rng)?;
        particles.discrete[i] = discrete;
        particles.set(i, &z, evaluated);
        Ok(())
    }
}

/// The particles: each one's unconstrained continuous coordinates, discrete latent values and log terms.
#[derive(Debug, Clone)]
struct Particles {
    d: usize,
    /// Row-major, one row of `d` coordinates per particle.
    z: Vec<f64>,
    discrete: Vec<Vec<Value>>,
    /// The constrained log prior, without the Jacobian.
    log_prior: Vec<f64>,
    log_det: Vec<f64>,
    log_likelihood: Vec<f64>,
}

impl Particles {
    /// The most prior draws tried for each particle before giving up.
    const ATTEMPTS: usize = 1000;

    /// `n` particles drawn from the prior, redrawing any where the model can't be evaluated (or whose
    /// unconstrained value would be infinite).
    fn from_prior<L: TemperedLikelihood + ?Sized>(
        target: &Target<'_, L>,
        n: usize,
        state: &mut State,
        rng: &mut KeanuRng,
    ) -> Result<Self> {
        let d = target.dimension();
        let mut particles = Self {
            d,
            z: Vec::with_capacity(n * d),
            discrete: Vec::with_capacity(n),
            log_prior: Vec::with_capacity(n),
            log_det: Vec::with_capacity(n),
            log_likelihood: Vec::with_capacity(n),
        };
        let model = target.model;
        let graph = model.graph();
        'particles: for _ in 0..n {
            'attempts: for _ in 0..Self::ATTEMPTS {
                for &latent in model.latents() {
                    let value = state.sample_prior(graph, latent, rng)?;
                    match model.set_and_cascade(state, [(latent, value)]) {
                        Err(e) if e.is_impossible_point() => continue 'attempts,
                        result => result?,
                    }
                }
                let z = match &target.param {
                    Some(param) => match param.from_state(state) {
                        Err(GraphError::Transform(_)) => continue 'attempts,
                        z => z?,
                    },
                    None => Vec::new(),
                };
                let evaluated = target.terms(&z, state, rng)?;
                if is_impossible(evaluated.log_prior) {
                    continue 'attempts;
                }
                particles.z.extend_from_slice(&z);
                particles.discrete.push(
                    target
                        .discrete
                        .iter()
                        .map(|&id| {
                            state
                                .value(id)
                                .cloned()
                                .ok_or(AlgorithmError::Graph(GraphError::MissingValue(id)))
                        })
                        .collect::<Result<_>>()?,
                );
                particles.log_prior.push(evaluated.log_prior);
                particles.log_det.push(evaluated.log_det);
                particles.log_likelihood.push(evaluated.log_likelihood);
                continue 'particles;
            }
            return Err(AlgorithmError::InvalidArgument(format!(
                "SMC could not draw a particle from the prior where the model can be evaluated in {} attempts",
                Self::ATTEMPTS
            )));
        }
        Ok(particles)
    }

    fn len(&self) -> usize {
        self.log_prior.len()
    }

    fn z(&self, i: usize) -> &[f64] {
        &self.z[i * self.d..(i + 1) * self.d]
    }

    fn tempered(&self, i: usize, beta: f64) -> f64 {
        tempered(
            self.log_prior[i] + self.log_det[i],
            beta,
            self.log_likelihood[i],
        )
    }

    fn set(&mut self, i: usize, z: &[f64], evaluated: Evaluated) {
        self.z[i * self.d..(i + 1) * self.d].copy_from_slice(z);
        self.log_prior[i] = evaluated.log_prior;
        self.log_det[i] = evaluated.log_det;
        self.log_likelihood[i] = evaluated.log_likelihood;
    }

    /// The particles at `indices`, in that order.
    fn resample(&self, indices: &[usize]) -> Self {
        let mut z = Vec::with_capacity(self.z.len());
        for &i in indices {
            z.extend_from_slice(self.z(i));
        }
        let pick = |v: &[f64]| indices.iter().map(|&i| v[i]).collect();
        Self {
            d: self.d,
            z,
            discrete: indices.iter().map(|&i| self.discrete[i].clone()).collect(),
            log_prior: pick(&self.log_prior),
            log_det: pick(&self.log_det),
            log_likelihood: pick(&self.log_likelihood),
        }
    }

    /// Every particle's coordinates: the continuous ones, then each discrete element as a number. One row per
    /// particle.
    fn coordinates(&self) -> Vec<Vec<f64>> {
        (0..self.len())
            .map(|i| {
                let mut row = self.z(i).to_vec();
                for value in &self.discrete[i] {
                    match value {
                        Value::I64(t) => row.extend(t.as_slice().iter().map(|&x| x as f64)),
                        Value::Bool(t) => {
                            row.extend(t.as_slice().iter().map(|&x| f64::from(u8::from(x))))
                        }
                        Value::F64(t) => row.extend_from_slice(t.as_slice()),
                        _ => {}
                    }
                }
                row
            })
            .collect()
    }
}

/// A multivariate normal fitted to the weighted particles: the proposal of the independent kernel, and the shape of
/// the random walk's steps.
#[derive(Debug, Clone)]
struct Proposal {
    d: usize,
    mean: Vec<f64>,
    /// The lower Cholesky factor of the covariance, row-major.
    factor: Vec<f64>,
}

impl Proposal {
    fn fit(particles: &Particles, weights: &[f64], d: usize) -> Result<Self> {
        let mut mean = vec![0.0; d];
        for (i, &w) in weights.iter().enumerate() {
            for (m, z) in mean.iter_mut().zip(particles.z(i)) {
                *m += w * z;
            }
        }
        let mut covariance = vec![0.0; d * d];
        for (i, &w) in weights.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let z = particles.z(i);
            for r in 0..d {
                let dr = z[r] - mean[r];
                for c in 0..=r {
                    covariance[r * d + c] += w * dr * (z[c] - mean[c]);
                }
            }
        }
        for r in 0..d {
            for c in 0..r {
                covariance[c * d + r] = covariance[r * d + c];
            }
        }
        // a little jitter, more if the particles have collapsed onto a lower dimensional set
        let scale = (0..d).map(|i| covariance[i * d + i]).sum::<f64>() / d as f64;
        let mut jitter = 1e-10
            * if scale > 0.0 && scale.is_finite() {
                scale
            } else {
                1.0
            };
        for _ in 0..12 {
            let mut a = covariance.clone();
            for i in 0..d {
                a[i * d + i] += jitter;
            }
            if let Some(factor) = cholesky(&a, d) {
                return Ok(Self { d, mean, factor });
            }
            jitter *= 100.0;
        }
        Err(AlgorithmError::InvalidArgument(
            "the SMC particles' covariance is not positive definite (or not finite)".into(),
        ))
    }

    /// `centre + scale · L ε`.
    fn draw(&self, centre: &[f64], epsilon: &[f64], scale: f64) -> Vec<f64> {
        (0..self.d)
            .map(|r| {
                let step: f64 = (0..=r)
                    .map(|c| self.factor[r * self.d + c] * epsilon[c])
                    .sum();
                centre[r] + scale * step
            })
            .collect()
    }

    /// The log density at `z`, up to a constant: −½ ‖L⁻¹ (z − mean)‖².
    fn log_density(&self, z: &[f64]) -> f64 {
        let mut y = vec![0.0; self.d];
        for r in 0..self.d {
            let sum: f64 = (0..r).map(|c| self.factor[r * self.d + c] * y[c]).sum();
            y[r] = (z[r] - self.mean[r] - sum) / self.factor[r * self.d + r];
        }
        -0.5 * crate::support::dot(&y, &y)
    }
}

/// The lower Cholesky factor of a symmetric n × n row-major matrix, `None` if it is not positive definite.
fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    DoubleTensor::new(&[n, n], a.to_vec())
        .and_then(|m| m.cholesky())
        .ok()
        .map(|l| l.as_slice().to_vec())
        .filter(|l| (0..n).all(|i| l[i * n + i] > 0.0 && l[i * n + i].is_finite()))
}

/// log Σ exp(xᵢ), −∞ for no terms or all −∞.
fn log_sum_exp(x: &[f64]) -> f64 {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY || max.is_nan() {
        return max;
    }
    if max == f64::INFINITY {
        return f64::INFINITY;
    }
    max + x.iter().map(|&v| (v - max).exp()).sum::<f64>().ln()
}

/// The normalised effective sample size, (Σw)² / (n Σw²), of weights `exp(log_weights)`; 0 if every weight is 0.
fn normalised_ess(log_weights: &[f64]) -> f64 {
    let max = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return 0.0;
    }
    let (sum, sum_squares) = log_weights.iter().fold((0.0, 0.0), |(s, s2), &w| {
        let w = (w - max).exp();
        (s + w, s2 + w * w)
    });
    sum * sum / (sum_squares * log_weights.len() as f64)
}

/// The next inverse temperature after `beta`: 1 if the incremental weights there keep the normalised ESS at least
/// `threshold`, otherwise (by bisection) where it falls to `threshold`. Always more than `beta`.
fn next_beta(beta: f64, log_likelihood: &[f64], threshold: f64) -> f64 {
    let ess = |next: f64| {
        let log_weights: Vec<f64> = log_likelihood
            .iter()
            .map(|&l| incremental(next - beta, l))
            .collect();
        normalised_ess(&log_weights)
    };
    if ess(1.0) >= threshold {
        return 1.0;
    }
    let (mut low, mut high) = (beta, 1.0);
    while high - low > 1e-12 * high.max(1e-300) {
        let mid = 0.5 * (low + high);
        if mid <= low || mid >= high {
            break;
        }
        if ess(mid) >= threshold {
            low = mid;
        } else {
            high = mid;
        }
    }
    if low > beta { low } else { high }
}

/// Systematic resampling: `n` indices, each particle appearing about n wᵢ times, from one uniform draw.
fn systematic(weights: &[f64], rng: &mut KeanuRng) -> Vec<usize> {
    let n = weights.len();
    let u = rng.next_f64();
    let mut indices = Vec::with_capacity(n);
    let mut cumulative = 0.0;
    let mut j = 0;
    for (i, &w) in weights.iter().enumerate() {
        cumulative += w * n as f64;
        while j < n && (j as f64 + u) < cumulative {
            indices.push(i);
            j += 1;
        }
    }
    // rounding can leave the total weight just under 1: fill with the last particle with weight
    let last = weights.iter().rposition(|&w| w > 0.0).unwrap_or(n - 1);
    indices.resize(n, last);
    indices
}

/// The mean over coordinates of the correlation across particles between each coordinate's values in
/// `before` and `after`. A coordinate that was the same in every particle before is skipped; one that is the same
/// in every particle after (but wasn't before) counts as fully correlated. 1 if every coordinate is skipped.
fn mean_correlation(before: &[Vec<f64>], after: &[Vec<f64>]) -> f64 {
    let n = before.len() as f64;
    let dims = before.first().map_or(0, Vec::len);
    let (mut total, mut counted) = (0.0, 0usize);
    for k in 0..dims {
        let mean_b = before.iter().map(|r| r[k]).sum::<f64>() / n;
        let mean_a = after.iter().map(|r| r[k]).sum::<f64>() / n;
        let (mut sab, mut sbb, mut saa) = (0.0, 0.0, 0.0);
        for (b, a) in before.iter().zip(after) {
            let (db, da) = (b[k] - mean_b, a[k] - mean_a);
            sab += db * da;
            sbb += db * db;
            saa += da * da;
        }
        if sbb <= 0.0 {
            continue;
        }
        counted += 1;
        total += if saa <= 0.0 {
            1.0
        } else {
            sab / (sbb * saa).sqrt()
        };
    }
    if counted == 0 {
        1.0
    } else {
        total / counted as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_beta_reaches_the_threshold() {
        let log_likelihood: Vec<f64> = (0..1000).map(|i| -0.01 * i as f64 * i as f64).collect();
        let beta = next_beta(0.0, &log_likelihood, 0.5);
        assert!(beta > 0.0 && beta < 1.0);
        let log_weights: Vec<f64> = log_likelihood.iter().map(|&l| beta * l).collect();
        assert!((normalised_ess(&log_weights) - 0.5).abs() < 1e-6);
        // a flat likelihood goes straight to 1
        assert_eq!(next_beta(0.3, &[-1.0; 10], 0.5), 1.0);
    }

    #[test]
    fn next_beta_progresses_past_impossible_particles() {
        // 80% of the particles are impossible, so no β > 0 keeps half the ESS: β still moves on
        let mut log_likelihood = vec![f64::NEG_INFINITY; 80];
        log_likelihood.extend((0..20).map(|i| -(i as f64)));
        let beta = next_beta(0.2, &log_likelihood, 0.5);
        assert!(beta > 0.2 && beta < 0.21);
    }

    #[test]
    fn systematic_resampling_keeps_proportions() {
        let mut rng = KeanuRng::seed_from_u64(3);
        let weights = [0.5, 0.0, 0.25, 0.25];
        let indices = systematic(&weights, &mut rng);
        assert_eq!(indices.len(), 4);
        assert_eq!(indices.iter().filter(|&&i| i == 0).count(), 2);
        assert!(!indices.contains(&1));
        assert_eq!(indices.iter().filter(|&&i| i == 2).count(), 1);
    }

    #[test]
    fn correlation_of_unchanged_and_independent_coordinates() {
        let before: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64, 1.0]).collect();
        assert!((mean_correlation(&before, &before) - 1.0).abs() < 1e-12);
        let mut rng = KeanuRng::seed_from_u64(1);
        let after: Vec<Vec<f64>> = (0..100).map(|_| vec![rng.next_gaussian(), 1.0]).collect();
        assert!(mean_correlation(&before, &after).abs() < 0.3);
    }

    #[test]
    fn proposal_density_and_draws() {
        let factor = vec![2.0, 0.0, 1.0, 3.0];
        let p = Proposal {
            d: 2,
            mean: vec![1.0, -1.0],
            factor,
        };
        let epsilon = [0.5, -0.25];
        let z = p.draw(&p.mean, &epsilon, 1.0);
        assert!((z[0] - 2.0).abs() < 1e-12 && (z[1] - (-1.0 + 0.5 - 0.75)).abs() < 1e-12);
        let expected = -0.5 * (0.25 + 0.0625);
        assert!((p.log_density(&z) - expected).abs() < 1e-12);
    }
}
