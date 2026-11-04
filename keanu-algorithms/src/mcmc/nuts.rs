//! The No-U-Turn sampler (NUTS): Hamiltonian Monte Carlo that chooses its own trajectory length, with dual
//! averaging step size adaptation and an adaptive diagonal mass matrix (Java's `NUTS`).

use crate::error::{AlgorithmError, Result};
use crate::mcmc::{SamplingAlgorithm, SamplingOptions, generate, is_impossible};
use crate::samples::NetworkSamples;
use crate::support::{check_nodes, evaluate_all, f64_values};
use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, Model, NodeId, State};
use keanu_tensor::DoubleTensor;
use std::cell::RefCell;
use std::rc::Rc;

/// The adaptive diagonal quadratic potential (mass matrix) settings (Java's `AdaptiveQuadraticPotential`).
///
/// During adaptation the potential's variance tracks the running variance of the samples, using two windows of
/// `adaptation_window` samples: the variance in use comes from the previous window, so it isn't biased by the
/// current position.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct QuadraticPotential {
    /// The starting estimate of each latent's mean.
    pub initial_mean: f64,
    /// The starting estimate of each latent's variance.
    pub initial_variance: f64,
    /// How many pseudo-samples the initial mean and variance count as.
    pub initial_weight: f64,
    /// Samples per adaptation window.
    pub adaptation_window: usize,
}

/// Setters for each setting, for use as `QuadraticPotential::default().with_...(...)`.
impl QuadraticPotential {
    /// Sets [`initial_mean`](Self::initial_mean).
    #[must_use]
    pub fn with_initial_mean(mut self, initial_mean: f64) -> Self {
        self.initial_mean = initial_mean;
        self
    }

    /// Sets [`initial_variance`](Self::initial_variance).
    #[must_use]
    pub fn with_initial_variance(mut self, initial_variance: f64) -> Self {
        self.initial_variance = initial_variance;
        self
    }

    /// Sets [`initial_weight`](Self::initial_weight).
    #[must_use]
    pub fn with_initial_weight(mut self, initial_weight: f64) -> Self {
        self.initial_weight = initial_weight;
        self
    }

    /// Sets [`adaptation_window`](Self::adaptation_window).
    #[must_use]
    pub fn with_adaptation_window(mut self, adaptation_window: usize) -> Self {
        self.adaptation_window = adaptation_window;
        self
    }
}

impl Default for QuadraticPotential {
    fn default() -> Self {
        Self {
            initial_mean: 0.0,
            initial_variance: 1.0,
            initial_weight: 10.0,
            adaptation_window: 100,
        }
    }
}

/// NUTS configuration. Defaults follow Java's `NUTS.builder()`. The configuration holds no state, so it can be
/// reused for any number of runs and shared by parallel chains; each run adapts its own step size and potential.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Nuts {
    /// Samples during which the step size and potential are adapted.
    pub adapt_count: usize,
    /// Whether to adapt the step size during the first `adapt_count` samples.
    pub adapt_step_size: bool,
    /// The initial step size; by default 0.25 / n^(1/4) for n latent elements.
    pub initial_step_size: Option<f64>,
    /// The kinetic energy (mass matrix) settings.
    pub potential: QuadraticPotential,
    /// Whether to adapt the potential during the first `adapt_count` samples.
    pub adapt_potential: bool,
    /// The acceptance probability step size adaptation aims for.
    pub target_acceptance_prob: f64,
    /// A trajectory whose energy changes by at least this much is divergent.
    pub max_energy_change: f64,
    /// The maximum tree height (at most 2^height leapfrog steps per sample).
    pub max_tree_height: usize,
    /// Record step sizes, log probs, mean tree acceptance and tree sizes for every step.
    pub save_statistics: bool,
}

/// Setters for each setting, for use as `Nuts::default().with_...(...)`.
impl Nuts {
    /// Sets [`adapt_count`](Self::adapt_count).
    #[must_use]
    pub fn with_adapt_count(mut self, adapt_count: usize) -> Self {
        self.adapt_count = adapt_count;
        self
    }

    /// Sets [`adapt_step_size`](Self::adapt_step_size).
    #[must_use]
    pub fn with_adapt_step_size(mut self, adapt_step_size: bool) -> Self {
        self.adapt_step_size = adapt_step_size;
        self
    }

    /// Sets [`initial_step_size`](Self::initial_step_size) to `Some(initial_step_size)`.
    #[must_use]
    pub fn with_initial_step_size(mut self, initial_step_size: f64) -> Self {
        self.initial_step_size = Some(initial_step_size);
        self
    }

    /// Sets [`potential`](Self::potential).
    #[must_use]
    pub fn with_potential(mut self, potential: QuadraticPotential) -> Self {
        self.potential = potential;
        self
    }

    /// Sets [`adapt_potential`](Self::adapt_potential).
    #[must_use]
    pub fn with_adapt_potential(mut self, adapt_potential: bool) -> Self {
        self.adapt_potential = adapt_potential;
        self
    }

    /// Sets [`target_acceptance_prob`](Self::target_acceptance_prob).
    #[must_use]
    pub fn with_target_acceptance_prob(mut self, target_acceptance_prob: f64) -> Self {
        self.target_acceptance_prob = target_acceptance_prob;
        self
    }

    /// Sets [`max_energy_change`](Self::max_energy_change).
    #[must_use]
    pub fn with_max_energy_change(mut self, max_energy_change: f64) -> Self {
        self.max_energy_change = max_energy_change;
        self
    }

    /// Sets [`max_tree_height`](Self::max_tree_height).
    #[must_use]
    pub fn with_max_tree_height(mut self, max_tree_height: usize) -> Self {
        self.max_tree_height = max_tree_height;
        self
    }

    /// Sets [`save_statistics`](Self::save_statistics).
    #[must_use]
    pub fn with_save_statistics(mut self, save_statistics: bool) -> Self {
        self.save_statistics = save_statistics;
        self
    }
}

impl Default for Nuts {
    fn default() -> Self {
        Self {
            adapt_count: 1000,
            adapt_step_size: true,
            initial_step_size: None,
            potential: QuadraticPotential::default(),
            adapt_potential: true,
            target_acceptance_prob: 0.8,
            max_energy_change: 1000.0,
            max_tree_height: 10,
            save_statistics: false,
        }
    }
}

/// What NUTS recorded while sampling (Java's NUTS `Statistics`).
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct NutsStatistics {
    /// The step size used at each step (when saving statistics).
    pub step_size: Vec<f64>,
    /// The mean Metropolis acceptance probability last used for step size adaptation, recorded at each step (as
    /// Java does). It is recorded before the step's own tree is used for adaptation, so `mean_tree_accept[i]` is the
    /// acceptance of step i - 1's tree (0 at step 0). Once adaptation ends (after `adapt_count` steps) it stays at
    /// the last adapted tree's value, and without step size adaptation it is always 0.
    pub mean_tree_accept: Vec<f64>,
    /// The log probability of each step's sample.
    pub log_prob: Vec<f64>,
    /// The number of leapfrog steps in each step's tree.
    pub tree_size: Vec<usize>,
    /// The (0-based) steps after adaptation whose trajectory diverged. Always recorded.
    pub divergent_steps: Vec<usize>,
}

impl Nuts {
    /// The default settings, as Java's NUTS builder.
    pub fn new() -> Self {
        Self::default()
    }

    fn validate(&self) -> Result<()> {
        let invalid = |m: &str| Err(AlgorithmError::InvalidSetting(m.into()));
        if !(0.0..=1.0).contains(&self.target_acceptance_prob) {
            return invalid("the target acceptance probability must be between 0 and 1");
        }
        if matches!(self.initial_step_size, Some(s) if s.is_nan() || s <= 0.0) {
            return invalid("the initial step size must be greater than 0");
        }
        if self.max_energy_change.is_nan() || self.max_energy_change <= 0.0 {
            return invalid("the max energy change must be greater than 0");
        }
        if self.max_tree_height == 0 {
            return invalid("the max tree height must be greater than 0");
        }
        if self.potential.adaptation_window <= 1 {
            return invalid("the potential's adaptation window must be greater than 1");
        }
        // a zero weight or variance gives a zero mass matrix entry, so momentum / mass is NaN and the chain freezes
        let positive = |x: f64| x.is_finite() && x > 0.0;
        if !positive(self.potential.initial_weight) {
            return invalid("the potential's initial weight must be finite and greater than 0");
        }
        if !positive(self.potential.initial_variance) {
            return invalid("the potential's initial variance must be finite and greater than 0");
        }
        Ok(())
    }

    /// A chain starting from the current state, recording the values of `nodes`. Every latent must be continuous.
    pub fn sampler<'m>(
        &self,
        model: &'m Model,
        state: &mut State,
        nodes: &[NodeId],
        rng: &mut KeanuRng,
    ) -> Result<NutsSampler<'m>> {
        self.validate()?;
        if nodes.is_empty() {
            return Err(AlgorithmError::InvalidArgument("no nodes to sample".into()));
        }
        check_nodes(model.graph(), nodes)?;
        if model.latents().is_empty() {
            return Err(AlgorithmError::NoLatents);
        }
        let discrete = model.discrete_latents();
        if !discrete.is_empty() {
            return Err(AlgorithmError::DiscreteLatents(
                discrete
                    .iter()
                    .map(|id| describe(model.graph(), *id))
                    .collect(),
            ));
        }
        let latents = model.continuous_latents();
        evaluate_all(model.graph(), state, rng)?;

        let position: Vec<DoubleTensor> = latents
            .iter()
            .map(|&id| state.f64(id).cloned().ok_or_else(|| missing(id)))
            .collect::<Result<_>>()?;
        let (log_prob, gradient) = log_prob_and_gradient_at(model, state, &latents)?;
        if is_impossible(log_prob) {
            return Err(AlgorithmError::ImpossibleStart);
        }

        let step_size = self.initial_step_size.unwrap_or_else(|| {
            let n: usize = position.iter().map(DoubleTensor::len).sum();
            0.25 / (n as f64).powf(0.25)
        });

        Ok(NutsSampler {
            model,
            config: *self,
            potential: PotentialState::new(&self.potential, &position),
            step_size: AdaptiveStepSize::new(
                step_size,
                self.target_acceptance_prob,
                self.adapt_count,
            ),
            latents,
            nodes: nodes.to_vec(),
            proposal: Candidate {
                position: Rc::new(position),
                gradient: Rc::new(gradient),
                log_prob,
            },
            step_count: 0,
            statistics: NutsStatistics::default(),
        })
    }

    /// Runs a chain of `count` steps from the current state, recording every step's values of `nodes`.
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

    /// Runs a chain, dropping and down-sampling as `options` says, and returns what NUTS recorded as well.
    pub fn sample_with_statistics(
        &self,
        model: &Model,
        state: &mut State,
        nodes: &[NodeId],
        count: usize,
        options: SamplingOptions,
        rng: &mut KeanuRng,
    ) -> Result<(NetworkSamples, NutsStatistics)> {
        let mut sampler = self.sampler(model, state, nodes, rng)?;
        let samples = generate(&mut sampler, state, count, options, rng)?;
        Ok((samples, sampler.statistics))
    }
}

fn describe(graph: &Graph, id: NodeId) -> String {
    graph[id]
        .label()
        .map(str::to_owned)
        .unwrap_or_else(|| format!("#{}", id.index()))
}

fn missing(id: NodeId) -> AlgorithmError {
    keanu_graph::GraphError::MissingValue(id).into()
}

/// The joint log probability and its gradient with respect to each latent, in latent order, in one pass. Where
/// the log probability is not finite the gradient is meaningless (the trajectory diverges anyway), so zeros are used.
fn log_prob_and_gradient_at(
    model: &Model,
    state: &State,
    latents: &[NodeId],
) -> Result<(f64, Vec<DoubleTensor>)> {
    let zeros = |id: NodeId| DoubleTensor::zeros(model.graph()[id].shape());
    let (log_prob, gradient) = model.log_prob_and_gradient(state)?;
    let gradient = match gradient {
        // the latents are distinct, so each gradient is moved out exactly once
        Some(mut gradient) if log_prob.is_finite() => latents
            .iter()
            .map(|&id| gradient.take(id).unwrap_or_else(|| zeros(id)))
            .collect(),
        _ => latents.iter().map(|&id| zeros(id)).collect(),
    };
    Ok((log_prob, gradient))
}

/// A chosen point of a trajectory.
#[derive(Debug, Clone)]
struct Candidate {
    position: Rc<Vec<DoubleTensor>>,
    gradient: Rc<Vec<DoubleTensor>>,
    log_prob: f64,
}

/// A point of a trajectory. Immutable once made: trees share them (and their position and gradient with
/// candidates) through reference counting rather than copying.
#[derive(Debug)]
struct LeapfrogState {
    position: Rc<Vec<DoubleTensor>>,
    momentum: Vec<DoubleTensor>,
    velocity: Vec<DoubleTensor>,
    gradient: Rc<Vec<DoubleTensor>>,
    log_prob: f64,
    energy: f64,
}

impl LeapfrogState {
    fn new(
        position: Rc<Vec<DoubleTensor>>,
        momentum: Vec<DoubleTensor>,
        gradient: Rc<Vec<DoubleTensor>>,
        log_prob: f64,
        potential: &PotentialState,
    ) -> Result<Self> {
        let velocity = potential.velocity(&momentum)?;
        let kinetic = 0.5 * dot(&momentum, &velocity)?;
        Ok(Self {
            position,
            momentum,
            velocity,
            gradient,
            log_prob,
            energy: kinetic - log_prob,
        })
    }
}

fn dot(a: &[DoubleTensor], b: &[DoubleTensor]) -> Result<f64> {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b) {
        sum += x.mul(y)?.sum();
    }
    Ok(sum)
}

/// x + dt · v for each latent.
fn axpy(x: &[DoubleTensor], dt: f64, v: &[DoubleTensor]) -> Result<Vec<DoubleTensor>> {
    x.iter().zip(v).map(|(x, v)| Ok((v * dt).add(x)?)).collect()
}

/// Running mean and variance (Welford), seeded with a prior mean and variance worth `weight` samples (Java's
/// `VarianceCalculator`).
#[derive(Debug, Clone)]
struct VarianceCalculator {
    count: f64,
    mean: Vec<DoubleTensor>,
    m2: Vec<DoubleTensor>,
}

impl VarianceCalculator {
    fn new(mean: Vec<DoubleTensor>, variance: Vec<DoubleTensor>, weight: f64) -> Self {
        let m2 = variance.iter().map(|v| v * weight).collect();
        Self {
            count: weight,
            mean,
            m2,
        }
    }

    fn add(&mut self, sample: &[DoubleTensor]) -> Result<()> {
        self.count += 1.0;
        for ((x, mean), m2) in sample.iter().zip(&mut self.mean).zip(&mut self.m2) {
            let delta = x.sub(mean)?;
            let new_mean = mean.add(&(&delta / self.count))?;
            let delta2 = x.sub(&new_mean)?;
            *m2 = m2.add(&delta.mul(&delta2)?)?;
            *mean = new_mean;
        }
        Ok(())
    }

    fn variance(&self) -> Vec<DoubleTensor> {
        self.m2.iter().map(|m2| m2 / self.count).collect()
    }
}

/// One run's adaptive quadratic potential.
#[derive(Debug, Clone)]
struct PotentialState {
    window: usize,
    variance: Vec<DoubleTensor>,
    standard_deviation: Vec<DoubleTensor>,
    forward: VarianceCalculator,
    background: VarianceCalculator,
    samples: usize,
}

impl PotentialState {
    fn new(settings: &QuadraticPotential, like: &[DoubleTensor]) -> Self {
        let full = |v: f64| {
            like.iter()
                .map(|t| DoubleTensor::full(t.shape(), v))
                .collect::<Vec<_>>()
        };
        let variance = full(settings.initial_variance);
        Self {
            window: settings.adaptation_window,
            standard_deviation: variance.iter().map(DoubleTensor::sqrt).collect(),
            forward: VarianceCalculator::new(
                full(settings.initial_mean),
                variance.clone(),
                settings.initial_weight,
            ),
            background: VarianceCalculator::new(full(0.0), full(0.0), 0.0),
            variance,
            samples: 0,
        }
    }

    fn update(&mut self, position: &[DoubleTensor]) -> Result<()> {
        if self.samples > 0 && self.samples % self.window == 0 {
            let zeros: Vec<DoubleTensor> = self
                .variance
                .iter()
                .map(|v| DoubleTensor::zeros(v.shape()))
                .collect();
            self.forward = std::mem::replace(
                &mut self.background,
                VarianceCalculator::new(zeros.clone(), zeros, 0.0),
            );
        }
        self.forward.add(position)?;
        self.background.add(position)?;
        self.variance = self.forward.variance();
        self.standard_deviation = self.variance.iter().map(DoubleTensor::sqrt).collect();
        self.samples += 1;
        Ok(())
    }

    fn random_momentum(&self, rng: &mut KeanuRng) -> Result<Vec<DoubleTensor>> {
        self.standard_deviation
            .iter()
            .map(|sd| Ok(rng.gaussian(sd.shape()).div(sd)?))
            .collect()
    }

    fn velocity(&self, momentum: &[DoubleTensor]) -> Result<Vec<DoubleTensor>> {
        self.variance
            .iter()
            .zip(momentum)
            .map(|(v, p)| Ok(v.mul(p)?))
            .collect()
    }
}

/// Dual averaging step size adaptation (Hoffman & Gelman 2014; Java's `AdaptiveStepSize`).
#[derive(Debug, Clone)]
struct AdaptiveStepSize {
    target: f64,
    adapt_count: usize,
    mu: f64,
    step_size: f64,
    h_bar: f64,
    accept_rate: f64,
    log_step_size: f64,
    log_step_size_bar: f64,
    step: usize,
}

impl AdaptiveStepSize {
    const T0: f64 = 10.0;
    const GAMMA: f64 = 0.05;
    const KAPPA: f64 = 0.75;

    fn new(step_size: f64, target: f64, adapt_count: usize) -> Self {
        Self {
            target,
            adapt_count,
            mu: (10.0 * step_size).ln(),
            step_size,
            h_bar: 0.0,
            accept_rate: 0.0,
            log_step_size: step_size.ln(),
            log_step_size_bar: step_size.ln(),
            step: 1,
        }
    }

    /// Adapts after a step. On the final adaptation step it switches to the averaged step size, so the first
    /// sample after adaptation already uses it.
    fn adapt(&mut self, tree: &Tree) {
        if self.step < self.adapt_count {
            self.step_size = self.update(tree).exp();
        } else if self.step == self.adapt_count {
            self.update(tree);
            self.step_size = self.log_step_size_bar.exp();
        }
        self.step += 1;
    }

    fn update(&mut self, tree: &Tree) -> f64 {
        let step = self.step as f64;
        let w = 1.0 / (step + Self::T0);
        self.accept_rate = tree.sum_metropolis_acceptance / tree.size as f64;
        self.h_bar = (1.0 - w) * self.h_bar + w * (self.target - self.accept_rate);
        self.log_step_size = self.mu - (step.sqrt() / Self::GAMMA) * self.h_bar;
        let tend_to_zero = step.powf(-Self::KAPPA);
        self.log_step_size_bar =
            tend_to_zero * self.log_step_size + (1.0 - tend_to_zero) * self.log_step_size_bar;
        self.log_step_size
    }
}

/// What a step of NUTS needs to build trajectories.
struct Integrator<'a, 'm> {
    model: &'m Model,
    latents: &'a [NodeId],
    potential: &'a PotentialState,
    start_energy: f64,
    max_energy_change: f64,
    /// The position the last leapfrog step left the state at, if any.
    in_state: RefCell<Option<Rc<Vec<DoubleTensor>>>>,
}

impl Integrator<'_, '_> {
    /// One leapfrog step of size `dt` (negative to go backwards). Leaves the state at the new position.
    fn leapfrog(&self, state: &mut State, from: &LeapfrogState, dt: f64) -> Result<LeapfrogState> {
        let half = dt / 2.0;
        let momentum = axpy(&from.momentum, half, &from.gradient)?;
        let velocity = self.potential.velocity(&momentum)?;
        let position = Rc::new(axpy(&from.position, dt, &velocity)?);

        // the state needs its own copy of the position: the trajectory keeps this one
        self.model
            .set_and_cascade(state, f64_values(self.latents, position.iter().cloned()))?;
        *self.in_state.borrow_mut() = Some(Rc::clone(&position));
        let (log_prob, gradient) = log_prob_and_gradient_at(self.model, state, self.latents)?;

        let momentum = axpy(&momentum, half, &gradient)?;
        LeapfrogState::new(
            position,
            momentum,
            Rc::new(gradient),
            log_prob,
            self.potential,
        )
    }
}

/// A (sub)trajectory and the candidate sampled from it.
#[derive(Debug, Clone)]
struct Tree {
    forward: Rc<LeapfrogState>,
    backward: Rc<LeapfrogState>,
    sum_momentum: Vec<DoubleTensor>,
    proposal: Option<Candidate>,
    log_sum_weight: f64,
    diverged: bool,
    u_turned: bool,
    sum_metropolis_acceptance: f64,
    size: usize,
    height: usize,
}

impl Tree {
    fn should_continue(&self) -> bool {
        !self.diverged && !self.u_turned
    }
}

fn log_sum_exp(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

fn accept_with_log_probability(log_probability: f64, rng: &mut KeanuRng) -> bool {
    rng.next_f64().ln() < log_probability
}

fn is_u_turning(
    forward: &[DoubleTensor],
    backward: &[DoubleTensor],
    rho: &[DoubleTensor],
) -> Result<bool> {
    Ok(dot(forward, rho)? < 0.0 || dot(backward, rho)? < 0.0)
}

fn add_all(a: &[DoubleTensor], b: &[DoubleTensor]) -> Result<Vec<DoubleTensor>> {
    a.iter().zip(b).map(|(x, y)| Ok(x.add(y)?)).collect()
}

impl Integrator<'_, '_> {
    fn base_case(
        &self,
        state: &mut State,
        from: &LeapfrogState,
        direction: f64,
        step_size: f64,
    ) -> Result<Tree> {
        let after = Rc::new(self.leapfrog(state, from, step_size * direction)?);
        let energy_change = after.energy - self.start_energy;
        if energy_change.abs() >= self.max_energy_change || energy_change.is_nan() {
            return Ok(Tree {
                sum_momentum: after.momentum.clone(),
                forward: Rc::clone(&after),
                backward: after,
                proposal: None,
                log_sum_weight: f64::NEG_INFINITY,
                diverged: true,
                u_turned: false,
                sum_metropolis_acceptance: 0.0,
                size: 1,
                height: 0,
            });
        }
        let log_sum_weight = -energy_change;
        Ok(Tree {
            sum_momentum: after.momentum.clone(),
            proposal: Some(Candidate {
                position: Rc::clone(&after.position),
                gradient: Rc::clone(&after.gradient),
                log_prob: after.log_prob,
            }),
            forward: Rc::clone(&after),
            backward: after,
            log_sum_weight,
            diverged: false,
            u_turned: false,
            sum_metropolis_acceptance: log_sum_weight.exp().min(1.0),
            size: 1,
            height: 0,
        })
    }

    /// Builds a subtree of 2^height leapfrog steps from `from` in `direction` (Java's `Tree.buildTree`).
    fn build(
        &self,
        state: &mut State,
        from: &LeapfrogState,
        direction: f64,
        height: usize,
        step_size: f64,
        rng: &mut KeanuRng,
    ) -> Result<Tree> {
        if height == 0 {
            return self.base_case(state, from, direction, step_size);
        }
        let mut tree = self.build(state, from, direction, height - 1, step_size, rng)?;
        if tree.should_continue() {
            let edge = Rc::clone(if direction < 0.0 {
                &tree.backward
            } else {
                &tree.forward
            });
            let extension = self.build(state, &edge, direction, height - 1, step_size, rng)?;
            if direction < 0.0 {
                tree.backward = Rc::clone(&extension.backward);
            } else {
                tree.forward = Rc::clone(&extension.forward);
            }
            tree.diverged = extension.diverged;
            tree.u_turned = extension.u_turned;
            if extension.should_continue() {
                tree.sum_momentum = add_all(&tree.sum_momentum, &extension.sum_momentum)?;
                tree.u_turned = is_u_turning(
                    &tree.forward.velocity,
                    &tree.backward.velocity,
                    &tree.sum_momentum,
                )?;
                let total = log_sum_exp(tree.log_sum_weight, extension.log_sum_weight);
                tree.log_sum_weight = total;
                if accept_with_log_probability(extension.log_sum_weight - total, rng) {
                    tree.proposal = extension.proposal;
                }
            }
            tree.sum_metropolis_acceptance += extension.sum_metropolis_acceptance;
            tree.size += extension.size;
        }
        Ok(tree)
    }

    /// Doubles the trajectory in a random direction (Java's `Tree.grow`).
    fn grow(
        &self,
        state: &mut State,
        tree: &mut Tree,
        step_size: f64,
        rng: &mut KeanuRng,
    ) -> Result<()> {
        let direction = if rng.next_bool() { 1.0 } else { -1.0 };
        let edge = Rc::clone(if direction < 0.0 {
            &tree.backward
        } else {
            &tree.forward
        });
        let other = self.build(state, &edge, direction, tree.height, step_size, rng)?;
        if direction < 0.0 {
            tree.backward = Rc::clone(&other.backward);
        } else {
            tree.forward = Rc::clone(&other.forward);
        }
        tree.sum_metropolis_acceptance += other.sum_metropolis_acceptance;
        tree.size += other.size;
        if other.should_continue() {
            if accept_with_log_probability(other.log_sum_weight - tree.log_sum_weight, rng) {
                tree.proposal = other.proposal;
            }
            tree.log_sum_weight = log_sum_exp(tree.log_sum_weight, other.log_sum_weight);
            tree.sum_momentum = add_all(&tree.sum_momentum, &other.sum_momentum)?;
        }
        tree.diverged = other.diverged;
        if !tree.diverged {
            tree.u_turned = other.u_turned
                || is_u_turning(
                    &tree.forward.velocity,
                    &tree.backward.velocity,
                    &tree.sum_momentum,
                )?;
        }
        tree.height += 1;
        Ok(())
    }
}

/// A running NUTS chain (Java's `NUTSSampler`). It owns its adapted step size and potential.
#[derive(Debug)]
pub struct NutsSampler<'m> {
    model: &'m Model,
    config: Nuts,
    potential: PotentialState,
    step_size: AdaptiveStepSize,
    latents: Vec<NodeId>,
    nodes: Vec<NodeId>,
    proposal: Candidate,
    step_count: usize,
    statistics: NutsStatistics,
}

impl NutsSampler<'_> {
    /// What has been recorded so far.
    pub fn statistics(&self) -> &NutsStatistics {
        &self.statistics
    }

    /// The step size the next step will use.
    pub fn step_size(&self) -> f64 {
        self.step_size.step_size
    }
}

impl SamplingAlgorithm for NutsSampler<'_> {
    fn step(&mut self, state: &mut State, rng: &mut KeanuRng) -> Result<()> {
        let momentum = self.potential.random_momentum(rng)?;
        let start = Rc::new(LeapfrogState::new(
            Rc::clone(&self.proposal.position),
            momentum,
            Rc::clone(&self.proposal.gradient),
            self.proposal.log_prob,
            &self.potential,
        )?);
        let integrator = Integrator {
            model: self.model,
            latents: &self.latents,
            potential: &self.potential,
            start_energy: start.energy,
            max_energy_change: self.config.max_energy_change,
            in_state: RefCell::new(None),
        };
        let mut tree = Tree {
            sum_momentum: start.momentum.clone(),
            forward: Rc::clone(&start),
            backward: start,
            proposal: Some(self.proposal.clone()),
            log_sum_weight: 0.0,
            diverged: false,
            u_turned: false,
            sum_metropolis_acceptance: 0.0,
            size: 0,
            height: 0,
        };
        let step_size = self.step_size.step_size;
        while tree.should_continue() && tree.height < self.config.max_tree_height {
            integrator.grow(state, &mut tree, step_size, rng)?;
        }
        if let Some(chosen) = tree.proposal.take() {
            self.proposal = chosen;
        }

        // leave the state at the chosen sample. The trajectory left it at its last leapfrog position; when that
        // is the chosen one (the same allocation, so the same values, and setting them again would cascade to the
        // same results) there is nothing to do.
        let in_state = integrator.in_state.into_inner();
        if !in_state.is_some_and(|p| Rc::ptr_eq(&p, &self.proposal.position)) {
            let values = f64_values(&self.latents, self.proposal.position.iter().cloned());
            self.model.set_and_cascade(state, values)?;
        }

        if self.config.save_statistics {
            self.statistics.step_size.push(self.step_size.step_size);
            self.statistics
                .mean_tree_accept
                .push(self.step_size.accept_rate);
            self.statistics.log_prob.push(self.proposal.log_prob);
            self.statistics.tree_size.push(tree.size);
        }
        if self.config.adapt_step_size {
            self.step_size.adapt(&tree);
        }
        if self.step_count < self.config.adapt_count && self.config.adapt_potential {
            self.potential.update(&self.proposal.position)?;
        }
        if self.step_count >= self.config.adapt_count && tree.diverged {
            self.statistics.divergent_steps.push(self.step_count);
        }
        self.step_count += 1;
        Ok(())
    }

    fn record(&self, state: &State, samples: &mut NetworkSamples) -> Result<()> {
        samples.record(state, self.proposal.log_prob)
    }

    fn graph(&self) -> &Graph {
        self.model.graph()
    }

    fn sampled_nodes(&self) -> &[NodeId] {
        &self.nodes
    }
}
