//! The No-U-Turn sampler (NUTS): Hamiltonian Monte Carlo that chooses its own trajectory length, with dual
//! averaging step size adaptation and an adaptive diagonal mass matrix.
//!
//! It follows Stan's NUTS: multinomial sampling from the trajectory, the generalised no-U-turn criterion with the
//! extra checks across subtree boundaries Stan added in 2.21, and Stan's windowed adaptation of the step size and
//! mass matrix (see [`QuadraticPotential`]).
//!
//! By default NUTS moves in unconstrained space ([`Space::Unconstrained`]): each latent is mapped to ℝⁿ through the
//! transform of its support (see [`Parameterisation`]), and the positions, the mass matrix and the leapfrog steps
//! are all in those coordinates, with the log-Jacobian of the transforms added to the log probability so that the
//! draws have the right distribution. Every point is inside the support, so trajectories never diverge by leaving
//! it. Draws are recorded, and the state left, at the constrained values, and the recorded log probabilities are the
//! model's (without the Jacobian). [`Nuts::with_space`]`(Space::Constrained)` moves in the latents' own values
//! instead, as NUTS did before transforms existed.

use crate::error::{AlgorithmError, Result};
use crate::mcmc::{SamplingAlgorithm, SamplingOptions, generate, is_impossible};
use crate::samples::NetworkSamples;
use crate::support::{
    block_shapes, check_latents, check_nodes, concat, evaluate_all, split, start_point,
};
use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, Model, NodeId, Parameterisation, Space, State};
use keanu_tensor::DoubleTensor;
use keanu_tensor::special::log_sum_exp;
use std::cell::RefCell;
use std::rc::Rc;

/// The adaptive diagonal quadratic potential (mass matrix) settings, and the schedule NUTS adapts on.
///
/// NUTS adapts as Stan does (its windowed adaptation, with Stan's defaults), over the first
/// [`Nuts::adapt_count`] steps:
///
/// 1. an initial fast interval of [`initial_buffer`](Self::initial_buffer) steps (75) in which only the step size
///    is adapted, while the chain finds the typical set;
/// 2. slow windows in which the draws are also collected to estimate each latent's (unconstrained) variance. The
///    first is [`adaptation_window`](Self::adaptation_window) steps (25) long and each one after it twice as long as
///    the one before, with the last stretched to fill the space left. At the end of each window the potential's
///    variance (the inverse mass matrix) becomes the window's sample variance, regularised towards a small value as
///    (n / (n + 5)) · var + 10⁻³ · 5 / (n + 5) for n draws; then a new reasonable step size is found for it and step
///    size adaptation restarts;
/// 3. a terminal fast interval of [`terminal_buffer`](Self::terminal_buffer) steps (50) in which only the step
///    size is adapted, to the final potential.
///
/// With fewer than 20 adaptation steps only the step size is adapted. When the three stages don't fit in
/// `adapt_count` steps, they take 15%, 75% and 10% of them instead (one slow window).
///
/// Until the first slow window ends every latent element has variance
/// [`initial_variance`](Self::initial_variance).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct QuadraticPotential {
    /// Ignored. The potential is estimated from each slow window alone (Keanu's earlier adaptation tracked a running
    /// mean that started here).
    #[deprecated(
        note = "ignored: the potential is estimated from each slow adaptation window alone"
    )]
    pub initial_mean: f64,
    /// Each latent element's variance (the inverse mass matrix) until the first slow window ends. Default 1.
    pub initial_variance: f64,
    /// Ignored. The variance estimates are regularised as Stan regularises them (Keanu's earlier adaptation counted
    /// the initial mean and variance as this many draws).
    #[deprecated(note = "ignored: the variance estimates are regularised as Stan's are")]
    pub initial_weight: f64,
    /// The length of the first slow window, in steps; each later window is twice as long as the one before
    /// (Stan's `window`). Default 25.
    pub adaptation_window: usize,
    /// The steps at the start of adaptation in which only the step size is adapted (Stan's `init_buffer`).
    /// Default 75.
    pub initial_buffer: usize,
    /// The steps at the end of adaptation in which only the step size is adapted (Stan's `term_buffer`).
    /// Default 50.
    pub terminal_buffer: usize,
}

/// Setters for each setting, for use as `QuadraticPotential::default().with_...(...)`.
impl QuadraticPotential {
    /// Sets [`initial_mean`](Self::initial_mean), which is ignored.
    #[must_use]
    #[deprecated(
        note = "ignored: the potential is estimated from each slow adaptation window alone"
    )]
    #[allow(deprecated)]
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

    /// Sets [`initial_weight`](Self::initial_weight), which is ignored.
    #[must_use]
    #[deprecated(note = "ignored: the variance estimates are regularised as Stan's are")]
    #[allow(deprecated)]
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

    /// Sets [`initial_buffer`](Self::initial_buffer).
    #[must_use]
    pub fn with_initial_buffer(mut self, initial_buffer: usize) -> Self {
        self.initial_buffer = initial_buffer;
        self
    }

    /// Sets [`terminal_buffer`](Self::terminal_buffer).
    #[must_use]
    pub fn with_terminal_buffer(mut self, terminal_buffer: usize) -> Self {
        self.terminal_buffer = terminal_buffer;
        self
    }
}

impl Default for QuadraticPotential {
    #[allow(deprecated)]
    fn default() -> Self {
        Self {
            initial_mean: 0.0,
            initial_variance: 1.0,
            initial_weight: 10.0,
            adaptation_window: 25,
            initial_buffer: 75,
            terminal_buffer: 50,
        }
    }
}

/// NUTS configuration. The defaults are Stan's where Stan has the setting (target acceptance probability 0.8, tree
/// height 10, divergence at an energy increase of 1000, and its adaptation schedule; see [`QuadraticPotential`]),
/// with 1000 adaptation steps. The configuration holds no state, so it can be reused for any number of runs and
/// shared by parallel chains; each run adapts its own step size and potential.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Nuts {
    /// The steps during which the step size and potential are adapted (Stan's `num_warmup`), on the schedule
    /// [`QuadraticPotential`] describes.
    pub adapt_count: usize,
    /// Whether to adapt the step size during the first `adapt_count` steps, by dual averaging (Hoffman & Gelman
    /// 2014, with Stan's constants γ = 0.05, t₀ = 10, κ = 0.75). Adaptation starts from a reasonable step size found
    /// from [`initial_step_size`](Self::initial_step_size) by doubling or halving it until one leapfrog step's
    /// acceptance probability crosses 0.8, as Stan does, and does so again after each update of the potential.
    /// The step size after adaptation is the dual averaging's average since the last restart.
    pub adapt_step_size: bool,
    /// The initial step size; by default 0.25 / n^(1/4) for n latent elements. With step size adaptation it is
    /// where the search for a reasonable step size starts.
    pub initial_step_size: Option<f64>,
    /// The kinetic energy (mass matrix) settings and the adaptation schedule.
    pub potential: QuadraticPotential,
    /// Whether to adapt the potential during the first `adapt_count` steps.
    pub adapt_potential: bool,
    /// The mean acceptance probability step size adaptation aims for (Stan's `delta`). Default 0.8.
    pub target_acceptance_prob: f64,
    /// A trajectory whose energy rises by more than this from where it started (or becomes NaN) is divergent, as
    /// in Stan (`max_deltaH`). A fall in energy, however large, is not a divergence.
    pub max_energy_change: f64,
    /// The maximum tree height (at most 2^height leapfrog steps per sample).
    pub max_tree_height: usize,
    /// Record step sizes, log probs, mean tree acceptance and tree sizes for every step.
    pub save_statistics: bool,
    /// The space the sampler moves in (default [`Space::Unconstrained`]; see the [module documentation](self)).
    pub space: Space,
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

    /// Sets [`space`](Self::space).
    #[must_use]
    pub fn with_space(mut self, space: Space) -> Self {
        self.space = space;
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
            space: Space::Unconstrained,
        }
    }
}

/// What NUTS recorded while sampling (Java's NUTS `Statistics`).
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct NutsStatistics {
    /// The step size used at each step (when saving statistics).
    pub step_size: Vec<f64>,
    /// The mean Metropolis acceptance probability last used for step size adaptation, recorded at each step. It is
    /// recorded before the step's own tree is used for adaptation, so `mean_tree_accept[i]` is the acceptance of
    /// step i - 1's tree (0 at step 0). Once adaptation ends (after `adapt_count` steps) it stays at the last
    /// adapted tree's value, and without step size adaptation it is always 0.
    pub mean_tree_accept: Vec<f64>,
    /// The log probability of each step's sample (the model's, without the Jacobian of any transform).
    pub log_prob: Vec<f64>,
    /// The number of leapfrog steps in each step's tree.
    pub tree_size: Vec<usize>,
    /// The (0-based) steps after adaptation whose trajectory diverged. Always recorded.
    ///
    /// A trajectory diverges when its Hamiltonian rises more than [`Nuts::max_energy_change`] above where it
    /// started (or becomes NaN): the leapfrog integrator has gone unstable, typically in a region of high curvature such as
    /// the neck of a funnel. Divergences mean the chain may not reach that region, so its draws can be biased even
    /// when other diagnostics look healthy; reparameterising the model (for example in non-centred form) or raising
    /// [`Nuts::target_acceptance_prob`] usually removes them. [`divergence_count`](Self::divergence_count) and
    /// [`divergent_flags`](Self::divergent_flags) are views of this list.
    pub divergent_steps: Vec<usize>,
    /// The Hamiltonian (potential plus kinetic energy) at each step's sample, with the momentum it was reached with.
    /// Always recorded, one per step, adaptation steps included.
    ///
    /// The potential is the negative log density being sampled, which in unconstrained space includes the Jacobian
    /// of the transforms, so this is not `-log_prob`. Pass the energies after adaptation to
    /// [`diagnostics::e_bfmi`](crate::diagnostics::e_bfmi) to check how well the chain explores the energy levels.
    pub energy: Vec<f64>,
}

impl NutsStatistics {
    /// The number of divergent steps after adaptation: the length of
    /// [`divergent_steps`](Self::divergent_steps).
    pub fn divergence_count(&self) -> usize {
        self.divergent_steps.len()
    }

    /// One flag per step taken (the length of [`energy`](Self::energy)), true for the steps in
    /// [`divergent_steps`](Self::divergent_steps). Adaptation steps are never flagged.
    ///
    /// When sampling with the default [`SamplingOptions`] each step is one draw, so flag `i` belongs to draw `i`;
    /// with a drop of `d` and down-sampling interval `k`, draw `i` is step `d + i·k`.
    pub fn divergent_flags(&self) -> Vec<bool> {
        let mut flags = vec![false; self.energy.len()];
        for &step in &self.divergent_steps {
            if let Some(flag) = flags.get_mut(step) {
                *flag = true;
            }
        }
        flags
    }
}

impl Nuts {
    /// The default settings.
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
        // a sample variance needs at least two draws
        if self.potential.adaptation_window <= 1 {
            return invalid("the potential's adaptation window must be greater than 1");
        }
        // a zero variance gives a zero mass matrix entry, so momentum / mass is NaN and the chain freezes
        let positive = |x: f64| x.is_finite() && x > 0.0;
        if !positive(self.potential.initial_variance) {
            return invalid("the potential's initial variance must be finite and greater than 0");
        }
        Ok(())
    }

    /// A chain starting from the current state, recording the values of `nodes`. Every latent must be continuous.
    ///
    /// With step size adaptation (and `adapt_count` above 0) this already takes a few single leapfrog steps from the
    /// starting point, to find a reasonable initial step size.
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::ImpossibleStart`] if the starting point has zero probability or, in unconstrained space,
    /// a latent's value is outside its support or on its boundary; and, in unconstrained space, an error from
    /// [`Parameterisation::new`] if a latent's support has no transform (for instance a bound that depends on another
    /// latent, which [`Space::Constrained`] can sample).
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
        check_latents(model)?;
        evaluate_all(model.graph(), state, rng)?;

        let param = Parameterisation::with_space(model, state, self.space)?;
        let shapes = block_shapes(&param);
        let z = start_point(&param, state)?;
        let (log_prob, gradient) = log_prob_and_gradient_at(&param, &shapes, state, &z)?;
        if is_impossible(log_prob) {
            return Err(AlgorithmError::ImpossibleStart);
        }
        let sample_log_prob = log_prob - param.log_det_jacobian(&z)?;
        let position = split(&z, &shapes);

        let step_size = self.initial_step_size.unwrap_or_else(|| {
            let n: usize = position.iter().map(DoubleTensor::len).sum();
            0.25 / (n as f64).powf(0.25)
        });

        let mut sampler = NutsSampler {
            param,
            shapes,
            config: *self,
            potential: PotentialState::new(&self.potential, self.adapt_count, &position),
            step_size: AdaptiveStepSize::new(step_size, self.target_acceptance_prob),
            nodes: nodes.to_vec(),
            proposal: Candidate {
                position: Rc::new(position),
                gradient: Rc::new(gradient),
                log_prob,
                energy: f64::NAN,
            },
            sample_log_prob,
            step_count: 0,
            statistics: NutsStatistics::default(),
        };
        if self.adapt_step_size && self.adapt_count > 0 {
            sampler.find_reasonable_step_size(state, rng)?;
        }
        Ok(sampler)
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

/// Sets the state to the point `z` and returns the log probability of `z` (with the Jacobian of the transforms) and
/// its gradient, one tensor per block of `shapes`, in one pass. Where the log probability is not finite the gradient
/// is meaningless (the trajectory diverges anyway), so zeros are used.
fn log_prob_and_gradient_at(
    param: &Parameterisation<'_>,
    shapes: &[Vec<usize>],
    state: &mut State,
    z: &[f64],
) -> Result<(f64, Vec<DoubleTensor>)> {
    let (log_prob, gradient) = param.log_prob_and_gradient(z, true, state)?;
    let gradient = match gradient {
        Some(gradient) if log_prob.is_finite() => split(&gradient, shapes),
        _ => shapes.iter().map(|s| DoubleTensor::zeros(s)).collect(),
    };
    Ok((log_prob, gradient))
}

/// A chosen point of a trajectory.
#[derive(Debug, Clone)]
struct Candidate {
    position: Rc<Vec<DoubleTensor>>,
    gradient: Rc<Vec<DoubleTensor>>,
    log_prob: f64,
    /// The Hamiltonian at this point with the momentum it was reached with (NaN until a step has given it one).
    energy: f64,
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

/// Running mean and sample variance (Welford), restarted for each slow adaptation window.
#[derive(Debug, Clone)]
struct VarianceEstimator {
    count: usize,
    mean: Vec<DoubleTensor>,
    m2: Vec<DoubleTensor>,
}

impl VarianceEstimator {
    fn new(like: &[DoubleTensor]) -> Self {
        let zeros: Vec<DoubleTensor> = like
            .iter()
            .map(|t| DoubleTensor::zeros(t.shape()))
            .collect();
        Self {
            count: 0,
            mean: zeros.clone(),
            m2: zeros,
        }
    }

    fn add(&mut self, sample: &[DoubleTensor]) -> Result<()> {
        self.count += 1;
        let count = self.count as f64;
        for ((x, mean), m2) in sample.iter().zip(&mut self.mean).zip(&mut self.m2) {
            let delta = x.sub(mean)?;
            let new_mean = mean.add(&(&delta / count))?;
            let delta2 = x.sub(&new_mean)?;
            *m2 = m2.add(&delta.mul(&delta2)?)?;
            *mean = new_mean;
        }
        Ok(())
    }

    /// The sample variance (with n − 1 in the denominator) regularised as Stan regularises it:
    /// (n / (n + 5)) · var + 10⁻³ · 5 / (n + 5). `None` with fewer than two draws.
    fn regularised_variance(&self) -> Option<Vec<DoubleTensor>> {
        if self.count < 2 {
            return None;
        }
        let n = self.count as f64;
        let shrink = n / (n + 5.0);
        let floor = 1e-3 * 5.0 / (n + 5.0);
        Some(
            self.m2
                .iter()
                .map(|m2| (m2 * (shrink / (n - 1.0))).map(|v| v + floor))
                .collect(),
        )
    }
}

/// Stan's adaptation windows (its `windowed_adaptation`): where the slow windows fall in the adaptation steps.
#[derive(Debug, Clone)]
struct AdaptationWindows {
    /// The adaptation steps, or 0 when the potential isn't adapted (fewer than 20 steps).
    num_warmup: usize,
    init_buffer: usize,
    term_buffer: usize,
    /// The adaptation step about to be used (0-based).
    counter: usize,
    window_size: usize,
    /// The last step of the current slow window.
    next_window: usize,
}

impl AdaptationWindows {
    fn new(settings: &QuadraticPotential, adapt_count: usize) -> Self {
        let (num_warmup, init_buffer, term_buffer, base_window) = if adapt_count < 20 {
            (0, 0, 0, 0)
        } else if settings.initial_buffer + settings.adaptation_window + settings.terminal_buffer
            > adapt_count
        {
            // the stages as fractions of the steps there are: 15%, 75% and 10%
            let init_buffer = (0.15 * adapt_count as f64) as usize;
            let term_buffer = (0.1 * adapt_count as f64) as usize;
            (
                adapt_count,
                init_buffer,
                term_buffer,
                adapt_count - init_buffer - term_buffer,
            )
        } else {
            (
                adapt_count,
                settings.initial_buffer,
                settings.terminal_buffer,
                settings.adaptation_window,
            )
        };
        Self {
            num_warmup,
            init_buffer,
            term_buffer,
            counter: 0,
            window_size: base_window,
            next_window: (init_buffer + base_window).saturating_sub(1),
        }
    }

    /// Whether the current step's draw goes into the variance estimate.
    fn in_window(&self) -> bool {
        self.counter >= self.init_buffer
            && self.counter < self.num_warmup - self.term_buffer
            && self.counter != self.num_warmup
    }

    /// Whether the current step ends a slow window.
    fn ends_window(&self) -> bool {
        self.counter == self.next_window && self.counter != self.num_warmup
    }

    /// Moves to the next slow window, twice as long, or stretched to the terminal buffer if the one after it wouldn't
    /// fit.
    fn compute_next_window(&mut self) {
        let last = self.num_warmup - self.term_buffer - 1;
        if self.next_window == last {
            return;
        }
        self.window_size *= 2;
        self.next_window = self.counter + self.window_size;
        if self.next_window != last && self.next_window + 2 * self.window_size > last {
            self.next_window = last;
        }
    }
}

/// One run's adaptive diagonal quadratic potential.
#[derive(Debug, Clone)]
struct PotentialState {
    /// The inverse mass matrix.
    variance: Vec<DoubleTensor>,
    standard_deviation: Vec<DoubleTensor>,
    estimator: VarianceEstimator,
    windows: AdaptationWindows,
}

impl PotentialState {
    fn new(settings: &QuadraticPotential, adapt_count: usize, like: &[DoubleTensor]) -> Self {
        let variance: Vec<DoubleTensor> = like
            .iter()
            .map(|t| DoubleTensor::full(t.shape(), settings.initial_variance))
            .collect();
        Self {
            standard_deviation: variance.iter().map(DoubleTensor::sqrt).collect(),
            variance,
            estimator: VarianceEstimator::new(like),
            windows: AdaptationWindows::new(settings, adapt_count),
        }
    }

    /// Learns from an adaptation step's draw (Stan's `var_adaptation::learn_variance`). Returns whether the step
    /// ended a slow window, so that the variance changed.
    fn update(&mut self, position: &[DoubleTensor]) -> Result<bool> {
        if self.windows.num_warmup == 0 {
            return Ok(false);
        }
        if self.windows.in_window() {
            self.estimator.add(position)?;
        }
        let ends_window = self.windows.ends_window();
        if ends_window {
            self.windows.compute_next_window();
            if let Some(variance) = self.estimator.regularised_variance() {
                self.standard_deviation = variance.iter().map(DoubleTensor::sqrt).collect();
                self.variance = variance;
            }
            self.estimator = VarianceEstimator::new(position);
        }
        self.windows.counter += 1;
        Ok(ends_window)
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

/// Dual averaging step size adaptation (Hoffman & Gelman 2014, as Stan's `stepsize_adaptation`).
#[derive(Debug, Clone)]
struct AdaptiveStepSize {
    target: f64,
    mu: f64,
    step_size: f64,
    h_bar: f64,
    /// The acceptance statistic of the last tree adapted to.
    accept_rate: f64,
    log_step_size_bar: f64,
    /// Trees adapted to since the last restart.
    counter: usize,
}

impl AdaptiveStepSize {
    const T0: f64 = 10.0;
    const GAMMA: f64 = 0.05;
    const KAPPA: f64 = 0.75;

    fn new(step_size: f64, target: f64) -> Self {
        let mut adaptive = Self {
            target,
            mu: 0.0,
            step_size,
            h_bar: 0.0,
            accept_rate: 0.0,
            log_step_size_bar: 0.0,
            counter: 0,
        };
        adaptive.restart();
        adaptive
    }

    /// Starts dual averaging again from the current step size, aiming its shrinkage at 10 times it.
    fn restart(&mut self) {
        self.mu = (10.0 * self.step_size).ln();
        self.h_bar = 0.0;
        self.log_step_size_bar = 0.0;
        self.counter = 0;
    }

    /// Adapts to a step's tree.
    fn learn(&mut self, tree: &Tree) {
        self.counter += 1;
        let count = self.counter as f64;
        self.accept_rate = (tree.sum_metropolis_acceptance / tree.size as f64).min(1.0);
        let w = 1.0 / (count + Self::T0);
        self.h_bar = (1.0 - w) * self.h_bar + w * (self.target - self.accept_rate);
        let log_step_size = self.mu - (count.sqrt() / Self::GAMMA) * self.h_bar;
        let tend_to_zero = count.powf(-Self::KAPPA);
        self.log_step_size_bar =
            tend_to_zero * log_step_size + (1.0 - tend_to_zero) * self.log_step_size_bar;
        self.step_size = log_step_size.exp();
    }

    /// Ends adaptation with the averaged step size.
    fn complete(&mut self) {
        if self.counter > 0 {
            self.step_size = self.log_step_size_bar.exp();
        }
    }
}
/// What a step of NUTS needs to build trajectories.
struct Integrator<'a, 'm> {
    param: &'a Parameterisation<'m>,
    shapes: &'a [Vec<usize>],
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

        let (log_prob, gradient) =
            log_prob_and_gradient_at(self.param, self.shapes, state, &concat(&position))?;
        *self.in_state.borrow_mut() = Some(Rc::clone(&position));

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

/// The two halves of a trajectory being merged: the subtree that was there first (`init`) and the one just built
/// on from its end (`last`).
struct Halves<'a> {
    /// The end of `init` away from `last`.
    init_outer: &'a LeapfrogState,
    /// The end of `init` that `last` was built from.
    init_inner: &'a LeapfrogState,
    /// The sum of `init`'s momenta.
    init_rho: &'a [DoubleTensor],
    /// The end of `last` next to `init`.
    last_inner: &'a LeapfrogState,
    /// The end of `last` away from `init`.
    last_outer: &'a LeapfrogState,
    /// The sum of `last`'s momenta.
    last_rho: &'a [DoubleTensor],
}

impl Halves<'_> {
    /// The generalised no-U-turn criterion on the merged trajectory, whose summed momentum is `rho`, with the extra
    /// checks across the boundary between the halves that Stan added in 2.21 (stan-dev/stan#2800): the merged
    /// trajectory's ends against `rho`, and then each half extended by the nearest point of the other. Without the
    /// extra checks a trajectory can run on past a U-turn when its halves together nearly complete an orbit, so
    /// that their momenta almost cancel.
    fn u_turned(&self, rho: &[DoubleTensor]) -> Result<bool> {
        if is_u_turning(&self.init_outer.velocity, &self.last_outer.velocity, rho)? {
            return Ok(true);
        }
        let init_extended = add_all(self.init_rho, &self.last_inner.momentum)?;
        if is_u_turning(
            &self.init_outer.velocity,
            &self.last_inner.velocity,
            &init_extended,
        )? {
            return Ok(true);
        }
        let last_extended = add_all(self.last_rho, &self.init_inner.momentum)?;
        is_u_turning(
            &self.init_inner.velocity,
            &self.last_outer.velocity,
            &last_extended,
        )
    }
}

/// A (sub)trajectory's ends as (the end behind, the end ahead) when travelling in `direction`.
fn ends(tree: &Tree, direction: f64) -> (Rc<LeapfrogState>, Rc<LeapfrogState>) {
    if direction < 0.0 {
        (Rc::clone(&tree.forward), Rc::clone(&tree.backward))
    } else {
        (Rc::clone(&tree.backward), Rc::clone(&tree.forward))
    }
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
        // as in Stan, only an increase is divergent: a large fall in energy is a very likely move, not instability
        if energy_change > self.max_energy_change || energy_change.is_nan() {
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
                energy: after.energy,
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
            let (init_outer, init_inner) = ends(&tree, direction);
            let extension =
                self.build(state, &init_inner, direction, height - 1, step_size, rng)?;
            let (last_inner, last_outer) = ends(&extension, direction);
            if direction < 0.0 {
                tree.backward = Rc::clone(&extension.backward);
            } else {
                tree.forward = Rc::clone(&extension.forward);
            }
            tree.diverged = extension.diverged;
            tree.u_turned = extension.u_turned;
            if extension.should_continue() {
                let rho = add_all(&tree.sum_momentum, &extension.sum_momentum)?;
                tree.u_turned = Halves {
                    init_outer: &init_outer,
                    init_inner: &init_inner,
                    init_rho: &tree.sum_momentum,
                    last_inner: &last_inner,
                    last_outer: &last_outer,
                    last_rho: &extension.sum_momentum,
                }
                .u_turned(&rho)?;
                tree.sum_momentum = rho;
                let total = log_sum_exp([tree.log_sum_weight, extension.log_sum_weight]);
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
        let (init_outer, init_inner) = ends(tree, direction);
        let other = self.build(state, &init_inner, direction, tree.height, step_size, rng)?;
        let (last_inner, last_outer) = ends(&other, direction);
        if direction < 0.0 {
            tree.backward = Rc::clone(&other.backward);
        } else {
            tree.forward = Rc::clone(&other.forward);
        }
        tree.sum_metropolis_acceptance += other.sum_metropolis_acceptance;
        tree.size += other.size;
        tree.diverged = other.diverged;
        tree.u_turned = other.u_turned;
        if other.should_continue() {
            if accept_with_log_probability(other.log_sum_weight - tree.log_sum_weight, rng) {
                tree.proposal = other.proposal;
            }
            tree.log_sum_weight = log_sum_exp([tree.log_sum_weight, other.log_sum_weight]);
            let rho = add_all(&tree.sum_momentum, &other.sum_momentum)?;
            tree.u_turned = Halves {
                init_outer: &init_outer,
                init_inner: &init_inner,
                init_rho: &tree.sum_momentum,
                last_inner: &last_inner,
                last_outer: &last_outer,
                last_rho: &other.sum_momentum,
            }
            .u_turned(&rho)?;
            tree.sum_momentum = rho;
        }
        tree.height += 1;
        Ok(())
    }
}

/// A running NUTS chain (Java's `NUTSSampler`). It owns its adapted step size and potential.
#[derive(Debug)]
pub struct NutsSampler<'m> {
    param: Parameterisation<'m>,
    /// The shape of each latent's block of the position.
    shapes: Vec<Vec<usize>>,
    config: Nuts,
    potential: PotentialState,
    step_size: AdaptiveStepSize,
    nodes: Vec<NodeId>,
    proposal: Candidate,
    /// The model's log probability at the proposal (without the Jacobian), as recorded.
    sample_log_prob: f64,
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

    /// Stan's heuristic for a reasonable step size (its `init_stepsize`, after Hoffman & Gelman 2014, Algorithm 4):
    /// from the current step size, doubles it while one leapfrog step from the current sample (with fresh momentum
    /// each time) has acceptance probability above 0.8, or halves it while it is below, then restarts dual averaging
    /// from it. The search stops at the bounds Stan stops at (above 10⁷, or 0) without going past them. Leaves the
    /// state at the current sample.
    fn find_reasonable_step_size(&mut self, state: &mut State, rng: &mut KeanuRng) -> Result<()> {
        let mut step_size = self.step_size.step_size;
        if step_size.is_finite() && step_size > 0.0 && step_size <= 1e7 {
            let integrator = Integrator {
                param: &self.param,
                shapes: &self.shapes,
                potential: &self.potential,
                start_energy: f64::NAN,
                max_energy_change: self.config.max_energy_change,
                in_state: RefCell::new(None),
            };
            let proposal = &self.proposal;
            let mut log_acceptance = |step_size: f64, rng: &mut KeanuRng| -> Result<f64> {
                let start = LeapfrogState::new(
                    Rc::clone(&proposal.position),
                    self.potential.random_momentum(rng)?,
                    Rc::clone(&proposal.gradient),
                    proposal.log_prob,
                    &self.potential,
                )?;
                let after = integrator.leapfrog(state, &start, step_size)?;
                let delta = start.energy - after.energy;
                Ok(if delta.is_nan() {
                    f64::NEG_INFINITY
                } else {
                    delta
                })
            };
            let threshold = 0.8f64.ln();
            let increase = log_acceptance(step_size, rng)? > threshold;
            loop {
                let delta = log_acceptance(step_size, rng)?;
                // delta is never NaN
                if (increase && delta <= threshold) || (!increase && delta >= threshold) {
                    break;
                }
                let next = if increase {
                    2.0 * step_size
                } else {
                    0.5 * step_size
                };
                if next > 1e7 || next == 0.0 {
                    break;
                }
                step_size = next;
            }
            self.param.to_state(&concat(&proposal.position), state)?;
        }
        self.step_size.step_size = step_size;
        self.step_size.restart();
        Ok(())
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
        let start_energy = start.energy;
        let integrator = Integrator {
            param: &self.param,
            shapes: &self.shapes,
            potential: &self.potential,
            start_energy,
            max_energy_change: self.config.max_energy_change,
            in_state: RefCell::new(None),
        };
        let mut tree = Tree {
            sum_momentum: start.momentum.clone(),
            forward: Rc::clone(&start),
            backward: start,
            // the previous sample stays the candidate unless the trajectory moves, with this step's momentum
            proposal: Some(Candidate {
                energy: start_energy,
                ..self.proposal.clone()
            }),
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
        let z = concat(&self.proposal.position);
        if !in_state.is_some_and(|p| Rc::ptr_eq(&p, &self.proposal.position)) {
            self.param.to_state(&z, state)?;
        }
        self.sample_log_prob = self.proposal.log_prob - self.param.log_det_jacobian(&z)?;

        if self.config.save_statistics {
            self.statistics.step_size.push(self.step_size.step_size);
            self.statistics
                .mean_tree_accept
                .push(self.step_size.accept_rate);
            self.statistics.log_prob.push(self.sample_log_prob);
            self.statistics.tree_size.push(tree.size);
        }
        if self.step_count < self.config.adapt_count {
            // as Stan's adaptive samplers: learn the step size from the tree, then the potential from the draw, and
            // when a slow window ends, find a step size for the new potential and restart dual averaging from it
            if self.config.adapt_step_size {
                self.step_size.learn(&tree);
            }
            let potential_changed =
                self.config.adapt_potential && self.potential.update(&self.proposal.position)?;
            if potential_changed && self.config.adapt_step_size {
                self.find_reasonable_step_size(state, rng)?;
            }
            if self.step_count + 1 == self.config.adapt_count && self.config.adapt_step_size {
                self.step_size.complete();
            }
        }
        self.statistics.energy.push(self.proposal.energy);
        if self.step_count >= self.config.adapt_count && tree.diverged {
            self.statistics.divergent_steps.push(self.step_count);
        }
        self.step_count += 1;
        Ok(())
    }

    fn record(&self, state: &State, samples: &mut NetworkSamples) -> Result<()> {
        samples.record(state, self.sample_log_prob)
    }

    fn graph(&self) -> &Graph {
        self.param.model().graph()
    }

    fn sampled_nodes(&self) -> &[NodeId] {
        &self.nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The (0-based) adaptation steps at which the potential is updated.
    fn window_ends(settings: &QuadraticPotential, adapt_count: usize) -> Vec<usize> {
        let like = [DoubleTensor::scalar(0.0)];
        let mut potential = PotentialState::new(settings, adapt_count, &like);
        (0..adapt_count)
            .filter(|&i| potential.update(&[DoubleTensor::scalar(i as f64)]).unwrap())
            .collect()
    }

    #[test]
    fn slow_windows_follow_stans_schedule() {
        let defaults = QuadraticPotential::default();
        // Stan's windows for 1000 warmup iterations: [75, 100), [100, 150), [150, 250), [250, 450) and [450, 950),
        // the last stretched to the terminal buffer of 50
        assert_eq!(window_ends(&defaults, 1000), [99, 149, 249, 449, 949]);
        assert_eq!(window_ends(&defaults, 500), [99, 149, 249, 449]);
        // a third window would end at 249, leaving no room for a fourth before 400, so it is stretched to 399
        assert_eq!(window_ends(&defaults, 450), [99, 149, 399]);
        assert_eq!(window_ends(&defaults, 200), [99, 149]);
        // 75 + 25 + 50 is exactly 150: one window
        assert_eq!(window_ends(&defaults, 150), [99]);
        // too short for the three stages: 15% (15), 75% (75) and 10% (10)
        assert_eq!(window_ends(&defaults, 100), [89]);
        assert_eq!(window_ends(&defaults, 20), [17]);
        // fewer than 20 steps: no potential adaptation
        assert!(window_ends(&defaults, 19).is_empty());
        assert!(window_ends(&defaults, 0).is_empty());
        let custom = defaults
            .with_initial_buffer(10)
            .with_adaptation_window(5)
            .with_terminal_buffer(5);
        assert_eq!(window_ends(&custom, 60), [14, 24, 54]);
    }

    #[test]
    fn the_variance_is_the_regularised_sample_variance_of_the_window() {
        let settings = QuadraticPotential::default();
        let like = [DoubleTensor::scalar(0.0)];
        // 150 steps: one slow window, steps 75 to 99
        let mut potential = PotentialState::new(&settings, 150, &like);
        let draws: Vec<f64> = (0..150).map(|i| ((i * 37) % 11) as f64).collect();
        for (i, &x) in draws.iter().enumerate() {
            let updated = potential.update(&[DoubleTensor::scalar(x)]).unwrap();
            assert_eq!(updated, i == 99, "step {i}");
            if i < 99 {
                assert_eq!(potential.variance[0].to_scalar().unwrap(), 1.0);
            }
        }
        let window = &draws[75..100];
        let n = window.len() as f64;
        let mean = window.iter().sum::<f64>() / n;
        let var = window.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
        let expected = n / (n + 5.0) * var + 1e-3 * 5.0 / (n + 5.0);
        let actual = potential.variance[0].to_scalar().unwrap();
        assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");
        let sd = potential.standard_deviation[0].to_scalar().unwrap();
        assert!((sd - expected.sqrt()).abs() < 1e-12);
    }

    /// Whether one leapfrog step from x = 0.3 with momentum 0.5, in a trajectory that started at energy
    /// `start_energy`, is divergent.
    fn diverges_from(start_energy: f64) -> bool {
        let b = keanu_graph::GraphBuilder::new();
        let x = b.gaussian(0.0, 1.0);
        let x = x.id();
        let (graph, mut state) = b.build().unwrap();
        state.set(&graph, x, DoubleTensor::scalar(0.3)).unwrap();
        let model = Model::new(graph, &state).unwrap();
        let param = Parameterisation::new(&model, &state).unwrap();
        let shapes = vec![vec![]];
        let position = [DoubleTensor::scalar(0.3)];
        let potential = PotentialState::new(&QuadraticPotential::default(), 0, &position);
        let (log_prob, gradient) =
            log_prob_and_gradient_at(&param, &shapes, &mut state, &concat(&position)).unwrap();
        let from = LeapfrogState::new(
            Rc::new(position.to_vec()),
            vec![DoubleTensor::scalar(0.5)],
            Rc::new(gradient),
            log_prob,
            &potential,
        )
        .unwrap();
        let integrator = Integrator {
            param: &param,
            shapes: &shapes,
            potential: &potential,
            start_energy,
            max_energy_change: 1000.0,
            in_state: RefCell::new(None),
        };
        let tree = integrator.base_case(&mut state, &from, 1.0, 0.1).unwrap();
        assert_eq!(tree.proposal.is_none(), tree.diverged);
        tree.diverged
    }

    #[test]
    fn only_an_energy_increase_is_a_divergence() {
        // the step's energy is about 1: a trajectory that started 2000 above it has fallen, which (as in Stan)
        // is not a divergence; one that started 2000 below has risen, which is
        assert!(!diverges_from(2000.0));
        assert!(diverges_from(-2000.0));
        assert!(!diverges_from(0.0));
        // Stan's comparison is strict, and a NaN energy change is divergent
        assert!(diverges_from(f64::NAN));
    }
}
