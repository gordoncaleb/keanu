//! Pathfinder variational inference, after Zhang, Carpenter, Gelman and Vehtari, "Pathfinder: Parallel
//! quasi-Newton variational inference", JMLR 23 (2022), with the paper's and Stan's defaults.

use crate::error::{AlgorithmError, Result};
use crate::optimizer::Lbfgs;
use crate::optimizer::lbfgs::Iterate;
use crate::psis::{khat_threshold, psis};
use crate::samples::NetworkSamples;
use crate::support::{
    Unevaluable, check_latents, dot, evaluate_all, sample_approximation, start_point,
};
use keanu_distributions::KeanuRng;
use keanu_graph::{Model, NodeId, Parameterisation, State};
use keanu_tensor::DoubleTensor;
use std::collections::VecDeque;

/// ln(2π).
const LN_2PI: f64 = 1.837_877_066_409_345_3;
/// The random starting points tried for each path before giving up, as Stan's initialisation tries.
const INIT_ATTEMPTS: usize = 100;

/// Pathfinder configuration. The defaults are the paper's: an L-BFGS history of 6, 5 draws for each ELBO
/// estimate, 4 paths, and 1000 draws (Stan's settings otherwise, with at most 1000 L-BFGS iterations). The
/// configuration holds no state, so it can be reused for any number of fits.
///
/// A single path ([`single_path`](Self::single_path)) maximises the log density of the model's continuous latents
/// in unconstrained space (the model's log probability plus the log-Jacobian of the transforms, see
/// [`Parameterisation`]) with L-BFGS, keeping every iterate and its gradient. At each iterate it forms a Gaussian
/// approximation of the posterior: its covariance is the L-BFGS estimate of the inverse Hessian, built from the
/// last [`history_size`](Lbfgs::history_size) steps and gradient changes whose curvature is positive (sᵀy > ε‖y‖²)
/// with the initial inverse Hessian diag(α) updated at each such pair by the diagonal BFGS update of Gilbert and
/// Lemaréchal (1989), in the compact low-rank-plus-diagonal form of Byrd, Nocedal and Schnabel (1994); its mean is
/// the Newton step from the iterate along that inverse Hessian. Each approximation's evidence lower bound (ELBO)
/// is estimated from [`elbo_draws`](Self::elbo_draws) draws, and the approximation with the highest is the path's
/// result, from which [`draws`](Self::draws) draws are taken. Draws are made through a thin QR factorisation of the
/// low-rank part, so each costs O(n·m) for n coordinates and m pairs (or, when 2m ≥ n, through the Cholesky
/// factor of the dense covariance).
///
/// Multi-path Pathfinder ([`multi_path`](Self::multi_path)) runs [`paths`](Self::paths) single paths from random
/// starting points (each coordinate uniform on (−2, 2), as Stan initialises), pools their draws and, with
/// [`resample`](Self::resample), resamples [`draws`](Self::draws) of them with replacement in proportion to their
/// Pareto-smoothed importance weights ([`psis`]: the log density minus the log density of the approximation each
/// came from). Several paths guard against a path stopping in a poor local optimum, and the importance weights
/// correct some of the Gaussian approximations' error. The result reports the Pareto k̂ of the weights: above
/// about 0.7 the approximation is too far from the posterior for importance sampling to fix.
///
/// Pathfinder is a fast approximation: much cheaper than ADVI (which needs thousands of stochastic gradient steps)
/// and NUTS, and a good way to start NUTS chains near the typical set
/// ([`PathfinderResult::initial_states`]). It handles continuous latents only.
///
/// ```
/// use keanu_algorithms::vi::Pathfinder;
/// use keanu_distributions::KeanuRng;
/// use keanu_graph::{GraphBuilder, Model};
/// use keanu_tensor::DoubleTensor;
///
/// let b = GraphBuilder::new();
/// let mu = b.gaussian(0.0, 10.0);
/// let x = b.gaussian_with_shape(&[3], mu, 1.0);
/// b.observe(x, DoubleTensor::vector(vec![4.1, 5.2, 4.7]));
/// let mu = mu.id();
/// let (graph, mut state) = b.build()?;
/// let model = Model::new(graph, &state)?;
///
/// let mut rng = KeanuRng::seed_from_u64(1);
/// let result = Pathfinder::default().multi_path(&model, &mut state, &mut rng)?;
/// // the posterior of mu is N(14 / 3.01, 1 / 3.01)
/// let samples = result.sample(&mut state, &[mu])?;
/// let mean = samples.f64(mu).unwrap().mean()?.to_scalar().unwrap();
/// assert!((mean - 14.0 / 3.01).abs() < 0.1);
/// assert!(result.is_reliable());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Pathfinder {
    /// The L-BFGS settings of each path: its history size (default 6), iteration limit (default 1000) and
    /// convergence tolerances (Stan's).
    pub lbfgs: Lbfgs,
    /// The draws each ELBO estimate averages (default 5).
    pub elbo_draws: usize,
    /// The draws to return (default 1000). Each path also draws this many from its approximation.
    pub draws: usize,
    /// The paths multi-path Pathfinder runs (default 4).
    pub paths: usize,
    /// Whether multi-path Pathfinder resamples its pooled draws by their importance weights (default true).
    /// Without it, every path's draws are returned, unweighted.
    pub resample: bool,
}

impl Default for Pathfinder {
    fn default() -> Self {
        Self {
            lbfgs: Lbfgs::default()
                .with_history_size(6)
                .with_max_iterations(1000),
            elbo_draws: 5,
            draws: 1000,
            paths: 4,
            resample: true,
        }
    }
}

/// Setters for each setting, for use as `Pathfinder::default().with_...(...)`.
impl Pathfinder {
    /// Sets [`lbfgs`](Self::lbfgs).
    #[must_use]
    pub fn with_lbfgs(mut self, lbfgs: Lbfgs) -> Self {
        self.lbfgs = lbfgs;
        self
    }

    /// Sets [`elbo_draws`](Self::elbo_draws).
    #[must_use]
    pub fn with_elbo_draws(mut self, elbo_draws: usize) -> Self {
        self.elbo_draws = elbo_draws;
        self
    }

    /// Sets [`draws`](Self::draws).
    #[must_use]
    pub fn with_draws(mut self, draws: usize) -> Self {
        self.draws = draws;
        self
    }

    /// Sets [`paths`](Self::paths).
    #[must_use]
    pub fn with_paths(mut self, paths: usize) -> Self {
        self.paths = paths;
        self
    }

    /// Sets [`resample`](Self::resample).
    #[must_use]
    pub fn with_resample(mut self, resample: bool) -> Self {
        self.resample = resample;
        self
    }
}

impl Pathfinder {
    /// The default settings.
    pub fn new() -> Self {
        Self::default()
    }

    fn validate(&self) -> Result<()> {
        let invalid = |m: &str| Err(AlgorithmError::InvalidSetting(m.into()));
        if self.elbo_draws == 0 {
            return invalid("the number of ELBO draws must be greater than 0");
        }
        if self.draws == 0 {
            return invalid("the number of draws must be greater than 0");
        }
        if self.paths == 0 {
            return invalid("the number of paths must be greater than 0");
        }
        Ok(())
    }

    /// Runs one path from the latents' current values in `state` (latents without a value are sampled from their
    /// priors first) and returns [`draws`](Self::draws) draws from its best approximation, unweighted, with the
    /// Pareto k̂ of their importance ratios. Leaves `state` at the constrained values of the approximation's mean
    /// or, where the model can't be evaluated there (a failed ODE solve), of the L-BFGS iterate the approximation
    /// was built at (see [`PathfinderPath::best_iteration`]), where it was evaluated.
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidSetting`] for an out-of-range setting; [`AlgorithmError::NoLatents`] if the model
    /// has no continuous latents; [`AlgorithmError::DiscreteLatents`] if it has discrete latents;
    /// [`AlgorithmError::ImpossibleStart`] if the starting point has zero probability or a latent's value is outside
    /// its support or on its boundary; an error from [`Parameterisation::new`] if a latent's support has no
    /// transform; and [`AlgorithmError::NonFiniteElbo`] if no approximation along the path has a finite ELBO.
    pub fn single_path<'m>(
        &self,
        model: &'m Model,
        state: &mut State,
        rng: &mut KeanuRng,
    ) -> Result<PathfinderResult<'m>> {
        let param = self.prepare(model, state, rng)?;
        let start = start_point(&param, state)?;
        let path = self.run_path(&param, state, start, rng)?;
        let (draws, log_density, approximation_log_density) =
            self.draw_from(&param, state, &path, rng)?;
        let pareto_k = pareto_k(&log_density, &approximation_log_density);
        // a mean where the model can't be evaluated (a failed ODE solve) leaves the state at the best iterate
        if !param.try_to_state(path.mean(), state)? {
            param.to_state(&path.iterate, state)?;
        }
        Ok(PathfinderResult {
            param,
            weighted_draws: draws.len(),
            evaluations: path.evaluations,
            paths: vec![path],
            failed_paths: 0,
            draws,
            log_density,
            approximation_log_density,
            pareto_k,
        })
    }

    /// Runs [`paths`](Self::paths) single paths, each from a random starting point (every coordinate uniform on
    /// (−2, 2) in unconstrained space; up to 100 points are tried until one has a finite log density and
    /// gradient), and pools their draws, resampled by importance weight with [`resample`](Self::resample) (see the
    /// [type documentation](Self)). Paths that fail (no finite ELBO, or no possible starting point) are skipped
    /// and counted ([`PathfinderResult::failed_paths`]). The paths run one after another. Leaves `state` at the
    /// constrained values of the last draw where the model can be evaluated (not a failed ODE solve), or of the
    /// best path's iterate if there is none.
    ///
    /// # Errors
    ///
    /// As [`single_path`](Self::single_path), except that the latents' values in `state` are not used; the error
    /// of the last path if every path fails; and [`AlgorithmError::NonFiniteElbo`] if every pooled draw has zero
    /// probability.
    pub fn multi_path<'m>(
        &self,
        model: &'m Model,
        state: &mut State,
        rng: &mut KeanuRng,
    ) -> Result<PathfinderResult<'m>> {
        let param = self.prepare(model, state, rng)?;
        let mut paths = Vec::with_capacity(self.paths);
        let mut failed_paths = 0;
        let mut last_error = None;
        let mut evaluations = 0;
        let (mut draws, mut log_density, mut approximation_log_density) =
            (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..self.paths {
            let outcome = random_start(&param, state, rng, &mut evaluations)
                .and_then(|start| self.run_path(&param, state, start, rng));
            match outcome {
                Ok(path) => {
                    evaluations += path.evaluations;
                    let (d, lp, lq) = self.draw_from(&param, state, &path, rng)?;
                    draws.extend(d);
                    log_density.extend(lp);
                    approximation_log_density.extend(lq);
                    paths.push(path);
                }
                Err(
                    error @ (AlgorithmError::ImpossibleStart | AlgorithmError::NonFiniteElbo(_)),
                ) => {
                    failed_paths += 1;
                    last_error = Some(error);
                }
                Err(error) => return Err(error),
            }
        }
        if paths.is_empty() {
            return Err(last_error.unwrap_or(AlgorithmError::ImpossibleStart));
        }
        let ratios: Vec<f64> = log_density
            .iter()
            .zip(&approximation_log_density)
            .map(|(lp, lq)| lp - lq)
            .collect();
        let smoothed = psis(&ratios, 1.0).map_err(|_| {
            AlgorithmError::NonFiniteElbo("every Pathfinder draw has zero probability".into())
        })?;
        let pareto_k = khat(&ratios, &smoothed);
        let weighted_draws = draws.len();
        if self.resample {
            let weights = smoothed.weights();
            let chosen = resample(&weights, self.draws, rng);
            draws = chosen.iter().map(|&i| draws[i].clone()).collect();
            log_density = chosen.iter().map(|&i| log_density[i]).collect();
            approximation_log_density = chosen
                .iter()
                .map(|&i| approximation_log_density[i])
                .collect();
        }
        // the last draw where the model can be evaluated (not a failed ODE solve), else the best path's iterate
        let mut placed = false;
        for draw in draws.iter().rev() {
            if param.try_to_state(draw, state)? {
                placed = true;
                break;
            }
        }
        if !placed {
            let best = paths
                .iter()
                .max_by(|a, b| a.elbo.total_cmp(&b.elbo))
                .expect("a path succeeded");
            param.to_state(&best.iterate, state)?;
        }
        Ok(PathfinderResult {
            param,
            paths,
            failed_paths,
            draws,
            log_density,
            approximation_log_density,
            pareto_k,
            weighted_draws,
            evaluations,
        })
    }

    /// Validates the settings and the model, and parameterises it.
    fn prepare<'m>(
        &self,
        model: &'m Model,
        state: &mut State,
        rng: &mut KeanuRng,
    ) -> Result<Parameterisation<'m>> {
        self.validate()?;
        check_latents(model)?;
        evaluate_all(model.graph(), state, rng)?;
        let param = Parameterisation::new(model, state)?;
        if param.dimension() == 0 {
            return Err(AlgorithmError::NoLatents);
        }
        Ok(param)
    }

    /// One path from `start`: the L-BFGS path, an approximation at each iterate, and the best of them by ELBO.
    fn run_path(
        &self,
        param: &Parameterisation<'_>,
        state: &mut State,
        start: Vec<f64>,
        rng: &mut KeanuRng,
    ) -> Result<PathfinderPath> {
        let n = param.dimension();
        let mut evaluations = 0;
        // L-BFGS minimises the negative log density; a point where it or its gradient is not finite counts as too
        // long a step
        let (iterates, converged) = self.lbfgs.minimize_path(start, |z| {
            evaluations += 1;
            let (log_density, gradient) = param.log_prob_and_gradient(z, true, state)?;
            Ok(match gradient {
                Some(g) if log_density.is_finite() && g.iter().all(|x| x.is_finite()) => {
                    (-log_density, g.iter().map(|x| -x).collect())
                }
                _ => (f64::INFINITY, vec![f64::NAN; n]),
            })
        })?;

        let mut alpha = vec![1.0; n];
        let mut pairs: VecDeque<(Vec<f64>, Vec<f64>)> =
            VecDeque::with_capacity(self.lbfgs.history_size);
        let mut elbo_trace = Vec::with_capacity(iterates.len());
        let mut best: Option<(f64, usize, Approximation)> = None;
        // without a step there is only the start to approximate at
        let first = if iterates.len() == 1 { 0 } else { 1 };
        for l in first..iterates.len() {
            if l > 0 {
                let (from, to) = (&iterates[l - 1], &iterates[l]);
                let s: Vec<f64> = to
                    .point
                    .iter()
                    .zip(&from.point)
                    .map(|(a, b)| a - b)
                    .collect();
                let y: Vec<f64> = to
                    .gradient
                    .iter()
                    .zip(&from.gradient)
                    .map(|(a, b)| a - b)
                    .collect();
                let (ys, yy) = (dot(&y, &s), dot(&y, &y));
                if ys > f64::EPSILON * yy && ys.is_finite() {
                    update_diagonal(&mut alpha, &s, &y);
                    if pairs.len() == self.lbfgs.history_size {
                        pairs.pop_front();
                    }
                    pairs.push_back((s, y));
                }
            }
            let elbo = match Approximation::new(&iterates[l], &alpha, &pairs) {
                Some(approximation) => {
                    let elbo = self.elbo(param, state, &approximation, rng)?;
                    evaluations += self.elbo_draws;
                    if best.as_ref().is_none_or(|(b, ..)| elbo > *b) && elbo.is_finite() {
                        best = Some((elbo, l, approximation));
                    }
                    elbo
                }
                None => f64::NEG_INFINITY,
            };
            elbo_trace.push(elbo);
        }
        let Some((elbo, best_iteration, approximation)) = best else {
            return Err(AlgorithmError::NonFiniteElbo(
                "no approximation along the Pathfinder path has a finite ELBO".into(),
            ));
        };
        Ok(PathfinderPath {
            iterate: iterates[best_iteration].point.clone(),
            approximation,
            elbo,
            elbo_trace,
            best_iteration,
            iterations: iterates.len() - 1,
            converged,
            evaluations,
        })
    }

    /// The ELBO of an approximation, estimated from [`elbo_draws`](Self::elbo_draws) draws.
    fn elbo(
        &self,
        param: &Parameterisation<'_>,
        state: &mut State,
        approximation: &Approximation,
        rng: &mut KeanuRng,
    ) -> Result<f64> {
        let mut sum = 0.0;
        for _ in 0..self.elbo_draws {
            let (z, log_q) = approximation.draw(rng);
            sum += log_density(param, state, &z)? - log_q;
        }
        Ok(sum / self.elbo_draws as f64)
    }

    /// [`draws`](Self::draws) draws from a path's approximation, with the log density and the approximation's log
    /// density at each.
    #[allow(clippy::type_complexity)]
    fn draw_from(
        &self,
        param: &Parameterisation<'_>,
        state: &mut State,
        path: &PathfinderPath,
        rng: &mut KeanuRng,
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>, Vec<f64>)> {
        let mut draws = Vec::with_capacity(self.draws);
        let mut log_densities = Vec::with_capacity(self.draws);
        let mut log_qs = Vec::with_capacity(self.draws);
        for _ in 0..self.draws {
            let (z, log_q) = path.approximation.draw(rng);
            log_densities.push(log_density(param, state, &z)?);
            log_qs.push(log_q);
            draws.push(z);
        }
        Ok((draws, log_densities, log_qs))
    }
}

/// The log density of the unconstrained point `z` (with the Jacobian), −∞ where it is impossible or NaN.
fn log_density(param: &Parameterisation<'_>, state: &mut State, z: &[f64]) -> Result<f64> {
    let lp = param.log_prob(z, true, state)?;
    Ok(if lp.is_nan() { f64::NEG_INFINITY } else { lp })
}

/// A random starting point with a finite log density and gradient.
fn random_start(
    param: &Parameterisation<'_>,
    state: &mut State,
    rng: &mut KeanuRng,
    evaluations: &mut usize,
) -> Result<Vec<f64>> {
    for _ in 0..INIT_ATTEMPTS {
        let z = param.uniform_initial_point(rng);
        *evaluations += 1;
        let (lp, gradient) = param.log_prob_and_gradient(&z, true, state)?;
        if lp.is_finite() && gradient.is_some_and(|g| g.iter().all(|x| x.is_finite())) {
            return Ok(z);
        }
    }
    Err(AlgorithmError::ImpossibleStart)
}

/// Log ratios whose finite values all lie within this of each other are constant: their weights are equal to this
/// precision, and k̂ fitted to the rounding errors of an exact approximation would be meaningless.
const CONSTANT_RATIOS: f64 = 1e-8;

/// The Pareto k̂ of PSIS on the log ratios, as [`Psis::pareto_k`](crate::psis::Psis::pareto_k), except that
/// ratios equal to within [`CONSTANT_RATIOS`] (an exact approximation) give −∞, PSIS's value for equal ratios.
fn khat(ratios: &[f64], smoothed: &crate::psis::Psis) -> f64 {
    let (low, high) = ratios
        .iter()
        .filter(|r| r.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &r| {
            (l.min(r), h.max(r))
        });
    if high - low < CONSTANT_RATIOS {
        f64::NEG_INFINITY
    } else {
        smoothed.pareto_k()
    }
}

/// The Pareto k̂ of the importance ratios of draws ([`khat`]), NaN if every draw is impossible.
fn pareto_k(log_density: &[f64], approximation_log_density: &[f64]) -> f64 {
    let ratios: Vec<f64> = log_density
        .iter()
        .zip(approximation_log_density)
        .map(|(lp, lq)| lp - lq)
        .collect();
    psis(&ratios, 1.0).map_or(f64::NAN, |p| khat(&ratios, &p))
}

/// `count` indices drawn with replacement in proportion to `weights` (which sum to 1).
fn resample(weights: &[f64], count: usize, rng: &mut KeanuRng) -> Vec<usize> {
    let mut cumulative = Vec::with_capacity(weights.len());
    let mut total = 0.0;
    for w in weights {
        total += w;
        cumulative.push(total);
    }
    (0..count)
        .map(|_| {
            let u = rng.next_f64() * total;
            cumulative
                .partition_point(|&c| c <= u)
                .min(weights.len() - 1)
        })
        .collect()
}

/// The diagonal BFGS update of the initial inverse Hessian diag(α) for the pair (s, y) (Gilbert and Lemaréchal,
/// 1989; Zhang et al., 2022, Algorithm 4): with a = yᵀ diag(α) y, b = yᵀ s and c = sᵀ diag(α)⁻¹ s,
/// αᵢ ← 1 / (a / (b αᵢ) + yᵢ² / b − a sᵢ² / (b c αᵢ²)). An element whose update is not finite and positive keeps
/// its value.
fn update_diagonal(alpha: &mut [f64], s: &[f64], y: &[f64]) {
    let a: f64 = alpha.iter().zip(y).map(|(al, y)| al * y * y).sum();
    let b = dot(y, s);
    let c: f64 = alpha.iter().zip(s).map(|(al, s)| s * s / al).sum();
    for ((al, s), y) in alpha.iter_mut().zip(s).zip(y) {
        let updated = 1.0 / (a / (b * *al) + y * y / b - a * s * s / (b * c * *al * *al));
        if updated.is_finite() && updated > 0.0 {
            *al = updated;
        }
    }
}

/// How a Gaussian approximation's draws are made.
#[derive(Debug, Clone)]
enum Factor {
    /// The lower Cholesky factor of the covariance, n × n row-major.
    Dense(Vec<f64>),
    /// Covariance diag(√α) (I + Q (L Lᵀ − I) Qᵀ) diag(√α), with Q n × k (column-major: one vector per column) with
    /// orthonormal columns and L k × k lower triangular (row-major).
    LowRank {
        sqrt_alpha: Vec<f64>,
        q: Vec<Vec<f64>>,
        l: Vec<f64>,
    },
}

/// A Gaussian approximation N(μ, Σ) of the posterior in unconstrained space.
#[derive(Debug, Clone)]
struct Approximation {
    mean: Vec<f64>,
    factor: Factor,
    log_det_covariance: f64,
}

impl Approximation {
    /// The approximation at an L-BFGS iterate, from the diagonal α and the curvature pairs (s, y) of the negative
    /// log density (Zhang et al., 2022, Algorithms 2 and 3). `None` if the covariance is not positive definite in
    /// floating point.
    fn new(at: &Iterate, alpha: &[f64], pairs: &VecDeque<(Vec<f64>, Vec<f64>)>) -> Option<Self> {
        let n = alpha.len();
        let m = pairs.len();
        // ∇ log p at the iterate
        let gradient: Vec<f64> = at.gradient.iter().map(|g| -g).collect();
        // R (upper triangular, R_ij = s_iᵀ y_j for i ≤ j), its inverse, D = diag(s_iᵀ y_i) and Yᵀ diag(α) Y
        let mut r = vec![0.0; m * m];
        let mut yay = vec![0.0; m * m];
        for (i, (s_i, y_i)) in pairs.iter().enumerate() {
            for (j, (_, y_j)) in pairs.iter().enumerate() {
                if i <= j {
                    r[i * m + j] = dot(s_i, y_j);
                }
                yay[i * m + j] = alpha
                    .iter()
                    .zip(y_i)
                    .zip(y_j)
                    .map(|((a, yi), yj)| a * yi * yj)
                    .sum();
            }
        }
        let r_inv = invert_upper(&r, m)?;
        // γ, 2m × 2m, for β = [diag(α) Y, S]: [[0, −R⁻¹], [−R⁻ᵀ, R⁻ᵀ (D + Yᵀ diag(α) Y) R⁻¹]]
        let k = 2 * m;
        let mut gamma = vec![0.0; k * k];
        let mut middle = yay;
        for i in 0..m {
            middle[i * m + i] += r[i * m + i];
        }
        let lower_right = matmul(&transpose(&r_inv, m), &matmul(&middle, &r_inv, m), m);
        for i in 0..m {
            for j in 0..m {
                gamma[i * k + (m + j)] = -r_inv[i * m + j];
                gamma[(m + i) * k + j] = -r_inv[j * m + i];
                gamma[(m + i) * k + (m + j)] = lower_right[i * m + j];
            }
        }
        // β's columns
        let beta: Vec<Vec<f64>> = pairs
            .iter()
            .map(|(_, y)| alpha.iter().zip(y).map(|(a, y)| a * y).collect())
            .chain(pairs.iter().map(|(s, _)| s.clone()))
            .collect();
        // μ = x + H ∇log p, H = diag(α) + β γ βᵀ
        let beta_t_g: Vec<f64> = beta.iter().map(|b| dot(b, &gradient)).collect();
        let coefficients: Vec<f64> = (0..k)
            .map(|i| (0..k).map(|j| gamma[i * k + j] * beta_t_g[j]).sum())
            .collect();
        let mut mean: Vec<f64> = at
            .point
            .iter()
            .zip(alpha)
            .zip(&gradient)
            .map(|((x, a), g)| x + a * g)
            .collect();
        for (column, c) in beta.iter().zip(&coefficients) {
            for (mu, b) in mean.iter_mut().zip(column) {
                *mu += c * b;
            }
        }

        let (factor, log_det_covariance) = if k >= n {
            let mut covariance = vec![0.0; n * n];
            for i in 0..n {
                covariance[i * n + i] = alpha[i];
            }
            for a in 0..k {
                for b in 0..k {
                    let g = gamma[a * k + b];
                    if g == 0.0 {
                        continue;
                    }
                    for i in 0..n {
                        let left = g * beta[a][i];
                        for j in 0..n {
                            covariance[i * n + j] += left * beta[b][j];
                        }
                    }
                }
            }
            let l = cholesky(&covariance, n)?;
            let log_det = 2.0 * (0..n).map(|i| l[i * n + i].ln()).sum::<f64>();
            (Factor::Dense(l), log_det)
        } else {
            let sqrt_alpha: Vec<f64> = alpha.iter().map(|a| a.sqrt()).collect();
            let scaled: Vec<Vec<f64>> = beta
                .iter()
                .map(|b| b.iter().zip(&sqrt_alpha).map(|(b, s)| b / s).collect())
                .collect();
            let (q, r_tilde) = thin_qr(scaled, n);
            // I + R̃ γ R̃ᵀ
            let mut inner = matmul(&matmul(&r_tilde, &gamma, k), &transpose(&r_tilde, k), k);
            for i in 0..k {
                inner[i * k + i] += 1.0;
            }
            let l = cholesky(&inner, k)?;
            let log_det = alpha.iter().map(|a| a.ln()).sum::<f64>()
                + 2.0 * (0..k).map(|i| l[i * k + i].ln()).sum::<f64>();
            (Factor::LowRank { sqrt_alpha, q, l }, log_det)
        };
        if !log_det_covariance.is_finite() || mean.iter().any(|m| !m.is_finite()) {
            return None;
        }
        Some(Self {
            mean,
            factor,
            log_det_covariance,
        })
    }

    /// A draw and its log density under the approximation.
    fn draw(&self, rng: &mut KeanuRng) -> (Vec<f64>, f64) {
        let n = self.mean.len();
        let u: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
        let mut z = self.mean.clone();
        match &self.factor {
            Factor::Dense(l) => {
                for (i, zi) in z.iter_mut().enumerate() {
                    *zi += (0..=i).map(|j| l[i * n + j] * u[j]).sum::<f64>();
                }
            }
            Factor::LowRank { sqrt_alpha, q, l } => {
                let k = q.len();
                // u + Q (L − I) Qᵀ u
                let u1: Vec<f64> = q.iter().map(|column| dot(column, &u)).collect();
                let lu1: Vec<f64> = (0..k)
                    .map(|i| (0..=i).map(|j| l[i * k + j] * u1[j]).sum::<f64>() - u1[i])
                    .collect();
                let mut w = u.clone();
                for (column, c) in q.iter().zip(&lu1) {
                    for (wi, qi) in w.iter_mut().zip(column) {
                        *wi += c * qi;
                    }
                }
                for ((zi, wi), s) in z.iter_mut().zip(&w).zip(sqrt_alpha) {
                    *zi += s * wi;
                }
            }
        }
        let log_q = -0.5 * (self.log_det_covariance + dot(&u, &u) + n as f64 * LN_2PI);
        (z, log_q)
    }

    /// The covariance, n × n row-major.
    fn covariance(&self) -> Vec<f64> {
        let n = self.mean.len();
        match &self.factor {
            Factor::Dense(l) => {
                let mut covariance = vec![0.0; n * n];
                for i in 0..n {
                    for j in 0..n {
                        covariance[i * n + j] =
                            (0..=i.min(j)).map(|k| l[i * n + k] * l[j * n + k]).sum();
                    }
                }
                covariance
            }
            Factor::LowRank { sqrt_alpha, q, l } => {
                let k = q.len();
                // L Lᵀ − I
                let mut inner = matmul(l, &transpose(l, k), k);
                for i in 0..k {
                    inner[i * k + i] -= 1.0;
                }
                let mut covariance = vec![0.0; n * n];
                for i in 0..n {
                    for j in 0..n {
                        let mut value = if i == j { 1.0 } else { 0.0 };
                        for a in 0..k {
                            for b in 0..k {
                                value += q[a][i] * inner[a * k + b] * q[b][j];
                            }
                        }
                        covariance[i * n + j] = sqrt_alpha[i] * value * sqrt_alpha[j];
                    }
                }
                covariance
            }
        }
    }
}

/// The product of two k × k row-major matrices.
fn matmul(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    let mut out = vec![0.0; k * k];
    for i in 0..k {
        for l in 0..k {
            let a_il = a[i * k + l];
            if a_il == 0.0 {
                continue;
            }
            for j in 0..k {
                out[i * k + j] += a_il * b[l * k + j];
            }
        }
    }
    out
}

/// The transpose of a k × k row-major matrix.
fn transpose(a: &[f64], k: usize) -> Vec<f64> {
    let mut out = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            out[j * k + i] = a[i * k + j];
        }
    }
    out
}

/// The inverse of an m × m upper triangular matrix, `None` if a diagonal element is not positive.
fn invert_upper(r: &[f64], m: usize) -> Option<Vec<f64>> {
    let mut inverse = vec![0.0; m * m];
    for j in (0..m).rev() {
        let diagonal = r[j * m + j];
        if !(diagonal > 0.0 && diagonal.is_finite()) {
            return None;
        }
        inverse[j * m + j] = 1.0 / diagonal;
        for i in (0..j).rev() {
            let sum: f64 = (i + 1..=j).map(|l| r[i * m + l] * inverse[l * m + j]).sum();
            inverse[i * m + j] = -sum / r[i * m + i];
        }
    }
    Some(inverse)
}

/// The lower Cholesky factor of a symmetric n × n row-major matrix, `None` if it is not positive definite.
fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    DoubleTensor::new(&[n, n], a.to_vec())
        .and_then(|m| m.cholesky())
        .ok()
        .map(|l| l.as_slice().to_vec())
        .filter(|l| (0..n).all(|i| l[i * n + i] > 0.0 && l[i * n + i].is_finite()))
}

/// The thin QR factorisation, by Householder reflections, of the n × k matrix (k < n) whose columns are `columns`:
/// Q's k orthonormal columns and R, k × k upper triangular (row-major).
fn thin_qr(mut columns: Vec<Vec<f64>>, n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let k = columns.len();
    let mut reflectors: Vec<Option<(Vec<f64>, f64)>> = Vec::with_capacity(k);
    for j in 0..k {
        let x = &columns[j][j..];
        let norm = dot(x, x).sqrt();
        let mut v = x.to_vec();
        let alpha = if x[0] > 0.0 { -norm } else { norm };
        v[0] -= alpha;
        let v_norm2 = dot(&v, &v);
        if norm == 0.0 || v_norm2 == 0.0 {
            reflectors.push(None);
            continue;
        }
        for column in columns.iter_mut().skip(j) {
            let projection = 2.0 * dot(&v, &column[j..]) / v_norm2;
            for (c, vi) in column[j..].iter_mut().zip(&v) {
                *c -= projection * vi;
            }
        }
        reflectors.push(Some((v, v_norm2)));
    }
    let mut r = vec![0.0; k * k];
    for (j, column) in columns.iter().enumerate() {
        for i in 0..=j {
            r[i * k + j] = column[i];
        }
    }
    let mut q: Vec<Vec<f64>> = (0..k)
        .map(|j| {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            e
        })
        .collect();
    for (j, reflector) in reflectors.iter().enumerate().rev() {
        if let Some((v, v_norm2)) = reflector {
            for column in &mut q {
                let projection = 2.0 * dot(v, &column[j..]) / v_norm2;
                for (c, vi) in column[j..].iter_mut().zip(v) {
                    *c -= projection * vi;
                }
            }
        }
    }
    (q, r)
}

/// One Pathfinder path: its best Gaussian approximation and what the path recorded.
#[derive(Debug, Clone)]
pub struct PathfinderPath {
    /// The L-BFGS iterate the best approximation was built at, where the log density was evaluated.
    iterate: Vec<f64>,
    approximation: Approximation,
    elbo: f64,
    elbo_trace: Vec<f64>,
    best_iteration: usize,
    iterations: usize,
    converged: bool,
    evaluations: usize,
}

impl PathfinderPath {
    /// The mean of the path's best approximation, in unconstrained space (laid out by
    /// [`PathfinderResult::parameterisation`]).
    pub fn mean(&self) -> &[f64] {
        &self.approximation.mean
    }

    /// The covariance of the path's best approximation in unconstrained space, as an n × n matrix.
    pub fn covariance(&self) -> DoubleTensor {
        let n = self.approximation.mean.len();
        DoubleTensor::new(&[n, n], self.approximation.covariance())
            .expect("an n × n matrix has n² elements")
    }

    /// The best approximation's ELBO estimate.
    pub fn elbo(&self) -> f64 {
        self.elbo
    }

    /// The ELBO estimate of the approximation at each L-BFGS iterate after the start (−∞ where it isn't
    /// positive definite), or at the start alone if L-BFGS took no step.
    pub fn elbo_trace(&self) -> &[f64] {
        &self.elbo_trace
    }

    /// The L-BFGS iterate (0 for the start) whose approximation was chosen.
    pub fn best_iteration(&self) -> usize {
        self.best_iteration
    }

    /// The L-BFGS iterations the path took.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Whether L-BFGS converged (rather than stopping at its iteration limit or failing to find a step).
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// The log density evaluations the path used: L-BFGS's (each with its gradient) and the ELBO estimates'.
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// One draw from the path's best approximation, in unconstrained space, with its log density under the
    /// approximation.
    pub fn draw(&self, rng: &mut KeanuRng) -> (Vec<f64>, f64) {
        self.approximation.draw(rng)
    }
}

/// The result of Pathfinder: approximate posterior draws in unconstrained space (laid out by
/// [`parameterisation`](Self::parameterisation)), each with its log density and the log density of the
/// approximation it came from, and the paths that produced them.
#[derive(Debug)]
pub struct PathfinderResult<'m> {
    param: Parameterisation<'m>,
    paths: Vec<PathfinderPath>,
    failed_paths: usize,
    draws: Vec<Vec<f64>>,
    log_density: Vec<f64>,
    approximation_log_density: Vec<f64>,
    pareto_k: f64,
    /// The draws k̂ was estimated from (before resampling).
    weighted_draws: usize,
    evaluations: usize,
}

impl<'m> PathfinderResult<'m> {
    /// The parameterisation the draws live in: [`Parameterisation::range`] gives a latent's coordinates in each draw.
    pub fn parameterisation(&self) -> &Parameterisation<'m> {
        &self.param
    }

    /// The paths that succeeded, in the order they ran (one for [`Pathfinder::single_path`]).
    pub fn paths(&self) -> &[PathfinderPath] {
        &self.paths
    }

    /// The paths of [`Pathfinder::multi_path`] that failed and were skipped.
    pub fn failed_paths(&self) -> usize {
        self.failed_paths
    }

    /// The draws, in unconstrained space. Map one to the latents' values with
    /// [`Parameterisation::to_constrained`] or [`Parameterisation::to_state`].
    pub fn draws(&self) -> &[Vec<f64>] {
        &self.draws
    }

    /// The log density of each draw: the model's log probability plus the log-Jacobian of the transforms (−∞ for
    /// an impossible draw).
    pub fn log_density(&self) -> &[f64] {
        &self.log_density
    }

    /// The log density of each draw under the Gaussian approximation it was drawn from.
    pub fn approximation_log_density(&self) -> &[f64] {
        &self.approximation_log_density
    }

    /// The Pareto k̂ of the importance ratios (log density minus approximation log density) of the draws before
    /// resampling ([`psis`]): the lower the better, and above [`khat_threshold`] (0.7 for many draws) the
    /// approximation is poor. −∞ when the ratios are equal to within 10⁻⁸ (the approximation is exact, as for a
    /// Gaussian posterior in one dimension), and NaN if every draw of a single path is impossible.
    pub fn pareto_k(&self) -> f64 {
        self.pareto_k
    }

    /// Whether [`pareto_k`](Self::pareto_k) is at most the [`khat_threshold`] for the draws (before resampling).
    pub fn is_reliable(&self) -> bool {
        self.pareto_k <= khat_threshold(self.weighted_draws)
    }

    /// The log density evaluations every path used, with the random starting points tried.
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// The mean of the draws, in unconstrained space.
    pub fn mean(&self) -> Vec<f64> {
        let n = self.param.dimension();
        let mut mean = vec![0.0; n];
        for draw in &self.draws {
            for (m, z) in mean.iter_mut().zip(draw) {
                *m += z;
            }
        }
        let count = self.draws.len().max(1) as f64;
        mean.iter_mut().for_each(|m| *m /= count);
        mean
    }

    /// The draws as samples of `nodes`: each draw is mapped to the latents' constrained values
    /// ([`Parameterisation::to_state`]) and their descendants are recomputed. The log probability recorded with each
    /// sample is the model's at the draw (without the Jacobian), as the samplers record it. Draws where the model
    /// can't be evaluated (a failed ODE solve, where [`log_density`](Self::log_density) is −∞), which have no values
    /// to record, are left out, so there can be fewer samples than draws. `state` must be the state Pathfinder ran
    /// with (or one with values for the same nodes); it is left at the last draw recorded.
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidArgument`] if `nodes` is empty, [`keanu_graph::GraphError::UnknownNode`] if a node
    /// is not in the model's graph, and any error evaluating the model.
    pub fn sample(&self, state: &mut State, nodes: &[NodeId]) -> Result<NetworkSamples> {
        let mut draws = self.draws.iter();
        sample_approximation(
            &self.param,
            || draws.next().cloned(),
            state,
            nodes,
            self.draws.len(),
            Unevaluable::Skip,
        )
    }

    /// Starting states for `count` MCMC chains: copies of `state` with the latents set to `count` draws chosen at
    /// random without replacement (preferring draws with distinct values, which resampled draws may repeat), so
    /// that each chain starts from its own point near the typical set. Pass them to
    /// [`run_chains_from`](crate::mcmc::chains::run_chains_from) or
    /// [`sample_chains_from`](crate::mcmc::chains::sample_chains_from).
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidArgument`] if `count` is 0 or more than the number of draws, and any error setting
    /// the latents.
    pub fn initial_states(
        &self,
        state: &State,
        count: usize,
        rng: &mut KeanuRng,
    ) -> Result<Vec<State>> {
        if count == 0 || count > self.draws.len() {
            return Err(AlgorithmError::InvalidArgument(format!(
                "cannot choose {count} starting points from {} draws",
                self.draws.len()
            )));
        }
        // a random order of the draws, the first `count` distinct ones chosen, topped up with repeats
        let mut order: Vec<usize> = (0..self.draws.len()).collect();
        for i in (1..order.len()).rev() {
            let j = rng.next_below(i as u64 + 1) as usize;
            order.swap(i, j);
        }
        let mut chosen: Vec<usize> = Vec::with_capacity(count);
        for &i in &order {
            if chosen.len() == count {
                break;
            }
            let possible = self.log_density[i].is_finite();
            if possible && !chosen.iter().any(|&c| self.draws[c] == self.draws[i]) {
                chosen.push(i);
            }
        }
        for &i in &order {
            if chosen.len() == count {
                break;
            }
            if !chosen.contains(&i) {
                chosen.push(i);
            }
        }
        chosen
            .iter()
            .map(|&i| {
                let mut start = state.clone();
                self.param.to_state(&self.draws[i], &mut start)?;
                Ok(start)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs_of(list: &[(Vec<f64>, Vec<f64>)]) -> VecDeque<(Vec<f64>, Vec<f64>)> {
        list.iter().cloned().collect()
    }

    /// The compact form equals the BFGS inverse Hessian built by the recursive update from diag(α).
    #[test]
    fn the_compact_inverse_hessian_is_the_bfgs_update() {
        let alpha = [0.7, 1.3, 0.4, 2.0, 1.1];
        let pairs = [
            (
                vec![0.3, -0.1, 0.2, 0.05, -0.4],
                vec![0.5, -0.05, 0.6, 0.02, -0.3],
            ),
            (
                vec![-0.2, 0.4, 0.1, -0.3, 0.1],
                vec![-0.1, 0.7, 0.3, -0.2, 0.05],
            ),
        ];
        // BFGS: H ← (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ
        let n = alpha.len();
        let mut h = vec![0.0; n * n];
        for i in 0..n {
            h[i * n + i] = alpha[i];
        }
        for (s, y) in &pairs {
            let rho = 1.0 / dot(s, y);
            let mut left = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    left[i * n + j] = f64::from(u8::from(i == j)) - rho * s[i] * y[j];
                }
            }
            let right = transpose(&left, n);
            let mut next = matmul(&matmul(&left, &h, n), &right, n);
            for i in 0..n {
                for j in 0..n {
                    next[i * n + j] += rho * s[i] * s[j];
                }
            }
            h = next;
        }
        let at = Iterate {
            point: vec![0.0; n],
            value: 0.0,
            gradient: vec![1.0, -2.0, 0.5, 0.3, -0.7],
        };
        // 2m = 4 < 5: the low-rank form
        let approximation = Approximation::new(&at, &alpha, &pairs_of(&pairs)).unwrap();
        assert!(matches!(approximation.factor, Factor::LowRank { .. }));
        let covariance = approximation.covariance();
        for (a, b) in covariance.iter().zip(&h) {
            assert!((a - b).abs() < 1e-10, "{covariance:?} vs {h:?}");
        }
        // the mean is the Newton step x − H ∇f, with ∇f the stored gradient of the negative log density
        for i in 0..n {
            let step: f64 = (0..n).map(|j| h[i * n + j] * at.gradient[j]).sum();
            assert!((approximation.mean[i] + step).abs() < 1e-10);
        }
        // the log determinant
        let l = cholesky(&h, n).unwrap();
        let log_det = 2.0 * (0..n).map(|i| l[i * n + i].ln()).sum::<f64>();
        assert!((approximation.log_det_covariance - log_det).abs() < 1e-10);
        // the dense form (2m ≥ n) gives the same covariance on the first three coordinates' problem
        let small: Vec<(Vec<f64>, Vec<f64>)> = pairs
            .iter()
            .map(|(s, y)| (s[..3].to_vec(), y[..3].to_vec()))
            .collect();
        let dense = Approximation::new(
            &Iterate {
                point: vec![0.0; 3],
                value: 0.0,
                gradient: vec![1.0, -2.0, 0.5],
            },
            &alpha[..3],
            &pairs_of(&small),
        )
        .unwrap();
        assert!(matches!(dense.factor, Factor::Dense(_)));
    }

    /// Draws from the low-rank form have the approximation's covariance, and their log densities are the
    /// Gaussian's.
    #[test]
    fn low_rank_draws_have_the_approximation_covariance() {
        let alpha = [0.7, 1.3, 0.4, 2.0, 1.1];
        let pairs = [(
            vec![0.3, -0.1, 0.2, 0.05, -0.4],
            vec![0.5, -0.05, 0.6, 0.02, -0.3],
        )];
        let at = Iterate {
            point: vec![1.0; 5],
            value: 0.0,
            gradient: vec![0.0; 5],
        };
        let approximation = Approximation::new(&at, &alpha, &pairs_of(&pairs)).unwrap();
        let covariance = approximation.covariance();
        let mut rng = KeanuRng::seed_from_u64(2);
        let count = 50_000;
        let mut sums = [0.0; 25];
        for _ in 0..count {
            let (z, log_q) = approximation.draw(&mut rng);
            for i in 0..5 {
                for j in 0..5 {
                    sums[i * 5 + j] += (z[i] - 1.0) * (z[j] - 1.0);
                }
            }
            assert!(log_q.is_finite());
        }
        for (sum, c) in sums.iter().zip(&covariance) {
            assert!((sum / count as f64 - c).abs() < 0.05, "{sum} vs {c}");
        }
        // each draw's log density is the Gaussian's: −½ (log det Σ + (z − μ)ᵀ Σ⁻¹ (z − μ) + n log 2π)
        let precision = DoubleTensor::new(&[5, 5], cholesky(&covariance, 5).unwrap())
            .unwrap()
            .cholesky_inverse()
            .unwrap();
        let precision = precision.as_slice();
        for _ in 0..10 {
            let (z, log_q) = approximation.draw(&mut rng);
            let d: Vec<f64> = z.iter().map(|z| z - 1.0).collect();
            let quadratic: f64 = (0..25).map(|k| d[k / 5] * precision[k] * d[k % 5]).sum();
            let log_det = 2.0
                * (0..5)
                    .map(|i| cholesky(&covariance, 5).unwrap()[i * 5 + i].ln())
                    .sum::<f64>();
            let expected = -0.5 * (log_det + quadratic + 5.0 * LN_2PI);
            assert!((log_q - expected).abs() < 1e-9, "{log_q} vs {expected}");
        }
    }

    #[test]
    fn the_diagonal_update_recovers_a_diagonal_hessian() {
        // f = Σ hᵢ xᵢ² / 2: y = h ⊙ s, and one update along a coordinate gives αᵢ = 1 / hᵢ there
        let mut alpha = vec![1.0, 1.0];
        update_diagonal(&mut alpha, &[1.0, 0.0], &[4.0, 0.0]);
        assert!((alpha[0] - 0.25).abs() < 1e-12, "{alpha:?}");
        let mut alpha = vec![1.0];
        update_diagonal(&mut alpha, &[0.5], &[1.5]);
        assert!((alpha[0] - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn thin_qr_factorises() {
        let columns = vec![
            vec![1.0, 2.0, 0.0, -1.0],
            vec![0.5, -1.0, 3.0, 2.0],
            vec![1.0, 2.0, 0.0, -1.0],
        ];
        let (q, r) = thin_qr(columns.clone(), 4);
        for (j, column) in columns.iter().enumerate() {
            for i in 0..4 {
                let value: f64 = (0..3).map(|a| q[a][i] * r[a * 3 + j]).sum();
                assert!((value - column[i]).abs() < 1e-12);
            }
        }
        for a in 0..3 {
            for b in 0..3 {
                let expected = f64::from(u8::from(a == b));
                assert!((dot(&q[a], &q[b]) - expected).abs() < 1e-12);
            }
        }
    }
}
