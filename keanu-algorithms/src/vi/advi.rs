//! Automatic differentiation variational inference (ADVI), after Kucukelbir, Tran, Ranganath, Gelman and Blei,
//! "Automatic Differentiation Variational Inference", JMLR 18 (2017), and Stan's implementation of it.

use crate::error::{AlgorithmError, Result};
use crate::samples::NetworkSamples;
use crate::support::{check_latents, evaluate_all, sample_approximation, start_point};
use crate::vi::minibatch::{Batches, Minibatch};
use keanu_distributions::KeanuRng;
use keanu_graph::{Model, NodeId, Parameterisation, State};
use keanu_tensor::DoubleTensor;
use std::collections::VecDeque;

/// The step sizes the adaptation tries, largest first, as Stan does.
const ETA_GRID: [f64; 5] = [100.0, 10.0, 1.0, 0.1, 0.01];
/// τ in the step size sequence: keeps the first steps finite when the gradient is tiny.
const TAU: f64 = 1.0;
/// The weight of the newest squared gradient in the step size's running average.
const NEW_WEIGHT: f64 = 0.1;

/// The Gaussian family ADVI fits, in unconstrained space.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VariationalFamily {
    /// Independent coordinates: N(μ, diag(exp(2ω))), with 2d parameters for d coordinates. The default, as in Stan.
    /// It cannot represent correlations, and underestimates the marginal variance of correlated latents.
    #[default]
    MeanField,
    /// A full covariance: N(μ, LLᵀ) with L lower triangular, with d + d(d + 1)/2 parameters. It captures linear
    /// correlations, at a cost quadratic in d per iteration.
    ///
    /// As in Stan, L's entries (its diagonal included) are optimized directly rather than through a log-diagonal
    /// Cholesky parameterisation: the entropy term log |L_ii| pushes each diagonal entry away from zero, so its sign
    /// never changes in practice, and LLᵀ is the covariance either way.
    FullRank,
}

/// ADVI configuration. The defaults are Stan's. The configuration holds no state, so it can be reused for any
/// number of fits.
///
/// Each iteration draws [`grad_samples`](Self::grad_samples) standard normal vectors ε, maps each to a point
/// ζ = μ + Lε of the approximation (the reparameterisation trick) and averages the gradient of the log density of
/// the unconstrained point there (the model's log probability plus the log-Jacobian of the transforms) to estimate
/// the ELBO's gradient; the entropy of the Gaussian, and its gradient, are exact. The parameters then take a step of
/// η·k^(−1/2) / (τ + √s_k) along the gradient, elementwise, where k is the iteration, τ = 1 and s_k is a running
/// average of the squared gradient: s_1 = g₁², s_k = 0.1·g_k² + 0.9·s_{k−1}. (The paper's exponent is −1/2 + ε
/// with ε = 10⁻¹⁶, which is −1/2 in floating point.)
///
/// With [`adapt`](Self::adapt) (the default), η is chosen first: starting from the initial approximation each
/// time, [`adapt_iterations`](Self::adapt_iterations) iterations are run with each η of 100, 10, 1, 0.1 and 0.01 in
/// turn, and the search stops at the first η whose ELBO is worse than the previous one's, taking the previous one,
/// provided that improved on the initial ELBO; if none is chosen that way, 0.01 is taken if it improved on the
/// initial ELBO, as in Stan. Where Stan then gives up, this takes the η with the highest ELBO that did not diverge
/// (the initial approximation can be close enough that the noisy ELBO estimates never beat it), and fails only if
/// every η diverged.
///
/// The default tolerance is loose, as in Stan: on a model whose ELBO is around −30 a relative change of 0.01 is 0.3
/// nats, which a fit can reach while still far from the optimum. Lower [`tol_rel_obj`](Self::tol_rel_obj) (to 0.001,
/// say) when the estimates matter, and raise [`grad_samples`](Self::grad_samples) for less noisy final iterates.
///
/// Every [`eval_elbo`](Self::eval_elbo) iterations the ELBO is estimated from [`elbo_samples`](Self::elbo_samples)
/// draws, and its relative change since the last estimate is kept in a buffer of the last
/// max(⌊0.1·`max_iterations` / `eval_elbo`⌋, 2) changes. The fit has converged when the mean or the median of the
/// buffer is below [`tol_rel_obj`](Self::tol_rel_obj), and stops anyway after
/// [`max_iterations`](Self::max_iterations).
///
/// ADVI handles continuous latents only: a model with discrete latents is rejected, so observe them or sum them out
/// (for instance with a mixture) first.
///
/// # Minibatches
///
/// With [`with_minibatch`](Self::with_minibatch), each iteration observes a fresh random batch of B of the N data
/// rows (see [`Minibatch`]) before estimating the gradient, and the log probability weighs the batched nodes' terms
/// by N / B (see [`Parameterisation::set_weight`]), so the gradient is an unbiased estimate of the full data's at
/// the cost of B rows. Each ELBO estimate is made on a fresh batch too, so the estimates are noisier: convergence is
/// judged on their running mean, the mean of the last max(⌊0.1·`max_iterations` / `eval_elbo`⌋, 2) estimates (the
/// change buffer's length), whose relative change goes in the buffer instead of the latest estimate's, with the same
/// mean-or-median rule. The adaptation runs on minibatches as well, each of its iterations on a fresh batch, but it
/// estimates the initial ELBO and every candidate η's on one fixed batch (the first), so that the comparison
/// between step sizes is not swamped by the differences between batches. The [ELBO trace](AdviResult::elbo_trace)
/// holds the noisy estimates; they seldom meet a tight tolerance, so bound the fit with
/// [`max_iterations`](Self::max_iterations). After the fit, the batched nodes get back the values they had before
/// it.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Advi {
    /// The Gaussian family to fit (default mean field).
    pub family: VariationalFamily,
    /// The draws averaged for each gradient estimate (default 1).
    pub grad_samples: usize,
    /// The draws averaged for each ELBO estimate (default 100).
    pub elbo_samples: usize,
    /// The iterations between ELBO estimates and convergence checks (default 100).
    pub eval_elbo: usize,
    /// The relative ELBO change below which the fit has converged (default 0.01).
    pub tol_rel_obj: f64,
    /// The most iterations to run, adaptation excluded (default 10 000).
    pub max_iterations: usize,
    /// Whether to choose the step size scale η by the grid search (default true).
    pub adapt: bool,
    /// The iterations run for each η tried by the adaptation (default 50).
    pub adapt_iterations: usize,
    /// The step size scale η used without adaptation (default 1).
    pub eta: f64,
    /// The data to subsample each iteration (default none: every iteration uses all of it). See the
    /// [type documentation](Self#minibatches).
    pub minibatch: Option<Minibatch>,
}

/// Setters for each setting, for use as `Advi::default().with_...(...)`.
impl Advi {
    /// Sets [`family`](Self::family).
    #[must_use]
    pub fn with_family(mut self, family: VariationalFamily) -> Self {
        self.family = family;
        self
    }

    /// Sets [`grad_samples`](Self::grad_samples).
    #[must_use]
    pub fn with_grad_samples(mut self, grad_samples: usize) -> Self {
        self.grad_samples = grad_samples;
        self
    }

    /// Sets [`elbo_samples`](Self::elbo_samples).
    #[must_use]
    pub fn with_elbo_samples(mut self, elbo_samples: usize) -> Self {
        self.elbo_samples = elbo_samples;
        self
    }

    /// Sets [`eval_elbo`](Self::eval_elbo).
    #[must_use]
    pub fn with_eval_elbo(mut self, eval_elbo: usize) -> Self {
        self.eval_elbo = eval_elbo;
        self
    }

    /// Sets [`tol_rel_obj`](Self::tol_rel_obj).
    #[must_use]
    pub fn with_tol_rel_obj(mut self, tol_rel_obj: f64) -> Self {
        self.tol_rel_obj = tol_rel_obj;
        self
    }

    /// Sets [`max_iterations`](Self::max_iterations).
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets [`adapt`](Self::adapt).
    #[must_use]
    pub fn with_adapt(mut self, adapt: bool) -> Self {
        self.adapt = adapt;
        self
    }

    /// Sets [`adapt_iterations`](Self::adapt_iterations).
    #[must_use]
    pub fn with_adapt_iterations(mut self, adapt_iterations: usize) -> Self {
        self.adapt_iterations = adapt_iterations;
        self
    }

    /// Sets [`eta`](Self::eta).
    #[must_use]
    pub fn with_eta(mut self, eta: f64) -> Self {
        self.eta = eta;
        self
    }

    /// Sets [`minibatch`](Self::minibatch): each iteration observes a random batch of the data, weighted up to the
    /// full data. A [`MinibatchGroup`](super::MinibatchGroup) converts into a [`Minibatch`].
    #[must_use]
    pub fn with_minibatch(mut self, minibatch: impl Into<Minibatch>) -> Self {
        self.minibatch = Some(minibatch.into());
        self
    }
}

impl Default for Advi {
    fn default() -> Self {
        Self {
            family: VariationalFamily::MeanField,
            grad_samples: 1,
            elbo_samples: 100,
            eval_elbo: 100,
            tol_rel_obj: 0.01,
            max_iterations: 10_000,
            adapt: true,
            adapt_iterations: 50,
            eta: 1.0,
            minibatch: None,
        }
    }
}

impl Advi {
    /// The default settings: mean field, with Stan's defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// The default settings with the mean-field family.
    pub fn mean_field() -> Self {
        Self::default()
    }

    /// The default settings with the full-rank family.
    pub fn full_rank() -> Self {
        Self::default().with_family(VariationalFamily::FullRank)
    }

    fn validate(&self) -> Result<()> {
        let invalid = |m: &str| Err(AlgorithmError::InvalidSetting(m.into()));
        if self.grad_samples == 0 {
            return invalid("the number of gradient samples must be greater than 0");
        }
        if self.elbo_samples == 0 {
            return invalid("the number of ELBO samples must be greater than 0");
        }
        if self.eval_elbo == 0 {
            return invalid("the ELBO evaluation interval must be greater than 0");
        }
        if !(self.tol_rel_obj.is_finite() && self.tol_rel_obj > 0.0) {
            return invalid("the relative ELBO tolerance must be finite and greater than 0");
        }
        if self.max_iterations == 0 {
            return invalid("the maximum number of iterations must be greater than 0");
        }
        if self.adapt && self.adapt_iterations == 0 {
            return invalid("the number of adaptation iterations must be greater than 0");
        }
        if !self.adapt && !(self.eta.is_finite() && self.eta > 0.0) {
            return invalid("the step size scale eta must be finite and greater than 0");
        }
        Ok(())
    }

    /// Fits the approximation to the posterior of `model`'s latents, starting with its mean at their current values
    /// in `state` (latents without a value are sampled from their priors first) and unit scale. Leaves `state` at
    /// the constrained values of the fitted mean (with a [minibatch](Self#minibatches), and the batched nodes at
    /// their values before the fit; after an error they can hold a batch).
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidSetting`] for an out-of-range setting; [`AlgorithmError::NoLatents`] if the model
    /// has no continuous latents to fit; [`AlgorithmError::DiscreteLatents`] if it has discrete latents (see
    /// above); [`AlgorithmError::ImpossibleStart`] if a latent's value is outside its support or on its boundary;
    /// an error from [`Parameterisation::new`] if a latent's support has no transform; and
    /// [`AlgorithmError::NonFiniteElbo`] if the ELBO of the initial approximation is not finite (a draw has zero
    /// probability or a NaN log probability), if every step size the adaptation tries diverges, or if a draw's log
    /// probability or gradient stops being finite during the fit. With a minibatch, any error from
    /// [`Minibatch::batches`] for an invalid one.
    pub fn fit<'m>(
        &self,
        model: &'m Model,
        state: &mut State,
        rng: &mut KeanuRng,
    ) -> Result<AdviResult<'m>> {
        self.validate()?;
        check_latents(model)?;
        evaluate_all(model.graph(), state, rng)?;
        let mut batches = match &self.minibatch {
            Some(minibatch) => {
                let mut batches = minibatch.batches(model, state)?;
                batches.next(state, rng)?;
                Some(batches)
            }
            None => None,
        };
        let mut param = Parameterisation::new(model, state)?;
        if param.dimension() == 0 {
            return Err(AlgorithmError::NoLatents);
        }
        if let Some(batches) = &batches {
            for &(node, weight) in batches.weights() {
                param.set_weight(node, weight)?;
            }
        }
        // the batch the initial ELBO and the adaptation's ELBOs are estimated on
        let fixed_rows = batches.as_ref().map(Batches::current_rows);
        let start = start_point(&param, state)?;
        let initial = Gaussian::initial(self.family, &start);
        let mut target = Target {
            param: &param,
            state,
        };

        let initial_elbo = target.elbo(&initial, self.elbo_samples, rng)?;
        if !initial_elbo.is_finite() {
            return Err(AlgorithmError::NonFiniteElbo(
                "the ELBO of the initial approximation is not finite; the model may be ill-conditioned or \
                 misspecified, or the starting values poor"
                    .into(),
            ));
        }
        let eta = if self.adapt {
            self.adapt_eta(
                &mut target,
                &mut batches,
                fixed_rows.as_deref(),
                &initial,
                initial_elbo,
                rng,
            )?
        } else {
            self.eta
        };

        let mut q = initial;
        let mut ascent = Ascent::new(q.params.len());
        let mut elbo_trace = vec![initial_elbo];
        let window = ((0.1 * self.max_iterations as f64 / self.eval_elbo as f64) as usize).max(2);
        let mut changes: VecDeque<f64> = VecDeque::with_capacity(window);
        // with minibatches, the last `window` ELBO estimates, whose mean is compared
        let mut recent: VecDeque<f64> = VecDeque::with_capacity(window);
        let mut elbo = 0.0;
        let mut converged = false;
        let mut iterations = 0;
        while iterations < self.max_iterations && !converged {
            iterations += 1;
            if let Some(batches) = &mut batches {
                batches.next(target.state, rng)?;
            }
            let gradient = target
                .elbo_gradient(&q, self.grad_samples, rng)?
                .ok_or_else(|| {
                    AlgorithmError::NonFiniteElbo(format!(
                        "a draw's log probability or gradient is not finite at iteration {iterations}"
                    ))
                })?;
            ascent.step(&mut q, &gradient, eta);
            if iterations % self.eval_elbo == 0 {
                let previous = elbo;
                if let Some(batches) = &mut batches {
                    batches.next(target.state, rng)?;
                }
                let estimate = target.elbo(&q, self.elbo_samples, rng)?;
                if !estimate.is_finite() {
                    return Err(AlgorithmError::NonFiniteElbo(format!(
                        "the ELBO is not finite at iteration {iterations}"
                    )));
                }
                elbo_trace.push(estimate);
                elbo = if batches.is_some() {
                    if recent.len() == window {
                        recent.pop_front();
                    }
                    recent.push_back(estimate);
                    recent.iter().sum::<f64>() / recent.len() as f64
                } else {
                    estimate
                };
                // Stan's relative difference, with the first estimate compared against 0 (a change of 1)
                if changes.len() == window {
                    changes.pop_front();
                }
                changes.push_back(((previous - elbo) / elbo).abs());
                let mean = changes.iter().sum::<f64>() / changes.len() as f64;
                converged = mean < self.tol_rel_obj || median(&changes) < self.tol_rel_obj;
            }
        }
        if let Some(batches) = &batches {
            batches.restore(state)?;
        }
        param.to_state(&q.params[..q.dim], state)?;
        Ok(AdviResult {
            param,
            q,
            elbo_trace,
            eta,
            iterations,
            converged,
        })
    }

    /// Stan's grid search for η: see the [type documentation](Self).
    fn adapt_eta(
        &self,
        target: &mut Target<'_, '_>,
        batches: &mut Option<Batches<'_>>,
        fixed_rows: Option<&[Vec<usize>]>,
        initial: &Gaussian,
        initial_elbo: f64,
        rng: &mut KeanuRng,
    ) -> Result<f64> {
        let mut best = (f64::NAN, f64::MIN);
        let mut highest = (f64::NAN, f64::MIN);
        for (index, &eta) in ETA_GRID.iter().enumerate() {
            let mut q = initial.clone();
            let mut ascent = Ascent::new(q.params.len());
            for _ in 0..self.adapt_iterations {
                if let Some(batches) = batches {
                    batches.next(target.state, rng)?;
                }
                // a diverging η gives non-finite gradients, or values the model cannot evaluate: zero the gradient
                // and let the ELBO below reject this η (the initial ELBO has already shown the model evaluates)
                let gradient = match target.elbo_gradient(&q, self.grad_samples, rng) {
                    Ok(Some(gradient)) => gradient,
                    _ => vec![0.0; q.params.len()],
                };
                ascent.step(&mut q, &gradient, eta);
            }
            if let (Some(batches), Some(rows)) = (batches.as_ref(), fixed_rows) {
                batches.observe_rows(target.state, rows)?;
            }
            let elbo = match target.elbo(&q, self.elbo_samples, rng) {
                Ok(elbo) if elbo.is_finite() => elbo,
                _ => f64::MIN,
            };
            if elbo > highest.1 {
                highest = (eta, elbo);
            }
            if elbo < best.1 && best.1 > initial_elbo {
                return Ok(best.0);
            }
            if index + 1 < ETA_GRID.len() {
                best = (eta, elbo);
            } else if elbo > initial_elbo {
                return Ok(eta);
            }
        }
        // Stan fails here; the initial approximation can already be close enough that 50 iterations do not beat
        // its noisy ELBO estimate, so take the best η that did not diverge
        if highest.1 > f64::MIN {
            return Ok(highest.0);
        }
        Err(AlgorithmError::NonFiniteElbo(
            "every step size the adaptation tried diverged; the model may be ill-conditioned or misspecified"
                .into(),
        ))
    }
}

fn median(values: &VecDeque<f64>) -> f64 {
    let mut sorted: Vec<f64> = values.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// The index of L_ij (j ≤ i) in the row-major packed lower triangle.
fn tri(i: usize, j: usize) -> usize {
    i * (i + 1) / 2 + j
}

/// A Gaussian over the unconstrained space: its flat parameters are μ (d values) followed by ω (d values, mean
/// field) or the packed lower triangle of L (d(d + 1)/2 values, full rank, row by row).
#[derive(Debug, Clone, PartialEq)]
struct Gaussian {
    family: VariationalFamily,
    dim: usize,
    params: Vec<f64>,
}

impl Gaussian {
    /// Mean `mu` and the identity covariance, as Stan starts.
    fn initial(family: VariationalFamily, mu: &[f64]) -> Self {
        let dim = mu.len();
        let mut params = mu.to_vec();
        match family {
            VariationalFamily::MeanField => params.extend(std::iter::repeat_n(0.0, dim)),
            VariationalFamily::FullRank => {
                params.extend(std::iter::repeat_n(0.0, dim * (dim + 1) / 2));
                for i in 0..dim {
                    params[dim + tri(i, i)] = 1.0;
                }
            }
        }
        Self {
            family,
            dim,
            params,
        }
    }

    fn mean(&self) -> &[f64] {
        &self.params[..self.dim]
    }

    /// ζ = μ + Lε.
    fn transform(&self, eps: &[f64]) -> Vec<f64> {
        let d = self.dim;
        let scale = &self.params[d..];
        match self.family {
            VariationalFamily::MeanField => (0..d)
                .map(|i| self.params[i] + scale[i].exp() * eps[i])
                .collect(),
            VariationalFamily::FullRank => (0..d)
                .map(|i| self.params[i] + (0..=i).map(|j| scale[tri(i, j)] * eps[j]).sum::<f64>())
                .collect(),
        }
    }

    /// The (differential) entropy: d/2·(1 + log 2π) + log |det L|.
    fn entropy(&self) -> f64 {
        let d = self.dim;
        let scale = &self.params[d..];
        let log_det: f64 = match self.family {
            VariationalFamily::MeanField => scale.iter().sum(),
            VariationalFamily::FullRank => (0..d).map(|i| scale[tri(i, i)].abs().ln()).sum(),
        };
        0.5 * d as f64 * (1.0 + (2.0 * std::f64::consts::PI).ln()) + log_det
    }

    /// The lower triangular factor L, as a d × d row-major matrix.
    fn factor(&self) -> Vec<f64> {
        let d = self.dim;
        let scale = &self.params[d..];
        let mut l = vec![0.0; d * d];
        for i in 0..d {
            match self.family {
                VariationalFamily::MeanField => l[i * d + i] = scale[i].exp(),
                VariationalFamily::FullRank => {
                    for j in 0..=i {
                        l[i * d + j] = scale[tri(i, j)];
                    }
                }
            }
        }
        l
    }

    fn draw(&self, rng: &mut KeanuRng) -> (Vec<f64>, Vec<f64>) {
        let eps: Vec<f64> = (0..self.dim).map(|_| rng.next_gaussian()).collect();
        let zeta = self.transform(&eps);
        (eps, zeta)
    }
}

/// The log density of the unconstrained point, evaluated in a state.
struct Target<'p, 's> {
    param: &'p Parameterisation<'p>,
    state: &'s mut State,
}

impl Target<'_, '_> {
    /// The Monte Carlo ELBO: the mean log density (with the Jacobian) of `samples` draws plus the exact entropy.
    /// Not finite if any draw's log density is not.
    fn elbo(&mut self, q: &Gaussian, samples: usize, rng: &mut KeanuRng) -> Result<f64> {
        let mut sum = 0.0;
        for _ in 0..samples {
            let (_, zeta) = q.draw(rng);
            let log_prob = self.param.log_prob(&zeta, true, self.state)?;
            if !log_prob.is_finite() {
                return Ok(f64::NEG_INFINITY);
            }
            sum += log_prob;
        }
        Ok(sum / samples as f64 + q.entropy())
    }

    /// The reparameterisation gradient of the ELBO with respect to the flat parameters of `q`, averaged over
    /// `samples` draws, with the entropy's exact gradient added; `None` if a draw's log density or gradient is not
    /// finite.
    fn elbo_gradient(
        &mut self,
        q: &Gaussian,
        samples: usize,
        rng: &mut KeanuRng,
    ) -> Result<Option<Vec<f64>>> {
        let d = q.dim;
        let mut gradient = vec![0.0; q.params.len()];
        for _ in 0..samples {
            let (eps, zeta) = q.draw(rng);
            let (log_prob, g) = self.param.log_prob_and_gradient(&zeta, true, self.state)?;
            let g = match g {
                Some(g) if log_prob.is_finite() && g.iter().all(|x| x.is_finite()) => g,
                _ => return Ok(None),
            };
            let (mu_grad, scale_grad) = gradient.split_at_mut(d);
            for (m, gi) in mu_grad.iter_mut().zip(&g) {
                *m += gi;
            }
            match q.family {
                VariationalFamily::MeanField => {
                    for i in 0..d {
                        scale_grad[i] += g[i] * eps[i];
                    }
                }
                VariationalFamily::FullRank => {
                    for i in 0..d {
                        for j in 0..=i {
                            scale_grad[tri(i, j)] += g[i] * eps[j];
                        }
                    }
                }
            }
        }
        for x in &mut gradient {
            *x /= samples as f64;
        }
        let scale = &q.params[d..];
        let scale_grad = &mut gradient[d..];
        match q.family {
            // the chain rule through exp(ω), plus the entropy's gradient, 1 per coordinate
            VariationalFamily::MeanField => {
                for (g, w) in scale_grad.iter_mut().zip(scale) {
                    *g = *g * w.exp() + 1.0;
                }
            }
            // the entropy's gradient, 1 / L_ii on the diagonal
            VariationalFamily::FullRank => {
                for i in 0..d {
                    scale_grad[tri(i, i)] += 1.0 / scale[tri(i, i)];
                }
            }
        }
        if gradient.iter().all(|x| x.is_finite()) {
            Ok(Some(gradient))
        } else {
            Ok(None)
        }
    }
}

/// Stan's adaptive step size sequence: see the [`Advi`] documentation.
struct Ascent {
    squared: Vec<f64>,
    iteration: usize,
}

impl Ascent {
    fn new(len: usize) -> Self {
        Self {
            squared: vec![0.0; len],
            iteration: 0,
        }
    }

    fn step(&mut self, q: &mut Gaussian, gradient: &[f64], eta: f64) {
        self.iteration += 1;
        let first = self.iteration == 1;
        let eta_k = eta / (self.iteration as f64).sqrt();
        for ((p, s), g) in q.params.iter_mut().zip(&mut self.squared).zip(gradient) {
            *s = if first {
                g * g
            } else {
                NEW_WEIGHT * g * g + (1.0 - NEW_WEIGHT) * *s
            };
            *p += eta_k * g / (TAU + s.sqrt());
        }
    }
}

/// A fitted ADVI approximation: a Gaussian over the unconstrained space of a model's continuous latents (laid out
/// by [`parameterisation`](Self::parameterisation)), with what the fit recorded.
#[derive(Debug)]
pub struct AdviResult<'m> {
    param: Parameterisation<'m>,
    q: Gaussian,
    elbo_trace: Vec<f64>,
    eta: f64,
    iterations: usize,
    converged: bool,
}

impl AdviResult<'_> {
    /// The parameterisation the approximation lives in: [`Parameterisation::range`] gives a latent's coordinates in
    /// [`mean`](Self::mean) and [`covariance`](Self::covariance).
    pub fn parameterisation(&self) -> &Parameterisation<'_> {
        &self.param
    }

    /// The family fitted.
    pub fn family(&self) -> VariationalFamily {
        self.q.family
    }

    /// The approximation's mean in unconstrained space. Map it to the latents' values with
    /// [`Parameterisation::to_constrained`]; that is not their posterior mean when a transform is nonlinear.
    pub fn mean(&self) -> &[f64] {
        self.q.mean()
    }

    /// The approximation's covariance in unconstrained space, as a d × d matrix. Diagonal for the mean-field family.
    pub fn covariance(&self) -> DoubleTensor {
        let d = self.q.dim;
        let l = self.q.factor();
        let mut covariance = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                covariance[i * d + j] = (0..=i.min(j)).map(|k| l[i * d + k] * l[j * d + k]).sum();
            }
        }
        DoubleTensor::new(&[d, d], covariance).expect("a d × d matrix has d² elements")
    }

    /// The lower triangular factor L of the covariance LLᵀ, as a d × d matrix: diag(exp(ω)) for the mean-field
    /// family.
    pub fn scale_factor(&self) -> DoubleTensor {
        let d = self.q.dim;
        DoubleTensor::new(&[d, d], self.q.factor()).expect("a d × d matrix has d² elements")
    }

    /// The ELBO estimates: the initial approximation's first, then one every [`Advi::eval_elbo`] iterations.
    pub fn elbo_trace(&self) -> &[f64] {
        &self.elbo_trace
    }

    /// The step size scale η used: the adaptation's choice, or [`Advi::eta`] without adaptation.
    pub fn eta(&self) -> f64 {
        self.eta
    }

    /// The iterations run, adaptation excluded.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Whether the relative ELBO change fell below [`Advi::tol_rel_obj`] before [`Advi::max_iterations`].
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// One draw from the approximation, in unconstrained space.
    pub fn draw(&self, rng: &mut KeanuRng) -> Vec<f64> {
        self.q.draw(rng).1
    }

    /// `count` independent draws from the approximation, recording the values of `nodes` at each: each draw is
    /// mapped to the latents' constrained values ([`Parameterisation::to_state`]) and their descendants are
    /// recomputed. The log probability recorded with each sample is the model's at the draw (without the Jacobian),
    /// as the samplers record it. `state` must be the state the approximation was fitted with (or one with values
    /// for the same nodes); it is left at the last draw.
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidArgument`] if `nodes` is empty, [`keanu_graph::GraphError::UnknownNode`] if a node
    /// is not in the model's graph, and any error evaluating the model.
    pub fn sample(
        &self,
        state: &mut State,
        nodes: &[NodeId],
        count: usize,
        rng: &mut KeanuRng,
    ) -> Result<NetworkSamples> {
        sample_approximation(&self.param, || self.draw(rng), state, nodes, count)
    }
}
