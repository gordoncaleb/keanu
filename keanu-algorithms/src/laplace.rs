//! The Laplace approximation: a Gaussian fitted to the posterior at its mode, in unconstrained space.
//!
//! [`Laplace::fit`] maps each continuous latent to the real line through the transform of its support (see
//! [`Parameterisation`]) and finds the mode ẑ of the density of that unconstrained point, the model's log
//! probability *plus* the log-Jacobian of the transforms, with [`Lbfgs`]. It then computes the exact Hessian H of
//! that log density (by forward-over-reverse automatic differentiation, [`Parameterisation::hessian`]), polishes
//! the mode with up to five Newton steps on it (L-BFGS stops on relative tolerances a little short of the mode; a
//! Gaussian posterior needs a single step), and approximates the posterior by the Gaussian N(ẑ, (−H)⁻¹), with H at
//! the final ẑ. A Newton step is taken only if it does not lower the log density. Draws z = ẑ + L⁻ᵀε, with L the lower Cholesky factor
//! of −H and ε standard normal, are mapped back to the latents' own values, so they always respect the supports.
//!
//! The mode in unconstrained space is not the MAP of [`GradientOptimizer`](crate::optimizer::GradientOptimizer),
//! which leaves the Jacobian out: for x ~ Gamma(θ, k) the Laplace mode is x = kθ (the mode of log x's density),
//! while the MAP is (k − 1)θ.
//!
//! The approximation is exact for a Gaussian posterior over unbounded latents, and good when the posterior is
//! close to Gaussian in unconstrained space, as it is for many models with plenty of data. It cannot represent
//! skewed or multimodal posteriors there; check important results with NUTS.
//!
//! Circular latents (such as a von Mises angle) are not supported: their transform's auxiliary radial density is
//! largest at the origin of the plane, where the angle is undefined, so there is no mode to approximate around.
//!
//! ```
//! use keanu_algorithms::Laplace;
//! use keanu_distributions::KeanuRng;
//! use keanu_graph::{GraphBuilder, Model};
//! use keanu_tensor::DoubleTensor;
//!
//! // mu ~ N(0, 10), observed once through x ~ N(mu, 1): the posterior is N(300 / 101, 100 / 101)
//! let b = GraphBuilder::new();
//! let mu = b.gaussian(0.0, 10.0);
//! let x = b.gaussian(mu, 1.0);
//! b.set_value(mu, DoubleTensor::scalar(0.0));
//! b.observe(x, DoubleTensor::scalar(3.0));
//! let mu = mu.id();
//! let (graph, mut state) = b.build()?;
//! let model = Model::new(graph, &state)?;
//!
//! let result = Laplace::new().fit(&model, &mut state)?;
//! assert!((result.mode()[0] - 300.0 / 101.0).abs() < 1e-6);
//! assert!((result.covariance().as_slice()[0] - 100.0 / 101.0).abs() < 1e-9);
//! let samples = result.sample(&mut state, &[mu], 100, &mut KeanuRng::seed_from_u64(1))?;
//! assert_eq!(samples.len(), 100);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{AlgorithmError, Result};
use crate::optimizer::{Fitness, Lbfgs, Objective};
use crate::samples::NetworkSamples;
use crate::support::{check_latents, compute_deterministic, node_name, norm, sample_approximation};
use keanu_distributions::KeanuRng;
use keanu_graph::{Model, NodeId, Parameterisation, State, Support, node_support};
use keanu_tensor::{DoubleTensor, TensorError};

/// Laplace approximation settings: the [`Lbfgs`] configuration that finds the mode. The configuration holds no
/// state, so it can be reused for any number of fits.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Laplace {
    lbfgs: Lbfgs,
}

impl Laplace {
    /// The default settings: L-BFGS with its defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the L-BFGS configuration used to find the mode.
    #[must_use]
    pub fn with_lbfgs(mut self, lbfgs: Lbfgs) -> Self {
        self.lbfgs = lbfgs;
        self
    }

    /// Fits the approximation to the posterior of `model`'s continuous latents, starting the search for the mode
    /// from their current values in `state`, and leaves `state` at the mode.
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidSetting`] for an out-of-range L-BFGS setting; [`AlgorithmError::NoLatents`] or
    /// [`AlgorithmError::DiscreteLatents`] if the model has no latents or discrete ones;
    /// [`AlgorithmError::InvalidArgument`], naming the latent, for a latent with a [`Support::Circular`] support
    /// (see the [module documentation](self));
    /// [`AlgorithmError::ImpossibleStart`] if a latent's value is outside its support (or on its boundary), or the
    /// log probability at the start is not finite; an error from [`Parameterisation::new`] if a latent's support has
    /// no transform; [`AlgorithmError::ModeOnBoundary`] if L-BFGS heads for the boundary of a latent's support;
    /// [`keanu_graph::GraphError::NoSecondOrderRule`], naming the node, if the model has a distribution or op on the
    /// path from a latent to the log probability without a second-order rule; and
    /// [`AlgorithmError::IndefiniteHessian`] if −H at the mode is not positive definite, as at a saddle point or
    /// along a direction in which the posterior is flat.
    pub fn fit<'m>(&self, model: &'m Model, state: &mut State) -> Result<LaplaceResult<'m>> {
        check_latents(model)?;
        compute_deterministic(model.graph(), state)?;
        for &id in model.latents() {
            if node_support(model.graph(), state, id)? == Support::Circular {
                return Err(AlgorithmError::InvalidArgument(format!(
                    "the Laplace approximation does not support the circular latent {}: its transform's \
                     auxiliary density has its mode at the origin, where the angle is undefined (sample the \
                     posterior with NUTS instead)",
                    model.graph().describe(id)
                )));
            }
        }
        let param = Parameterisation::new(model, state)?;
        if param.dimension() == 0 {
            return Err(AlgorithmError::NoLatents);
        }
        let objective = Objective::new(param, state, Fitness::LogProb, true)?;
        let start = objective.flat_point(state)?;
        let (mut mode, mut log_density, mut converged) =
            self.lbfgs.maximize(&objective, state, start, None)?;
        if let Some(latent) = objective.boundary(state, &mode)? {
            return Err(AlgorithmError::ModeOnBoundary(node_name(
                model.graph(),
                latent,
            )));
        }
        let name = |i: usize| {
            objective
                .parameterisation()
                .coordinate_latent(i)
                .map_or_else(
                    || format!("coordinate {i}"),
                    |id| model.graph().describe(id),
                )
        };
        let mut hessian = objective.parameterisation().hessian(&mode, true, state)?;
        let mut factor = negated_cholesky(&hessian, name)?;
        // polish the mode with Newton steps on the exact Hessian: L-BFGS stops on its relative tolerances, a little
        // short of the mode, and a quadratic log density (a Gaussian posterior) takes one step
        let (_, mut gradient) = objective.value_and_gradient_flat(state, &mode)?;
        for _ in 0..NEWTON_STEPS {
            let step = solve_with_factor(&factor, &gradient)?;
            let size = norm(&step);
            if !size.is_finite() {
                break;
            }
            if size <= NEWTON_TOLERANCE * (1.0 + norm(&mode)) {
                converged = true;
                break;
            }
            let trial: Vec<f64> = mode.iter().zip(&step).map(|(z, s)| z + s).collect();
            let (value, trial_gradient) = objective.value_and_gradient_flat(state, &trial)?;
            if !(value.is_finite() && value >= log_density)
                || !trial_gradient.iter().all(|g| g.is_finite())
                || objective.boundary(state, &trial)?.is_some()
            {
                break;
            }
            (mode, log_density, gradient) = (trial, value, trial_gradient);
            hessian = objective.parameterisation().hessian(&mode, true, state)?;
            factor = negated_cholesky(&hessian, name)?;
        }
        let evaluations = objective.evaluations();
        let param = objective.into_parameterisation();
        param.to_state(&mode, state)?;
        let covariance = factor.cholesky_inverse()?;
        let values = param
            .latents()
            .iter()
            .copied()
            .zip(param.to_constrained(&mode)?)
            .collect();
        Ok(LaplaceResult {
            param,
            mode,
            values,
            log_density,
            hessian,
            factor,
            covariance,
            converged,
            evaluations,
        })
    }
}

/// The most Newton steps taken to polish the mode L-BFGS finds.
const NEWTON_STEPS: usize = 5;

/// A Newton step shorter than this, relative to 1 + |ẑ|, means the mode has converged.
const NEWTON_TOLERANCE: f64 = 1e-12;

/// (L Lᵀ)⁻¹ b, for a lower triangular L (n × n) with a positive diagonal.
fn solve_with_factor(factor: &DoubleTensor, b: &[f64]) -> Result<Vec<f64>> {
    let y = factor.solve_triangular(&DoubleTensor::vector(b.to_vec()), true, false)?;
    Ok(factor.solve_triangular(&y, true, true)?.as_slice().to_vec())
}

/// The lower Cholesky factor L of −H (symmetrised), where `hessian` is n × n: L Lᵀ = −(H + Hᵀ) / 2. `name` names
/// the latent of a coordinate, for the error when −H is not positive definite.
fn negated_cholesky(
    hessian: &DoubleTensor,
    name: impl Fn(usize) -> String,
) -> Result<DoubleTensor> {
    let n = hessian.shape()[0];
    let h = hessian.as_slice();
    let mut negated = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            negated[i * n + j] = -0.5 * (h[i * n + j] + h[j * n + i]);
        }
    }
    if !negated.iter().all(|x| x.is_finite()) {
        return Err(AlgorithmError::IndefiniteHessian(
            "the Hessian of the log density at the mode has a NaN or infinite element, so there is no Gaussian \
             approximation there; the model may be ill-conditioned at the mode"
                .into(),
        ));
    }
    let negated = DoubleTensor::new(&[n, n], negated)?;
    match negated.cholesky() {
        Ok(factor) => Ok(factor),
        Err(TensorError::NotPositiveDefinite) => Err(indefinite(&negated, n, name)),
        Err(e) => Err(e.into()),
    }
}

/// The error for a −H that is not positive definite, naming the latents along the eigenvector of its smallest
/// eigenvalue.
fn indefinite(negated: &DoubleTensor, n: usize, name: impl Fn(usize) -> String) -> AlgorithmError {
    let along = match negated.eigh() {
        Ok((values, vectors)) => {
            let lambda = values.as_slice()[0];
            let column: Vec<f64> = (0..n).map(|i| vectors.as_slice()[i * n]).collect();
            let largest = column.iter().copied().map(f64::abs).fold(0.0, f64::max);
            let mut names: Vec<String> = Vec::new();
            for (i, c) in column.iter().enumerate() {
                let latent = name(i);
                if c.abs() >= 0.1 * largest && !names.contains(&latent) {
                    names.push(latent);
                }
            }
            format!(
                " (the smallest eigenvalue of the negated Hessian is {lambda:e}, along a direction involving {})",
                names.join(", ")
            )
        }
        Err(_) => String::new(),
    };
    AlgorithmError::IndefiniteHessian(format!(
        "the Hessian of the log density at the mode is not negative definite{along}: the point found is a saddle \
         point, or the posterior is flat in that direction (a latent the priors and data do not identify), so it \
         has no Laplace approximation. Give those latents informative priors or reparameterise the model, or try \
         another starting point"
    ))
}

/// A fitted Laplace approximation: the Gaussian N(ẑ, (−H)⁻¹) over the unconstrained space of a model's continuous
/// latents (laid out by [`parameterisation`](Self::parameterisation)), where ẑ is the mode of the Jacobian-adjusted
/// log density and H its Hessian there.
#[derive(Debug)]
pub struct LaplaceResult<'m> {
    param: Parameterisation<'m>,
    mode: Vec<f64>,
    values: Vec<(NodeId, DoubleTensor)>,
    log_density: f64,
    hessian: DoubleTensor,
    factor: DoubleTensor,
    covariance: DoubleTensor,
    converged: bool,
    evaluations: usize,
}

impl LaplaceResult<'_> {
    /// The parameterisation the approximation lives in: [`Parameterisation::range`] gives a latent's coordinates in
    /// [`mode`](Self::mode) and [`covariance`](Self::covariance).
    pub fn parameterisation(&self) -> &Parameterisation<'_> {
        &self.param
    }

    /// The mode ẑ in unconstrained space: the approximation's mean.
    pub fn mode(&self) -> &[f64] {
        &self.mode
    }

    /// Each latent's value at the mode: the constrained values of [`mode`](Self::mode). That is not the latents'
    /// posterior mean, nor their MAP, when a transform is nonlinear.
    pub fn values(&self) -> &[(NodeId, DoubleTensor)] {
        &self.values
    }

    /// A latent's value at the mode (see [`values`](Self::values)).
    pub fn get(&self, id: NodeId) -> Option<&DoubleTensor> {
        self.values.iter().find(|(n, _)| *n == id).map(|(_, v)| v)
    }

    /// The log density of the unconstrained point at the mode: the model's log probability plus the log-Jacobian
    /// of the transforms.
    pub fn log_density(&self) -> f64 {
        self.log_density
    }

    /// The exact Hessian H of the log density at the mode, as a d × d matrix (symmetric up to rounding).
    pub fn hessian(&self) -> &DoubleTensor {
        &self.hessian
    }

    /// The lower triangular Cholesky factor L of the precision −H (symmetrised), as a d × d matrix: L Lᵀ = −H.
    pub fn precision_factor(&self) -> &DoubleTensor {
        &self.factor
    }

    /// The approximation's covariance (−H)⁻¹ in unconstrained space, as a d × d matrix.
    pub fn covariance(&self) -> &DoubleTensor {
        &self.covariance
    }

    /// Whether the mode converged: L-BFGS met its convergence criterion, or the Newton polish reached a step
    /// shorter than 10⁻¹² relative to 1 + |ẑ|. When neither did, the mode, and so the approximation, may be
    /// inaccurate.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// The number of log density and gradient evaluations L-BFGS and the Newton polish used (Hessians are not
    /// counted).
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// One draw z = ẑ + L⁻ᵀε from the approximation, in unconstrained space, where ε is standard normal and L is
    /// the [`precision_factor`](Self::precision_factor).
    pub fn draw(&self, rng: &mut KeanuRng) -> Vec<f64> {
        let n = self.mode.len();
        let eps: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
        let x = self
            .factor
            .solve_triangular(&DoubleTensor::vector(eps), true, true)
            .expect("the precision factor is n × n with a positive diagonal");
        x.as_slice()
            .iter()
            .zip(&self.mode)
            .map(|(xi, m)| xi + m)
            .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(n: usize, data: &[f64]) -> DoubleTensor {
        DoubleTensor::new(&[n, n], data.to_vec()).unwrap()
    }

    #[test]
    fn the_factor_of_a_negative_definite_hessian() {
        // an asymmetric H (as rounding can make it) is symmetrised first
        let h = matrix(2, &[-4.0, 1.0 + 1e-12, 1.0 - 1e-12, -3.0]);
        let l = negated_cholesky(&h, |i| format!("z{i}")).unwrap();
        let l = l.as_slice();
        let product = [
            l[0] * l[0],
            l[0] * l[2],
            l[2] * l[0],
            l[2] * l[2] + l[3] * l[3],
        ];
        for (p, e) in product.iter().zip([4.0, -1.0, -1.0, 3.0]) {
            assert!((p - e).abs() < 1e-12, "{product:?}");
        }
        assert_eq!(l[1], 0.0);
    }

    #[test]
    fn a_saddle_names_the_offending_direction() {
        // a maximum in z0 and a minimum in z1
        let h = matrix(2, &[-2.0, 0.0, 0.0, 0.5]);
        match negated_cholesky(&h, |i| format!("z{i}")) {
            Err(AlgorithmError::IndefiniteHessian(message)) => {
                assert!(message.contains("saddle"), "{message}");
                assert!(
                    message.contains("z1") && !message.contains("z0"),
                    "{message}"
                );
                assert!(message.contains("-5e-1"), "{message}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_non_finite_hessian_is_an_error() {
        let h = matrix(1, &[f64::NAN]);
        assert!(matches!(
            negated_cholesky(&h, |i| format!("z{i}")),
            Err(AlgorithmError::IndefiniteHessian(m)) if m.contains("NaN")
        ));
    }
}
