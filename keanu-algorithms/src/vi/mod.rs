//! Variational inference: fitting a simple distribution to the posterior by optimization instead of sampling it.
//!
//! [`Advi`] is automatic differentiation variational inference (Kucukelbir et al. 2017) with Stan's defaults. It
//! fits a Gaussian, either with independent coordinates ([`VariationalFamily::MeanField`]) or with a full covariance
//! ([`VariationalFamily::FullRank`]), to the posterior of a model's continuous latents in unconstrained space (see
//! [`Parameterisation`](keanu_graph::Parameterisation)), by maximising the evidence lower bound (ELBO) with
//! stochastic gradient ascent. The fitted [`AdviResult`] holds the Gaussian's mean and covariance and draws
//! constrained samples from it.
//!
//! ADVI is usually much faster than NUTS, but it is an approximation: a mean-field fit ignores correlations between
//! latents and underestimates their marginal variances, and either family is Gaussian in unconstrained space, so it
//! cannot represent skewed or multimodal posteriors there. Use it for quick estimates, large models, or initialising
//! a sampler, and check important results with NUTS.
//!
//! With a [`Minibatch`], ADVI subsamples the data each iteration: it observes a random batch of B of N rows and
//! weighs the batch's log likelihood by N / B, so that each iteration costs B rows rather than N.
//!
//! [`Pathfinder`] (Zhang et al. 2022) is cheaper still: it follows an L-BFGS optimization path, fits a Gaussian
//! with a low-rank-plus-diagonal covariance at each point of it from the optimizer's curvature estimates, and keeps
//! the one with the best ELBO. Several paths from random starting points, with their pooled draws resampled by
//! Pareto-smoothed importance weights, make it more robust. Its draws are good starting points for NUTS chains.
//!
//! ```
//! use keanu_algorithms::vi::Advi;
//! use keanu_distributions::KeanuRng;
//! use keanu_graph::{GraphBuilder, Model};
//! use keanu_tensor::DoubleTensor;
//!
//! let b = GraphBuilder::new();
//! let mu = b.gaussian(0.0, 10.0);
//! let x = b.gaussian(mu, 1.0);
//! b.set_value(mu, DoubleTensor::scalar(0.0));
//! b.observe(x, DoubleTensor::scalar(3.0));
//! let mu = mu.id();
//! let (graph, mut state) = b.build()?;
//! let model = Model::new(graph, &state)?;
//!
//! let mut rng = KeanuRng::seed_from_u64(1);
//! let result = Advi::mean_field().with_tol_rel_obj(0.001).fit(&model, &mut state, &mut rng)?;
//! // the posterior of mu is N(300 / 101, 100 / 101)
//! assert!((result.mean()[0] - 300.0 / 101.0).abs() < 0.2);
//! let samples = result.sample(&mut state, &[mu], 1000, &mut rng)?;
//! assert_eq!(samples.len(), 1000);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod advi;
mod minibatch;
mod pathfinder;

pub use advi::{Advi, AdviResult, VariationalFamily};
pub use minibatch::{Batches, Minibatch, MinibatchGroup};
pub use pathfinder::{Pathfinder, PathfinderPath, PathfinderResult};
