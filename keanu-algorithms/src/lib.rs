//! Inference algorithms for [Keanu](https://github.com/gordoncaleb/keanu) models: gradient optimizers and a
//! non-gradient optimizer (BOBYQA) for MAP and maximum likelihood, MCMC samplers (Metropolis–Hastings, NUTS, Gibbs and discrete random-walk steps) and simulated annealing, tempered sequential Monte Carlo with marginal likelihood estimates and approximate Bayesian computation (SMC-ABC) for simulator models, variational inference (ADVI and Pathfinder), the Laplace approximation, a particle filter,
//! forward sampling from the prior, prior-, posterior- and interventional predictive sampling, sample containers, chain statistics,
//! multi-chain convergence diagnostics, the pointwise log likelihood, Pareto-smoothed importance sampling, model comparison by PSIS-LOO and WAIC, export of draws to CmdStan CSV, recovery of marginalised discrete latents, and smoothing, decoding and forecasting of time-series nodes over posterior draws.
//!
//! ```
//! use keanu_algorithms::optimizer::{Adam, GradientOptimizer};
//! use keanu_graph::{GraphBuilder, Model};
//! use keanu_tensor::DoubleTensor;
//!
//! let b = GraphBuilder::new();
//! let mu = b.gaussian(0.0, 10.0);
//! let x = b.gaussian(mu, 1.0);
//! b.set_value(mu, DoubleTensor::scalar(1.0));
//! b.observe(x, DoubleTensor::scalar(3.0));
//! let (graph, state) = b.build()?;
//! let mut state = state;
//! let model = Model::new(graph, &state)?;
//! let result = GradientOptimizer::new(&model).algorithm(Adam::default()).max_a_posteriori(&mut state)?;
//! // the posterior mode of mu is 3 * 100 / 101
//! let mu_map = result.values[0].1.to_scalar().unwrap();
//! assert!((mu_map - 300.0 / 101.0).abs() < 1e-2);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Licence
//!
//! This crate is MIT licensed, except for [`optimizer::bobyqa::prima`], a Rust translation of the BOBYQA solver
//! from [PRIMA](https://github.com/libprima/prima), which remains under PRIMA's BSD 3-Clause licence. See
//! `THIRD-PARTY-NOTICES.md` in the crate.

#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod abc;
pub mod diagnostics;
mod error;
pub mod export;
pub mod forward;
pub mod laplace;
pub mod loglik;
pub mod loo;
pub mod marginal;
pub mod mcmc;
pub mod optimizer;
pub mod particle_filter;
pub mod predictive;
pub mod psis;
pub mod regression;
pub mod samples;
pub mod smc;
pub mod statistics;
pub mod sts;
mod support;
pub mod timeseries;
pub mod vi;

pub use abc::{Abc, AbcKernel, SummaryStatistics};
pub use error::{AlgorithmError, Result};
pub use export::{CmdStanCsv, write_labeled_trace};
pub use forward::ForwardSampler;
/// How the Laplace approximation and the trust-region optimizer compute Hessians (exact, finite differences, or
/// exact where possible); see [`Laplace::with_hessian_method`].
pub use keanu_graph::HessianMethod;
/// The space NUTS and the gradient optimizers move in (unconstrained by default); see [`mcmc::nuts`] and
/// [`optimizer`].
pub use keanu_graph::Space;
pub use laplace::{Laplace, LaplaceResult};
pub use loglik::{LogLikPoint, PointwiseLogLik};
pub use loo::{Comparison, Loo, RefitCrossValidation, Waic, kfold, refit_loo};
pub use marginal::{LatentMarginals, recover_marginals};
pub use predictive::PredictiveSampler;
pub use psis::Psis;
pub use samples::{ChainSamples, ChainSummary, NetworkSample, NetworkSamples, Samples, SummaryRow};
pub use smc::{Smc, SmcKernel, SmcResult};
pub use vi::{
    Advi, AdviResult, Minibatch, MinibatchGroup, Pathfinder, PathfinderResult, VariationalFamily,
};
