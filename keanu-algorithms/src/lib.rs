//! Inference algorithms for [Keanu](https://github.com/gordoncaleb/keanu) models: gradient optimizers and a
//! non-gradient optimizer (BOBYQA) for MAP and maximum likelihood, MCMC samplers (Metropolis–Hastings, NUTS) and simulated annealing, a particle filter,
//! forward sampling from the prior, prior- and posterior-predictive sampling, sample containers, chain statistics and
//! multi-chain convergence diagnostics.
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

pub mod diagnostics;
mod error;
pub mod forward;
pub mod mcmc;
pub mod optimizer;
pub mod particle_filter;
pub mod predictive;
pub mod regression;
pub mod samples;
pub mod statistics;
mod support;

pub use error::{AlgorithmError, Result};
pub use forward::ForwardSampler;
pub use predictive::PredictiveSampler;
pub use samples::{ChainSamples, ChainSummary, NetworkSample, NetworkSamples, Samples, SummaryRow};
