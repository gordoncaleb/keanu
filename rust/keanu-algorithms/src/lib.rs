//! Inference algorithms for [Keanu](https://github.com/gordoncaleb/keanu) models: gradient optimizers for MAP and
//! maximum likelihood, MCMC samplers (Metropolis–Hastings, NUTS) and simulated annealing, a particle filter,
//! forward sampling from the prior, sample containers and chain statistics.
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

#![warn(missing_docs)]

mod error;
pub mod forward;
pub mod mcmc;
pub mod optimizer;
pub mod particle_filter;
pub mod samples;
pub mod statistics;

pub use error::{AlgorithmError, Result};
pub use forward::ForwardSampler;
pub use samples::{NetworkSample, NetworkSamples, Samples};
