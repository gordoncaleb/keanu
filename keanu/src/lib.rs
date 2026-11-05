//! # Keanu
//!
//! A probabilistic programming library: describe a model as a graph of random variables and the deterministic
//! operations that combine them, observe some of them, then find the most probable values of the rest
//! (optimization) or sample from their posterior (MCMC).
//!
//! ## Quick start
//!
//! Two thermometers read a room's temperature, each with its own noise. What is the temperature?
//!
//! ```
//! use keanu::prelude::*;
//!
//! # fn main() -> keanu::Result<()> {
//! // describe the model
//! let b = GraphBuilder::new();
//! let temperature = b.uniform(20.0, 30.0).label("temperature");
//! let first = b.gaussian(temperature, 2.5);
//! let second = b.gaussian(temperature, 5.0);
//! b.observe(first, DoubleTensor::scalar(25.0));
//! b.observe(second, DoubleTensor::scalar(30.0));
//! let temperature = temperature.id();
//!
//! // build it, starting from a value inside the prior's support
//! let (graph, mut state) = b.build()?;
//! state.set(&graph, temperature, DoubleTensor::scalar(25.0))?;
//! let model = Model::new(graph, &state)?;
//!
//! // the most probable temperature
//! let map = GradientOptimizer::new(&model).max_a_posteriori(&mut state)?;
//! let most_probable = map.get(temperature).unwrap().to_scalar().unwrap();
//! assert!((most_probable - 26.0).abs() < 1e-3);
//!
//! // samples from the posterior
//! let mut rng = KeanuRng::seed_from_u64(1);
//! let samples = Nuts::default().sample(&model, &mut state, &[temperature], 2000, &mut rng)?;
//! let mean = samples.f64(temperature).unwrap().mean()?.to_scalar().unwrap();
//! assert!((mean - 26.0).abs() < 0.5);
//! # Ok(())
//! # }
//! ```
//!
//! ## Layers
//!
//! | Module | Contents |
//! |---|---|
//! | [`tensor`] | N-dimensional tensors of `f64`, `i64` and `bool` with numpy-style broadcasting and slicing |
//! | [`distributions`] | Probability distributions: log probability, gradients and sampling, and [`KeanuRng`](distributions::KeanuRng) |
//! | [`graph`] | Building graphs ([`GraphBuilder`](graph::GraphBuilder)), their [`State`](graph::State), auto diff, log probabilities, [`Model`](graph::Model) and sequences (plates) |
//! | [`algorithms`] | Optimizers, MCMC (Metropolis–Hastings, NUTS, simulated annealing), forward sampling, the particle filter and sample statistics |
//! | [`io`] | Saving and loading models (JSON, protobuf) and Graphviz export |
//!
//! ## Features
//!
//! - `json` (default): load and save models as JSON.
//! - `protobuf`: load and save models as binary protobuf.
//! - `dot`: Graphviz DOT export.
//! - `parallel`: run independent MCMC chains in parallel with rayon.
//!
//! A [`Graph`](graph::Graph) is immutable and `Send + Sync`; all mutable values live in a
//! [`State`](graph::State), so many chains can share one graph, each with its own state.

#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub use keanu_algorithms as algorithms;
pub use keanu_distributions as distributions;
pub use keanu_graph as graph;
pub use keanu_io as io;
pub use keanu_tensor as tensor;

mod error;
pub use error::{Error, Result};

/// The types most programs need.
pub mod prelude {
    pub use crate::algorithms::forward::ForwardSampler;
    pub use crate::algorithms::mcmc::annealing::{AnnealingSchedule, SimulatedAnnealing};
    pub use crate::algorithms::mcmc::metropolis::{
        MetropolisHastings, RejectionStrategy, VariableSelector,
    };
    pub use crate::algorithms::mcmc::nuts::Nuts;
    pub use crate::algorithms::mcmc::proposal::{GaussianProposal, PriorProposal};
    pub use crate::algorithms::mcmc::select::Sampler;
    pub use crate::algorithms::mcmc::{SamplingOptions, generate};
    pub use crate::algorithms::optimizer::{
        Adam, Bobyqa, ConjugateGradient, GradientOptimizer, NonGradientOptimizer, OptimizedResult,
        OptimizerBounds,
    };
    pub use crate::algorithms::regression::{
        Fitter, LinearRegression, LogisticRegression, PredictiveModel, Prior, RegressionModel,
        Regularization, SamplingFitting, accuracy, coefficient_of_determination,
    };
    pub use crate::algorithms::{NetworkSample, NetworkSamples, Samples};
    pub use crate::distributions::KeanuRng;
    pub use crate::graph::{
        BoolVar, CustomDistribution, CustomGradient, DType, DoubleVar, Graph, GraphBuilder, IntVar,
        Model, NodeId, State, Support, Value, Var,
    };
    pub use crate::tensor::{BoolTensor, DoubleTensor, IntTensor, Tensor};
    pub use crate::{Error, Result};
}

/// Compiles and runs the code in the repository's README as a doctest.
#[cfg(doctest)]
#[doc = include_str!("../../README.md")]
struct ReadmeDoctests;
