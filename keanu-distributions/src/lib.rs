//! Probability distributions for [Keanu](https://github.com/gordoncaleb/keanu).
//!
//! Each distribution is a struct of parameter tensors, owned or borrowed (see [`Param`]), with:
//! - `log_prob(x)`: the element-wise log density or mass, broadcasting x with the parameters
//!   (-infinity outside the support),
//! - `d_log_prob(x)`: derivatives of the log density with respect to x and each parameter, as a typed gradient
//!   struct (where the distribution is differentiable),
//! - `sample(shape, rng)`: draws using Keanu's own algorithms and an explicit [`KeanuRng`].
//!   Sampling with parameters outside their valid range (e.g. a non-positive scale) is an
//!   [`DistributionError::InvalidParameter`] error.
//!
//! ```
//! use keanu_distributions::{Gaussian, KeanuRng};
//! use keanu_tensor::DoubleTensor;
//!
//! let normal = Gaussian::new(DoubleTensor::scalar(0.0), DoubleTensor::scalar(1.0));
//! let log_prob = normal.log_prob(&DoubleTensor::vector(vec![0.0, 1.0]))?;
//! assert!((log_prob.as_slice()[0] + 0.918_938_533).abs() < 1e-9);
//!
//! let mut rng = KeanuRng::seed_from_u64(1);
//! let samples = normal.sample(&[1000], &mut rng)?;
//! assert_eq!(samples.shape(), &[1000]);
//! # Ok::<(), keanu_distributions::DistributionError>(())
//! ```

#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod cdf;
pub mod continuous;
pub mod directional;
pub mod discrete;
mod error;
mod event;
mod jet;
pub mod kde;
pub mod matrix;
pub mod multivariate;
pub mod multivariate_discrete;
mod param;
pub mod processes;
mod quadrature;
mod real;
mod rng;
pub mod spatial;
pub mod timeseries;
mod util;
pub mod wiener;

pub use cdf::BetaBinomialTail;
pub use continuous::*;
pub use directional::*;
pub use discrete::*;
pub use error::{DistributionError, Result};
pub use jet::ElementDerivatives;
pub use kde::{Kde, KdeGradient};
pub use matrix::*;
pub use multivariate::*;
pub use multivariate_discrete::*;
pub use param::{IntoParam, Param};
pub use processes::{
    Autoregressive, AutoregressiveGradient, Garch11, Garch11Gradient, GaussianRandomWalk,
    GaussianRandomWalkGradient,
};
pub use rng::KeanuRng;
pub use spatial::{Adjacency, Car, CarGradient, Icar, IcarGradient};
pub use timeseries::{
    FutureStep, HiddenMarkovChain, HiddenMarkovChainGradient, KalmanFilter, KalmanSmoother,
    LinearGaussianStateSpace, LinearGaussianStateSpaceGradient, StateSpaceForecast, StateSpacePath,
};
pub use wiener::{Wiener, WienerFull, WienerFullGradient, WienerGradient, sample_wiener};
