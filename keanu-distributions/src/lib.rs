//! Probability distributions for [Keanu](https://github.com/gordoncaleb/keanu).
//!
//! Each distribution is a struct of parameter tensors with:
//! - `log_prob(x)`: the element-wise log density or mass, broadcasting x with the parameters
//!   (-infinity outside the support),
//! - `d_log_prob(x)`: derivatives of the log density with respect to x and each parameter, as a typed gradient
//!   struct (where the distribution is differentiable),
//! - `sample(shape, rng)`: draws using Keanu's own algorithms and an explicit [`KeanuRng`].
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

pub mod continuous;
pub mod discrete;
mod error;
pub mod kde;
pub mod multivariate;
mod rng;
mod util;

pub use continuous::*;
pub use discrete::*;
pub use error::{DistributionError, Result};
pub use kde::{Kde, KdeGradient};
pub use multivariate::*;
pub use rng::KeanuRng;
