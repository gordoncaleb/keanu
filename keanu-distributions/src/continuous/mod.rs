//! Univariate continuous distributions.
//!
//! `log_prob` is element-wise and broadcasts x with the parameters. Gradients are element-wise too, shaped like
//! the broadcast of x and the parameters; they are unspecified where `log_prob` is -infinity.

use crate::error::{Result, invalid_parameter};
use crate::param::{IntoParam, Param};
use crate::rng::KeanuRng;
use crate::util::{
    digamma, ln_gamma, map, map_n, require_ordered, require_positive, safe_ln_times, sample_each,
    sample_shape, scalar, sigmoid, softplus,
};
use keanu_tensor::DoubleTensor;
use keanu_tensor::special::{log_bessel_i0, log_bessel_i1};
use std::f64::consts::{LN_2, PI};

mod bounded;
mod circular;
mod location_scale;
mod positive;

pub use bounded::*;
pub use circular::*;
pub use location_scale::*;
pub use positive::*;

pub(crate) const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;
