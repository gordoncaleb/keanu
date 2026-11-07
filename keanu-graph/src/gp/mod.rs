//! Exact Gaussian processes: covariance kernels, the marginal likelihood and the predictive distribution, all built
//! from ordinary graph nodes, so their hyperparameters can be latent variables with gradients.
//!
//! A [`Kernel`] builds the covariance matrix k(X1, X2) between two sets of inputs: `[n, d]` matrices with one row
//! per point, or `[n]` vectors for 1-D inputs. Kernels combine with `+` and `*`. A [`GaussianProcess`] adds a mean,
//! observation noise and a small diagonal jitter to a kernel, and builds
//!
//! - [`marginal`](GaussianProcess::marginal): an observed multivariate Gaussian over the targets y with covariance
//!   K(X, X) + (noise² + jitter)·I, whose log density is the GP marginal likelihood;
//! - [`predict`](GaussianProcess::predict): the mean and covariance of the latent function at new inputs given the
//!   training data, by a Cholesky factorisation and triangular solves (no explicit inverse).
//!
//! [`gp_marginal`] and [`gp_predict`] do the same with a constant mean in one call.
//!
//! For more points than an exact GP can handle, [`Hsgp`] approximates a GP with an exponentiated quadratic or
//! Matérn kernel by a fixed set of basis functions with Gaussian weights (Riutort-Mayol et al., 2023): the function
//! is an ordinary latent of the model, at a cost linear in the number of points (see the [`hsgp`] module).
//!
//! ```
//! use keanu_distributions::KeanuRng;
//! use keanu_graph::GraphBuilder;
//! use keanu_graph::gp::{GaussianProcess, Kernel};
//! use keanu_tensor::DoubleTensor;
//!
//! let x = DoubleTensor::vector(vec![0.0, 1.0, 2.0]);
//! let y = DoubleTensor::vector(vec![0.5, -0.2, 0.3]);
//! let b = GraphBuilder::new();
//! let length_scale = b.half_gaussian(2.0);
//! b.set_value(length_scale, DoubleTensor::scalar(1.0));
//! let gp = GaussianProcess::new(Kernel::exp_quad(&b, 1.0, length_scale)).with_noise(0.1);
//! gp.marginal(x.clone(), y.clone());
//! let prediction = gp.predict(x, y, DoubleTensor::vector(vec![0.5, 1.5]));
//! let (mean, sd) = (prediction.mean.id(), prediction.variance().sqrt().id());
//! let (graph, mut state) = b.build()?;
//! state.lazy_eval(&graph, &[mean, sd], &mut KeanuRng::seed_from_u64(1))?;
//! assert_eq!(state.f64(mean).unwrap().shape(), &[2]);
//! # Ok::<(), keanu_graph::GraphError>(())
//! ```
//!
//! Shape errors (inputs with different feature counts, hyperparameters or targets of the wrong shape) are recorded
//! by the builder like any other invalid node, so [`GraphBuilder::build`] returns them. Hyperparameter values (e.g.
//! positivity of a length scale) are not checked: give them priors with the right support.

pub mod hsgp;

pub use hsgp::{Hsgp, HsgpDomain, HsgpPrior};

use crate::builder::GraphBuilder;
use crate::error::GraphError;
use crate::vars::{DoubleVar, IntoDoubleVar};
use keanu_tensor::DoubleTensor;
use std::f64::consts::PI;
use std::fmt;
use std::ops::{Add, Mul};

/// The default jitter added to the diagonal of a GP's training covariance to keep its Cholesky factorisation stable.
pub const DEFAULT_JITTER: f64 = 1e-6;

/// The Matérn distance r is computed as √max(r², this), so that the gradient through the square root stays finite
/// (zero) where two inputs coincide, and where r² is at most this the correlation is exactly 1 (with gradient zero):
/// at r = 1e-15 a Matérn correlation with a small smoothness ν is still well below 1 (it falls as r^{2ν}), so the
/// clamped distance alone would put the covariance at zero distance below σ².
const MIN_SQUARED_DISTANCE: f64 = 1e-30;

const KERNEL: &str = "gaussian process kernel";
const GP: &str = "gaussian process";

/// A covariance function k(x, x') of a Gaussian process, whose hyperparameters are graph nodes (constants or
/// latent variables).
///
/// In the formulas below, σ is the amplitude and r² = Σ_d ((x_d − x'_d) / ℓ_d)² is the squared distance scaled by
/// the length scale ℓ, which is a scalar or has one element per feature (automatic relevance determination).
/// Kernels add and multiply with `+` and `*`; sums and products of kernels are kernels.
///
/// ```
/// use keanu_graph::GraphBuilder;
/// use keanu_graph::gp::Kernel;
/// use keanu_tensor::DoubleTensor;
///
/// let b = GraphBuilder::new();
/// let kernel = Kernel::exp_quad(&b, 1.0, 2.0) * Kernel::periodic(&b, 1.0, 0.5, 12.0)
///     + Kernel::linear(&b, 0.1, 0.3, 0.0);
/// let x = DoubleTensor::zeros(&[5, 2]);
/// let x_new = DoubleTensor::zeros(&[3, 2]);
/// assert_eq!(kernel.covariance(x, x_new).shape(), &[5, 3]);
/// ```
#[derive(Clone, Debug)]
pub struct Kernel<'b> {
    builder: &'b GraphBuilder,
    kind: Kind<'b>,
}

#[derive(Clone, Copy, Debug)]
enum Form {
    ExpQuad,
    Matern12,
    Matern32,
    Matern52,
    /// The Matérn kernel of any smoothness ν > 0, through the Bessel function K_ν.
    Matern(f64),
}

#[derive(Clone, Debug)]
enum Kind<'b> {
    Stationary {
        form: Form,
        amplitude: DoubleVar<'b>,
        length_scale: DoubleVar<'b>,
    },
    RationalQuadratic {
        amplitude: DoubleVar<'b>,
        length_scale: DoubleVar<'b>,
        alpha: DoubleVar<'b>,
    },
    Periodic {
        amplitude: DoubleVar<'b>,
        length_scale: DoubleVar<'b>,
        period: DoubleVar<'b>,
    },
    Linear {
        bias_amplitude: DoubleVar<'b>,
        slope_amplitude: DoubleVar<'b>,
        shift: DoubleVar<'b>,
    },
    Sum(Box<Kernel<'b>>, Box<Kernel<'b>>),
    Product(Box<Kernel<'b>>, Box<Kernel<'b>>),
}

/// Records an invalid-node error and returns a placeholder so that building can continue; the build fails.
fn fail<'b>(b: &'b GraphBuilder, op: &'static str, message: String) -> DoubleVar<'b> {
    b.record(GraphError::InvalidNode { op, message });
    b.scalar(0.0)
}

/// Whether `var` is a scalar, or (if `per_feature`) a vector of `d` elements; records an error if not.
fn check_parameter(
    b: &GraphBuilder,
    var: DoubleVar<'_>,
    name: &str,
    d: usize,
    per_feature: bool,
) -> bool {
    let shape = var.shape();
    if shape.is_empty() || (per_feature && shape == [d]) {
        return true;
    }
    let expected = if per_feature {
        format!("a scalar or one element per feature ([{d}])")
    } else {
        "a scalar".to_string()
    };
    fail(
        b,
        KERNEL,
        format!("{name} must be {expected}, not {shape:?}"),
    );
    false
}

/// Inputs as an `[n, d]` matrix with n and d, or None (with an error recorded) unless they are `[n]` or `[n, d]`.
fn as_matrix<'b>(
    b: &'b GraphBuilder,
    x: DoubleVar<'b>,
    name: &str,
) -> Option<(DoubleVar<'b>, usize, usize)> {
    match x.shape()[..] {
        [n] => Some((x.reshape(&[n as isize, 1]), n, 1)),
        [n, d] => Some((x, n, d)),
        ref shape => {
            fail(
                b,
                KERNEL,
                format!("{name} must be a vector [n] or a matrix [n, features], not {shape:?}"),
            );
            None
        }
    }
}

/// x1_i − x2_j for every pair of rows, `[n, m, d]`.
fn pairwise_differences<'b>(x1: DoubleVar<'b>, x2: DoubleVar<'b>, shape: Shapes) -> DoubleVar<'b> {
    let (n, m, d) = (shape.n as isize, shape.m as isize, shape.d as isize);
    x1.reshape(&[n, 1, d]) - x2.reshape(&[1, m, d])
}

#[derive(Clone, Copy)]
struct Shapes {
    n: usize,
    m: usize,
    d: usize,
}

impl<'b> Kernel<'b> {
    fn new(builder: &'b GraphBuilder, kind: Kind<'b>) -> Self {
        Self { builder, kind }
    }

    fn stationary(
        b: &'b GraphBuilder,
        form: Form,
        amplitude: impl IntoDoubleVar<'b>,
        length_scale: impl IntoDoubleVar<'b>,
    ) -> Self {
        let kind = Kind::Stationary {
            form,
            amplitude: amplitude.into_var(b),
            length_scale: length_scale.into_var(b),
        };
        Self::new(b, kind)
    }

    /// The exponentiated quadratic (squared exponential, RBF) kernel σ² exp(−r²/2).
    pub fn exp_quad(
        b: &'b GraphBuilder,
        amplitude: impl IntoDoubleVar<'b>,
        length_scale: impl IntoDoubleVar<'b>,
    ) -> Self {
        Self::stationary(b, Form::ExpQuad, amplitude, length_scale)
    }

    /// The Matérn ν = 1/2 (exponential) kernel σ² exp(−r).
    pub fn matern12(
        b: &'b GraphBuilder,
        amplitude: impl IntoDoubleVar<'b>,
        length_scale: impl IntoDoubleVar<'b>,
    ) -> Self {
        Self::stationary(b, Form::Matern12, amplitude, length_scale)
    }

    /// The Matérn ν = 3/2 kernel σ² (1 + √3 r) exp(−√3 r).
    pub fn matern32(
        b: &'b GraphBuilder,
        amplitude: impl IntoDoubleVar<'b>,
        length_scale: impl IntoDoubleVar<'b>,
    ) -> Self {
        Self::stationary(b, Form::Matern32, amplitude, length_scale)
    }

    /// The Matérn ν = 5/2 kernel σ² (1 + √5 r + 5r²/3) exp(−√5 r).
    pub fn matern52(
        b: &'b GraphBuilder,
        amplitude: impl IntoDoubleVar<'b>,
        length_scale: impl IntoDoubleVar<'b>,
    ) -> Self {
        Self::stationary(b, Form::Matern52, amplitude, length_scale)
    }

    /// The Matérn kernel of smoothness ν > 0 (a constant): σ² 2^{1−ν}/Γ(ν) (√(2ν) r)^ν K_ν(√(2ν) r), with K_ν the
    /// modified Bessel function of the second kind (the correlation is [`DoubleVar::matern_correlation`], an
    /// [`Op::Bessel`](crate::op::Op::Bessel) node, differentiable to second order in the distance, so in the
    /// amplitude, length scale and inputs). Its sample paths are ⌈ν⌉ − 1 times differentiable, and it is the
    /// exponentiated quadratic in the limit ν → ∞. At ν = ½, 3/2 and 5/2 it equals [`matern12`](Self::matern12),
    /// [`matern32`](Self::matern32) and [`matern52`](Self::matern52) to rounding, which are cheaper. It has a
    /// spectral density, so [`Hsgp`] approximates it for any ν. A ν that is not finite and positive is a build
    /// error.
    pub fn matern(
        b: &'b GraphBuilder,
        nu: f64,
        amplitude: impl IntoDoubleVar<'b>,
        length_scale: impl IntoDoubleVar<'b>,
    ) -> Self {
        if !(nu.is_finite() && nu > 0.0) {
            fail(
                b,
                KERNEL,
                format!("the Matérn smoothness ν must be finite and positive, not {nu}"),
            );
        }
        Self::stationary(b, Form::Matern(nu), amplitude, length_scale)
    }

    /// The rational quadratic kernel σ² (1 + r² / (2α))^(−α), a scale mixture of exponentiated quadratics that
    /// tends to [`exp_quad`](Self::exp_quad) as the (scalar) α grows.
    pub fn rational_quadratic(
        b: &'b GraphBuilder,
        amplitude: impl IntoDoubleVar<'b>,
        length_scale: impl IntoDoubleVar<'b>,
        alpha: impl IntoDoubleVar<'b>,
    ) -> Self {
        let kind = Kind::RationalQuadratic {
            amplitude: amplitude.into_var(b),
            length_scale: length_scale.into_var(b),
            alpha: alpha.into_var(b),
        };
        Self::new(b, kind)
    }

    /// The periodic (exp-sine-squared) kernel σ² exp(−2 Σ_d sin²(π (x_d − x'_d) / p) / ℓ_d²), with period p a
    /// scalar or one per feature. For 1-D inputs this is the usual σ² exp(−2 sin²(π |x − x'| / p) / ℓ²).
    pub fn periodic(
        b: &'b GraphBuilder,
        amplitude: impl IntoDoubleVar<'b>,
        length_scale: impl IntoDoubleVar<'b>,
        period: impl IntoDoubleVar<'b>,
    ) -> Self {
        let kind = Kind::Periodic {
            amplitude: amplitude.into_var(b),
            length_scale: length_scale.into_var(b),
            period: period.into_var(b),
        };
        Self::new(b, kind)
    }

    /// The linear (dot product) kernel σ_b² + σ_s² (x − c)·(x' − c), with bias amplitude σ_b, slope amplitude σ_s
    /// and a shift c that is a scalar or one per feature. Its covariance matrices have rank at most d + 1.
    pub fn linear(
        b: &'b GraphBuilder,
        bias_amplitude: impl IntoDoubleVar<'b>,
        slope_amplitude: impl IntoDoubleVar<'b>,
        shift: impl IntoDoubleVar<'b>,
    ) -> Self {
        let kind = Kind::Linear {
            bias_amplitude: bias_amplitude.into_var(b),
            slope_amplitude: slope_amplitude.into_var(b),
            shift: shift.into_var(b),
        };
        Self::new(b, kind)
    }

    /// The builder this kernel's nodes belong to.
    pub fn builder(&self) -> &'b GraphBuilder {
        self.builder
    }

    /// The covariance matrix `[n, m]` between inputs `x1` (`[n, d]`, or `[n]` for one feature) and `x2` (`[m, d]`
    /// or `[m]`), as a node. Inputs with different feature counts, or of another rank, and hyperparameters of the
    /// wrong shape are recorded as build errors.
    pub fn covariance(
        &self,
        x1: impl IntoDoubleVar<'b>,
        x2: impl IntoDoubleVar<'b>,
    ) -> DoubleVar<'b> {
        let b = self.builder;
        let (x1, x2) = (x1.into_var(b), x2.into_var(b));
        let (Some((x1, n, d1)), Some((x2, m, d2))) =
            (as_matrix(b, x1, "x1"), as_matrix(b, x2, "x2"))
        else {
            return b.scalar(0.0);
        };
        if d1 != d2 {
            return fail(b, KERNEL, format!("x1 has {d1} features but x2 has {d2}"));
        }
        self.matrix_covariance(x1, x2, Shapes { n, m, d: d1 })
    }

    fn matrix_covariance(
        &self,
        x1: DoubleVar<'b>,
        x2: DoubleVar<'b>,
        shape: Shapes,
    ) -> DoubleVar<'b> {
        let b = self.builder;
        let d = shape.d;
        let scaled_squared_distance = |length_scale: DoubleVar<'b>| {
            let scaled = pairwise_differences(x1, x2, shape) / length_scale;
            (scaled * scaled).sum_axes(&[-1])
        };
        match &self.kind {
            &Kind::Stationary {
                form,
                amplitude,
                length_scale,
            } => {
                if !(check_parameter(b, amplitude, "the amplitude", d, false)
                    && check_parameter(b, length_scale, "the length scale", d, true))
                {
                    return b.scalar(0.0);
                }
                let r2 = scaled_squared_distance(length_scale);
                let distance = || r2.maximum(MIN_SQUARED_DISTANCE).sqrt();
                let correlation = match form {
                    Form::ExpQuad => (r2 * -0.5).exp(),
                    Form::Matern12 => (-distance()).exp(),
                    Form::Matern32 => {
                        let a = distance() * 3f64.sqrt();
                        (a + 1.0) * (-a).exp()
                    }
                    Form::Matern52 => {
                        let a = distance() * 5f64.sqrt();
                        (a + 1.0 + r2 * (5.0 / 3.0)) * (-a).exp()
                    }
                    Form::Matern(nu) => (distance() * (2.0 * nu).sqrt()).matern_correlation(nu),
                };
                // exactly 1 at (clamped) zero distance
                let correlation = match form {
                    Form::ExpQuad => correlation,
                    _ => b.where_(r2.le(MIN_SQUARED_DISTANCE), b.scalar(1.0), correlation),
                };
                correlation * (amplitude * amplitude)
            }
            &Kind::RationalQuadratic {
                amplitude,
                length_scale,
                alpha,
            } => {
                if !(check_parameter(b, amplitude, "the amplitude", d, false)
                    && check_parameter(b, length_scale, "the length scale", d, true)
                    && check_parameter(b, alpha, "alpha", d, false))
                {
                    return b.scalar(0.0);
                }
                let base = scaled_squared_distance(length_scale) / (alpha * 2.0) + 1.0;
                base.pow(-alpha) * (amplitude * amplitude)
            }
            &Kind::Periodic {
                amplitude,
                length_scale,
                period,
            } => {
                if !(check_parameter(b, amplitude, "the amplitude", d, false)
                    && check_parameter(b, length_scale, "the length scale", d, true)
                    && check_parameter(b, period, "the period", d, true))
                {
                    return b.scalar(0.0);
                }
                let sines =
                    (pairwise_differences(x1, x2, shape) * PI / period).sin() / length_scale;
                ((sines * sines).sum_axes(&[-1]) * -2.0).exp() * (amplitude * amplitude)
            }
            &Kind::Linear {
                bias_amplitude,
                slope_amplitude,
                shift,
            } => {
                if !(check_parameter(b, bias_amplitude, "the bias amplitude", d, false)
                    && check_parameter(b, slope_amplitude, "the slope amplitude", d, false)
                    && check_parameter(b, shift, "the shift", d, true))
                {
                    return b.scalar(0.0);
                }
                let dot = (x1 - shift).matmul_transposed(x2 - shift, false, true);
                dot * (slope_amplitude * slope_amplitude) + bias_amplitude * bias_amplitude
            }
            Kind::Sum(left, right) => {
                left.matrix_covariance(x1, x2, shape) + right.matrix_covariance(x1, x2, shape)
            }
            Kind::Product(left, right) => {
                left.matrix_covariance(x1, x2, shape) * right.matrix_covariance(x1, x2, shape)
            }
        }
    }
}

/// The sum of two kernels, k(x, x') = k₁(x, x') + k₂(x, x').
impl<'b> Add for Kernel<'b> {
    type Output = Kernel<'b>;
    fn add(self, other: Kernel<'b>) -> Kernel<'b> {
        Kernel::new(self.builder, Kind::Sum(Box::new(self), Box::new(other)))
    }
}

/// The product of two kernels, k(x, x') = k₁(x, x') · k₂(x, x').
impl<'b> Mul for Kernel<'b> {
    type Output = Kernel<'b>;
    fn mul(self, other: Kernel<'b>) -> Kernel<'b> {
        Kernel::new(self.builder, Kind::Product(Box::new(self), Box::new(other)))
    }
}

/// A Gaussian process prior f ~ GP(m, k) with Gaussian observation noise, y = f(x) + ε, ε ~ N(0, noise²), which
/// builds its marginal likelihood and predictive distribution as graph nodes.
///
/// The mean defaults to zero, the noise to zero and the jitter to [`DEFAULT_JITTER`]. The jitter is added to the
/// diagonal of the training covariance as well as noise², so that its Cholesky factorisation succeeds when the
/// noise is zero or tiny.
///
/// `F` is the type of the mean function set by [`with_mean_fn`](Self::with_mean_fn).
#[derive(Clone)]
pub struct GaussianProcess<'b, F = fn(DoubleVar<'b>) -> DoubleVar<'b>> {
    kernel: Kernel<'b>,
    constant_mean: Option<DoubleVar<'b>>,
    mean_fn: Option<F>,
    noise: Option<DoubleVar<'b>>,
    jitter: f64,
}

impl<F> fmt::Debug for GaussianProcess<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GaussianProcess")
            .field("kernel", &self.kernel)
            .field("constant_mean", &self.constant_mean)
            .field("mean_fn", &self.mean_fn.as_ref().map(|_| "<function>"))
            .field("noise", &self.noise)
            .field("jitter", &self.jitter)
            .finish()
    }
}

/// The predictive distribution of a GP's latent function at new inputs: a multivariate Gaussian with this mean
/// (`[m]`) and covariance (`[m, m]`). It excludes the observation noise; add noise² to the diagonal for the
/// distribution of new observations.
#[derive(Clone, Copy, Debug)]
pub struct GpPrediction<'b> {
    /// The predictive mean, `[m]`.
    pub mean: DoubleVar<'b>,
    /// The predictive covariance, `[m, m]`.
    pub covariance: DoubleVar<'b>,
}

impl<'b> GpPrediction<'b> {
    /// The predictive variance at each new input, `[m]` (the diagonal of the covariance).
    pub fn variance(&self) -> DoubleVar<'b> {
        self.covariance.diag_part()
    }
}

impl<'b> GaussianProcess<'b> {
    /// A zero-mean, noise-free Gaussian process with this kernel.
    pub fn new(kernel: Kernel<'b>) -> Self {
        Self {
            kernel,
            constant_mean: None,
            mean_fn: None,
            noise: None,
            jitter: DEFAULT_JITTER,
        }
    }
}

impl<'b, F: Fn(DoubleVar<'b>) -> DoubleVar<'b>> GaussianProcess<'b, F> {
    /// With a constant mean, a scalar node or value (it may be a latent variable). Replaces any mean function.
    pub fn with_mean(mut self, mean: impl IntoDoubleVar<'b>) -> Self {
        self.constant_mean = Some(mean.into_var(self.kernel.builder));
        self.mean_fn = None;
        self
    }

    /// With a mean function m(x), called with the inputs as given (`[n]` or `[n, d]`) and returning a scalar or
    /// one mean per input (`[n]`). It may use other nodes of the builder, e.g. latent coefficients. Replaces any
    /// constant mean.
    pub fn with_mean_fn<G: Fn(DoubleVar<'b>) -> DoubleVar<'b>>(
        self,
        mean: G,
    ) -> GaussianProcess<'b, G> {
        GaussianProcess {
            kernel: self.kernel,
            constant_mean: None,
            mean_fn: Some(mean),
            noise: self.noise,
            jitter: self.jitter,
        }
    }

    /// With observation noise of this standard deviation, a scalar node or value (it may be a latent variable).
    pub fn with_noise(mut self, noise: impl IntoDoubleVar<'b>) -> Self {
        self.noise = Some(noise.into_var(self.kernel.builder));
        self
    }

    /// With this jitter (default [`DEFAULT_JITTER`]) added to the diagonal of the training covariance. It must be
    /// finite and non-negative, or building the marginal or predictive records an error.
    pub fn with_jitter(mut self, jitter: f64) -> Self {
        self.jitter = jitter;
        self
    }

    /// The kernel.
    pub fn kernel(&self) -> &Kernel<'b> {
        &self.kernel
    }

    /// The jitter added to the diagonal of the training covariance.
    pub fn jitter(&self) -> f64 {
        self.jitter
    }

    fn fail(&self, message: String) -> DoubleVar<'b> {
        fail(self.kernel.builder, GP, message)
    }

    /// The number of points in training inputs, or None (with an error recorded) if there are none or they are not
    /// a vector or matrix.
    fn points(&self, x: DoubleVar<'b>, name: &str) -> Option<usize> {
        match x.shape()[..] {
            [0] | [0, _] => {
                self.fail(format!("{name} has no points"));
                None
            }
            [n] | [n, _] => Some(n),
            ref shape => {
                self.fail(format!(
                    "{name} must be a vector [n] or a matrix [n, features], not {shape:?}"
                ));
                None
            }
        }
    }

    /// Whether `var` is a vector of `n` elements; records an error if not.
    fn check_vector(&self, var: DoubleVar<'b>, name: &str, n: usize) -> bool {
        let shape = var.shape();
        if shape == [n] {
            return true;
        }
        self.fail(format!(
            "{name} must be a vector of {n} elements, not {shape:?}"
        ));
        false
    }

    /// The mean at inputs `x` of `n` points, `[n]`.
    fn mean_at(&self, x: DoubleVar<'b>, n: usize) -> Option<DoubleVar<'b>> {
        let b = self.kernel.builder;
        let value = match (&self.mean_fn, self.constant_mean) {
            (Some(mean), _) => mean(x),
            (None, Some(mean)) => mean,
            (None, None) => return Some(b.constant(DoubleTensor::zeros(&[n]))),
        };
        let shape = value.shape();
        if shape.is_empty() || shape == [n] {
            Some(value.broadcast_to(&[n]))
        } else {
            self.fail(format!(
                "the mean must be a scalar or one value per input ([{n}]), not {shape:?}"
            ));
            None
        }
    }

    /// K(x, x) + (noise² + jitter)·I for inputs of `n` points.
    fn training_covariance(&self, x: DoubleVar<'b>, n: usize) -> Option<DoubleVar<'b>> {
        let b = self.kernel.builder;
        if !(self.jitter.is_finite() && self.jitter >= 0.0) {
            self.fail(format!(
                "the jitter must be finite and non-negative, not {}",
                self.jitter
            ));
            return None;
        }
        let diagonal = match self.noise {
            Some(noise) => {
                let shape = noise.shape();
                if !shape.is_empty() {
                    self.fail(format!("the noise must be a scalar, not {shape:?}"));
                    return None;
                }
                noise * noise + self.jitter
            }
            None => b.scalar(self.jitter),
        };
        let identity = match DoubleTensor::eye(n) {
            Ok(identity) => b.constant(identity),
            Err(e) => {
                b.record(e.into());
                return None;
            }
        };
        Some(self.kernel.covariance(x, x) + identity * diagonal)
    }

    /// The (unobserved) multivariate Gaussian over the noisy values y = f(x) + ε at inputs `x` (`[n]` or
    /// `[n, d]`): mean m(x) and covariance K(x, x) + (noise² + jitter)·I. Use it to sample from the prior, or
    /// observe it yourself; [`marginal`](Self::marginal) does both in one call.
    pub fn prior(&self, x: impl IntoDoubleVar<'b>) -> DoubleVar<'b> {
        let b = self.kernel.builder;
        let x = x.into_var(b);
        let Some(n) = self.points(x, "x") else {
            return b.scalar(0.0);
        };
        match (self.mean_at(x, n), self.training_covariance(x, n)) {
            (Some(mean), Some(covariance)) => b.multivariate_gaussian(mean, covariance),
            _ => b.scalar(0.0),
        }
    }

    /// The multivariate Gaussian over the targets `y` (`[n]`) at inputs `x` (`[n]` or `[n, d]`), observed to be
    /// `y`. Its log density, which joins the model's log likelihood, is the GP marginal likelihood
    /// ln N(y; m(x), K(x, x) + (noise² + jitter)·I).
    pub fn marginal(&self, x: impl IntoDoubleVar<'b>, y: impl Into<DoubleTensor>) -> DoubleVar<'b> {
        let b = self.kernel.builder;
        let (x, y) = (x.into_var(b), y.into());
        let Some(n) = self.points(x, "x") else {
            return b.scalar(0.0);
        };
        if y.shape() != [n] {
            return self.fail(format!(
                "y must be a vector of {n} elements (one per input), not {:?}",
                y.shape()
            ));
        }
        let observed = self.prior(x);
        b.observe(observed, y);
        observed
    }

    /// The predictive distribution of the latent function at `x_new` (`[m]` or `[m, d]`), given targets `y_train`
    /// (`[n]`, a constant or a node) at `x_train` (`[n]` or `[n, d]`). With L the Cholesky factor of
    /// K = K(x_train, x_train) + (noise² + jitter)·I and K* = K(x_train, x_new):
    ///
    /// - mean = m(x_new) + K*ᵀ L⁻ᵀ L⁻¹ (y_train − m(x_train));
    /// - covariance = K(x_new, x_new) − VᵀV, with V = L⁻¹ K*.
    ///
    /// Both use triangular solves against L; no matrix is inverted. With zero noise the predictive at the training
    /// inputs returns the targets with (up to the jitter) zero variance.
    pub fn predict(
        &self,
        x_train: impl IntoDoubleVar<'b>,
        y_train: impl IntoDoubleVar<'b>,
        x_new: impl IntoDoubleVar<'b>,
    ) -> GpPrediction<'b> {
        let b = self.kernel.builder;
        let (x_train, y_train, x_new) =
            (x_train.into_var(b), y_train.into_var(b), x_new.into_var(b));
        let failed = || GpPrediction {
            mean: b.scalar(0.0),
            covariance: b.scalar(0.0),
        };
        let Some(n) = self.points(x_train, "x_train") else {
            return failed();
        };
        let m = match x_new.shape()[..] {
            [m] | [m, _] => m,
            ref shape => {
                self.fail(format!(
                    "x_new must be a vector [m] or a matrix [m, features], not {shape:?}"
                ));
                return failed();
            }
        };
        if !self.check_vector(y_train, "y_train", n) {
            return failed();
        }
        let (Some(train_mean), Some(new_mean), Some(covariance)) = (
            self.mean_at(x_train, n),
            self.mean_at(x_new, m),
            self.training_covariance(x_train, n),
        ) else {
            return failed();
        };
        let cholesky = covariance.cholesky();
        let solve =
            |rhs: DoubleVar<'b>, transpose: bool| cholesky.solve_triangular(rhs, true, transpose);
        // α = K⁻¹ (y − m) as a column [n, 1]
        let residual = (y_train - train_mean).reshape(&[n as isize, 1]);
        let alpha = solve(solve(residual, false), true);
        let cross = self.kernel.covariance(x_train, x_new);
        let mean = new_mean
            + cross
                .matmul_transposed(alpha, true, false)
                .reshape(&[m as isize]);
        let v = solve(cross, false);
        let covariance = self.kernel.covariance(x_new, x_new) - v.matmul_transposed(v, true, false);
        GpPrediction { mean, covariance }
    }
}

/// The GP marginal likelihood in one call: an observed multivariate Gaussian over targets `y` (`[n]`) at inputs
/// `x` (`[n]` or `[n, d]`) with a constant `mean`, the `kernel`, observation noise standard deviation `noise` and
/// the default jitter. See [`GaussianProcess::marginal`].
pub fn gp_marginal<'b>(
    x: impl IntoDoubleVar<'b>,
    y: impl Into<DoubleTensor>,
    mean: impl IntoDoubleVar<'b>,
    kernel: &Kernel<'b>,
    noise: impl IntoDoubleVar<'b>,
) -> DoubleVar<'b> {
    GaussianProcess::new(kernel.clone())
        .with_mean(mean)
        .with_noise(noise)
        .marginal(x, y)
}

/// The GP predictive distribution in one call, with a constant `mean`, observation noise standard deviation
/// `noise` and the default jitter. See [`GaussianProcess::predict`].
pub fn gp_predict<'b>(
    x_train: impl IntoDoubleVar<'b>,
    y_train: impl IntoDoubleVar<'b>,
    x_new: impl IntoDoubleVar<'b>,
    mean: impl IntoDoubleVar<'b>,
    kernel: &Kernel<'b>,
    noise: impl IntoDoubleVar<'b>,
) -> GpPrediction<'b> {
    GaussianProcess::new(kernel.clone())
        .with_mean(mean)
        .with_noise(noise)
        .predict(x_train, y_train, x_new)
}
