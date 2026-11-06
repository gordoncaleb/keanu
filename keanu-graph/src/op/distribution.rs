//! Probabilistic nodes: a distribution whose parameters are the node's parents.

use super::hmm;
use super::mixture;
use super::nested::nested_kinds;
use crate::error::{GraphError, Result};
use crate::value::{DType, Value};
use keanu_distributions as d;
use keanu_distributions::KeanuRng;
use keanu_tensor::{DoubleTensor, Shape};
use std::borrow::Cow;
use std::sync::Arc;

/// The set of values a distribution gives positive density (or mass), which decides how a gradient based algorithm
/// maps the node to unconstrained space.
///
/// Bounds are tensors that broadcast with the node's value, so a distribution whose bounds are parameters (Uniform,
/// Triangular, Pareto) reports them element-wise. Bounds are reported as the parameters give them, valid or not (a
/// Uniform with `x_max <= x_min` reports that interval anyway). Whether a bound itself is in the support (Uniform's
/// `[x_min, x_max)`) is not recorded: it has probability zero for a continuous distribution.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Support {
    /// Every real number.
    Real,
    /// Real numbers greater than the lower bound.
    GreaterThan(DoubleTensor),
    /// Real numbers less than the upper bound.
    LessThan(DoubleTensor),
    /// Real numbers between the lower and the upper bound.
    Interval(DoubleTensor, DoubleTensor),
    /// Non-negative vectors summing to 1, over the value's last dimension.
    Simplex,
    /// Strictly increasing vectors, over the value's last dimension.
    Ordered,
    /// Vectors whose elements sum to zero, over the value's last dimension.
    SumToZero,
    /// Values whose elements sum to zero within each listed group of flat indices (disjoint), the elements in no
    /// group being unconstrained: an [ICAR](DistributionKind::Icar)'s support, a group per connected component of
    /// two or more areas.
    SumToZeroComponents(Arc<[Vec<usize>]>),
    /// Lower triangular Cholesky factors of correlation matrices, over the value's last two dimensions: a positive
    /// diagonal and rows of unit length.
    CorrelationCholesky,
    /// Symmetric positive definite matrices over the value's last two dimensions (covariance matrices, such as a
    /// Wishart's values).
    CovarianceMatrix,
    /// Lower triangular matrices with a positive diagonal over the value's last two dimensions: Cholesky factors of
    /// covariance matrices.
    CovarianceCholesky,
    /// Vectors of unit length over the value's last dimension, for distributions of directions such as the von
    /// Mises–Fisher. The transform is Stan's: an unconstrained vector normalised, with a standard normal density on
    /// it (not a bijection).
    UnitVector,
    /// Angles, for distributions of directions such as the von Mises: a value and the same value plus any multiple
    /// of 2π are the same point, so the value is mapped to unconstrained space with a circular transform.
    Circular,
    /// Integer or boolean values: the node is not continuous, so it has no unconstrained form.
    Discrete,
}

impl Support {
    /// Real numbers greater than a scalar lower bound.
    pub fn greater_than(lower: f64) -> Self {
        Support::GreaterThan(DoubleTensor::scalar(lower))
    }

    /// Real numbers less than a scalar upper bound.
    pub fn less_than(upper: f64) -> Self {
        Support::LessThan(DoubleTensor::scalar(upper))
    }

    /// Real numbers between scalar lower and upper bounds.
    pub fn interval(lower: f64, upper: f64) -> Self {
        Support::Interval(DoubleTensor::scalar(lower), DoubleTensor::scalar(upper))
    }
}

/// The distribution of a probabilistic node. The node's parents are the parameters, in the order listed.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DistributionKind {
    /// Parameters: `mu`, `sigma`. Support: real.
    Gaussian,
    /// Parameters: `sigma`. Support: greater than 0.
    HalfGaussian,
    /// Parameters: `location`, `scale`. Support: real.
    Cauchy,
    /// Parameters: `scale`. Support: greater than 0.
    HalfCauchy,
    /// Parameters: `alpha`, `beta`. Support: the interval [0, 1].
    Beta,
    /// Parameters: `k (i64)`. Support: greater than 0.
    ChiSquared,
    /// Parameters: `concentration`; the value has the concentration's shape. Support: the simplex over the last
    /// dimension.
    Dirichlet,
    /// Parameters: `rate`. Support: greater than 0.
    Exponential,
    /// Parameters: `theta (scale)`, `k (shape)`. Support: greater than 0.
    Gamma,
    /// Parameters: `alpha`, `beta`. Support: greater than 0.
    InverseGamma,
    /// Parameters: `mu`, `beta`. Support: real.
    Laplace,
    /// Parameters: `mu`, `s`. Support: real.
    Logistic,
    /// Parameters: `mu`, `sigma`. Support: greater than 0.
    LogNormal,
    /// Parameters: `eta`; the value is a K × K lower triangular Cholesky factor of a correlation matrix (or a batch
    /// of them, shape [..., K, K], with `eta` broadcasting to the batch shape), K set by the node's shape. Support:
    /// correlation Cholesky factors. See [`keanu_distributions::LkjCholesky`].
    LkjCholesky,
    /// Parameters: `mu`, `covariance`; the value's last dimension is the event dimension. Support: real.
    MultivariateGaussian,
    /// Parameters: `location` (the minimum value), `scale` (the shape). Support: greater than `location`.
    Pareto,
    /// Parameters: `v (i64)`. Support: real.
    StudentT,
    /// Parameters: `x_min`, `x_max`, `c`. Support: the interval [`x_min`, `x_max`].
    Triangular,
    /// Parameters: `x_min`, `x_max`. Support: the interval [`x_min`, `x_max`).
    Uniform,
    /// Parameters: `alpha` (shape), `beta` (scale). Support: greater than 0.
    Weibull,
    /// Parameters: `sigma`; the value is a vector of N elements summing to zero (or a batch of them, shape [..., N],
    /// with `sigma` broadcasting to the batch shape), N set by the node's shape: N(0, σ² I) restricted to the
    /// zero-sum subspace. Support: sum to zero. See [`keanu_distributions::ZeroSumNormal`].
    ZeroSumNormal,
    /// Parameters: `mu` (mean direction), `kappa` (concentration). Support: circular (angles; the density is
    /// periodic with period 2π).
    VonMises,
    /// Parameters: `xi` (location), `omega` (scale), `alpha` (shape). Support: real. See
    /// [`keanu_distributions::SkewNormal`].
    SkewNormal,
    /// Parameters: `mu`, `sigma` (the normal part), `lambda` (the exponential part's rate). Support: real. See
    /// [`keanu_distributions::ExGaussian`].
    ExGaussian,
    /// Parameters: `mu` (location), `beta` (scale). Support: real. See [`keanu_distributions::Gumbel`].
    Gumbel,
    /// Parameters: `mu` (location), `sigma` (scale), `xi` (shape). Support: greater than μ − σ/ξ for ξ > 0, less
    /// than it for ξ < 0 and real for ξ = 0, element-wise. See [`keanu_distributions::GeneralizedExtremeValue`].
    GeneralizedExtremeValue,
    /// Parameters: `mu` (location, the threshold), `sigma` (scale), `xi` (shape). Support: greater than μ, and less
    /// than μ − σ/ξ for ξ < 0, element-wise. See [`keanu_distributions::GeneralizedPareto`].
    GeneralizedPareto,
    /// Parameters: `mu` (mean), `lambda` (shape). Support: greater than 0. See
    /// [`keanu_distributions::InverseGaussian`].
    InverseGaussian,
    /// Parameters: `a`, `b` (shapes). Support: the interval [0, 1]. See [`keanu_distributions::Kumaraswamy`].
    Kumaraswamy,
    /// Parameters: `mu`, `sigma` (of the logit). Support: the interval [0, 1]. See
    /// [`keanu_distributions::LogitNormal`].
    LogitNormal,
    /// Parameters: `nu` (degrees of freedom, real), `sigma` (scale). Support: greater than 0. See
    /// [`keanu_distributions::HalfStudentT`].
    HalfStudentT,
    /// Parameters: `mu` (location, the τ-quantile), `sigma` (scale), `tau` (skewness, in (0, 1)). Support: real. See
    /// [`keanu_distributions::AsymmetricLaplace`].
    AsymmetricLaplace,
    /// Parameters: `alpha` (scale), `beta` (shape). Support: greater than 0. See
    /// [`keanu_distributions::LogLogistic`].
    LogLogistic,
    /// Parameters: `nu` (noncentrality), `sigma` (scale). Support: greater than 0. See
    /// [`keanu_distributions::Rice`].
    Rice,
    /// Parameters: `theta` (scale), `k` (shape), `p` (power). Support: greater than 0. See
    /// [`keanu_distributions::GeneralizedGamma`].
    GeneralizedGamma,
    /// Parameters: `nu` (degrees of freedom, real), `mu` [..., K], `scale` (the scale matrix Σ, [..., K, K]); the
    /// value's last axis is the event (K elements), with any leading axes a batch that the parameters' leading axes
    /// broadcast to. Support: real. See [`keanu_distributions::MultivariateStudentT`].
    MultivariateStudentT,
    /// Parameters: `nu`, `mu` [..., K], `scale_cholesky` (the lower triangular Cholesky factor L of the scale
    /// matrix, [..., K, K]). Support: real. See [`keanu_distributions::MultivariateStudentTCholesky`].
    MultivariateStudentTCholesky,
    /// Parameters: `nu` (degrees of freedom, > K − 1), `scale` [..., K, K]; the value is a symmetric positive definite
    /// matrix [..., K, K]. Support: covariance matrices. See [`keanu_distributions::Wishart`].
    Wishart,
    /// Parameters: `nu`, `scale_cholesky` [..., K, K]; the value is the Cholesky factor of a Wishart matrix,
    /// [..., K, K]. Support: covariance Cholesky factors. See [`keanu_distributions::WishartCholesky`].
    WishartCholesky,
    /// Parameters: `nu` (> K − 1), `scale` [..., K, K]; the value is a symmetric positive definite matrix
    /// [..., K, K]. Support: covariance matrices. See [`keanu_distributions::InverseWishart`].
    InverseWishart,
    /// Parameters: `nu`, `scale_cholesky` [..., K, K]; the value is the Cholesky factor of an inverse Wishart matrix.
    /// Support: covariance Cholesky factors. See [`keanu_distributions::InverseWishartCholesky`].
    InverseWishartCholesky,
    /// Parameters: `mu` [..., n, p], `row_covariance` [..., n, n], `column_covariance` [..., p, p]; the value is an
    /// n × p matrix [..., n, p]. Support: real. See [`keanu_distributions::MatrixNormal`].
    MatrixNormal,
    /// Parameters: `mu` (the mean direction, a unit vector [..., p], p >= 2), `kappa` (the concentration); the value
    /// is a unit vector [..., p]. Support: unit vectors. See [`keanu_distributions::VonMisesFisher`].
    VonMisesFisher,
    /// Parameters: `a` (boundary separation), `t0` (non-decision time), `w` (relative starting point), `v` (drift),
    /// `sv` (the drift's standard deviation across trials, 0 for the four-parameter model), `boundary (i64)` (the
    /// boundary reached, 0 the lower and 1 the upper, observed data); the value is the response time. The Wiener
    /// first-passage time (drift-diffusion model) density at the given boundary, with Stan's parameterisation.
    /// Support: greater than `t0`. See [`keanu_distributions::Wiener`] and
    /// [`GraphBuilder::wiener`](crate::GraphBuilder::wiener).
    Wiener,
    /// No parameters: a kernel density estimate from fixed samples. Support: real.
    Kde(Arc<d::Kde>),
    /// Parameters: `sigma` (a scalar); the value has one element per area of the neighbourhood, shape `[n]`: the
    /// intrinsic conditional autoregressive distribution, −½ Σ_{i~j} (φᵢ − φⱼ)² / σ² − (n − c) ln σ. Support:
    /// sum to zero on each connected component of two or more areas. See [`keanu_distributions::Icar`] and
    /// [`GraphBuilder::icar`](crate::GraphBuilder::icar).
    Icar(Arc<d::Adjacency>),
    /// Parameters: `tau`, `alpha` (scalars); the value has one element per area, shape `[n]`: the proper
    /// conditional autoregressive distribution, a zero-mean multivariate normal with precision τ(D − αW). Every
    /// area must have a neighbour. Support: real. See [`keanu_distributions::Car`] and
    /// [`GraphBuilder::car`](crate::GraphBuilder::car).
    Car(Arc<d::Adjacency>),
    /// Parameters: `probability of true` -> bool. Support: discrete.
    Bernoulli,
    /// Parameters: `p`, `n (i64)` -> i64. Support: discrete.
    Binomial,
    /// Parameters: `probability of each category...` -> i64 category index. Support: discrete.
    Categorical,
    /// Parameters: `p` -> i64. Support: discrete.
    Geometric,
    /// Parameters: `n (i64)`, `p` -> i64. Support: discrete.
    Multinomial,
    /// Parameters: `mu` -> i64. Support: discrete.
    Poisson,
    /// Parameters: `mu` (mean), `alpha` (dispersion) -> i64, with variance mu + mu²/alpha. Support: discrete.
    NegativeBinomial,
    /// Parameters: `eta`, `cutpoints` (strictly increasing along the last axis, K - 1 of them) -> i64 category index
    /// in 0..K. The value's shape is the broadcast of eta's shape and the cutpoints' leading axes. Support:
    /// discrete.
    OrderedLogistic,
    /// Parameters: `x_min (i64)`, `x_max (i64)` -> i64 in [x_min, x_max). Support: discrete.
    UniformInt,
    /// Parameters: `eta`, `cutpoints` (strictly increasing along the last axis, K - 1 of them) -> i64 category index
    /// in 0..K, as [`OrderedLogistic`](Self::OrderedLogistic) with the standard normal CDF for the logistic one.
    /// Support: discrete. See [`keanu_distributions::OrderedProbit`].
    OrderedProbit,
    /// Parameters: `population (i64)` N, `successes (i64)` K, `draws (i64)` n -> i64, the successes among n draws
    /// without replacement. Support: discrete. See [`keanu_distributions::HyperGeometric`].
    HyperGeometric,
    /// Parameters: `mu1`, `mu2` -> i64, the difference of Poisson(mu1) and Poisson(mu2) counts. Support: discrete.
    /// See [`keanu_distributions::Skellam`].
    Skellam,
    /// Parameters: `n (i64)`, `alpha` (the concentrations, [..., K]) -> i64 counts [..., K] summing to n. Support:
    /// discrete. See [`keanu_distributions::DirichletMultinomial`].
    DirichletMultinomial,
    /// Parameters: `worths` (positive, [..., K]) -> i64 ranking [..., K], a permutation of 0..K listing the items
    /// from first to last. Support: discrete. See [`keanu_distributions::PlackettLuce`].
    PlackettLuce,
    /// Parameters: `n (i64)`, `alpha`, `beta` -> i64 in 0..=n: a binomial count whose success probability is
    /// Beta(alpha, beta) distributed. Support: discrete. See [`keanu_distributions::BetaBinomial`].
    BetaBinomial,
    /// Parameters: `logit` (the log odds of true) -> bool. Support: discrete. See
    /// [`keanu_distributions::BernoulliLogit`].
    BernoulliLogit,
    /// Parameters: `logit` (the log odds of success), `n (i64)` -> i64. Support: discrete. See
    /// [`keanu_distributions::BinomialLogit`].
    BinomialLogit,
    /// Parameters: `log_mu` (the log of the mean) -> i64. Support: discrete. See
    /// [`keanu_distributions::PoissonLog`].
    PoissonLog,
    /// Parameters: `logit of each category...` (unnormalised log probabilities, all of the same shape) -> i64
    /// category index. Support: discrete. See [`keanu_distributions::CategoricalLogit`].
    CategoricalLogit,
    /// A user-defined distribution; its parameters are whatever nodes it was given. Support: whatever
    /// [`CustomDistribution::support`] returns.
    Custom(Arc<dyn CustomDistribution>),
    /// A simulator: its value is drawn by a user function of its parameters (whatever nodes it was given) and has
    /// no likelihood, so its log probability is [`GraphError::NoLikelihood`]. It can be sampled and observed, and
    /// its parameters inferred by approximate Bayesian computation. Support: real for f64 values, discrete
    /// otherwise. See [`Simulator`](super::Simulator) and
    /// [`GraphBuilder::simulator`](crate::GraphBuilder::simulator).
    Simulator(Arc<super::Simulator>),
    /// The base distribution restricted to `[lower, upper]` and renormalised by the probability it gives that
    /// range. Parameters: the base's parameters, then `lower`, `upper` (f64, either may be infinite). Support: the
    /// base's support intersected with the bounds (discrete for a discrete base). See
    /// [`GraphBuilder::truncated`](crate::GraphBuilder::truncated).
    Truncated(Arc<DistributionKind>),
    /// The base distribution with values at or beyond the bounds recorded as the bounds: a value equal to `lower`
    /// has the probability P(X <= lower) and a value equal to `upper` the probability P(X >= upper). Parameters:
    /// the base's parameters, then `lower`, `upper` (f64, either may be infinite). Support: as for
    /// [`Truncated`](Self::Truncated). See [`GraphBuilder::censored`](crate::GraphBuilder::censored).
    Censored(Arc<DistributionKind>),
    /// A mixture of C distributions of the component kind: density Σ_c w_c p_c(x). Parameters: the component's
    /// parameters, each with a trailing component axis of length C (or 1, or no axes at all, to share the value
    /// between components), then `weights` (non-negative and summing to 1 along their last axis, of length C).
    /// The component is an element-wise univariate kind (a distribution of one value per element, or a truncated
    /// or censored one); the value type is the component's. Support: the component's support widened to hold
    /// every component's. See [`GraphBuilder::mixture`](crate::GraphBuilder::mixture).
    Mixture(Arc<DistributionKind>),
    /// The inner count distribution with extra zeros: a structural zero with probability ψ, else a draw from the
    /// inner kind, so P(0) = ψ + (1 − ψ) p(0) and P(k) = (1 − ψ) p(k) for k ≠ 0. Parameters: the inner kind's
    /// parameters, then `psi` (in [0, 1]). The inner kind is an element-wise discrete kind with integer values
    /// (a Poisson, negative binomial, binomial, ...); the value type is i64. Support: discrete. See
    /// [`GraphBuilder::zero_inflated`](crate::GraphBuilder::zero_inflated).
    ZeroInflated(Arc<DistributionKind>),
    /// A hurdle model: zero with probability ψ, else a draw from the inner kind restricted to positive values. For
    /// a count inner kind P(0) = ψ and P(k) = (1 − ψ) p(k) / (1 − p(0)) for k > 0; for a positive continuous inner
    /// kind (a gamma, log-normal, ...) a value of exactly 0 has probability ψ and a positive value the density
    /// (1 − ψ) p(x). Parameters: the inner kind's parameters, then `psi` (in [0, 1]); the value type is the inner
    /// kind's. Support: discrete for a count, the inner kind's for a continuous one. See
    /// [`GraphBuilder::hurdle`](crate::GraphBuilder::hurdle).
    Hurdle(Arc<DistributionKind>),
    /// A hidden Markov model whose observations are drawn from the emission kind with the hidden state's
    /// parameters. Parameters: the emission's parameters, each with a trailing state axis of length K (or 1, or no
    /// axes at all, to share the value between states), then `initial` (the first state's probabilities, \[K\]) and
    /// `transition` (the transition probabilities, [K, K], row i the probabilities of moving from state i). Both
    /// are probabilities, not log probabilities: non-negative, summing to 1 along their last axis (within 1e-5).
    /// The value is the observation series, [T, ...] with time first; the emission is an element-wise univariate
    /// kind, as for a mixture, and the value type is the emission's. Support: the emission's support widened to
    /// hold every state's. See [`GraphBuilder::hidden_markov_model`](crate::GraphBuilder::hidden_markov_model).
    HiddenMarkovModel(Arc<DistributionKind>),
    /// Parameters: `initial_mean` \[d\], `initial_covariance` [d, d], `transition_matrix` [d, d],
    /// `transition_covariance` [d, d], `observation_matrix` [m, d], `observation_covariance` [m, m]; the value is
    /// the observation series, [T, m]. The covariances must be symmetric positive semidefinite. Support: real. See
    /// [`keanu_distributions::LinearGaussianStateSpace`] and
    /// [`GraphBuilder::linear_gaussian_state_space`](crate::GraphBuilder::linear_gaussian_state_space).
    LinearGaussianStateSpace,
    /// Parameters: `mu` (the drift), `sigma`, `init_mu`, `init_sigma`; the value is a series [..., T] (time last,
    /// T > 0, set by the node's shape) whose first value is N(`init_mu`, `init_sigma`) and each later one the one
    /// before plus N(`mu`, `sigma`). Each parameter broadcasts to the batch shape [...]. Support: real. See
    /// [`keanu_distributions::GaussianRandomWalk`].
    GaussianRandomWalk,
    /// Parameters: `constant`, `rho` (the coefficients of lags 1 to p, [..., p]), `sigma`, `init_mu`, `init_sigma`;
    /// the value is a series [..., T] (time last) whose first p values are N(`init_mu`, `init_sigma`) and each
    /// later one N(c + Σᵢ ρᵢ x_{t−i}, σ). Every parameter (`rho` without its lag axis) broadcasts to the batch
    /// shape. Support: real. See [`keanu_distributions::Autoregressive`].
    Autoregressive,
    /// Parameters: `omega`, `alpha`, `beta`, `initial_vol`; the value is a series [..., T] (time last) with
    /// r_t ~ N(0, σ_t), σ_0 = `initial_vol` and σ_t² = ω + α r_{t−1}² + β σ_{t−1}². Each parameter broadcasts to
    /// the batch shape; parameters outside ω > 0, α, β ≥ 0, α + β < 1, `initial_vol` > 0 make the value impossible.
    /// Support: real. See [`keanu_distributions::Garch11`].
    Garch11,
}

/// A user-defined distribution for a probabilistic node.
///
/// Only [`log_prob`](Self::log_prob) is required. Implement [`log_prob_gradient`](Self::log_prob_gradient) to use
/// the node with gradient based algorithms (NUTS, gradient optimizers), and [`sample`](Self::sample) to sample it
/// from its prior (forward sampling, prior proposals, evaluating a graph without a value for it).
pub trait CustomDistribution: Send + Sync + std::fmt::Debug {
    /// The type of the node's value. f64 by default.
    fn value_type(&self) -> DType {
        DType::F64
    }

    /// The log probability of `value` (summed over its elements) given the parameters' values.
    fn log_prob(&self, value: &Value, parameters: &[&Value]) -> std::result::Result<f64, String>;

    /// The gradient of [`log_prob`](Self::log_prob) with respect to the value (f64 values only) and each f64
    /// parameter, shaped like the value and the parameters. `None` (the default) means the distribution has no
    /// gradient.
    fn log_prob_gradient(
        &self,
        value: &Value,
        parameters: &[&Value],
    ) -> std::result::Result<Option<CustomGradient>, String> {
        let _ = (value, parameters);
        Ok(None)
    }

    /// Draws a value of the given shape. By default the distribution can't be sampled.
    fn sample(
        &self,
        shape: &[usize],
        parameters: &[&Value],
        rng: &mut KeanuRng,
    ) -> std::result::Result<Value, String> {
        let _ = (shape, parameters, rng);
        Err("this custom distribution cannot be sampled".into())
    }

    /// The support of the distribution given the parameters' values. By default every real number for f64 values
    /// and [`Support::Discrete`] for i64 and bool values. Override it for a bounded distribution so that gradient
    /// based algorithms keep the node inside its support.
    fn support(&self, parameters: &[&Value]) -> std::result::Result<Support, String> {
        let _ = parameters;
        Ok(match self.value_type() {
            DType::F64 => Support::Real,
            DType::I64 | DType::Bool => Support::Discrete,
        })
    }
}

/// The error for the log probability of a simulator, which has none. The node is named where it is known
/// ([`node_log_prob`](crate::node_log_prob) and the model's log probabilities).
fn no_likelihood() -> GraphError {
    GraphError::NoLikelihood("the node".into())
}

/// The gradient of a [`CustomDistribution`]'s log probability.
#[derive(Debug, Clone, Default)]
pub struct CustomGradient {
    /// With respect to the value, if it is f64 and differentiable.
    pub value: Option<keanu_tensor::DoubleTensor>,
    /// With respect to each parameter, in order (`None` where there is no gradient).
    pub parameters: Vec<Option<keanu_tensor::DoubleTensor>>,
}

/// Per-parameter log prob gradients, inline for up to four parameters.
pub(crate) type ParameterGradients = smallvec::SmallVec<[Option<DoubleTensor>; 4]>;

/// A node's log prob gradient with respect to its value (None for non-f64 values and distributions without one)
/// and to each parameter (None where there is no derivative).
pub(crate) type LogProbGradientParts = (Option<DoubleTensor>, ParameterGradients);

/// The first bracketed tokens if there are any (an optional override), else the second.
macro_rules! override_or {
    ([] [$($default:tt)*]) => { $($default)* };
    ([$($given:tt)+] [$($default:tt)*]) => { $($given)+ };
}

/// The tensor of a value of the given [`DType`] variant.
macro_rules! tensor {
    (F64, $value:expr) => {
        $value.f64()?
    };
    (I64, $value:expr) => {
        $value.i64()?
    };
    (Bool, $value:expr) => {
        $value.bool()?
    };
}

/// A log prob gradient with respect to a value or parameter of the given [`DType`] variant: only f64 ones have one.
macro_rules! gradient {
    (F64, $gradient:expr) => {
        Some($gradient)
    };
    ($dtype:ident, $gradient:expr) => {
        None
    };
}

/// Describes each distribution with a fixed parameter list once, as
/// `Kind "name" (parameter: dtype, ...) -> value dtype => constructor`, and generates [`DistributionKind`]'s name,
/// types, sampling, log probability and gradient from the table (the remaining kinds are written out here).
///
/// The parameters are bound to references to their tensors, which the distribution borrows (nothing is copied). By
/// default a distribution is sampled with `sample(shape, rng)`, and its gradient is the `d_log_prob` field named after each f64 parameter, and `x`
/// for an f64 value. A row can override either: `sample = f` is called as `f(&distribution, shape, rng)` and
/// `gradient = f` as `f(&distribution, x)` (both non-capturing closures or functions). `fused = f`, called as
/// `f(&distribution, x)`, computes the summed log probability and the gradient together for
/// `log_prob_and_gradient`; it must return exactly what the separate methods do. Every row states its support as
/// `support = expression`, an expression of the parameters' tensors (bound to the parameters' names) giving a
/// [`Support`].
///
/// A univariate distribution with `log_cdf`, `log_ccdf`, `d_log_cdf` and `d_log_ccdf` methods (which can then be
/// truncated or censored) says so with `cdf = inverse` if it also has `inverse_cdf` and `inverse_ccdf` (so a
/// truncation is sampled by inverting the CDF), or `cdf = rejection` if not (so it is sampled by rejection). A
/// distribution of one value per element whose log density and gradient are element-wise but which has no CDF says
/// `cdf = none`: it can't be truncated or censored, but it can be a mixture component.
macro_rules! distributions {
    ($(
        $kind:ident $name:literal ($($param:ident: $ptype:ident),+) -> $vtype:ident => $build:expr,
        support = $support:expr
        $(, cdf = $cdf:ident)?
        $(, sample = $sample:expr)? $(, gradient = $gradient:expr)? $(, fused = $fused:expr)?;
    )+) => {
        impl DistributionKind {
            /// The parameter types, or None for a variable number of f64 parameters (Categorical).
            pub(crate) fn parameter_types(&self) -> Option<Cow<'static, [DType]>> {
                match self {
                    $(DistributionKind::$kind => Some(Cow::Borrowed(&[$(DType::$ptype),+])),)+
                    DistributionKind::Kde(_) => Some(Cow::Borrowed(&[])),
                    DistributionKind::Icar(_) => Some(Cow::Borrowed(&[DType::F64])),
                    DistributionKind::Car(_) => Some(Cow::Borrowed(&[DType::F64, DType::F64])),
                    DistributionKind::Categorical
                    | DistributionKind::CategoricalLogit
                    | DistributionKind::Custom(_)
                    | DistributionKind::Simulator(_) => None,
                    nested_kinds!() => {
                        let (nested, inner) = self.nested_parts();
                        nested.parameter_types(inner)
                    }
                }
            }

            /// The parameter names, in parent order (e.g. `["mu", "sigma"]` for a Gaussian), or None for a variable
            /// number of parameters (Categorical, CategoricalLogit, Custom, Simulator). A kind built around another one has its
            /// base's names followed by its own (`lower` and `upper` for a truncated or censored kind, `weights` for
            /// a mixture, `initial` and `transition` for a hidden Markov model, `psi` for a zero-inflated or hurdle
            /// kind).
            pub fn parameter_names(&self) -> Option<Cow<'static, [&'static str]>> {
                match self {
                    $(DistributionKind::$kind => Some(Cow::Borrowed(&[$(stringify!($param)),+])),)+
                    DistributionKind::Kde(_) => Some(Cow::Borrowed(&[])),
                    DistributionKind::Icar(_) => Some(Cow::Borrowed(&["sigma"])),
                    DistributionKind::Car(_) => Some(Cow::Borrowed(&["tau", "alpha"])),
                    DistributionKind::Categorical
                    | DistributionKind::CategoricalLogit
                    | DistributionKind::Custom(_)
                    | DistributionKind::Simulator(_) => None,
                    nested_kinds!() => {
                        let (nested, inner) = self.nested_parts();
                        nested.parameter_names(inner)
                    }
                }
            }

            /// Whether the distribution is univariate and has a CDF, so that it can be truncated or censored.
            pub(crate) fn has_cdf(&self) -> bool {
                match self {
                    $(DistributionKind::$kind => has_cdf!([$($cdf)?]),)+
                    _ => false,
                }
            }

            /// Whether the distribution has one value per element and an element-wise log density and gradient
            /// (see [`Self::univariate`]), so that it can be a mixture component.
            pub(crate) fn is_elementwise(&self) -> bool {
                match self {
                    $(DistributionKind::$kind => is_elementwise!([$($cdf)?]),)+
                    nested_kinds!() => self.nested_parts().0.is_elementwise(),
                    _ => false,
                }
            }

            /// `function` of an element-wise distribution at `x`, a value of the distribution's type, element-wise
            /// and broadcast with the parameters; with `gradient`, also its derivatives with respect to x and each
            /// parameter, element-wise. The log density works for every [element-wise](Self::is_elementwise) kind,
            /// the CDF and survival function only for those [with a CDF](Self::has_cdf).
            pub(crate) fn univariate(
                &self,
                function: Univariate,
                x: &Value,
                parameters: &[&Value],
                gradient: bool,
            ) -> Result<(DoubleTensor, Option<LogProbGradientParts>)> {
                Ok(match self {
                    $(DistributionKind::$kind => univariate!(
                        [$($cdf)?], self, function, x, parameters, gradient, $vtype,
                        [$($param: $ptype),+], [$($gradient)?], $build
                    ),)+
                    nested_kinds!() if function == Univariate::Density => {
                        let (nested, inner) = self.nested_parts();
                        match nested.elementwise(inner, x, parameters, gradient) {
                            Some(result) => result?,
                            None => return Err(self.no_cdf()),
                        }
                    }
                    _ => return Err(self.no_cdf()),
                })
            }

            /// The inverse CDF at probabilities `p` (or the inverse survival function if `upper`), element-wise
            /// and broadcast with the parameters, if the distribution has one in closed form.
            pub(crate) fn inverse_cdf(
                &self,
                p: &DoubleTensor,
                upper: bool,
                parameters: &[&Value],
            ) -> Result<Option<DoubleTensor>> {
                Ok(match self {
                    $(DistributionKind::$kind => inverse_cdf!(
                        [$($cdf)?], self, p, upper, parameters, [$($param: $ptype),+], $build
                    ),)+
                    _ => None,
                })
            }

            /// The type of the node's value.
            pub fn value_type(&self) -> DType {
                match self {
                    $(DistributionKind::$kind => DType::$vtype,)+
                    DistributionKind::Kde(_) | DistributionKind::Icar(_) | DistributionKind::Car(_) => DType::F64,
                    DistributionKind::Categorical | DistributionKind::CategoricalLogit => DType::I64,
                    DistributionKind::Custom(c) => c.value_type(),
                    DistributionKind::Simulator(s) => s.value_type(),
                    nested_kinds!() => self.nested_parts().1.value_type(),
                }
            }

            /// Every kind's [`name`](Self::name), the custom kind's included.
            pub const NAMES: &'static [&'static str] = &[
                $($name,)+
                "kde",
                "icar",
                "car",
                "categorical",
                "categorical_logit",
                "custom",
                "simulator",
                "truncated",
                "censored",
                "mixture",
                "hidden_markov_model",
                "zero_inflated",
                "hurdle",
            ];

            /// The [names](Self::name) of the kinds with a CDF, which can be truncated or censored.
            pub const CDF_NAMES: &'static [&'static str] = {
                const ROWS: &[(&str, bool)] = &[$(($name, has_cdf!([$($cdf)?]))),+];
                const COUNT: usize = {
                    let mut count = 0;
                    let mut i = 0;
                    while i < ROWS.len() {
                        if ROWS[i].1 {
                            count += 1;
                        }
                        i += 1;
                    }
                    count
                };
                const NAMES: [&str; COUNT] = {
                    let mut names = [""; COUNT];
                    let (mut i, mut j) = (0, 0);
                    while i < ROWS.len() {
                        if ROWS[i].1 {
                            names[j] = ROWS[i].0;
                            j += 1;
                        }
                        i += 1;
                    }
                    names
                };
                &NAMES
            };

            /// The kind's name in snake case, as in error messages (`"gaussian"`, `"truncated"`).
            pub fn name(&self) -> &'static str {
                match self {
                    $(DistributionKind::$kind => $name,)+
                    DistributionKind::Kde(_) => "kde",
                    DistributionKind::Icar(_) => "icar",
                    DistributionKind::Car(_) => "car",
                    DistributionKind::Categorical => "categorical",
                    DistributionKind::CategoricalLogit => "categorical_logit",
                    DistributionKind::Custom(_) => "custom",
                    DistributionKind::Simulator(_) => "simulator",
                    nested_kinds!() => self.nested_parts().0.name(),
                }
            }

            /// The distribution's support given its parameters' values (see [`Support`]). Bounds that are parameters
            /// are the parameters' values.
            pub fn support(&self, parameters: &[&Value]) -> Result<Support> {
                Ok(match self {
                    $(DistributionKind::$kind => {
                        let &[$($param),+] = parameters else { return Err(self.wrong_parameters()) };
                        $(let $param = tensor!($ptype, $param);)+
                        let _ = ($($param,)+);
                        $support
                    })+
                    DistributionKind::Kde(_) | DistributionKind::Car(_) => Support::Real,
                    DistributionKind::Icar(a) => {
                        Support::SumToZeroComponents(a.connected_groups().cloned().collect())
                    }
                    DistributionKind::Categorical | DistributionKind::CategoricalLogit => Support::Discrete,
                    DistributionKind::Custom(c) => c.support(parameters).map_err(GraphError::Custom)?,
                    DistributionKind::Simulator(s) => match s.value_type() {
                        DType::F64 => Support::Real,
                        DType::I64 | DType::Bool => Support::Discrete,
                    },
                    nested_kinds!() => {
                        let (nested, inner) = self.nested_parts();
                        nested.support(inner, parameters)?
                    }
                })
            }

            /// Draws a value of the given shape from the distribution given its parameters' values.
            pub(crate) fn sample(&self, shape: &[usize], parameters: &[&Value], rng: &mut KeanuRng) -> Result<Value> {
                Ok(match self {
                    $(DistributionKind::$kind => {
                        let &[$($param),+] = parameters else { return Err(self.wrong_parameters()) };
                        $(let $param = tensor!($ptype, $param);)+
                        let dist = $build;
                        Value::$vtype(override_or!(
                            [$({
                                let sample: fn(&_, &[usize], &mut KeanuRng) -> d::Result<_> = $sample;
                                sample(&dist, shape, rng)
                            })?]
                            [dist.sample(shape, rng)]
                        )?)
                    })+
                    DistributionKind::Kde(kde) => {
                        let count: usize = shape.iter().product();
                        let signed: Vec<isize> = shape.iter().map(|&s| s as isize).collect();
                        Value::F64(kde.sample(count, rng).reshape(&signed)?)
                    }
                    DistributionKind::Icar(a) => {
                        let &[sigma] = parameters else { return Err(self.wrong_parameters()) };
                        Value::F64(d::Icar::new(a, sigma.f64()?).sample(shape, rng)?)
                    }
                    DistributionKind::Car(a) => {
                        let &[tau, alpha] = parameters else { return Err(self.wrong_parameters()) };
                        Value::F64(d::Car::new(a, tau.f64()?, alpha.f64()?).sample(shape, rng)?)
                    }
                    DistributionKind::Categorical => {
                        Value::I64(d::Categorical::new(f64_tensors(parameters)?)?.sample(shape, rng)?)
                    }
                    DistributionKind::CategoricalLogit => {
                        Value::I64(d::CategoricalLogit::new(f64_tensors(parameters)?)?.sample(shape, rng)?)
                    }
                    DistributionKind::Custom(c) => {
                        let value = c
                            .sample(shape, parameters, rng)
                            .map_err(GraphError::Custom)?;
                        if value.dtype() != c.value_type() || value.shape() != shape {
                            return Err(GraphError::Custom(format!(
                                "sample returned a {:?} value of shape {:?}, expected {:?} of shape {shape:?}",
                                value.dtype(),
                                value.shape(),
                                c.value_type()
                            )));
                        }
                        value
                    }
                    DistributionKind::Simulator(s) => s.simulate(shape, parameters, rng)?,
                    nested_kinds!() => {
                        let (nested, inner) = self.nested_parts();
                        nested.sample(inner, shape, parameters, rng)?
                    }
                })
            }

            /// The log probability of `value` given the parameters' values, summed over its elements.
            pub(crate) fn log_prob(&self, value: &Value, parameters: &[&Value]) -> Result<f64> {
                Ok(match self {
                    $(DistributionKind::$kind => {
                        let &[$($param),+] = parameters else { return Err(self.wrong_parameters()) };
                        $(let $param = tensor!($ptype, $param);)+
                        $build.log_prob(tensor!($vtype, value))?.total()
                    })+
                    DistributionKind::Kde(kde) => kde.log_prob(value.f64()?).sum(),
                    DistributionKind::Icar(_) | DistributionKind::Car(_) => self.spatial_log_prob(value, parameters)?,
                    DistributionKind::Categorical => {
                        d::Categorical::new(f64_tensors(parameters)?)?.log_prob(value.i64()?)?.sum()
                    }
                    DistributionKind::CategoricalLogit => {
                        d::CategoricalLogit::new(f64_tensors(parameters)?)?.log_prob(value.i64()?)?.sum()
                    }
                    DistributionKind::Custom(c) => c.log_prob(value, parameters).map_err(GraphError::Custom)?,
                    DistributionKind::Simulator(_) => return Err(no_likelihood()),
                    nested_kinds!() => {
                        let (nested, inner) = self.nested_parts();
                        nested.log_prob(inner, value, parameters, false)?.0
                    }
                })
            }

            /// The terms whose sum is [`Self::log_prob`], unreduced: the log probability of each element of the
            /// value (broadcast with the parameters) for a distribution of one value per element, and a scalar for
            /// the others (the multivariate kinds, a hidden Markov model, the linear Gaussian state space model, a
            /// custom distribution).
            pub(crate) fn log_prob_terms(&self, value: &Value, parameters: &[&Value]) -> Result<DoubleTensor> {
                Ok(match self {
                    $(DistributionKind::$kind => {
                        let &[$($param),+] = parameters else { return Err(self.wrong_parameters()) };
                        $(let $param = tensor!($ptype, $param);)+
                        $build.log_prob(tensor!($vtype, value))?.terms()
                    })+
                    DistributionKind::Kde(kde) => kde.log_prob(value.f64()?),
                    DistributionKind::Icar(_) | DistributionKind::Car(_) => {
                        DoubleTensor::scalar(self.spatial_log_prob(value, parameters)?)
                    }
                    DistributionKind::Categorical => {
                        d::Categorical::new(f64_tensors(parameters)?)?.log_prob(value.i64()?)?
                    }
                    DistributionKind::CategoricalLogit => {
                        d::CategoricalLogit::new(f64_tensors(parameters)?)?.log_prob(value.i64()?)?
                    }
                    DistributionKind::Custom(c) => {
                        DoubleTensor::scalar(c.log_prob(value, parameters).map_err(GraphError::Custom)?)
                    }
                    DistributionKind::Simulator(_) => return Err(no_likelihood()),
                    nested_kinds!() => {
                        let (nested, inner) = self.nested_parts();
                        nested.log_prob_terms(inner, value, parameters)?
                    }
                })
            }

            /// The gradient of [`Self::log_prob`] with respect to `value` and each parameter.
            pub(crate) fn log_prob_gradient(
                &self,
                value: &Value,
                parameters: &[&Value],
            ) -> Result<LogProbGradientParts> {
                Ok(match self {
                    $(DistributionKind::$kind => {
                        let &[$($param),+] = parameters else { return Err(self.wrong_parameters()) };
                        $(let $param = tensor!($ptype, $param);)+
                        let dist = $build;
                        let x = tensor!($vtype, value);
                        distribution_gradient!(dist, x, $vtype, [$($param: $ptype),+], [$($gradient)?])
                    })+
                    DistributionKind::Kde(kde) => (Some(kde.d_log_prob(value.f64()?)?.x), smallvec::smallvec![]),
                    DistributionKind::Icar(a) => {
                        let &[sigma] = parameters else { return Err(self.wrong_parameters()) };
                        let g = d::Icar::new(a, sigma.f64()?).d_log_prob(value.f64()?)?;
                        (Some(g.x), smallvec::smallvec![Some(g.sigma)])
                    }
                    DistributionKind::Car(a) => {
                        let &[tau, alpha] = parameters else { return Err(self.wrong_parameters()) };
                        let g = d::Car::new(a, tau.f64()?, alpha.f64()?).d_log_prob(value.f64()?)?;
                        (Some(g.x), smallvec::smallvec![Some(g.tau), Some(g.alpha)])
                    }
                    DistributionKind::Categorical => {
                        super::distribution_hvp::categorical_gradient(value.i64()?, parameters)?
                    }
                    DistributionKind::CategoricalLogit => {
                        super::distribution_hvp::categorical_logit_gradient(value.i64()?, parameters)?
                    }
                    DistributionKind::Custom(c) => {
                        match c.log_prob_gradient(value, parameters).map_err(GraphError::Custom)? {
                            Some(g) => {
                                let mut gradients: ParameterGradients = g.parameters.into_iter().collect();
                                gradients.resize(parameters.len(), None);
                                (g.value, gradients)
                            }
                            None => no_gradient(parameters.len()),
                        }
                    }
                    DistributionKind::Simulator(_) => return Err(no_likelihood()),
                    // the gradient even where the log probability is impossible
                    nested_kinds!() => {
                        let (nested, inner) = self.nested_parts();
                        let (_, gradient) = nested.log_prob(inner, value, parameters, true)?;
                        gradient.unwrap_or_else(|| no_gradient(parameters.len()))
                    }
                })
            }

            /// [`Self::log_prob`] and, unless it is impossible (-infinity or NaN), [`Self::log_prob_gradient`],
            /// sharing the work of reading the parameters and building the distribution. Each is exactly what the
            /// separate method returns.
            pub(crate) fn log_prob_and_gradient(
                &self,
                value: &Value,
                parameters: &[&Value],
            ) -> Result<(f64, Option<LogProbGradientParts>)> {
                Ok(match self {
                    $(DistributionKind::$kind => {
                        let &[$($param),+] = parameters else { return Err(self.wrong_parameters()) };
                        $(let $param = tensor!($ptype, $param);)+
                        let dist = $build;
                        let x = tensor!($vtype, value);
                        override_or!(
                            [$({
                                let fused: fn(&_, &_) -> Result<(f64, LogProbGradientParts)> = $fused;
                                let (log_prob, gradient) = fused(&dist, x)?;
                                (log_prob, (!is_impossible(log_prob)).then_some(gradient))
                            })?]
                            [{
                                let log_prob = dist.log_prob(x)?.total();
                                if is_impossible(log_prob) {
                                    return Ok((log_prob, None));
                                }
                                let gradient = distribution_gradient!(
                                    dist, x, $vtype, [$($param: $ptype),+], [$($gradient)?]
                                );
                                (log_prob, Some(gradient))
                            }]
                        )
                    })+
                    DistributionKind::Kde(_)
                    | DistributionKind::Icar(_)
                    | DistributionKind::Car(_)
                    | DistributionKind::Categorical
                    | DistributionKind::CategoricalLogit
                    | DistributionKind::Custom(_)
                    | DistributionKind::Simulator(_) => {
                        let log_prob = self.log_prob(value, parameters)?;
                        if is_impossible(log_prob) {
                            return Ok((log_prob, None));
                        }
                        (log_prob, Some(self.log_prob_gradient(value, parameters)?))
                    }
                    nested_kinds!() => {
                        let (nested, inner) = self.nested_parts();
                        let (log_prob, gradient) = nested.log_prob(inner, value, parameters, true)?;
                        (log_prob, gradient.filter(|_| !is_impossible(log_prob)))
                    }
                })
            }
        }
    };
}

/// The log prob gradient parts of a built distribution `$dist` at `$x`: the row's `gradient` override if it has
/// one, else the `d_log_prob` field named after each f64 parameter (and `x` for an f64 value).
macro_rules! distribution_gradient {
    ($dist:ident, $x:ident, $vtype:ident, [$($param:ident: $ptype:ident),+], [$($gradient:expr)?]) => {
        override_or!(
            [$({
                let gradient: fn(&_, &_) -> Result<LogProbGradientParts> = $gradient;
                gradient(&$dist, $x)?
            })?]
            [{
                let g = $dist.d_log_prob($x)?;
                (gradient!($vtype, g.x), smallvec::smallvec![$(gradient!($ptype, g.$param)),+])
            }]
        )
    };
}

/// `true` for a row with a CDF.
macro_rules! has_cdf {
    ([]) => {
        false
    };
    ([none]) => {
        false
    };
    ([$cdf:ident]) => {
        true
    };
}

/// `true` for an element-wise row: one with a `cdf` entry, even `none`.
macro_rules! is_elementwise {
    ([]) => {
        false
    };
    ([$cdf:ident]) => {
        true
    };
}

/// The body of [`DistributionKind::univariate`] for a row: an error without a CDF, else the requested function of
/// the built distribution and, if asked for, its gradient parts (from `d_log_prob`, or the row's gradient override,
/// for the log density, and from `d_log_cdf` / `d_log_ccdf` for the others).
macro_rules! univariate {
    ([], $kind:ident, $($rest:tt)*) => {
        return Err($kind.no_cdf())
    };
    ([none], $kind:ident, $function:ident, $x:ident, $parameters:ident, $want:ident, $vtype:ident,
     [$($param:ident: $ptype:ident),+], [$($gradient:expr)?], $build:expr) => {{
        if $function != Univariate::Density {
            return Err($kind.no_cdf());
        }
        let &[$($param),+] = $parameters else { return Err($kind.wrong_parameters()) };
        $(let $param = tensor!($ptype, $param);)+
        let dist = $build;
        let x = tensor!($vtype, $x);
        (
            dist.log_prob(x)?,
            if $want {
                Some(distribution_gradient!(dist, x, $vtype, [$($param: $ptype),+], [$($gradient)?]))
            } else {
                None
            },
        )
    }};
    ([$cdf:ident], $kind:ident, $function:ident, $x:ident, $parameters:ident, $want:ident, $vtype:ident,
     [$($param:ident: $ptype:ident),+], [$($gradient:expr)?], $build:expr) => {{
        let &[$($param),+] = $parameters else { return Err($kind.wrong_parameters()) };
        $(let $param = tensor!($ptype, $param);)+
        let dist = $build;
        let x = tensor!($vtype, $x);
        match $function {
            Univariate::Density => (
                dist.log_prob(x)?,
                if $want {
                    Some(distribution_gradient!(dist, x, $vtype, [$($param: $ptype),+], [$($gradient)?]))
                } else {
                    None
                },
            ),
            Univariate::Cdf => (
                dist.log_cdf(x)?,
                if $want {
                    let g = dist.d_log_cdf(x)?;
                    Some((gradient!($vtype, g.x), smallvec::smallvec![$(gradient!($ptype, g.$param)),+]))
                } else {
                    None
                },
            ),
            Univariate::Ccdf => (
                dist.log_ccdf(x)?,
                if $want {
                    let g = dist.d_log_ccdf(x)?;
                    Some((gradient!($vtype, g.x), smallvec::smallvec![$(gradient!($ptype, g.$param)),+]))
                } else {
                    None
                },
            ),
        }
    }};
}

/// The body of [`DistributionKind::inverse_cdf`] for a row.
macro_rules! inverse_cdf {
    ([inverse], $kind:ident, $p:ident, $upper:ident, $parameters:ident, [$($param:ident: $ptype:ident),+],
     $build:expr) => {{
        let &[$($param),+] = $parameters else { return Err($kind.wrong_parameters()) };
        $(let $param = tensor!($ptype, $param);)+
        let dist = $build;
        Some(if $upper { dist.inverse_ccdf($p)? } else { dist.inverse_cdf($p)? })
    }};
    ([$($other:ident)?], $($rest:tt)*) => {
        None
    };
}

/// A function of a univariate distribution, evaluated element-wise by [`DistributionKind::univariate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Univariate {
    /// The log density (or mass), ln p(x).
    Density,
    /// The log CDF, ln P(X <= x).
    Cdf,
    /// The log survival function, ln P(X > x).
    Ccdf,
}

/// Whether a log probability makes the state impossible: -infinity or NaN.
pub(crate) fn is_impossible(log_prob: f64) -> bool {
    log_prob == f64::NEG_INFINITY || log_prob.is_nan()
}

distributions! {
    Gaussian "gaussian" (mu: F64, sigma: F64) -> F64 => d::Gaussian::new(mu, sigma),
        support = Support::Real,
        cdf = inverse,
        fused = gaussian_log_prob_and_gradient;
    HalfGaussian "half_gaussian" (sigma: F64) -> F64 => d::HalfGaussian::new(sigma),
        support = Support::greater_than(0.0),
        cdf = none;
    Cauchy "cauchy" (location: F64, scale: F64) -> F64 => d::Cauchy::new(location, scale),
        support = Support::Real,
        cdf = inverse;
    HalfCauchy "half_cauchy" (scale: F64) -> F64 => d::HalfCauchy::new(scale),
        support = Support::greater_than(0.0),
        cdf = none;
    Beta "beta" (alpha: F64, beta: F64) -> F64 => d::Beta::standard(alpha, beta),
        support = Support::interval(0.0, 1.0),
        cdf = rejection;
    ChiSquared "chi_squared" (k: I64) -> F64 => d::ChiSquared::new(k),
        support = Support::greater_than(0.0),
        cdf = none;
    Dirichlet "dirichlet" (concentration: F64) -> F64 => d::Dirichlet::new(concentration),
        support = Support::Simplex,
        sample = |dist: &d::Dirichlet<'_>, _, rng| dist.sample(rng);
    Exponential "exponential" (rate: F64) -> F64 => d::Exponential::new(rate),
        support = Support::greater_than(0.0),
        cdf = inverse;
    Gamma "gamma" (theta: F64, k: F64) -> F64 => d::Gamma::new(theta, k),
        support = Support::greater_than(0.0),
        cdf = rejection;
    InverseGamma "inverse_gamma" (alpha: F64, beta: F64) -> F64 => d::InverseGamma::new(alpha, beta),
        support = Support::greater_than(0.0),
        cdf = none;
    Laplace "laplace" (mu: F64, beta: F64) -> F64 => d::Laplace::new(mu, beta),
        support = Support::Real,
        cdf = inverse;
    Logistic "logistic" (mu: F64, s: F64) -> F64 => d::Logistic::new(mu, s),
        support = Support::Real,
        cdf = inverse;
    LogNormal "log_normal" (mu: F64, sigma: F64) -> F64 => d::LogNormal::new(mu, sigma),
        support = Support::greater_than(0.0),
        cdf = inverse;
    LkjCholesky "lkj_cholesky" (eta: F64) -> F64 => d::LkjCholesky::new(eta),
        support = Support::CorrelationCholesky;
    MultivariateGaussian "multivariate_gaussian" (mu: F64, covariance: F64) -> F64 =>
        d::MultivariateGaussian::new(mu, covariance),
        support = Support::Real;
    LinearGaussianStateSpace "linear_gaussian_state_space" (
        initial_mean: F64, initial_covariance: F64, transition_matrix: F64, transition_covariance: F64,
        observation_matrix: F64, observation_covariance: F64
    ) -> F64 => d::LinearGaussianStateSpace::new(
            initial_mean, initial_covariance, transition_matrix, transition_covariance, observation_matrix,
            observation_covariance
        ),
        support = Support::Real, fused = linear_gaussian_state_space_log_prob_and_gradient;
    GaussianRandomWalk "gaussian_random_walk" (mu: F64, sigma: F64, init_mu: F64, init_sigma: F64) -> F64 =>
        d::GaussianRandomWalk::new(mu, sigma, init_mu, init_sigma),
        support = Support::Real;
    Autoregressive "autoregressive" (
        constant: F64, rho: F64, sigma: F64, init_mu: F64, init_sigma: F64
    ) -> F64 => d::Autoregressive::new(constant, rho, sigma, init_mu, init_sigma),
        support = Support::Real;
    Garch11 "garch11" (omega: F64, alpha: F64, beta: F64, initial_vol: F64) -> F64 =>
        d::Garch11::new(omega, alpha, beta, initial_vol),
        support = Support::Real;
    Pareto "pareto" (location: F64, scale: F64) -> F64 => d::Pareto::new(location, scale),
        support = Support::GreaterThan(location.clone()),
        cdf = none;
    StudentT "student_t" (v: I64) -> F64 => d::StudentT::new(v),
        support = Support::Real,
        cdf = rejection;
    Triangular "triangular" (x_min: F64, x_max: F64, c: F64) -> F64 => d::Triangular::new(x_min, x_max, c),
        support = Support::Interval(x_min.clone(), x_max.clone()),
        cdf = none;
    Uniform "uniform" (x_min: F64, x_max: F64) -> F64 => d::Uniform::new(x_min, x_max),
        support = Support::Interval(x_min.clone(), x_max.clone()),
        cdf = inverse,
        gradient = uniform_gradient;
    Weibull "weibull" (alpha: F64, beta: F64) -> F64 => d::Weibull::new(alpha, beta),
        support = Support::greater_than(0.0),
        cdf = inverse;
    ZeroSumNormal "zero_sum_normal" (sigma: F64) -> F64 => d::ZeroSumNormal::new(sigma),
        support = Support::SumToZero;
    VonMises "von_mises" (mu: F64, kappa: F64) -> F64 => d::VonMises::new(mu, kappa),
        support = Support::Circular,
        cdf = none;
    SkewNormal "skew_normal" (xi: F64, omega: F64, alpha: F64) -> F64 => d::SkewNormal::new(xi, omega, alpha),
        support = Support::Real,
        cdf = rejection;
    ExGaussian "ex_gaussian" (mu: F64, sigma: F64, lambda: F64) -> F64 => d::ExGaussian::new(mu, sigma, lambda),
        support = Support::Real,
        cdf = rejection;
    Gumbel "gumbel" (mu: F64, beta: F64) -> F64 => d::Gumbel::new(mu, beta),
        support = Support::Real,
        cdf = inverse;
    GeneralizedExtremeValue "generalized_extreme_value" (mu: F64, sigma: F64, xi: F64) -> F64 =>
        d::GeneralizedExtremeValue::new(mu, sigma, xi),
        support = shape_dependent_support(mu, sigma, xi, false)?,
        cdf = inverse;
    GeneralizedPareto "generalized_pareto" (mu: F64, sigma: F64, xi: F64) -> F64 =>
        d::GeneralizedPareto::new(mu, sigma, xi),
        support = shape_dependent_support(mu, sigma, xi, true)?,
        cdf = inverse;
    InverseGaussian "inverse_gaussian" (mu: F64, lambda: F64) -> F64 => d::InverseGaussian::new(mu, lambda),
        support = Support::greater_than(0.0),
        cdf = rejection;
    Kumaraswamy "kumaraswamy" (a: F64, b: F64) -> F64 => d::Kumaraswamy::new(a, b),
        support = Support::interval(0.0, 1.0),
        cdf = inverse;
    LogitNormal "logit_normal" (mu: F64, sigma: F64) -> F64 => d::LogitNormal::new(mu, sigma),
        support = Support::interval(0.0, 1.0),
        cdf = inverse;
    HalfStudentT "half_student_t" (nu: F64, sigma: F64) -> F64 => d::HalfStudentT::new(nu, sigma),
        support = Support::greater_than(0.0),
        cdf = rejection;
    AsymmetricLaplace "asymmetric_laplace" (mu: F64, sigma: F64, tau: F64) -> F64 =>
        d::AsymmetricLaplace::new(mu, sigma, tau),
        support = Support::Real,
        cdf = inverse;
    LogLogistic "log_logistic" (alpha: F64, beta: F64) -> F64 => d::LogLogistic::new(alpha, beta),
        support = Support::greater_than(0.0),
        cdf = inverse;
    Rice "rice" (nu: F64, sigma: F64) -> F64 => d::Rice::new(nu, sigma),
        support = Support::greater_than(0.0),
        cdf = none;
    GeneralizedGamma "generalized_gamma" (theta: F64, k: F64, p: F64) -> F64 =>
        d::GeneralizedGamma::new(theta, k, p),
        support = Support::greater_than(0.0),
        cdf = rejection;
    Wiener "wiener" (a: F64, t0: F64, w: F64, v: F64, sv: F64, boundary: I64) -> F64 =>
        d::Wiener::new(a, t0, w, v, sv, boundary),
        support = Support::GreaterThan(t0.clone()),
        cdf = none,
        fused = wiener_log_prob_and_gradient;
    MultivariateStudentT "multivariate_student_t" (nu: F64, mu: F64, scale: F64) -> F64 =>
        d::MultivariateStudentT::new(nu, mu, scale),
        support = Support::Real;
    MultivariateStudentTCholesky "multivariate_student_t_cholesky" (nu: F64, mu: F64, scale_cholesky: F64) -> F64 =>
        d::MultivariateStudentTCholesky::new(nu, mu, scale_cholesky),
        support = Support::Real;
    Wishart "wishart" (nu: F64, scale: F64) -> F64 => d::Wishart::new(nu, scale),
        support = Support::CovarianceMatrix;
    WishartCholesky "wishart_cholesky" (nu: F64, scale_cholesky: F64) -> F64 =>
        d::WishartCholesky::new(nu, scale_cholesky),
        support = Support::CovarianceCholesky;
    InverseWishart "inverse_wishart" (nu: F64, scale: F64) -> F64 => d::InverseWishart::new(nu, scale),
        support = Support::CovarianceMatrix;
    InverseWishartCholesky "inverse_wishart_cholesky" (nu: F64, scale_cholesky: F64) -> F64 =>
        d::InverseWishartCholesky::new(nu, scale_cholesky),
        support = Support::CovarianceCholesky;
    MatrixNormal "matrix_normal" (mu: F64, row_covariance: F64, column_covariance: F64) -> F64 =>
        d::MatrixNormal::new(mu, row_covariance, column_covariance),
        support = Support::Real;
    VonMisesFisher "von_mises_fisher" (mu: F64, kappa: F64) -> F64 => d::VonMisesFisher::new(mu, kappa),
        support = Support::UnitVector;
    Bernoulli "bernoulli" (prob_true: F64) -> Bool => d::Bernoulli::new(prob_true),
        support = Support::Discrete;
    Binomial "binomial" (p: F64, n: I64) -> I64 => d::Binomial::new(p, n),
        support = Support::Discrete,
        cdf = rejection;
    Geometric "geometric" (p: F64) -> I64 => d::Geometric::new(p),
        support = Support::Discrete,
        cdf = none;
    Multinomial "multinomial" (n: I64, p: F64) -> I64 => d::Multinomial::new(n, p)?,
        support = Support::Discrete;
    Poisson "poisson" (mu: F64) -> I64 => d::Poisson::new(mu),
        support = Support::Discrete,
        cdf = rejection;
    NegativeBinomial "negative_binomial" (mu: F64, alpha: F64) -> I64 => d::NegativeBinomial::new(mu, alpha),
        support = Support::Discrete,
        cdf = rejection;
    OrderedLogistic "ordered_logistic" (eta: F64, cutpoints: F64) -> I64 =>
        d::OrderedLogistic::new(eta, cutpoints),
        support = Support::Discrete;
    OrderedProbit "ordered_probit" (eta: F64, cutpoints: F64) -> I64 => d::OrderedProbit::new(eta, cutpoints),
        support = Support::Discrete;
    HyperGeometric "hyper_geometric" (population: I64, successes: I64, draws: I64) -> I64 =>
        d::HyperGeometric::new(population, successes, draws),
        support = Support::Discrete,
        cdf = none,
        gradient = |_, _| Ok(no_gradient(3));
    Skellam "skellam" (mu1: F64, mu2: F64) -> I64 => d::Skellam::new(mu1, mu2),
        support = Support::Discrete,
        cdf = none;
    DirichletMultinomial "dirichlet_multinomial" (n: I64, alpha: F64) -> I64 => d::DirichletMultinomial::new(n, alpha),
        support = Support::Discrete;
    PlackettLuce "plackett_luce" (worths: F64) -> I64 => d::PlackettLuce::new(worths),
        support = Support::Discrete;
    UniformInt "uniform_int" (x_min: I64, x_max: I64) -> I64 => d::UniformInt::new(x_min, x_max),
        support = Support::Discrete,
        cdf = none,
        gradient = |_, _| Ok(no_gradient(2));
    BetaBinomial "beta_binomial" (n: I64, alpha: F64, beta: F64) -> I64 => d::BetaBinomial::new(n, alpha, beta),
        support = Support::Discrete,
        cdf = rejection;
    BernoulliLogit "bernoulli_logit" (logit: F64) -> Bool => d::BernoulliLogit::new(logit),
        support = Support::Discrete;
    BinomialLogit "binomial_logit" (logit: F64, n: I64) -> I64 => d::BinomialLogit::new(logit, n),
        support = Support::Discrete,
        cdf = none;
    PoissonLog "poisson_log" (log_mu: F64) -> I64 => d::PoissonLog::new(log_mu),
        support = Support::Discrete,
        cdf = none;
}

/// The support of a generalized extreme value (`pareto` false) or generalized Pareto (`pareto` true)
/// distribution, element-wise over the broadcast of its parameters: 1 + ξ(x − μ)/σ > 0, so x > μ − σ/ξ where ξ > 0
/// and x < μ − σ/ξ where ξ < 0, and for the generalized Pareto also x > μ. Reported as real, greater than or less
/// than where every element has the same kind of bound, else as an interval with infinite ends where unbounded.
fn shape_dependent_support(
    mu: &DoubleTensor,
    sigma: &DoubleTensor,
    xi: &DoubleTensor,
    pareto: bool,
) -> Result<Support> {
    let shape = keanu_tensor::broadcast_shape(
        &keanu_tensor::broadcast_shape(mu.shape(), sigma.shape())?,
        xi.shape(),
    )?;
    let (mu, sigma, xi) = (
        mu.broadcast_to(&shape)?,
        sigma.broadcast_to(&shape)?,
        xi.broadcast_to(&shape)?,
    );
    let ends: Vec<(f64, f64)> = mu
        .as_slice()
        .iter()
        .zip(sigma.as_slice())
        .zip(xi.as_slice())
        .map(|((&m, &s), &x)| {
            let end = m - s / x;
            let lower = if pareto {
                m
            } else if x > 0.0 {
                end
            } else {
                f64::NEG_INFINITY
            };
            let upper = if x < 0.0 { end } else { f64::INFINITY };
            (lower, upper)
        })
        .collect();
    let lower = DoubleTensor::new(&shape, ends.iter().map(|e| e.0).collect())?;
    let upper = DoubleTensor::new(&shape, ends.iter().map(|e| e.1).collect())?;
    let unbounded = |t: &DoubleTensor| t.as_slice().iter().all(|v| v.is_infinite());
    Ok(match (unbounded(&lower), unbounded(&upper)) {
        (true, true) => Support::Real,
        (false, true) => Support::GreaterThan(lower),
        (true, false) => Support::LessThan(upper),
        (false, false) => Support::Interval(lower, upper),
    })
}

impl DistributionKind {
    /// The log density of an ICAR or proper CAR.
    fn spatial_log_prob(&self, value: &Value, parameters: &[&Value]) -> Result<f64> {
        Ok(match (self, parameters) {
            (DistributionKind::Icar(a), &[sigma]) => {
                d::Icar::new(a, sigma.f64()?).log_prob(value.f64()?)?
            }
            (DistributionKind::Car(a), &[tau, alpha]) => {
                d::Car::new(a, tau.f64()?, alpha.f64()?).log_prob(value.f64()?)?
            }
            _ => return Err(self.wrong_parameters()),
        })
    }

    /// True for a [simulator](Self::Simulator), which has no likelihood.
    pub fn is_simulator(&self) -> bool {
        matches!(self, Self::Simulator(_))
    }

    /// The error for a distribution without a CDF used where one is needed.
    fn no_cdf(&self) -> GraphError {
        GraphError::InvalidNode {
            op: self.name(),
            message: "the distribution has no CDF, so it cannot be truncated or censored".into(),
        }
    }

    /// The error for a parameter list that doesn't match the distribution (checked when the node is built).
    pub(crate) fn wrong_parameters(&self) -> GraphError {
        GraphError::InvalidNode {
            op: self.name(),
            message: format!("needs parameters of types {:?}", self.parameter_types()),
        }
    }
}

/// The sum of an element-wise log probability (distributions that reduce over their event dimension return the
/// sum already).
trait Total {
    fn total(self) -> f64;
    /// The terms of the sum, unreduced: a scalar for a sum already taken.
    fn terms(self) -> DoubleTensor;
}

impl Total for DoubleTensor {
    fn total(self) -> f64 {
        self.sum()
    }
    fn terms(self) -> DoubleTensor {
        self
    }
}

impl Total for f64 {
    fn total(self) -> f64 {
        self
    }
    fn terms(self) -> DoubleTensor {
        DoubleTensor::scalar(self)
    }
}

/// Every parameter's f64 tensor (Categorical's probabilities).
fn f64_tensors<'v>(parameters: &[&'v Value]) -> Result<Vec<&'v DoubleTensor>> {
    parameters.iter().map(|p| p.f64()).collect()
}

/// No gradient with respect to the value or any of `parameters` parameters.
fn no_gradient(parameters: usize) -> LogProbGradientParts {
    (None, smallvec::smallvec![None; parameters])
}

/// The Gaussian log probability and gradient in one pass.
fn gaussian_log_prob_and_gradient(
    gaussian: &d::Gaussian<'_>,
    x: &DoubleTensor,
) -> Result<(f64, LogProbGradientParts)> {
    let (log_prob, g) = gaussian.log_prob_and_d_log_prob(x)?;
    Ok((
        log_prob.sum(),
        (Some(g.x), smallvec::smallvec![Some(g.mu), Some(g.sigma)]),
    ))
}

/// The Wiener log density and gradient from one pass over the series.
fn wiener_log_prob_and_gradient(
    wiener: &d::Wiener<'_>,
    x: &DoubleTensor,
) -> Result<(f64, LogProbGradientParts)> {
    let (log_prob, g) = wiener.log_prob_and_d_log_prob(x)?;
    Ok((
        log_prob.sum(),
        (
            Some(g.x),
            smallvec::smallvec![
                Some(g.a),
                Some(g.t0),
                Some(g.w),
                Some(g.v),
                Some(g.sv),
                None
            ],
        ),
    ))
}

/// The linear Gaussian state space model's log density and gradient from one pass of the Kalman filter (the
/// gradient's parts are empty where the observations are impossible, which the caller discards).
fn linear_gaussian_state_space_log_prob_and_gradient(
    model: &d::LinearGaussianStateSpace<'_>,
    x: &DoubleTensor,
) -> Result<(f64, LogProbGradientParts)> {
    let (log_prob, gradient) = model.log_prob_and_d_log_prob(x)?;
    let parts = match gradient {
        Some(g) => (
            Some(g.x),
            smallvec::smallvec![
                Some(g.initial_mean),
                Some(g.initial_covariance),
                Some(g.transition_matrix),
                Some(g.transition_covariance),
                Some(g.observation_matrix),
                Some(g.observation_covariance),
            ],
        ),
        None => (None, smallvec::smallvec![]),
    };
    Ok((log_prob, parts))
}

/// As UniformVertex: zero inside the support, -inf above x_max and +inf at or below x_min, and no gradient with
/// respect to the bounds.
fn uniform_gradient(uniform: &d::Uniform<'_>, x: &DoubleTensor) -> Result<LogProbGradientParts> {
    let dx = x
        .zip_with(&*uniform.x_max, |x, max| {
            if x > max { f64::NEG_INFINITY } else { 0.0 }
        })?
        .zip_with(&x.le(&uniform.x_min)?, |d, below| {
            if below { f64::INFINITY } else { d }
        })?;
    Ok((Some(dx), smallvec::smallvec![None, None]))
}

/// The value shape of a kind whose value is a batch of events ([`DistributionKind::MultivariateStudentT`], the
/// Wishart forms, ...): the event shape from the parameters, every parameter's trailing axes checked against its
/// event shape and its leading axes broadcast into the batch, with the requested shape (if any) broadcast in too.
fn event_value_shape(
    kind: &DistributionKind,
    requested: Option<&[usize]>,
    shapes: &[&[usize]],
) -> std::result::Result<Shape, String> {
    use DistributionKind as K;
    let last = |name: &str, s: &[usize], minimum: usize| match s.last() {
        Some(&k) if k >= minimum => Ok(k),
        _ => Err(format!(
            "{name} must have a last axis of at least {minimum}, not shape {s:?}"
        )),
    };
    let square = |name: &str, s: &[usize]| match s {
        [.., rows, cols] if rows == cols && *rows > 0 => Ok(*rows),
        _ => Err(format!(
            "{name} must end in two equal, non-zero axes, not shape {s:?}"
        )),
    };
    let (event, expected): (Vec<usize>, Vec<Vec<usize>>) = match kind {
        K::DirichletMultinomial => {
            let k = last("alpha", shapes[1], 1)?;
            (vec![k], vec![vec![], vec![k]])
        }
        K::PlackettLuce => {
            let k = last("worths", shapes[0], 1)?;
            (vec![k], vec![vec![k]])
        }
        K::MultivariateStudentT | K::MultivariateStudentTCholesky => {
            let k = last("mu", shapes[1], 1)?;
            (vec![k], vec![vec![], vec![k], vec![k, k]])
        }
        K::Wishart | K::WishartCholesky | K::InverseWishart | K::InverseWishartCholesky => {
            let k = square("the scale", shapes[1])?;
            (vec![k, k], vec![vec![], vec![k, k]])
        }
        K::MatrixNormal => {
            let [.., n, p] = shapes[0] else {
                return Err(format!(
                    "mu must be a matrix [..., n, p], not shape {:?}",
                    shapes[0]
                ));
            };
            (vec![*n, *p], vec![vec![*n, *p], vec![*n, *n], vec![*p, *p]])
        }
        K::VonMisesFisher => {
            let p = last("mu", shapes[0], 2)?;
            (vec![p], vec![vec![p], vec![]])
        }
        _ => unreachable!("only called for kinds over events"),
    };
    let names = kind.parameter_names().unwrap_or_default();
    let mut batch = Shape::new();
    for ((name, shape), expected) in names.iter().zip(shapes).zip(&expected) {
        let r = expected.len();
        if shape.len() < r || shape[shape.len() - r..] != expected[..] {
            return Err(format!(
                "{name} of shape {shape:?} must end in {expected:?} for a value with events of shape {event:?}"
            ));
        }
        batch = keanu_tensor::broadcast_shape(&batch, &shape[..shape.len() - r])
            .map_err(|e| e.to_string())?;
    }
    if let Some(requested) = requested {
        let r = event.len();
        if requested.len() < r || requested[requested.len() - r..] != event[..] {
            return Err(format!(
                "the requested shape {requested:?} must end in the event shape {event:?}"
            ));
        }
        batch = keanu_tensor::broadcast_shape(&batch, &requested[..requested.len() - r])
            .map_err(|e| e.to_string())?;
    }
    batch.extend_from_slice(&event);
    Ok(batch)
}

/// A probabilistic node's distribution and the shape of its value.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DistributionOp {
    /// The distribution.
    pub kind: DistributionKind,
    /// The shape of the node's value.
    pub shape: Shape,
}

impl DistributionOp {
    /// The node's value shape given the requested shape and the parameter shapes: the broadcast of all of them,
    /// except for distributions whose value shape is determined by a parameter.
    pub(crate) fn value_shape(
        kind: &DistributionKind,
        requested: Option<&[usize]>,
        parameters: &[(DType, &[usize])],
    ) -> Result<Shape> {
        let invalid = |message: String| GraphError::InvalidNode {
            op: kind.name(),
            message,
        };
        if let DistributionKind::Custom(_) | DistributionKind::Simulator(_) = kind {
            return Ok(Shape::from_slice(requested.unwrap_or(&[])));
        }
        if let Some((nested, inner)) = kind.nested() {
            nested.check_inner(inner).map_err(invalid)?;
        }
        match kind.parameter_types() {
            Some(types) => {
                if parameters.len() != types.len()
                    || parameters
                        .iter()
                        .zip(types.iter())
                        .any(|((t, _), e)| t != e)
                {
                    return Err(invalid(format!("needs parameters of types {types:?}")));
                }
            }
            None => {
                if parameters.is_empty()
                    || parameters
                        .iter()
                        .any(|(t, s)| *t != DType::F64 || *s != parameters[0].1)
                {
                    return Err(invalid(
                        "needs at least one f64 probability, all of the same shape".into(),
                    ));
                }
            }
        }
        match kind {
            DistributionKind::Dirichlet => Ok(Shape::from_slice(parameters[0].1)),
            DistributionKind::MultivariateGaussian => {
                let (mu, covariance) = (parameters[0].1, parameters[1].1);
                if mu.is_empty() || covariance.len() < 2 {
                    return Err(invalid(
                        "mu must be a vector and the covariance a matrix".into(),
                    ));
                }
                let n = mu[mu.len() - 1];
                let c = covariance.len();
                if covariance[c - 1] != n || covariance[c - 2] != n {
                    return Err(invalid(format!(
                        "covariance {covariance:?} does not match mu {mu:?}"
                    )));
                }
                let batch =
                    keanu_tensor::broadcast_shape(&mu[..mu.len() - 1], &covariance[..c - 2])?;
                let mut shape = batch;
                shape.push(n);
                Ok(match requested {
                    Some(r) => keanu_tensor::broadcast_shape(r, &shape)?,
                    None => shape,
                })
            }
            DistributionKind::LkjCholesky => {
                let shape = requested.unwrap_or(&[]);
                let [batch @ .., rows, cols] = shape else {
                    return Err(invalid(format!(
                        "needs a shape [..., K, K] for its K × K Cholesky factors, not {shape:?}"
                    )));
                };
                if rows != cols || *rows == 0 {
                    return Err(invalid(format!(
                        "needs a shape [..., K, K] for its K × K Cholesky factors, with K > 0, not {shape:?}"
                    )));
                }
                let eta = parameters[0].1;
                if keanu_tensor::broadcast_shape(batch, eta)?.as_slice() != batch {
                    return Err(invalid(format!(
                        "eta of shape {eta:?} does not broadcast to the batch shape {batch:?}"
                    )));
                }
                Ok(Shape::from_slice(shape))
            }
            DistributionKind::ZeroSumNormal => {
                let shape = requested.unwrap_or(&[]);
                let [batch @ .., n] = shape else {
                    return Err(invalid(format!(
                        "needs a shape [..., N] for its vectors of N elements, not {shape:?}"
                    )));
                };
                if *n == 0 {
                    return Err(invalid(format!(
                        "needs a shape [..., N] for its vectors of N elements, with N > 0, not {shape:?}"
                    )));
                }
                let sigma = parameters[0].1;
                if keanu_tensor::broadcast_shape(batch, sigma)?.as_slice() != batch {
                    return Err(invalid(format!(
                        "sigma of shape {sigma:?} does not broadcast to the batch shape {batch:?}"
                    )));
                }
                Ok(Shape::from_slice(shape))
            }
            DistributionKind::GaussianRandomWalk
            | DistributionKind::Autoregressive
            | DistributionKind::Garch11 => {
                let shape = requested.unwrap_or(&[]);
                let [batch @ .., steps] = shape else {
                    return Err(invalid(format!(
                        "needs a shape [..., T] for its series of T steps, not {shape:?}"
                    )));
                };
                if *steps == 0 {
                    return Err(invalid(format!(
                        "needs a shape [..., T] for its series of T steps, with T > 0, not {shape:?}"
                    )));
                }
                let names = kind.parameter_names().unwrap_or_default();
                for (name, (_, p)) in names.iter().zip(parameters) {
                    let p_batch = if *name == "rho" {
                        match p.split_last() {
                            Some((&lags, b)) if lags > 0 => b,
                            _ => {
                                return Err(invalid(format!(
                                    "rho must have a trailing axis of at least one lag, not shape {p:?}"
                                )));
                            }
                        }
                    } else {
                        p
                    };
                    let broadcast = keanu_tensor::broadcast_shape(batch, p_batch).ok();
                    if broadcast.as_ref().map(|b| b.as_slice()) != Some(batch) {
                        return Err(invalid(format!(
                            "{name} of shape {p:?} does not broadcast to the batch shape {batch:?}"
                        )));
                    }
                }
                Ok(Shape::from_slice(shape))
            }
            DistributionKind::Kde(_) => Ok(Shape::from_slice(requested.unwrap_or(&[]))),
            DistributionKind::Icar(a) | DistributionKind::Car(a) => {
                let n = a.len();
                if requested.is_some_and(|r| r != [n]) {
                    return Err(invalid(format!(
                        "has one value per area, shape [{n}], not {:?}",
                        requested.unwrap_or_default()
                    )));
                }
                let names = kind.parameter_names().unwrap_or_default();
                for (name, (_, p)) in names.iter().zip(parameters) {
                    if p.iter().product::<usize>() != 1 {
                        return Err(invalid(format!(
                            "{name} must be a scalar, not of shape {p:?}"
                        )));
                    }
                }
                Ok(Shape::from_slice(&[n]))
            }
            DistributionKind::Mixture(_) => {
                mixture::value_shape(requested, parameters).map_err(invalid)
            }
            DistributionKind::HiddenMarkovModel(_) => {
                hmm::value_shape(requested, parameters).map_err(invalid)
            }
            DistributionKind::LinearGaussianStateSpace => {
                let shapes: Vec<&[usize]> = parameters.iter().map(|(_, s)| *s).collect();
                let (d, m) = match *shapes.as_slice() {
                    [[d], _, _, _, [m, _], _] => (*d, *m),
                    _ => {
                        return Err(invalid(
                            "the initial mean must be a vector [d] and the observation matrix a matrix [m, d]"
                                .into(),
                        ));
                    }
                };
                let expected: [&[usize]; 6] = [&[d], &[d, d], &[d, d], &[d, d], &[m, d], &[m, m]];
                let names = kind.parameter_names().unwrap_or_default();
                for ((name, shape), expected) in names.iter().zip(&shapes).zip(expected) {
                    if *shape != expected {
                        return Err(invalid(format!(
                            "{name} has shape {shape:?}, but a state of dimension {d} observed in dimension {m} \
                             needs {expected:?}"
                        )));
                    }
                }
                match requested {
                    Some(&[t]) => Ok(Shape::from_slice(&[t, m])),
                    Some(&[t, mm]) if mm == m => Ok(Shape::from_slice(&[t, m])),
                    _ => Err(invalid(format!(
                        "needs a shape [T] or [T, {m}] for its observations, not {requested:?}"
                    ))),
                }
            }
            DistributionKind::MultivariateStudentT
            | DistributionKind::MultivariateStudentTCholesky
            | DistributionKind::Wishart
            | DistributionKind::WishartCholesky
            | DistributionKind::InverseWishart
            | DistributionKind::InverseWishartCholesky
            | DistributionKind::MatrixNormal
            | DistributionKind::VonMisesFisher
            | DistributionKind::DirichletMultinomial
            | DistributionKind::PlackettLuce => {
                let shapes: Vec<&[usize]> = parameters.iter().map(|(_, s)| *s).collect();
                event_value_shape(kind, requested, &shapes).map_err(invalid)
            }
            DistributionKind::OrderedLogistic | DistributionKind::OrderedProbit => {
                let (eta, cutpoints) = (parameters[0].1, parameters[1].1);
                let Some((&count, leading)) = cutpoints.split_last() else {
                    return Err(invalid("the cutpoints must have at least one axis".into()));
                };
                if count == 0 {
                    return Err(invalid("there must be at least one cutpoint".into()));
                }
                let mut shape = keanu_tensor::broadcast_shape(eta, leading)?;
                if let Some(r) = requested {
                    shape = keanu_tensor::broadcast_shape(r, &shape)?;
                }
                Ok(shape)
            }
            _ => {
                let mut shape = Shape::from_slice(requested.unwrap_or(&[]));
                for (_, p) in parameters {
                    shape = keanu_tensor::broadcast_shape(&shape, p)?;
                }
                Ok(shape)
            }
        }
    }

    /// Draws a value from the distribution given its parameters' values.
    pub(crate) fn sample(&self, parameters: &[&Value], rng: &mut KeanuRng) -> Result<Value> {
        self.kind.sample(self.shape.as_slice(), parameters, rng)
    }
}
