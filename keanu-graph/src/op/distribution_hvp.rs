//! Second derivatives of distributions' log densities: Hessian-vector products in the value and the parameters,
//! for exact Hessian-vector products of a joint log probability (see
//! [`Parameterisation::hvp`](crate::Parameterisation::hvp)).
//!
//! Kept apart from the distribution table so that each kind's rule can be added without touching it. Every kind
//! but a custom distribution has a rule:
//!
//! - element-wise kinds (one value per element, with an element-wise log density) contract a per-element Hessian
//!   in the value and the parameters, from the scalar rules in [`rules`], with the tangents;
//! - truncated and censored kinds add the second derivatives of the base's log CDF and log survival function (see
//!   the wrappers module), a mixture those of a logsumexp of its components (see the mixture module), and a
//!   hidden Markov model those of the forward-backward algorithm (see the hmm module), all on top of the
//!   element-wise contraction of their base, component or emission;
//! - the multivariate kinds (multivariate Gaussian, Dirichlet, LKJ Cholesky, zero-sum normal, ICAR, proper CAR,
//!   multinomial, ordered logistic and probit, KDE) have closed forms of their own here, and the multivariate
//!   Student-t, Wishart and inverse Wishart (and their Cholesky forms), matrix normal, von Mises–Fisher,
//!   Dirichlet-multinomial and Plackett–Luce differentiate their hand-written gradients in forward mode (see
//!   [`keanu_distributions::Wishart::hvp`]);
//! - the linear Gaussian state space model differentiates the Kalman filter's reverse pass in forward mode (see
//!   [`keanu_distributions::LinearGaussianStateSpace::hvp`]), and the Gaussian random walk, AR(p) and GARCH(1,1)
//!   their hand-written reverse passes, on dual numbers (see [`keanu_distributions::processes`]).
//!
//! A rule's derivatives are those of the first-order gradient it sits next to, including where that gradient
//! follows a convention: the uniform has no derivative with respect to its bounds, so its second derivatives are
//! zero too, and the Laplace density's kink at its
//! location takes the almost-everywhere second derivative 0. The second derivatives of a gamma, beta or negative
//! binomial CDF with respect to a shape parameter come from the second shape derivatives of the regularised
//! incomplete gamma and beta functions, and the linear Gaussian state space model differentiates the Kalman
//! filter in forward mode over its reverse pass. Only a custom distribution returns
//! [`GraphError::NoSecondOrderRule`], naming the kind.

pub(crate) mod rules;

use super::distribution::{DistributionKind, Univariate};
use super::nested::{Nested, nested_kinds};
use super::{inflated, wrappers};
use crate::autodiff::{batched_shape, fit, lift, reduce_to};
use crate::error::{GraphError, Result};
use crate::value::Value;
use keanu_tensor::special::trigamma;
use keanu_tensor::{DoubleTensor, Shape, broadcast_shape};
use rules::Hessian;
use smallvec::SmallVec;

/// Hessian-vector products of a node's log density along `b` directions: with respect to its value (None for
/// non-f64 values or where zero) and to each parameter (None where there is none), each shaped
/// `[b, value or parameter shape...]`.
pub(crate) type LogProbHvpParts = (Option<DoubleTensor>, SmallVec<[Option<DoubleTensor>; 4]>);

impl DistributionKind {
    /// The Hessian-vector products of [`Self::log_prob`] (summed over the value) with respect to the value and each
    /// parameter, along `b` directions given as the tangents of the value and of each parameter
    /// (`[b, value or parameter shape...]`, None for a zero tangent): for each input u,
    /// `Σᵥ ∂² log p / ∂u ∂v · v̇` over the inputs v. These are the derivatives along the tangents of
    /// [`Self::log_prob_gradient`].
    ///
    /// # Errors
    ///
    /// [`GraphError::NoSecondOrderRule`] naming the kind for a custom distribution (see the
    /// [module documentation](self)).
    pub(crate) fn log_prob_hvp(
        &self,
        value: &Value,
        parameters: &[&Value],
        value_tangent: Option<&DoubleTensor>,
        parameter_tangents: &[Option<&DoubleTensor>],
        b: usize,
    ) -> Result<LogProbHvpParts> {
        use DistributionKind as K;
        // exhaustive, so that a new kind needs a rule (or an explicit error) to compile
        match self {
            K::Gaussian
            | K::HalfGaussian
            | K::Cauchy
            | K::HalfCauchy
            | K::Beta
            | K::ChiSquared
            | K::Exponential
            | K::Gamma
            | K::InverseGamma
            | K::Laplace
            | K::Logistic
            | K::LogNormal
            | K::Pareto
            | K::StudentT
            | K::Triangular
            | K::Uniform
            | K::Weibull
            | K::VonMises
            | K::SkewNormal
            | K::ExGaussian
            | K::Gumbel
            | K::GeneralizedExtremeValue
            | K::GeneralizedPareto
            | K::InverseGaussian
            | K::Kumaraswamy
            | K::LogitNormal
            | K::HalfStudentT
            | K::LocationScaleStudentT
            | K::AsymmetricLaplace
            | K::LogLogistic
            | K::Rice
            | K::GeneralizedGamma
            | K::Bernoulli
            | K::Binomial
            | K::Geometric
            | K::Poisson
            | K::NegativeBinomial
            | K::SkewStudentT
            | K::PolyaGamma1
            | K::PolyaGamma
            | K::DiscreteWeibull
            | K::NegativeBinomialLog
            | K::UniformInt
            | K::BetaBinomial
            | K::BernoulliLogit
            | K::BinomialLogit
            | K::PoissonLog
            | K::HyperGeometric
            | K::Skellam
            | K::Wiener
            | K::WienerFull => elementwise_hvp(
                self,
                value,
                parameters,
                value_tangent,
                parameter_tangents,
                b,
            ),
            K::MultivariateGaussianPrecision => {
                let &[mu, precision] = parameters else {
                    return Err(self.wrong_parameters());
                };
                gaussian_precision_hvp(
                    value.f64()?,
                    mu.f64()?,
                    precision.f64()?,
                    value_tangent,
                    parameter_tangents,
                    b,
                )
            }
            K::MultivariateGaussian => {
                let &[mu, covariance] = parameters else {
                    return Err(self.wrong_parameters());
                };
                multivariate_gaussian_hvp(
                    value.f64()?,
                    mu.f64()?,
                    covariance.f64()?,
                    value_tangent,
                    parameter_tangents,
                )
            }
            K::Dirichlet => {
                let &[concentration] = parameters else {
                    return Err(self.wrong_parameters());
                };
                dirichlet_hvp(
                    value.f64()?,
                    concentration.f64()?,
                    value_tangent,
                    parameter_tangents[0],
                    b,
                )
            }
            K::LkjCholesky => {
                let &[eta] = parameters else {
                    return Err(self.wrong_parameters());
                };
                lkj_cholesky_hvp(
                    value.f64()?,
                    eta.f64()?,
                    value_tangent,
                    parameter_tangents[0],
                    b,
                )
            }
            K::ZeroSumNormal => {
                let &[sigma] = parameters else {
                    return Err(self.wrong_parameters());
                };
                zero_sum_normal_hvp(
                    value.f64()?,
                    sigma.f64()?,
                    value_tangent,
                    parameter_tangents[0],
                    b,
                )
            }
            K::PoissonBinomial => {
                let &[p] = parameters else {
                    return Err(self.wrong_parameters());
                };
                let Some(tangent) = parameter_tangents[0] else {
                    return Ok((None, smallvec::smallvec![None]));
                };
                let product = keanu_distributions::PoissonBinomial::new(p.f64()?)?
                    .hvp(value.i64()?, tangent)?;
                Ok((None, smallvec::smallvec![Some(product)]))
            }
            K::MultinomialLogit => {
                let &[n, logits] = parameters else {
                    return Err(self.wrong_parameters());
                };
                multinomial_logit_hvp(
                    value.shape(),
                    &n.i64()?.to_double(),
                    logits.f64()?,
                    parameter_tangents[1],
                    b,
                )
            }
            K::Multinomial => {
                let &[_, p] = parameters else {
                    return Err(self.wrong_parameters());
                };
                multinomial_hvp(
                    &value.i64()?.to_double(),
                    p.f64()?,
                    parameter_tangents[1],
                    b,
                )
            }
            K::OrderedLogistic => {
                let &[eta, cutpoints] = parameters else {
                    return Err(self.wrong_parameters());
                };
                ordered_logistic_hvp(
                    value.i64()?,
                    eta.f64()?,
                    cutpoints.f64()?,
                    parameter_tangents,
                    b,
                )
            }
            K::OrderedProbit => {
                let &[eta, cutpoints] = parameters else {
                    return Err(self.wrong_parameters());
                };
                ordered_probit_hvp(
                    value.i64()?,
                    eta.f64()?,
                    cutpoints.f64()?,
                    parameter_tangents,
                    b,
                )
            }
            K::MultivariateStudentT
            | K::MultivariateStudentTCholesky
            | K::Wishart
            | K::WishartCholesky
            | K::InverseWishart
            | K::InverseWishartCholesky
            | K::MatrixNormal
            | K::VonMisesFisher
            | K::DirichletMultinomial
            | K::PlackettLuce => event_hvp(
                self,
                value,
                parameters,
                value_tangent,
                parameter_tangents,
                b,
            ),
            K::Icar(adjacency) => {
                let &[sigma] = parameters else {
                    return Err(self.wrong_parameters());
                };
                icar_hvp(
                    self,
                    adjacency,
                    value.f64()?,
                    sigma.f64()?,
                    value_tangent,
                    parameter_tangents[0],
                    b,
                )
            }
            K::Car(adjacency) => car_hvp(
                self,
                adjacency,
                value.f64()?,
                parameters,
                value_tangent,
                parameter_tangents,
                b,
            ),
            K::Kde(kde) => Ok((
                value_tangent
                    .map(|t| kde_hvp(kde, value.f64()?, t, b))
                    .transpose()?,
                SmallVec::new(),
            )),
            K::Categorical => categorical_hvp(value.i64()?, parameters, parameter_tangents, b),
            K::CategoricalLogit => {
                categorical_logit_hvp(value.i64()?, parameters, parameter_tangents, b)
            }
            nested_kinds!() => Nested::log_prob_hvp(
                self,
                value,
                parameters,
                value_tangent,
                parameter_tangents,
                b,
            ),
            K::MixtureOf(components) => super::mixture_of::log_prob_hvp(
                components,
                value,
                parameters,
                value_tangent,
                parameter_tangents,
                b,
            ),
            K::Transformed(..) => elementwise_hvp(
                self,
                value,
                parameters,
                value_tangent,
                parameter_tangents,
                b,
            ),
            K::GaussianRandomWalk | K::Autoregressive | K::Garch11 => process_hvp(
                self,
                value.f64()?,
                parameters,
                value_tangent,
                parameter_tangents,
                b,
            ),
            K::LinearGaussianStateSpace => linear_gaussian_state_space_hvp(
                value.f64()?,
                parameters,
                value_tangent,
                parameter_tangents,
                b,
            ),
            K::Custom(_) => Err(GraphError::NoSecondOrderRule(format!(
                "the {} distribution",
                self.name()
            ))),
            K::Simulator(_) => Err(GraphError::NoLikelihood("the node".into())),
        }
    }
}

/// An element-wise kind's log density and its first and second derivatives at each element of a broadcast shape.
pub(crate) struct Elementwise {
    /// The log density at each element.
    pub(crate) values: Vec<f64>,
    /// The gradient with respect to each input (the value, then each parameter) at each element, None where there
    /// is none.
    pub(crate) gradients: Vec<Option<Vec<f64>>>,
    /// The Hessian-vector products with respect to each input, `[b, shape...]` flattened, None exactly where the
    /// gradient is.
    pub(crate) hvps: Vec<Option<Vec<f64>>>,
}

/// The second-order parts of an [element-wise](DistributionKind::is_elementwise) kind at `x` broadcast with the
/// parameters to `shape`, along the inputs' `tangents` (the value's, then each parameter's, each `[b, shape...]`
/// flattened, None for a zero tangent).
pub(crate) fn elementwise(
    kind: &DistributionKind,
    x: &Value,
    parameters: &[&Value],
    tangents: &[Option<Vec<f64>>],
    b: usize,
    shape: &[usize],
) -> Result<Elementwise> {
    if let DistributionKind::Transformed(base, bijector) = kind {
        return super::transformed::second_order(
            base, *bijector, x, parameters, tangents, b, shape,
        );
    }
    if let Some((nested, inner)) = kind.nested() {
        if let Some(wrapper) = nested.wrapper() {
            return wrappers::second_order(wrapper, inner, x, parameters, tangents, b, shape);
        }
        if let Some(inflation) = nested.inflation() {
            return inflated::second_order(inflation, inner, x, parameters, tangents, b, shape);
        }
    }
    let Some(rule) = rules::density_rule(kind) else {
        return Err(GraphError::NoSecondOrderRule(format!(
            "the {} distribution",
            kind.name()
        )));
    };
    let (values, parts) = if kind.is_elementwise() {
        let (values, parts) = kind.univariate(Univariate::Density, x, parameters, true)?;
        (spread(&values, shape)?, parts)
    } else {
        // an element-wise density that can't be a mixture component (the Bernoulli and its logit form), whose
        // values are never read
        let len = shape.iter().product();
        (vec![0.0; len], Some(kind.log_prob_gradient(x, parameters)?))
    };
    let (x_gradient, parameter_gradients) =
        parts.unwrap_or_else(|| (None, smallvec::smallvec![None; parameters.len()]));
    let gradients: Vec<Option<Vec<f64>>> = std::iter::once(x_gradient)
        .chain(parameter_gradients)
        .map(|g| g.map(|g| spread(&g, shape)).transpose())
        .collect::<Result<_>>()?;
    let inputs: Vec<Vec<f64>> = std::iter::once(x)
        .chain(parameters.iter().copied())
        .map(|v| spread(&as_f64(v), shape))
        .collect::<Result<_>>()?;
    let differentiable: Vec<bool> = gradients.iter().map(Option::is_some).collect();
    let mut point = vec![0.0; parameters.len()];
    let hvps = contract(&differentiable, tangents, values.len(), b, |i, h| {
        for (p, input) in point.iter_mut().zip(&inputs[1..]) {
            *p = input[i];
        }
        rule(inputs[0][i], &point, h);
        Ok(())
    })?;
    Ok(Elementwise {
        values,
        gradients,
        hvps,
    })
}

/// The Hessian-vector products of a log density over `len` elements given each element's Hessian over the
/// inputs (filled by `hessian`, which gets a zeroed one): for each `differentiable` input u, `[b, len]` flattened
/// with `Σᵥ H_uv t_v` at [k, i]; None for the other inputs.
pub(crate) fn contract(
    differentiable: &[bool],
    tangents: &[Option<Vec<f64>>],
    len: usize,
    b: usize,
    mut hessian: impl FnMut(usize, &mut Hessian) -> Result<()>,
) -> Result<Vec<Option<Vec<f64>>>> {
    let n = differentiable.len();
    let mut out: Vec<Option<Vec<f64>>> = differentiable
        .iter()
        .map(|&d| d.then(|| vec![0.0; b * len]))
        .collect();
    let moving: Vec<(usize, &Vec<f64>)> = tangents
        .iter()
        .enumerate()
        .filter_map(|(v, t)| t.as_ref().map(|t| (v, t)))
        .collect();
    if moving.is_empty() {
        return Ok(out);
    }
    let mut h = Hessian::new(n);
    for i in 0..len {
        h.clear();
        hessian(i, &mut h)?;
        for (u, o) in out.iter_mut().enumerate() {
            let Some(o) = o else { continue };
            for k in 0..b {
                let j = k * len + i;
                o[j] = moving.iter().map(|&(v, t)| h.get(u, v) * t[j]).sum();
            }
        }
    }
    Ok(out)
}

/// A tensor broadcast to `shape`, flattened.
pub(crate) fn spread(t: &DoubleTensor, shape: &[usize]) -> Result<Vec<f64>> {
    Ok(if t.shape() == shape {
        t.as_slice().to_vec()
    } else {
        t.broadcast_to(shape)?.into_vec()
    })
}

/// A batched tangent `[b, s...]` broadcast to `[b, shape...]`, flattened.
pub(crate) fn spread_tangent(
    t: Option<&DoubleTensor>,
    b: usize,
    shape: &[usize],
) -> Result<Option<Vec<f64>>> {
    t.map(|t| Ok(fit(t.clone(), b, shape)?.into_vec()))
        .transpose()
}

/// A value as f64s (1 and 0 for booleans).
pub(crate) fn as_f64(value: &Value) -> DoubleTensor {
    match value {
        Value::F64(t) => t.clone(),
        Value::I64(t) => t.to_double(),
        Value::Bool(t) => t.to_double_mask(),
    }
}

/// A flattened `[b, shape...]` Hessian-vector product summed to `[b, target...]`.
pub(crate) fn reduce(
    h: Option<Vec<f64>>,
    b: usize,
    shape: &[usize],
    target: &[usize],
) -> Result<Option<DoubleTensor>> {
    h.map(|h| reduce_to(DoubleTensor::new(&batched_shape(b, shape), h)?, target))
        .transpose()
}

/// An element-wise kind's Hessian-vector products, from [`elementwise`] over the broadcast of the value and the
/// parameters, summed to each input's shape.
pub(super) fn elementwise_hvp(
    kind: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
    value_tangent: Option<&DoubleTensor>,
    parameter_tangents: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<LogProbHvpParts> {
    let mut shape = Shape::from_slice(value.shape());
    for p in parameters {
        shape = broadcast_shape(&shape, p.shape())?;
    }
    let tangents: Vec<Option<Vec<f64>>> = std::iter::once(value_tangent)
        .chain(parameter_tangents.iter().copied())
        .map(|t| spread_tangent(t, b, &shape))
        .collect::<Result<_>>()?;
    let parts = elementwise(kind, value, parameters, &tangents, b, &shape)?;
    let mut hvps = parts.hvps.into_iter();
    let value_hvp = reduce(hvps.next().flatten(), b, &shape, value.shape())?;
    let parameter_hvps = hvps
        .zip(parameters)
        .map(|(h, p)| reduce(h, b, &shape, p.shape()))
        .collect::<Result<_>>()?;
    Ok((value_hvp, parameter_hvps))
}

/// The Dirichlet over the whole value (as its log density and gradient are): the gradient is ((c − 1)/x,
/// ln x − ψ(c) + ψ(Σc)) in (x, c), so the products are −(c − 1)/x² ẋ + ċ/x and ẋ/x − ψ₁(c) ċ + ψ₁(Σc) Σċ.
fn dirichlet_hvp(
    x: &DoubleTensor,
    concentration: &DoubleTensor,
    x_tangent: Option<&DoubleTensor>,
    c_tangent: Option<&DoubleTensor>,
    b: usize,
) -> Result<LogProbHvpParts> {
    let shape = x.shape();
    let c = spread(concentration, shape)?;
    let tx = spread_tangent(x_tangent, b, shape)?;
    let tc = spread_tangent(c_tangent, b, shape)?;
    let n = x.len();
    let xs = x.as_slice();
    let total = trigamma(c.iter().sum());
    let mut hx = vec![0.0; b * n];
    let mut hc = vec![0.0; b * n];
    for k in 0..b {
        let row = k * n..(k + 1) * n;
        let c_sum: f64 = tc.as_ref().map_or(0.0, |t| t[row.clone()].iter().sum());
        for i in 0..n {
            let j = k * n + i;
            let (dx, dc) = (
                tx.as_ref().map_or(0.0, |t| t[j]),
                tc.as_ref().map_or(0.0, |t| t[j]),
            );
            hx[j] = -(c[i] - 1.0) / (xs[i] * xs[i]) * dx + dc / xs[i];
            hc[j] = dx / xs[i] - trigamma(c[i]) * dc + total * c_sum;
        }
    }
    Ok((
        reduce(Some(hx), b, shape, shape)?,
        smallvec::smallvec![reduce(Some(hc), b, shape, concentration.shape())?],
    ))
}

/// The LKJ Cholesky density Σ_{i≥1} (K − i − 1 + 2η − 2) ln L_ii − ln N(K, η) per factor: the gradient is
/// (power_i / L_ii, 2 Σ ln L_ii − d ln N/dη), so the products are −power_i/L_ii² L̇_ii + 2η̇/L_ii on the
/// diagonal and Σ 2 L̇_ii/L_ii − d² ln N/dη² η̇, with
/// ln N = Σ_m m((2η − 2 + m) ln 2 + ln B(β_m, β_m)), β_m = η + (m − 1)/2, so
/// d² ln N/dη² = Σ_m 2m(ψ₁(β_m) − 2ψ₁(2β_m)).
fn lkj_cholesky_hvp(
    x: &DoubleTensor,
    eta: &DoubleTensor,
    x_tangent: Option<&DoubleTensor>,
    eta_tangent: Option<&DoubleTensor>,
    b: usize,
) -> Result<LogProbHvpParts> {
    let shape = x.shape();
    let (batch, k) = match shape {
        [batch @ .., rows, _] => (batch, *rows),
        _ => return Err(DistributionKind::LkjCholesky.wrong_parameters()),
    };
    let etas = spread(eta, batch)?;
    let tx = spread_tangent(x_tangent, b, shape)?;
    let te = spread_tangent(eta_tangent, b, batch)?;
    let (n, factors) = (x.len(), etas.len());
    let xs = x.as_slice();
    let mut hx = vec![0.0; b * n];
    let mut he = vec![0.0; b * factors];
    for (f, &eta) in etas.iter().enumerate() {
        let d2_normaliser: f64 = (1..k)
            .map(|level| {
                let m = (k - level) as f64;
                let beta = eta + (m - 1.0) / 2.0;
                2.0 * m * (trigamma(beta) - 2.0 * trigamma(2.0 * beta))
            })
            .sum();
        for s in 0..b {
            let de = te.as_ref().map_or(0.0, |t| t[s * factors + f]);
            let mut acc = -d2_normaliser * de;
            for i in 1..k {
                let at = f * k * k + i * k + i;
                let diagonal = xs[at];
                let power = (k - i - 1) as f64 + 2.0 * eta - 2.0;
                let dx = tx.as_ref().map_or(0.0, |t| t[s * n + at]);
                hx[s * n + at] = -power / (diagonal * diagonal) * dx + 2.0 * de / diagonal;
                acc += 2.0 * dx / diagonal;
            }
            he[s * factors + f] = acc;
        }
    }
    Ok((
        reduce(Some(hx), b, shape, shape)?,
        smallvec::smallvec![reduce(Some(he), b, batch, eta.shape())?],
    ))
}

/// The zero-sum normal density −Σ x²/(2σ²) − (N − 1)(ln σ + ½ ln 2π) per vector: the gradient is
/// (−x/σ², Σx²/σ³ − (N − 1)/σ), so the products are −ẋ/σ² + 2xσ̇/σ³ for each element and
/// 2 Σ x ẋ/σ³ + ((N − 1)/σ² − 3Σx²/σ⁴) σ̇ for σ.
fn zero_sum_normal_hvp(
    x: &DoubleTensor,
    sigma: &DoubleTensor,
    x_tangent: Option<&DoubleTensor>,
    sigma_tangent: Option<&DoubleTensor>,
    b: usize,
) -> Result<LogProbHvpParts> {
    let shape = x.shape();
    let (batch, n) = match shape {
        [batch @ .., n] if *n > 0 => (batch, *n),
        _ => return Err(DistributionKind::ZeroSumNormal.wrong_parameters()),
    };
    let sigmas = spread(sigma, batch)?;
    let tx = spread_tangent(x_tangent, b, shape)?;
    let ts = spread_tangent(sigma_tangent, b, batch)?;
    let (len, vectors) = (x.len(), sigmas.len());
    let xs = x.as_slice();
    let mut hx = vec![0.0; b * len];
    let mut hs = vec![0.0; b * vectors];
    for (v, &s) in sigmas.iter().enumerate() {
        let row = v * n..(v + 1) * n;
        let squares: f64 = xs[row.clone()].iter().map(|x| x * x).sum();
        let (s2, s3) = (s * s, s * s * s);
        for k in 0..b {
            let ds = ts.as_ref().map_or(0.0, |t| t[k * vectors + v]);
            let mut cross = 0.0;
            for i in row.clone() {
                let dx = tx.as_ref().map_or(0.0, |t| t[k * len + i]);
                hx[k * len + i] = -dx / s2 + 2.0 * xs[i] * ds / s3;
                cross += xs[i] * dx;
            }
            hs[k * vectors + v] =
                2.0 * cross / s3 + ((n - 1) as f64 / s2 - 3.0 * squares / (s2 * s2)) * ds;
        }
    }
    Ok((
        reduce(Some(hx), b, shape, shape)?,
        smallvec::smallvec![reduce(Some(hs), b, batch, sigma.shape())?],
    ))
}

/// Differentiates x − n p along logits: −n p ⊙ (t − Σ p t).
fn multinomial_logit_hvp(
    shape: &[usize],
    n: &DoubleTensor,
    logits: &DoubleTensor,
    tangent: Option<&DoubleTensor>,
    b: usize,
) -> Result<LogProbHvpParts> {
    let Some(tangent) = tangent else {
        return Ok((None, smallvec::smallvec![None, None]));
    };
    let column = |x: &DoubleTensor| -> Result<DoubleTensor> {
        let shape = x
            .shape()
            .iter()
            .map(|&d| d as isize)
            .chain([1])
            .collect::<Vec<_>>();
        Ok(x.reshape(&shape)?)
    };
    let p = logits.softmax(-1)?;
    let t = fit(tangent.clone(), b, shape)?;
    let mean = p.mul(&t)?.sum_axes(&[-1])?;
    let h = p.mul(&t.sub(&column(&mean)?)?)?.mul(&column(n)?)?.neg();
    Ok((
        None,
        smallvec::smallvec![None, Some(reduce_to(h, logits.shape())?)],
    ))
}

/// The multinomial in its probabilities, over the last axis: the gradient is x_i/p_i − Σx/Σp, so the product is
/// −x_i/p_i² ṗ_i + Σx/(Σp)² Σṗ (with 0/p taken as 0, as in first order).
fn multinomial_hvp(
    x: &DoubleTensor,
    p: &DoubleTensor,
    p_tangent: Option<&DoubleTensor>,
    b: usize,
) -> Result<LogProbHvpParts> {
    let shape = broadcast_shape(x.shape(), p.shape())?;
    let Some(tp) = spread_tangent(p_tangent, b, &shape)? else {
        return Ok((None, smallvec::smallvec![None, None]));
    };
    let xs = spread(x, &shape)?;
    let ps = spread(p, &shape)?;
    let count = shape.last().copied().unwrap_or(1).max(1);
    let len = xs.len();
    let mut hp = vec![0.0; b * len];
    for row in (0..len).step_by(count) {
        let range = row..row + count;
        let x_sum: f64 = xs[range.clone()].iter().sum();
        let p_sum: f64 = ps[range.clone()].iter().sum();
        let shared = x_sum / (p_sum * p_sum);
        for k in 0..b {
            let t = &tp[k * len..(k + 1) * len];
            let dp_sum: f64 = t[range.clone()].iter().sum();
            for i in range.clone() {
                let own = if xs[i] == 0.0 {
                    0.0
                } else {
                    -xs[i] / (ps[i] * ps[i]) * t[i]
                };
                hp[k * len + i] = own + shared * dp_sum;
            }
        }
    }
    Ok((
        None,
        smallvec::smallvec![None, reduce(Some(hp), b, &shape, p.shape())?],
    ))
}

/// The ordered logistic: each element's log probability is f(a, b) with a = η − c_{k−1} and b = η − c_k, the
/// cutpoints either side of category k (one of them only at the ends). With g = 1/(e^(a−b) − 1):
/// f = ln σ(a) + ln σ(−b) + ln(1 − e^(b−a)), f_aa = −σ(a)σ(−a) − g(1 + g), f_ab = g(1 + g),
/// f_bb = −σ(b)σ(−b) − g(1 + g); f = ln σ(−a) at k = 0 and ln σ(a) at the last category, each with second
/// derivative −σ(a)σ(−a).
fn ordered_logistic_hvp(
    k: &keanu_tensor::IntTensor,
    eta: &DoubleTensor,
    cutpoints: &DoubleTensor,
    tangents: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<LogProbHvpParts> {
    let Some((&count, leading)) = cutpoints.shape().split_last() else {
        return Err(DistributionKind::OrderedLogistic.wrong_parameters());
    };
    let batch = broadcast_shape(&broadcast_shape(k.shape(), eta.shape())?, leading)?;
    let mut cut_shape = batch.to_vec();
    cut_shape.push(count);
    let ks = k.broadcast_to(&batch)?.into_vec();
    let etas = spread(eta, &batch)?;
    let cuts = spread(cutpoints, &cut_shape)?;
    let te = spread_tangent(tangents[0], b, &batch)?;
    let tc = spread_tangent(tangents[1], b, &cut_shape)?;
    let len = ks.len();
    let mut he = vec![0.0; b * len];
    let mut hc = vec![0.0; b * len * count];
    let sigmoid = |z: f64| 1.0 / (1.0 + (-z).exp());
    for i in 0..len {
        let Ok(category) = usize::try_from(ks[i]) else {
            continue;
        };
        if category > count {
            continue;
        }
        let c = &cuts[i * count..(i + 1) * count];
        // (cutpoint index, its coordinate's second derivative) for a and b, and f_ab
        let (lower, upper, cross) = if category == 0 {
            let a = etas[i] - c[0];
            (None, Some((0, -sigmoid(a) * sigmoid(-a))), 0.0)
        } else if category == count {
            let a = etas[i] - c[count - 1];
            (Some((count - 1, -sigmoid(a) * sigmoid(-a))), None, 0.0)
        } else {
            let (a, bb) = (etas[i] - c[category - 1], etas[i] - c[category]);
            let g = 1.0 / (a - bb).exp_m1();
            let gg = g * (1.0 + g);
            (
                Some((category - 1, -sigmoid(a) * sigmoid(-a) - gg)),
                Some((category, -sigmoid(bb) * sigmoid(-bb) - gg)),
                gg,
            )
        };
        for s in 0..b {
            let de = te.as_ref().map_or(0.0, |t| t[s * len + i]);
            let dc = |j: usize| tc.as_ref().map_or(0.0, |t| t[(s * len + i) * count + j]);
            // the tangents of the two coordinates, η̇ − ċ_j
            let da = lower.map_or(0.0, |(j, _)| de - dc(j));
            let db = upper.map_or(0.0, |(j, _)| de - dc(j));
            let mut eta_out = 0.0;
            if let Some((j, faa)) = lower {
                let out = faa * da + cross * db;
                eta_out += out;
                hc[(s * len + i) * count + j] = -out;
            }
            if let Some((j, fbb)) = upper {
                let out = cross * da + fbb * db;
                eta_out += out;
                hc[(s * len + i) * count + j] = -out;
            }
            he[s * len + i] = eta_out;
        }
    }
    Ok((
        None,
        smallvec::smallvec![
            reduce(Some(he), b, &batch, eta.shape())?,
            reduce(Some(hc), b, &cut_shape, cutpoints.shape())?,
        ],
    ))
}

/// The ordered probit: each element's log probability depends on η and the cutpoints either side of its category,
/// with the Hessian in (η, c_(k−1), c_k) from [`keanu_distributions::OrderedProbit::category_derivatives`],
/// contracted with the tangents.
fn ordered_probit_hvp(
    k: &keanu_tensor::IntTensor,
    eta: &DoubleTensor,
    cutpoints: &DoubleTensor,
    tangents: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<LogProbHvpParts> {
    let Some((&count, leading)) = cutpoints.shape().split_last() else {
        return Err(DistributionKind::OrderedProbit.wrong_parameters());
    };
    let batch = broadcast_shape(&broadcast_shape(k.shape(), eta.shape())?, leading)?;
    let mut cut_shape = batch.to_vec();
    cut_shape.push(count);
    let ks = k.broadcast_to(&batch)?.into_vec();
    let etas = spread(eta, &batch)?;
    let cuts = spread(cutpoints, &cut_shape)?;
    let te = spread_tangent(tangents[0], b, &batch)?;
    let tc = spread_tangent(tangents[1], b, &cut_shape)?;
    let len = ks.len();
    let mut he = vec![0.0; b * len];
    let mut hc = vec![0.0; b * len * count];
    for i in 0..len {
        let c = &cuts[i * count..(i + 1) * count];
        let d = keanu_distributions::OrderedProbit::category_derivatives(ks[i], etas[i], c);
        if d.value == f64::NEG_INFINITY || !c.windows(2).all(|w| w[0] < w[1]) {
            continue;
        }
        let category = ks[i] as usize;
        // the cutpoint index of each input (none for η, and none for a missing cutpoint at the ends)
        let index = [
            None,
            category.checked_sub(1),
            (category < count).then_some(category),
        ];
        for s in 0..b {
            let tangent = |input: usize| match (input, index[input]) {
                (0, _) => te.as_ref().map_or(0.0, |t| t[s * len + i]),
                (_, Some(j)) => tc.as_ref().map_or(0.0, |t| t[(s * len + i) * count + j]),
                (_, None) => 0.0,
            };
            let moved = [tangent(0), tangent(1), tangent(2)];
            for (input, row) in d.hessian.iter().enumerate() {
                let out: f64 = row.iter().zip(&moved).map(|(h, t)| h * t).sum();
                match (input, index[input]) {
                    (0, _) => he[s * len + i] = out,
                    (_, Some(j)) => hc[(s * len + i) * count + j] = out,
                    (_, None) => {}
                }
            }
        }
    }
    Ok((
        None,
        smallvec::smallvec![
            reduce(Some(he), b, &batch, eta.shape())?,
            reduce(Some(hc), b, &cut_shape, cutpoints.shape())?,
        ],
    ))
}

/// The kinds over batches of events whose hand-written gradients are differentiated in forward mode by
/// `keanu_distributions` (see [`keanu_distributions::Wishart::hvp`]).
fn event_hvp(
    kind: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
    x_tangent: Option<&DoubleTensor>,
    tangents: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<LogProbHvpParts> {
    use DistributionKind as K;
    use keanu_distributions as d;
    // each tangent as [b, input shape]
    let x_tangent = match value {
        Value::F64(x) => x_tangent
            .map(|t| fit(t.clone(), b, x.shape()))
            .transpose()?,
        _ => None,
    };
    let fitted: Vec<Option<DoubleTensor>> = tangents
        .iter()
        .zip(parameters)
        .map(|(t, p)| match p {
            Value::F64(p) => t.map(|t| fit(t.clone(), b, p.shape())).transpose(),
            _ => Ok(None),
        })
        .collect::<Result<_>>()?;
    let t: Vec<Option<&DoubleTensor>> = fitted.iter().map(Option::as_ref).collect();
    let xt = x_tangent.as_ref();
    let all = |parts: (DoubleTensor, Vec<DoubleTensor>)| -> LogProbHvpParts {
        (Some(parts.0), parts.1.into_iter().map(Some).collect())
    };
    Ok(match (kind, parameters) {
        (K::MultivariateStudentT, &[nu, mu, scale]) => {
            let h = d::MultivariateStudentT::new(nu.f64()?, mu.f64()?, scale.f64()?).hvp(
                value.f64()?,
                xt,
                &t,
                b,
            )?;
            all((h.x, vec![h.nu, h.mu, h.scale]))
        }
        (K::MultivariateStudentTCholesky, &[nu, mu, scale]) => {
            let h = d::MultivariateStudentTCholesky::new(nu.f64()?, mu.f64()?, scale.f64()?).hvp(
                value.f64()?,
                xt,
                &t,
                b,
            )?;
            all((h.x, vec![h.nu, h.mu, h.scale_cholesky]))
        }
        (K::Wishart, &[nu, scale]) => {
            let h = d::Wishart::new(nu.f64()?, scale.f64()?).hvp(value.f64()?, xt, &t, b)?;
            all((h.x, vec![h.nu, h.scale]))
        }
        (K::WishartCholesky, &[nu, scale]) => {
            let h =
                d::WishartCholesky::new(nu.f64()?, scale.f64()?).hvp(value.f64()?, xt, &t, b)?;
            all((h.x, vec![h.nu, h.scale_cholesky]))
        }
        (K::InverseWishart, &[nu, scale]) => {
            let h = d::InverseWishart::new(nu.f64()?, scale.f64()?).hvp(value.f64()?, xt, &t, b)?;
            all((h.x, vec![h.nu, h.scale]))
        }
        (K::InverseWishartCholesky, &[nu, scale]) => {
            let h = d::InverseWishartCholesky::new(nu.f64()?, scale.f64()?).hvp(
                value.f64()?,
                xt,
                &t,
                b,
            )?;
            all((h.x, vec![h.nu, h.scale_cholesky]))
        }
        (K::MatrixNormal, &[mu, row, column]) => {
            let h = d::MatrixNormal::new(mu.f64()?, row.f64()?, column.f64()?).hvp(
                value.f64()?,
                xt,
                &t,
                b,
            )?;
            all((h.x, vec![h.mu, h.row_covariance, h.column_covariance]))
        }
        (K::VonMisesFisher, &[mu, kappa]) => {
            let h = d::VonMisesFisher::new(mu.f64()?, kappa.f64()?).hvp(value.f64()?, xt, &t, b)?;
            all((h.x, vec![h.mu, h.kappa]))
        }
        (K::DirichletMultinomial, &[n, alpha]) => {
            let h =
                d::DirichletMultinomial::new(n.i64()?, alpha.f64()?).hvp(value.i64()?, t[1], b)?;
            (None, smallvec::smallvec![None, Some(h.alpha)])
        }
        (K::PlackettLuce, &[worths]) => {
            let h = d::PlackettLuce::new(worths.f64()?).hvp(value.i64()?, t[0], b)?;
            (None, smallvec::smallvec![Some(h.worths)])
        }
        _ => return Err(kind.wrong_parameters()),
    })
}

/// A categorical distribution's broadcast shape, its probabilities spread to that shape (one flat vector per
/// category), and each element's category (`None` where the value is not a category).
type CategoricalParts = (Shape, Vec<Vec<f64>>, Vec<Option<usize>>);

/// The probabilities of a categorical distribution broadcast to the shape of the value and every probability,
/// flattened, with that shape, and each element's category (`None` where it is not a category).
fn categorical_parts(
    x: &keanu_tensor::IntTensor,
    parameters: &[&Value],
) -> Result<CategoricalParts> {
    let mut shape = Shape::from_slice(x.shape());
    for p in parameters {
        shape = broadcast_shape(&shape, p.shape())?;
    }
    let probabilities = parameters
        .iter()
        .map(|p| spread(p.f64()?, &shape))
        .collect::<Result<Vec<_>>>()?;
    let categories = x
        .broadcast_to(&shape)?
        .into_vec()
        .into_iter()
        .map(|c| usize::try_from(c).ok().filter(|&c| c < parameters.len()))
        .collect();
    Ok((shape, probabilities, categories))
}

/// The gradient of the categorical log probability ln(p_x / Σ_j p_j) with respect to each probability:
/// [j = x] / p_x − 1 / Σ p, element-wise, summed to each probability's shape (zero where x is not a category).
pub(super) fn categorical_gradient(
    x: &keanu_tensor::IntTensor,
    parameters: &[&Value],
) -> Result<super::distribution::LogProbGradientParts> {
    let (shape, probabilities, categories) = categorical_parts(x, parameters)?;
    let len = categories.len();
    let mut gradients = vec![vec![0.0; len]; parameters.len()];
    for (i, category) in categories.iter().enumerate() {
        let Some(c) = *category else { continue };
        let total: f64 = probabilities.iter().map(|p| p[i]).sum();
        for (j, gradient) in gradients.iter_mut().enumerate() {
            gradient[i] = f64::from(u8::from(j == c)) / probabilities[c][i] - 1.0 / total;
        }
    }
    let mut parts = SmallVec::new();
    for (gradient, p) in gradients.into_iter().zip(parameters) {
        let summed = reduce(Some(gradient), 1, &shape, p.shape())?.expect("a gradient was given");
        parts.push(Some(summed.reshape(
            &p.shape().iter().map(|&d| d as isize).collect::<Vec<_>>(),
        )?));
    }
    Ok((None, parts))
}

/// The categorical log probability's second derivatives in its probabilities, −[j = k = x] / p_x² + 1 / (Σ p)²,
/// contracted with the probabilities' tangents.
fn categorical_hvp(
    x: &keanu_tensor::IntTensor,
    parameters: &[&Value],
    tangents: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<LogProbHvpParts> {
    let (shape, probabilities, categories) = categorical_parts(x, parameters)?;
    let len = categories.len();
    let tangents = tangents
        .iter()
        .map(|t| spread_tangent(*t, b, &shape))
        .collect::<Result<Vec<_>>>()?;
    let mut products = vec![vec![0.0; b * len]; parameters.len()];
    for (i, category) in categories.iter().enumerate() {
        let Some(c) = *category else { continue };
        let total: f64 = probabilities.iter().map(|p| p[i]).sum();
        for s in 0..b {
            let tangent = |j: usize| tangents[j].as_ref().map_or(0.0, |t| t[s * len + i]);
            let shared = (0..parameters.len()).map(tangent).sum::<f64>() / (total * total);
            for (j, product) in products.iter_mut().enumerate() {
                product[s * len + i] = shared;
                if j == c {
                    product[s * len + i] -=
                        tangent(c) / (probabilities[c][i] * probabilities[c][i]);
                }
            }
        }
    }
    let mut parts = SmallVec::new();
    for (product, p) in products.into_iter().zip(parameters) {
        parts.push(reduce(Some(product), b, &shape, p.shape())?);
    }
    Ok((None, parts))
}

/// The gradient of the categorical log probability η_x − logsumexp(η) with respect to each logit:
/// [j = x] − softmax(η)ⱼ, element-wise, summed to each logit's shape (zero where x is not a category).
pub(super) fn categorical_logit_gradient(
    x: &keanu_tensor::IntTensor,
    parameters: &[&Value],
) -> Result<super::distribution::LogProbGradientParts> {
    let logits: Vec<&DoubleTensor> = parameters.iter().map(|p| p.f64()).collect::<Result<_>>()?;
    let gradients = keanu_distributions::CategoricalLogit::new(logits)?.d_log_prob(x)?;
    let mut parts = SmallVec::new();
    for (gradient, p) in gradients.into_iter().zip(parameters) {
        // each gradient has the broadcast shape; sum it to the logit's shape as a batch of one, then drop the batch
        let shape = Shape::from_slice(gradient.shape());
        let summed =
            reduce(Some(gradient.into_vec()), 1, &shape, p.shape())?.expect("a gradient was given");
        parts.push(Some(summed.reshape(
            &p.shape().iter().map(|&d| d as isize).collect::<Vec<_>>(),
        )?));
    }
    Ok((None, parts))
}

/// The categorical log probability's second derivatives in its logits, −(diag(s) − s sᵀ) with s = softmax(η)
/// (the same for every category x), contracted with the logits' tangents: −sⱼ ṫⱼ + sⱼ Σₖ sₖ ṫₖ.
fn categorical_logit_hvp(
    x: &keanu_tensor::IntTensor,
    parameters: &[&Value],
    tangents: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<LogProbHvpParts> {
    let (shape, logits, categories) = categorical_parts(x, parameters)?;
    let len = categories.len();
    let tangents = tangents
        .iter()
        .map(|t| spread_tangent(*t, b, &shape))
        .collect::<Result<Vec<_>>>()?;
    let count = parameters.len();
    let mut products = vec![vec![0.0; b * len]; count];
    let mut softmax = vec![0.0; count];
    for (i, category) in categories.iter().enumerate() {
        if category.is_none() {
            continue;
        }
        let total = keanu_tensor::special::log_sum_exp(logits.iter().map(|l| l[i]));
        for (s, l) in softmax.iter_mut().zip(&logits) {
            *s = (l[i] - total).exp();
        }
        for k in 0..b {
            let tangent = |j: usize| tangents[j].as_ref().map_or(0.0, |t| t[k * len + i]);
            let mean: f64 = (0..count).map(|j| softmax[j] * tangent(j)).sum();
            for (j, product) in products.iter_mut().enumerate() {
                product[k * len + i] = softmax[j] * (mean - tangent(j));
            }
        }
    }
    let mut parts = SmallVec::new();
    for (product, p) in products.into_iter().zip(parameters) {
        parts.push(reduce(Some(product), b, &shape, p.shape())?);
    }
    Ok((None, parts))
}

/// The KDE log density ln Σ_s exp(−d_s²/2) + c with d_s = (x − s)/h, whose second derivative is
/// (Σ w_s (d_s² − 1) − (Σ w_s d_s)²)/h² with the normalised weights w_s.
fn kde_hvp(
    kde: &keanu_distributions::Kde,
    x: &DoubleTensor,
    tangent: &DoubleTensor,
    b: usize,
) -> Result<DoubleTensor> {
    let h = kde.bandwidth;
    let second: Vec<f64> = x
        .as_slice()
        .iter()
        .map(|&x| {
            let exponents: Vec<f64> = kde
                .samples
                .iter()
                .map(|&s| -0.5 * ((x - s) / h).powi(2))
                .collect();
            let max = exponents.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let (mut total, mut first, mut squares) = (0.0, 0.0, 0.0);
            for (&s, &e) in kde.samples.iter().zip(&exponents) {
                let w = (e - max).exp();
                let d = (x - s) / h;
                total += w;
                first += w * d;
                squares += w * (d * d - 1.0);
            }
            let mean = first / total;
            (squares / total - mean * mean) / (h * h)
        })
        .collect();
    let second = DoubleTensor::new(x.shape(), second)?;
    let t = fit(tangent.clone(), b, x.shape())?;
    Ok(lift(&t, x.rank())?.mul(&second)?)
}

/// The linear Gaussian state space model: the Kalman filter's gradient differentiated in forward mode (see
/// [`keanu_distributions::LinearGaussianStateSpace::hvp`]), with covariance tangents taken to be symmetric.
fn linear_gaussian_state_space_hvp(
    x: &DoubleTensor,
    parameters: &[&Value],
    x_tangent: Option<&DoubleTensor>,
    tangents: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<LogProbHvpParts> {
    let &[m0, p0, f, q, h, r] = parameters else {
        return Err(DistributionKind::LinearGaussianStateSpace.wrong_parameters());
    };
    let model = keanu_distributions::LinearGaussianStateSpace::new(
        m0.f64()?,
        p0.f64()?,
        f.f64()?,
        q.f64()?,
        h.f64()?,
        r.f64()?,
    );
    let x_tangent = x_tangent
        .map(|t| fit(t.clone(), b, x.shape()))
        .transpose()?;
    let tangents: Vec<Option<DoubleTensor>> = tangents
        .iter()
        .zip(parameters)
        .map(|(t, p)| t.map(|t| fit(t.clone(), b, p.shape())).transpose())
        .collect::<Result<_>>()?;
    let tangents: Vec<Option<&DoubleTensor>> = tangents.iter().map(Option::as_ref).collect();
    let g = model.hvp(x, x_tangent.as_ref(), &tangents, b)?;
    Ok((
        Some(g.x),
        smallvec::smallvec![
            Some(g.initial_mean),
            Some(g.initial_covariance),
            Some(g.transition_matrix),
            Some(g.transition_covariance),
            Some(g.observation_matrix),
            Some(g.observation_covariance),
        ],
    ))
}

/// The Gaussian random walk, AR(p) and GARCH(1,1): their gradients differentiated in forward mode (see
/// [`keanu_distributions::GaussianRandomWalk::hvp`]).
fn process_hvp(
    kind: &DistributionKind,
    x: &DoubleTensor,
    parameters: &[&Value],
    x_tangent: Option<&DoubleTensor>,
    tangents: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<LogProbHvpParts> {
    let p: Vec<&DoubleTensor> = parameters.iter().map(|p| p.f64()).collect::<Result<_>>()?;
    let x_tangent = x_tangent
        .map(|t| fit(t.clone(), b, x.shape()))
        .transpose()?;
    let tangents: Vec<Option<DoubleTensor>> = tangents
        .iter()
        .zip(&p)
        .map(|(t, p)| t.map(|t| fit(t.clone(), b, p.shape())).transpose())
        .collect::<Result<_>>()?;
    let tangents: Vec<Option<&DoubleTensor>> = tangents.iter().map(Option::as_ref).collect();
    let x_tangent = x_tangent.as_ref();
    let (hx, hp): (DoubleTensor, Vec<DoubleTensor>) = match (kind, p.as_slice()) {
        (DistributionKind::GaussianRandomWalk, &[mu, sigma, m0, s0]) => {
            let g = keanu_distributions::GaussianRandomWalk::new(mu, sigma, m0, s0)
                .hvp(x, x_tangent, &tangents, b)?;
            (g.x, vec![g.mu, g.sigma, g.init_mu, g.init_sigma])
        }
        (DistributionKind::Autoregressive, &[c, rho, sigma, m0, s0]) => {
            let g = keanu_distributions::Autoregressive::new(c, rho, sigma, m0, s0)
                .hvp(x, x_tangent, &tangents, b)?;
            (
                g.x,
                vec![g.constant, g.rho, g.sigma, g.init_mu, g.init_sigma],
            )
        }
        (DistributionKind::Garch11, &[omega, alpha, beta, v0]) => {
            let g = keanu_distributions::Garch11::new(omega, alpha, beta, v0)
                .hvp(x, x_tangent, &tangents, b)?;
            (g.x, vec![g.omega, g.alpha, g.beta, g.initial_vol])
        }
        _ => return Err(kind.wrong_parameters()),
    };
    Ok((Some(hx), hp.into_iter().map(Some).collect()))
}

/// The multivariate Gaussian, with r = x − μ, P = Σ⁻¹ and a = P r: the gradient is (−a, a, ½(a aᵀ − P)) in
/// (x, μ, Σ), so along (ẋ, μ̇, Σ̇), with ȧ = P (ṙ − Σ̇ a), the products are (−ȧ, ȧ, ½(ȧ aᵀ + a ȧᵀ + P Σ̇ P)),
/// summed over batch dimensions to each input's shape. Σ̇ is taken to be symmetric, as the tangent of a covariance
/// is.
fn multivariate_gaussian_hvp(
    x: &DoubleTensor,
    mu: &DoubleTensor,
    covariance: &DoubleTensor,
    x_tangent: Option<&DoubleTensor>,
    tangents: &[Option<&DoubleTensor>],
) -> Result<LogProbHvpParts> {
    let precision = covariance.cholesky()?.cholesky_inverse()?;
    let r = x.sub(mu)?;
    let column = |t: &DoubleTensor| -> Result<DoubleTensor> {
        let shape: Vec<isize> = t
            .shape()
            .iter()
            .map(|&d| d as isize)
            .chain(std::iter::once(1))
            .collect();
        Ok(t.reshape(&shape)?)
    };
    // a = P r, [batch..., n, 1], with every batch dimension of x, μ and Σ
    let a = precision.matmul(&column(&r)?)?;
    let full = a.rank();
    // ṙ = ẋ − μ̇
    let rank = r.rank();
    let from_x = x_tangent
        .map(|t| lift(t, rank).map(|t| t.into_owned()))
        .transpose()?;
    let from_mu = tangents[0]
        .map(|t| lift(t, rank).map(|t| t.neg()))
        .transpose()?;
    let r_dot = match (from_x, from_mu) {
        (Some(x), Some(m)) => Some(x.add(&m)?),
        (x, None) => x,
        (None, m) => m,
    };
    let cov_dot = tangents[1]
        .map(|t| lift(t, full).map(|t| t.into_owned()))
        .transpose()?;
    let mut inner = r_dot
        .map(|r| -> Result<DoubleTensor> { Ok(lift(&column(&r)?, full)?.into_owned()) })
        .transpose()?;
    if let Some(cd) = &cov_dot {
        let shifted = cd.matmul(&a)?;
        inner = Some(match inner {
            Some(i) => i.sub(&shifted)?,
            None => shifted.neg(),
        });
    }
    let Some(inner) = inner else {
        return Ok((None, smallvec::smallvec![None, None]));
    };
    let a_dot = precision.matmul(&inner)?;
    // [b, batch..., n, 1] -> [b, batch..., n]
    let vector_shape: Vec<isize> = a_dot.shape()[..a_dot.rank() - 1]
        .iter()
        .map(|&d| d as isize)
        .collect();
    let a_dot_vector = a_dot.reshape(&vector_shape)?;
    let hx = reduce_to(a_dot_vector.neg(), x.shape())?;
    let hm = reduce_to(a_dot_vector, mu.shape())?;
    let outer = a_dot.matmul_transposed(&a, false, true)?;
    let mut hc = outer.add(&outer.transpose()?)?;
    if let Some(cd) = &cov_dot {
        hc = hc.add(&precision.matmul(cd)?.matmul(&precision)?)?;
    }
    let hc = reduce_to(hc * 0.5, covariance.shape())?;
    Ok((Some(hx), smallvec::smallvec![Some(hm), Some(hc)]))
}

/// With r = x − μ and P the symmetric precision read from its lower triangle, the gradient is
/// (−P r, P r, ½(P⁻¹ − r rᵀ)). The matrix gradient folds symmetric entries into the lower triangle.
fn gaussian_precision_hvp(
    x: &DoubleTensor,
    mu: &DoubleTensor,
    precision: &DoubleTensor,
    value_tangent: Option<&DoubleTensor>,
    tangents: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<LogProbHvpParts> {
    if value_tangent.is_none() && tangents.iter().all(Option::is_none) {
        return Ok((None, smallvec::smallvec![None, None]));
    }
    let residual = x.sub(mu)?;
    let n = residual.shape()[residual.rank() - 1];
    let mut batch = broadcast_shape(
        &residual.shape()[..residual.rank() - 1],
        &precision.shape()[..precision.rank() - 2],
    )?;
    batch.push(n);
    let column = |t: &DoubleTensor| -> Result<DoubleTensor> {
        let shape = t
            .shape()
            .iter()
            .map(|&d| d as isize)
            .chain([1])
            .collect::<Vec<_>>();
        Ok(t.reshape(&shape)?)
    };
    let tangent = |t: Option<&DoubleTensor>| -> Result<DoubleTensor> {
        match t {
            Some(t) => fit(t.clone(), b, &batch),
            None => Ok(DoubleTensor::zeros(&batched_shape(b, &batch))),
        }
    };
    let r = column(&residual)?;
    let dr = column(&tangent(value_tangent)?.sub(&tangent(tangents[0])?)?)?;
    let symmetric = |lower: &DoubleTensor| -> Result<DoubleTensor> {
        Ok(lower.tril(0)?.add(&lower.tril(-1)?.transpose()?)?)
    };
    let p = symmetric(precision)?;
    let mut da = p.matmul(&dr)?;
    let cross = dr.matmul_transposed(&r, false, true)?;
    let mut hp = cross.add(&cross.transpose()?)?;
    if let Some(dp) = tangents[1] {
        let dp = symmetric(&*lift(dp, batch.len() + 1)?)?;
        let covariance = precision.cholesky()?.cholesky_inverse()?;
        da = da.add(&dp.matmul(&r)?)?;
        hp = hp.add(&covariance.matmul(&dp)?.matmul(&covariance)?)?;
    }
    let vector = da.shape()[..da.rank() - 1]
        .iter()
        .map(|&d| d as isize)
        .collect::<Vec<_>>();
    let da = da.reshape(&vector)?;
    let hp = hp.tril(0)?.add(&hp.tril(-1)?)? * -0.5;
    Ok((
        Some(reduce_to(da.neg(), x.shape())?),
        smallvec::smallvec![
            Some(reduce_to(da, mu.shape())?),
            Some(reduce_to(hp, precision.shape())?)
        ],
    ))
}

/// Pv for the ICAR's precision P: the graph Laplacian plus 1 on the diagonal of each island.
fn icar_precision_times(a: &keanu_distributions::Adjacency, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for &(i, j) in a.edges() {
        let d = v[i] - v[j];
        out[i] += d;
        out[j] -= d;
    }
    for (i, &d) in a.degrees().iter().enumerate() {
        if d == 0.0 {
            out[i] += v[i];
        }
    }
    out
}

/// The ICAR density −½ xᵀPx/σ² − r ln σ: the gradient is (−Px/σ², xᵀPx/σ³ − r/σ), so the products are
/// −Pẋ/σ² + 2Px σ̇/σ³ for x and 2 (Px)·ẋ/σ³ + (r/σ² − 3 xᵀPx/σ⁴) σ̇ for σ.
fn icar_hvp(
    kind: &DistributionKind,
    a: &keanu_distributions::Adjacency,
    x: &DoubleTensor,
    sigma: &DoubleTensor,
    x_tangent: Option<&DoubleTensor>,
    sigma_tangent: Option<&DoubleTensor>,
    b: usize,
) -> Result<LogProbHvpParts> {
    let n = a.len();
    let &[s] = sigma.as_slice() else {
        return Err(kind.wrong_parameters());
    };
    let xs = x.as_slice();
    let px = icar_precision_times(a, xs);
    let quadratic: f64 = px.iter().zip(xs).map(|(p, x)| p * x).sum();
    let rank = a.icar_rank() as f64;
    let tx = spread_tangent(x_tangent, b, x.shape())?;
    let ts = spread_tangent(sigma_tangent, b, sigma.shape())?;
    let (s2, s3) = (s * s, s * s * s);
    let mut hx = vec![0.0; b * n];
    let mut hs = vec![0.0; b];
    for k in 0..b {
        let ds = ts.as_ref().map_or(0.0, |t| t[k]);
        let mut cross = 0.0;
        let p_dx = tx
            .as_ref()
            .map(|t| icar_precision_times(a, &t[k * n..(k + 1) * n]));
        for i in 0..n {
            let pdx = p_dx.as_ref().map_or(0.0, |p| p[i]);
            hx[k * n + i] = -pdx / s2 + 2.0 * px[i] * ds / s3;
            cross += tx.as_ref().map_or(0.0, |t| t[k * n + i]) * px[i];
        }
        hs[k] = 2.0 * cross / s3 + (rank / s2 - 3.0 * quadratic / (s2 * s2)) * ds;
    }
    Ok((
        reduce(Some(hx), b, x.shape(), x.shape())?,
        smallvec::smallvec![reduce(Some(hs), b, sigma.shape(), sigma.shape())?],
    ))
}

/// Dv and Wv for a proper CAR.
fn car_parts(a: &keanu_distributions::Adjacency, v: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let dv = a.degrees().iter().zip(v).map(|(d, v)| d * v).collect();
    let mut wv = vec![0.0; v.len()];
    for &(i, j) in a.edges() {
        wv[i] += v[j];
        wv[j] += v[i];
    }
    (dv, wv)
}

/// The proper CAR density ½ Σ ln(1 − αλ) + ½ n ln τ − ½ τ xᵀ(D − αW)x (+ constants), with M = D − αW: the
/// gradient is (−τMx, n/(2τ) − ½ xᵀMx, −½ Σ λ/(1 − αλ) + ½ τ xᵀWx), so with second derivatives
/// H_xx = −τM, H_xτ = −Mx, H_xα = τWx, H_ττ = −n/(2τ²), H_τα = ½ xᵀWx and H_αα = −½ Σ λ²/(1 − αλ)².
fn car_hvp(
    kind: &DistributionKind,
    a: &keanu_distributions::Adjacency,
    x: &DoubleTensor,
    parameters: &[&Value],
    x_tangent: Option<&DoubleTensor>,
    parameter_tangents: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<LogProbHvpParts> {
    let &[tau, alpha] = parameters else {
        return Err(kind.wrong_parameters());
    };
    let (tau, alpha) = (tau.f64()?, alpha.f64()?);
    let n = a.len();
    let (&[t], &[al]) = (tau.as_slice(), alpha.as_slice()) else {
        return Err(kind.wrong_parameters());
    };
    let lambda = a.car_eigenvalues()?;
    let xs = x.as_slice();
    let (dx, wx) = car_parts(a, xs);
    let mx: Vec<f64> = dx.iter().zip(&wx).map(|(d, w)| d - al * w).collect();
    let xwx: f64 = wx.iter().zip(xs).map(|(w, x)| w * x).sum();
    let h_tt = -0.5 * n as f64 / (t * t);
    let h_ta = 0.5 * xwx;
    let h_aa = -0.5
        * lambda
            .iter()
            .map(|l| (l / (1.0 - al * l)).powi(2))
            .sum::<f64>();
    let tx = spread_tangent(x_tangent, b, x.shape())?;
    let tt = spread_tangent(parameter_tangents[0], b, tau.shape())?;
    let ta = spread_tangent(
        parameter_tangents.get(1).copied().flatten(),
        b,
        alpha.shape(),
    )?;
    let mut hx = vec![0.0; b * n];
    let mut h_tau = vec![0.0; b];
    let mut h_alpha = vec![0.0; b];
    for k in 0..b {
        let d_tau = tt.as_ref().map_or(0.0, |v| v[k]);
        let d_alpha = ta.as_ref().map_or(0.0, |v| v[k]);
        let (m_dx, from_x_tau, from_x_alpha) = match &tx {
            Some(v) => {
                let v = &v[k * n..(k + 1) * n];
                let (dv, wv) = car_parts(a, v);
                let m_dx: Vec<f64> = dv.iter().zip(&wv).map(|(d, w)| d - al * w).collect();
                let from_tau: f64 = mx.iter().zip(v).map(|(m, v)| m * v).sum();
                let from_alpha: f64 = wx.iter().zip(v).map(|(w, v)| w * v).sum();
                (Some(m_dx), -from_tau, t * from_alpha)
            }
            None => (None, 0.0, 0.0),
        };
        for i in 0..n {
            let curvature = m_dx.as_ref().map_or(0.0, |m| -t * m[i]);
            hx[k * n + i] = curvature - mx[i] * d_tau + t * wx[i] * d_alpha;
        }
        h_tau[k] = from_x_tau + h_tt * d_tau + h_ta * d_alpha;
        h_alpha[k] = from_x_alpha + h_ta * d_tau + h_aa * d_alpha;
    }
    Ok((
        reduce(Some(hx), b, x.shape(), x.shape())?,
        smallvec::smallvec![
            reduce(Some(h_tau), b, tau.shape(), tau.shape())?,
            reduce(Some(h_alpha), b, alpha.shape(), alpha.shape())?,
        ],
    ))
}
