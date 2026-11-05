//! Second derivatives of distributions' log densities: Hessian-vector products in the value and the parameters,
//! for exact Hessian-vector products of a joint log probability (see
//! [`Parameterisation::hvp`](crate::Parameterisation::hvp)).
//!
//! Kept apart from the distribution table so that each kind's rule can be added without touching it. Kinds without
//! a rule return [`GraphError::NoSecondOrderRule`].

use super::distribution::DistributionKind;
use crate::autodiff::{batched_shape, fit, lift, reduce_to};
use crate::error::{GraphError, Result};
use crate::value::Value;
use keanu_tensor::DoubleTensor;
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
    /// [`GraphError::NoSecondOrderRule`] naming the kind for a distribution without a rule (every kind but the
    /// Gaussian and the multivariate Gaussian, for now).
    pub(crate) fn log_prob_hvp(
        &self,
        value: &Value,
        parameters: &[&Value],
        value_tangent: Option<&DoubleTensor>,
        parameter_tangents: &[Option<&DoubleTensor>],
        b: usize,
    ) -> Result<LogProbHvpParts> {
        match self {
            DistributionKind::Gaussian => {
                let &[mu, sigma] = parameters else {
                    return Err(self.wrong_parameters());
                };
                gaussian_hvp(
                    value.f64()?,
                    mu.f64()?,
                    sigma.f64()?,
                    value_tangent,
                    parameter_tangents,
                    b,
                )
            }
            DistributionKind::MultivariateGaussian => {
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
            _ => Err(GraphError::NoSecondOrderRule(format!(
                "the {} distribution",
                self.name()
            ))),
        }
    }
}

/// A batched tangent `[b, s...]` broadcast to `[b, shape...]`, or zeros for a zero tangent.
fn broadcast_tangent(t: Option<&DoubleTensor>, b: usize, shape: &[usize]) -> Result<DoubleTensor> {
    match t {
        Some(t) => fit(t.clone(), b, shape),
        None => Ok(DoubleTensor::zeros(&batched_shape(b, shape))),
    }
}

/// The Gaussian, element-wise with u = (x − μ)/σ: the gradient is (−u/σ, u/σ, (u² − 1)/σ) in (x, μ, σ), and its
/// derivatives are ∂²/∂x² = ∂²/∂μ² = −1/σ², ∂²/∂x∂μ = 1/σ², ∂²/∂x∂σ = −∂²/∂μ∂σ = 2u/σ² and
/// ∂²/∂σ² = (1 − 3u²)/σ².
fn gaussian_hvp(
    x: &DoubleTensor,
    mu: &DoubleTensor,
    sigma: &DoubleTensor,
    x_tangent: Option<&DoubleTensor>,
    tangents: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<LogProbHvpParts> {
    let shape = x.shape();
    let (mu_b, sigma_b) = (mu.broadcast_to(shape)?, sigma.broadcast_to(shape)?);
    let tx = broadcast_tangent(x_tangent, b, shape)?;
    let tm = broadcast_tangent(tangents[0], b, shape)?;
    let ts = broadcast_tangent(tangents[1], b, shape)?;
    let n = x.len();
    let (xs, ms, ss) = (x.as_slice(), mu_b.as_slice(), sigma_b.as_slice());
    let (txs, tms, tss) = (tx.as_slice(), tm.as_slice(), ts.as_slice());
    let mut hx = vec![0.0; b * n];
    let mut hs = vec![0.0; b * n];
    for k in 0..b {
        for i in 0..n {
            let j = k * n + i;
            let s = ss[i];
            let u = (xs[i] - ms[i]) / s;
            let s2 = s * s;
            let (dr, ds) = (txs[j] - tms[j], tss[j]);
            hx[j] = (-dr + 2.0 * u * ds) / s2;
            hs[j] = (2.0 * u * dr + (1.0 - 3.0 * u * u) * ds) / s2;
        }
    }
    let batched = batched_shape(b, shape);
    let hx = DoubleTensor::new(&batched, hx)?;
    let hm = reduce_to(hx.neg(), mu.shape())?;
    let hs = reduce_to(DoubleTensor::new(&batched, hs)?, sigma.shape())?;
    Ok((Some(hx), smallvec::smallvec![Some(hm), Some(hs)]))
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
