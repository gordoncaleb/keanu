//! Distributions over vectors.

use crate::continuous::sample_gamma;
use crate::error::{DistributionError, Result, invalid_parameter};
use crate::param::{IntoParam, Param};
use crate::rng::KeanuRng;
use crate::util::{digamma, ln_gamma, map, safe_ln_times};
use keanu_tensor::special::log_beta;
use keanu_tensor::{DoubleTensor, IntTensor, TensorError, broadcast_shape};
use std::f64::consts::{LN_2, PI};

/// Multivariate normal distribution over the last dimension. Leading dimensions of x, mu and covariance are
/// batch dimensions and broadcast.
#[derive(Debug, Clone, PartialEq)]
pub struct MultivariateGaussian<'a> {
    /// Mean, shape [..., n].
    pub mu: Param<'a>,
    /// Covariance, shape [..., n, n].
    pub covariance: Param<'a>,
}

/// Derivatives of [`MultivariateGaussian::log_prob`], each shaped like the corresponding input.
#[derive(Debug, Clone, PartialEq)]
pub struct MultivariateGaussianGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `mu`.
    pub mu: DoubleTensor,
    /// With respect to `covariance`.
    pub covariance: DoubleTensor,
}

fn batch(shape: &[usize], event_rank: usize) -> &[usize] {
    &shape[..shape.len() - event_rank]
}

fn with_suffix(prefix: &[usize], suffix: &[usize]) -> Vec<isize> {
    prefix.iter().chain(suffix).map(|&d| d as isize).collect()
}

/// Reduces a gradient computed at the broadcast shape to `target_shape` (which broadcasts to it): sums the
/// extra leading dimensions and every dimension that the target has as size 1.
fn sum_over_batch(batched: &DoubleTensor, target_shape: &[usize]) -> Result<DoubleTensor> {
    let extra = batched.rank() - target_shape.len();
    let axes: Vec<isize> = (0..batched.rank())
        .filter(|&axis| {
            axis < extra || (target_shape[axis - extra] == 1 && batched.shape()[axis] != 1)
        })
        .map(|axis| axis as isize)
        .collect();
    let target: Vec<isize> = target_shape.iter().map(|&d| d as isize).collect();
    Ok(batched.sum_axes(&axes)?.reshape(&target)?)
}

impl<'a> MultivariateGaussian<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(mu: impl IntoParam<'a, f64>, covariance: impl IntoParam<'a, f64>) -> Self {
        Self {
            mu: mu.into_param(),
            covariance: covariance.into_param(),
        }
    }

    /// Checks the shapes and returns the broadcast shape of a sample.
    fn validate(&self, x_shape: &[usize]) -> Result<Vec<usize>> {
        let (mu, cov) = (self.mu.shape(), self.covariance.shape());
        let invalid = |m: &str| Err(DistributionError::InvalidValue(m.to_string()));
        if x_shape.is_empty() {
            return invalid("x must be at least a vector; use a Gaussian for scalar x");
        }
        if mu.is_empty() {
            return invalid("mu must be at least a vector");
        }
        if cov.len() < 2 {
            return invalid("covariance must be a matrix or a batch of matrices");
        }
        let n = x_shape[x_shape.len() - 1];
        if mu[mu.len() - 1] != n || cov[cov.len() - 1] != n || cov[cov.len() - 2] != n {
            return invalid(
                "x, mu and covariance must have the same event size and covariance must be square",
            );
        }
        let batch_shape = broadcast_shape(
            &broadcast_shape(batch(x_shape, 1), batch(mu, 1))?,
            batch(cov, 2),
        )?;
        Ok(batch_shape.iter().copied().chain([n]).collect())
    }

    /// Total log density of every vector in x (summed over batch dimensions). -infinity if the covariance is not
    /// positive definite.
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<f64> {
        self.validate(x.shape())?;
        let n = x.shape()[x.rank() - 1];
        let cholesky = match self.covariance.cholesky() {
            Ok(l) => l,
            Err(TensorError::NotPositiveDefinite) => return Ok(f64::NEG_INFINITY),
            Err(e) => return Err(e.into()),
        };
        let log_cov_det = &cholesky.diag_part()?.ln().sum_axes(&[-1])? * 2.0;
        // (x - mu)ᵀ Σ⁻¹ (x - mu) = ‖z‖² where L z = x - mu: one triangular solve, no inverse
        let x_minus_mu = x.sub(&self.mu)?;
        let column = x_minus_mu.reshape(&with_suffix(x_minus_mu.shape(), &[1]))?;
        let z = cholesky.solve_triangular(&column, true, false)?;
        let quadratic = z.mul(&z)?.sum_axes(&[-2, -1])?;
        let k_log_2pi = n as f64 * (2.0 * PI).ln();
        Ok(quadratic.add(&(&log_cov_det + k_log_2pi))?.sum() * -0.5)
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<MultivariateGaussianGradient> {
        let result_shape = self.validate(x.shape())?;
        let n = x.shape()[x.rank() - 1];
        let cholesky = self.covariance.cholesky()?;
        let x_minus_mu = x.sub(&self.mu)?;

        // Σ⁻¹ (x - mu) by two triangular solves: L y = x - mu, then Lᵀ a = y
        let column = x_minus_mu.reshape(&with_suffix(batch(x_minus_mu.shape(), 1), &[n, 1]))?;
        let y = cholesky.solve_triangular(&column, true, false)?;
        let precision_times_diff = cholesky.solve_triangular(&y, true, true)?;
        let d_mu_batched =
            precision_times_diff.reshape(&with_suffix(batch(&result_shape, 1), &[n]))?;

        // dΣ = ½ (Σ⁻¹ (x - mu)(x - mu)ᵀ Σ⁻¹ - Σ⁻¹) = ½ (a aᵀ - Σ⁻¹) with a = Σ⁻¹ (x - mu). The second term is
        // the full inverse itself, so it is formed (from the Cholesky factor, as solving against the identity
        // would cost the same); the outer product replaces the two n³ products of the explicit form.
        let cov_inv = cholesky.cholesky_inverse()?;
        let outer = precision_times_diff.matmul_transposed(&precision_times_diff, false, true)?;
        let d_cov_batched = &outer.sub(&cov_inv)? * 0.5;

        Ok(MultivariateGaussianGradient {
            x: sum_over_batch(&d_mu_batched, x.shape())?.neg(),
            mu: sum_over_batch(&d_mu_batched, self.mu.shape())?,
            covariance: sum_over_batch(&d_cov_batched, self.covariance.shape())?,
        })
    }

    /// Samples with the given x shape ([..., n]).
    pub fn sample(&self, x_shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        self.validate(x_shape)?;
        let n = x_shape[x_shape.len() - 1];
        let cholesky = self.covariance.cholesky()?;
        let mut variate_shape = batch(x_shape, 1).to_vec();
        variate_shape.extend([n, 1]);
        let variates = rng.gaussian(&variate_shape);
        let x_shape_signed: Vec<isize> = x_shape.iter().map(|&d| d as isize).collect();
        Ok(cholesky
            .matmul(&variates)?
            .reshape(&x_shape_signed)?
            .add(&self.mu)?)
    }
}

/// Dirichlet distribution over a single probability vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Dirichlet<'a> {
    /// The concentration.
    pub concentration: Param<'a>,
}

/// Derivatives of [`Dirichlet::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct DirichletGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `concentration`.
    pub concentration: DoubleTensor,
}

const DIRICHLET_EPSILON: f64 = 1e-5;

impl<'a> Dirichlet<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(concentration: impl IntoParam<'a, f64>) -> Self {
        Self {
            concentration: concentration.into_param(),
        }
    }

    /// Log density of x. x must sum to 1 (within 1e-5); -infinity if any element of x is negative.
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<f64> {
        if (x.sum() - 1.0).abs() > DIRICHLET_EPSILON {
            return Err(DistributionError::InvalidValue(
                "a Dirichlet value must sum to 1".into(),
            ));
        }
        if x.as_slice().iter().any(|&x| x < 0.0) {
            return Ok(f64::NEG_INFINITY);
        }
        let c: &DoubleTensor = &self.concentration;
        // (c - 1) ln x, taken as 0 where c is 1 (even at x = 0)
        let sum_concentration_logged = x.zip_with(c, |x, c| safe_ln_times(x, c - 1.0))?.sum();
        let sum_ln_gamma_concentration = c.ln_gamma().sum();
        Ok(sum_concentration_logged - sum_ln_gamma_concentration + ln_gamma(c.sum()))
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<DirichletGradient> {
        let c: &DoubleTensor = &self.concentration;
        Ok(DirichletGradient {
            concentration: &x.ln().sub(&c.digamma())? + digamma(c.sum()),
            x: (c - 1.0).div(x)?,
        })
    }

    /// One probability vector, from normalised Gamma(1, concentration) draws.
    pub fn sample(&self, rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let draws = self
            .concentration
            .as_slice()
            .iter()
            .map(|&k| sample_gamma(1.0, k, rng))
            .collect::<Result<Vec<_>>>()?;
        let gammas = DoubleTensor::new(self.concentration.shape(), draws)?;
        let total = gammas.sum();
        Ok(&gammas / total)
    }
}

/// The LKJ distribution of Lewandowski, Kurowicka & Joe (2009) over lower triangular Cholesky factors `L` of
/// correlation matrices `Ω = L Lᵀ`, over the value's last two dimensions (K × K). Leading dimensions are a batch,
/// with `eta` broadcasting to the batch shape.
///
/// `LKJ(η)` gives a correlation matrix a density proportional to `det(Ω)^(η − 1)`: η = 1 is uniform over
/// correlation matrices, larger η concentrates on the identity and η < 1 favours strong correlations. As a density
/// over the strictly lower triangle of `L` (the factor's free elements; the diagonal follows from the unit row
/// lengths) it is, as Stan's `lkj_corr_cholesky`,
///
/// `p(L | η) = Π_{k=2}^{K} L_kk^(K − k + 2η − 2) / c_K(η)`  (one-based k),
///
/// normalised by Lewandowski et al.'s constant (their equation 16, written over the levels of the C-vine)
///
/// `c_K(η) = Π_{k=1}^{K−1} [2^(2η − 2 + K − k) B(η + (K − k − 1) / 2, η + (K − k − 1) / 2)]^(K − k)`,
///
/// so that `eta` can be a latent variable. For K = 2 this is the density of the correlation `r = L₁₀`,
/// `(1 − r²)^(η − 1) / (2^(2η − 1) B(η, η))` on (−1, 1).
#[derive(Debug, Clone, PartialEq)]
pub struct LkjCholesky<'a> {
    /// The concentration η > 0, a scalar or one per matrix of the batch.
    pub eta: Param<'a>,
}

/// Derivatives of [`LkjCholesky::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct LkjCholeskyGradient {
    /// With respect to x (non-zero only on the diagonal below the first row).
    pub x: DoubleTensor,
    /// With respect to `eta`, shaped like it.
    pub eta: DoubleTensor,
}

/// How far a row's squared length may be from 1 for [`LkjCholesky::log_prob`] to treat the value as in the
/// support.
const LKJ_ROW_TOLERANCE: f64 = 1e-5;

impl<'a> LkjCholesky<'a> {
    /// Creates the distribution from its parameter.
    pub fn new(eta: impl IntoParam<'a, f64>) -> Self {
        Self {
            eta: eta.into_param(),
        }
    }

    /// Checks a value shape [..., K, K] (K ≥ 1) against `eta`, returning K and `eta` broadcast to the batch shape.
    fn validate(&self, shape: &[usize]) -> Result<(usize, DoubleTensor)> {
        let k = match shape {
            [.., rows, cols] if rows == cols && *rows > 0 => *rows,
            _ => {
                return Err(DistributionError::InvalidValue(format!(
                    "an LKJ Cholesky factor must have a shape ending in two equal, non-zero dimensions, \
                     not {shape:?}"
                )));
            }
        };
        let batch = batch(shape, 2);
        if broadcast_shape(batch, self.eta.shape())?.as_slice() != batch {
            return invalid_parameter(format!(
                "eta of shape {:?} does not broadcast to the batch shape {batch:?}",
                self.eta.shape()
            ));
        }
        Ok((k, self.eta.broadcast_to(batch)?))
    }

    /// The total log density of every factor in x (summed over the batch). -infinity for a factor outside the
    /// support: a non-positive diagonal, a non-zero upper triangle, or a row whose squared length is not within
    /// 1e-5 of 1. NaN where `eta` is not positive. An error if x is not a (batch of) non-empty square matrices or
    /// `eta` does not broadcast to its batch shape.
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<f64> {
        let (k, eta) = self.validate(x.shape())?;
        let mut total = 0.0;
        for (l, &eta) in x.as_slice().chunks(k * k).zip(eta.as_slice()) {
            if eta.is_nan() || eta <= 0.0 {
                return Ok(f64::NAN);
            }
            if !in_correlation_cholesky_support(l, k) {
                return Ok(f64::NEG_INFINITY);
            }
            total += (1..k)
                .map(|i| lkj_diagonal_power(k, i, eta) * l[i * k + i].ln())
                .sum::<f64>()
                - lkj_log_normaliser(k, eta);
        }
        Ok(total)
    }

    /// The derivatives of [`Self::log_prob`] with respect to x and `eta` (the support is not checked).
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<LkjCholeskyGradient> {
        let (k, eta) = self.validate(x.shape())?;
        let mut dx = vec![0.0; x.len()];
        let mut d_eta = Vec::with_capacity(eta.len());
        for ((l, dl), &eta) in x
            .as_slice()
            .chunks(k * k)
            .zip(dx.chunks_mut(k * k))
            .zip(eta.as_slice())
        {
            let mut d = -lkj_log_normaliser_d_eta(k, eta);
            for i in 1..k {
                let diagonal = l[i * k + i];
                dl[i * k + i] = lkj_diagonal_power(k, i, eta) / diagonal;
                d += 2.0 * diagonal.ln();
            }
            d_eta.push(d);
        }
        Ok(LkjCholeskyGradient {
            x: DoubleTensor::new(x.shape(), dx)?,
            eta: sum_over_batch(&DoubleTensor::new(eta.shape(), d_eta)?, self.eta.shape())?,
        })
    }

    /// Samples factors of the given shape ([..., K, K]) by the onion method (Lewandowski et al. 2009, section 3.2,
    /// written for the Cholesky factor directly as in TensorFlow Probability): the first correlation is
    /// `2 Beta(β, β) − 1` with `β = η + (K − 2) / 2`, and each later zero-based row i ≥ 2 is `√y · u` with
    /// `y ~ Beta(i / 2, β − (i − 1) / 2)` and `u` uniform on the unit sphere in i dimensions, and diagonal `√(1 − y)`.
    /// An error unless `eta` is positive and finite.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let (k, eta) = self.validate(shape)?;
        let mut samples = vec![0.0; eta.len() * k * k];
        for (l, &eta) in samples.chunks_mut(k * k).zip(eta.as_slice()) {
            if !eta.is_finite() || eta <= 0.0 {
                return invalid_parameter(format!("eta must be positive, got {eta}"));
            }
            l[0] = 1.0;
            let mut beta = eta + (k as f64 - 2.0) / 2.0;
            for i in 1..k {
                let row = &mut l[i * k..i * k + i + 1];
                if i == 1 {
                    let r = 2.0 * sample_beta(beta, beta, rng)? - 1.0;
                    row[0] = r;
                    row[1] = (1.0 - r * r).sqrt();
                    continue;
                }
                beta -= 0.5;
                let y = sample_beta(i as f64 / 2.0, beta, rng)?;
                for u in &mut row[..i] {
                    *u = rng.next_gaussian();
                }
                let scale = (y / row[..i].iter().map(|u| u * u).sum::<f64>()).sqrt();
                for u in &mut row[..i] {
                    *u *= scale;
                }
                row[i] = (1.0 - y).sqrt();
            }
        }
        Ok(DoubleTensor::new(shape, samples)?)
    }
}

/// The exponent of the zero-based diagonal element `L_ii` (i ≥ 1) in the LKJ density: `K − k + 2η − 2` with
/// one-based k = i + 1.
fn lkj_diagonal_power(k: usize, i: usize, eta: f64) -> f64 {
    (k - i - 1) as f64 + 2.0 * eta - 2.0
}

/// `ln c_K(η)` (see [`LkjCholesky`]): `Σ_{k=1}^{K−1} (K − k) [(2η − 2 + K − k) ln 2 + ln B(b_k, b_k)]` with
/// `b_k = η + (K − k − 1) / 2`.
fn lkj_log_normaliser(k: usize, eta: f64) -> f64 {
    (1..k)
        .map(|level| {
            let m = (k - level) as f64;
            let b = eta + (m - 1.0) / 2.0;
            m * ((2.0 * eta - 2.0 + m) * LN_2 + log_beta(b, b))
        })
        .sum()
}

/// The derivative of [`lkj_log_normaliser`] in η: `Σ (K − k) [2 ln 2 + 2 ψ(b_k) − 2 ψ(2 b_k)]`.
fn lkj_log_normaliser_d_eta(k: usize, eta: f64) -> f64 {
    (1..k)
        .map(|level| {
            let m = (k - level) as f64;
            let b = eta + (m - 1.0) / 2.0;
            2.0 * m * (LN_2 + digamma(b) - digamma(2.0 * b))
        })
        .sum()
}

/// Whether the row-major K × K matrix `l` is lower triangular with a positive diagonal and rows of unit length
/// (within [`LKJ_ROW_TOLERANCE`]).
fn in_correlation_cholesky_support(l: &[f64], k: usize) -> bool {
    l.chunks(k).enumerate().all(|(i, row)| {
        let length: f64 = row.iter().map(|x| x * x).sum();
        row[i] > 0.0
            && row[i + 1..].iter().all(|&x| x == 0.0)
            && (length - 1.0).abs() <= LKJ_ROW_TOLERANCE
    })
}

/// A Beta(a, b) draw, as the ratio of Gamma draws.
fn sample_beta(a: f64, b: f64, rng: &mut KeanuRng) -> Result<f64> {
    let x = sample_gamma(1.0, a, rng)?;
    let y = sample_gamma(1.0, b, rng)?;
    Ok(x / (x + y))
}

/// Multinomial distribution: counts of `n` draws over the categories in the last dimension of `p`.
/// `p` need not be normalised.
#[derive(Debug, Clone, PartialEq)]
pub struct Multinomial<'a> {
    /// The number of trials.
    pub n: Param<'a, i64>,
    /// The probability parameter.
    pub p: Param<'a>,
}

/// Derivative of [`Multinomial::log_prob`] with respect to `p`.
#[derive(Debug, Clone, PartialEq)]
pub struct MultinomialGradient {
    /// With respect to `p`.
    pub p: DoubleTensor,
}

impl<'a> Multinomial<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(n: impl IntoParam<'a, i64>, p: impl IntoParam<'a, f64>) -> Result<Self> {
        let (n, p) = (n.into_param(), p.into_param());
        if p.rank() == 0 {
            return invalid_parameter("p must be a vector or batch of vectors");
        }
        Ok(Self { n, p })
    }

    fn normalised_p(&self) -> Result<DoubleTensor> {
        let totals = self.p.sum_axes(&[-1])?;
        let mut shape: Vec<isize> = totals.shape().iter().map(|&d| d as isize).collect();
        shape.push(1);
        Ok(self.p.div(&totals.reshape(&shape)?)?)
    }

    /// Log probability of each count vector in x (the last dimension of x is the categories). -infinity for a
    /// count vector with a negative count or whose counts do not sum to n. An error if the last dimension of x is
    /// not the number of categories.
    pub fn log_prob(&self, x: &IntTensor) -> Result<DoubleTensor> {
        let k = self.p.shape()[self.p.rank() - 1];
        if x.shape().last() != Some(&k) {
            return Err(DistributionError::InvalidValue(format!(
                "the last dimension of x must be the number of categories {k}, got shape {:?}",
                x.shape()
            )));
        }
        let p = self.normalised_p()?;
        let x_double = x.to_double();
        let x_ln_p = p.safe_ln_times(&x_double)?.sum_axes(&[-1])?;
        let n = self.n.to_double();
        let ln_gamma_n = n.map(|n| ln_gamma(n + 1.0));
        let ln_gamma_x = x_double.map(|x| ln_gamma(x + 1.0)).sum_axes(&[-1])?;
        let log_prob = x_ln_p.add(&ln_gamma_n)?.sub(&ln_gamma_x)?;
        // outside the support: a negative count, or counts not summing to n (which also covers a count above n)
        let x_sum = x_double.sum_axes(&[-1])?;
        let negative_counts = x_double
            .map(|x| if x < 0.0 { 1.0 } else { 0.0 })
            .sum_axes(&[-1])?;
        map(
            (&log_prob, &x_sum, &negative_counts, &n),
            |(lp, sum, negatives, n)| {
                if negatives > 0.0 || sum != n {
                    f64::NEG_INFINITY
                } else {
                    lp
                }
            },
        )
    }

    /// With S = sum(p): d/dp_j = x_j / p_j - sum(x) / S (0 / p_j taken as 0).
    pub fn d_log_prob(&self, x: &IntTensor) -> Result<MultinomialGradient> {
        let x_double = x.to_double();
        let expand = |t: DoubleTensor| -> Result<DoubleTensor> {
            let mut shape: Vec<isize> = t.shape().iter().map(|&d| d as isize).collect();
            shape.push(1);
            Ok(t.reshape(&shape)?)
        };
        let p_sum = expand(self.p.sum_axes(&[-1])?)?;
        let x_sum = expand(x_double.sum_axes(&[-1])?)?;
        let x_div_p = x_double.zip_with(&*self.p, |x, p| if x == 0.0 { 0.0 } else { x / p })?;
        Ok(MultinomialGradient {
            p: x_div_p.sub(&x_sum.div(&p_sum)?)?,
        })
    }

    /// Samples count vectors. `shape` includes the category dimension.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        let k = self.p.shape()[self.p.rank() - 1];
        if shape.last() != Some(&k) {
            return invalid_parameter(format!(
                "the last dimension of the sample shape must be the number of categories {k}"
            ));
        }
        let batch_shape = &shape[..shape.len() - 1];
        let n = self.n.broadcast_to(batch_shape)?;
        let mut p_shape = batch_shape.to_vec();
        p_shape.push(k);
        let p = self.normalised_p()?.broadcast_to(&p_shape)?;
        let mut samples = vec![0i64; n.len() * k];
        for (i, &trials) in n.as_slice().iter().enumerate() {
            let probabilities = &p.as_slice()[i * k..(i + 1) * k];
            for _ in 0..trials.max(0) {
                samples[i * k + draw_category(probabilities, rng)] += 1;
            }
        }
        Ok(IntTensor::new(shape, samples)?)
    }
}

/// Index drawn in proportion to (normalised) probabilities, skipping zero probability categories. If rounding
/// leaves the cumulative sum below the draw, the last category with non-zero probability is returned.
fn draw_category(probabilities: &[f64], rng: &mut KeanuRng) -> usize {
    let value = rng.next_f64();
    let mut cumulative = 0.0;
    let mut last_possible = None;
    for (index, &p) in probabilities.iter().enumerate() {
        if p == 0.0 {
            continue;
        }
        last_possible = Some(index);
        cumulative += p;
        if cumulative >= value {
            return index;
        }
    }
    // all-zero probabilities (a NaN normalisation) fall back to the last category, as before
    last_possible.unwrap_or(probabilities.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::draw_category;
    use crate::rng::KeanuRng;

    #[test]
    fn rounding_shortfall_never_draws_a_zero_probability_category() {
        // probabilities summing to less than 1 (as rounding can leave them): draws above the sum must fall back
        // to the last category with non-zero probability, never a trailing zero probability one
        let mut rng = KeanuRng::seed_from_u64(7);
        for _ in 0..1000 {
            assert_eq!(draw_category(&[0.5, 0.0], &mut rng), 0);
            assert!(draw_category(&[0.3, 0.4, 0.0, 0.0], &mut rng) < 2);
        }
    }
}
