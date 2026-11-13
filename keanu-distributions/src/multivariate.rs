//! Distributions over vectors.

use crate::continuous::{sample_beta_ratio, sample_dirichlet};
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
pub(crate) fn sum_over_batch(
    batched: &DoubleTensor,
    target_shape: &[usize],
) -> Result<DoubleTensor> {
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

fn gaussian_event_shape(
    mu: &[usize],
    cov: &[usize],
    x_shape: &[usize],
    matrix_name: &str,
) -> Result<Vec<usize>> {
    let invalid = |m: &str| Err(DistributionError::InvalidValue(m.to_string()));
    if x_shape.is_empty() {
        return invalid("x must be at least a vector; use a Gaussian for scalar x");
    }
    if mu.is_empty() {
        return invalid("mu must be at least a vector");
    }
    if cov.len() < 2 {
        return invalid(&format!(
            "{matrix_name} must be a matrix or a batch of matrices"
        ));
    }
    let n = x_shape[x_shape.len() - 1];
    if mu[mu.len() - 1] != n || cov[cov.len() - 1] != n || cov[cov.len() - 2] != n {
        return invalid(&format!(
            "x, mu and {matrix_name} must have the same event size and {matrix_name} must be square"
        ));
    }
    let batch_shape = broadcast_shape(
        &broadcast_shape(batch(x_shape, 1), batch(mu, 1))?,
        batch(cov, 2),
    )?;
    Ok(batch_shape.iter().copied().chain([n]).collect())
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
        gaussian_event_shape(
            self.mu.shape(),
            self.covariance.shape(),
            x_shape,
            "covariance",
        )
    }

    /// Total log density of every vector in x (summed over batch dimensions). -infinity if the covariance is not
    /// positive definite.
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<f64> {
        match self.log_prob_summands(x) {
            Ok(terms) => Ok(terms.sum() * -0.5),
            Err(DistributionError::Tensor(TensorError::NotPositiveDefinite)) => {
                Ok(f64::NEG_INFINITY)
            }
            Err(e) => Err(e),
        }
    }

    /// One log density per vector, shaped like the broadcast batch dimensions. An error if any covariance
    /// matrix is not positive definite.
    pub fn log_prob_terms(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        Ok(self.log_prob_summands(x)? * -0.5)
    }

    fn log_prob_summands(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        self.validate(x.shape())?;
        let n = x.shape()[x.rank() - 1];
        let cholesky = self.covariance.cholesky()?;
        let log_cov_det = &cholesky.diag_part()?.ln().sum_axes(&[-1])? * 2.0;
        // (x - mu)ᵀ Σ⁻¹ (x - mu) = ‖z‖² where L z = x - mu: one triangular solve, no inverse
        let x_minus_mu = x.sub(&self.mu)?;
        let column = x_minus_mu.reshape(&with_suffix(x_minus_mu.shape(), &[1]))?;
        let z = cholesky.solve_triangular(&column, true, false)?;
        let quadratic = z.mul(&z)?.sum_axes(&[-2, -1])?;
        let k_log_2pi = n as f64 * (2.0 * PI).ln();
        Ok(quadratic.add(&(&log_cov_det + k_log_2pi))?)
    }

    /// [`Self::log_prob`] and, unless it is −∞ (a covariance that is not positive definite), [`Self::d_log_prob`],
    /// from one Cholesky factorisation and one solve against it: exactly the same numbers as the two calls.
    pub fn log_prob_and_gradient(
        &self,
        x: &DoubleTensor,
    ) -> Result<(f64, Option<MultivariateGaussianGradient>)> {
        let result_shape = self.validate(x.shape())?;
        let n = x.shape()[x.rank() - 1];
        let cholesky = match self.covariance.cholesky() {
            Ok(cholesky) => cholesky,
            Err(TensorError::NotPositiveDefinite) => return Ok((f64::NEG_INFINITY, None)),
            Err(e) => return Err(e.into()),
        };
        // the log density, as log_prob_summands
        let log_cov_det = &cholesky.diag_part()?.ln().sum_axes(&[-1])? * 2.0;
        let x_minus_mu = x.sub(&self.mu)?;
        let column = x_minus_mu.reshape(&with_suffix(x_minus_mu.shape(), &[1]))?;
        let z = cholesky.solve_triangular(&column, true, false)?;
        let quadratic = z.mul(&z)?.sum_axes(&[-2, -1])?;
        let k_log_2pi = n as f64 * (2.0 * PI).ln();
        let log_prob = quadratic.add(&(&log_cov_det + k_log_2pi))?.sum() * -0.5;
        // the gradient, as d_log_prob, whose first solve is z's
        let precision_times_diff = cholesky.solve_triangular(&z, true, true)?;
        let d_mu_batched =
            precision_times_diff.reshape(&with_suffix(batch(&result_shape, 1), &[n]))?;
        let cov_inv = cholesky.cholesky_inverse()?;
        let outer = precision_times_diff.matmul_transposed(&precision_times_diff, false, true)?;
        let d_cov_batched = &outer.sub(&cov_inv)? * 0.5;
        Ok((
            log_prob,
            Some(MultivariateGaussianGradient {
                x: sum_over_batch(&d_mu_batched, x.shape())?.neg(),
                mu: sum_over_batch(&d_mu_batched, self.mu.shape())?,
                covariance: sum_over_batch(&d_cov_batched, self.covariance.shape())?,
            }),
        ))
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

/// Multivariate Gaussian parameterized by a positive definite precision matrix P.
/// Leading dimensions broadcast. The lower triangle of P specifies its symmetric entries.
#[derive(Debug, Clone, PartialEq)]
pub struct MultivariateGaussianPrecision<'a> {
    /// Mean vector or batch of mean vectors.
    pub mu: Param<'a>,
    /// Precision matrix or batch of precision matrices.
    pub precision: Param<'a>,
}

/// Derivatives of the precision-form Gaussian log density, reduced to each input's shape.
#[derive(Debug, Clone, PartialEq)]
pub struct MultivariateGaussianPrecisionGradient {
    /// Derivative with respect to the value.
    pub x: DoubleTensor,
    /// Derivative with respect to the mean.
    pub mu: DoubleTensor,
    /// Derivative with respect to the lower triangle of the precision.
    pub precision: DoubleTensor,
}

impl<'a> MultivariateGaussianPrecision<'a> {
    /// Creates the distribution from a mean and precision.
    pub fn new(mu: impl IntoParam<'a, f64>, precision: impl IntoParam<'a, f64>) -> Self {
        Self {
            mu: mu.into_param(),
            precision: precision.into_param(),
        }
    }

    fn factor(&self, shape: &[usize]) -> Result<(Vec<usize>, DoubleTensor)> {
        let shape =
            gaussian_event_shape(self.mu.shape(), self.precision.shape(), shape, "precision")?;
        if shape.last() == Some(&0) {
            return invalid_parameter("the precision Gaussian needs a nonempty event");
        }
        let l = self.precision.cholesky()?;
        if l.diag_part()?
            .as_slice()
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(TensorError::NotPositiveDefinite.into());
        }
        Ok((shape, l))
    }

    fn residual_column(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        let r = x.sub(&self.mu)?;
        Ok(r.reshape(&with_suffix(r.shape(), &[1]))?)
    }

    fn terms(&self, l: &DoubleTensor, z: &DoubleTensor, n: usize) -> Result<DoubleTensor> {
        let half_log_det = l.diag_part()?.ln().sum_axes(&[-1])?;
        let quadratic = z.mul(z)?.sum_axes(&[-2, -1])?;
        Ok(half_log_det.sub(&(&quadratic * 0.5 + 0.5 * n as f64 * (2.0 * PI).ln()))?)
    }

    /// Log density per broadcast vector. An error if the precision cannot be factored.
    pub fn log_prob_terms(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        let (_, l) = self.factor(x.shape())?;
        let z = l.matmul_transposed(&self.residual_column(x)?, true, false)?;
        self.terms(&l, &z, x.shape()[x.rank() - 1])
    }

    /// Joint log density. Negative infinity for a precision that is not positive definite.
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<f64> {
        match self.log_prob_terms(x) {
            Ok(terms) => Ok(terms.sum()),
            Err(DistributionError::Tensor(TensorError::NotPositiveDefinite)) => {
                Ok(f64::NEG_INFINITY)
            }
            Err(e) => Err(e),
        }
    }

    fn gradient(
        &self,
        x: &DoubleTensor,
        shape: &[usize],
        l: &DoubleTensor,
        r: &DoubleTensor,
        z: &DoubleTensor,
    ) -> Result<MultivariateGaussianPrecisionGradient> {
        let a = l.matmul(z)?.reshape(&with_suffix(shape, &[]))?;
        let outer = r.matmul_transposed(r, false, true)?;
        let symmetric = l.cholesky_inverse()?.sub(&outer)? * 0.5;
        // A strict-lower entry represents both symmetric entries.
        let lower = symmetric.tril(0)?.add(&symmetric.tril(-1)?)?;
        Ok(MultivariateGaussianPrecisionGradient {
            x: sum_over_batch(&a, x.shape())?.neg(),
            mu: sum_over_batch(&a, self.mu.shape())?,
            precision: sum_over_batch(&lower, self.precision.shape())?,
        })
    }

    /// Joint density and its derivatives using a single factorization.
    /// Impossible points return negative infinity and no gradient.
    pub fn log_prob_and_gradient(
        &self,
        x: &DoubleTensor,
    ) -> Result<(f64, Option<MultivariateGaussianPrecisionGradient>)> {
        let (shape, l) = match self.factor(x.shape()) {
            Ok(parts) => parts,
            Err(DistributionError::Tensor(TensorError::NotPositiveDefinite)) => {
                return Ok((f64::NEG_INFINITY, None));
            }
            Err(e) => return Err(e),
        };
        let r = self.residual_column(x)?;
        let z = l.matmul_transposed(&r, true, false)?;
        let lp = self.terms(&l, &z, shape[shape.len() - 1])?.sum();
        let gradient = if lp.is_finite() {
            Some(self.gradient(x, &shape, &l, &r, &z)?)
        } else {
            None
        };
        Ok((lp, gradient))
    }

    /// Derivatives at a point with positive definite precision.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<MultivariateGaussianPrecisionGradient> {
        let (shape, l) = self.factor(x.shape())?;
        let r = self.residual_column(x)?;
        let z = l.matmul_transposed(&r, true, false)?;
        self.gradient(x, &shape, &l, &r, &z)
    }

    /// Draws independent vectors with covariance P⁻¹, using a triangular solve.
    /// The requested shape must contain the entire parameter batch.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let (broadcast, l) = self.factor(shape)?;
        if broadcast != shape {
            return invalid_parameter("sample shape must include the precision and mean batch");
        }
        let columns: Vec<usize> = shape.iter().copied().chain([1]).collect();
        let z = rng.gaussian(&columns);
        let x = l
            .solve_triangular(&z, true, true)?
            .reshape(&with_suffix(shape, &[]))?;
        Ok(x.add(&self.mu)?)
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

    /// One probability vector, from normalised Gamma(1, concentration) draws (normalised from their logs, so the
    /// vector sums to 1 even for concentrations so small that every Gamma draw would underflow).
    pub fn sample(&self, rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let draws = sample_dirichlet(self.concentration.as_slice(), rng)?;
        Ok(DoubleTensor::new(self.concentration.shape(), draws)?)
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
    sample_beta_ratio(a, b, rng)
}

/// The zero-sum normal distribution over the last dimension, as PyMC's `ZeroSumNormal`: a Gaussian N(0, σ² I)
/// restricted to vectors whose elements sum to zero. Leading dimensions are a batch, and `sigma` is a scalar or
/// broadcasts to the batch shape (one σ per vector).
///
/// For vectors of N elements the density, with respect to the volume of the (N − 1)-dimensional zero-sum subspace,
/// is that of N(0, σ² I) in an orthonormal basis of the subspace:
///
/// `ln p(x) = −Σᵢ xᵢ² / (2σ²) − (N − 1) (ln σ + ½ ln 2π)`.
///
/// Each element's marginal is N(0, σ² (N − 1) / N), and a draw is a draw of N(0, σ² I) minus its mean. It is the
/// usual prior for group effects next to an intercept: the effects are identified, and their scale is σ.
#[derive(Debug, Clone, PartialEq)]
pub struct ZeroSumNormal<'a> {
    /// The scale σ > 0, a scalar or one per vector of the batch.
    pub sigma: Param<'a>,
}

/// Derivatives of [`ZeroSumNormal::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct ZeroSumNormalGradient {
    /// With respect to x, `−x / σ²` (in the ambient space; its projection onto the zero-sum subspace is the
    /// derivative within it).
    pub x: DoubleTensor,
    /// With respect to `sigma`, shaped like it.
    pub sigma: DoubleTensor,
}

/// How far a vector's sum may be from zero, relative to the larger of 1 and the sum of its elements' magnitudes, for
/// [`ZeroSumNormal::log_prob`] to treat it as in the support.
pub const ZERO_SUM_TOLERANCE: f64 = 1e-8;

/// Whether `x` sums to zero within [`ZERO_SUM_TOLERANCE`].
pub fn sums_to_zero(x: &[f64]) -> bool {
    let (sum, magnitude) = x
        .iter()
        .fold((0.0, 0.0), |(s, m): (f64, f64), v| (s + v, m + v.abs()));
    sum.abs() <= ZERO_SUM_TOLERANCE * magnitude.max(1.0)
}

impl<'a> ZeroSumNormal<'a> {
    /// Creates the distribution from its parameter.
    pub fn new(sigma: impl IntoParam<'a, f64>) -> Self {
        Self {
            sigma: sigma.into_param(),
        }
    }

    /// Checks a value shape [..., N] (N ≥ 1) against `sigma`, returning N and `sigma` broadcast to the batch shape.
    fn validate(&self, shape: &[usize]) -> Result<(usize, DoubleTensor)> {
        let n = match shape.last() {
            Some(&n) if n > 0 => n,
            _ => {
                return Err(DistributionError::InvalidValue(format!(
                    "a zero-sum normal value must have at least one dimension, of at least one element, not \
                     {shape:?}"
                )));
            }
        };
        let batch = batch(shape, 1);
        if broadcast_shape(batch, self.sigma.shape())?.as_slice() != batch {
            return invalid_parameter(format!(
                "sigma of shape {:?} does not broadcast to the batch shape {batch:?}",
                self.sigma.shape()
            ));
        }
        Ok((n, self.sigma.broadcast_to(batch)?))
    }

    /// The total log density of every vector in x (summed over the batch). -infinity for a vector that is not
    /// finite or doesn't sum to zero (within [`ZERO_SUM_TOLERANCE`]), NaN where `sigma` is not positive. An error
    /// if x is a scalar or has an empty last dimension, or `sigma` does not broadcast to its batch shape.
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<f64> {
        let (n, sigma) = self.validate(x.shape())?;
        let normaliser = 0.5 * (2.0 * PI).ln();
        let mut total = 0.0;
        for (row, &s) in x.as_slice().chunks(n).zip(sigma.as_slice()) {
            if s.is_nan() || s <= 0.0 {
                return Ok(f64::NAN);
            }
            if !row.iter().all(|v| v.is_finite()) || !sums_to_zero(row) {
                return Ok(f64::NEG_INFINITY);
            }
            let squares: f64 = row.iter().map(|v| v * v).sum();
            total += -0.5 * squares / (s * s) - (n - 1) as f64 * (s.ln() + normaliser);
        }
        Ok(total)
    }

    /// The derivatives of [`Self::log_prob`] with respect to x and `sigma` (the support is not checked).
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<ZeroSumNormalGradient> {
        let (n, sigma) = self.validate(x.shape())?;
        let mut dx = vec![0.0; x.len()];
        let mut d_sigma = Vec::with_capacity(sigma.len());
        for ((row, d), &s) in x
            .as_slice()
            .chunks(n)
            .zip(dx.chunks_mut(n))
            .zip(sigma.as_slice())
        {
            let mut squares = 0.0;
            for (d, v) in d.iter_mut().zip(row) {
                *d = -v / (s * s);
                squares += v * v;
            }
            d_sigma.push(squares / (s * s * s) - (n - 1) as f64 / s);
        }
        Ok(ZeroSumNormalGradient {
            x: DoubleTensor::new(x.shape(), dx)?,
            sigma: sum_over_batch(
                &DoubleTensor::new(sigma.shape(), d_sigma)?,
                self.sigma.shape(),
            )?,
        })
    }

    /// Draws vectors of the given shape ([..., N]): each is a draw of N(0, σ² I) minus its mean. An error unless
    /// `sigma` is positive and finite.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let (n, sigma) = self.validate(shape)?;
        let mut samples = vec![0.0; sigma.len() * n];
        for (row, &s) in samples.chunks_mut(n).zip(sigma.as_slice()) {
            if !s.is_finite() || s <= 0.0 {
                return invalid_parameter(format!("sigma must be positive, got {s}"));
            }
            for v in row.iter_mut() {
                *v = s * rng.next_gaussian();
            }
            let mean = row.iter().sum::<f64>() / n as f64;
            for v in row.iter_mut() {
                *v -= mean;
            }
        }
        Ok(DoubleTensor::new(shape, samples)?)
    }
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

/// Multinomial counts parameterized by finite logits over the last axis.
/// The count totals and leading dimensions of the logits and values broadcast.
#[derive(Debug, Clone, PartialEq)]
pub struct MultinomialLogit<'a> {
    n: Param<'a, i64>,
    logits: Param<'a>,
}

/// Derivatives of the joint log mass with respect to the logits.
#[derive(Debug, Clone, PartialEq)]
pub struct MultinomialLogitGradient {
    /// Derivatives summed to the logits' original shape.
    pub logits: DoubleTensor,
}

impl<'a> MultinomialLogit<'a> {
    /// Creates a distribution with a nonnegative total for each event and a last category axis of logits.
    pub fn new(n: impl IntoParam<'a, i64>, logits: impl IntoParam<'a, f64>) -> Result<Self> {
        let (n, logits) = (n.into_param(), logits.into_param());
        if logits.shape().last().is_none_or(|&k| k == 0) {
            return invalid_parameter("logits need a nonempty category axis");
        }
        Ok(Self { n, logits })
    }

    fn batch_shape(&self, shape: &[usize]) -> Result<Vec<usize>> {
        if shape.last() != self.logits.shape().last() {
            return Err(DistributionError::InvalidValue(format!(
                "count vectors of shape {shape:?} must end in the logits' category axis"
            )));
        }
        let batch_shape = broadcast_shape(batch(shape, 1), batch(self.logits.shape(), 1))?;
        Ok(broadcast_shape(&batch_shape, self.n.shape())?.to_vec())
    }

    /// One log mass per count vector. Impossible counts give negative infinity.
    /// Nonfinite logits or a negative total give NaN for their events.
    pub fn log_prob(&self, x: &IntTensor) -> Result<DoubleTensor> {
        let batch = self.batch_shape(x.shape())?;
        let k = x.shape()[x.rank() - 1];
        let shape = batch.iter().copied().chain([k]).collect::<Vec<_>>();
        let counts = x.broadcast_to(&shape)?;
        let totals = self.n.broadcast_to(&batch)?;
        let log_p = self.logits.log_softmax(-1)?.broadcast_to(&shape)?;
        let logits = self.logits.broadcast_to(&shape)?;
        let values = counts
            .as_slice()
            .chunks_exact(k)
            .zip(log_p.as_slice().chunks_exact(k))
            .zip(logits.as_slice().chunks_exact(k))
            .zip(totals.as_slice())
            .map(|(((counts, log_p), logits), &n)| {
                if n < 0 || logits.iter().any(|l| !l.is_finite()) {
                    return f64::NAN;
                }
                let mut remaining = n;
                let mut lp = 0.0;
                for (&count, &log_p) in counts.iter().zip(log_p) {
                    if count < 0 || count > remaining {
                        return f64::NEG_INFINITY;
                    }
                    if count != 0 {
                        let smaller = count.min(remaining - count);
                        if smaller != 0 {
                            // ln C(remaining, count) via beta, avoiding subtraction of large factorials.
                            lp -= (remaining as f64 + 1.0).ln()
                                + log_beta(
                                    smaller as f64 + 1.0,
                                    (remaining - smaller) as f64 + 1.0,
                                );
                        }
                        lp += count as f64 * log_p;
                    }
                    remaining -= count;
                }
                if remaining == 0 {
                    lp
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect();
        Ok(DoubleTensor::new(&batch, values)?)
    }

    /// Joint log-mass derivatives, x − n softmax(logits), at points in the support.
    pub fn d_log_prob(&self, x: &IntTensor) -> Result<MultinomialLogitGradient> {
        let batch = self.batch_shape(x.shape())?;
        let totals = self
            .n
            .broadcast_to(&batch)?
            .to_double()
            .reshape(&with_suffix(&batch, &[1]))?;
        let probabilities = self.logits.softmax(-1)?;
        let gradient = x.to_double().sub(&probabilities.mul(&totals)?)?;
        Ok(MultinomialLogitGradient {
            logits: sum_over_batch(&gradient, self.logits.shape())?,
        })
    }

    /// Draws counts whose last axis matches the logits and whose totals equal n.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        let batch = self.batch_shape(shape)?;
        if batch != shape[..shape.len() - 1] {
            return invalid_parameter("sample shape must include the multinomial parameter batch");
        }
        if self.logits.as_slice().iter().any(|l| !l.is_finite())
            || self.n.as_slice().iter().any(|&n| n < 0)
        {
            return invalid_parameter("multinomial logits must be finite and totals nonnegative");
        }
        Multinomial::new(&*self.n, self.logits.softmax(-1)?)?.sample(shape, rng)
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
