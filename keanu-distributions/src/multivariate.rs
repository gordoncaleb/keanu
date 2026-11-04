//! Distributions over vectors.

use crate::continuous::sample_gamma;
use crate::error::{DistributionError, Result, invalid_parameter};
use crate::rng::KeanuRng;
use crate::util::{digamma, ln_gamma, map4, safe_ln_times};
use keanu_tensor::{DoubleTensor, IntTensor, TensorError, broadcast_shape};
use std::f64::consts::PI;

/// Multivariate normal distribution over the last dimension. Leading dimensions of x, mu and covariance are
/// batch dimensions and broadcast.
#[derive(Debug, Clone, PartialEq)]
pub struct MultivariateGaussian {
    /// Mean, shape [..., n].
    pub mu: DoubleTensor,
    /// Covariance, shape [..., n, n].
    pub covariance: DoubleTensor,
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

impl MultivariateGaussian {
    /// Creates the distribution from its parameters.
    pub fn new(mu: DoubleTensor, covariance: DoubleTensor) -> Self {
        Self { mu, covariance }
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
        let cov_inv = cholesky.cholesky_inverse()?;
        let x_minus_mu = x.sub(&self.mu)?;
        let shape = x_minus_mu.shape().to_vec();
        let row = x_minus_mu.reshape(&with_suffix(&shape[..shape.len() - 1], &[1, n]))?;
        let column = x_minus_mu.reshape(&with_suffix(&shape, &[1]))?;
        let quadratic = row.matmul(&cov_inv.matmul(&column)?)?;
        let quadratic = quadratic.reshape(&with_suffix(batch(quadratic.shape(), 2), &[]))?;
        let k_log_2pi = n as f64 * (2.0 * PI).ln();
        Ok(quadratic.add(&(&log_cov_det + k_log_2pi))?.sum() * -0.5)
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<MultivariateGaussianGradient> {
        let result_shape = self.validate(x.shape())?;
        let n = x.shape()[x.rank() - 1];
        let cov_inv = self.covariance.cholesky()?.cholesky_inverse()?;
        let x_minus_mu = x.sub(&self.mu)?;

        let column = x_minus_mu.reshape(&with_suffix(batch(x_minus_mu.shape(), 1), &[n, 1]))?;
        let d_mu_batched = cov_inv
            .matmul(&column)?
            .reshape(&with_suffix(batch(&result_shape, 1), &[n]))?;

        let cov_inv_t = cov_inv.transpose()?;
        let outer = column.matmul(&column.transpose()?)?;
        let d_cov_batched = &cov_inv_t
            .matmul(&outer.matmul(&cov_inv_t)?)?
            .sub(&cov_inv_t)?
            * 0.5;

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
pub struct Dirichlet {
    /// The concentration.
    pub concentration: DoubleTensor,
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

impl Dirichlet {
    /// Creates the distribution from its parameters.
    pub fn new(concentration: DoubleTensor) -> Self {
        Self { concentration }
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
        let c = &self.concentration;
        // (c - 1) ln x, taken as 0 where c is 1 (even at x = 0)
        let sum_concentration_logged = x.zip_with(c, |x, c| safe_ln_times(x, c - 1.0))?.sum();
        let sum_ln_gamma_concentration = c.ln_gamma().sum();
        Ok(sum_concentration_logged - sum_ln_gamma_concentration + ln_gamma(c.sum()))
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<DirichletGradient> {
        let c = &self.concentration;
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

/// Multinomial distribution: counts of `n` draws over the categories in the last dimension of `p`.
/// `p` need not be normalised.
#[derive(Debug, Clone, PartialEq)]
pub struct Multinomial {
    /// The number of trials.
    pub n: IntTensor,
    /// The probability parameter.
    pub p: DoubleTensor,
}

/// Derivative of [`Multinomial::log_prob`] with respect to `p`.
#[derive(Debug, Clone, PartialEq)]
pub struct MultinomialGradient {
    /// With respect to `p`.
    pub p: DoubleTensor,
}

impl Multinomial {
    /// Creates the distribution from its parameters.
    pub fn new(n: IntTensor, p: DoubleTensor) -> Result<Self> {
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
        map4(
            &log_prob,
            &x_sum,
            &negative_counts,
            &n,
            |lp, sum, negatives, n| {
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
        let x_div_p = x_double.zip_with(&self.p, |x, p| if x == 0.0 { 0.0 } else { x / p })?;
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
