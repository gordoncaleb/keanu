//! Multivariate Student-t, Wishart, inverse Wishart and matrix normal distributions, each with a Cholesky-factor
//! form where Stan has one.
//!
//! Each is a distribution over the value's last one (vectors) or two (matrices) axes, with any leading axes a batch
//! of independent events; each parameter's leading axes broadcast to the batch. Their gradients are written by hand
//! and their Hessian-vector products are the same code on dual numbers (forward mode over the reverse pass), so
//! both are exact. Gradients with respect to a matrix are with respect to each of its entries, as for
//! [`MultivariateGaussian`](crate::MultivariateGaussian): those of a symmetric matrix (a covariance, a scale) are
//! symmetric, and those of a Cholesky factor are zero above the diagonal.
//!
//! Symmetric matrices (a covariance or scale parameter, a Wishart value) must be symmetric to a relative tolerance
//! of 1e-8, and only their lower triangles are factorised; Cholesky factors are lower triangular with a positive
//! diagonal, and the upper triangle of a Cholesky-factor *parameter* is ignored, as in Stan.

// the negated comparisons (`!(kappa >= 0.0)`) are deliberate: they also catch NaN
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use crate::continuous::sample_gamma;
use crate::error::{DistributionError, Result, invalid_parameter};
use crate::event::{EventFamily, EventGradient, event_distribution};
use crate::param::{IntoParam, Param};
use crate::real::{Real, dense};
use crate::rng::KeanuRng;
use keanu_tensor::DoubleTensor;
use std::f64::consts::{LN_2, PI};

/// How far a symmetric matrix's mirrored entries may differ, relative to the larger of 1 and their magnitude.
pub const SYMMETRY_TOLERANCE: f64 = 1e-8;

/// ln Γ_K(a) = K(K − 1)/4 ln π + Σ_{j<K} ln Γ(a − j/2), the multivariate gamma function.
fn log_multivariate_gamma<S: Real>(k: usize, a: S) -> S {
    (0..k).fold(
        S::constant(k as f64 * (k as f64 - 1.0) / 4.0 * PI.ln()),
        |s, j| s + (a - j as f64 / 2.0).ln_gamma(),
    )
}

/// d ln Γ_K(a) / da = Σ_{j<K} ψ(a − j/2).
fn multivariate_digamma<S: Real>(k: usize, a: S) -> S {
    (0..k).fold(S::zero(), |s, j| s + (a - j as f64 / 2.0).digamma())
}

/// The size K of a square event [K, K], or an error.
fn square(what: &str, event: &[usize]) -> Result<usize> {
    match event {
        [rows, cols] if rows == cols && *rows > 0 => Ok(*rows),
        _ => Err(DistributionError::InvalidValue(format!(
            "a {what} value must end in two equal, non-zero axes, not {event:?}"
        ))),
    }
}

/// Whether a K × K matrix is finite, symmetric (to [`SYMMETRY_TOLERANCE`]) and positive definite.
fn symmetric_positive_definite(a: &[f64], k: usize) -> bool {
    a.iter().all(|v| v.is_finite())
        && dense::is_symmetric(a, k, SYMMETRY_TOLERANCE)
        && dense::cholesky(a, k).iter().all(|v| !v.is_nan())
}

/// Whether a Cholesky-factor parameter has a positive, finite diagonal and a finite lower triangle.
fn valid_factor_parameter(l: &[f64], k: usize) -> bool {
    (0..k).all(|i| l[i * k + i] > 0.0 && l[i * k..=i * k + i].iter().all(|v| v.is_finite()))
}

/// An error for a sampler's invalid parameters.
fn invalid_sampling(what: &str) -> Result<()> {
    invalid_parameter(format!("cannot sample a {what} with invalid parameters"))
}

/// Draws a K × K Bartlett factor A: lower triangular, A_ii = √χ²(ν − i) (zero-based i) and A_ij ~ N(0, 1) below the
/// diagonal, so that L A Aᵀ Lᵀ ~ Wishart(ν, L Lᵀ).
fn bartlett(nu: f64, k: usize, rng: &mut KeanuRng) -> Result<Vec<f64>> {
    let mut a = vec![0.0; k * k];
    for i in 0..k {
        a[i * k + i] = sample_gamma(2.0, (nu - i as f64) / 2.0, rng)?.sqrt();
        for j in 0..i {
            a[i * k + j] = rng.next_gaussian();
        }
    }
    Ok(a)
}

// ---------------------------------------------------------------------------------------------------------------
// Multivariate Student-t

/// The multivariate Student-t kernel; `cholesky` if the third parameter is the scale's Cholesky factor.
struct StudentTKernel {
    cholesky: bool,
}

impl EventFamily for StudentTKernel {
    const EVENT_RANK: usize = 1;

    fn validate_event(&self, event: &[usize]) -> Result<()> {
        if event[0] == 0 {
            return Err(DistributionError::InvalidValue(
                "a multivariate Student-t value needs at least one element".into(),
            ));
        }
        Ok(())
    }

    fn parameter_events(&self, event: &[usize]) -> Vec<Vec<usize>> {
        let k = event[0];
        vec![vec![], vec![k], vec![k, k]]
    }

    fn check(&self, event: &[usize], x: &[f64], p: &[&[f64]]) -> Option<f64> {
        let k = event[0];
        let (nu, mu, scale) = (p[0][0], p[1], p[2]);
        let scale_ok = if self.cholesky {
            valid_factor_parameter(scale, k)
        } else {
            symmetric_positive_definite(scale, k)
        };
        if !(nu > 0.0 && nu.is_finite()) || !scale_ok || mu.iter().any(|m| !m.is_finite()) {
            return Some(f64::NAN);
        }
        x.iter()
            .any(|v| !v.is_finite())
            .then_some(f64::NEG_INFINITY)
    }

    fn gradient<S: Real>(&self, event: &[usize], x: &[S], p: &[&[S]]) -> EventGradient<S> {
        let k = event[0];
        let kf = k as f64;
        let (nu, mu) = (p[0][0], p[1]);
        let l = if self.cholesky {
            dense::lower(p[2].to_vec(), k)
        } else {
            dense::cholesky(p[2], k)
        };
        let r: Vec<S> = x.iter().zip(mu).map(|(&x, &m)| x - m).collect();
        let z = dense::solve_lower(&l, &r, k, 1);
        let q = dense::dot(&z, &z);
        let a = dense::solve_lower_transposed(&l, &z, k, 1);
        let half_log_det = dense::log_diagonal(&l, k);
        let log1p_q = (q / nu + 1.0).ln();
        let log_prob = ((nu + kf) * 0.5).ln_gamma()
            - (nu * 0.5).ln_gamma()
            - (nu * PI).ln() * (kf / 2.0)
            - half_log_det
            - (nu + kf) * 0.5 * log1p_q;
        let w = (nu + kf) / (nu + q);
        let dx: Vec<S> = a.iter().map(|&a| -(w * a)).collect();
        let dmu: Vec<S> = a.iter().map(|&a| w * a).collect();
        let d_scale = if self.cholesky {
            // ∂/∂L = −diag(1/L_ii) + w a zᵀ, below the diagonal
            let mut d = vec![S::zero(); k * k];
            for i in 0..k {
                for j in 0..=i {
                    d[i * k + j] = w * a[i] * z[j];
                }
                d[i * k + i] = d[i * k + i] - S::constant(1.0) / l[i * k + i];
            }
            d
        } else {
            // ∂/∂Σ = −½ Σ⁻¹ + ½ w a aᵀ
            let inverse = dense::cholesky_inverse(&l, k);
            let mut d = vec![S::zero(); k * k];
            for i in 0..k {
                for j in 0..k {
                    d[i * k + j] = (w * a[i] * a[j] - inverse[i * k + j]) * 0.5;
                }
            }
            d
        };
        let d_nu = ((nu + kf) * 0.5).digamma() * 0.5
            - (nu * 0.5).digamma() * 0.5
            - S::constant(kf / 2.0) / nu
            - log1p_q * 0.5
            + (nu + kf) * q / (nu * (nu + q) * 2.0);
        EventGradient {
            log_prob,
            x: dx,
            parameters: vec![vec![d_nu], dmu, d_scale],
        }
    }
}

/// Draws one multivariate Student-t vector: μ + L z √(ν/u), z ~ N(0, I) and u ~ χ²(ν).
fn sample_student_t(
    cholesky: bool,
    event: &[usize],
    p: &[&[f64]],
    out: &mut [f64],
    rng: &mut KeanuRng,
) -> Result<()> {
    let kernel = StudentTKernel { cholesky };
    if kernel.check(event, &vec![0.0; event[0]], p).is_some() {
        return invalid_sampling("multivariate Student-t");
    }
    let k = event[0];
    let (nu, mu) = (p[0][0], p[1]);
    let l = if cholesky {
        dense::lower(p[2].to_vec(), k)
    } else {
        dense::cholesky(p[2], k)
    };
    let z: Vec<f64> = (0..k).map(|_| rng.next_gaussian()).collect();
    let scale = (nu / sample_gamma(2.0, nu / 2.0, rng)?).sqrt();
    for i in 0..k {
        let lz: f64 = (0..=i).map(|j| l[i * k + j] * z[j]).sum();
        out[i] = mu[i] + lz * scale;
    }
    Ok(())
}

event_distribution! {
    /// The multivariate Student-t distribution with ν degrees of freedom, location μ and scale matrix Σ, as Stan's
    /// `multi_student_t(ν, μ, Σ)`, over the value's last axis (K elements):
    ///
    /// ```text
    /// p(x) = Γ((ν + K)/2) / (Γ(ν/2) (νπ)^(K/2) |Σ|^(1/2)) (1 + (x − μ)ᵀ Σ⁻¹ (x − μ) / ν)^(−(ν + K)/2)
    /// ```
    ///
    /// for real ν > 0 and a symmetric positive definite Σ (the covariance is νΣ/(ν − 2) for ν > 2). `nu` is a
    /// scalar or one per vector of the batch, `mu` is [..., K] and `scale` [..., K, K]. Draws are
    /// μ + L z √(ν/u), with L the Cholesky factor of Σ, z ~ N(0, I) and u ~ χ²(ν).
    MultivariateStudentT, MultivariateStudentTGradient,
    kernel = StudentTKernel { cholesky: false }, what = "multivariate Student-t",
    (
        /// The degrees of freedom ν > 0.
        nu,
        /// The location μ, [..., K].
        mu,
        /// The scale matrix Σ, [..., K, K].
        scale,
    )
}

impl MultivariateStudentT<'_> {
    /// Draws vectors of the given shape ([..., K]). An error unless ν > 0 and the scale is symmetric positive
    /// definite.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        self.events().sample(shape, rng, |e, p, out, rng| {
            sample_student_t(false, e, p, out, rng)
        })
    }
}

event_distribution! {
    /// The multivariate Student-t distribution given the Cholesky factor L of its scale matrix Σ = L Lᵀ, as Stan's
    /// `multi_student_t_cholesky(ν, μ, L)`: the density of [`MultivariateStudentT`] with |Σ|^(1/2) = Π Lᵢᵢ and the
    /// quadratic form ‖L⁻¹(x − μ)‖², so no factorisation is needed. L is lower triangular with a positive diagonal
    /// (its upper triangle is ignored), and the gradient with respect to it is zero above the diagonal.
    MultivariateStudentTCholesky, MultivariateStudentTCholeskyGradient,
    kernel = StudentTKernel { cholesky: true }, what = "multivariate Student-t",
    (
        /// The degrees of freedom ν > 0.
        nu,
        /// The location μ, [..., K].
        mu,
        /// The Cholesky factor L of the scale matrix, [..., K, K].
        scale_cholesky,
    )
}

impl MultivariateStudentTCholesky<'_> {
    /// Draws vectors of the given shape ([..., K]). An error unless ν > 0 and the factor's diagonal is positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        self.events().sample(shape, rng, |e, p, out, rng| {
            sample_student_t(true, e, p, out, rng)
        })
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Wishart and inverse Wishart

/// The Wishart (or, if `inverse`, inverse Wishart) kernel over symmetric matrices (or, if `cholesky`, over their
/// Cholesky factors, with the scale given by its Cholesky factor).
struct WishartKernel {
    inverse: bool,
    cholesky: bool,
}

impl WishartKernel {
    fn what(&self) -> &'static str {
        match (self.inverse, self.cholesky) {
            (false, false) => "Wishart",
            (false, true) => "Wishart Cholesky",
            (true, false) => "inverse Wishart",
            (true, true) => "inverse Wishart Cholesky",
        }
    }
}

impl EventFamily for WishartKernel {
    const EVENT_RANK: usize = 2;

    fn validate_event(&self, event: &[usize]) -> Result<()> {
        square(self.what(), event).map(|_| ())
    }

    fn parameter_events(&self, event: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![], event.to_vec()]
    }

    fn check(&self, event: &[usize], x: &[f64], p: &[&[f64]]) -> Option<f64> {
        let k = event[0];
        let (nu, scale) = (p[0][0], p[1]);
        let scale_ok = if self.cholesky {
            valid_factor_parameter(scale, k)
        } else {
            symmetric_positive_definite(scale, k)
        };
        if !(nu > k as f64 - 1.0 && nu.is_finite()) || !scale_ok {
            return Some(f64::NAN);
        }
        let in_support = if self.cholesky {
            dense::is_cholesky_factor(x, k)
        } else {
            symmetric_positive_definite(x, k)
        };
        (!in_support).then_some(f64::NEG_INFINITY)
    }

    fn gradient<S: Real>(&self, event: &[usize], x: &[S], p: &[&[S]]) -> EventGradient<S> {
        let k = event[0];
        let kf = k as f64;
        let nu = p[0][0];
        // the constant and ν terms shared by every form: −νK/2 ln 2 − ln Γ_K(ν/2), and their ν derivative
        let common = -(nu * (kf / 2.0 * LN_2)) - log_multivariate_gamma(k, nu * 0.5);
        let d_common = S::constant(-kf / 2.0 * LN_2) - multivariate_digamma(k, nu * 0.5) * 0.5;
        match (self.inverse, self.cholesky) {
            (false, false) => {
                // (ν − K − 1)/2 ln|W| − ½ tr(S⁻¹W) − ν/2 ln|S|
                let ls = dense::cholesky(p[1], k);
                let lw = dense::cholesky(x, k);
                let s_inv = dense::cholesky_inverse(&ls, k);
                let w_inv = dense::cholesky_inverse(&lw, k);
                let (log_det_w, log_det_s) = (
                    dense::log_diagonal(&lw, k) * 2.0,
                    dense::log_diagonal(&ls, k) * 2.0,
                );
                let power = (nu - (kf + 1.0)) * 0.5;
                let log_prob =
                    power * log_det_w - dense::dot(&s_inv, x) * 0.5 - nu * 0.5 * log_det_s + common;
                let dx = (0..k * k)
                    .map(|i| power * w_inv[i] - s_inv[i] * 0.5)
                    .collect();
                let sws = dense::matmul(&dense::matmul(&s_inv, x, k, k, k), &s_inv, k, k, k);
                let d_scale = (0..k * k).map(|i| (sws[i] - nu * s_inv[i]) * 0.5).collect();
                let d_nu = (log_det_w - log_det_s) * 0.5 + d_common;
                EventGradient {
                    log_prob,
                    x: dx,
                    parameters: vec![vec![d_nu], d_scale],
                }
            }
            (true, false) => {
                // ν/2 ln|S| − (ν + K + 1)/2 ln|W| − ½ tr(S W⁻¹)
                let ls = dense::cholesky(p[1], k);
                let lw = dense::cholesky(x, k);
                let s_inv = dense::cholesky_inverse(&ls, k);
                let w_inv = dense::cholesky_inverse(&lw, k);
                let (log_det_w, log_det_s) = (
                    dense::log_diagonal(&lw, k) * 2.0,
                    dense::log_diagonal(&ls, k) * 2.0,
                );
                let power = (nu + (kf + 1.0)) * 0.5;
                let log_prob =
                    nu * 0.5 * log_det_s - power * log_det_w - dense::dot(p[1], &w_inv) * 0.5
                        + common;
                let wsw = dense::matmul(&dense::matmul(&w_inv, p[1], k, k, k), &w_inv, k, k, k);
                let dx = (0..k * k)
                    .map(|i| wsw[i] * 0.5 - power * w_inv[i])
                    .collect();
                let d_scale = (0..k * k)
                    .map(|i| (nu * s_inv[i] - w_inv[i]) * 0.5)
                    .collect();
                let d_nu = (log_det_s - log_det_w) * 0.5 + d_common;
                EventGradient {
                    log_prob,
                    x: dx,
                    parameters: vec![vec![d_nu], d_scale],
                }
            }
            (false, true) => {
                // Σ_k (ν − k) ln L_kk (one-based k) − ½ ‖L_S⁻¹ L‖² + K ln 2 − ν Σ ln L_S,kk
                let ls = dense::lower(p[1].to_vec(), k);
                let lw = dense::lower(x.to_vec(), k);
                let m = dense::solve_lower(&ls, &lw, k, k);
                let mut log_prob =
                    common - nu * dense::log_diagonal(&ls, k) - dense::dot(&m, &m) * 0.5
                        + kf * LN_2;
                for i in 0..k {
                    log_prob = log_prob + (nu - (i as f64 + 1.0)) * lw[i * k + i].ln();
                }
                // ∂/∂L = diag((ν − k)/L_kk) − S⁻¹ L, ∂/∂L_S = L_S⁻ᵀ M Mᵀ − ν diag(1/L_S,kk), below the diagonal
                let s_inv_l = dense::solve_lower_transposed(&ls, &m, k, k);
                let mut dx: Vec<S> = s_inv_l.iter().map(|&v| -v).collect();
                let mmt = dense::matmul(&m, &dense::transpose(&m, k, k), k, k, k);
                let mut d_scale = dense::solve_lower_transposed(&ls, &mmt, k, k);
                for i in 0..k {
                    dx[i * k + i] = dx[i * k + i] + (nu - (i as f64 + 1.0)) / lw[i * k + i];
                    d_scale[i * k + i] = d_scale[i * k + i] - nu / ls[i * k + i];
                }
                let d_nu = dense::log_diagonal(&lw, k) - dense::log_diagonal(&ls, k) + d_common;
                EventGradient {
                    log_prob,
                    x: dense::lower(dx, k),
                    parameters: vec![vec![d_nu], dense::lower(d_scale, k)],
                }
            }
            (true, true) => {
                // ν Σ ln L_S,kk − Σ_k (ν + k) ln L_kk (one-based k) − ½ ‖L⁻¹ L_S‖² + K ln 2
                let ls = dense::lower(p[1].to_vec(), k);
                let lw = dense::lower(x.to_vec(), k);
                let n = dense::solve_lower(&lw, &ls, k, k);
                let mut log_prob = common + nu * dense::log_diagonal(&ls, k)
                    - dense::dot(&n, &n) * 0.5
                    + kf * LN_2;
                for i in 0..k {
                    log_prob = log_prob - (nu + (i as f64 + 1.0)) * lw[i * k + i].ln();
                }
                // ∂/∂L = L⁻ᵀ N Nᵀ − diag((ν + k)/L_kk), ∂/∂L_S = −L⁻ᵀ N + ν diag(1/L_S,kk), below the diagonal
                let nnt = dense::matmul(&n, &dense::transpose(&n, k, k), k, k, k);
                let mut dx = dense::solve_lower_transposed(&lw, &nnt, k, k);
                let mut d_scale: Vec<S> = dense::solve_lower_transposed(&lw, &n, k, k)
                    .iter()
                    .map(|&v| -v)
                    .collect();
                for i in 0..k {
                    dx[i * k + i] = dx[i * k + i] - (nu + (i as f64 + 1.0)) / lw[i * k + i];
                    d_scale[i * k + i] = d_scale[i * k + i] + nu / ls[i * k + i];
                }
                let d_nu = dense::log_diagonal(&ls, k) - dense::log_diagonal(&lw, k) + d_common;
                EventGradient {
                    log_prob,
                    x: dense::lower(dx, k),
                    parameters: vec![vec![d_nu], dense::lower(d_scale, k)],
                }
            }
        }
    }
}

/// Draws one Wishart-family matrix by the Bartlett decomposition.
fn sample_wishart(
    kernel: &WishartKernel,
    event: &[usize],
    p: &[&[f64]],
    out: &mut [f64],
    rng: &mut KeanuRng,
) -> Result<()> {
    let k = event[0];
    // the identity is in every form's support, so only the parameters are checked
    if kernel.check(event, &dense::identity::<f64>(k), p).is_some() {
        return invalid_sampling(kernel.what());
    }
    let nu = p[0][0];
    let ls = if kernel.cholesky {
        dense::lower(p[1].to_vec(), k)
    } else {
        dense::cholesky(p[1], k)
    };
    let a = bartlett(nu, k, rng)?;
    let value = if kernel.inverse {
        // W⁻¹ ~ Wishart(ν, S⁻¹): with C the Cholesky factor of S⁻¹, W = (C A Aᵀ Cᵀ)⁻¹
        let c = dense::cholesky(&dense::cholesky_inverse(&ls, k), k);
        let ca = dense::matmul(&c, &a, k, k, k);
        let w = dense::cholesky_inverse(&ca, k);
        // symmetrise the rounding
        let w: Vec<f64> = (0..k * k)
            .map(|i| 0.5 * (w[i] + w[(i % k) * k + i / k]))
            .collect();
        if kernel.cholesky {
            dense::cholesky(&w, k)
        } else {
            w
        }
    } else {
        // the Cholesky factor of L_S A Aᵀ L_Sᵀ is L_S A (lower triangular, with a positive diagonal)
        let la = dense::matmul(&ls, &a, k, k, k);
        if kernel.cholesky {
            la
        } else {
            let w = dense::matmul(&la, &dense::transpose(&la, k, k), k, k, k);
            (0..k * k)
                .map(|i| 0.5 * (w[i] + w[(i % k) * k + i / k]))
                .collect()
        }
    };
    out.copy_from_slice(&value);
    Ok(())
}

event_distribution! {
    /// The Wishart distribution with ν degrees of freedom and scale matrix S over symmetric positive definite K × K
    /// matrices W, as Stan's `wishart(ν, S)`:
    ///
    /// ```text
    /// ln p(W) = (ν − K − 1)/2 ln|W| − ½ tr(S⁻¹W) − νK/2 ln 2 − ν/2 ln|S| − ln Γ_K(ν/2)
    /// ```
    ///
    /// for real ν > K − 1, with Γ_K the multivariate gamma function; the mean is νS. It is a density over the
    /// K(K + 1)/2 distinct elements of W. The value is [..., K, K]; `nu` is a scalar or one per matrix and `scale`
    /// [..., K, K]. A value that is not symmetric (to 1e-8) and positive definite is impossible; invalid
    /// parameters give NaN. Draws use the Bartlett decomposition, W = L A Aᵀ Lᵀ with L the Cholesky factor of S,
    /// Aᵢᵢ² ~ χ²(ν − i) (zero-based i) and N(0, 1) entries below the diagonal of A.
    Wishart, WishartGradient,
    kernel = WishartKernel { inverse: false, cholesky: false }, what = "Wishart",
    (
        /// The degrees of freedom ν > K − 1.
        nu,
        /// The scale matrix S, [..., K, K].
        scale,
    )
}

event_distribution! {
    /// The Wishart distribution over the Cholesky factor L of W = L Lᵀ, given the Cholesky factor L_S of the scale,
    /// as Stan's `wishart_cholesky(ν, L_S)`: the [`Wishart`] density of L Lᵀ times the Jacobian of the map from L to
    /// W, 2^K Π_k L_kk^(K − k + 1) (one-based k), so
    ///
    /// ```text
    /// ln p(L) = Σ_k (ν − k) ln L_kk − ½ ‖L_S⁻¹ L‖²_F + K ln 2 − νK/2 ln 2 − ν Σ_k ln L_S,kk − ln Γ_K(ν/2)
    /// ```
    ///
    /// a density over the lower triangle of L. The value must be lower triangular with a positive diagonal; the
    /// upper triangle of `scale_cholesky` is ignored. Draws are L_S A, A the Bartlett factor (see [`Wishart`]).
    WishartCholesky, WishartCholeskyGradient,
    kernel = WishartKernel { inverse: false, cholesky: true }, what = "Wishart Cholesky",
    (
        /// The degrees of freedom ν > K − 1.
        nu,
        /// The Cholesky factor L_S of the scale matrix, [..., K, K].
        scale_cholesky,
    )
}

event_distribution! {
    /// The inverse Wishart distribution with ν degrees of freedom and scale matrix S over symmetric positive
    /// definite K × K matrices W, as Stan's `inv_wishart(ν, S)`:
    ///
    /// ```text
    /// ln p(W) = ν/2 ln|S| − νK/2 ln 2 − ln Γ_K(ν/2) − (ν + K + 1)/2 ln|W| − ½ tr(S W⁻¹)
    /// ```
    ///
    /// for real ν > K − 1; W⁻¹ is Wishart(ν, S⁻¹) and the mean is S/(ν − K − 1) for ν > K + 1. Shapes and
    /// conventions are as for [`Wishart`]. Draws invert a Wishart(ν, S⁻¹) draw.
    InverseWishart, InverseWishartGradient,
    kernel = WishartKernel { inverse: true, cholesky: false }, what = "inverse Wishart",
    (
        /// The degrees of freedom ν > K − 1.
        nu,
        /// The scale matrix S, [..., K, K].
        scale,
    )
}

event_distribution! {
    /// The inverse Wishart distribution over the Cholesky factor L of W = L Lᵀ, given the Cholesky factor L_S of the
    /// scale, as Stan's `inv_wishart_cholesky(ν, L_S)`: the [`InverseWishart`] density of L Lᵀ times the Jacobian
    /// 2^K Π_k L_kk^(K − k + 1) (one-based k), so
    ///
    /// ```text
    /// ln p(L) = ν Σ_k ln L_S,kk − νK/2 ln 2 − ln Γ_K(ν/2) − Σ_k (ν + k) ln L_kk − ½ ‖L⁻¹ L_S‖²_F + K ln 2
    /// ```
    ///
    /// Conventions are as for [`WishartCholesky`]. Draws factorise an inverse Wishart draw.
    InverseWishartCholesky, InverseWishartCholeskyGradient,
    kernel = WishartKernel { inverse: true, cholesky: true }, what = "inverse Wishart Cholesky",
    (
        /// The degrees of freedom ν > K − 1.
        nu,
        /// The Cholesky factor L_S of the scale matrix, [..., K, K].
        scale_cholesky,
    )
}

macro_rules! wishart_sampler {
    ($($name:ident: $inverse:literal, $cholesky:literal;)+) => {$(
        impl $name<'_> {
            /// Draws matrices of the given shape ([..., K, K]). An error unless ν > K − 1 and the scale (or its
            /// Cholesky factor) is valid.
            pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
                let kernel = WishartKernel { inverse: $inverse, cholesky: $cholesky };
                self.events().sample(shape, rng, |e, p, out, rng| sample_wishart(&kernel, e, p, out, rng))
            }
        }
    )+};
}

wishart_sampler! {
    Wishart: false, false;
    WishartCholesky: false, true;
    InverseWishart: true, false;
    InverseWishartCholesky: true, true;
}

// ---------------------------------------------------------------------------------------------------------------
// Matrix normal

struct MatrixNormalKernel;

impl EventFamily for MatrixNormalKernel {
    const EVENT_RANK: usize = 2;

    fn validate_event(&self, event: &[usize]) -> Result<()> {
        if event.contains(&0) {
            return Err(DistributionError::InvalidValue(format!(
                "a matrix normal value must have non-empty axes, not {event:?}"
            )));
        }
        Ok(())
    }

    fn parameter_events(&self, event: &[usize]) -> Vec<Vec<usize>> {
        let (n, p) = (event[0], event[1]);
        vec![vec![n, p], vec![n, n], vec![p, p]]
    }

    fn check(&self, event: &[usize], x: &[f64], p: &[&[f64]]) -> Option<f64> {
        let (n, q) = (event[0], event[1]);
        if p[0].iter().any(|m| !m.is_finite())
            || !symmetric_positive_definite(p[1], n)
            || !symmetric_positive_definite(p[2], q)
        {
            return Some(f64::NAN);
        }
        x.iter()
            .any(|v| !v.is_finite())
            .then_some(f64::NEG_INFINITY)
    }

    fn gradient<S: Real>(&self, event: &[usize], x: &[S], p: &[&[S]]) -> EventGradient<S> {
        let (n, q) = (event[0], event[1]);
        let (lu, lv) = (dense::cholesky(p[1], n), dense::cholesky(p[2], q));
        let r: Vec<S> = x.iter().zip(p[0]).map(|(&x, &m)| x - m).collect();
        // B = U⁻¹ R, A = B V⁻¹, D = R V⁻¹
        let b = dense::solve_lower_transposed(&lu, &dense::solve_lower(&lu, &r, n, q), n, q);
        let (u_inv, v_inv) = (
            dense::cholesky_inverse(&lu, n),
            dense::cholesky_inverse(&lv, q),
        );
        let a = dense::matmul(&b, &v_inv, n, q, q);
        let d = dense::matmul(&r, &v_inv, n, q, q);
        let (nf, qf) = (n as f64, q as f64);
        let log_prob = S::constant(-nf * qf / 2.0 * (2.0 * PI).ln())
            - dense::log_diagonal(&lu, n) * qf
            - dense::log_diagonal(&lv, q) * nf
            - dense::dot(&a, &r) * 0.5;
        let abt = dense::matmul(&a, &dense::transpose(&b, n, q), n, q, n);
        let dta = dense::matmul(&dense::transpose(&d, n, q), &a, q, n, q);
        EventGradient {
            log_prob,
            x: a.iter().map(|&v| -v).collect(),
            parameters: vec![
                a.clone(),
                (0..n * n).map(|i| (abt[i] - u_inv[i] * qf) * 0.5).collect(),
                (0..q * q).map(|i| (dta[i] - v_inv[i] * nf) * 0.5).collect(),
            ],
        }
    }
}

event_distribution! {
    /// The matrix normal distribution of n × p matrices X with mean M, row covariance U (n × n) and column
    /// covariance V (p × p), as PyMC's `MatrixNormal(mu, rowcov, colcov)`: vec(X) is multivariate normal with
    /// covariance V ⊗ U, so
    ///
    /// ```text
    /// ln p(X) = −np/2 ln 2π − p/2 ln|U| − n/2 ln|V| − ½ tr(V⁻¹ (X − M)ᵀ U⁻¹ (X − M))
    /// ```
    ///
    /// The value is [..., n, p], `mu` [..., n, p], `row_covariance` [..., n, n] and `column_covariance`
    /// [..., p, p]; both covariances must be symmetric positive definite (U and V are identified only up to a
    /// factor c, U c and V / c giving the same distribution). Draws are M + L_U Z L_Vᵀ with Z standard normal.
    MatrixNormal, MatrixNormalGradient,
    kernel = MatrixNormalKernel, what = "matrix normal",
    (
        /// The mean M, [..., n, p].
        mu,
        /// The row covariance U, [..., n, n].
        row_covariance,
        /// The column covariance V, [..., p, p].
        column_covariance,
    )
}

impl MatrixNormal<'_> {
    /// Draws matrices of the given shape ([..., n, p]). An error unless both covariances are symmetric positive
    /// definite.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        self.events().sample(shape, rng, |event, p, out, rng| {
            let (n, q) = (event[0], event[1]);
            if MatrixNormalKernel
                .check(event, &vec![0.0; n * q], p)
                .is_some()
            {
                return invalid_sampling("matrix normal");
            }
            let (lu, lv) = (dense::cholesky(p[1], n), dense::cholesky(p[2], q));
            let z: Vec<f64> = (0..n * q).map(|_| rng.next_gaussian()).collect();
            let lz = dense::matmul(&lu, &z, n, n, q);
            let x = dense::matmul(&lz, &dense::transpose(&lv, q, q), n, q, q);
            for ((o, x), m) in out.iter_mut().zip(x).zip(p[0]) {
                *o = m + x;
            }
            Ok(())
        })
    }
}
