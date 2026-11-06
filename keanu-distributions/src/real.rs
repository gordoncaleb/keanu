//! The arithmetic that hand-written gradients are written in, so that the same code gives the gradient (on plain
//! numbers) and its directional derivatives, that is Hessian-vector products (on [`Dual`] numbers: forward mode over
//! the hand-written reverse pass), plus small dense matrix routines generic over it.

use keanu_tensor::special::{bessel_i_ratio, digamma, log_bessel_i, trigamma};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A real number type for gradient code: f64, or a [`Dual`] number carrying one tangent.
pub(crate) trait Real:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    fn constant(v: f64) -> Self;
    /// The value, without its tangent.
    fn value(self) -> f64;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    /// ln Γ(self).
    fn ln_gamma(self) -> Self;
    /// ψ(self), the digamma function.
    fn digamma(self) -> Self;
    /// ln I_ν(self), for a constant order ν >= 0.
    fn log_bessel_i(self, nu: f64) -> Self;
    /// I_{ν+1}(self) / I_ν(self), for a constant order ν >= 0.
    fn bessel_i_ratio(self, nu: f64) -> Self;

    fn zero() -> Self {
        Self::constant(0.0)
    }
}

/// The derivative of A(x) = I_{ν+1}(x)/I_ν(x), A' = 1 − A² − (2ν + 1) A/x (from the recurrences of I_ν), with A/x
/// from its series 1/(2ν + 2) − x²/((2ν + 2)²(2ν + 4)) near 0.
pub(crate) fn bessel_i_ratio_dx(nu: f64, x: f64, a: f64) -> f64 {
    let p = 2.0 * nu + 2.0;
    let a_over_x = if x < 1e-4 {
        1.0 / p - x * x / (p * p * (p + 2.0))
    } else {
        a / x
    };
    1.0 - a * a - (2.0 * nu + 1.0) * a_over_x
}

impl Real for f64 {
    fn constant(v: f64) -> Self {
        v
    }
    fn value(self) -> f64 {
        self
    }
    fn ln(self) -> Self {
        f64::ln(self)
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn ln_gamma(self) -> Self {
        keanu_tensor::special::ln_gamma(self)
    }
    fn digamma(self) -> Self {
        digamma(self)
    }
    fn log_bessel_i(self, nu: f64) -> Self {
        log_bessel_i(nu, self)
    }
    fn bessel_i_ratio(self, nu: f64) -> Self {
        bessel_i_ratio(nu, self)
    }
}

/// A number with one tangent: `v + d ε` with `ε² = 0`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Dual {
    pub(crate) v: f64,
    pub(crate) d: f64,
}

impl Dual {
    pub(crate) fn new(v: f64, d: f64) -> Self {
        Self { v, d }
    }

    /// f(self) given f and f' at the value.
    fn chain(self, f: f64, df: f64) -> Self {
        Self {
            v: f,
            d: if self.d == 0.0 { 0.0 } else { df * self.d },
        }
    }
}

impl Add for Dual {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.v + o.v, self.d + o.d)
    }
}

impl Sub for Dual {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.v - o.v, self.d - o.d)
    }
}

impl Mul for Dual {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self::new(self.v * o.v, self.d * o.v + self.v * o.d)
    }
}

impl Div for Dual {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        let v = self.v / o.v;
        Self::new(v, (self.d - v * o.d) / o.v)
    }
}

impl Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.v, -self.d)
    }
}

impl Add<f64> for Dual {
    type Output = Self;
    fn add(self, o: f64) -> Self {
        Self::new(self.v + o, self.d)
    }
}

impl Sub<f64> for Dual {
    type Output = Self;
    fn sub(self, o: f64) -> Self {
        Self::new(self.v - o, self.d)
    }
}

impl Mul<f64> for Dual {
    type Output = Self;
    fn mul(self, o: f64) -> Self {
        Self::new(self.v * o, self.d * o)
    }
}

impl Div<f64> for Dual {
    type Output = Self;
    fn div(self, o: f64) -> Self {
        Self::new(self.v / o, self.d / o)
    }
}

impl Real for Dual {
    fn constant(v: f64) -> Self {
        Self::new(v, 0.0)
    }
    fn value(self) -> f64 {
        self.v
    }
    fn ln(self) -> Self {
        self.chain(self.v.ln(), 1.0 / self.v)
    }
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        self.chain(s, 0.5 / s)
    }
    fn ln_gamma(self) -> Self {
        self.chain(keanu_tensor::special::ln_gamma(self.v), digamma(self.v))
    }
    fn digamma(self) -> Self {
        self.chain(digamma(self.v), trigamma(self.v))
    }
    fn log_bessel_i(self, nu: f64) -> Self {
        let a = bessel_i_ratio(nu, self.v);
        self.chain(log_bessel_i(nu, self.v), a + nu / self.v)
    }
    fn bessel_i_ratio(self, nu: f64) -> Self {
        let a = bessel_i_ratio(nu, self.v);
        self.chain(a, bessel_i_ratio_dx(nu, self.v, a))
    }
}

/// Row-major dense matrices of [`Real`]s, for the multivariate families' gradients. Matrices are slices of `n × m`
/// elements; a K × K Cholesky factor is lower triangular, its upper triangle never read.
pub(crate) mod dense {
    use super::Real;

    /// The lower triangular Cholesky factor of the symmetric K × K matrix `a` (only its lower triangle is read),
    /// with NaNs from the first non-positive pivot on (so a matrix that is not positive definite gives NaN results,
    /// never a panic).
    pub(crate) fn cholesky<S: Real>(a: &[S], k: usize) -> Vec<S> {
        let mut l = vec![S::zero(); k * k];
        for i in 0..k {
            for j in 0..=i {
                let mut s = a[i * k + j];
                for m in 0..j {
                    s = s - l[i * k + m] * l[j * k + m];
                }
                l[i * k + j] = if i == j {
                    if s.value() > 0.0 {
                        s.sqrt()
                    } else {
                        S::constant(f64::NAN)
                    }
                } else {
                    s / l[j * k + j]
                };
            }
        }
        l
    }

    /// X with L X = B, for a lower triangular K × K `l` and a K × m `b`.
    pub(crate) fn solve_lower<S: Real>(l: &[S], b: &[S], k: usize, m: usize) -> Vec<S> {
        let mut x = b.to_vec();
        for c in 0..m {
            for i in 0..k {
                let mut s = x[i * m + c];
                for j in 0..i {
                    s = s - l[i * k + j] * x[j * m + c];
                }
                x[i * m + c] = s / l[i * k + i];
            }
        }
        x
    }

    /// X with Lᵀ X = B, for a lower triangular K × K `l` and a K × m `b`.
    pub(crate) fn solve_lower_transposed<S: Real>(l: &[S], b: &[S], k: usize, m: usize) -> Vec<S> {
        let mut x = b.to_vec();
        for c in 0..m {
            for i in (0..k).rev() {
                let mut s = x[i * m + c];
                for j in i + 1..k {
                    s = s - l[j * k + i] * x[j * m + c];
                }
                x[i * m + c] = s / l[i * k + i];
            }
        }
        x
    }

    /// The K × K identity.
    pub(crate) fn identity<S: Real>(k: usize) -> Vec<S> {
        let mut eye = vec![S::zero(); k * k];
        for i in 0..k {
            eye[i * k + i] = S::constant(1.0);
        }
        eye
    }

    /// (L Lᵀ)⁻¹ = L⁻ᵀ L⁻¹ from the Cholesky factor `l`.
    pub(crate) fn cholesky_inverse<S: Real>(l: &[S], k: usize) -> Vec<S> {
        let inverse = solve_lower(l, &identity(k), k, k);
        solve_lower_transposed(l, &inverse, k, k)
    }

    /// Σ ln L_ii.
    pub(crate) fn log_diagonal<S: Real>(l: &[S], k: usize) -> S {
        (0..k).fold(S::zero(), |s, i| s + l[i * k + i].ln())
    }

    /// The product of an n × m `a` and an m × p `b`.
    pub(crate) fn matmul<S: Real>(a: &[S], b: &[S], n: usize, m: usize, p: usize) -> Vec<S> {
        let mut c = vec![S::zero(); n * p];
        for i in 0..n {
            for j in 0..m {
                let a_ij = a[i * m + j];
                for q in 0..p {
                    c[i * p + q] = c[i * p + q] + a_ij * b[j * p + q];
                }
            }
        }
        c
    }

    /// The transpose of an n × m `a`.
    pub(crate) fn transpose<S: Real>(a: &[S], n: usize, m: usize) -> Vec<S> {
        let mut t = vec![S::zero(); n * m];
        for i in 0..n {
            for j in 0..m {
                t[j * n + i] = a[i * m + j];
            }
        }
        t
    }

    /// The lower triangle of a K × K matrix (its upper triangle set to zero).
    pub(crate) fn lower<S: Real>(mut a: Vec<S>, k: usize) -> Vec<S> {
        for i in 0..k {
            for j in i + 1..k {
                a[i * k + j] = S::zero();
            }
        }
        a
    }

    /// Σ a_ij b_ij.
    pub(crate) fn dot<S: Real>(a: &[S], b: &[S]) -> S {
        a.iter().zip(b).fold(S::zero(), |s, (&x, &y)| s + x * y)
    }

    /// Whether the K × K `a` (f64) is symmetric to a relative tolerance.
    pub(crate) fn is_symmetric(a: &[f64], k: usize, tolerance: f64) -> bool {
        (0..k).all(|i| {
            (0..i).all(|j| {
                let (x, y) = (a[i * k + j], a[j * k + i]);
                (x - y).abs() <= tolerance * x.abs().max(y.abs()).max(1.0)
            })
        })
    }

    /// Whether `l` is lower triangular with a positive, finite diagonal and finite entries.
    pub(crate) fn is_cholesky_factor(l: &[f64], k: usize) -> bool {
        (0..k).all(|i| {
            l[i * k + i] > 0.0
                && l[i * k..(i + 1) * k].iter().all(|v| v.is_finite())
                && l[i * k + i + 1..(i + 1) * k].iter().all(|&v| v == 0.0)
        })
    }
}
