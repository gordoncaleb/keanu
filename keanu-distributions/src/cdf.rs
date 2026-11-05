//! Log cumulative distribution functions, log survival functions, their gradients and inverse CDFs.
//!
//! For a univariate distribution, `log_cdf(x)` is ln P(X <= x) and `log_ccdf(x)` is ln P(X > x), element-wise and
//! broadcasting x with the parameters like `log_prob`. Both are computed so that they stay finite and accurate
//! in the tail where the probability is small, even where it underflows (from ln Φ for the Gaussian family, and
//! for the incomplete gamma and beta families by computing whichever of the CDF and survival function is the
//! smaller directly, in log space). `d_log_cdf` and `d_log_ccdf` return the same gradient struct as `d_log_prob`,
//! holding the derivatives of the log CDF (or survival function) with respect to x and each parameter, formed as
//! ratios that stay finite in those tails. They are unspecified where the log CDF (or survival function) is
//! -infinity.
//!
//! Continuous distributions whose quantile function has a closed form also provide `inverse_cdf(p)`, the x with
//! P(X <= x) = p, and `inverse_ccdf(q)`, the x with P(X > x) = q, each accurate for small probabilities.

use crate::continuous::LN_SQRT_2PI;
use crate::error::Result;
use crate::util::{ln_gamma, map, map_n, safe_ln_times, softplus};
use crate::{
    Beta, BetaGradient, Binomial, BinomialGradient, Cauchy, CauchyGradient, Exponential,
    ExponentialGradient, Gamma, GammaGradient, Gaussian, GaussianGradient, Laplace,
    LaplaceGradient, LogNormal, LogNormalGradient, Logistic, LogisticGradient, NegativeBinomial,
    NegativeBinomialGradient, Poisson, PoissonGradient, StudentT, StudentTGradient, Uniform,
    UniformGradient, Weibull, WeibullGradient,
};
use keanu_tensor::special::{
    log_beta, log_gamma_p, log_gamma_p_da, log_gamma_q, log_gamma_q_da, log_inc_beta,
    log_inc_beta_da, log_inc_beta_db, log_ndtr, log1mexp, logit, ndtri,
};
use keanu_tensor::{DoubleTensor, IntTensor};
use std::f64::consts::{LN_2, PI};

/// ln p given ln p and (lazily) ln q for its complement q = 1 - p, each computed directly: ln p itself while
/// p < ½, where it keeps its precision, and ln(1 - q) beyond. Working with logs keeps far tails finite.
#[inline]
fn log_complement(log_p: f64, log_q: impl FnOnce() -> f64) -> f64 {
    if log_p < -LN_2 {
        log_p
    } else {
        log1mexp(log_q())
    }
}

/// e^(log_numerator - log_denominator), taken to be 0 where either is -infinity (where a log CDF gradient is
/// unspecified, or the density is 0).
#[inline]
pub(crate) fn ratio(log_numerator: f64, log_denominator: f64) -> f64 {
    if log_numerator == f64::NEG_INFINITY || log_denominator == f64::NEG_INFINITY {
        0.0
    } else {
        (log_numerator - log_denominator).exp()
    }
}

/// A standardised symmetric distribution (location 0, scale 1), from which a location-scale family's CDF
/// functions follow: P(X > z) = P(X <= -z).
struct Standard {
    log_cdf: fn(f64) -> f64,
    log_pdf: fn(f64) -> f64,
    quantile: fn(f64) -> f64,
}

impl Standard {
    /// ln P(X <= x), or ln P(X > x) if `upper`, for location `l` and scale `s`.
    #[inline]
    fn log_cdf(&self, x: f64, l: f64, s: f64, upper: bool) -> f64 {
        let z = (x - l) / s;
        (self.log_cdf)(if upper { -z } else { z })
    }

    /// The derivatives of [`Self::log_cdf`] with respect to x, the location and the scale.
    #[inline]
    fn gradient(&self, x: f64, l: f64, s: f64, upper: bool) -> [f64; 3] {
        let z = (x - l) / s;
        let d_z = if upper {
            -ratio((self.log_pdf)(z), (self.log_cdf)(-z))
        } else {
            ratio((self.log_pdf)(z), (self.log_cdf)(z))
        };
        [d_z / s, -d_z / s, -z * d_z / s]
    }
}

const STANDARD_GAUSSIAN: Standard = Standard {
    log_cdf: log_ndtr,
    log_pdf: |z| -0.5 * z * z - LN_SQRT_2PI,
    quantile: ndtri,
};

const STANDARD_CAUCHY: Standard = Standard {
    // atan2(1, -z) / π is ½ + atan(z) / π, without cancellation for very negative z
    log_cdf: |z| (1f64.atan2(-z) / PI).ln(),
    log_pdf: |z| -PI.ln() - z.mul_add(z, 1.0).ln(),
    // from the nearer tail, so tiny probabilities keep their precision
    quantile: |p| {
        if p < 0.5 {
            -1.0 / (PI * p).tan()
        } else {
            1.0 / (PI * (1.0 - p)).tan()
        }
    },
};

const STANDARD_LAPLACE: Standard = Standard {
    log_cdf: |z| {
        if z < 0.0 {
            z - LN_2
        } else {
            (-0.5 * (-z).exp()).ln_1p()
        }
    },
    log_pdf: |z| -z.abs() - LN_2,
    quantile: |p| {
        if p < 0.5 {
            (2.0 * p).ln()
        } else {
            -(2.0 * (1.0 - p)).ln()
        }
    },
};

const STANDARD_LOGISTIC: Standard = Standard {
    log_cdf: |z| -softplus(-z),
    log_pdf: |z| -z.abs() - 2.0 * softplus(-z.abs()),
    quantile: logit,
};

/// The CDF methods of a location-scale family with a symmetric standard form.
macro_rules! location_scale_cdf {
    ($ty:ident, $gradient:ident, $loc:ident, $scale:ident, $standard:ident) => {
        impl $ty<'_> {
            /// The element-wise log CDF, ln P(X <= x), broadcasting x with the parameters.
            pub fn log_cdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
                map((x, &*self.$loc, &*self.$scale), |(x, l, s)| {
                    $standard.log_cdf(x, l, s, false)
                })
            }

            /// The element-wise log survival function, ln P(X > x), broadcasting x with the parameters.
            pub fn log_ccdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
                map((x, &*self.$loc, &*self.$scale), |(x, l, s)| {
                    $standard.log_cdf(x, l, s, true)
                })
            }

            /// The element-wise derivatives of [`Self::log_cdf`] with respect to x and each parameter.
            pub fn d_log_cdf(&self, x: &DoubleTensor) -> Result<$gradient> {
                let [x, $loc, $scale] = map_n((x, &*self.$loc, &*self.$scale), |(x, l, s)| {
                    $standard.gradient(x, l, s, false)
                })?;
                Ok($gradient { x, $loc, $scale })
            }

            /// The element-wise derivatives of [`Self::log_ccdf`] with respect to x and each parameter.
            pub fn d_log_ccdf(&self, x: &DoubleTensor) -> Result<$gradient> {
                let [x, $loc, $scale] = map_n((x, &*self.$loc, &*self.$scale), |(x, l, s)| {
                    $standard.gradient(x, l, s, true)
                })?;
                Ok($gradient { x, $loc, $scale })
            }

            /// The element-wise inverse CDF: the x with P(X <= x) = p, for p in [0, 1].
            pub fn inverse_cdf(&self, p: &DoubleTensor) -> Result<DoubleTensor> {
                map((p, &*self.$loc, &*self.$scale), |(p, l, s)| {
                    l + s * ($standard.quantile)(p)
                })
            }

            /// The element-wise inverse survival function: the x with P(X > x) = q, for q in [0, 1].
            pub fn inverse_ccdf(&self, q: &DoubleTensor) -> Result<DoubleTensor> {
                map((q, &*self.$loc, &*self.$scale), |(q, l, s)| {
                    l - s * ($standard.quantile)(q)
                })
            }
        }
    };
}

location_scale_cdf!(Gaussian, GaussianGradient, mu, sigma, STANDARD_GAUSSIAN);
location_scale_cdf!(Cauchy, CauchyGradient, location, scale, STANDARD_CAUCHY);
location_scale_cdf!(Laplace, LaplaceGradient, mu, beta, STANDARD_LAPLACE);
location_scale_cdf!(Logistic, LogisticGradient, mu, s, STANDARD_LOGISTIC);

impl LogNormal<'_> {
    /// The element-wise log CDF, ln P(X <= x), broadcasting x with the parameters (-infinity for x <= 0).
    pub fn log_cdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.mu, &*self.sigma), |(x, m, s)| {
            if x <= 0.0 {
                f64::NEG_INFINITY
            } else {
                STANDARD_GAUSSIAN.log_cdf(x.ln(), m, s, false)
            }
        })
    }

    /// The element-wise log survival function, ln P(X > x), broadcasting x with the parameters (0 for x <= 0).
    pub fn log_ccdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.mu, &*self.sigma), |(x, m, s)| {
            if x <= 0.0 {
                0.0
            } else {
                STANDARD_GAUSSIAN.log_cdf(x.ln(), m, s, true)
            }
        })
    }

    fn cdf_gradient(&self, x: &DoubleTensor, upper: bool) -> Result<LogNormalGradient> {
        let [x, mu, sigma] = map_n((x, &*self.mu, &*self.sigma), |(x, m, s)| {
            if x <= 0.0 {
                return [0.0; 3];
            }
            let [d_ln_x, d_mu, d_sigma] = STANDARD_GAUSSIAN.gradient(x.ln(), m, s, upper);
            [d_ln_x / x, d_mu, d_sigma]
        })?;
        Ok(LogNormalGradient { x, mu, sigma })
    }

    /// The element-wise derivatives of [`Self::log_cdf`] with respect to x and each parameter.
    pub fn d_log_cdf(&self, x: &DoubleTensor) -> Result<LogNormalGradient> {
        self.cdf_gradient(x, false)
    }

    /// The element-wise derivatives of [`Self::log_ccdf`] with respect to x and each parameter.
    pub fn d_log_ccdf(&self, x: &DoubleTensor) -> Result<LogNormalGradient> {
        self.cdf_gradient(x, true)
    }

    /// The element-wise inverse CDF: the x with P(X <= x) = p, for p in [0, 1].
    pub fn inverse_cdf(&self, p: &DoubleTensor) -> Result<DoubleTensor> {
        map((p, &*self.mu, &*self.sigma), |(p, m, s)| {
            (m + s * ndtri(p)).exp()
        })
    }

    /// The element-wise inverse survival function: the x with P(X > x) = q, for q in [0, 1].
    pub fn inverse_ccdf(&self, q: &DoubleTensor) -> Result<DoubleTensor> {
        map((q, &*self.mu, &*self.sigma), |(q, m, s)| {
            (m - s * ndtri(q)).exp()
        })
    }
}

impl Exponential<'_> {
    /// The element-wise log CDF, ln P(X <= x) = ln(1 - e^(-rate x)), broadcasting x with the parameters
    /// (-infinity for x <= 0).
    pub fn log_cdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.rate), |(x, rate)| {
            if x <= 0.0 {
                f64::NEG_INFINITY
            } else {
                log1mexp(-rate * x)
            }
        })
    }

    /// The element-wise log survival function, ln P(X > x) = -rate x, broadcasting x with the parameters (0 for
    /// x <= 0).
    pub fn log_ccdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map(
            (x, &*self.rate),
            |(x, rate)| if x <= 0.0 { 0.0 } else { -rate * x },
        )
    }

    /// The element-wise derivatives of [`Self::log_cdf`] with respect to x and each parameter.
    pub fn d_log_cdf(&self, x: &DoubleTensor) -> Result<ExponentialGradient> {
        let [x, rate] = map_n((x, &*self.rate), |(x, rate)| {
            if x <= 0.0 {
                return [0.0; 2];
            }
            // d/dx ln(1 - e^(-rate x)) = rate / (e^(rate x) - 1)
            let e = (rate * x).exp_m1();
            [rate / e, x / e]
        })?;
        Ok(ExponentialGradient { x, rate })
    }

    /// The element-wise derivatives of [`Self::log_ccdf`] with respect to x and each parameter.
    pub fn d_log_ccdf(&self, x: &DoubleTensor) -> Result<ExponentialGradient> {
        let [x, rate] = map_n((x, &*self.rate), |(x, rate)| {
            if x <= 0.0 { [0.0; 2] } else { [-rate, -x] }
        })?;
        Ok(ExponentialGradient { x, rate })
    }

    /// The element-wise inverse CDF: the x with P(X <= x) = p, for p in [0, 1].
    pub fn inverse_cdf(&self, p: &DoubleTensor) -> Result<DoubleTensor> {
        map((p, &*self.rate), |(p, rate)| -(-p).ln_1p() / rate)
    }

    /// The element-wise inverse survival function: the x with P(X > x) = q, for q in [0, 1].
    pub fn inverse_ccdf(&self, q: &DoubleTensor) -> Result<DoubleTensor> {
        map((q, &*self.rate), |(q, rate)| -q.ln() / rate)
    }
}

/// The gamma log density with scale `theta` and shape `k` at x > 0.
fn gamma_log_pdf(x: f64, theta: f64, k: f64) -> f64 {
    (k - 1.0) * x.ln() - ln_gamma(k) - x / theta - k * theta.ln()
}

impl Gamma<'_> {
    /// ln P(X <= x), or ln P(X > x) if `upper`, from the regularised incomplete gamma functions.
    fn log_cdf_at(x: f64, theta: f64, k: f64, upper: bool) -> f64 {
        if x <= 0.0 {
            return if upper { 0.0 } else { f64::NEG_INFINITY };
        }
        let y = x / theta;
        if upper {
            log_gamma_q(k, y)
        } else {
            log_gamma_p(k, y)
        }
    }

    /// The element-wise log CDF, ln P(X <= x) = ln P(k, x / theta), broadcasting x with the parameters
    /// (-infinity for x <= 0).
    pub fn log_cdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.theta, &*self.k), |(x, t, k)| {
            Self::log_cdf_at(x, t, k, false)
        })
    }

    /// The element-wise log survival function, ln P(X > x) = ln Q(k, x / theta), broadcasting x with the
    /// parameters (0 for x <= 0).
    pub fn log_ccdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.theta, &*self.k), |(x, t, k)| {
            Self::log_cdf_at(x, t, k, true)
        })
    }

    fn cdf_gradient(&self, x: &DoubleTensor, upper: bool) -> Result<GammaGradient> {
        let [x, theta, k] = map_n((x, &*self.theta, &*self.k), |(x, t, k)| {
            if x <= 0.0 || x == f64::INFINITY {
                return [0.0; 3];
            }
            let log_f = Self::log_cdf_at(x, t, k, upper);
            let sign = if upper { -1.0 } else { 1.0 };
            // d P(k, x/θ) / dx is the density
            let d_x = sign * ratio(gamma_log_pdf(x, t, k), log_f);
            // ∂ ln P/∂k or ∂ ln Q/∂k, formed stably in the tails
            let d_k = if upper {
                log_gamma_q_da(k, x / t)
            } else {
                log_gamma_p_da(k, x / t)
            };
            [d_x, -x / t * d_x, d_k]
        })?;
        Ok(GammaGradient { x, theta, k })
    }

    /// The element-wise derivatives of [`Self::log_cdf`] with respect to x and each parameter (the shape's from
    /// the exact derivative of the incomplete gamma function).
    pub fn d_log_cdf(&self, x: &DoubleTensor) -> Result<GammaGradient> {
        self.cdf_gradient(x, false)
    }

    /// The element-wise derivatives of [`Self::log_ccdf`] with respect to x and each parameter.
    pub fn d_log_ccdf(&self, x: &DoubleTensor) -> Result<GammaGradient> {
        self.cdf_gradient(x, true)
    }
}

impl Beta<'_> {
    /// ln P(X <= x), or ln P(X > x) if `upper`, with y = (x - x_min) / (x_max - x_min). The survival function
    /// is I₁₋ᵧ(β, α), computed directly.
    fn log_cdf_at(y: f64, a: f64, b: f64, upper: bool) -> f64 {
        let (lower_end, upper_end) = if upper {
            (0.0, f64::NEG_INFINITY)
        } else {
            (f64::NEG_INFINITY, 0.0)
        };
        if y <= 0.0 {
            return lower_end;
        }
        if y >= 1.0 {
            return upper_end;
        }
        let (a, b, y) = if upper { (b, a, 1.0 - y) } else { (a, b, y) };
        log_complement(log_inc_beta(a, b, y), || log_inc_beta(b, a, 1.0 - y))
    }

    /// The element-wise log CDF, ln P(X <= x) = ln Iᵧ(α, β) with y = (x - x_min) / (x_max - x_min), broadcasting
    /// x with the parameters.
    pub fn log_cdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map(
            (x, &*self.alpha, &*self.beta, &*self.x_min, &*self.x_max),
            |(x, a, b, min, max)| Self::log_cdf_at((x - min) / (max - min), a, b, false),
        )
    }

    /// The element-wise log survival function, ln P(X > x), broadcasting x with the parameters.
    pub fn log_ccdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map(
            (x, &*self.alpha, &*self.beta, &*self.x_min, &*self.x_max),
            |(x, a, b, min, max)| Self::log_cdf_at((x - min) / (max - min), a, b, true),
        )
    }

    fn cdf_gradient(&self, x: &DoubleTensor, upper: bool) -> Result<BetaGradient> {
        let [x, alpha, beta] = map_n(
            (x, &*self.alpha, &*self.beta, &*self.x_min, &*self.x_max),
            |(x, a, b, min, max)| {
                let range = max - min;
                let y = (x - min) / range;
                if y <= 0.0 || y >= 1.0 {
                    return [0.0; 3];
                }
                let log_f = Self::log_cdf_at(y, a, b, upper);
                let log_pdf =
                    (a - 1.0) * y.ln() + (b - 1.0) * (1.0 - y).ln() - log_beta(a, b) - range.ln();
                if upper {
                    // S = I₁₋ᵧ(β, α)
                    [
                        -ratio(log_pdf, log_f),
                        log_inc_beta_db(b, a, 1.0 - y),
                        log_inc_beta_da(b, a, 1.0 - y),
                    ]
                } else {
                    [
                        ratio(log_pdf, log_f),
                        log_inc_beta_da(a, b, y),
                        log_inc_beta_db(a, b, y),
                    ]
                }
            },
        )?;
        Ok(BetaGradient { x, alpha, beta })
    }

    /// The element-wise derivatives of [`Self::log_cdf`] with respect to x and the shape parameters (from the
    /// exact derivatives of the incomplete beta function).
    pub fn d_log_cdf(&self, x: &DoubleTensor) -> Result<BetaGradient> {
        self.cdf_gradient(x, false)
    }

    /// The element-wise derivatives of [`Self::log_ccdf`] with respect to x and the shape parameters.
    pub fn d_log_ccdf(&self, x: &DoubleTensor) -> Result<BetaGradient> {
        self.cdf_gradient(x, true)
    }
}

/// The Student's t log density with v degrees of freedom.
fn student_t_log_pdf(t: f64, v: f64) -> f64 {
    ln_gamma((v + 1.0) / 2.0)
        - v.ln() / 2.0
        - PI.ln() / 2.0
        - ln_gamma(v / 2.0)
        - (v + 1.0) / 2.0 * (t * t / v).ln_1p()
}

/// ln P(T <= t) for Student's t with v degrees of freedom: ½ I_(v / (v + t²))(v/2, ½) is the smaller tail.
fn student_t_log_cdf(t: f64, v: f64) -> f64 {
    if t == f64::INFINITY {
        return 0.0;
    }
    if t == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let log_tail = log_inc_beta(v / 2.0, 0.5, v / (v + t * t)) - LN_2;
    if t <= 0.0 {
        log_tail
    } else {
        log1mexp(log_tail)
    }
}

impl StudentT<'_> {
    /// The element-wise log CDF, ln P(X <= x), broadcasting x with the parameters.
    pub fn log_cdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.v), |(t, v)| student_t_log_cdf(t, v as f64))
    }

    /// The element-wise log survival function, ln P(X > x), broadcasting x with the parameters.
    pub fn log_ccdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.v), |(t, v)| student_t_log_cdf(-t, v as f64))
    }

    /// The element-wise derivative of [`Self::log_cdf`] with respect to x.
    pub fn d_log_cdf(&self, x: &DoubleTensor) -> Result<StudentTGradient> {
        Ok(StudentTGradient {
            x: map((x, &*self.v), |(t, v)| {
                let v = v as f64;
                ratio(student_t_log_pdf(t, v), student_t_log_cdf(t, v))
            })?,
        })
    }

    /// The element-wise derivative of [`Self::log_ccdf`] with respect to x.
    pub fn d_log_ccdf(&self, x: &DoubleTensor) -> Result<StudentTGradient> {
        Ok(StudentTGradient {
            x: map((x, &*self.v), |(t, v)| {
                let v = v as f64;
                -ratio(student_t_log_pdf(t, v), student_t_log_cdf(-t, v))
            })?,
        })
    }
}

impl Uniform<'_> {
    /// The element-wise log CDF, ln P(X <= x) = ln((x - x_min) / (x_max - x_min)) inside the bounds, broadcasting
    /// x with the parameters.
    pub fn log_cdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.x_min, &*self.x_max), |(x, a, b)| {
            if x <= a {
                f64::NEG_INFINITY
            } else if x >= b {
                0.0
            } else {
                ((x - a) / (b - a)).ln()
            }
        })
    }

    /// The element-wise log survival function, ln P(X > x) = ln((x_max - x) / (x_max - x_min)) inside the
    /// bounds, broadcasting x with the parameters.
    pub fn log_ccdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.x_min, &*self.x_max), |(x, a, b)| {
            if x <= a {
                0.0
            } else if x >= b {
                f64::NEG_INFINITY
            } else {
                ((b - x) / (b - a)).ln()
            }
        })
    }

    /// The element-wise derivatives of [`Self::log_cdf`] with respect to x and each bound (zero outside them).
    pub fn d_log_cdf(&self, x: &DoubleTensor) -> Result<UniformGradient> {
        let [x, x_min, x_max] = map_n((x, &*self.x_min, &*self.x_max), |(x, a, b)| {
            if x <= a || x >= b {
                [0.0; 3]
            } else {
                let (from_min, range) = (1.0 / (x - a), 1.0 / (b - a));
                [from_min, range - from_min, -range]
            }
        })?;
        Ok(UniformGradient { x, x_min, x_max })
    }

    /// The element-wise derivatives of [`Self::log_ccdf`] with respect to x and each bound (zero outside them).
    pub fn d_log_ccdf(&self, x: &DoubleTensor) -> Result<UniformGradient> {
        let [x, x_min, x_max] = map_n((x, &*self.x_min, &*self.x_max), |(x, a, b)| {
            if x <= a || x >= b {
                [0.0; 3]
            } else {
                let (to_max, range) = (1.0 / (b - x), 1.0 / (b - a));
                [-to_max, range, to_max - range]
            }
        })?;
        Ok(UniformGradient { x, x_min, x_max })
    }

    /// The element-wise inverse CDF: the x with P(X <= x) = p, for p in [0, 1].
    pub fn inverse_cdf(&self, p: &DoubleTensor) -> Result<DoubleTensor> {
        map((p, &*self.x_min, &*self.x_max), |(p, a, b)| a + p * (b - a))
    }

    /// The element-wise inverse survival function: the x with P(X > x) = q, for q in [0, 1].
    pub fn inverse_ccdf(&self, q: &DoubleTensor) -> Result<DoubleTensor> {
        map((q, &*self.x_min, &*self.x_max), |(q, a, b)| b - q * (b - a))
    }
}

/// The Poisson log mass at k >= 0.
fn poisson_log_pmf(k: f64, mu: f64) -> f64 {
    safe_ln_times(mu, k) - mu - ln_gamma(k + 1.0)
}

impl Poisson<'_> {
    /// ln P(X <= k) = ln Q(k + 1, mu), or ln P(X > k) = ln P(k + 1, mu) if `upper`.
    fn log_cdf_at(k: i64, mu: f64, upper: bool) -> f64 {
        if k < 0 {
            return if upper { 0.0 } else { f64::NEG_INFINITY };
        }
        let a = k as f64 + 1.0;
        if upper {
            log_gamma_p(a, mu)
        } else {
            log_gamma_q(a, mu)
        }
    }

    /// The element-wise log CDF, ln P(X <= k), broadcasting k with the parameters.
    pub fn log_cdf(&self, k: &IntTensor) -> Result<DoubleTensor> {
        map((k, &*self.mu), |(k, mu)| Self::log_cdf_at(k, mu, false))
    }

    /// The element-wise log survival function, ln P(X > k), broadcasting k with the parameters.
    pub fn log_ccdf(&self, k: &IntTensor) -> Result<DoubleTensor> {
        map((k, &*self.mu), |(k, mu)| Self::log_cdf_at(k, mu, true))
    }

    fn cdf_gradient(&self, k: &IntTensor, upper: bool) -> Result<PoissonGradient> {
        Ok(PoissonGradient {
            mu: map((k, &*self.mu), |(k, mu)| {
                if k < 0 {
                    return 0.0;
                }
                // d Q(k + 1, mu) / d mu = -P(X = k)
                let d = ratio(
                    poisson_log_pmf(k as f64, mu),
                    Self::log_cdf_at(k, mu, upper),
                );
                if upper { d } else { -d }
            })?,
        })
    }

    /// The element-wise derivative of [`Self::log_cdf`] with respect to mu.
    pub fn d_log_cdf(&self, k: &IntTensor) -> Result<PoissonGradient> {
        self.cdf_gradient(k, false)
    }

    /// The element-wise derivative of [`Self::log_ccdf`] with respect to mu.
    pub fn d_log_ccdf(&self, k: &IntTensor) -> Result<PoissonGradient> {
        self.cdf_gradient(k, true)
    }
}

impl Binomial<'_> {
    /// ln P(X <= k) = ln I₁₋ₚ(n - k, k + 1), or ln P(X > k) = ln Iₚ(k + 1, n - k) if `upper`.
    fn log_cdf_at(k: i64, p: f64, n: i64, upper: bool) -> f64 {
        let (below, above) = if upper {
            (0.0, f64::NEG_INFINITY)
        } else {
            (f64::NEG_INFINITY, 0.0)
        };
        if k < 0 {
            return below;
        }
        if k >= n {
            return above;
        }
        let (k, n) = (k as f64, n as f64);
        let cdf = || log_inc_beta(n - k, k + 1.0, 1.0 - p);
        let survival = || log_inc_beta(k + 1.0, n - k, p);
        if upper {
            log_complement(survival(), cdf)
        } else {
            log_complement(cdf(), survival)
        }
    }

    /// The element-wise log CDF, ln P(X <= k), broadcasting k with the parameters.
    pub fn log_cdf(&self, k: &IntTensor) -> Result<DoubleTensor> {
        map((k, &*self.p, &*self.n), |(k, p, n)| {
            Self::log_cdf_at(k, p, n, false)
        })
    }

    /// The element-wise log survival function, ln P(X > k), broadcasting k with the parameters.
    pub fn log_ccdf(&self, k: &IntTensor) -> Result<DoubleTensor> {
        map((k, &*self.p, &*self.n), |(k, p, n)| {
            Self::log_cdf_at(k, p, n, true)
        })
    }

    fn cdf_gradient(&self, k: &IntTensor, upper: bool) -> Result<BinomialGradient> {
        Ok(BinomialGradient {
            p: map((k, &*self.p, &*self.n), |(k, p, n)| {
                if k < 0 || k >= n {
                    return 0.0;
                }
                // d Iₚ(k + 1, n - k) / dp is the Beta(k + 1, n - k) density at p
                let (kf, nf) = (k as f64, n as f64);
                let log_density = safe_ln_times(p, kf) + safe_ln_times(1.0 - p, nf - kf - 1.0)
                    - log_beta(kf + 1.0, nf - kf);
                let d = ratio(log_density, Self::log_cdf_at(k, p, n, upper));
                if upper { d } else { -d }
            })?,
        })
    }

    /// The element-wise derivative of [`Self::log_cdf`] with respect to p.
    pub fn d_log_cdf(&self, k: &IntTensor) -> Result<BinomialGradient> {
        self.cdf_gradient(k, false)
    }

    /// The element-wise derivative of [`Self::log_ccdf`] with respect to p.
    pub fn d_log_ccdf(&self, k: &IntTensor) -> Result<BinomialGradient> {
        self.cdf_gradient(k, true)
    }
}

impl Weibull<'_> {
    /// The element-wise log CDF, ln P(X <= x) = ln(1 - e^(-(x/β)^α)), broadcasting x with the parameters
    /// (-infinity for x <= 0).
    pub fn log_cdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.alpha, &*self.beta), |(x, a, b)| {
            if x <= 0.0 {
                f64::NEG_INFINITY
            } else {
                log1mexp(-(x / b).powf(a))
            }
        })
    }

    /// The element-wise log survival function, ln P(X > x) = -(x/β)^α, broadcasting x with the parameters (0 for
    /// x <= 0).
    pub fn log_ccdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.alpha, &*self.beta), |(x, a, b)| {
            if x <= 0.0 { 0.0 } else { -(x / b).powf(a) }
        })
    }

    /// With t = (x/β)^α, the derivatives of -t with respect to x, α and β, scaled by `factor(t)`.
    fn cdf_gradient(&self, x: &DoubleTensor, factor: fn(f64) -> f64) -> Result<WeibullGradient> {
        let [x, alpha, beta] = map_n((x, &*self.alpha, &*self.beta), |(x, a, b)| {
            if x <= 0.0 {
                return [0.0; 3];
            }
            let z = x / b;
            let t = z.powf(a);
            let f = factor(t);
            [-a * t / x * f, -t * z.ln() * f, a * t / b * f]
        })?;
        Ok(WeibullGradient { x, alpha, beta })
    }

    /// The element-wise derivatives of [`Self::log_cdf`] with respect to x and each parameter.
    pub fn d_log_cdf(&self, x: &DoubleTensor) -> Result<WeibullGradient> {
        // d ln(1 - e^-t) = -d(-t) / (e^t - 1)
        self.cdf_gradient(x, |t| -1.0 / t.exp_m1())
    }

    /// The element-wise derivatives of [`Self::log_ccdf`] with respect to x and each parameter.
    pub fn d_log_ccdf(&self, x: &DoubleTensor) -> Result<WeibullGradient> {
        self.cdf_gradient(x, |_| 1.0)
    }

    /// The element-wise inverse CDF: the x with P(X <= x) = p, for p in [0, 1].
    pub fn inverse_cdf(&self, p: &DoubleTensor) -> Result<DoubleTensor> {
        map((p, &*self.alpha, &*self.beta), |(p, a, b)| {
            b * (-(-p).ln_1p()).powf(1.0 / a)
        })
    }

    /// The element-wise inverse survival function: the x with P(X > x) = q, for q in [0, 1].
    pub fn inverse_ccdf(&self, q: &DoubleTensor) -> Result<DoubleTensor> {
        map((q, &*self.alpha, &*self.beta), |(q, a, b)| {
            b * (-q.ln()).powf(1.0 / a)
        })
    }
}

impl NegativeBinomial<'_> {
    /// ln P(X <= k) = ln I_p(α, k + 1) with p = α / (α + μ), or ln P(X > k) = ln I_(1-p)(k + 1, α) if `upper`.
    fn log_cdf_at(k: i64, mu: f64, alpha: f64, upper: bool) -> f64 {
        if k < 0 {
            return if upper { 0.0 } else { f64::NEG_INFINITY };
        }
        let k = k as f64 + 1.0;
        let (p, q) = (alpha / (alpha + mu), mu / (alpha + mu));
        let cdf = || log_inc_beta(alpha, k, p);
        let survival = || log_inc_beta(k, alpha, q);
        if upper {
            log_complement(survival(), cdf)
        } else {
            log_complement(cdf(), survival)
        }
    }

    /// The element-wise log CDF, ln P(X <= k), broadcasting k with the parameters.
    pub fn log_cdf(&self, k: &IntTensor) -> Result<DoubleTensor> {
        map((k, &*self.mu, &*self.alpha), |(k, mu, alpha)| {
            Self::log_cdf_at(k, mu, alpha, false)
        })
    }

    /// The element-wise log survival function, ln P(X > k), broadcasting k with the parameters.
    pub fn log_ccdf(&self, k: &IntTensor) -> Result<DoubleTensor> {
        map((k, &*self.mu, &*self.alpha), |(k, mu, alpha)| {
            Self::log_cdf_at(k, mu, alpha, true)
        })
    }

    fn cdf_gradient(&self, k: &IntTensor, upper: bool) -> Result<NegativeBinomialGradient> {
        let [mu, alpha] = map_n((k, &*self.mu, &*self.alpha), |(k, mu, alpha)| {
            if k < 0 {
                return [0.0; 2];
            }
            let log_f = Self::log_cdf_at(k, mu, alpha, upper);
            if log_f == f64::NEG_INFINITY {
                // unspecified (see the module docs); 0, like `ratio`, rather than the NaN the incomplete beta
                // derivatives give there (the survival function at μ = 0)
                return [0.0; 2];
            }
            let k = k as f64 + 1.0;
            let total = alpha + mu;
            let (p, q) = (alpha / total, mu / total);
            // the Beta(α, k + 1) density at p, which is d I_p(α, k + 1) / dp (with q^0 = 1 at μ = 0, q = 0)
            let log_density =
                safe_ln_times(p, alpha - 1.0) + safe_ln_times(q, k - 1.0) - log_beta(alpha, k);
            let density = ratio(log_density, log_f);
            // dp/dμ = -α / (α + μ)², dp/dα = μ / (α + μ)²
            let (dp_dmu, dp_dalpha) = (-alpha / (total * total), mu / (total * total));
            if upper {
                // S = I_q(k + 1, α), q = 1 - p
                [
                    -density * dp_dmu,
                    log_inc_beta_db(k, alpha, q) - density * dp_dalpha,
                ]
            } else {
                [
                    density * dp_dmu,
                    log_inc_beta_da(alpha, k, p) + density * dp_dalpha,
                ]
            }
        })?;
        Ok(NegativeBinomialGradient { mu, alpha })
    }

    /// The element-wise derivatives of [`Self::log_cdf`] with respect to each parameter.
    pub fn d_log_cdf(&self, k: &IntTensor) -> Result<NegativeBinomialGradient> {
        self.cdf_gradient(k, false)
    }

    /// The element-wise derivatives of [`Self::log_ccdf`] with respect to each parameter.
    pub fn d_log_ccdf(&self, k: &IntTensor) -> Result<NegativeBinomialGradient> {
        self.cdf_gradient(k, true)
    }
}
