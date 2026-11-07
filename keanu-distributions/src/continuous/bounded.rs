//! Distributions on a bounded interval.

use super::*;

/// Beta distribution with shape parameters `alpha` and `beta`, on the interval [x_min, x_max].
#[derive(Debug, Clone, PartialEq)]
pub struct Beta<'a> {
    /// The alpha (first shape) parameter.
    pub alpha: Param<'a>,
    /// The beta parameter.
    pub beta: Param<'a>,
    /// The lower bound.
    pub x_min: Param<'a>,
    /// The upper bound.
    pub x_max: Param<'a>,
}

/// Derivatives of [`Beta::log_prob`] (with respect to x and the shape parameters).
#[derive(Debug, Clone, PartialEq)]
pub struct BetaGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `alpha`.
    pub alpha: DoubleTensor,
    /// With respect to `beta`.
    pub beta: DoubleTensor,
}

impl<'a> Beta<'a> {
    /// Beta on [x_min, x_max].
    pub fn new(
        alpha: impl IntoParam<'a, f64>,
        beta: impl IntoParam<'a, f64>,
        x_min: impl IntoParam<'a, f64>,
        x_max: impl IntoParam<'a, f64>,
    ) -> Self {
        Self {
            alpha: alpha.into_param(),
            beta: beta.into_param(),
            x_min: x_min.into_param(),
            x_max: x_max.into_param(),
        }
    }

    /// The standard Beta on [0, 1].
    pub fn standard(alpha: impl IntoParam<'a, f64>, beta: impl IntoParam<'a, f64>) -> Self {
        Self::new(alpha, beta, scalar(0.0), scalar(1.0))
    }

    /// With y = (x - x_min) / (x_max - x_min):
    /// (alpha - 1) ln y + (beta - 1) ln(1 - y) - ln B(alpha, beta) - ln(x_max - x_min), -infinity outside the bounds
    /// (everywhere if x_max <= x_min).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map(
            (x, &*self.alpha, &*self.beta, &*self.x_min, &*self.x_max),
            |(x, a, b, min, max)| {
                // `contains` is false for NaN bounds too
                if max <= min || !(min..=max).contains(&x) {
                    return f64::NEG_INFINITY;
                }
                let range = max - min;
                let y = (x - min) / range;
                let beta_function = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
                safe_ln_times(y, a - 1.0) + safe_ln_times(1.0 - y, b - 1.0)
                    - beta_function
                    - range.ln()
            },
        )
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<BetaGradient> {
        let [x, alpha, beta] = map_n(
            (x, &*self.alpha, &*self.beta, &*self.x_min, &*self.x_max),
            |(x, a, b, min, max)| {
                let range = max - min;
                let y = (x - min) / range;
                [
                    ((a - 1.0) / y - (b - 1.0) / (1.0 - y)) / range,
                    y.ln() + (digamma(a + b) - digamma(a)),
                    (1.0 - y).ln() + (digamma(a + b) - digamma(b)),
                ]
            },
        )?;
        Ok(BetaGradient { x, alpha, beta })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless alpha and beta
    /// are positive and x_max > x_min.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        require_positive("alpha", &self.alpha)?;
        require_positive("beta", &self.beta)?;
        require_ordered(&self.x_min, &self.x_max, "x_max must be greater than x_min")?;
        let shape = sample_shape(
            shape,
            &[
                self.alpha.shape(),
                self.beta.shape(),
                self.x_min.shape(),
                self.x_max.shape(),
            ],
        )?;
        sample_each(
            &shape,
            &[&self.alpha, &self.beta, &self.x_min, &self.x_max],
            |p| {
                let (alpha, beta, min, max) = (p[0], p[1], p[2], p[3]);
                // the share of the range from the bound the draw is likelier to be far from, from the Gamma draws'
                // logs (which don't underflow for small shapes), so values near either bound keep their precision
                let x = if alpha < beta {
                    max - sample_beta_ratio(beta, alpha, rng)? * (max - min)
                } else {
                    min + sample_beta_ratio(alpha, beta, rng)? * (max - min)
                };
                // the support is open: a draw that rounded onto a bound is moved just inside it
                Ok(strictly_inside(x, min, max))
            },
        )
    }
}

/// Triangular distribution on [x_min, x_max] with mode `c`.
#[derive(Debug, Clone, PartialEq)]
pub struct Triangular<'a> {
    /// The lower bound.
    pub x_min: Param<'a>,
    /// The upper bound.
    pub x_max: Param<'a>,
    /// The mode.
    pub c: Param<'a>,
}

/// Derivatives of [`Triangular::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct TriangularGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `x_min`.
    pub x_min: DoubleTensor,
    /// With respect to `x_max`.
    pub x_max: DoubleTensor,
    /// With respect to `c`.
    pub c: DoubleTensor,
}

impl<'a> Triangular<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(
        x_min: impl IntoParam<'a, f64>,
        x_max: impl IntoParam<'a, f64>,
        c: impl IntoParam<'a, f64>,
    ) -> Self {
        Self {
            x_min: x_min.into_param(),
            x_max: x_max.into_param(),
            c: c.into_param(),
        }
    }

    /// 2(x - a)/((b - a)(c - a)) below the mode, 2/(b - a) at it, 2(b - x)/((b - a)(b - c)) above it; -infinity
    /// everywhere unless a < b and a <= c <= b.
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.x_min, &*self.x_max, &*self.c), |(x, a, b, c)| {
            // written so that NaN parameters are rejected too
            if !(a < b && a <= c && c <= b) {
                return f64::NEG_INFINITY;
            }
            let range = b - a;
            let pdf = if x >= a && x < c {
                2.0 * (x - a) / (range * (c - a))
            } else if x == c {
                2.0 / range
            } else if x > c && x <= b {
                2.0 * (b - x) / (range * (b - c))
            } else {
                0.0
            };
            pdf.ln()
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<TriangularGradient> {
        // (dx, da, db, dc) for each piece of the pdf, zero outside the support
        let [x, x_min, x_max, c] =
            map_n((x, &*self.x_min, &*self.x_max, &*self.c), |(x, a, b, c)| {
                let inv_range = 1.0 / (b - a);
                if x >= a && x < c {
                    let inv_x_minus_a = if x - a > 0.0 { 1.0 / (x - a) } else { 1.0 };
                    let inv_c_minus_a = 1.0 / (c - a);
                    [
                        inv_x_minus_a,
                        -inv_x_minus_a + inv_range + inv_c_minus_a,
                        -inv_range,
                        -inv_c_minus_a,
                    ]
                } else if x == c {
                    [0.0, inv_range, -inv_range, 0.0]
                } else if x > c && x <= b {
                    let inv_b_minus_x = if b - x > 0.0 { 1.0 / (b - x) } else { 1.0 };
                    let inv_b_minus_c = 1.0 / (b - c);
                    [
                        -inv_b_minus_x,
                        inv_range,
                        inv_b_minus_x - inv_range - inv_b_minus_c,
                        inv_b_minus_c,
                    ]
                } else {
                    [0.0; 4]
                }
            })?;
        Ok(TriangularGradient { x, x_min, x_max, c })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless x_min < x_max
    /// and x_min <= c <= x_max.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        require_ordered(&self.x_min, &self.x_max, "x_max must be greater than x_min")?;
        let valid = self.c.ge(&self.x_min)?.all() && self.c.le(&self.x_max)?.all();
        if !valid {
            return invalid_parameter("the mode c must be between x_min and x_max");
        }
        let shape = sample_shape(
            shape,
            &[self.x_min.shape(), self.x_max.shape(), self.c.shape()],
        )?;
        let p = rng.uniform(&shape);
        map(
            (&p, &*self.x_min, &*self.x_max, &*self.c),
            |(p, a, b, c)| {
                let range = b - a;
                if p <= (c - a) / range {
                    a + (range * (c - a) * p).sqrt()
                } else {
                    b - (range * (b - c) * (1.0 - p)).sqrt()
                }
            },
        )
    }
}

/// Uniform distribution on [x_min, x_max).
#[derive(Debug, Clone, PartialEq)]
pub struct Uniform<'a> {
    /// The lower bound.
    pub x_min: Param<'a>,
    /// The upper bound.
    pub x_max: Param<'a>,
}

/// Derivatives of [`Uniform::log_cdf`] and [`Uniform::log_ccdf`] (the log density has no useful derivative: it
/// is constant inside the support).
#[derive(Debug, Clone, PartialEq)]
pub struct UniformGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `x_min`.
    pub x_min: DoubleTensor,
    /// With respect to `x_max`.
    pub x_max: DoubleTensor,
}

impl<'a> Uniform<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(x_min: impl IntoParam<'a, f64>, x_max: impl IntoParam<'a, f64>) -> Self {
        Self {
            x_min: x_min.into_param(),
            x_max: x_max.into_param(),
        }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support,
    /// so everywhere if x_max <= x_min).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.x_min, &*self.x_max), |(x, min, max)| {
            // `contains` is false for NaN bounds too
            if max <= min || !(min..max).contains(&x) {
                f64::NEG_INFINITY
            } else {
                -(max - min).ln()
            }
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless
    /// x_max > x_min.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        require_ordered(&self.x_min, &self.x_max, "x_max must be greater than x_min")?;
        let shape = sample_shape(shape, &[self.x_min.shape(), self.x_max.shape()])?;
        Ok(rng
            .uniform(&shape)
            .mul(&self.x_max.sub(&self.x_min)?)?
            .add(&self.x_min)?)
    }
}
