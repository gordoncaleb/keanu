use crate::error::{DistributionError, Result, invalid_parameter};
use crate::jet::{self, ElementDerivatives, Jet};
use crate::param::{IntoParam, Param};
use crate::rng::KeanuRng;
use crate::util::{map, map_n, sample_shape};
use keanu_tensor::{DoubleTensor, IntTensor};

/// Discrete Weibull (type I), with P(X ≥ k) = q^(k^β) for k ≥ 0.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteWeibull<'a> {
    /// Survival probability past zero, strictly between zero and one.
    pub q: Param<'a>,
    /// Positive shape.
    pub beta: Param<'a>,
}

/// Derivatives of an elementwise log mass or log tail.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteWeibullGradient {
    /// Derivative with respect to q.
    pub q: DoubleTensor,
    /// Derivative with respect to beta.
    pub beta: DoubleTensor,
}

#[derive(Clone, Copy)]
enum Function {
    Mass,
    Cdf,
    Survival,
}

fn valid(q: f64, beta: f64) -> bool {
    q > 0.0 && q < 1.0 && beta.is_finite() && beta > 0.0
}

/// ln(1 − exp(−exp(log_rate))).
fn log_one_minus_survival<const K: usize>(log_rate: Jet<K>) -> Jet<K> {
    if log_rate.v > 7.0 {
        // exp(−exp(log_rate)) is below the floating-point range.
        Jet::constant(-0.0)
    } else {
        (-log_rate).log_gumbel_ccdf()
    }
}

fn evaluate<const K: usize>(k: f64, [q, beta]: [Jet<K>; 2], function: Function) -> Jet<K> {
    if !valid(q.v, beta.v) {
        return Jet::constant(f64::NAN);
    }
    if k < 0.0 {
        return Jet::constant(match function {
            Function::Mass | Function::Cdf => f64::NEG_INFINITY,
            Function::Survival => 0.0,
        });
    }
    if k == 0.0 {
        return match function {
            Function::Mass | Function::Cdf => (-q).ln_1p(),
            Function::Survival => q.ln(),
        };
    }
    let log_rate = (-q.ln()).ln();
    match function {
        Function::Mass => {
            let log_t = log_rate + beta * k.ln();
            let t = log_t.exp();
            if !t.v.is_finite() {
                return Jet::constant(f64::NEG_INFINITY);
            }
            // (k + 1)^β − k^β = k^β expm1(β log1p(1/k)).
            let step = k.recip().ln_1p();
            let delta = beta * step;
            let log_gap = if delta.v > std::f64::consts::LN_2 {
                log_t + delta + (-delta).log1mexp()
            } else if delta.v < 1e-8 {
                log_t + beta.ln() + step.ln() + delta * 0.5
            } else {
                log_t + delta.exp_m1().ln()
            };
            -t + log_one_minus_survival(log_gap)
        }
        Function::Cdf | Function::Survival => {
            let log_t = log_rate + beta * k.ln_1p();
            match function {
                Function::Cdf => log_one_minus_survival(log_t),
                _ => {
                    let t = log_t.exp();
                    if t.v.is_finite() {
                        -t
                    } else {
                        Jet::constant(f64::NEG_INFINITY)
                    }
                }
            }
        }
    }
}

impl<'a> DiscreteWeibull<'a> {
    /// Creates a distribution with the given parameters.
    pub fn new(q: impl IntoParam<'a, f64>, beta: impl IntoParam<'a, f64>) -> Self {
        Self {
            q: q.into_param(),
            beta: beta.into_param(),
        }
    }

    fn values(&self, k: &IntTensor, function: Function) -> Result<DoubleTensor> {
        map((k, &*self.q, &*self.beta), |(k, q, beta)| {
            jet::value(k as f64, [q, beta], |k, p| evaluate(k.v, p, function))
        })
    }

    fn gradient(&self, k: &IntTensor, function: Function) -> Result<DiscreteWeibullGradient> {
        let [q, beta] = map_n((k, &*self.q, &*self.beta), |(k, q, beta)| {
            let [_, q, beta] = Self::derivatives(k as f64, [q, beta], function).gradient;
            [q, beta]
        })?;
        Ok(DiscreteWeibullGradient { q, beta })
    }

    fn derivatives(k: f64, parameters: [f64; 2], function: Function) -> ElementDerivatives<3> {
        jet::derivatives(k, parameters, |k, p| evaluate(k.v, p, function))
    }

    /// Elementwise log mass. Negative counts are outside the support.
    pub fn log_prob(&self, k: &IntTensor) -> Result<DoubleTensor> {
        self.values(k, Function::Mass)
    }

    /// Elementwise log P(X ≤ k).
    pub fn log_cdf(&self, k: &IntTensor) -> Result<DoubleTensor> {
        self.values(k, Function::Cdf)
    }

    /// Elementwise log P(X > k).
    pub fn log_ccdf(&self, k: &IntTensor) -> Result<DoubleTensor> {
        self.values(k, Function::Survival)
    }

    /// Elementwise derivatives of the log mass.
    pub fn d_log_prob(&self, k: &IntTensor) -> Result<DiscreteWeibullGradient> {
        self.gradient(k, Function::Mass)
    }

    /// Elementwise derivatives of the log CDF.
    pub fn d_log_cdf(&self, k: &IntTensor) -> Result<DiscreteWeibullGradient> {
        self.gradient(k, Function::Cdf)
    }

    /// Elementwise derivatives of the log survival probability.
    pub fn d_log_ccdf(&self, k: &IntTensor) -> Result<DiscreteWeibullGradient> {
        self.gradient(k, Function::Survival)
    }

    /// Log mass and parameter derivatives of a count. The count coordinate has zero derivatives.
    pub fn log_prob_derivatives(k: f64, parameters: [f64; 2]) -> ElementDerivatives<3> {
        Self::derivatives(k, parameters, Function::Mass)
    }

    /// Log CDF (or log survival probability if upper) and parameter derivatives of a count.
    pub fn log_cdf_derivatives(k: f64, parameters: [f64; 2], upper: bool) -> ElementDerivatives<3> {
        Self::derivatives(
            k,
            parameters,
            if upper {
                Function::Survival
            } else {
                Function::Cdf
            },
        )
    }

    /// Samples integer counts by rounding down a continuous Weibull draw.
    /// Returns an error if a draw is outside the range of i64.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        let shape = sample_shape(shape, &[self.q.shape(), self.beta.shape()])?;
        if self.q.as_slice().iter().any(|&q| !valid(q, 1.0))
            || self.beta.as_slice().iter().any(|&b| !valid(0.5, b))
        {
            return invalid_parameter("discrete Weibull needs q in (0, 1) and finite beta > 0");
        }
        let q = self.q.broadcast_to(&shape)?;
        let beta = self.beta.broadcast_to(&shape)?;
        let mut samples = Vec::with_capacity(q.len());
        for (&q, &beta) in q.as_slice().iter().zip(beta.as_slice()) {
            let e = -rng.next_f64_non_zero().ln();
            let value = ((e.ln() - (-q.ln()).ln()) / beta).exp();
            if value >= -(i64::MIN as f64) {
                return Err(DistributionError::InvalidValue(
                    "discrete Weibull draw exceeds the i64 range".into(),
                ));
            }
            samples.push(value as i64);
        }
        Ok(IntTensor::new(&shape, samples)?)
    }
}
