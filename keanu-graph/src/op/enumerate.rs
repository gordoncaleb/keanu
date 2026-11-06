//! The values each element of a discrete distribution can take, for samplers that enumerate them (Gibbs steps).

use super::DistributionKind;
use crate::error::{GraphError, Result};
use crate::value::Value;
use keanu_tensor::{DoubleTensor, IntTensor};
use std::ops::RangeInclusive;

impl DistributionKind {
    /// Whether every element of the distribution's value has finitely many possible values whatever its
    /// parameters, and the elements can be changed one at a time: the Bernoulli, binomial, beta-binomial,
    /// categorical, ordered logistic and probit, hypergeometric and discrete uniform (and their logit forms), and a
    /// truncated, zero-inflated or hurdle form of one of these. A truncated Poisson or negative binomial can have finitely many values too, when
    /// its upper bound is finite, which only [`finite_support`](Self::finite_support) can tell.
    ///
    /// The elements of a multinomial or Dirichlet-multinomial draw must sum to its count, and those of a
    /// Plackett–Luce ranking must be a permutation, so they can't be changed one at a time, and they are not
    /// counted.
    pub fn has_finite_support(&self) -> bool {
        match self {
            DistributionKind::Bernoulli
            | DistributionKind::Binomial
            | DistributionKind::Categorical
            | DistributionKind::OrderedLogistic
            | DistributionKind::OrderedProbit
            | DistributionKind::HyperGeometric
            | DistributionKind::UniformInt
            | DistributionKind::BetaBinomial
            | DistributionKind::BernoulliLogit
            | DistributionKind::BinomialLogit
            | DistributionKind::CategoricalLogit => true,
            DistributionKind::Truncated(base)
            | DistributionKind::ZeroInflated(base)
            | DistributionKind::Hurdle(base) => base.has_finite_support(),
            _ => false,
        }
    }

    /// The values each element of a value of `shape` can take, given the parameters' values, when there are finitely
    /// many: for each element in row-major order, the smallest and largest value (0 and 1, for false and true, for
    /// a boolean value). A zero-inflated or hurdle kind takes its inner kind's values and 0. `None` when an element has infinitely many values (a Poisson, negative binomial or
    /// geometric count, or a truncation of one with no finite upper bound), the value is continuous, or the elements
    /// can't be enumerated one at a time (a multinomial, mixture, hidden Markov model, censored or custom
    /// distribution).
    ///
    /// A range is empty (its start above its end) where the parameters allow no value at all, such as a discrete
    /// uniform with `x_max <= x_min`.
    ///
    /// # Errors
    ///
    /// An error if the parameters don't match the distribution or don't broadcast to `shape`.
    pub fn finite_support(
        &self,
        parameters: &[&Value],
        shape: &[usize],
    ) -> Result<Option<Vec<RangeInclusive<i64>>>> {
        let len: usize = shape.iter().product();
        let fixed = |lo: i64, hi: i64| Some(vec![lo..=hi; len]);
        Ok(match self {
            DistributionKind::Bernoulli | DistributionKind::BernoulliLogit => fixed(0, 1),
            DistributionKind::Categorical | DistributionKind::CategoricalLogit => {
                fixed(0, parameters.len() as i64 - 1)
            }
            DistributionKind::OrderedLogistic | DistributionKind::OrderedProbit => {
                let &[_, cutpoints] = parameters else {
                    return Err(self.wrong_parameters());
                };
                let categories = cutpoints.f64()?.shape().last().copied().unwrap_or(0) + 1;
                fixed(0, categories as i64 - 1)
            }
            DistributionKind::Binomial
            | DistributionKind::BinomialLogit
            | DistributionKind::BetaBinomial => {
                let n = match (self, parameters) {
                    (DistributionKind::BetaBinomial, &[n, _, _]) => n,
                    (DistributionKind::Binomial | DistributionKind::BinomialLogit, &[_, n]) => n,
                    _ => return Err(self.wrong_parameters()),
                };
                Some(
                    broadcast_int(n.i64()?, shape)?
                        .into_iter()
                        .map(|n| 0..=n)
                        .collect(),
                )
            }
            DistributionKind::HyperGeometric => {
                let &[population, successes, draws] = parameters else {
                    return Err(self.wrong_parameters());
                };
                let n = broadcast_int(population.i64()?, shape)?;
                let k = broadcast_int(successes.i64()?, shape)?;
                let d = broadcast_int(draws.i64()?, shape)?;
                Some(
                    n.into_iter()
                        .zip(k)
                        .zip(d)
                        .map(|((n, k), d)| keanu_distributions::HyperGeometric::range(n, k, d))
                        .collect(),
                )
            }
            DistributionKind::UniformInt => {
                let &[x_min, x_max] = parameters else {
                    return Err(self.wrong_parameters());
                };
                let lower = broadcast_int(x_min.i64()?, shape)?;
                let upper = broadcast_int(x_max.i64()?, shape)?;
                Some(
                    lower
                        .into_iter()
                        .zip(upper)
                        .map(|(lo, hi)| lo..=hi - 1)
                        .collect(),
                )
            }
            DistributionKind::Truncated(base) => {
                let Some((base_parameters, bounds)) = parameters.split_last_chunk::<2>() else {
                    return Err(self.wrong_parameters());
                };
                let base_support = match base.as_ref() {
                    DistributionKind::Poisson | DistributionKind::NegativeBinomial => {
                        vec![0..=i64::MAX; len]
                    }
                    other => match other.finite_support(base_parameters, shape)? {
                        Some(support) => support,
                        None => return Ok(None),
                    },
                };
                let lower = broadcast_double(bounds[0].f64()?, shape)?;
                let upper = broadcast_double(bounds[1].f64()?, shape)?;
                let mut support = Vec::with_capacity(len);
                for ((range, lo), hi) in base_support.into_iter().zip(lower).zip(upper) {
                    let hi = clamp_bound(hi.floor()).min(*range.end());
                    if hi == i64::MAX {
                        return Ok(None);
                    }
                    support.push(clamp_bound(lo.ceil()).max(*range.start())..=hi);
                }
                Some(support)
            }
            DistributionKind::ZeroInflated(inner) | DistributionKind::Hurdle(inner) => {
                let Some((_, inner_parameters)) = parameters.split_last() else {
                    return Err(self.wrong_parameters());
                };
                inner
                    .finite_support(inner_parameters, shape)?
                    .map(|ranges| {
                        ranges
                            .into_iter()
                            .map(|r| (*r.start()).min(0)..=(*r.end()).max(0))
                            .collect()
                    })
            }
            _ => None,
        })
    }
}

impl DistributionKind {
    /// Of a node of this kind with `count` parameters, which parameters [`finite_support`](Self::finite_support)
    /// reads the values of: the count of a binomial (or its logit form) and a beta-binomial, the bounds of a
    /// discrete uniform and of a truncation, every parameter of a hypergeometric, and for a truncated,
    /// zero-inflated or hurdle kind the inner kind's own (so nested kinds are followed: in a zero-inflated truncated
    /// Poisson, whose parameters are the rate, the bounds and ψ, the bounds). The values set by the other
    /// parameters, or by the number of parameters (a categorical's categories, an ordered logistic's or probit's
    /// cutpoints' shape), can't change. Everything false for a kind without finitely many values.
    pub(crate) fn support_parameters(&self, count: usize) -> Vec<bool> {
        let mut reads = vec![false; count];
        match self {
            DistributionKind::Binomial | DistributionKind::BinomialLogit if count == 2 => {
                reads[1] = true;
            }
            DistributionKind::BetaBinomial if count == 3 => reads[0] = true,
            DistributionKind::UniformInt | DistributionKind::HyperGeometric => reads.fill(true),
            DistributionKind::Truncated(base) if count >= 2 => {
                reads[..count - 2].copy_from_slice(&base.support_parameters(count - 2));
                reads[count - 2..].fill(true);
            }
            DistributionKind::ZeroInflated(inner) | DistributionKind::Hurdle(inner)
                if count >= 1 =>
            {
                reads[..count - 1].copy_from_slice(&inner.support_parameters(count - 1));
            }
            _ => {}
        }
        reads
    }
}

/// A real bound as an integer, saturating (NaN as the largest, so a NaN bound gives no finite support).
fn clamp_bound(x: f64) -> i64 {
    if x.is_nan() {
        i64::MAX
    } else {
        // `as` saturates at the integer limits
        x as i64
    }
}

fn broadcast_int(tensor: &IntTensor, shape: &[usize]) -> Result<Vec<i64>> {
    Ok(tensor
        .broadcast_to(shape)
        .map_err(GraphError::from)?
        .into_vec())
}

fn broadcast_double(tensor: &DoubleTensor, shape: &[usize]) -> Result<Vec<f64>> {
    Ok(tensor
        .broadcast_to(shape)
        .map_err(GraphError::from)?
        .into_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn int(values: &[i64]) -> Value {
        Value::I64(IntTensor::vector(values.to_vec()))
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)] // one range per element, and these have one element
    fn finite_supports_are_enumerated_per_element() {
        let p = Value::F64(DoubleTensor::scalar(0.3));
        let bernoulli = DistributionKind::Bernoulli.finite_support(&[&p], &[2]);
        assert_eq!(bernoulli.unwrap(), Some(vec![0..=1, 0..=1]));
        let n = int(&[2, 5]);
        let binomial = DistributionKind::Binomial.finite_support(&[&p, &n], &[2]);
        assert_eq!(binomial.unwrap(), Some(vec![0..=2, 0..=5]));
        let (lo, hi) = (Value::I64(IntTensor::scalar(-1)), int(&[1, 3]));
        let uniform = DistributionKind::UniformInt.finite_support(&[&lo, &hi], &[2]);
        assert_eq!(uniform.unwrap(), Some(vec![-1..=0, -1..=2]));
        let categories = DistributionKind::Categorical.finite_support(&[&p, &p, &p], &[]);
        assert_eq!(categories.unwrap(), Some(vec![0..=2]));
        let cutpoints = Value::F64(DoubleTensor::vector(vec![-1.0, 0.0, 1.0]));
        let ordered = DistributionKind::OrderedLogistic.finite_support(&[&p, &cutpoints], &[]);
        assert_eq!(ordered.unwrap(), Some(vec![0..=3]));

        // a truncated count with a finite upper bound has finitely many values; without one it has not
        let poisson = DistributionKind::Truncated(Arc::new(DistributionKind::Poisson));
        let (lower, upper) = (
            Value::F64(DoubleTensor::scalar(0.5)),
            Value::F64(DoubleTensor::scalar(4.0)),
        );
        let truncated = poisson.finite_support(&[&p, &lower, &upper], &[]);
        assert_eq!(truncated.unwrap(), Some(vec![1..=4]));
        let infinite = Value::F64(DoubleTensor::scalar(f64::INFINITY));
        assert_eq!(
            poisson
                .finite_support(&[&p, &lower, &infinite], &[])
                .unwrap(),
            None
        );
        let binomial = DistributionKind::Truncated(Arc::new(DistributionKind::Binomial));
        let truncated = binomial.finite_support(&[&p, &n, &lower, &infinite], &[2]);
        assert_eq!(truncated.unwrap(), Some(vec![1..=2, 1..=5]));
        assert!(binomial.has_finite_support() && !poisson.has_finite_support());

        for kind in [
            DistributionKind::Poisson,
            DistributionKind::Geometric,
            DistributionKind::Gaussian,
        ] {
            assert!(!kind.has_finite_support());
            assert_eq!(kind.finite_support(&[&p, &p], &[]).unwrap(), None);
        }
    }
}
