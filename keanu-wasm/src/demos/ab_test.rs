//! Should we ship variant B? A Bayesian A/B test on conversion counts.
//!
//! Each variant's conversion rate has a uniform Beta(1, 1) prior and its conversions are binomial in its visitors.
//! The decision quantities come from the joint posterior: the probability that B converts better, the relative
//! lift, and the expected loss of shipping each variant (the conversion rate given up if it is in fact worse).

use crate::run::{ChainRun, Histogram, Interval, mean, scalar_draws};
use keanu::prelude::*;
use serde::Serialize;

/// Observed traffic for one variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arm {
    /// Visitors who saw the variant.
    pub visitors: i64,
    /// Visitors who converted.
    pub conversions: i64,
}

/// A fit in progress.
#[derive(Debug)]
pub struct AbTest {
    chain: ChainRun,
    rate_a: NodeId,
    rate_b: NodeId,
}

/// What the decision maker sees.
#[derive(Debug, Clone, Serialize)]
pub struct AbReport {
    /// Variant A's conversion rate.
    pub rate_a: Interval,
    /// Variant B's conversion rate.
    pub rate_b: Interval,
    /// The probability that B's rate is higher than A's.
    pub probability_b_better: f64,
    /// B's rate relative to A's, minus one.
    pub lift: Interval,
    /// The distribution of the lift.
    pub lift_histogram: Histogram,
    /// The expected conversion rate lost by shipping A (if B is better).
    pub expected_loss_a: f64,
    /// The expected conversion rate lost by shipping B (if A is better).
    pub expected_loss_b: f64,
    /// The number of posterior draws.
    pub draws: usize,
}

impl AbTest {
    /// Starts fitting the two arms with `draws` NUTS draws after as many adaptation steps.
    pub fn new(a: Arm, b: Arm, draws: usize, seed: u64) -> keanu::Result<Self> {
        for arm in [a, b] {
            if arm.visitors <= 0 || arm.conversions < 0 || arm.conversions > arm.visitors {
                return Err(keanu::Error::from(keanu::graph::GraphError::InvalidValue(
                    format!(
                        "each variant needs visitors > 0 and 0 ≤ conversions ≤ visitors, got {} of {}",
                        arm.conversions, arm.visitors
                    ),
                )));
            }
        }
        let builder = GraphBuilder::new();
        let rate = |arm: Arm| {
            let rate = builder.beta(1.0, 1.0);
            builder.set_value(rate, DoubleTensor::scalar(0.5));
            let conversions = builder.binomial(rate, arm.visitors);
            builder.observe(conversions, IntTensor::scalar(arm.conversions));
            rate.id()
        };
        let (rate_a, rate_b) = (rate(a), rate(b));
        let (graph, state) = builder.build()?;
        let model = Model::new(graph, &state)?;
        let chain = ChainRun::new(
            model,
            state,
            &[rate_a, rate_b],
            Nuts::default(),
            draws,
            draws,
            seed,
        )?;
        Ok(Self {
            chain,
            rate_a,
            rate_b,
        })
    }

    /// The chain.
    pub fn chain(&mut self) -> &mut ChainRun {
        &mut self.chain
    }

    /// The decision report from the draws so far.
    pub fn report(&self) -> AbReport {
        let samples = self.chain.samples();
        let a = scalar_draws(samples, self.rate_a);
        let b = scalar_draws(samples, self.rate_b);
        let lift: Vec<f64> = a.iter().zip(&b).map(|(a, b)| b / a - 1.0).collect();
        let better = a.iter().zip(&b).filter(|(a, b)| b > a).count();
        AbReport {
            rate_a: Interval::of(&a),
            rate_b: Interval::of(&b),
            probability_b_better: better as f64 / a.len().max(1) as f64,
            lift: Interval::of(&lift),
            lift_histogram: Histogram::of(&lift, 40),
            expected_loss_a: mean(
                &a.iter()
                    .zip(&b)
                    .map(|(a, b)| (b - a).max(0.0))
                    .collect::<Vec<_>>(),
            ),
            expected_loss_b: mean(
                &a.iter()
                    .zip(&b)
                    .map(|(a, b)| (a - b).max(0.0))
                    .collect::<Vec<_>>(),
            ),
            draws: a.len(),
        }
    }
}
