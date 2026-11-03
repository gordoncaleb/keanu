//! A model with a user-defined distribution.
//!
//! A hierarchical Beta–Binomial model, whose hyperprior over the Beta's (a, b) is p(a, b) ∝ (a + b)^-2.5. That
//! hyperprior isn't a built-in distribution, so it's implemented with [`CustomDistribution`]: its log probability,
//! and (so NUTS can be used) its gradient. It can't be sampled, so its starting value is set explicitly.
//!
//! In Java this needed a custom probabilistic vertex class. Run with `cargo run --example custom_distribution`.

use keanu::prelude::*;

/// p(a, b) ∝ (a + b)^-2.5 for a, b >= 0, over a vector value [a, b].
#[derive(Debug)]
pub struct HyperPrior;

impl CustomDistribution for HyperPrior {
    fn log_prob(&self, value: &Value, _parameters: &[&Value]) -> Result<f64, String> {
        let v = value.as_f64().ok_or("the value must be f64")?.as_slice();
        let (a, b) = (v[0], v[1]);
        Ok(if a < 0.0 || b < 0.0 {
            f64::NEG_INFINITY
        } else {
            -2.5 * (a + b).ln()
        })
    }

    fn log_prob_gradient(
        &self,
        value: &Value,
        _parameters: &[&Value],
    ) -> Result<Option<CustomGradient>, String> {
        let v = value.as_f64().ok_or("the value must be f64")?.as_slice();
        let (a, b) = (v[0], v[1]);
        let d = if a < 0.0 || b < 0.0 {
            0.0
        } else {
            -2.5 / (a + b)
        };
        Ok(Some(CustomGradient {
            value: Some(DoubleTensor::vector(vec![d, d])),
            parameters: vec![],
        }))
    }
}

/// Number of successes out of 100 trials in each of 10 experiments.
pub const SUCCESSES: [i64; 10] = [40, 44, 47, 54, 63, 46, 44, 49, 58, 50];

/// Posterior means of each experiment's true rate, from Metropolis–Hastings and from NUTS.
#[derive(Debug)]
pub struct Results {
    pub metropolis_hastings: Vec<f64>,
    pub nuts: Vec<f64>,
}

pub fn run(seed: u64, mh_samples: usize, nuts_samples: usize) -> keanu::Result<Results> {
    let b = GraphBuilder::new();
    let hyper_priors: DoubleVar = b.custom(HyperPrior, &[2], &[]);
    b.set_value(hyper_priors, DoubleTensor::vector(vec![0.5, 0.5]));
    let a = hyper_priors.slice("0");
    let b_param = hyper_priors.slice("1");
    let true_rates = b.beta_with_shape(&[10], a, b_param);
    let trials = b.int_constant(IntTensor::vector(vec![100; 10]));
    let observed = b.binomial(true_rates, trials);
    b.observe(observed, IntTensor::vector(SUCCESSES.to_vec()));
    let (hyper_priors, true_rates) = (hyper_priors.id(), true_rates.id());

    let (graph, mut initial) = b.build()?;
    let mut rng = KeanuRng::seed_from_u64(seed);
    // start every true rate at 0.5, inside the support
    initial.set_and_cascade(&graph, true_rates, DoubleTensor::full(&[10], 0.5))?;
    let model = Model::new(graph, &initial)?;

    // Metropolis–Hastings with Gaussian proposals (the hyperprior can't be sampled, so the prior proposal won't do)
    let mut state = initial.clone();
    let mh = MetropolisHastings::default().proposal(GaussianProposal::new(0.05));
    let mh_samples = mh.sample_with(
        &model,
        &mut state,
        &[hyper_priors, true_rates],
        mh_samples,
        SamplingOptions::default()
            .with_drop(mh_samples / 2)
            .with_down_sample_interval(1),
        &mut rng,
    )?;

    // NUTS, which uses the custom distribution's gradient
    let mut state = initial;
    let nuts = Nuts::default().with_max_tree_height(5);
    let (nuts_samples, _statistics) = nuts.sample_with_statistics(
        &model,
        &mut state,
        &[true_rates],
        nuts_samples,
        SamplingOptions::default()
            .with_drop(1000)
            .with_down_sample_interval(1),
        &mut rng,
    )?;

    let means = |samples: &NetworkSamples| -> keanu::Result<Vec<f64>> {
        Ok(samples.f64(true_rates).unwrap().mean()?.into_vec())
    };
    Ok(Results {
        metropolis_hastings: means(&mh_samples)?,
        nuts: means(&nuts_samples)?,
    })
}

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    println!("{:?}", run(1, 100_000, 5000)?);
    Ok(())
}
