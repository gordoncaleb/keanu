//! A model with user-defined distributions.
//!
//! A hierarchical Beta–Binomial model (like the rat tumour model in Gelman et al., *Bayesian Data Analysis*, ch. 5):
//! each experiment's true rate θⱼ ~ Beta(a, b), with the hyperprior p(a, b) ∝ (a + b)^-5/2. Neither the hyperprior
//! nor the reparameterised rates below are built-in distributions, so they're implemented with
//! [`CustomDistribution`]: their log probabilities, and (so NUTS can be used) their gradients. They can't be
//! sampled, so their starting values are set explicitly.
//!
//! Written directly in (a, b) and θ, the posterior is hard to sample: it has a long tail towards large a + b, where
//! the rates are squeezed against their mean (a funnel). So, as in BDA, it's sampled in unconstrained coordinates
//! with the same posterior:
//!
//! - the hyperparameters as φ = (logit μ, ln(a + b)), where μ = a / (a + b). Their density is p(a, b) times the
//!   Jacobian |∂(a, b) / ∂φ| = a b, that is p(φ) ∝ a b (a + b)^-5/2 = μ (1 - μ) (a + b)^-1/2.
//! - each rate as its standardised offset from the mean on the logit scale, zⱼ = (logit θⱼ - logit μ) / k, where
//!   k = √(1/a + 1/b) is about the standard deviation of logit θⱼ. Its density is Beta(θⱼ; a, b) times
//!   |dθⱼ / dzⱼ| = θⱼ (1 - θⱼ) k, roughly a standard Gaussian whatever a and b are, so there's no funnel.
//!
//! (The hyperprior is truncated at a very large a + b, see [`MAX_LN_SIZE`], which changes nothing measurable.)
//!
//! In Java this needed custom probabilistic vertex classes. Run with `cargo run --example custom_distribution`.

use keanu::prelude::*;
use keanu::tensor::special::{digamma, ln_gamma};

/// The hyperprior p(a, b) ∝ (a + b)^-5/2, as a density over φ = [logit(a / (a + b)), ln(a + b)]:
/// ln p(φ) = ln σ(φ₀) + ln σ(-φ₀) - φ₁ / 2 (plus a constant), for φ₁ up to [`MAX_LN_SIZE`].
#[derive(Debug)]
pub struct HyperPrior;

/// The hyperprior is truncated at a + b = e^20 (about 5 × 10⁸), beyond which ln B(a, b) (in [`RateOffsets`]) is
/// computed as a difference of numbers so large that f64 can't hold it precisely, and the log probability becomes
/// noise a sampler can get lost in. The truncated tail has posterior probability below 10⁻⁴, and changes the rates'
/// posterior means by less than 10⁻⁵.
pub const MAX_LN_SIZE: f64 = 20.0;

impl CustomDistribution for HyperPrior {
    fn log_prob(&self, value: &Value, _parameters: &[&Value]) -> Result<f64, String> {
        let phi = value.as_f64().ok_or("the value must be f64")?.as_slice();
        if phi[1] > MAX_LN_SIZE {
            return Ok(f64::NEG_INFINITY);
        }
        Ok(ln_sigmoid(phi[0]) + ln_sigmoid(-phi[0]) - 0.5 * phi[1])
    }

    fn log_prob_gradient(
        &self,
        value: &Value,
        _parameters: &[&Value],
    ) -> Result<Option<CustomGradient>, String> {
        let phi = value.as_f64().ok_or("the value must be f64")?.as_slice();
        // d/dx ln σ(x) = σ(-x)
        Ok(Some(CustomGradient {
            value: Some(DoubleTensor::vector(vec![
                sigmoid(-phi[0]) - sigmoid(phi[0]),
                -0.5,
            ])),
            parameters: vec![],
        }))
    }
}

/// The rates θⱼ ~ Beta(a, b), as standardised offsets zⱼ = (logit θⱼ - logit μ) / k, with k = √(1/a + 1/b) and
/// one parameter, φ (see [`HyperPrior`]). With xⱼ = logit θⱼ = φ₀ + k zⱼ:
/// ln p(zⱼ | φ) = a ln σ(xⱼ) + b ln σ(-xⱼ) - ln B(a, b) + ln k.
#[derive(Debug)]
pub struct RateOffsets;

/// a, b and k = √(1/a + 1/b) from φ.
fn hyperparameters(phi: &[f64]) -> (f64, f64, f64) {
    let (mu, size) = (sigmoid(phi[0]), phi[1].exp());
    let (a, b) = (mu * size, (1.0 - mu) * size);
    (a, b, (1.0 / a + 1.0 / b).sqrt())
}

impl CustomDistribution for RateOffsets {
    fn log_prob(&self, value: &Value, parameters: &[&Value]) -> Result<f64, String> {
        let z = value.as_f64().ok_or("the value must be f64")?.as_slice();
        let phi = parameters[0].as_f64().ok_or("φ must be f64")?.as_slice();
        let (a, b, k) = hyperparameters(phi);
        let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
        Ok(z.iter()
            .map(|&z| {
                let x = phi[0] + k * z;
                a * ln_sigmoid(x) + b * ln_sigmoid(-x) - ln_beta + k.ln()
            })
            .sum())
    }

    fn log_prob_gradient(
        &self,
        value: &Value,
        parameters: &[&Value],
    ) -> Result<Option<CustomGradient>, String> {
        let z = value.as_f64().ok_or("the value must be f64")?.as_slice();
        let phi = parameters[0].as_f64().ok_or("φ must be f64")?.as_slice();
        let (a, b, k) = hyperparameters(phi);
        // φ₀ moves a and b in opposite directions (keeping a + b), by ±a b / (a + b); φ₁ scales them both
        let da_dphi0 = a * b / (a + b);
        // d ln k / dφ₀ and d ln k / dφ₁
        let (dlnk_dphi0, dlnk_dphi1) = (-0.5 * (b - a) / (a + b), -0.5);
        // d ln B(a, b) / da and d ln B(a, b) / db
        let dlnb_da = digamma(a) - digamma(a + b);
        let dlnb_db = digamma(b) - digamma(a + b);

        let mut dz = Vec::with_capacity(z.len());
        let (mut dphi0, mut dphi1) = (0.0, 0.0);
        for &z in z {
            let x = phi[0] + k * z;
            // d/dx (a ln σ(x) + b ln σ(-x))
            let dx = a * sigmoid(-x) - b * sigmoid(x);
            dz.push(dx * k);
            // through a and b, through x = φ₀ + k z, and through ln k
            dphi0 += da_dphi0 * (ln_sigmoid(x) - dlnb_da - ln_sigmoid(-x) + dlnb_db)
                + dx * (1.0 + k * z * dlnk_dphi0)
                + dlnk_dphi0;
            dphi1 += a * (ln_sigmoid(x) - dlnb_da)
                + b * (ln_sigmoid(-x) - dlnb_db)
                + dx * k * z * dlnk_dphi1
                + dlnk_dphi1;
        }
        Ok(Some(CustomGradient {
            value: Some(DoubleTensor::vector(dz)),
            parameters: vec![Some(DoubleTensor::vector(vec![dphi0, dphi1]))],
        }))
    }
}

/// The logistic sigmoid σ(x) = 1 / (1 + e⁻ˣ).
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// ln σ(x) = -ln(1 + e⁻ˣ), without overflow for large negative x.
fn ln_sigmoid(x: f64) -> f64 {
    x.min(0.0) - (-x.abs()).exp().ln_1p()
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
    let phi: DoubleVar = b.custom(HyperPrior, &[2], &[]);
    let offsets: DoubleVar = b.custom(RateOffsets, &[10], &[phi.id()]);
    // start at μ = 0.5 and a + b = e, with every rate at the mean
    b.set_value(phi, DoubleTensor::vector(vec![0.0, 1.0]));
    b.set_value(offsets, DoubleTensor::zeros(&[10]));
    // the rates, θⱼ = σ(φ₀ + k zⱼ), with k = √(1/a + 1/b) = 1 / √(μ (1 - μ) (a + b))
    let (logit_mean, ln_size) = (phi.slice("0"), phi.slice("1"));
    let k = (logit_mean.sigmoid() * (-logit_mean).sigmoid() * ln_size.exp())
        .sqrt()
        .reciprocal();
    let true_rates = (logit_mean + k * offsets).sigmoid();
    let trials = b.int_constant(IntTensor::vector(vec![100; 10]));
    let observed = b.binomial(true_rates, trials);
    b.observe(observed, IntTensor::vector(SUCCESSES.to_vec()));
    let true_rates = true_rates.id();

    let (graph, initial) = b.build()?;
    let mut rng = KeanuRng::seed_from_u64(seed);
    let model = Model::new(graph, &initial)?;

    // Metropolis–Hastings with Gaussian proposals (the custom distributions can't be sampled, so the prior proposal
    // won't do), sized for the unconstrained coordinates, which vary on a scale of about 1
    let mut state = initial.clone();
    let mh = MetropolisHastings::default().proposal(GaussianProposal::new(0.5));
    let mh_samples = mh.sample_with(
        &model,
        &mut state,
        &[true_rates],
        mh_samples,
        SamplingOptions::default()
            .with_drop(mh_samples / 2)
            .with_down_sample_interval(1),
        &mut rng,
    )?;

    // NUTS, which uses the custom distributions' gradients
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
