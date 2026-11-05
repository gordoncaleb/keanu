//! How much do coaching programmes at eight schools differ in their effect on test scores?
//!
//! The SAT coaching study of Rubin (1981), as analysed in Gelman et al., *Bayesian Data Analysis* (3rd ed., §5.5):
//! each school j reports an estimated effect y_j with standard error sigma_j, and the true effects theta_j share a
//! population mean mu and spread tau:
//!
//! ```text
//! mu ~ N(0, 5)    tau ~ HalfCauchy(5)    theta_j ~ N(mu, tau)    y_j ~ N(theta_j, sigma_j)
//! ```
//!
//! Written this way (the centred form) the posterior is a funnel: when tau is small the theta_j are squeezed
//! together, and NUTS cannot pick one step size that works in both the neck and the mouth, so its trajectories
//! diverge. The non-centred form samples standardised offsets eta_j ~ N(0, 1) and sets theta_j = mu + tau·eta_j,
//! which has the same posterior without the funnel. Both run four NUTS chains; the example reports each form's
//! divergences, the non-centred form's R-hat and effective sample size summary, and each form's PSIS-LOO estimate
//! of the expected log predictive density with the Pareto k̂ of each school. The two forms are the same model, so
//! their LOO estimates agree (ArviZ reports about −30.7 for both).
//!
//! Run with `cargo run --release --example eight_schools`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::loo::compare;
use keanu::algorithms::mcmc::chains::run_chains;
use keanu::prelude::*;

/// Each school's estimated coaching effect (points on the SAT-V).
pub const EFFECTS: [f64; 8] = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0];
/// The standard error of each school's estimate.
pub const STANDARD_ERRORS: [f64; 8] = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0];

/// The number of chains each form runs.
pub const CHAINS: usize = 4;

/// The result of sampling one form of the model.
#[derive(Debug)]
pub struct Fit {
    /// The draws of every chain.
    pub samples: ChainSamples,
    /// The number of divergent transitions after adaptation, over all chains.
    pub divergences: usize,
    /// The population mean.
    pub mu: NodeId,
    /// The population spread.
    pub tau: NodeId,
    /// The latent per-school variable: theta in the centred form, eta in the non-centred form.
    pub schools: NodeId,
    /// The PSIS-LOO estimate of the expected log predictive density, with a point (and a Pareto k̂) per school.
    pub loo: Loo,
}

/// Both forms of the model, sampled with the same settings.
#[derive(Debug)]
pub struct EightSchools {
    /// The centred form, which diverges.
    pub centred: Fit,
    /// The non-centred form, which does not.
    pub non_centred: Fit,
}

/// Samples one form of the model with `CHAINS` chains of `draws` draws each, after as many adaptation steps.
pub fn fit(centred: bool, seed: u64, draws: usize) -> keanu::Result<Fit> {
    let schools = EFFECTS.len();
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 5.0);
    let tau = b.half_cauchy(5.0);
    b.set_value(mu, DoubleTensor::scalar(0.0));
    b.set_value(tau, DoubleTensor::scalar(1.0));
    let (theta, latent) = if centred {
        let theta = b.gaussian_with_shape(&[schools], mu, tau);
        b.set_value(theta, DoubleTensor::vector(vec![0.0; schools]));
        (theta, theta.id())
    } else {
        let eta = b.gaussian_with_shape(&[schools], 0.0, 1.0);
        b.set_value(eta, DoubleTensor::vector(vec![0.0; schools]));
        (mu + tau * eta, eta.id())
    };
    let sigma = b.constant(DoubleTensor::vector(STANDARD_ERRORS.to_vec()));
    let y = b.gaussian_with_shape(&[schools], theta, sigma).label("y");
    b.observe(y, DoubleTensor::vector(EFFECTS.to_vec()));
    let (mu, tau) = (mu.id(), tau.id());

    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let latents = [mu, tau, latent];
    let adapt = draws;
    // A high target acceptance makes NUTS take smaller steps, as is usual for hierarchical models. At 0.9 the
    // non-centred form still diverges about once in 4000 draws for some seeds; at 0.95 it did not over seeds 1 to 12,
    // while the centred form diverged for every one of them.
    let nuts = Nuts::default()
        .with_adapt_count(adapt)
        .with_target_acceptance_prob(0.95);
    let chains = run_chains(CHAINS, seed, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &latents,
            adapt + draws,
            SamplingOptions::default().with_drop(adapt),
            rng,
        )
    })?;
    let divergences = chains.iter().map(|(_, s)| s.divergence_count()).sum();
    let samples = ChainSamples::new(chains.into_iter().map(|(samples, _)| samples).collect())?;
    let loo = loo(&PointwiseLogLik::from_chains(&model, &state, &samples)?)?;
    Ok(Fit {
        samples,
        divergences,
        mu,
        tau,
        schools: latent,
        loo,
    })
}

/// Samples both forms with `draws` draws per chain.
pub fn run(seed: u64, draws: usize) -> keanu::Result<EightSchools> {
    Ok(EightSchools {
        centred: fit(true, seed, draws)?,
        non_centred: fit(false, seed, draws)?,
    })
}

/// The number of draws per chain (after as many adaptation steps).
pub const DEFAULT_SAMPLES: usize = 1000;

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let result = run(1, DEFAULT_SAMPLES)?;
    for (name, fit) in [
        ("centred", &result.centred),
        ("non-centred", &result.non_centred),
    ] {
        let mu = fit.samples.summarize(fit.mu, &[])?;
        let tau = fit.samples.summarize(fit.tau, &[])?;
        println!(
            "{name}: {} divergences; mu {:.2} (R-hat {:.3}), tau {:.2} (R-hat {:.3})",
            fit.divergences, mu.mean, mu.r_hat, tau.mean, tau.r_hat
        );
    }
    let fit = &result.non_centred;
    println!(
        "\nnon-centred summary (mu = {:?}, tau = {:?}, eta = {:?}):",
        fit.mu, fit.tau, fit.schools
    );
    println!("{}", fit.samples.summary()?);
    for (name, fit) in [
        ("centred", &result.centred),
        ("non-centred", &result.non_centred),
    ] {
        let k: Vec<String> = fit
            .loo
            .pareto_k()
            .iter()
            .map(|k| format!("{k:.2}"))
            .collect();
        println!("\n{name}: {}\nPareto k̂ by school: {}", fit.loo, k.join(" "));
    }
    let table = compare(&[
        ("centred", &result.centred.loo),
        ("non-centred", &result.non_centred.loo),
    ])?;
    println!("\n{table}");
    Ok(())
}
