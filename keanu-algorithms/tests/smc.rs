//! Tempered SMC against closed-form evidence and exact posteriors: a conjugate Gaussian (both kernels), a Bayes
//! factor, a symmetric bimodal posterior, mixed discrete models (Gibbs and the discrete random walk), impossible
//! regions, a support that moves with a latent, a plugged-in likelihood, and the settings' errors.

use keanu_algorithms::smc::{ModelLikelihood, Smc, SmcKernel, SmcResult, TemperedLikelihood};
use keanu_algorithms::{AlgorithmError, Result};
use keanu_distributions::KeanuRng;
use keanu_graph::{GraphBuilder, Model, NodeId, State};
use keanu_tensor::DoubleTensor;
use std::f64::consts::PI;

const DATA: [f64; 5] = [0.8, 1.9, 1.1, 2.4, 1.3];

/// mu ~ N(0, tau), y_i ~ N(mu, 1) observed at `DATA`.
fn conjugate(tau: f64) -> (Model, State, NodeId) {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, tau).label("mu");
    let y = b.gaussian_with_shape(&[DATA.len()], mu, 1.0);
    b.observe(y, DoubleTensor::vector(DATA.to_vec()));
    let mu = mu.id();
    let (graph, state) = b.build().unwrap();
    (Model::new(graph, &state).unwrap(), state, mu)
}

/// The log evidence of `conjugate(tau)`: y ~ N(0, I + tau² 11ᵀ).
fn conjugate_log_evidence(data: &[f64], tau: f64) -> f64 {
    let n = data.len() as f64;
    let t2 = tau * tau;
    let sum: f64 = data.iter().sum();
    let squares: f64 = data.iter().map(|y| y * y).sum();
    let quadratic = squares - t2 / (1.0 + n * t2) * sum * sum;
    -0.5 * n * (2.0 * PI).ln() - 0.5 * (1.0 + n * t2).ln() - 0.5 * quadratic
}

fn pooled(result: &SmcResult, id: NodeId) -> Vec<f64> {
    result
        .samples
        .chains()
        .iter()
        .flat_map(|c| {
            c.f64(id)
                .unwrap()
                .as_slice()
                .iter()
                .map(|t| t.as_slice()[0])
                .collect::<Vec<_>>()
        })
        .collect()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64]) -> f64 {
    let m = mean(xs);
    xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64
}

fn smc() -> Smc {
    Smc::default().with_particles(1000).with_chains(4)
}

fn check_conjugate(result: &SmcResult, mu: NodeId, tau: f64, tolerance: f64) {
    let exact = conjugate_log_evidence(&DATA, tau);
    for (run, lml) in result.log_marginal_likelihood.iter().enumerate() {
        assert!(
            (lml - exact).abs() < 2.0 * tolerance,
            "run {run}: log evidence {lml}, exact {exact}"
        );
    }
    let combined = result.combined_log_marginal_likelihood();
    assert!(
        (combined - exact).abs() < tolerance,
        "log evidence {combined}, exact {exact}"
    );
    let n = DATA.len() as f64;
    let precision = n + 1.0 / (tau * tau);
    let posterior_mean = DATA.iter().sum::<f64>() / precision;
    let xs = pooled(result, mu);
    assert!(
        (mean(&xs) - posterior_mean).abs() < 0.03,
        "mean {}",
        mean(&xs)
    );
    assert!(
        (variance(&xs) * precision - 1.0).abs() < 0.1,
        "variance {}",
        variance(&xs)
    );
    for betas in &result.betas {
        assert_eq!(betas[0], 0.0);
        assert_eq!(*betas.last().unwrap(), 1.0);
        assert!(betas.windows(2).all(|w| w[0] < w[1]));
    }
    assert!(result.samples.rhat(mu, &[]).unwrap() < 1.01);
}

#[test]
fn log_evidence_of_a_conjugate_gaussian_matches_the_closed_form() {
    let (model, state, mu) = conjugate(2.0);
    let result = smc().sample(&model, &state, &[mu], 1).unwrap();
    check_conjugate(&result, mu, 2.0, 0.1);
    assert_eq!(result.samples.num_chains(), 4);
    assert_eq!(result.samples.draws(), 1000);
    // the independent proposal fits a Gaussian posterior well
    for rates in &result.acceptance_rates {
        assert!(rates.iter().all(|&r| r > 0.5), "{rates:?}");
    }
}

#[test]
fn the_random_walk_kernel_also_matches_the_closed_form() {
    let (model, state, mu) = conjugate(2.0);
    let result = smc()
        .with_kernel(SmcKernel::RandomWalkMetropolis)
        .sample(&model, &state, &[mu], 2)
        .unwrap();
    check_conjugate(&result, mu, 2.0, 0.1);
}

#[test]
fn a_bayes_factor_between_conjugate_models_matches_the_closed_form() {
    let (narrow, narrow_state, mu) = conjugate(1.0);
    let (wide, wide_state, _) = conjugate(10.0);
    let narrow = smc().sample(&narrow, &narrow_state, &[mu], 3).unwrap();
    let wide = smc().sample(&wide, &wide_state, &[mu], 4).unwrap();
    let exact = conjugate_log_evidence(&DATA, 1.0) - conjugate_log_evidence(&DATA, 10.0);
    let estimate =
        narrow.combined_log_marginal_likelihood() - wide.combined_log_marginal_likelihood();
    // the data favour the narrow prior by a factor of about 3.6
    assert!(exact > 1.2);
    assert!(
        (estimate - exact).abs() < 0.15,
        "log Bayes factor {estimate}, exact {exact}"
    );
}

#[test]
fn both_modes_of_a_symmetric_bimodal_posterior_are_sampled_in_proportion() {
    // x ~ N(0, 3), y ~ N(x², 0.5) observed at 4: modes at x = ±2 of equal mass
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 3.0);
    let y = b.gaussian(x * x, 0.5);
    b.observe(y, DoubleTensor::scalar(4.0));
    let x = x.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let result = smc().sample(&model, &state, &[x], 5).unwrap();
    for (run, chain) in result.samples.chains().iter().enumerate() {
        let positive = chain.f64(x).unwrap().probability(|t| t.as_slice()[0] > 0.0);
        assert!(
            (positive - 0.5).abs() < 0.08,
            "run {run}: {positive} of the draws in the positive mode"
        );
    }
    let xs = pooled(&result, x);
    let positive = xs.iter().filter(|&&v| v > 0.0).count() as f64 / xs.len() as f64;
    assert!((positive - 0.5).abs() < 0.04, "{positive}");
    let magnitude = mean(&xs.iter().map(|v| v.abs()).collect::<Vec<_>>());
    assert!((magnitude - 2.0).abs() < 0.1, "mean |x| {magnitude}");
}

#[test]
fn a_mixed_discrete_model_matches_the_exact_posterior() {
    // z ~ Bernoulli(0.3), mu ~ N(0, 1), y_i ~ N(mu + 2 z, 1) observed
    let data = [1.9, 2.6, 1.4];
    let b = GraphBuilder::new();
    let z = b.bernoulli(0.3);
    let mu = b.gaussian(0.0, 1.0);
    let shift = b.where_(z, b.scalar(2.0), b.scalar(0.0));
    let y = b.gaussian_with_shape(&[data.len()], mu + shift, 1.0);
    b.observe(y, DoubleTensor::vector(data.to_vec()));
    let (z, mu) = (z.id(), mu.id());
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let result = smc().sample(&model, &state, &[z, mu], 6).unwrap();

    // given z, y - 2z ~ N(0, I + 11ᵀ)
    let shifted = |s: f64| data.iter().map(|y| y - s).collect::<Vec<_>>();
    let (z0, z1) = (
        conjugate_log_evidence(&shifted(0.0), 1.0),
        conjugate_log_evidence(&shifted(2.0), 1.0),
    );
    let evidence = (0.7 * z0.exp() + 0.3 * z1.exp()).ln();
    let p1 = 0.3 * z1.exp() / evidence.exp();
    let mut probability = 0.0;
    for chain in result.samples.chains() {
        probability += chain.bool(z).unwrap().probability(|t| t.as_slice()[0]) / 4.0;
    }
    assert!(
        (probability - p1).abs() < 0.03,
        "P(z = 1) {probability}, exact {p1}"
    );
    let combined = result.combined_log_marginal_likelihood();
    assert!(
        (combined - evidence).abs() < 0.1,
        "log evidence {combined}, exact {evidence}"
    );
    // the posterior mean of mu mixes the two conditionals' means, sum(y - 2z) / 4
    let exact_mean = (1.0 - p1) * shifted(0.0).iter().sum::<f64>() / 4.0
        + p1 * shifted(2.0).iter().sum::<f64>() / 4.0;
    let mu_mean = mean(&pooled(&result, mu));
    assert!(
        (mu_mean - exact_mean).abs() < 0.05,
        "{mu_mean} {exact_mean}"
    );
    assert!(result.samples.rhat(z, &[]).unwrap() < 1.02);
}

#[test]
fn an_integer_latent_is_mutated_by_the_discrete_random_walk() {
    // n ~ Poisson(5), y ~ N(n, 1) observed at 8: enumerate the posterior
    let b = GraphBuilder::new();
    let n = b.poisson(5.0);
    let y = b.gaussian(n.to_double(), 1.0);
    b.observe(y, DoubleTensor::scalar(8.0));
    let n = n.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let result = smc().sample(&model, &state, &[n], 7).unwrap();

    let mut log_factorial = 0.0;
    let terms: Vec<f64> = (0..80)
        .map(|k| {
            if k > 0 {
                log_factorial += (k as f64).ln();
            }
            let prior = k as f64 * 5f64.ln() - 5.0 - log_factorial;
            let d = 8.0 - k as f64;
            prior - 0.5 * (2.0 * PI).ln() - 0.5 * d * d
        })
        .collect();
    let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = terms.iter().map(|t| (t - max).exp()).sum();
    let evidence = max + total.ln();
    let exact_mean: f64 = terms
        .iter()
        .enumerate()
        .map(|(k, t)| k as f64 * (t - evidence).exp())
        .sum();
    let draws: Vec<f64> = result
        .samples
        .chains()
        .iter()
        .flat_map(|c| {
            c.i64(n)
                .unwrap()
                .as_slice()
                .iter()
                .map(|t| t.as_slice()[0] as f64)
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(
        (mean(&draws) - exact_mean).abs() < 0.1,
        "{} {exact_mean}",
        mean(&draws)
    );
    let combined = result.combined_log_marginal_likelihood();
    assert!(
        (combined - evidence).abs() < 0.1,
        "log evidence {combined}, exact {evidence}"
    );
    assert!(result.acceptance_rates.iter().flatten().all(|r| r.is_nan()));
}

#[test]
fn particles_in_impossible_regions_are_rejected() {
    // x ~ N(0, 1), y ~ Uniform(0, x) observed at 0.5: the likelihood is 1/x for x > 0.5 and zero elsewhere, so
    // most prior draws are impossible
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    let y = b.uniform(0.0, x);
    b.observe(y, DoubleTensor::scalar(0.5));
    let x = x.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let result = smc().sample(&model, &state, &[x], 8).unwrap();
    let xs = pooled(&result, x);
    assert!(xs.iter().all(|&v| v > 0.5));

    // the evidence ∫ φ(x) / x dx over (0.5, ∞) and the posterior mean ∫ φ(x) dx / evidence, by quadrature
    let h: f64 = 1e-4;
    let (mut evidence, mut first) = (0.0f64, 0.0f64);
    let mut v = 0.5 + h / 2.0;
    while v < 10.0 {
        let phi = (-0.5 * v * v).exp() / (2.0 * PI).sqrt();
        evidence += phi / v * h;
        first += phi * h;
        v += h;
    }
    let combined = result.combined_log_marginal_likelihood();
    assert!(
        (combined - evidence.ln()).abs() < 0.1,
        "log evidence {combined}, exact {}",
        evidence.ln()
    );
    assert!((mean(&xs) - first / evidence).abs() < 0.03);
}

#[test]
fn a_support_that_moves_with_a_latent_is_sampled_in_constrained_space() {
    // theta ~ Uniform(1, 2), x ~ Uniform(0, theta), y ~ N(x, 0.1) observed at 0.5: the evidence is about
    // ∫ 1/theta dtheta = ln 2
    let b = GraphBuilder::new();
    let theta = b.uniform(1.0, 2.0);
    let x = b.uniform(0.0, theta);
    let y = b.gaussian(x, 0.1);
    b.observe(y, DoubleTensor::scalar(0.5));
    let x = x.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let result = smc().sample(&model, &state, &[x], 9).unwrap();
    let combined = result.combined_log_marginal_likelihood();
    let exact = 2f64.ln().ln();
    assert!(
        (combined - exact).abs() < 0.1,
        "log evidence {combined}, exact {exact}"
    );
    assert!((mean(&pooled(&result, x)) - 0.5).abs() < 0.02);
}

/// The Gaussian likelihood of `DATA` given the latent `mu`, computed outside the model.
struct ExternalGaussian {
    mu: NodeId,
}

impl TemperedLikelihood for ExternalGaussian {
    fn log_likelihood(&self, _: &Model, state: &mut State, _: &mut KeanuRng) -> Result<f64> {
        let mu = state.f64(self.mu).unwrap().as_slice()[0];
        Ok(DATA
            .iter()
            .map(|y| -0.5 * (2.0 * PI).ln() - 0.5 * (y - mu) * (y - mu))
            .sum())
    }
}

#[test]
fn a_plugged_in_likelihood_is_tempered_like_the_models_own() {
    // the model has the prior alone; the likelihood comes from outside
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 2.0).id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let result = smc()
        .sample_with(&model, &state, &[mu], &ExternalGaussian { mu }, 10)
        .unwrap();
    check_conjugate(&result, mu, 2.0, 0.1);
}

#[test]
fn runs_are_reproducible() {
    let (model, state, mu) = conjugate(2.0);
    let smc = Smc::default().with_particles(200).with_chains(2);
    let a = smc.sample(&model, &state, &[mu], 11).unwrap();
    let b = smc.sample(&model, &state, &[mu], 11).unwrap();
    assert_eq!(a, b);
    let c = smc.sample(&model, &state, &[mu], 12).unwrap();
    assert_ne!(a.log_marginal_likelihood, c.log_marginal_likelihood);
    // one run on its own
    let mut state = state;
    let mut rng = KeanuRng::seed_from_u64(1);
    let run = smc
        .run(&model, &mut state, &[mu], &ModelLikelihood, &mut rng)
        .unwrap();
    assert_eq!(run.samples.len(), 200);
    assert_eq!(run.steps.len(), run.betas.len() - 1);
    assert!(run.steps.iter().all(|&s| (1..=25).contains(&s)));
}

#[test]
fn invalid_settings_and_models_are_errors() {
    let (model, state, mu) = conjugate(2.0);
    let invalid = |smc: Smc| {
        matches!(
            smc.sample(&model, &state, &[mu], 1),
            Err(AlgorithmError::InvalidSetting(_))
        )
    };
    assert!(invalid(Smc::default().with_particles(1)));
    assert!(invalid(Smc::default().with_ess_threshold(1.0)));
    assert!(invalid(Smc::default().with_ess_threshold(0.0)));
    assert!(invalid(Smc::default().with_correlation_threshold(-0.1)));
    assert!(invalid(Smc::default().with_max_steps(0)));
    assert!(invalid(Smc::default().with_max_stages(0)));
    assert!(matches!(
        Smc::default()
            .with_chains(0)
            .sample(&model, &state, &[mu], 1),
        Err(AlgorithmError::InvalidArgument(_))
    ));
    assert!(matches!(
        Smc::default().sample(&model, &state, &[], 1),
        Err(AlgorithmError::InvalidArgument(_))
    ));
    // too few stages for an informative likelihood
    assert!(matches!(
        Smc::default()
            .with_particles(100)
            .with_chains(1)
            .with_max_stages(1)
            .sample(&model, &state, &[mu], 1),
        Err(AlgorithmError::InvalidArgument(message)) if message.contains("stages")
    ));

    // no latents
    let b = GraphBuilder::new();
    let y = b.gaussian(0.0, 1.0);
    b.observe(y, DoubleTensor::scalar(0.0));
    let y = y.id();
    let (graph, state) = b.build().unwrap();
    let observed_only = Model::new(graph, &state).unwrap();
    assert!(matches!(
        Smc::default().sample(&observed_only, &state, &[y], 1),
        Err(AlgorithmError::NoLatents)
    ));

    // discrete latents need a likelihood of model terms
    let b = GraphBuilder::new();
    let z = b.bernoulli(0.5);
    let mu = b.gaussian(0.0, 1.0);
    let _ = b.where_(z, mu, b.scalar(0.0));
    let (z, mu) = (z.id(), mu.id());
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let error = Smc::default()
        .sample_with(&model, &state, &[z], &ExternalGaussian { mu }, 1)
        .unwrap_err();
    assert!(error.to_string().contains("model_terms"), "{error}");
}
