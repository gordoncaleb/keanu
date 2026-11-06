//! Is the trend in some data a straight line or a curve? Model comparison by the marginal likelihood (the
//! evidence), which tempered sequential Monte Carlo estimates as it samples.
//!
//! Thirty points y_i at x_i evenly spaced on [−1, 1] are simulated from a quadratic trend with Gaussian noise of
//! known scale σ = 0.3 ([`data`]). Two polynomial regressions with independent standard normal priors on their
//! coefficients compete:
//!
//! ```text
//! linear:     y_i ~ N(w₀ + w₁ x_i, σ)
//! quadratic:  y_i ~ N(w₀ + w₁ x_i + w₂ x_i², σ)          w_j ~ N(0, 1)
//! ```
//!
//! [`Smc`] moves 4 independent populations of particles from each model's prior to its posterior and estimates the
//! log evidence log p(y | model). The difference of two log evidences is the log Bayes factor, and with equal prior
//! odds the posterior probability of the quadratic model is 1 / (1 + e^(−log Bayes factor)). Both models are
//! linear-Gaussian, so their evidence has a closed form, y ~ N(0, σ² I + X Xᵀ) for the design matrix X
//! ([`exact_log_evidence`]), which the estimates match to about 0.1. The example also scores both models by PSIS-LOO
//! on the SMC particles, which prefers the quadratic model too.
//!
//! Run with `cargo run --release --example smc_model_comparison`.

use keanu::algorithms::loo::loo;
use keanu::prelude::*;

/// The number of points.
pub const POINTS: usize = 30;

/// The noise scale, known to both models.
pub const SIGMA: f64 = 0.3;

/// The quadratic trend the data are simulated from: 0.5 + x + 0.6 x².
pub const TRUTH: [f64; 3] = [0.5, 1.0, 0.6];

/// The inputs, evenly spaced on [−1, 1].
pub fn inputs() -> Vec<f64> {
    (0..POINTS)
        .map(|i| -1.0 + 2.0 * i as f64 / (POINTS - 1) as f64)
        .collect()
}

/// The data: the quadratic trend at [`inputs`] plus noise of scale [`SIGMA`], from a generator seeded with `seed`.
pub fn data(seed: u64) -> Vec<f64> {
    let mut rng = KeanuRng::seed_from_u64(seed);
    inputs()
        .iter()
        .map(|&x| TRUTH[0] + TRUTH[1] * x + TRUTH[2] * x * x + SIGMA * rng.next_gaussian())
        .collect()
}

/// The design matrix [POINTS, degree + 1]: 1, x, x², ... in each row.
fn design(degree: usize) -> DoubleTensor {
    let values = inputs()
        .iter()
        .flat_map(|&x| (0..=degree).map(move |j| x.powi(j as i32)))
        .collect();
    DoubleTensor::new(&[POINTS, degree + 1], values).expect("the design matrix's shape")
}

/// The polynomial regression of the given degree on `y`, and its coefficients' node.
pub fn model(degree: usize, y: &[f64]) -> keanu::Result<(Model, State, NodeId)> {
    let b = GraphBuilder::new();
    // ANCHOR: smc_regression
    let weights = b
        .gaussian_with_shape(&[degree + 1, 1], 0.0, 1.0)
        .label("weights");
    let mean = b.constant(design(degree)).matmul(weights);
    let observed = b.gaussian(mean, SIGMA);
    b.observe(observed, DoubleTensor::new(&[POINTS, 1], y.to_vec())?);
    // ANCHOR_END: smc_regression
    let weights = weights.id();
    let (graph, state) = b.build()?;
    Ok((Model::new(graph, &state)?, state, weights))
}

/// The closed-form log evidence of the regression of the given degree: y ~ N(0, σ² I + X Xᵀ).
pub fn exact_log_evidence(degree: usize, y: &[f64]) -> keanu::Result<f64> {
    let x = design(degree);
    let mut covariance = x.matmul_transposed(&x, false, true)?.into_vec();
    for i in 0..POINTS {
        covariance[i * POINTS + i] += SIGMA * SIGMA;
    }
    let factor = DoubleTensor::new(&[POINTS, POINTS], covariance)?.cholesky()?;
    let whitened =
        factor.solve_triangular(&DoubleTensor::new(&[POINTS, 1], y.to_vec())?, true, false)?;
    let log_det: f64 = (0..POINTS)
        .map(|i| factor.as_slice()[i * POINTS + i].ln())
        .sum();
    let quadratic: f64 = whitened.as_slice().iter().map(|v| v * v).sum();
    Ok(-0.5 * POINTS as f64 * (2.0 * std::f64::consts::PI).ln() - log_det - 0.5 * quadratic)
}

/// One model's fit: its SMC log evidence (the runs combined), the closed form, and its PSIS-LOO elpd and standard
/// error from the particles.
#[derive(Debug, Clone, Copy)]
pub struct Fit {
    /// The polynomial's degree.
    pub degree: usize,
    /// The SMC estimate of the log evidence.
    pub log_evidence: f64,
    /// The closed-form log evidence.
    pub exact_log_evidence: f64,
    /// The expected log pointwise predictive density by PSIS-LOO.
    pub elpd_loo: f64,
    /// Its standard error.
    pub elpd_loo_se: f64,
}

/// Fits the regression of the given degree to `y` with SMC (4 runs of `particles` particles).
pub fn fit(degree: usize, y: &[f64], particles: usize, seed: u64) -> keanu::Result<Fit> {
    let (model, state, weights) = model(degree, y)?;
    // ANCHOR: smc_evidence
    let result =
        Smc::default()
            .with_particles(particles)
            .sample(&model, &state, &[weights], seed)?;
    let log_evidence = result.combined_log_marginal_likelihood();
    // ANCHOR_END: smc_evidence
    let loo = loo(&PointwiseLogLik::from_chains(
        &model,
        &state,
        &result.samples,
    )?)?;
    Ok(Fit {
        degree,
        log_evidence,
        exact_log_evidence: exact_log_evidence(degree, y)?,
        elpd_loo: loo.elpd,
        elpd_loo_se: loo.se,
    })
}

/// The linear and quadratic fits to the data simulated with `seed`.
pub fn run(particles: usize, seed: u64) -> keanu::Result<[Fit; 2]> {
    let y = data(seed);
    Ok([fit(1, &y, particles, seed)?, fit(2, &y, particles, seed)?])
}

/// The posterior probability of the second model given equal prior odds, from a log Bayes factor of the second
/// over the first.
pub fn probability(log_bayes_factor: f64) -> f64 {
    1.0 / (1.0 + (-log_bayes_factor).exp())
}

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let [linear, quadratic] = run(2000, 1)?;
    println!("model       log evidence (SMC)  (exact)   elpd LOO (se)");
    for (name, f) in [("linear", linear), ("quadratic", quadratic)] {
        println!(
            "{name:<10}  {:>18.2}  {:>7.2}   {:>7.2} ({:.2})",
            f.log_evidence, f.exact_log_evidence, f.elpd_loo, f.elpd_loo_se
        );
    }
    let log_bayes_factor = quadratic.log_evidence - linear.log_evidence;
    println!(
        "\nlog Bayes factor, quadratic over linear: {log_bayes_factor:.2} (exact {:.2})",
        quadratic.exact_log_evidence - linear.exact_log_evidence
    );
    println!(
        "posterior probability of the quadratic model: {:.4}",
        probability(log_bayes_factor)
    );
    Ok(())
}
