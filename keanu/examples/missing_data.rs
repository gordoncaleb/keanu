//! What do the data say when some of them are missing?
//!
//! A Gaussian regression of an outcome y on two covariates, y ~ N(β₀ + β₁x₁ + β₂x₂, σ), on [`ROWS`] simulated
//! rows, where a fifth of the outcomes and a tenth of the first covariate are missing at random. Dropping every row
//! with a gap would throw away almost 30% of the data; instead the missing values are treated as unknowns and
//! imputed along with the parameters.
//!
//! The outcome already has a distribution. The covariate x₁ is given one too, x₁ ~ N(μₓ, σₓ), so that its missing
//! elements have a prior to be imputed from (x₂, fully observed, stays a constant). Both are observed with
//! [`observe_masked`], from values with NaN where an element is missing: each becomes an observed part
//! (`y_observed`), a latent of its missing elements (`y_missing`) and a node of the original shape and label (`y`)
//! that puts them back in place. The default sampler is then NUTS over the parameters and both missing parts, and
//! the draws of `y` and `x1` hold the imputed values in their original positions.
//!
//! The data are simulated from known values, so the coefficients and the imputed outcomes can be checked against
//! the truth. Run with `cargo run --release --example missing_data`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::mcmc::chains::sample_chains;
use keanu::prelude::*;

/// The number of rows.
pub const ROWS: usize = 200;
/// The coefficients the outcomes are simulated with: intercept, x₁ and x₂.
pub const TRUE_BETA: [f64; 3] = [0.5, 1.5, -1.0];
/// The outcome's noise.
pub const TRUE_SIGMA: f64 = 0.5;
/// The fraction of outcomes missing.
pub const MISSING_OUTCOMES: f64 = 0.2;
/// The fraction of the first covariate missing.
pub const MISSING_COVARIATE: f64 = 0.1;
/// The seed the example's data are simulated with.
pub const DATA_SEED: u64 = 1;

/// A simulated data set: the complete values, and which are missing.
#[derive(Debug, Clone)]
pub struct Data {
    pub x1: Vec<f64>,
    pub x2: Vec<f64>,
    pub y: Vec<f64>,
    /// The rows whose outcome is missing.
    pub y_missing: Vec<usize>,
    /// The rows whose first covariate is missing.
    pub x1_missing: Vec<usize>,
}

impl Data {
    /// The values as recorded: NaN where missing.
    fn recorded(values: &[f64], missing: &[usize]) -> Vec<f64> {
        let mut values = values.to_vec();
        for &i in missing {
            values[i] = f64::NAN;
        }
        values
    }
}

/// `count` distinct rows chosen at random.
fn choose(count: usize, rng: &mut KeanuRng) -> Vec<usize> {
    let mut rows: Vec<usize> = (0..ROWS).collect();
    for i in 0..count {
        let j = i + rng.next_below((ROWS - i) as u64) as usize;
        rows.swap(i, j);
    }
    let mut chosen = rows[..count].to_vec();
    chosen.sort_unstable();
    chosen
}

/// Simulates the complete data and which values are missing (completely at random).
pub fn data(seed: u64) -> Data {
    let mut rng = KeanuRng::seed_from_u64(seed);
    let x1: Vec<f64> = (0..ROWS).map(|_| 1.0 + rng.next_gaussian()).collect();
    let x2: Vec<f64> = (0..ROWS).map(|_| rng.next_gaussian()).collect();
    let y = x1
        .iter()
        .zip(&x2)
        .map(|(a, b)| {
            TRUE_BETA[0] + TRUE_BETA[1] * a + TRUE_BETA[2] * b + TRUE_SIGMA * rng.next_gaussian()
        })
        .collect();
    let y_missing = choose((MISSING_OUTCOMES * ROWS as f64) as usize, &mut rng);
    let x1_missing = choose((MISSING_COVARIATE * ROWS as f64) as usize, &mut rng);
    Data {
        x1,
        x2,
        y,
        y_missing,
        x1_missing,
    }
}

/// A fit: every chain's draws of the coefficients, σ and the completed outcome and covariate.
#[derive(Debug)]
pub struct Fit {
    pub chains: ChainSamples,
    pub beta: NodeId,
    pub sigma: NodeId,
    /// The outcome with its imputed elements in place, shape [ROWS].
    pub y: NodeId,
    /// The first covariate with its imputed elements in place, shape [ROWS].
    pub x1: NodeId,
}

/// Fits the regression with the missing values imputed, by NUTS with `chains` chains of `draws` draws after as many
/// warm-up steps.
pub fn run(data: &Data, chains: usize, draws: usize, seed: u64) -> keanu::Result<Fit> {
    let b = GraphBuilder::new();
    let mu_x = b.gaussian(0.0, 5.0).label("mu_x");
    let sigma_x = b.half_gaussian(2.0).label("sigma_x");
    let x1 = b.gaussian_with_shape(&[ROWS], mu_x, sigma_x).label("x1");
    let x2 = b.constant(DoubleTensor::vector(data.x2.clone()));
    let beta = b.gaussian_with_shape(&[3], 0.0, 5.0).label("beta");
    let sigma = b.half_gaussian(2.0).label("sigma");
    let mean = beta.take(&[0]) + beta.take(&[1]) * x1 + beta.take(&[2]) * x2;
    let y = b.gaussian_with_shape(&[ROWS], mean, sigma).label("y");
    b.set_value(sigma_x, DoubleTensor::scalar(1.0));
    b.set_value(sigma, DoubleTensor::scalar(1.0));
    b.set_value(mu_x, DoubleTensor::scalar(0.0));
    b.set_value(beta, DoubleTensor::vector(vec![0.0; 3]));
    let (x1, y) = (x1.id(), y.id());
    let (graph, state) = b.build()?;

    // observe what was recorded; the NaNs become the latents x1_missing and y_missing
    let (graph, state, imputed) = observe_masked(
        &graph,
        &state,
        &[
            MaskedObservation::with_nan_missing(
                y,
                DoubleTensor::vector(Data::recorded(&data.y, &data.y_missing)),
            ),
            MaskedObservation::with_nan_missing(
                x1,
                DoubleTensor::vector(Data::recorded(&data.x1, &data.x1_missing)),
            ),
        ],
    )?;
    let model = Model::new(graph, &state)?;
    let find = |label: &str| model.find(label).expect("labels are kept");
    let (beta, sigma) = (find("beta"), find("sigma"));
    let (y, x1) = (imputed.node(y).expect("y"), imputed.node(x1).expect("x1"));
    let Sampler::Nuts(nuts) = Sampler::for_model(&model, &state) else {
        unreachable!("every latent is continuous and differentiable");
    };
    let nuts = nuts.with_adapt_count(draws);
    let latents = model.latents().to_vec();
    let mut recorded = latents.clone();
    recorded.extend([y, x1]);
    let chains = sample_chains(chains, seed, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &recorded,
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            rng,
        )
        .map(|(samples, _)| samples)
    })?;
    Ok(Fit {
        chains,
        beta,
        sigma,
        y,
        x1,
    })
}

/// The posterior mean of each element of a node, over every chain's draws.
pub fn posterior_mean(chains: &ChainSamples, node: NodeId) -> Vec<f64> {
    let mut total = vec![0.0; ROWS];
    let mut count = 0;
    for chain in chains.chains() {
        for draw in chain.f64(node).expect("recorded").as_slice() {
            for (t, v) in total.iter_mut().zip(draw.as_slice()) {
                *t += v;
            }
            count += 1;
        }
    }
    total.into_iter().map(|t| t / count as f64).collect()
}

/// The correlation of two equally long series.
pub fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
    }
    sab / (saa * sbb).sqrt()
}

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let data = data(DATA_SEED);
    println!(
        "{ROWS} rows, {} outcomes and {} values of x1 missing",
        data.y_missing.len(),
        data.x1_missing.len()
    );
    let fit = run(&data, 4, 1000, 1)?;
    for (i, truth) in TRUE_BETA.iter().enumerate() {
        let s = fit.chains.summarize(fit.beta, &[i])?;
        println!(
            "beta[{i}]: mean {:.3}, 90% interval [{:.3}, {:.3}], R-hat {:.3} (truth {truth})",
            s.mean, s.q5, s.q95, s.r_hat
        );
    }
    let s = fit.chains.summarize(fit.sigma, &[])?;
    println!(
        "sigma: mean {:.3}, 90% interval [{:.3}, {:.3}] (truth {TRUE_SIGMA})",
        s.mean, s.q5, s.q95
    );
    let y = posterior_mean(&fit.chains, fit.y);
    let pick = |values: &[f64], rows: &[usize]| rows.iter().map(|&i| values[i]).collect::<Vec<_>>();
    println!(
        "imputed outcomes: correlation {:.3} with the held-out values",
        correlation(&pick(&y, &data.y_missing), &pick(&data.y, &data.y_missing))
    );
    let x1 = posterior_mean(&fit.chains, fit.x1);
    println!(
        "imputed x1: correlation {:.3} with the held-out values",
        correlation(
            &pick(&x1, &data.x1_missing),
            &pick(&data.x1, &data.x1_missing)
        )
    );
    Ok(())
}
