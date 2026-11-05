//! What smooth function produced these noisy measurements, and how sure can we be of it between them?
//!
//! Gaussian process regression on synthetic data: noisy observations y_i = sin(x_i) + ε_i of a sine curve, with
//! a zero-mean GP prior on the function and an exponentiated quadratic (squared exponential) kernel. The kernel's
//! amplitude and length scale and the observation noise are unknown and get half-Gaussian priors:
//!
//! ```text
//! amplitude ~ HalfGaussian(2)    length_scale ~ HalfGaussian(2)    noise ~ HalfGaussian(1)
//! f ~ GP(0, amplitude² exp(−(x − x')² / (2 length_scale²)))    y_i ~ N(f(x_i), noise)
//! ```
//!
//! The function is integrated out analytically, so NUTS samples only the three hyperparameters from the GP
//! marginal likelihood (four chains). The prediction on a grid then averages over the posterior: for each draw of
//! the hyperparameters the GP's conditional mean and variance of f at the grid are computed in closed form, and the
//! posterior-predictive mean is the average of the conditional means, with variance the average conditional
//! variance plus the variance of the conditional means (the law of total variance).
//!
//! The predictive nodes are built in a separate graph whose hyperparameters are constants, set to each draw in turn.
//! Keeping them out of the sampled model means NUTS doesn't recompute the prediction at every leapfrog step.
//!
//! The second half fits the same model to 200 points with a Hilbert space approximate GP (HSGP, Riutort-Mayol et
//! al. 2023), as one would for a series too long for an exact GP: the function is a weighted sum of 25 fixed basis
//! functions whose weights are latent, so NUTS samples the weights with the hyperparameters, and the prediction on
//! the grid is the same weighted sum there, evaluated at every draw.
//!
//! Run with `cargo run --release --example gp_regression`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::mcmc::chains::sample_chains;
use keanu::graph::gp::{GaussianProcess, Hsgp, Kernel};
use keanu::prelude::*;

/// The number of chains.
pub const CHAINS: usize = 4;
/// The number of training points.
pub const POINTS: usize = 40;
/// The standard deviation of the noise added to the observations.
pub const NOISE: f64 = 0.2;
/// The training inputs are evenly spaced on [0, `RANGE`].
pub const RANGE: f64 = 10.0;
/// The number of grid points the function is predicted at.
pub const GRID_POINTS: usize = 21;

/// The function the data are generated from.
pub fn truth(x: f64) -> f64 {
    x.sin()
}

/// The posterior and the posterior-predictive distribution of the function on the grid.
#[derive(Debug)]
pub struct GpRegression {
    /// The draws of every chain.
    pub samples: ChainSamples,
    /// The kernel amplitude.
    pub amplitude: NodeId,
    /// The kernel length scale.
    pub length_scale: NodeId,
    /// The observation noise standard deviation.
    pub noise: NodeId,
    /// The grid the function is predicted at, evenly spaced on [0, `RANGE`].
    pub grid: Vec<f64>,
    /// The posterior-predictive mean of the function at each grid point.
    pub mean: Vec<f64>,
    /// The posterior-predictive standard deviation of the function at each grid point.
    pub sd: Vec<f64>,
}

/// Noisy observations of [`truth`] at `POINTS` evenly spaced inputs, from a generator seeded with `seed`.
pub fn data(seed: u64) -> (Vec<f64>, Vec<f64>) {
    data_at(POINTS, seed)
}

/// Noisy observations of [`truth`] at `points` evenly spaced inputs on [0, `RANGE`].
fn data_at(points: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = KeanuRng::seed_from_u64(seed);
    let x: Vec<f64> = (0..points)
        .map(|i| RANGE * i as f64 / (points - 1) as f64)
        .collect();
    let y = x
        .iter()
        .map(|&x| truth(x) + NOISE * rng.next_gaussian())
        .collect();
    (x, y)
}

/// The GP with the given hyperparameters: the same kernel in the sampled model and the predictive graph.
fn gaussian_process<'b>(
    b: &'b GraphBuilder,
    amplitude: DoubleVar<'b>,
    length_scale: DoubleVar<'b>,
    noise: DoubleVar<'b>,
) -> GaussianProcess<'b> {
    GaussianProcess::new(Kernel::exp_quad(b, amplitude, length_scale)).with_noise(noise)
}

/// Samples the hyperparameters with `CHAINS` NUTS chains of `draws` draws each (after as many adaptation steps)
/// and averages the GP's prediction over every draw.
pub fn run(seed: u64, draws: usize) -> keanu::Result<GpRegression> {
    let (x, y) = data(seed);
    let (x, y) = (DoubleTensor::vector(x), DoubleTensor::vector(y));

    // the model: the hyperparameters' priors and the GP marginal likelihood of the observations
    let b = GraphBuilder::new();
    let amplitude = b.half_gaussian(2.0).label("amplitude");
    let length_scale = b.half_gaussian(2.0).label("length_scale");
    let noise = b.half_gaussian(1.0).label("noise");
    b.set_value(amplitude, DoubleTensor::scalar(1.0));
    b.set_value(length_scale, DoubleTensor::scalar(1.0));
    b.set_value(noise, DoubleTensor::scalar(0.5));
    gaussian_process(&b, amplitude, length_scale, noise).marginal(x.clone(), y.clone());
    let latents = [amplitude.id(), length_scale.id(), noise.id()];
    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;

    let nuts = Nuts::default().with_adapt_count(draws);
    let samples = sample_chains(CHAINS, seed, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &latents,
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            rng,
        )
        .map(|(samples, _)| samples)
    })?;

    // the prediction: the same GP with constant hyperparameters, conditioned on the data, at the grid
    let grid = grid();
    let p = GraphBuilder::new();
    let hyperparameters = [p.scalar(1.0), p.scalar(1.0), p.scalar(1.0)];
    let [a, l, s] = hyperparameters;
    let prediction =
        gaussian_process(&p, a, l, s).predict(x, y, DoubleTensor::vector(grid.clone()));
    let (mean, variance) = (prediction.mean.id(), prediction.variance().id());
    let hyperparameters = hyperparameters.map(|h| h.id());
    let (predictive, mut values) = p.build()?;

    // average the conditional mean and variance over every draw of every chain
    let mut rng = KeanuRng::seed_from_u64(seed);
    let (mut sum, mut sum_squares, mut sum_variances) = (
        vec![0.0; GRID_POINTS],
        vec![0.0; GRID_POINTS],
        vec![0.0; GRID_POINTS],
    );
    let mut count = 0.0;
    for chain in samples.chains() {
        for draw in chain.iter() {
            for (&target, &source) in hyperparameters.iter().zip(&latents) {
                let value = draw.f64(source).expect("every latent is sampled").clone();
                values.set(&predictive, target, value)?;
            }
            values.eval(&predictive, &[mean, variance], &mut rng)?;
            let (m, v) = (values.f64(mean).unwrap(), values.f64(variance).unwrap());
            for i in 0..GRID_POINTS {
                let m = m.as_slice()[i];
                sum[i] += m;
                sum_squares[i] += m * m;
                sum_variances[i] += v.as_slice()[i];
            }
            count += 1.0;
        }
    }
    let mean: Vec<f64> = sum.iter().map(|s| s / count).collect();
    let sd = (0..GRID_POINTS)
        .map(|i| {
            let variance_of_means = (sum_squares[i] / count - mean[i] * mean[i]).max(0.0);
            (sum_variances[i] / count + variance_of_means).sqrt()
        })
        .collect();

    Ok(GpRegression {
        samples,
        amplitude: latents[0],
        length_scale: latents[1],
        noise: latents[2],
        grid,
        mean,
        sd,
    })
}

/// `GRID_POINTS` evenly spaced points on [0, `RANGE`].
fn grid() -> Vec<f64> {
    (0..GRID_POINTS)
        .map(|i| RANGE * i as f64 / (GRID_POINTS - 1) as f64)
        .collect()
}

/// The number of training points of the HSGP fit.
pub const HSGP_POINTS: usize = 200;
/// The number of HSGP basis functions. With the domain extended by `HSGP_C` and length scales down to 0.2 of the
/// inputs' half range, Riutort-Mayol et al.'s rule m ≥ 1.75 c / (ℓ / S) asks for at least 14.
pub const BASIS_FUNCTIONS: usize = 25;
/// The factor by which the HSGP's domain extends the inputs' range.
pub const HSGP_C: f64 = 1.5;

/// The same model on `HSGP_POINTS` points with an HSGP in place of the exact GP: NUTS samples the hyperparameters
/// and the basis functions' weights with `CHAINS` chains of `draws` draws (after as many adaptation steps), and the
/// prediction is the function's posterior mean and standard deviation on the grid.
pub fn run_hsgp(seed: u64, draws: usize) -> keanu::Result<GpRegression> {
    let (x, y) = data_at(HSGP_POINTS, seed);
    let grid = grid();

    let b = GraphBuilder::new();
    let amplitude = b.half_gaussian(2.0).label("amplitude");
    let length_scale = b.half_gaussian(2.0).label("length_scale");
    let noise = b.half_gaussian(1.0).label("noise");
    b.set_value(amplitude, DoubleTensor::scalar(1.0));
    b.set_value(length_scale, DoubleTensor::scalar(1.0));
    b.set_value(noise, DoubleTensor::scalar(0.5));
    let kernel = Kernel::exp_quad(&b, amplitude, length_scale);
    let hsgp = Hsgp::new(&kernel, BASIS_FUNCTIONS, HSGP_C)?.prior(DoubleTensor::vector(x));
    // the function on the grid, with the same weights
    let f_grid = hsgp.predict(DoubleTensor::vector(grid.clone())).id();
    let y_node = b.gaussian(hsgp.f, noise);
    b.observe(y_node, DoubleTensor::vector(y));
    let latents = [amplitude.id(), length_scale.id(), noise.id()];
    drop((hsgp, kernel));
    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;

    let nuts = Nuts::default().with_adapt_count(draws);
    let recorded = [latents[0], latents[1], latents[2], f_grid];
    let samples = sample_chains(CHAINS, seed, &state, |state, rng| {
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
    let (mut mean, mut sd) = (Vec::new(), Vec::new());
    for i in 0..GRID_POINTS {
        let s = samples.summarize(f_grid, &[i])?;
        mean.push(s.mean);
        sd.push(s.sd);
    }
    Ok(GpRegression {
        samples,
        amplitude: latents[0],
        length_scale: latents[1],
        noise: latents[2],
        grid,
        mean,
        sd,
    })
}

/// The number of draws per chain (after as many adaptation steps).
pub const DEFAULT_SAMPLES: usize = 1000;

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    println!("exact GP, {POINTS} points:");
    report(&run(1, DEFAULT_SAMPLES)?)?;
    println!("\nHSGP with {BASIS_FUNCTIONS} basis functions, {HSGP_POINTS} points:");
    report(&run_hsgp(1, DEFAULT_SAMPLES)?)
}

/// Prints the hyperparameters' posterior summaries and the prediction on the grid.
#[allow(dead_code)]
fn report(result: &GpRegression) -> keanu::Result<()> {
    for (name, id) in [
        ("amplitude", result.amplitude),
        ("length scale", result.length_scale),
        ("noise", result.noise),
    ] {
        let s = result.samples.summarize(id, &[])?;
        println!(
            "{name}: {:.3} (90% interval {:.3} to {:.3}, R-hat {:.3})",
            s.mean, s.q5, s.q95, s.r_hat
        );
    }
    println!("\n    x   sin(x)   predicted ± sd");
    for ((x, m), s) in result.grid.iter().zip(&result.mean).zip(&result.sd) {
        println!("{x:5.1} {:8.3} {m:8.3} ± {s:.3}", truth(*x));
    }
    Ok(())
}
