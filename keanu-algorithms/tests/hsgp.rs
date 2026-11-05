//! NUTS on a Hilbert space approximate GP over a 1,000-point series: the chains converge, and the posterior mean of
//! the function agrees with the exact GP's on a 200-point subset. The hyperparameters are fixed, so that the exact
//! GP's posterior mean is a closed form to compare with; the GP regression example samples them.

use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::chains::sample_chains;
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_distributions::KeanuRng;
use keanu_graph::gp::{GaussianProcess, Hsgp, Kernel};
use keanu_graph::{GraphBuilder, Model};
use keanu_tensor::DoubleTensor;

const POINTS: usize = 1000;
const RANGE: f64 = 10.0;
const NOISE: f64 = 0.3;
const AMPLITUDE: f64 = 1.0;
const LENGTH_SCALE: f64 = 0.8;
/// Riutort-Mayol et al.'s rule for c = 1.5 and ℓ / S = 0.16 asks for at least 17.
const BASIS_FUNCTIONS: usize = 20;
const C: f64 = 1.5;

/// A smooth series observed with noise at `POINTS` evenly spaced inputs on [0, `RANGE`].
fn series() -> (DoubleTensor, DoubleTensor) {
    let mut rng = KeanuRng::seed_from_u64(3);
    let x: Vec<f64> = (0..POINTS)
        .map(|i| RANGE * i as f64 / (POINTS - 1) as f64)
        .collect();
    let y = x
        .iter()
        .map(|&x| x.sin() + 0.5 * (2.2 * x).cos() + NOISE * rng.next_gaussian())
        .collect();
    (DoubleTensor::vector(x), DoubleTensor::vector(y))
}

/// Every fifth input: the 200-point subset the exact GP's posterior mean is compared on.
fn subset(x: &DoubleTensor) -> (Vec<usize>, DoubleTensor) {
    let indices: Vec<usize> = (0..POINTS).step_by(5).collect();
    let values = indices.iter().map(|&i| x.as_slice()[i]).collect();
    (indices, DoubleTensor::vector(values))
}

/// The exact GP's posterior mean of the function at `x_new` given the whole series, with these hyperparameters.
fn exact_posterior_mean(
    x: &DoubleTensor,
    y: &DoubleTensor,
    x_new: &DoubleTensor,
    amplitude: f64,
    length_scale: f64,
    noise: f64,
) -> Vec<f64> {
    let b = GraphBuilder::new();
    let gp = GaussianProcess::new(Kernel::exp_quad(&b, amplitude, length_scale)).with_noise(noise);
    let mean = gp.predict(x.clone(), y.clone(), x_new.clone()).mean.id();
    drop(gp);
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &[mean], &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    state.f64(mean).unwrap().as_slice().to_vec()
}

#[test]
fn nuts_on_a_thousand_point_series_matches_the_exact_gp_posterior_mean() {
    let (x, y) = series();
    let (indices, x_subset) = subset(&x);
    let b = GraphBuilder::new();
    let kernel = Kernel::exp_quad(&b, AMPLITUDE, LENGTH_SCALE);
    let prior = Hsgp::new(&kernel, BASIS_FUNCTIONS, C)
        .unwrap()
        .prior(x.clone());
    let observed = b.gaussian(prior.f, NOISE);
    b.observe(observed, y.clone());
    let (f, beta) = (prior.f.id(), prior.beta.id());
    drop((prior, kernel));
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    // 4 chains of 300 draws after 200 adaptation steps: the weights' posterior is correlated (the basis functions
    // are not orthogonal on the data's part of the domain), so trees are about 7 deep, and this takes ~20 s
    let nuts = Nuts::default().with_adapt_count(200);
    let samples = sample_chains(4, 7, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &[f, beta],
            500,
            SamplingOptions::default().with_drop(200),
            rng,
        )
        .map(|(samples, _)| samples)
    })
    .unwrap();
    for j in 0..BASIS_FUNCTIONS {
        let s = samples.summarize(beta, &[j]).unwrap();
        assert!(s.r_hat < 1.01, "R-hat of beta[{j}] is {}", s.r_hat);
    }

    let exact = exact_posterior_mean(&x, &y, &x_subset, AMPLITUDE, LENGTH_SCALE, NOISE);
    for (&i, &exact) in indices.iter().zip(&exact) {
        let s = samples.summarize(f, &[i]).unwrap();
        // the posterior sd of f is about 0.06 and the Monte Carlo error of its mean a few thousandths
        let error = (s.mean - exact).abs();
        assert!(
            error < 0.02 && error < 5.0 * s.mcse_mean + 1e-3,
            "f({}): HSGP posterior mean {} vs exact {exact} (MCSE {})",
            x.as_slice()[i],
            s.mean,
            s.mcse_mean
        );
    }
}
