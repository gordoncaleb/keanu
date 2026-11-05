//! NUTS on constrained latents: truncated, circular, half-Gaussian, Dirichlet and mixture-weight latents, constrained and unconstrained space, and supports it can't sample.

mod mcmc_common;

use keanu_algorithms::AlgorithmError;
use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_algorithms::statistics::effective_sample_size;
use keanu_graph::{GraphBuilder, Model, NodeId, Space, State};
use keanu_tensor::{DoubleTensor, IntTensor};
use mcmc_common::*;

/// A latent N(0, 1) truncated to [a, b] (either bound may be infinite): NUTS stays inside the bounds through the
/// truncation's support transform and finds the truncated mean and variance.
#[test]
fn nuts_keeps_a_truncated_latent_inside_its_bounds() {
    use keanu_graph::DoubleVar;
    use keanu_graph::op::DistributionKind;
    let phi = |x: f64| {
        if x.is_finite() {
            (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
        } else {
            0.0
        }
    };
    let x_phi = |x: f64| if x.is_finite() { x * phi(x) } else { 0.0 };
    let cdf = keanu_tensor::special::ndtr;
    for (a, b, seed) in [(0.5, 2.0, 21), (-f64::INFINITY, -1.0, 22)] {
        let builder = GraphBuilder::new();
        let (mu, sigma) = (builder.scalar(0.0), builder.scalar(1.0));
        let x: DoubleVar =
            builder.truncated(DistributionKind::Gaussian, &[mu.id(), sigma.id()], a, b);
        let start = if b.is_finite() && a.is_finite() {
            (a + b) / 2.0
        } else {
            b - 1.0
        };
        builder.set_value(x, DoubleTensor::scalar(start));
        let x = x.id();
        let (graph, mut state) = builder.build().unwrap();
        let model = Model::new(graph, &state).unwrap();
        let samples = Nuts::default()
            .with_adapt_count(500)
            .sample_with_statistics(
                &model,
                &mut state,
                &[x],
                4000,
                SamplingOptions::default()
                    .with_drop(500)
                    .with_down_sample_interval(1),
                &mut rng(seed),
            )
            .unwrap()
            .0;
        let xs = series(&samples, x);
        assert!(
            xs.iter().all(|&x| x > a && x < b),
            "[{a}, {b}]: a sample left the bounds"
        );
        let z = cdf(b) - cdf(a);
        let m = (phi(a) - phi(b)) / z;
        let v = 1.0 + (x_phi(a) - x_phi(b)) / z - m * m;
        assert_moments(&xs, m, v, &format!("truncated [{a}, {b}]"));
    }
}

/// A vector latent N(0, 1) truncated element-wise to [0, ∞) and (-∞, 1]: the elements have different bound kinds,
/// and NUTS still samples it in unconstrained space and finds each element's truncated moments.
#[test]
fn nuts_samples_a_truncated_vector_latent_with_mixed_bounds() {
    use keanu_graph::DoubleVar;
    use keanu_graph::op::DistributionKind;
    let phi = |x: f64| {
        if x.is_finite() {
            (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
        } else {
            0.0
        }
    };
    let x_phi = |x: f64| if x.is_finite() { x * phi(x) } else { 0.0 };
    let cdf = keanu_tensor::special::ndtr;
    let (lower, upper) = ([0.0, f64::NEG_INFINITY], [f64::INFINITY, 1.0]);
    let builder = GraphBuilder::new();
    let (mu, sigma) = (builder.scalar(0.0), builder.scalar(1.0));
    let x: DoubleVar = builder.truncated(
        DistributionKind::Gaussian,
        &[mu.id(), sigma.id()],
        DoubleTensor::vector(lower.to_vec()),
        DoubleTensor::vector(upper.to_vec()),
    );
    builder.set_value(x, DoubleTensor::vector(vec![0.5, 0.0]));
    let x = x.id();
    let (graph, mut state) = builder.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let samples = Nuts::default()
        .with_adapt_count(500)
        .sample_with_statistics(
            &model,
            &mut state,
            &[x],
            4000,
            SamplingOptions::default()
                .with_drop(500)
                .with_down_sample_interval(1),
            &mut rng(23),
        )
        .unwrap()
        .0;
    let chain = samples.f64(x).unwrap();
    for i in 0..2 {
        let (a, b) = (lower[i], upper[i]);
        let xs = chain.element(&[i]).unwrap();
        assert!(xs.iter().all(|&x| x > a && x < b), "[{a}, {b}]");
        let z = cdf(b) - cdf(a);
        let m = (phi(a) - phi(b)) / z;
        let v = 1.0 + (x_phi(a) - x_phi(b)) / z - m * m;
        assert_moments(&xs, m, v, &format!("element {i} truncated to [{a}, {b}]"));
    }
}

/// A von Mises latent whose mean direction is near the ±π seam: NUTS moves through its circular transform across
/// the seam and recovers the circular mean and the mean resultant length I₁(κ)/I₀(κ).
#[test]
fn nuts_recovers_the_circular_mean_of_a_von_mises_latent() {
    let (mu, kappa) = (std::f64::consts::PI - 0.1, 10.0);
    let b = GraphBuilder::new();
    let theta = b.von_mises(mu, kappa);
    b.set_value(theta, DoubleTensor::scalar(0.0));
    let theta = theta.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let samples = Nuts::default()
        .with_adapt_count(500)
        .sample_with_statistics(
            &model,
            &mut state,
            &[theta],
            4000,
            SamplingOptions::default()
                .with_drop(500)
                .with_down_sample_interval(1),
            &mut rng(23),
        )
        .unwrap()
        .0;
    let xs = series(&samples, theta);
    let cos: Vec<f64> = xs.iter().map(|x| x.cos()).collect();
    let sin: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
    let circular_mean = mean(&sin).atan2(mean(&cos));
    // the angular distance, so a mean on the other side of the seam counts as close
    let error = (circular_mean - mu)
        .sin()
        .atan2((circular_mean - mu).cos())
        .abs();
    let deviations: Vec<f64> = xs.iter().map(|x| (x - mu).sin()).collect();
    let ess = effective_sample_size(&deviations).max(10.0);
    // the circular standard error is about sqrt(E[sin²(x - mu)] / ess) / R
    let resultant = (keanu_tensor::special::log_bessel_i1(kappa)
        - keanu_tensor::special::log_bessel_i0(kappa))
    .exp();
    let tolerance = 5.0 * (variance(&deviations) / ess).sqrt() / resultant;
    assert!(
        error < tolerance,
        "circular mean {circular_mean} vs {mu} (tolerance {tolerance})"
    );
    let r = mean(&cos).hypot(mean(&sin));
    assert!((r - resultant).abs() < 0.02, "resultant {r} vs {resultant}");
    // some samples fall on each side of the seam
    let (lo, hi) = xs
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        });
    assert!(lo < 0.0 && hi > 0.0, "{lo} {hi}");
}

/// mu ~ N(0, 3²) (two elements), sigma ~ Gamma(theta = 1, k = 2), x ~ N(mu, sigma) observed: a model with a
/// vector latent and a positive one, for checking chains step for step.
fn mixed_support_model() -> (Model, State, NodeId, NodeId) {
    let b = GraphBuilder::new();
    let mu = b.gaussian_with_shape(&[2], 0.0, 3.0);
    let sigma = b.gamma(1.0, 2.0);
    let x = b.gaussian_with_shape(&[2], mu, sigma);
    b.observe(x, DoubleTensor::vector(vec![0.8, -1.3]));
    b.set_value(mu, DoubleTensor::vector(vec![0.1, 0.2]));
    b.set_value(sigma, DoubleTensor::scalar(1.5));
    let (mu, sigma) = (mu.id(), sigma.id());
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    (model, state, mu, sigma)
}

/// `Space::Constrained` moves in the latents' own values. Where every latent is unbounded the transforms are the
/// identity, so it must give exactly the unconstrained chain. (This replaces a test that compared its chain bit for
/// bit with constants recorded from NUTS before transforms existed; NUTS's tree building and adaptation have since
/// changed, so those constants no longer describe anything.)
#[test]
fn nuts_in_constrained_space_matches_unconstrained_for_unbounded_latents() {
    let b = GraphBuilder::new();
    let mu = b.gaussian_with_shape(&[2], 0.0, 3.0);
    let x = b.gaussian_with_shape(&[2], mu, 1.5);
    b.observe(x, DoubleTensor::vector(vec![0.8, -1.3]));
    b.set_value(mu, DoubleTensor::vector(vec![0.1, 0.2]));
    let mu = mu.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let run = |space| {
        let samples = Nuts::default()
            .with_adapt_count(100)
            .with_space(space)
            .sample(&model, &mut state.clone(), &[mu], 200, &mut rng(42))
            .unwrap();
        (
            samples.f64(mu).unwrap().as_slice().to_vec(),
            samples.log_probs().to_vec(),
        )
    };
    assert_eq!(run(Space::Constrained), run(Space::Unconstrained));
}

/// With a bounded latent the two spaces give different chains, but the same posterior: their means agree within
/// Monte Carlo error.
#[test]
fn nuts_in_constrained_and_unconstrained_space_agree_on_the_posterior() {
    let (model, state, mu, sigma) = mixed_support_model();
    let run = |space| {
        let samples = Nuts::default()
            .with_adapt_count(500)
            .with_space(space)
            .sample_with_statistics(
                &model,
                &mut state.clone(),
                &[mu, sigma],
                4500,
                SamplingOptions::default().with_drop(500),
                &mut rng(42),
            )
            .unwrap()
            .0;
        let mu = samples.f64(mu).unwrap();
        [
            mu.element(&[0]).unwrap(),
            mu.element(&[1]).unwrap(),
            series(&samples, sigma),
        ]
    };
    let constrained = run(Space::Constrained);
    let unconstrained = run(Space::Unconstrained);
    assert!(constrained[2].iter().all(|&s| s > 0.0));
    for ((label, a), b) in ["mu[0]", "mu[1]", "sigma"]
        .iter()
        .zip(&constrained)
        .zip(&unconstrained)
    {
        let (ess_a, ess_b) = (effective_sample_size(a), effective_sample_size(b));
        let v = variance(b);
        let tolerance = 5.0 * (v / ess_a + v / ess_b).sqrt();
        assert!(
            (mean(a) - mean(b)).abs() < tolerance,
            "{label}: constrained mean {} vs unconstrained {} (tolerance {tolerance})",
            mean(a),
            mean(b)
        );
    }
}

/// The posterior mean and variance of sigma ~ HalfGaussian(1) given y_i ~ N(0, sigma), by quadrature of
/// p(σ | y) ∝ exp(−σ²/2) σ⁻ⁿ exp(−Σy² / 2σ²) over σ > 0.
fn half_gaussian_scale_posterior(data: &[f64]) -> (f64, f64) {
    let n = data.len() as f64;
    let s: f64 = data.iter().map(|y| y * y).sum();
    let log_density =
        |sigma: f64| -sigma * sigma / 2.0 - n * sigma.ln() - s / (2.0 * sigma * sigma);
    // the trapezoidal rule on (0, 12]: the density vanishes faster than any power at both ends
    let steps = 200_000;
    let h = 12.0 / steps as f64;
    let (mut z, mut m1, mut m2) = (0.0, 0.0, 0.0);
    for i in 1..=steps {
        let sigma = i as f64 * h;
        let w = if i == steps { 0.5 } else { 1.0 } * log_density(sigma).exp();
        z += w;
        m1 += w * sigma;
        m2 += w * sigma * sigma;
    }
    let mean = m1 / z;
    (mean, m2 / z - mean * mean)
}

/// AE1: sampled in unconstrained space, a positive scale never leaves its support, no trajectory diverges, and the
/// draws match the posterior computed by quadrature.
#[test]
fn nuts_samples_a_half_gaussian_scale_inside_its_support() {
    let data = [0.4, -1.1, 0.7, 1.6, -0.3];
    let b = GraphBuilder::new();
    let sigma = b.half_gaussian(1.0);
    let y = b.gaussian_with_shape(&[data.len()], 0.0, sigma);
    b.observe(y, DoubleTensor::vector(data.to_vec()));
    b.set_value(sigma, DoubleTensor::scalar(1.0));
    let sigma = sigma.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let (samples, statistics) = Nuts::default()
        .with_adapt_count(500)
        .sample_with_statistics(
            &model,
            &mut state,
            &[sigma],
            5000,
            SamplingOptions::default()
                .with_drop(500)
                .with_down_sample_interval(1),
            &mut rng(11),
        )
        .unwrap();
    let draws = series(&samples, sigma);
    assert!(draws.iter().all(|&s| s > 0.0));
    assert!(
        statistics.divergent_steps.is_empty(),
        "{:?}",
        statistics.divergent_steps
    );
    // the recorded log probabilities are the model's at the constrained values, without the Jacobian
    let mut at = state.clone();
    for sample in samples.iter().take(3) {
        at.set_and_cascade(model.graph(), sigma, sample.f64(sigma).unwrap().clone())
            .unwrap();
        let expected = model.log_prob(&at).unwrap();
        assert!(
            (sample.log_prob() - expected).abs() < 1e-9,
            "{} vs {expected}",
            sample.log_prob()
        );
    }
    let (m, v) = half_gaussian_scale_posterior(&data);
    assert_moments(&draws, m, v, "HalfGaussian scale NUTS");
}

/// p ~ Dirichlet(1, 2, 3), k ~ Multinomial(20, p) observed: the posterior is Dirichlet(α + k). In constrained
/// space every move would leave the simplex; in unconstrained space every draw is on it.
#[test]
fn nuts_finds_the_dirichlet_multinomial_posterior() {
    let alpha = [1.0, 2.0, 3.0];
    let counts = [4i64, 7, 9];
    let b = GraphBuilder::new();
    let p = b.dirichlet(DoubleTensor::vector(alpha.to_vec()));
    let k = b.multinomial(counts.iter().sum::<i64>(), p);
    b.observe(k, IntTensor::vector(counts.to_vec()));
    b.set_value(p, DoubleTensor::vector(vec![1.0 / 3.0; 3]));
    let p = p.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let samples = Nuts::default()
        .with_adapt_count(500)
        .sample_with_statistics(
            &model,
            &mut state,
            &[p],
            4000,
            SamplingOptions::default()
                .with_drop(500)
                .with_down_sample_interval(1),
            &mut rng(12),
        )
        .unwrap()
        .0;
    let chain = samples.f64(p).unwrap();
    for draw in chain.as_slice() {
        assert!(draw.as_slice().iter().all(|&x| x > 0.0));
        assert!((draw.sum() - 1.0).abs() < 1e-12);
    }
    let posterior: Vec<f64> = alpha
        .iter()
        .zip(&counts)
        .map(|(a, &k)| a + k as f64)
        .collect();
    let total: f64 = posterior.iter().sum();
    for (i, a) in posterior.iter().enumerate() {
        let m = a / total;
        let v = a * (total - a) / (total * total * (total + 1.0));
        assert_moments(
            &chain.element(&[i]).unwrap(),
            m,
            v,
            &format!("Dirichlet p[{i}]"),
        );
    }
}

/// A start outside the support has no unconstrained point; in either space it is an impossible start.
#[test]
fn nuts_rejects_a_start_outside_the_support() {
    let b = GraphBuilder::new();
    let a = b.uniform(0.0, 1.0);
    b.set_value(a, DoubleTensor::scalar(5.0));
    let a = a.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    for space in [Space::Unconstrained, Space::Constrained] {
        assert_eq!(
            Nuts::default()
                .with_space(space)
                .sampler(&model, &mut state.clone(), &[a], &mut rng(1))
                .unwrap_err(),
            AlgorithmError::ImpossibleStart,
            "{space:?}"
        );
    }
}

/// A bound that moves with another latent has no fixed transform: the error says so and suggests the opt-out,
/// which still samples.
#[test]
fn nuts_reports_a_support_that_depends_on_a_latent() {
    let b = GraphBuilder::new();
    let hi = b.uniform(1.0, 2.0);
    let x = b.uniform(0.0, hi);
    b.set_value(hi, DoubleTensor::scalar(1.5));
    b.set_value(x, DoubleTensor::scalar(0.5));
    let x = x.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let error = Nuts::default()
        .sampler(&model, &mut state.clone(), &[x], &mut rng(1))
        .unwrap_err();
    assert!(error.to_string().contains("Space::Constrained"), "{error}");
    Nuts::default()
        .with_adapt_count(20)
        .with_space(Space::Constrained)
        .sample(&model, &mut state.clone(), &[x], 20, &mut rng(1))
        .unwrap();
}

/// Latent draws of a two-component Gaussian mixture whose weights are a latent Dirichlet, each observed with noise:
/// the weights don't move the mixture's support, so NUTS runs in unconstrained space and every draw of the weights
/// is on the simplex.
#[test]
fn nuts_samples_latent_mixture_weights() {
    use keanu_graph::DoubleVar;
    use keanu_graph::op::DistributionKind;
    let b = GraphBuilder::new();
    let w = b.dirichlet(DoubleTensor::vector(vec![1.0, 1.0]));
    let mu = b.constant(DoubleTensor::vector(vec![-2.0, 2.0]));
    let sigma = b.scalar(1.0);
    let data = [1.5, 2.5, -1.8, 2.2];
    let x: DoubleVar = b.mixture_with_shape(
        &[data.len()],
        DistributionKind::Gaussian,
        &[mu.id(), sigma.id()],
        w,
    );
    let y = b.gaussian(x, 0.3);
    b.observe(y, DoubleTensor::vector(data.to_vec()));
    b.set_value(w, DoubleTensor::vector(vec![0.5, 0.5]));
    b.set_value(x, DoubleTensor::vector(data.to_vec()));
    let w = w.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let samples = Nuts::default()
        .with_adapt_count(100)
        .sample(&model, &mut state, &[w], 400, &mut rng(31))
        .unwrap();
    let chain = samples.f64(w).unwrap();
    for draw in chain.as_slice() {
        assert!(draw.as_slice().iter().all(|&x| x > 0.0));
        assert!((draw.sum() - 1.0).abs() < 1e-12);
    }
    // three of the four observations favour the second component
    let second = chain.element(&[1]).unwrap();
    assert!(mean(&second) > 0.5, "{}", mean(&second));
}
