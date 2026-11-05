//! NUTS against analytic posteriors: adaptation, settings, divergences, energies and E-BFMI, and its trajectories.

mod mcmc_common;

use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::chains::run_chains;
use keanu_algorithms::mcmc::nuts::{Nuts, QuadraticPotential};
use keanu_algorithms::statistics::effective_sample_size;
use keanu_algorithms::{AlgorithmError, diagnostics};
use keanu_graph::{GraphBuilder, Model, NodeId, State};
use keanu_tensor::{DoubleTensor, IntTensor};
use mcmc_common::*;

#[test]
fn nuts_finds_the_gaussian_posterior() {
    let (model, mut state, mu, m, v) = gaussian_model();
    let nuts = Nuts::default().with_adapt_count(500);
    let samples = nuts
        .sample_with_statistics(
            &model,
            &mut state,
            &[mu],
            5000,
            SamplingOptions::default()
                .with_drop(500)
                .with_down_sample_interval(1),
            &mut rng(4),
        )
        .unwrap()
        .0;
    assert_moments(&series(&samples, mu), m, v, "NUTS");
}

/// Regression: a NUTS configuration (in Java, its adaptive potential) could not be reused for a second run.
#[test]
fn nuts_can_be_reused() {
    let (model, state, mu, ..) = gaussian_model();
    let nuts = Nuts::default().with_adapt_count(100);
    let mut r = rng(14);
    nuts.sample(&model, &mut state.clone(), &[mu], 200, &mut r)
        .unwrap();
    let second = nuts
        .sample(&model, &mut state.clone(), &[mu], 200, &mut r)
        .unwrap();
    let values = series(&second, mu);
    assert!(values.iter().all(|v| v.is_finite()));
    let mut distinct = values.clone();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    assert!(
        distinct.len() > 50,
        "only {} distinct values",
        distinct.len()
    );
}

/// Regression: the first sample after adaptation must already use the averaged step size, which then stays fixed.
#[test]
fn nuts_switches_to_the_averaged_step_size_when_adaptation_ends() {
    let (model, mut state, mu, ..) = gaussian_model();
    let adapt_count = 50;
    let nuts = Nuts::default()
        .with_adapt_count(adapt_count)
        .with_save_statistics(true);
    let (_, statistics) = nuts
        .sample_with_statistics(
            &model,
            &mut state,
            &[mu],
            80,
            SamplingOptions::default(),
            &mut rng(15),
        )
        .unwrap();
    let after = &statistics.step_size[adapt_count..];
    assert!(
        after.iter().all(|&s| s == after[0]),
        "step sizes after adaptation: {after:?}"
    );
    assert_ne!(statistics.step_size[adapt_count - 1], after[0]);
    assert_eq!(statistics.tree_size.len(), 80);
    assert_eq!(statistics.log_prob.len(), 80);
    assert!(
        statistics
            .mean_tree_accept
            .iter()
            .all(|a| (0.0..=1.0).contains(a))
    );
    // as documented: each entry is the previous tree's acceptance (none before the first), frozen once adaptation
    // ends at the last adapted tree's (step adapt_count - 1, recorded at step adapt_count)
    let accept = &statistics.mean_tree_accept;
    assert_eq!(accept[0], 0.0);
    assert_ne!(accept[1], 0.0);
    assert!(
        accept[adapt_count..]
            .iter()
            .all(|&a| a == accept[adapt_count]),
        "{:?}",
        &accept[adapt_count..]
    );
    assert_ne!(accept[adapt_count - 1], accept[adapt_count]);
}

#[test]
fn nuts_records_divergent_steps_after_adaptation() {
    let (model, mut state, mu, ..) = gaussian_model();
    let nuts = Nuts::default()
        .with_adapt_count(0)
        .with_adapt_step_size(false)
        .with_initial_step_size(5.0)
        .with_max_energy_change(1e-3);
    let (_, statistics) = nuts
        .sample_with_statistics(
            &model,
            &mut state,
            &[mu],
            50,
            SamplingOptions::default(),
            &mut rng(16),
        )
        .unwrap();
    assert!(!statistics.divergent_steps.is_empty());
}

#[test]
fn nuts_rejects_invalid_settings_and_discrete_latents() {
    let (model, mut state, mu, ..) = gaussian_model();
    let bad = Nuts::default().with_max_tree_height(0);
    assert!(matches!(
        bad.sampler(&model, &mut state, &[mu], &mut rng(1)),
        Err(AlgorithmError::InvalidSetting(_))
    ));

    let b = GraphBuilder::new();
    let k = b.uniform_int(0, 4);
    let x = b.poisson(k.to_double() + 1.0);
    b.observe(x, IntTensor::scalar(2));
    let k = k.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert!(matches!(
        Nuts::default().sampler(&model, &mut state, &[k], &mut rng(1)),
        Err(AlgorithmError::DiscreteLatents(_))
    ));
}

/// The potential's initial mean and weight belonged to the earlier adaptation and are ignored.
#[test]
#[allow(deprecated)]
fn nuts_ignores_the_deprecated_potential_settings() {
    let (model, state, mu, ..) = gaussian_model();
    let run = |potential| {
        let samples = Nuts::default()
            .with_adapt_count(100)
            .with_potential(potential)
            .sample(&model, &mut state.clone(), &[mu], 200, &mut rng(5))
            .unwrap();
        series(&samples, mu)
    };
    let ignored = QuadraticPotential::default()
        .with_initial_weight(0.0)
        .with_initial_mean(f64::NAN);
    assert_eq!(run(ignored), run(QuadraticPotential::default()));
}

/// Regression: a non-positive initial variance gave a zero mass matrix entry, so momentum / mass was NaN and the
/// chain never moved. (The initial weight could do the same, but is now ignored.)
#[test]
fn nuts_rejects_a_degenerate_initial_potential() {
    let (model, mut state, mu, ..) = gaussian_model();
    for potential in [
        QuadraticPotential::default().with_initial_variance(0.0),
        QuadraticPotential::default().with_initial_variance(-1.0),
        QuadraticPotential::default().with_initial_variance(f64::NAN),
        QuadraticPotential::default().with_initial_variance(f64::INFINITY),
        QuadraticPotential::default().with_adaptation_window(1),
    ] {
        let nuts = Nuts::default().with_potential(potential);
        assert!(
            matches!(
                nuts.sampler(&model, &mut state, &[mu], &mut rng(1)),
                Err(AlgorithmError::InvalidSetting(_))
            ),
            "{potential:?}"
        );
    }
}

const SCHOOL_EFFECTS: [f64; 8] = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0];
const SCHOOL_SIGMAS: [f64; 8] = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0];

/// Eight schools (Rubin 1981): mu ~ N(0, 5), tau ~ HalfCauchy(5), theta ~ N(mu, tau), y ~ N(theta, sigma). The
/// centred form samples theta directly, giving the funnel between theta and tau; the non-centred form samples
/// eta ~ N(0, 1) and sets theta = mu + tau·eta. Returns the model, its state and the latents.
fn eight_schools(centred: bool) -> (Model, State, Vec<NodeId>) {
    let schools = SCHOOL_EFFECTS.len();
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 5.0);
    let tau = b.half_cauchy(5.0);
    b.set_value(mu, DoubleTensor::scalar(0.0));
    b.set_value(tau, DoubleTensor::scalar(1.0));
    let (theta, latents) = if centred {
        let theta = b.gaussian_with_shape(&[schools], mu, tau);
        b.set_value(theta, DoubleTensor::vector(vec![0.0; schools]));
        (theta, vec![mu.id(), tau.id(), theta.id()])
    } else {
        let eta = b.gaussian_with_shape(&[schools], 0.0, 1.0);
        b.set_value(eta, DoubleTensor::vector(vec![0.0; schools]));
        (mu + tau * eta, vec![mu.id(), tau.id(), eta.id()])
    };
    let sigma = b.constant(DoubleTensor::vector(SCHOOL_SIGMAS.to_vec()));
    let y = b.gaussian_with_shape(&[schools], theta, sigma);
    b.observe(y, DoubleTensor::vector(SCHOOL_EFFECTS.to_vec()));
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    (model, state, latents)
}

/// The divergences of four NUTS chains of eight schools after adaptation, and each chain's draw count and
/// energy series length.
fn eight_schools_divergences(centred: bool) -> (usize, Vec<(usize, usize)>) {
    let (model, state, latents) = eight_schools(centred);
    let (adapt, draws) = (500, 500);
    let nuts = Nuts::default()
        .with_adapt_count(adapt)
        .with_target_acceptance_prob(0.9);
    let chains = run_chains(4, 8, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &latents,
            adapt + draws,
            SamplingOptions::default().with_drop(adapt),
            rng,
        )
    })
    .unwrap();
    let divergences = chains.iter().map(|(_, s)| s.divergence_count()).sum();
    let lengths = chains
        .iter()
        .map(|(samples, s)| (samples.len(), s.energy[adapt..].len()))
        .collect();
    (divergences, lengths)
}

#[test]
fn centred_eight_schools_diverges_and_non_centred_does_not() {
    let (centred, _) = eight_schools_divergences(true);
    assert!(centred > 0, "centred eight schools reported no divergences");
    let (non_centred, lengths) = eight_schools_divergences(false);
    assert_eq!(non_centred, 0, "non-centred eight schools diverged");
    for (draws, energies) in lengths {
        assert_eq!(draws, energies);
    }
}

#[test]
fn nuts_records_an_energy_per_step_and_flags_divergences() {
    let (model, mut state, mu, ..) = gaussian_model();
    let nuts = Nuts::default()
        .with_adapt_count(0)
        .with_adapt_step_size(false)
        .with_initial_step_size(5.0)
        .with_max_energy_change(1e-3);
    let (samples, statistics) = nuts
        .sample_with_statistics(
            &model,
            &mut state,
            &[mu],
            50,
            SamplingOptions::default(),
            &mut rng(16),
        )
        .unwrap();
    assert_eq!(statistics.energy.len(), samples.len());
    assert!(statistics.energy.iter().all(|e| e.is_finite()));
    let flags = statistics.divergent_flags();
    assert_eq!(flags.len(), samples.len());
    assert_eq!(
        flags.iter().filter(|&&d| d).count(),
        statistics.divergence_count()
    );
    assert!(statistics.divergence_count() > 0);
    for &step in &statistics.divergent_steps {
        assert!(flags[step]);
    }
}

#[test]
fn e_bfmi_of_a_standard_gaussian_is_healthy() {
    let b = GraphBuilder::new();
    let x = b.gaussian_with_shape(&[10], 0.0, 1.0);
    b.set_value(x, DoubleTensor::vector(vec![0.0; 10]));
    let x = x.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let (adapt, draws) = (500, 2000);
    let (samples, statistics) = Nuts::default()
        .with_adapt_count(adapt)
        .sample_with_statistics(
            &model,
            &mut state,
            &[x],
            adapt + draws,
            SamplingOptions::default().with_drop(adapt),
            &mut rng(21),
        )
        .unwrap();
    assert_eq!(statistics.energy.len(), adapt + samples.len());
    let e_bfmi = diagnostics::e_bfmi(&statistics.energy[adapt..]);
    assert!(e_bfmi > diagnostics::E_BFMI_THRESHOLD, "E-BFMI {e_bfmi}");
}

#[test]
fn e_bfmi_is_nan_for_too_few_or_degenerate_energies() {
    assert!(diagnostics::e_bfmi(&[]).is_nan());
    assert!(diagnostics::e_bfmi(&[1.0]).is_nan());
    assert!(diagnostics::e_bfmi(&[2.0, 2.0, 2.0]).is_nan());
    assert!(diagnostics::e_bfmi(&[1.0, f64::NAN, 2.0]).is_nan());
    assert!(diagnostics::e_bfmi(&[1.0, f64::INFINITY, 2.0]).is_nan());
    // alternating energies: three successive differences of ±2 over four unit deviations, 12 / 4
    let e = diagnostics::e_bfmi(&[1.0, -1.0, 1.0, -1.0]);
    assert!((e - 3.0).abs() < 1e-12, "{e}");
    // a slow drift: successive differences are small next to the spread
    let drift: Vec<f64> = (0..100).map(f64::from).collect();
    assert!(diagnostics::e_bfmi(&drift) < 0.01);
}

/// The AR(1) model x_t = φ x_{t-1} + ε_t, φ ~ N(0.5, 0.3²), ε ~ N(0, 1), y_t ~ N(x_t, 0.5) observed, with x built by
/// a scan or unrolled. Returns the model, its state and (φ, ε).
fn ar1_model(scan: bool) -> (Model, State, NodeId, NodeId) {
    let data = [0.4, 0.9, 0.2, -0.5, -0.1];
    let b = GraphBuilder::new();
    let phi = b.gaussian(0.5, 0.3);
    let eps = b.gaussian_with_shape(&[data.len()], 0.0, 1.0);
    let x0 = b.scalar(0.2);
    let x = if scan {
        b.scan(x0, &[eps], &[phi], |_, x, e, c| c[0] * x + e[0])
    } else {
        let mut x = x0;
        let items: Vec<_> = (0..data.len())
            .map(|t| {
                x = phi * x + eps.take(&[t]);
                x
            })
            .collect();
        b.stack(0, &items)
    };
    let y = b.gaussian(x, 0.5);
    b.observe(y, DoubleTensor::vector(data.to_vec()));
    b.set_value(phi, DoubleTensor::scalar(0.5));
    b.set_value(eps, DoubleTensor::vector(vec![0.0; data.len()]));
    let (phi, eps) = (phi.id(), eps.id());
    let (graph, state) = b.build().unwrap();
    (Model::new(graph, &state).unwrap(), state, phi, eps)
}

#[test]
fn nuts_on_a_scan_model_matches_the_unrolled_model() {
    let run = |scan: bool, seed: u64| {
        let (model, mut state, phi, eps) = ar1_model(scan);
        let samples = Nuts::default()
            .with_adapt_count(300)
            .sample_with_statistics(
                &model,
                &mut state,
                &[phi, eps],
                3000,
                SamplingOptions::default()
                    .with_drop(300)
                    .with_down_sample_interval(1),
                &mut rng(seed),
            )
            .unwrap()
            .0;
        let eps0 = samples.f64(eps).unwrap().element(&[0]).unwrap();
        (series(&samples, phi), eps0)
    };
    // the scan's log probability and gradient are bit for bit the unrolled model's, so with the same seed the
    // chains are identical
    let (phi_scan, eps_scan) = run(true, 21);
    let (phi_unrolled, eps_unrolled) = run(false, 21);
    assert_eq!(phi_scan, phi_unrolled);
    assert_eq!(eps_scan, eps_unrolled);
    // and an independent unrolled chain agrees on the posterior means within Monte Carlo error
    let (phi_other, eps_other) = run(false, 22);
    for (label, a, b) in [
        ("phi", &phi_scan, &phi_other),
        ("eps[0]", &eps_scan, &eps_other),
    ] {
        let (ess_a, ess_b) = (effective_sample_size(a), effective_sample_size(b));
        let v = variance(b);
        let tolerance = 5.0 * (v / ess_a + v / ess_b).sqrt();
        assert!(
            (mean(a) - mean(b)).abs() < tolerance,
            "{label}: scan mean {} vs unrolled {} (tolerance {tolerance})",
            mean(a),
            mean(b)
        );
    }
}

/// Regression: without the checks across subtree boundaries that Stan added in 2.21, the no-U-turn criterion could
/// miss a U-turn when a trajectory's two halves together nearly complete an orbit, so that their summed momentum
/// almost cancels, and the tree then kept doubling. On a 100-dimensional standard Gaussian with the identity metric
/// and step size 0.2, an orbit takes 2π / 0.2 ≈ 31 leapfrog steps; about a fifth of the trees ran to the maximum of
/// 1023 steps. With the checks every tree stops within two orbits.
#[test]
fn nuts_trees_stop_within_two_orbits_of_a_standard_gaussian() {
    let d = 100;
    let b = GraphBuilder::new();
    let x = b.gaussian_with_shape(&[d], 0.0, 1.0);
    b.set_value(x, DoubleTensor::vector(vec![0.5; d]));
    let x = x.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let (_, statistics) = Nuts::default()
        .with_adapt_count(0)
        .with_adapt_step_size(false)
        .with_adapt_potential(false)
        .with_initial_step_size(0.2)
        .with_save_statistics(true)
        .sample_with_statistics(
            &model,
            &mut state,
            &[x],
            300,
            SamplingOptions::default(),
            &mut rng(3),
        )
        .unwrap();
    let longest = statistics.tree_size.iter().max().unwrap();
    assert!(*longest <= 63, "a tree took {longest} leapfrog steps");
}
