//! Pareto-smoothed importance sampling: k̂ for draws from known generalized Pareto tails, agreement with
//! arviz-stats on fixed vectors, the invariants of the smoothing, and the edge cases.

use keanu_algorithms::AlgorithmError;
use keanu_algorithms::psis::{GeneralizedPareto, gpdfit, khat_threshold, psis, tail_length};
use keanu_distributions::KeanuRng;

/// `n` draws of a generalized Pareto distribution with shape `k` and scale 1, by inversion.
fn pareto_draws(k: f64, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = KeanuRng::seed_from_u64(seed);
    let d = GeneralizedPareto::new(k, 1.0);
    (0..n).map(|_| d.quantile(rng.next_f64())).collect()
}

/// The fixture's deterministic sequence u(i, a, p) = ((a·i + 13) mod p + ½) / p.
fn u(i: usize, a: usize, p: usize) -> f64 {
    ((i * a + 13) % p) as f64 / p as f64 + 0.5 / p as f64
}

/// k̂ of log ratios whose ratios are generalized Pareto with shape 0.3 or 0.8: the tail above any threshold of a
/// generalized Pareto distribution is generalized Pareto with the same shape, so k̂ estimates it. With 50 000 draws
/// the tail has 671, and the standard error of k̂ is about (1 + k) / √671 ≈ 0.05–0.07.
#[test]
fn khat_recovers_known_pareto_shapes() {
    for (k, seed) in [(0.3, 1), (0.8, 2)] {
        let log_ratios: Vec<f64> = pareto_draws(k, 50_000, seed)
            .iter()
            .map(|x| x.ln())
            .collect();
        let result = psis(&log_ratios, 1.0).unwrap();
        assert_eq!(result.tail_length(), 671);
        assert!(
            (result.pareto_k() - k).abs() < 0.1,
            "k = {k}: k̂ = {}",
            result.pareto_k()
        );
        assert_eq!(result.is_reliable(), k < 0.7);
    }
}

/// The fit itself on 5000 generalized Pareto draws: the standard error of the shape is about (1 + k) / √5000 ≈ 0.02,
/// and that of the scale about 2.5%.
#[test]
fn gpdfit_recovers_shape_and_scale() {
    for (k, seed) in [(-0.2, 3), (0.3, 4), (0.8, 5)] {
        let draws: Vec<f64> = pareto_draws(k, 5000, seed)
            .iter()
            .map(|x| 2.0 * x)
            .collect();
        let fit = gpdfit(&draws);
        assert!((fit.k - k).abs() < 0.07, "k = {k}: {fit:?}");
        assert!((fit.sigma / 2.0 - 1.0).abs() < 0.1, "k = {k}: {fit:?}");
    }
    // the order of the values does not matter
    let mut draws = pareto_draws(0.5, 500, 6);
    let fit = gpdfit(&draws);
    draws.reverse();
    assert_eq!(gpdfit(&draws), fit);
}

/// `tests/data/arviz_loo.json` was generated offline by `tests/data/arviz_loo.py` with arviz-stats 1.3.3, whose
/// `psislw` follows the same algorithm as `loo` (it computes the tail length as ⌊3·√S⌋, which equals `loo`'s
/// ⌈3·√S⌉ for the S = 1600 used). Its weights are normalised.
#[test]
fn psis_matches_arviz() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/arviz_loo.json");
    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cases = fixture["psis"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let log_ratios: Vec<f64> = (0..1600)
            .map(|i| match name {
                "gpd_shape_0_6" => (((1.0 - u(i, 7919, 1601)).powf(-0.6) - 1.0) / 0.6).ln(),
                "bounded_trig" => 2.0 * (i as f64 * 0.37).sin() + 0.5 * (i as f64 * 1.3).cos(),
                other => panic!("unknown case {other}"),
            })
            .collect();
        let result = psis(&log_ratios, 1.0).unwrap();
        let expected = case["pareto_k"].as_f64().unwrap();
        assert!(
            (result.pareto_k() - expected).abs() < 1e-9,
            "{name}: k̂ {} vs ArviZ {expected}",
            result.pareto_k()
        );
        let normalized = result.normalized_log_weights();
        for (index, value) in case["normalized_log_weights"].as_object().unwrap() {
            let index: usize = index.parse().unwrap();
            let expected = value.as_f64().unwrap();
            assert!(
                (normalized[index] - expected).abs() < 1e-9,
                "{name}: log weight {index} {} vs ArviZ {expected}",
                normalized[index]
            );
        }
    }
}

/// Smoothing replaces only the tail, keeps the order of the weights, never raises a weight above the largest raw
/// one, and does not depend on the scale of the ratios.
#[test]
fn smoothing_invariants() {
    let log_ratios: Vec<f64> = pareto_draws(0.6, 4000, 7).iter().map(|x| x.ln()).collect();
    let result = psis(&log_ratios, 1.0).unwrap();
    let weights = result.log_weights();
    let tail = tail_length(4000, 1.0);
    let mut order: Vec<usize> = (0..log_ratios.len()).collect();
    order.sort_by(|&a, &b| log_ratios[a].total_cmp(&log_ratios[b]));
    // outside the tail the log weights are the log ratios (to the rounding of shifting by the largest and back)
    for &i in &order[..4000 - tail] {
        assert!((weights[i] - log_ratios[i]).abs() < 1e-12);
    }
    // in the raw ratios' order the smoothed weights never decrease
    for pair in order.windows(2) {
        assert!(weights[pair[0]] <= weights[pair[1]]);
    }
    let raw_max = log_ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(weights.iter().all(|&w| w <= raw_max));
    // the tail did change
    assert!(
        order[4000 - tail..]
            .iter()
            .any(|&i| (weights[i] - log_ratios[i]).abs() > 1e-3)
    );
    let total: f64 = result.weights().iter().sum();
    assert!((total - 1.0).abs() < 1e-12);

    // shifting the log ratios shifts the log weights and leaves k̂
    let shifted: Vec<f64> = log_ratios.iter().map(|r| r - 50.0).collect();
    let moved = psis(&shifted, 1.0).unwrap();
    assert!((moved.pareto_k() - result.pareto_k()).abs() < 1e-9);
    for (a, b) in moved
        .normalized_log_weights()
        .iter()
        .zip(result.normalized_log_weights())
    {
        assert!((a - b).abs() < 1e-9);
    }
}

/// A lower relative efficiency lengthens the tail.
#[test]
fn relative_efficiency_lengthens_the_tail() {
    let log_ratios: Vec<f64> = pareto_draws(0.3, 4000, 8).iter().map(|x| x.ln()).collect();
    assert_eq!(psis(&log_ratios, 1.0).unwrap().tail_length(), 190);
    assert_eq!(psis(&log_ratios, 0.5).unwrap().tail_length(), 269);
}

#[test]
fn constant_and_short_inputs() {
    // equal ratios have a bounded tail: k̂ is −∞ and the weights are uniform
    let result = psis(&[1.5; 1000], 1.0).unwrap();
    assert_eq!(result.pareto_k(), f64::NEG_INFINITY);
    assert!(result.is_reliable());
    assert!(result.weights().iter().all(|w| (w - 1e-3).abs() < 1e-15));
    // 20 draws have a tail of 4, too short to fit: k̂ is +∞ and the weights are the raw ratios
    let short: Vec<f64> = (0..20).map(|i| (i as f64 * 0.7).sin()).collect();
    let result = psis(&short, 1.0).unwrap();
    assert_eq!(result.pareto_k(), f64::INFINITY);
    assert!(!result.is_reliable());
    for (w, r) in result.log_weights().iter().zip(&short) {
        assert!((w - r).abs() < 1e-15);
    }
    // −∞ ratios are zero weights
    let mut with_zero = short.clone();
    with_zero[3] = f64::NEG_INFINITY;
    assert_eq!(psis(&with_zero, 1.0).unwrap().weights()[3], 0.0);
}

#[test]
fn thresholds() {
    assert_eq!(khat_threshold(4000), 0.7);
    // 1 − 1 / log₁₀(1000)
    assert!((khat_threshold(1000) - 2.0 / 3.0).abs() < 1e-15);
}

#[test]
fn invalid_inputs_are_rejected() {
    let invalid = |log_ratios: &[f64], r_eff: f64| {
        assert!(matches!(
            psis(log_ratios, r_eff),
            Err(AlgorithmError::InvalidArgument(_))
        ));
    };
    invalid(&[], 1.0);
    invalid(&[0.0, f64::NAN], 1.0);
    invalid(&[0.0, f64::INFINITY], 1.0);
    invalid(&[f64::NEG_INFINITY; 3], 1.0);
    invalid(&[0.0, 1.0], 0.0);
    invalid(&[0.0, 1.0], f64::NAN);
}
