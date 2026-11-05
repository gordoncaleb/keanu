//! PSIS-LOO, WAIC and model comparison: agreement with arviz-stats on fixed log likelihoods, with the exact
//! leave-one-out predictive density of a conjugate model, the ranking of a correct and a wrong model, the weights,
//! the warnings and the errors.

use keanu_algorithms::AlgorithmError;
use keanu_algorithms::loglik::PointwiseLogLik;
use keanu_algorithms::loo::{
    ElpdEstimate, Loo, compare, loo, loo_from_values, pseudo_bma_weights, relative_efficiency,
    stacking_weights, waic, waic_from_values,
};
use keanu_algorithms::mcmc::chains::sample_chains;
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_algorithms::psis::GeneralizedPareto;
use keanu_distributions::KeanuRng;
use keanu_graph::{GraphBuilder, Model};
use keanu_tensor::DoubleTensor;
use std::f64::consts::PI;

/// The fixture's deterministic sequence u(i, a, p) = ((a·i + 13) mod p + ½) / p.
fn u(i: usize, a: usize, p: usize) -> f64 {
    ((i * a + 13) % p) as f64 / p as f64 + 0.5 / p as f64
}

fn normal_log_density(y: f64, mean: f64, sd: f64) -> f64 {
    -0.5 * (2.0 * PI).ln() - sd.ln() - 0.5 * ((y - mean) / sd).powi(2)
}

/// The fixture's two log likelihoods of six observations at 1600 draws, as one chain `[chain][draw][point]`.
fn fixture_values(model: &str) -> Vec<Vec<Vec<f64>>> {
    let y = [-0.4, 0.1, 0.3, 0.8, 1.2, 4.0];
    let draws = (0..1600)
        .map(|s| {
            let m = 0.2 + 0.8 * (u(s, 7919, 1601) - 0.5);
            let sd = match model {
                "narrow" => 0.8 + 0.4 * u(s, 104_729, 1607),
                "wide" => 2.0,
                other => panic!("unknown model {other}"),
            };
            y.iter().map(|&y| normal_log_density(y, m, sd)).collect()
        })
        .collect();
    vec![draws]
}

fn names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("y[{i}]")).collect()
}

fn close(what: &str, actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: {actual} vs {expected}"
    );
}

/// `tests/data/arviz_loo.json` was generated offline by `tests/data/arviz_loo.py` with arviz-stats 1.3.3 (PSIS by
/// `psislw` with r_eff = 1, on 1600 draws so that its tail length ⌊3·√S⌋ equals `loo`'s ⌈3·√S⌉, then elpdᵢ = logsumexp(log weights + log likelihood)); its stacking weights are computed
/// as arviz-stats' `compare` computes them, with SciPy's SLSQP.
#[test]
fn loo_matches_arviz() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/arviz_loo.json");
    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut fits = Vec::new();
    for case in fixture["loo"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let result = loo_from_values(&fixture_values(name), &names(6), Some(&[1.0; 6])).unwrap();
        close(name, result.elpd, case["elpd"].as_f64().unwrap(), 1e-9);
        close(name, result.se, case["se"].as_f64().unwrap(), 1e-9);
        close(name, result.p, case["p_loo"].as_f64().unwrap(), 1e-9);
        let elpd_i = case["elpd_i"].as_array().unwrap();
        let k = case["pareto_k"].as_array().unwrap();
        for (i, point) in result.points.iter().enumerate() {
            close(name, point.elpd, elpd_i[i].as_f64().unwrap(), 1e-9);
            close(name, point.pareto_k, k[i].as_f64().unwrap(), 1e-9);
        }
        assert_eq!(result.looic, -2.0 * result.elpd);
        assert!(result.warnings.is_empty());
        fits.push(result);
    }
    let weights = stacking_weights(&[&fits[0], &fits[1]]).unwrap();
    let expected = fixture["stacking_weights"].as_array().unwrap();
    for (w, e) in weights.iter().zip(expected) {
        close("stacking weight", *w, e.as_f64().unwrap(), 1e-5);
    }
}

/// y ~ N(mu, 1) with mu ~ N(0, 10) for eight observations, sampled by NUTS, and the same model with the wrong
/// standard deviation 0.3.
fn conjugate_fit(sigma: f64) -> (Loo, PointwiseLogLik) {
    let data = vec![0.3, -0.8, 1.6, 0.9, 0.1, 2.4, -0.2, 1.1];
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 10.0);
    let y = b.gaussian_with_shape(&[8], mu, sigma).label("y");
    b.observe(y, DoubleTensor::vector(data));
    let mu = mu.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let nuts = Nuts::default().with_adapt_count(500);
    let chains = sample_chains(4, 11, &state, |state, rng| {
        nuts.sample(&model, state, &[mu], 1500, rng)
            .and_then(|s| s.drop(500))
    })
    .unwrap();
    let loglik = PointwiseLogLik::from_chains(&model, &state, &chains).unwrap();
    (loo(&loglik).unwrap(), loglik)
}

/// The exact leave-one-out predictive density of the conjugate model: without point i the posterior of mu is
/// normal with precision 1/100 + (n − 1) and mean Σⱼ≠ᵢ yⱼ over it, so yᵢ is predicted by a normal with that mean
/// and variance 1 + 1 / precision.
#[test]
fn loo_matches_the_exact_leave_one_out_density() {
    let data = [0.3, -0.8, 1.6, 0.9, 0.1, 2.4, -0.2, 1.1];
    let (result, loglik) = conjugate_fit(1.0);
    assert_eq!(result.draws, 4000);
    assert_eq!(result.points.len(), 8);
    assert_eq!(result.points[2].name, "y[2]");
    let total: f64 = data.iter().sum();
    let mut exact = 0.0;
    for (i, &y) in data.iter().enumerate() {
        let precision = 0.01 + 7.0;
        let mean = (total - y) / precision;
        let pointwise = normal_log_density(y, mean, (1.0 + 1.0 / precision).sqrt());
        let point = &result.points[i];
        // four Monte Carlo standard errors, or a little more for points whose MCSE is tiny
        close(
            &point.name,
            point.elpd,
            pointwise,
            (4.0 * point.mcse_elpd).max(0.01),
        );
        assert!(point.pareto_k < 0.5, "{point:?}");
        assert!(point.r_eff > 0.3, "{point:?}");
        exact += pointwise;
    }
    close("elpd", result.elpd, exact, 4.0 * result.mcse_elpd + 0.01);
    assert!(result.mcse_elpd > 0.0 && result.mcse_elpd < 0.1);
    // one parameter
    close("p_loo", result.p, 1.0, 0.2);
    assert!(result.warnings.is_empty());
    assert!(result.to_string().contains("elpd_loo"));

    // WAIC is close to LOO for a model this regular
    let w = waic(&loglik).unwrap();
    close("waic", w.elpd, result.elpd, 0.1);
    assert_eq!(w.waic, -2.0 * w.elpd);
    assert!(w.warnings.is_empty());
}

/// The correct model ranks first, the wrong one's difference is negative and more than a standard error from 0, and
/// both kinds of weights put almost everything on the correct model.
#[test]
fn compare_ranks_the_correct_model_first() {
    let (good, _) = conjugate_fit(1.0);
    let (bad, _) = conjugate_fit(0.3);
    let table = compare(&[("wrong", &bad), ("right", &good)]).unwrap();
    let rows = table.rows();
    assert_eq!(rows[0].name, "right");
    assert_eq!(
        (rows[0].rank, rows[0].elpd_diff, rows[0].se_diff),
        (0, 0.0, 0.0)
    );
    assert_eq!(table.best().name, "right");
    let wrong = table.get("wrong").unwrap();
    assert_eq!(wrong.rank, 1);
    close("elpd_diff", wrong.elpd_diff, bad.elpd - good.elpd, 1e-12);
    assert!(wrong.elpd_diff < -wrong.se_diff, "{wrong:?}");
    assert!(rows[0].weight >= 0.9, "{table}");
    close("weights", rows[0].weight + rows[1].weight, 1.0, 1e-12);
    assert!(table.to_string().contains("elpd_diff"));

    let mut rng = KeanuRng::seed_from_u64(1);
    let bma = pseudo_bma_weights(&[&good, &bad], &mut rng).unwrap();
    close("pseudo-BMA weights", bma.iter().sum(), 1.0, 1e-12);
    assert!(bma[0] >= 0.9, "{bma:?}");
    // one model has all the weight
    assert_eq!(stacking_weights(&[&good]).unwrap(), vec![1.0]);
}

/// Stacking weights of two models that each predict half the points well are interior, and pseudo-BMA+ weights
/// of identical models are equal.
#[test]
fn weights_of_complementary_and_identical_models() {
    let names = names(6);
    let narrow = loo_from_values(&fixture_values("narrow"), &names, Some(&[1.0; 6])).unwrap();
    let wide = loo_from_values(&fixture_values("wide"), &names, Some(&[1.0; 6])).unwrap();
    let w = stacking_weights(&[&narrow, &wide]).unwrap();
    assert!(w.iter().all(|&w| w > 0.3 && w < 0.7), "{w:?}");
    let mut rng = KeanuRng::seed_from_u64(2);
    let same = pseudo_bma_weights(&[&narrow, &narrow], &mut rng).unwrap();
    close("pseudo-BMA", same[0], 0.5, 1e-12);
}

/// A point whose importance ratios have a very heavy tail gets a k̂ above the threshold and a warning naming it.
#[test]
fn influential_points_are_reported() {
    let d = GeneralizedPareto::new(1.2, 1.0);
    let mut rng = KeanuRng::seed_from_u64(3);
    let draws: Vec<Vec<f64>> = (0..2000)
        .map(|_| {
            // log p(y₁ | θ) = −log(ratio) for a generalized Pareto ratio; the other point is regular
            let ratio = d.quantile(rng.next_f64()) + 1e-3;
            vec![-0.1 * rng.next_gaussian().powi(2), -ratio.ln()]
        })
        .collect();
    let result = loo_from_values(std::slice::from_ref(&draws), &names(2), None).unwrap();
    assert!(result.points[1].pareto_k > 0.7, "{:?}", result.points[1]);
    assert!(result.points[0].pareto_k < 0.7, "{:?}", result.points[0]);
    let bad: Vec<&str> = result
        .bad_points()
        .iter()
        .map(|p| p.name.as_str())
        .collect();
    assert_eq!(bad, ["y[1]"]);
    assert_eq!(result.warnings.len(), 1);
    assert!(
        result.warnings[0].contains("y[1]"),
        "{}",
        result.warnings[0]
    );
    assert!(result.has_warnings());
    // the same point's log likelihood varies a lot, which WAIC warns about too
    let w = waic_from_values(&[draws], &names(2)).unwrap();
    assert!(w.warnings[0].contains("y[1]"), "{:?}", w.warnings);
}

/// Independent draws have a relative efficiency near 1, strongly autocorrelated ones much less.
#[test]
fn relative_efficiency_reflects_autocorrelation() {
    let mut rng = KeanuRng::seed_from_u64(4);
    let mut x = 0.0;
    let chains: Vec<Vec<Vec<f64>>> = (0..2)
        .map(|_| {
            (0..1000)
                .map(|_| {
                    x = 0.95 * x + rng.next_gaussian() * 0.1;
                    vec![-0.5 * rng.next_gaussian().powi(2), x]
                })
                .collect()
        })
        .collect();
    let r_eff = relative_efficiency(&chains).unwrap();
    assert!((r_eff[0] - 1.0).abs() < 0.2, "{r_eff:?}");
    assert!(r_eff[1] < 0.2, "{r_eff:?}");
}

#[test]
fn invalid_inputs_are_rejected() {
    let invalid = |result: Result<_, AlgorithmError>| {
        assert!(matches!(result, Err(AlgorithmError::InvalidArgument(_))));
    };
    let values = fixture_values("narrow");
    invalid(loo_from_values(&values, &names(5), None).map(|_| ()));
    invalid(loo_from_values(&values, &names(6), Some(&[1.0; 2])).map(|_| ()));
    invalid(loo_from_values(&[vec![vec![0.0; 6]]], &names(6), None).map(|_| ()));
    invalid(loo_from_values(&[], &names(6), None).map(|_| ()));
    let mut with_nan = values.clone();
    with_nan[0][3][2] = f64::NAN;
    invalid(loo_from_values(&with_nan, &names(6), None).map(|_| ()));
    invalid(waic_from_values(&with_nan, &names(6)).map(|_| ()));
    let mut ragged = values.clone();
    ragged[0][3].pop();
    invalid(relative_efficiency(&ragged).map(|_| ()));
    // models evaluated on different points can't be compared
    let six = loo_from_values(&values, &names(6), Some(&[1.0; 6])).unwrap();
    let three: Vec<Vec<Vec<f64>>> = vec![values[0].iter().map(|d| d[..3].to_vec()).collect()];
    let three = loo_from_values(&three, &names(3), Some(&[1.0; 3])).unwrap();
    invalid(compare(&[("six", &six), ("three", &three)]).map(|_| ()));
    invalid(stacking_weights::<Loo>(&[]).map(|_| ()));
}
