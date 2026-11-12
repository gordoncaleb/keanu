//! The demos recover what their synthetic data were generated with, and their reports behave as a decision maker
//! would expect.

use keanu_wasm::demos::ab_test::{AbTest, Arm};
use keanu_wasm::demos::credit::{self, CreditModel};
use keanu_wasm::demos::volatility::{VolatilityKind, VolatilityModel, sample_returns};
use keanu_wasm::run::quantile;

#[test]
fn an_ab_test_finds_the_better_variant_and_runs_in_chunks() {
    let mut test = AbTest::new(
        Arm {
            visitors: 4000,
            conversions: 200,
        },
        Arm {
            visitors: 4000,
            conversions: 260,
        },
        500,
        1,
    )
    .unwrap();
    let mut calls = 0;
    while !test.chain().advance(37).unwrap() {
        calls += 1;
        assert!(test.chain().progress() < 1.0);
    }
    assert!(calls > 20);
    let report = test.report();
    assert_eq!(report.draws, 500);
    assert!(report.probability_b_better > 0.99, "{report:?}");
    assert!((report.rate_a.mean - 0.05).abs() < 0.005);
    assert!((report.rate_b.mean - 0.065).abs() < 0.005);
    assert!(
        report.lift.low < 0.3 && 0.3 < report.lift.high,
        "{:?}",
        report.lift
    );
    assert!(report.expected_loss_b < report.expected_loss_a);
    assert!((report.lift_histogram.fraction.iter().sum::<f64>() - 1.0).abs() < 1e-9);
}

#[test]
fn an_ab_test_rejects_impossible_counts() {
    let arm = Arm {
        visitors: 10,
        conversions: 11,
    };
    assert!(AbTest::new(arm, arm, 10, 1).is_err());
}

#[test]
fn the_credit_model_recovers_grade_default_rates_and_stress_raises_losses() {
    let history = credit::history(3000, 7);
    let book = credit::book(250, 7);
    let mut model = CreditModel::new(&history, 500, 7).unwrap();
    while !model.chain().advance(100).unwrap() {}
    assert_eq!(model.chain().divergences(), 0);

    let base = model.portfolio(&history, &book, 0.0, 15, 400, 50).unwrap();
    for grade in &base.grades {
        let interval = &grade.default_probability;
        // generous: the 90% intervals of five grades should almost all cover the truth
        assert!(
            interval.low - 0.01 < grade.true_probability
                && grade.true_probability < interval.high + 0.01,
            "{grade:?}"
        );
    }
    assert!(base.dti_odds_ratio.low < 1.57 && 1.57 < base.dti_odds_ratio.high);
    // worse grades default more
    let means: Vec<f64> = base
        .grades
        .iter()
        .map(|g| g.default_probability.mean)
        .collect();
    assert!(means.windows(2).all(|w| w[0] < w[1]), "{means:?}");
    assert!(base.value_at_risk_99 > base.expected_loss);
    assert!(base.expected_shortfall_99 >= base.value_at_risk_99);

    // the exact Poisson-binomial tail agrees with the simulated default counts
    let report = model.portfolio(&history, &book, 0.0, 15, 400, 400).unwrap();
    let simulated: f64 = report.defaults_histogram.fraction[16..].iter().sum();
    assert!(report.probability_defaults_exceed > 0.0 && report.probability_defaults_exceed < 1.0);
    assert!(
        (report.probability_defaults_exceed - simulated).abs() < 0.08,
        "exact {} vs simulated {simulated}",
        report.probability_defaults_exceed
    );

    let stressed = model.portfolio(&history, &book, 1.0, 15, 400, 50).unwrap();
    assert!(stressed.expected_loss > 1.8 * base.expected_loss);
    assert!(stressed.probability_defaults_exceed > base.probability_defaults_exceed);
    assert!(stressed.expected_defaults.mean > base.expected_defaults.mean);
}

#[test]
fn both_volatility_models_recover_the_volatility_and_give_a_sensible_value_at_risk() {
    let returns = sample_returns();
    assert_eq!(returns.len(), 200);
    for (kind, draws) in [
        (VolatilityKind::Garch, 500),
        (VolatilityKind::Stochastic, 300),
    ] {
        let mut model = VolatilityModel::new(kind, &returns, draws, 1).unwrap();
        while !model.chain().advance(50).unwrap() {}
        let report = model.report(1_000_000.0, 10, 20).unwrap();
        assert_eq!(report.volatility.len(), 200, "{kind:?}");
        assert_eq!(report.forecast.len(), 10);
        assert!(
            report.divergences < draws / 50,
            "{kind:?}: {} divergences",
            report.divergences
        );
        // simulated by stochastic volatility with phi = 0.9 (a half-life of about 6.6 days) and a daily volatility
        // near 0.6%, or 10% a year; GARCH has its own persistence, but shocks still fade within months
        let half_life = &report.half_life_days;
        match kind {
            VolatilityKind::Stochastic => {
                assert!(half_life.low < 6.6 && 6.6 < half_life.high, "{half_life:?}")
            }
            VolatilityKind::Garch => {
                assert!(
                    half_life.mean > 2.0 && half_life.mean < 120.0,
                    "{half_life:?}"
                )
            }
        }
        assert!(
            report.long_run_volatility.mean > 0.06 && report.long_run_volatility.mean < 0.16,
            "{kind:?}: {:?}",
            report.long_run_volatility
        );
        // a 99% loss of a 1,000,000 position at a daily volatility of 0.3% to 1.5%
        assert!(report.value_at_risk_95 > 0.0);
        assert!(report.value_at_risk_99 > report.value_at_risk_95);
        assert!(
            report.value_at_risk_99 > 7_000.0 && report.value_at_risk_99 < 35_000.0,
            "{kind:?}: {}",
            report.value_at_risk_99
        );
        assert!(report.expected_shortfall_975 > report.value_at_risk_95);
        // the forecast band widens as it reverts toward the long-run level
        let first = &report.forecast[0].volatility;
        let last = &report.forecast[9].volatility;
        assert!(last.high - last.low > first.high - first.low, "{kind:?}");
        // reports are repeatable
        let again = model.report(1_000_000.0, 10, 20).unwrap();
        assert_eq!(again.value_at_risk_99, report.value_at_risk_99);
    }
}

#[test]
fn a_volatility_model_needs_a_month_of_returns() {
    assert!(VolatilityModel::new(VolatilityKind::Garch, &[0.01; 10], 10, 1).is_err());
}

#[test]
fn quantiles_interpolate() {
    assert_eq!(quantile(&[3.0, 1.0, 2.0], 0.5), 2.0);
    assert_eq!(quantile(&[0.0, 10.0], 0.25), 2.5);
}
