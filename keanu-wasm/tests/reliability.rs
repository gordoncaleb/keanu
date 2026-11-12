//! The reliability demo recovers the lifetimes its synthetic fleet was generated with, prices a warranty sensibly,
//! and shows the bias of ignoring the pumps still running.

use keanu_wasm::demos::reliability::{
    self, ReliabilityModel, TRUE_HIGH_LOAD_FACTOR, TRUE_SCALE, TRUE_SHAPE, Weibull,
};

fn fitted() -> ReliabilityModel {
    let mut model = ReliabilityModel::new(reliability::fleet(500, 11), 500, 11).unwrap();
    while !model.chain().advance(100).unwrap() {}
    model
}

#[test]
fn the_fit_recovers_the_true_lifetimes_and_beats_ignoring_running_pumps() {
    let mut model = fitted();
    assert_eq!(model.chain().divergences(), 0);
    let report = model.report(12.0, 5000, 350.0, 100_000.0, 20).unwrap();
    eprintln!(
        "failed {} of {}; shape {:?} scale {:?} factor {:?}; naive {:?}; claims {} ({}..{}) naive {} true {}",
        report.failed,
        report.units,
        report.shape,
        report.scale_normal,
        report.high_load_factor,
        report.naive,
        report.expected_claims,
        report.claims_low,
        report.claims_high,
        report.naive_expected_claims,
        report.true_expected_claims
    );
    assert!(report.failed > 50 && report.failed < report.units / 2);
    let covers = |i: &keanu_wasm::run::Interval, truth: f64| i.low < truth && truth < i.high;
    assert!(covers(&report.shape, TRUE_SHAPE), "{:?}", report.shape);
    assert!(
        covers(&report.scale_normal, TRUE_SCALE),
        "{:?}",
        report.scale_normal
    );
    assert!(
        covers(&report.high_load_factor, TRUE_HIGH_LOAD_FACTOR),
        "{:?}",
        report.high_load_factor
    );
    assert!(report.shape.low > 1.0, "wear-out is detected");
    for life in &report.lives {
        assert!(covers(&life.b10, life.true_b10), "{life:?}");
        // ignoring running pumps makes lives look shorter than they are
        assert!(life.naive_b10 < life.b10.low, "{life:?}");
    }
    // so it predicts far more warranty claims than the truth, and than the censored fit
    assert!(
        report.naive.scale_normal < 0.75 * TRUE_SCALE,
        "{:?}",
        report.naive
    );
    assert!(report.naive_expected_claims > 1.3 * report.true_expected_claims);
    assert!(report.claims_low < report.true_expected_claims);
    assert!(report.true_expected_claims < report.claims_high);
    assert!(
        report.claims_low <= report.expected_claims && report.expected_claims <= report.claims_high
    );
    assert!((report.expected_cost - 350.0 * report.expected_claims).abs() < 1e-6);
    assert!((report.claims_histogram.fraction.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    // the survival curve starts at 1, falls, and its band covers the truth
    assert_eq!(report.survival.len(), reliability::CURVE_MONTHS + 1);
    assert!((report.survival[0].mean - 1.0).abs() < 1e-12);
    assert!(report.survival.windows(2).all(|w| w[1].mean <= w[0].mean));
    let covered = report
        .survival
        .iter()
        .filter(|p| p.low <= p.truth && p.truth <= p.high)
        .count();
    assert!(covered as f64 > 0.9 * report.survival.len() as f64);
}

#[test]
fn longer_warranties_and_bigger_fleets_cost_more_and_reports_repeat() {
    let model = fitted();
    let short = model.report(6.0, 5000, 350.0, 100_000.0, 20).unwrap();
    let long = model.report(24.0, 5000, 350.0, 100_000.0, 20).unwrap();
    assert!(long.expected_claims > short.expected_claims);
    assert!(long.probability_over_budget >= short.probability_over_budget);
    let bigger = model.report(24.0, 10_000, 350.0, 100_000.0, 20).unwrap();
    assert!((bigger.expected_claims - 2.0 * long.expected_claims).abs() < 1e-6);
    let again = model.report(24.0, 5000, 350.0, 100_000.0, 20).unwrap();
    assert_eq!(
        serde_json::to_string(&long).unwrap(),
        serde_json::to_string(&again).unwrap()
    );
    assert!(model.report(-1.0, 5000, 350.0, 1.0, 20).is_err());
}

#[test]
fn the_fleet_b10_life_lies_between_the_loads() {
    let w = Weibull::truth();
    let fleet = w.fleet_life(0.1);
    assert!((w.survival(fleet) - 0.9).abs() < 1e-9);
    let high = w.scale_high * (-(0.9f64).ln()).powf(1.0 / w.shape);
    let normal = w.scale_normal * (-(0.9f64).ln()).powf(1.0 / w.shape);
    assert!(high < fleet && fleet < normal);
}
