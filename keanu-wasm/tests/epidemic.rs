//! The epidemic demo recovers the outbreak its synthetic cases were generated with, and its forecast responds to
//! beds and interventions as a planner would expect.

use keanu_wasm::demos::epidemic::{self, EpidemicModel, Intervention, TRUE_R0};
use std::time::Instant;

fn fitted() -> EpidemicModel {
    let cases = epidemic::reported_cases(35, 3);
    let mut model = EpidemicModel::new(&cases, 500, 3).unwrap();
    let started = Instant::now();
    while !model.chain().advance(50).unwrap() {}
    eprintln!("fit in {:.2} s", started.elapsed().as_secs_f64());
    model
}

const NONE: Intervention = Intervention::NONE;

#[test]
fn the_epidemic_model_recovers_r0_and_responds_to_beds_and_interventions() {
    let model = fitted();
    let started = Instant::now();
    let base = model.report(900.0, NONE).unwrap();
    eprintln!("report in {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
    eprintln!(
        "R0 {:?} D {:?} rho {:?} I0 {:?} peak {:?} day {:?} p {} div {}",
        base.r0,
        base.recovery_days,
        base.reporting,
        base.initial,
        base.peak_occupancy,
        base.peak_day,
        base.probability_exceeds,
        base.divergences
    );
    assert!(base.divergences <= 5, "{}", base.divergences);
    assert!(
        base.r0.low < TRUE_R0 && TRUE_R0 < base.r0.high,
        "{:?}",
        base.r0
    );
    let truth = &base.truth;
    assert!(
        base.recovery_days.low < truth.recovery_days
            && truth.recovery_days < base.recovery_days.high
    );
    assert!(base.reporting.low < truth.reporting && truth.reporting < base.reporting.high);
    // the observed counts sit inside the predictive band on most days
    let inside = base
        .observed
        .iter()
        .zip(&base.cases)
        .filter(|(o, b)| b.low <= **o as f64 && **o as f64 <= b.high)
        .count();
    assert!(
        inside as f64 >= 0.8 * base.observed.len() as f64,
        "{inside}"
    );
    assert_eq!(base.cases.len(), epidemic::HORIZON);
    assert!((base.peak_histogram.fraction.iter().sum::<f64>() - 1.0).abs() < 1e-9);

    // more beds, less chance of running out
    let fewer = model.report(600.0, NONE).unwrap();
    let more = model.report(1200.0, NONE).unwrap();
    assert!(fewer.probability_exceeds > base.probability_exceeds);
    assert!(more.probability_exceeds < base.probability_exceeds);
    assert!(fewer.days_over_capacity > more.days_over_capacity);

    // a stronger or earlier cut in contacts lowers the peak
    let cut = |cut, from_day| model.report(900.0, Intervention { cut, from_day }).unwrap();
    let (mild, strong, earlier) = (cut(0.2, 50), cut(0.4, 50), cut(0.2, 40));
    assert!(mild.peak_occupancy.mean < base.peak_occupancy.mean);
    assert!(strong.peak_occupancy.mean < mild.peak_occupancy.mean);
    assert!(earlier.peak_occupancy.mean < mild.peak_occupancy.mean);
    assert!(strong.probability_exceeds <= mild.probability_exceeds);
    assert_eq!(mild.probability_exceeds_without, base.probability_exceeds);
    // an intervention changes nothing before it starts
    let means = |r: &epidemic::EpidemicReport| -> Vec<f64> {
        r.occupancy[..49].iter().map(|i| i.mean).collect()
    };
    assert_eq!(means(&mild), means(&base));
}

#[test]
fn epidemic_reports_are_repeatable() {
    let model = fitted();
    let intervention = Intervention {
        cut: 0.3,
        from_day: 45,
    };
    let first = serde_json::to_string(&model.report(600.0, intervention).unwrap()).unwrap();
    let second = serde_json::to_string(&model.report(600.0, intervention).unwrap()).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        epidemic::reported_cases(35, 3),
        epidemic::reported_cases(35, 3)
    );
}
