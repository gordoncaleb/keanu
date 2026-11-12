//! The dosing demo recovers the synthetic patient and makes dosing decisions that behave sensibly.

use keanu_wasm::demos::dosing::{
    self, DosingModel, DosingReport, MIN_EFFECTIVE, SAMPLE_HOURS, TOXIC, TRUE_PATIENT,
};

const DRAWS: usize = 1000;
const SEED: u64 = 7;

fn fitted() -> DosingModel {
    let mut model = DosingModel::new(dosing::patient_samples(SEED), DRAWS, SEED).unwrap();
    while !model.chain().advance(100).unwrap() {}
    model
}

fn report(model: &DosingModel, dose: f64, interval: f64) -> DosingReport {
    model.report(dose, interval, false).unwrap()
}

#[test]
fn the_patient_is_recovered_and_dosing_decisions_behave() {
    let samples = dosing::patient_samples(SEED);
    assert_eq!(samples.len(), SAMPLE_HOURS.len());
    for s in &samples {
        let truth = TRUE_PATIENT.single_dose(dosing::FIRST_DOSE, s.hours);
        assert!(
            (s.concentration / truth).ln().abs() < 0.6,
            "{s:?} vs {truth}"
        );
    }

    let model = fitted();
    let r = report(&model, 300.0, 12.0);
    assert_eq!(r.draws, DRAWS);
    assert!(r.divergences < DRAWS / 50, "{} divergences", r.divergences);

    // every parameter's 90% interval covers the truth, and the samples narrow the population's spread
    for p in &r.parameters {
        assert!(
            p.patient.low < p.truth && p.truth < p.patient.high,
            "{}: {:?} vs {}",
            p.name,
            p.patient,
            p.truth
        );
        assert!(p.patient.low > 0.0);
    }
    let clearance = &r.parameters[0];
    assert!(
        clearance.patient.high - clearance.patient.low
            < 0.6 * (clearance.population.high - clearance.population.low),
        "{clearance:?}"
    );
    // the first-dose band covers the true curve almost everywhere after the first hour
    let curve = &r.first_dose_curve;
    let covered = (0..curve.hours.len())
        .filter(|&i| curve.hours[i] >= 1.0)
        .filter(|&i| {
            let (c, t) = (&curve.concentration[i], curve.truth[i]);
            c.low <= t && t <= c.high
        })
        .count();
    let after_first_hour = r
        .first_dose_curve
        .hours
        .iter()
        .filter(|&&h| h >= 1.0)
        .count();
    assert!(
        covered as f64 > 0.9 * after_first_hour as f64,
        "{covered} of {after_first_hour}"
    );

    // a higher dose raises the toxicity risk and lowers the risk of under-dosing
    let low = report(&model, 150.0, 12.0).chosen;
    let high = report(&model, 600.0, 12.0).chosen;
    assert!(
        high.probability_peak_high > low.probability_peak_high,
        "{low:?} {high:?}"
    );
    assert!(
        high.probability_trough_low < low.probability_trough_low,
        "{low:?} {high:?}"
    );
    for pair in r.doses.windows(2) {
        assert!(pair[1].probability_peak_high >= pair[0].probability_peak_high);
        assert!(pair[1].probability_trough_low <= pair[0].probability_trough_low);
    }

    // the suggested dose is sensible: it keeps the true patient in range most of the time with small risks
    let suggested = r.suggested;
    assert!(
        (150.0..=600.0).contains(&suggested.dose),
        "suggested {suggested:?}"
    );
    assert!(suggested.time_in_range > 0.7, "{suggested:?}");
    assert!(suggested.probability_peak_high < 0.25, "{suggested:?}");
    assert!(suggested.probability_trough_low < 0.25, "{suggested:?}");
    let at_suggested = report(&model, suggested.dose, 12.0);
    assert!(
        at_suggested.true_time_in_range > 0.8,
        "{}",
        at_suggested.true_time_in_range
    );
    let c = &at_suggested.steady_state.concentration;
    assert_eq!(c.len(), at_suggested.steady_state.hours.len());
    assert!(at_suggested.trough.mean > MIN_EFFECTIVE * 0.8);
    assert!(at_suggested.peak.mean < TOXIC * 1.1);

    // the population alone, which expects faster clearance, suggests more drug than this patient needs
    let population = model.report(300.0, 12.0, true).unwrap();
    assert!(
        population.suggested.dose > suggested.dose,
        "population {:?}, patient {suggested:?}",
        population.suggested
    );

    // a report is repeatable, and the same seed gives the same fit
    let again = report(&model, 300.0, 12.0);
    assert_eq!(
        serde_json::to_string(&r).unwrap(),
        serde_json::to_string(&again).unwrap()
    );
    let refit = report(&fitted(), 300.0, 12.0);
    assert_eq!(
        serde_json::to_string(&r).unwrap(),
        serde_json::to_string(&refit).unwrap()
    );
}

#[test]
fn bad_inputs_are_rejected() {
    let model = DosingModel::new(dosing::patient_samples(1), 20, 1).unwrap();
    assert!(model.report(300.0, 12.0, false).is_err(), "no draws yet");
    let model = fitted();
    assert!(model.report(-1.0, 12.0, false).is_err());
    assert!(model.report(300.0, 0.0, false).is_err());
    let empty = DosingModel::new(Vec::new(), 20, 1);
    assert!(empty.is_err());
}

#[test]
fn steady_state_is_the_sum_of_repeated_doses() {
    let (dose, interval) = (250.0, 8.0);
    for t in [0.0, 1.5, 4.0, 7.9] {
        let summed: f64 = (0..400)
            .map(|k| TRUE_PATIENT.single_dose(dose, t + k as f64 * interval))
            .sum();
        let closed = TRUE_PATIENT.steady_state(dose, interval, t);
        assert!(
            (summed - closed).abs() < 1e-9 * closed.max(1.0),
            "{summed} vs {closed}"
        );
    }
}
