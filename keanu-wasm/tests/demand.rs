//! The demand model recovers what its synthetic sales were generated with, and its stock decision behaves as a
//! planner would expect.

use keanu_wasm::demos::demand::{self, DemandModel, MAX_PERIOD, TRUE_WEEKDAY};

fn fitted() -> (DemandModel, demand::Sales) {
    let sales = demand::sales(13, 1).unwrap();
    let mut model = DemandModel::new(&sales.history, 300, 1).unwrap();
    let mut calls = 0;
    while !model.chain().advance(50).unwrap() {
        calls += 1;
        assert!(model.chain().progress() < 1.0);
    }
    assert!(calls > 10);
    (model, sales)
}

#[test]
fn the_demand_model_recovers_the_truth_and_orders_more_for_more_cover() {
    let (model, sales) = fitted();
    let report = model.report(&sales, 28, 0.95, 20).unwrap();
    assert_eq!(report.draws, 300);
    assert_eq!(report.futures, 6000);
    assert!(
        report.divergences <= 3,
        "{} divergences",
        report.divergences
    );

    // the weekday pattern: every true multiplier inside its 90% interval, and the ordering recovered
    for (effect, truth) in report.weekdays.iter().zip(TRUE_WEEKDAY) {
        let m = effect.multiplier;
        assert!(
            m.low < truth.exp() && truth.exp() < m.high,
            "{}: {m:?} vs {}",
            effect.day,
            truth.exp()
        );
    }
    let saturday = report.weekdays[5].multiplier.mean;
    assert!(
        report
            .weekdays
            .iter()
            .all(|w| w.multiplier.mean <= saturday)
    );

    // growth over the year, dispersion and the size of the level's steps
    let g = report.growth;
    assert!(
        g.low < report.true_growth && report.true_growth < g.high,
        "{g:?}"
    );
    let s = report.drift;
    assert!(
        s.low < report.true_drift && report.true_drift < s.high,
        "{s:?}"
    );
    let d = report.dispersion;
    assert!(
        d.low < report.true_dispersion && report.true_dispersion < d.high,
        "{d:?}"
    );

    // the expected sales cover the true mean on most days of the history
    let covered = report
        .history
        .iter()
        .filter(|day| day.interval.low <= day.truth && day.truth <= day.interval.high)
        .count();
    assert!(
        covered * 10 >= report.history.len() * 8,
        "{covered} of {}",
        report.history.len()
    );

    // the held-out month: its total is a plausible draw, and most days are inside their band
    assert!(report.demand.low < report.actual_demand as f64 * 1.05);
    assert!((report.actual_demand as f64) < report.demand.high * 1.05);
    let inside = report
        .forecast
        .iter()
        .filter(|day| {
            day.interval.low <= day.actual as f64 && day.actual as f64 <= day.interval.high
        })
        .count();
    assert!(
        inside >= 22,
        "{inside} of 28 held-out days in their 90% band"
    );

    // the decision
    assert!(report.stock > report.expected_demand);
    assert!((report.stock - report.safety_stock - report.expected_demand).abs() < 1e-9);
    assert!(report.fill_rate > 0.99 && report.fill_rate < 1.0);
    assert!(report.expected_short < report.expected_left_over);
    assert!((report.demand_histogram.fraction.iter().sum::<f64>() - 1.0).abs() < 1e-9);

    let levels: Vec<f64> = [0.8, 0.9, 0.95, 0.99]
        .iter()
        .map(|&s| model.report(&sales, 28, s, 20).unwrap().stock)
        .collect();
    assert!(levels.windows(2).all(|w| w[0] < w[1]), "{levels:?}");
    let periods: Vec<f64> = [7, 14, 28, MAX_PERIOD]
        .iter()
        .map(|&p| model.report(&sales, p, 0.95, 20).unwrap().stock)
        .collect();
    assert!(periods.windows(2).all(|w| w[0] < w[1]), "{periods:?}");

    // repeatable
    let again = model.report(&sales, 28, 0.95, 20).unwrap();
    assert_eq!(again.stock, report.stock);
    assert_eq!(again.expected_short, report.expected_short);

    assert!(model.report(&sales, 0, 0.95, 20).is_err());
    assert!(model.report(&sales, MAX_PERIOD + 1, 0.95, 20).is_err());
    assert!(model.report(&sales, 28, 1.0, 20).is_err());
}

#[test]
fn the_demand_model_needs_whole_periods() {
    let sales = demand::sales(4, 1).unwrap();
    assert!(DemandModel::new(&sales.history[..56], 10, 1).is_err());
    assert!(DemandModel::new(&sales.history[..100], 10, 1).is_err());
    assert!(DemandModel::new(&sales.history[..84], 10, 1).is_ok());
    assert_eq!(sales.history.len(), 112);
    assert_eq!(sales.future.len(), MAX_PERIOD);
}
