//! Tests of the BOBYQA solver (the PRIMA translation) on standard problems with known solutions.

use keanu_algorithms::optimizer::bobyqa::prima::{ExitStatus, Settings, minimize};

fn rosenbrock(x: &[f64]) -> f64 {
    // PRIMA's chained Rosenbrock (`chrosen` in its tests)
    (0..x.len() - 1)
        .map(|i| 4.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
        .sum()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() <= tol,
            "{actual:?} != {expected:?} within {tol}"
        );
    }
}

#[test]
fn prima_example_one_bound_active() {
    // PRIMA's bobyqa_example_1: (x1 - 5)^2 + (x2 - 4)^2 on [-1, 4.5]^2
    let f = |x: &[f64]| (x[0] - 5.0).powi(2) + (x[1] - 4.0).powi(2);
    for settings in [Settings::default(), Settings::default().with_rhobeg(1.0)] {
        let out = minimize(f, &[0.0, 0.0], &[-1.0, -1.0], &[4.5, 4.5], &settings);
        assert_eq!(out.status, ExitStatus::SmallTrustRegionRadius);
        assert_eq!(out.x[0], 4.5, "the active bound is hit exactly");
        assert_close(&out.x, &[4.5, 4.0], 1e-6);
        assert!((out.f - 0.25).abs() < 1e-10);
        assert!(out.evaluations < 100);
    }
}

#[test]
fn unconstrained_rosenbrock() {
    for (n, start) in [(2usize, -1.0), (3, -1.0), (6, 0.0)] {
        let x0 = vec![start; n];
        let out = minimize(
            rosenbrock,
            &x0,
            &[],
            &[],
            &Settings::default().with_rhoend(1e-8),
        );
        assert_eq!(out.status, ExitStatus::SmallTrustRegionRadius, "n = {n}");
        assert_close(&out.x, &vec![1.0; n], 1e-5);
        assert!(out.f < 1e-10, "n = {n}: f = {}", out.f);
    }
}

#[test]
fn chained_rosenbrock_local_minimum_is_stationary() {
    // for n >= 4 the chained Rosenbrock has a second local minimum near x1 = -0.77, which a start at -1 finds
    let out = minimize(
        rosenbrock,
        &[-1.0; 6],
        &[],
        &[],
        &Settings::default().with_rhoend(1e-8),
    );
    let h = 1e-6;
    for i in 0..6 {
        let (mut up, mut down) = (out.x.clone(), out.x.clone());
        up[i] += h;
        down[i] -= h;
        let g = (rosenbrock(&up) - rosenbrock(&down)) / (2.0 * h);
        assert!(g.abs() < 1e-5, "gradient {g} in coordinate {i}");
    }
}

#[test]
fn rosenbrock_with_every_valid_number_of_interpolation_points() {
    let n = 3;
    for npt in [n + 2, 2 * n + 1, 2 * n + 3, (n + 1) * (n + 2) / 2] {
        let out = minimize(
            rosenbrock,
            &[-1.2, 1.0, -0.5],
            &[],
            &[],
            &Settings::default()
                .with_npt(npt)
                .with_rhoend(1e-8)
                .with_maxfun(20_000),
        );
        assert_close(&out.x, &[1.0, 1.0, 1.0], 1e-4);
        assert!(out.f < 1e-8, "npt = {npt}: f = {}", out.f);
    }
}

#[test]
fn bound_constrained_rosenbrock() {
    // with x1 <= 0.5 the minimum is on the bound: f(0.5, x2) is minimised at x2 = 0.25
    let out = minimize(
        rosenbrock,
        &[-1.2, 1.0],
        &[-2.0, -2.0],
        &[0.5, 2.0],
        &Settings::default().with_rhoend(1e-9),
    );
    assert_eq!(out.x[0], 0.5);
    assert_close(&out.x, &[0.5, 0.25], 1e-6);
    assert!((out.f - 0.25).abs() < 1e-10);
}

/// The Chebyquad problem (Fletcher 1965), whose minimum is 0 for n <= 7. Row i of the recurrence holds the
/// shifted Chebyshev polynomial T_i(2x - 1) at each coordinate.
fn chebyquad(x: &[f64]) -> f64 {
    let n = x.len();
    let mut rows: Vec<Vec<f64>> = vec![vec![1.0; n], x.iter().map(|v| 2.0 * v - 1.0).collect()];
    while rows.len() <= n {
        let k = rows.len();
        let next: Vec<f64> = (0..n)
            .map(|j| 2.0 * rows[1][j] * rows[k - 1][j] - rows[k - 2][j])
            .collect();
        rows.push(next);
    }
    rows.iter()
        .enumerate()
        .skip(1)
        .map(|(i, row)| {
            let mut sum = row.iter().sum::<f64>() / n as f64;
            if i % 2 == 0 {
                sum += 1.0 / ((i * i) as f64 - 1.0);
            }
            sum * sum
        })
        .sum()
}

#[test]
fn chebyquad_reaches_zero() {
    for n in [2usize, 4, 6] {
        let x0: Vec<f64> = (1..=n).map(|j| j as f64 / (n + 1) as f64).collect();
        let out = minimize(
            chebyquad,
            &x0,
            &vec![0.0; n],
            &vec![1.0; n],
            &Settings::default()
                .with_rhobeg(0.2 / (n as f64))
                .with_rhoend(1e-8),
        );
        assert!(out.f < 1e-10, "n = {n}: f = {}", out.f);
        assert!(out.x.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }
}

#[test]
fn prima_example_two_points_in_a_square() {
    // PRIMA's bobyqa_example_2 (Powell's test from the BOBYQA paper): 10 points in [-1, 1]^2 pushed apart
    let f = |x: &[f64]| {
        let n = x.len();
        let mut f = 0.0;
        let mut i = 4;
        while i <= n {
            let mut j = 2;
            while j <= i - 2 {
                let temp =
                    ((x[i - 2] - x[j - 2]).powi(2) + (x[i - 1] - x[j - 1]).powi(2)).max(1.0e-4);
                f += 1.0 / temp.sqrt();
                j += 2;
            }
            i += 2;
        }
        f
    };
    let n = 20;
    let mut x0 = vec![0.0; n];
    for i in 1..=n / 2 {
        let angle = i as f64 * 8.0 * 1f64.atan() / (n / 2) as f64;
        x0[2 * i - 2] = angle.cos();
        x0[2 * i - 1] = angle.sin();
    }
    let f0 = f(&x0);
    let out = minimize(
        f,
        &x0,
        &vec![-1.0; n],
        &vec![1.0; n],
        &Settings::default().with_rhobeg(0.1),
    );
    assert!(
        out.f < f0 - 1.0,
        "{} should be well below the starting value {f0}",
        out.f
    );
    assert!(out.x.iter().all(|&v| (-1.0..=1.0).contains(&v)));

    // local optimality: no small feasible coordinate move improves f
    for i in 0..n {
        for h in [1e-4, -1e-4] {
            let mut y = out.x.clone();
            y[i] = (y[i] + h).clamp(-1.0, 1.0);
            assert!(
                f(&y) >= out.f - 1e-7,
                "moving coordinate {i} by {h} improves f"
            );
        }
    }
}

#[test]
fn one_dimension() {
    let out = minimize(
        |x| (x[0] - 2.0).powi(2) + 1.0,
        &[0.0],
        &[0.0],
        &[5.0],
        &Settings::default(),
    );
    assert_close(&out.x, &[2.0], 1e-5);
    assert!((out.f - 1.0).abs() < 1e-10);
}

#[test]
fn nan_values_are_treated_as_large() {
    // undefined (NaN) for x1 < 0.2; the minimum (0.5, 0.5) is in the defined region
    let f = |x: &[f64]| {
        if x[0] < 0.2 {
            f64::NAN
        } else {
            (x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2)
        }
    };
    let out = minimize(
        f,
        &[0.9, 0.9],
        &[],
        &[],
        &Settings::default().with_rhobeg(0.3),
    );
    assert_close(&out.x, &[0.5, 0.5], 1e-5);
}

#[test]
fn exit_conditions() {
    let quad = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
    let out = minimize(
        quad,
        &[3.0, -2.0],
        &[],
        &[],
        &Settings::default().with_maxfun(10),
    );
    assert_eq!(out.status, ExitStatus::MaxEvaluationsReached);
    assert_eq!(out.evaluations, 10);

    let out = minimize(
        quad,
        &[3.0, -2.0],
        &[],
        &[],
        &Settings::default().with_ftarget(1.0),
    );
    assert_eq!(out.status, ExitStatus::TargetReached);
    assert!(out.f <= 1.0);

    let out = minimize(
        quad,
        &[0.0, 0.0],
        &[0.0, 1.0],
        &[0.0, 2.0],
        &Settings::default(),
    );
    assert_eq!(out.status, ExitStatus::NoSpaceBetweenBounds);
    assert_eq!(out.evaluations, 0);
}

#[test]
fn starting_point_outside_the_bounds_is_projected() {
    let out = minimize(
        |x| (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2),
        &[10.0, -10.0],
        &[0.0, 0.0],
        &[3.0, 3.0],
        &Settings::default(),
    );
    assert_close(&out.x, &[1.0, 1.0], 1e-5);
}

/// Regression: bound cleaning replaced any bound with |v| >= BOUNDMAX by "no bound", so a lower bound of +infinity
/// (or an upper bound of -infinity) was silently dropped. As PRIMA, only a very low lower bound or a very high
/// upper bound means no bound.
#[test]
fn an_infinite_bound_on_the_wrong_side_leaves_no_space() {
    let quad = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
    for (xl, xu) in [
        ([f64::INFINITY, 0.0], [f64::INFINITY, 1.0]),
        ([0.0, 0.0], [f64::NEG_INFINITY, 1.0]),
    ] {
        let out = minimize(quad, &[0.5, 0.5], &xl, &xu, &Settings::default());
        assert_eq!(out.status, ExitStatus::NoSpaceBetweenBounds);
    }
    // infinite bounds on the right side, and NaN bounds, are still no bound
    let out = minimize(
        quad,
        &[3.0, -2.0],
        &[f64::NEG_INFINITY, f64::NAN],
        &[f64::INFINITY, f64::NAN],
        &Settings::default(),
    );
    assert_eq!(out.status, ExitStatus::SmallTrustRegionRadius);
    assert_close(&out.x, &[0.0, 0.0], 1e-6);
}

/// Regression: checking the number of interpolation points computed 2 · npt, which overflowed for a huge npt.
#[test]
fn a_huge_number_of_interpolation_points_falls_back_to_the_default() {
    let quad = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
    let out = minimize(
        quad,
        &[3.0, -2.0],
        &[],
        &[],
        &Settings::default()
            .with_maxfun(usize::MAX)
            .with_npt(usize::MAX - 2),
    );
    assert_eq!(out.status, ExitStatus::SmallTrustRegionRadius);
    assert_close(&out.x, &[0.0, 0.0], 1e-6);
}
