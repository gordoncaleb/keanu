//! Samplers built on Gamma draws with tiny shapes, whose draws underflow, and samplers on the unit interval whose
//! draws can round onto a bound: the draws stay valid values of their distributions.

use keanu_distributions::*;
use keanu_tensor::{DoubleTensor, IntTensor};

mod common;
use common::s;

fn rng(seed: u64) -> KeanuRng {
    KeanuRng::seed_from_u64(seed)
}

#[test]
fn dirichlet_multinomial_counts_sum_to_n_for_tiny_concentrations() {
    let alpha = DoubleTensor::vector(vec![0.001; 3]);
    let d = DirichletMultinomial::new(IntTensor::scalar(5), &alpha);
    let draws = d.sample(&[2000, 3], &mut rng(1)).unwrap();
    let mut first = 0usize;
    for (i, row) in draws.as_slice().chunks(3).enumerate() {
        assert_eq!(row.iter().sum::<i64>(), 5, "draw {i}: {row:?}");
        // with α this small, the counts nearly always all fall in one category
        first += usize::from(row[0] == 5);
    }
    // the total over the draws, finite only if every draw's is
    assert!(d.log_prob(&draws).unwrap().is_finite());
    // each category takes all the counts a third of the time
    let share = first as f64 / 2000.0;
    assert!((share - 1.0 / 3.0).abs() < 0.05, "share {share}");
}

#[test]
fn dirichlet_draws_are_probability_vectors_for_tiny_concentrations() {
    let d = Dirichlet::new(DoubleTensor::vector(vec![0.001; 3]));
    let mut r = rng(2);
    let mut mean_first = 0.0;
    for _ in 0..2000 {
        let x = d.sample(&mut r).unwrap();
        let x = x.as_slice();
        assert!(x.iter().all(|v| v.is_finite() && *v >= 0.0), "{x:?}");
        assert!((x.iter().sum::<f64>() - 1.0).abs() < 1e-12, "{x:?}");
        mean_first += x[0] / 2000.0;
    }
    assert!((mean_first - 1.0 / 3.0).abs() < 0.05, "mean {mean_first}");
}

#[test]
fn gamma_draws_are_positive_for_tiny_shapes() {
    let draws = Gamma::new(s(1.0), s(0.001))
        .sample(&[5000], &mut rng(3))
        .unwrap();
    assert!(draws.as_slice().iter().all(|&x| x > 0.0));
}

#[test]
fn beta_draws_stay_strictly_inside_the_interval_for_tiny_shapes() {
    for (min, max) in [(0.0, 1.0), (2.0, 5.0)] {
        let d = Beta::new(s(0.001), s(0.001), s(min), s(max));
        let draws = d.sample(&[5000], &mut rng(4)).unwrap();
        let xs = draws.as_slice();
        assert!(xs.iter().all(|&x| x > min && x < max), "[{min}, {max}]");
        // the mass is split evenly between the two ends
        let low = xs.iter().filter(|&&x| x < (min + max) / 2.0).count() as f64 / 5000.0;
        assert!((low - 0.5).abs() < 0.05, "share near the lower bound {low}");
    }
    let binomial = BetaBinomial::new(IntTensor::scalar(10), s(0.001), s(0.001));
    let draws = binomial.sample(&[2000], &mut rng(5)).unwrap();
    // p is nearly always within 1e-3 of 0 or 1, so the counts nearly always all fall on one side, evenly
    let ends = draws
        .as_slice()
        .iter()
        .filter(|&&k| k == 0 || k == 10)
        .count();
    let zeros = draws.as_slice().iter().filter(|&&k| k == 0).count() as f64;
    assert!(ends > 1950, "{ends} draws of 0 or 10");
    assert!((zeros / ends as f64 - 0.5).abs() < 0.05, "{zeros} zeros");
}

#[test]
fn student_t_draws_stay_finite_unless_the_true_draw_overflows() {
    // for ν = 0.01 the χ² draw is below the least positive float about 2.4% of the time, but the draw itself
    // exceeds the largest float only about 0.1% of the time
    let nu = 0.01;
    let (mu, scale) = (
        DoubleTensor::vector(vec![0.0]),
        DoubleTensor::new(&[1, 1], vec![1.0]).unwrap(),
    );
    let mvt = MultivariateStudentT::new(s(nu), &mu, &scale);
    let draws = mvt.sample(&[20_000, 1], &mut rng(6)).unwrap();
    let infinite = draws.as_slice().iter().filter(|x| !x.is_finite()).count();
    assert!(draws.as_slice().iter().all(|x| !x.is_nan()));
    assert!(infinite < 100, "{infinite} infinite draws");
    let half = HalfStudentT::new(s(nu), s(1.0))
        .sample(&[20_000], &mut rng(7))
        .unwrap();
    let infinite = half.as_slice().iter().filter(|x| !x.is_finite()).count();
    assert!(infinite < 100, "{infinite} infinite half-t draws");
}

#[test]
fn unit_interval_draws_stay_strictly_inside_it() {
    for mu in [-800.0, -40.0, 40.0, 800.0] {
        let d = LogitNormal::new(s(mu), s(1.0));
        let draws = d.sample(&[1000], &mut rng(8)).unwrap();
        assert!(
            draws.as_slice().iter().all(|&x| x > 0.0 && x < 1.0),
            "mu = {mu}"
        );
        let q = d
            .inverse_cdf(&DoubleTensor::vector(vec![1e-10, 0.5, 1.0 - 1e-10]))
            .unwrap();
        assert!(
            q.as_slice().iter().all(|&x| x > 0.0 && x < 1.0),
            "mu = {mu}: {q:?}"
        );
        let q = d
            .inverse_ccdf(&DoubleTensor::vector(vec![1e-10, 0.5]))
            .unwrap();
        assert!(
            q.as_slice().iter().all(|&x| x > 0.0 && x < 1.0),
            "mu = {mu}: {q:?}"
        );
        assert!(
            d.log_prob(&draws)
                .unwrap()
                .as_slice()
                .iter()
                .all(|l| !l.is_nan())
        );
    }
    // the bounds are still the quantiles of 0 and 1
    let d = LogitNormal::new(s(0.0), s(1.0));
    let q = d
        .inverse_cdf(&DoubleTensor::vector(vec![0.0, 1.0]))
        .unwrap();
    assert_eq!(q.as_slice(), &[0.0, 1.0]);
    for (a, b) in [(0.01, 1.0), (1.0, 0.01), (100.0, 1e-3)] {
        let d = Kumaraswamy::new(s(a), s(b));
        let draws = d.sample(&[2000], &mut rng(9)).unwrap();
        assert!(
            draws.as_slice().iter().all(|&x| x > 0.0 && x < 1.0),
            "a = {a}, b = {b}"
        );
    }
}
