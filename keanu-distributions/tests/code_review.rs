//! Regression tests for the code review fixes: borrowed parameters, the O(1) binomial and Poisson samplers,
//! one pass Beta sampling and Categorical log prob, and consistent parameter validation.

use keanu_distributions::*;
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};

mod common;
use common::{s, v};

fn ln_gamma(x: f64) -> f64 {
    keanu_tensor::special::ln_gamma(x)
}

#[test]
fn borrowed_and_owned_parameters_agree() {
    let (mu, sigma) = (v(&[0.0, 1.0, -2.0]), v(&[1.0, 0.5, 3.0]));
    let x = v(&[0.3, 0.9, 1.0]);
    let owned = Gaussian::new(mu.clone(), sigma.clone());
    let borrowed = Gaussian::new(&mu, &sigma);
    assert_eq!(owned, borrowed);
    assert_eq!(owned.log_prob(&x).unwrap(), borrowed.log_prob(&x).unwrap());
    assert_eq!(
        owned.d_log_prob(&x).unwrap(),
        borrowed.d_log_prob(&x).unwrap()
    );
    // a borrowed parameter stays borrowed: nothing is copied
    assert!(matches!(borrowed.mu, std::borrow::Cow::Borrowed(_)));
    let probabilities = [s(1.0), s(3.0)];
    let categorical = Categorical::new(&probabilities).unwrap();
    assert_eq!(
        categorical
            .log_prob(&IntTensor::scalar(1))
            .unwrap()
            .as_slice(),
        &[(0.75f64).ln()]
    );
}

fn moments(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    (mean, var)
}

/// Mean within 5 standard errors and variance within `var_tolerance` (relative).
fn assert_moments(name: &str, values: &[i64], mean: f64, var: f64, var_tolerance: f64) {
    let values: Vec<f64> = values.iter().map(|&k| k as f64).collect();
    let (m, v) = moments(&values);
    let standard_error = (var / values.len() as f64).sqrt();
    assert!(
        (m - mean).abs() < 5.0 * standard_error,
        "{name}: mean {m}, expected {mean} (se {standard_error})"
    );
    assert!(
        (v - var).abs() <= var_tolerance * var,
        "{name}: variance {v}, expected {var}"
    );
}

/// Pearson's chi-square statistic of the counts of each value against a pmf, over the values with an expected
/// count of at least 5 (the rest pooled into one bin), must be within 5 standard deviations of its mean (the
/// degrees of freedom).
fn assert_chi_square(name: &str, values: &[i64], pmf: impl Fn(i64) -> f64) {
    let n = values.len() as f64;
    let (min, max) = (*values.iter().min().unwrap(), *values.iter().max().unwrap());
    let mut counts = std::collections::HashMap::new();
    for &k in values {
        *counts.entry(k).or_insert(0usize) += 1;
    }
    let (mut statistic, mut bins) = (0.0, 0usize);
    let (mut pooled_observed, mut pooled_expected) = (0.0, 1.0);
    for k in min..=max {
        let expected = n * pmf(k);
        let observed = counts.get(&k).copied().unwrap_or(0) as f64;
        pooled_expected -= pmf(k);
        if expected >= 5.0 {
            statistic += (observed - expected).powi(2) / expected;
            bins += 1;
        } else {
            pooled_observed += observed;
            pooled_expected += pmf(k);
        }
    }
    let pooled_expected = pooled_expected * n;
    if pooled_expected >= 5.0 {
        statistic += (pooled_observed - pooled_expected).powi(2) / pooled_expected;
        bins += 1;
    }
    let df = (bins - 1) as f64;
    assert!(bins > 10, "{name}: only {bins} bins");
    assert!(
        statistic < df + 5.0 * (2.0 * df).sqrt(),
        "{name}: chi-square {statistic} with {df} degrees of freedom"
    );
}

fn binomial_pmf(n: i64, p: f64) -> impl Fn(i64) -> f64 {
    move |k| {
        if k < 0 || k > n {
            return 0.0;
        }
        let (n, k) = (n as f64, k as f64);
        (ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
            + k * p.ln()
            + (n - k) * (1.0 - p).ln())
        .exp()
    }
}

fn poisson_pmf(mu: f64) -> impl Fn(i64) -> f64 {
    move |k| {
        if k < 0 {
            0.0
        } else {
            (k as f64 * mu.ln() - mu - ln_gamma(k as f64 + 1.0)).exp()
        }
    }
}

const DRAWS: usize = 200_000;

fn binomial_draws(n: i64, p: f64, seed: u64) -> Vec<i64> {
    Binomial::new(s(p), IntTensor::scalar(n))
        .sample(&[DRAWS], &mut KeanuRng::seed_from_u64(seed))
        .unwrap()
        .into_vec()
}

#[test]
fn binomial_with_large_n_matches_its_distribution() {
    // BTRS (n min(p, 1 - p) >= 10), on both sides of p = 0.5, and inversion (smaller means) above 50 trials
    for (seed, n, p) in [
        (1, 100, 0.3),
        (2, 1000, 0.7),
        (3, 100_000_000, 0.5),
        (4, 1_000_000_000, 2e-8),
        (5, 200, 0.98),
        (6, 60, 0.1),
    ] {
        let draws = binomial_draws(n, p, seed);
        let name = format!("Binomial({n}, {p})");
        assert!(draws.iter().all(|&k| (0..=n).contains(&k)), "{name}");
        let (n_f, q) = (n as f64, 1.0 - p);
        assert_moments(&name, &draws, n_f * p, n_f * p * q, 0.03);
        if n <= 1000 {
            assert_chi_square(&name, &draws, binomial_pmf(n, p));
        }
    }
}

#[test]
fn poisson_with_large_mu_matches_its_distribution() {
    for (seed, mu) in [(1, 10.0), (2, 15.5), (3, 100.0), (4, 1e9)] {
        let draws = Poisson::new(s(mu))
            .sample(&[DRAWS], &mut KeanuRng::seed_from_u64(seed))
            .unwrap()
            .into_vec();
        let name = format!("Poisson({mu})");
        assert_moments(&name, &draws, mu, mu, 0.03);
        if mu <= 100.0 {
            assert_chi_square(&name, &draws, poisson_pmf(mu));
        }
    }
}

/// The samplers before the O(1) algorithms, which small parameters still use draw for draw.
fn counting_binomial(n: i64, p: f64, rng: &mut KeanuRng) -> i64 {
    (0..n).filter(|_| rng.next_f64() < p).count() as i64
}

fn knuth_poisson(mu: f64, rng: &mut KeanuRng) -> i64 {
    let (mut mu_left, mut k, mut p) = (mu, 0i64, 1.0);
    loop {
        k += 1;
        p *= rng.next_f64_non_zero();
        while p < 1.0 && mu_left > 0.0 {
            if mu_left > 500.0 {
                p *= 500f64.exp();
                mu_left -= 500.0;
            } else {
                p *= mu_left.exp();
                mu_left = 0.0;
            }
        }
        if p <= 1.0 {
            return k - 1;
        }
    }
}

#[test]
fn small_parameters_draw_the_same_values_as_before() {
    for (n, p) in [(1, 0.5), (10, 0.3), (50, 0.9), (50, 0.5)] {
        let mut reference = KeanuRng::seed_from_u64(9);
        let expected: Vec<i64> = (0..1000)
            .map(|_| counting_binomial(n, p, &mut reference))
            .collect();
        let draws = Binomial::new(s(p), IntTensor::scalar(n))
            .sample(&[1000], &mut KeanuRng::seed_from_u64(9))
            .unwrap();
        assert_eq!(draws.as_slice(), &expected[..], "Binomial({n}, {p})");
    }
    for mu in [0.1, 1.0, 3.0, 9.99] {
        let mut reference = KeanuRng::seed_from_u64(9);
        let expected: Vec<i64> = (0..1000)
            .map(|_| knuth_poisson(mu, &mut reference))
            .collect();
        let draws = Poisson::new(s(mu))
            .sample(&[1000], &mut KeanuRng::seed_from_u64(9))
            .unwrap();
        assert_eq!(draws.as_slice(), &expected[..], "Poisson({mu})");
    }
}

#[test]
fn beta_sample_is_unchanged() {
    // the reference: Java's arithmetic on Gamma draws of the same stream
    let (alpha, beta) = (v(&[0.5, 2.0, 3.0]), v(&[2.0, 2.0, 0.7]));
    let (min, max) = (s(-1.0), v(&[1.0, 2.0, 4.0]));
    let mut reference = KeanuRng::seed_from_u64(5);
    let y1 = Gamma::new(s(1.0), &alpha)
        .sample(&[4, 3], &mut reference)
        .unwrap();
    let y2 = Gamma::new(s(1.0), &beta)
        .sample(&[4, 3], &mut reference)
        .unwrap();
    let samples = Beta::new(&alpha, &beta, &min, &max)
        .sample(&[4, 3], &mut KeanuRng::seed_from_u64(5))
        .unwrap();
    for i in 0..12 {
        let (y1, y2) = (y1.as_slice()[i], y2.as_slice()[i]);
        let (a, b, hi) = (
            alpha.as_slice()[i % 3],
            beta.as_slice()[i % 3],
            max.as_slice()[i % 3],
        );
        let lo = -1.0;
        let expected = if a < b {
            hi - (y2 / (y1 + y2)) * (hi - lo)
        } else {
            lo + (y1 / (y1 + y2)) * (hi - lo)
        };
        assert_eq!(samples.as_slice()[i].to_bits(), expected.to_bits());
    }
}

#[test]
fn categorical_log_prob_in_one_pass() {
    let probabilities = [v(&[1.0, 0.0, 2.0]), s(2.0), v(&[3.0, 1.0, 0.5])];
    let d = Categorical::new(&probabilities).unwrap();
    let x = IntTensor::new(&[2, 3], vec![0, 1, 2, 3, -1, 1]).unwrap();
    let log_prob = d.log_prob(&x).unwrap();
    assert_eq!(log_prob.shape(), &[2, 3]);
    for (i, &category) in x.as_slice().iter().enumerate() {
        let column = i % 3;
        let p = |c: usize| probabilities[c].as_slice()[if c == 1 { 0 } else { column }];
        let total = 0.0 + p(0) + p(1) + p(2);
        let expected = if (0..3).contains(&category) {
            (p(category as usize) / total).ln()
        } else {
            f64::NEG_INFINITY
        };
        assert_eq!(log_prob.as_slice()[i].to_bits(), expected.to_bits(), "{i}");
    }
}

#[test]
fn negative_categorical_probabilities_are_errors() {
    let d = Categorical::new(vec![s(1.0), s(-0.5), s(2.0)]).unwrap();
    assert!(matches!(
        d.log_prob(&IntTensor::scalar(0)),
        Err(DistributionError::InvalidParameter(_))
    ));
    let mut rng = KeanuRng::seed_from_u64(1);
    assert!(matches!(
        d.sample(&[3], &mut rng),
        Err(DistributionError::InvalidParameter(_))
    ));
    let nan = Categorical::new(vec![s(f64::NAN), s(1.0)]).unwrap();
    assert!(nan.log_prob(&IntTensor::scalar(1)).is_err());
}

#[test]
fn every_sampler_rejects_invalid_parameters() {
    let mut rng = KeanuRng::seed_from_u64(1);
    let invalid =
        |result: Result<()>| matches!(result, Err(DistributionError::InvalidParameter(_)));
    let r = &mut rng;
    let checks: Vec<(&str, Result<()>)> = vec![
        (
            "Gaussian",
            Gaussian::new(s(0.0), s(-1.0)).sample(&[2], r).map(drop),
        ),
        (
            "Gaussian NaN",
            Gaussian::new(s(0.0), s(f64::NAN)).sample(&[2], r).map(drop),
        ),
        (
            "HalfGaussian",
            HalfGaussian::new(s(0.0)).sample(&[2], r).map(drop),
        ),
        (
            "HalfCauchy",
            HalfCauchy::new(s(-1.0)).sample(&[2], r).map(drop),
        ),
        (
            "LogNormal",
            LogNormal::new(s(0.0), s(-1.0)).sample(&[2], r).map(drop),
        ),
        (
            "Exponential",
            Exponential::new(s(-1.0)).sample(&[2], r).map(drop),
        ),
        (
            "Exponential 0",
            Exponential::new(s(0.0)).sample(&[2], r).map(drop),
        ),
        (
            "Logistic",
            Logistic::new(s(0.0), s(-1.0)).sample(&[2], r).map(drop),
        ),
        (
            "Pareto location",
            Pareto::new(s(0.0), s(1.0)).sample(&[2], r).map(drop),
        ),
        (
            "Pareto scale",
            Pareto::new(s(1.0), s(-1.0)).sample(&[2], r).map(drop),
        ),
        (
            "Uniform",
            Uniform::new(s(1.0), s(1.0)).sample(&[2], r).map(drop),
        ),
        (
            "Uniform reversed",
            Uniform::new(s(1.0), s(0.0)).sample(&[2], r).map(drop),
        ),
        (
            "Beta range",
            Beta::new(s(1.0), s(1.0), s(1.0), s(0.0))
                .sample(&[2], r)
                .map(drop),
        ),
        (
            "Triangular range",
            Triangular::new(s(1.0), s(1.0), s(1.0))
                .sample(&[2], r)
                .map(drop),
        ),
        (
            "InverseGamma",
            InverseGamma::new(s(1.0), s(-1.0)).sample(&[2], r).map(drop),
        ),
        (
            "ChiSquared",
            ChiSquared::new(IntTensor::scalar(0))
                .sample(&[2], r)
                .map(drop),
        ),
        (
            "Bernoulli",
            Bernoulli::new(s(1.5)).sample(&[2], r).map(drop),
        ),
        (
            "Bernoulli NaN",
            Bernoulli::new(s(f64::NAN)).sample(&[2], r).map(drop),
        ),
        (
            "Geometric",
            Geometric::new(s(0.0)).sample(&[2], r).map(drop),
        ),
        (
            "Geometric above 1",
            Geometric::new(s(1.5)).sample(&[2], r).map(drop),
        ),
        (
            "Binomial p",
            Binomial::new(s(-0.1), IntTensor::scalar(5))
                .sample(&[2], r)
                .map(drop),
        ),
        (
            "Binomial n",
            Binomial::new(s(0.5), IntTensor::scalar(-5))
                .sample(&[2], r)
                .map(drop),
        ),
        (
            "Poisson infinite",
            Poisson::new(s(f64::INFINITY)).sample(&[2], r).map(drop),
        ),
        (
            "UniformInt",
            UniformInt::new(IntTensor::scalar(3), IntTensor::scalar(3))
                .sample(&[2], r)
                .map(drop),
        ),
    ];
    for (name, result) in checks {
        assert!(invalid(result), "{name}");
    }
}

#[test]
fn degenerate_ranges_have_no_density() {
    let x = v(&[0.0, 0.5, 1.0, 2.0]);
    let all_impossible = |t: DoubleTensor| t.as_slice().iter().all(|&lp| lp == f64::NEG_INFINITY);
    for (min, max) in [(1.0, 1.0), (1.0, 0.0), (f64::NAN, 1.0)] {
        assert!(all_impossible(
            Uniform::new(s(min), s(max)).log_prob(&x).unwrap()
        ));
        assert!(all_impossible(
            Beta::new(s(2.0), s(2.0), s(min), s(max))
                .log_prob(&x)
                .unwrap()
        ));
        assert!(all_impossible(
            Triangular::new(s(min), s(max), s(1.0))
                .log_prob(&x)
                .unwrap()
        ));
    }
    // a mode outside the range
    assert!(all_impossible(
        Triangular::new(s(0.0), s(1.0), s(2.0))
            .log_prob(&x)
            .unwrap()
    ));
    // valid ranges are unchanged
    assert!(
        Beta::standard(s(2.0), s(2.0))
            .log_prob(&x)
            .unwrap()
            .as_slice()[1]
            .is_finite()
    );
}

#[test]
fn bernoulli_log_prob_still_clamps() {
    // only sampling rejects a probability outside [0, 1]
    let lp = Bernoulli::new(s(1.5))
        .log_prob(&BoolTensor::scalar(true))
        .unwrap();
    assert_eq!(lp.as_slice(), &[0.0]);
}

#[test]
fn gaussian_log_prob_and_gradient_in_one_pass_are_exact() {
    let d = Gaussian::new(v(&[0.0, 1.5, -2.0]), s(0.7));
    let x = DoubleTensor::new(&[2, 3], vec![0.1, -3.0, 2.5, 1.0, 1.5, 0.0]).unwrap();
    let (log_prob, gradient) = d.log_prob_and_d_log_prob(&x).unwrap();
    assert_eq!(log_prob, d.log_prob(&x).unwrap());
    assert_eq!(gradient, d.d_log_prob(&x).unwrap());
}
