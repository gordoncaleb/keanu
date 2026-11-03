//! Statistical tests of every sampler: sample moments against the analytic mean and variance (tolerances from
//! the standard error), and Kolmogorov–Smirnov tests where the CDF is simple.

use keanu_distributions::*;
use keanu_tensor::{DoubleTensor, IntTensor};
use std::f64::consts::PI;

const N: usize = 200_000;

fn s(v: f64) -> DoubleTensor {
    DoubleTensor::scalar(v)
}

fn rng() -> KeanuRng {
    KeanuRng::seed_from_u64(42)
}

fn moments(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    (mean, var)
}

/// Mean within 5 standard errors and variance within `var_tolerance` (relative).
fn assert_moments(name: &str, values: &[f64], mean: f64, var: f64, var_tolerance: f64) {
    let (m, v) = moments(values);
    let standard_error = (var / values.len() as f64).sqrt();
    assert!(
        (m - mean).abs() < 5.0 * standard_error + 1e-12,
        "{name}: mean {m}, expected {mean} (se {standard_error})"
    );
    assert!(
        (v - var).abs() <= var_tolerance * var,
        "{name}: variance {v}, expected {var}"
    );
}

fn ints_as_f64(t: &IntTensor) -> Vec<f64> {
    t.as_slice().iter().map(|&x| x as f64).collect()
}

/// Kolmogorov–Smirnov statistic against a CDF must be below the 1% critical value.
fn assert_ks(name: &str, values: &[f64], cdf: impl Fn(f64) -> f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len() as f64;
    let d = sorted
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let f = cdf(x);
            (f - i as f64 / n).abs().max(((i + 1) as f64 / n - f).abs())
        })
        .fold(0.0, f64::max);
    assert!(d < 1.63 / n.sqrt(), "{name}: KS statistic {d}");
}

fn erf(x: f64) -> f64 {
    // Numerical Recipes' Chebyshev fit of erfc (error < 1.2e-7, far below the KS critical value here)
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let y = 1.0
        - t * (-x * x - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 { y } else { -y }
}

#[test]
fn gaussian() {
    let x = Gaussian::new(s(1.5), s(2.0))
        .sample(&[N], &mut rng())
        .unwrap();
    assert_moments("Gaussian", x.as_slice(), 1.5, 4.0, 0.02);
    assert_ks("Gaussian", x.as_slice(), |v| {
        0.5 * (1.0 + erf((v - 1.5) / (2.0 * 2f64.sqrt())))
    });
}

#[test]
fn gaussian_broadcasts_parameters() {
    let x = Gaussian::new(DoubleTensor::vector(vec![-5.0, 5.0]), s(1.0))
        .sample(&[N / 2, 2], &mut rng())
        .unwrap();
    let column = |c: usize| {
        x.as_slice()
            .iter()
            .skip(c)
            .step_by(2)
            .copied()
            .collect::<Vec<_>>()
    };
    assert_moments("Gaussian column 0", &column(0), -5.0, 1.0, 0.03);
    assert_moments("Gaussian column 1", &column(1), 5.0, 1.0, 0.03);
}

#[test]
fn half_gaussian() {
    let sigma: f64 = 2.0;
    let x = HalfGaussian::new(s(sigma))
        .sample(&[N], &mut rng())
        .unwrap();
    assert!(x.as_slice().iter().all(|&v| v >= 0.0));
    assert_moments(
        "HalfGaussian",
        x.as_slice(),
        sigma * (2.0 / PI).sqrt(),
        sigma * sigma * (1.0 - 2.0 / PI),
        0.02,
    );
}

#[test]
fn cauchy_and_half_cauchy_quantiles() {
    let x = Cauchy::new(s(2.0), s(0.5))
        .sample(&[N], &mut rng())
        .unwrap();
    assert_ks("Cauchy", x.as_slice(), |v| {
        0.5 + ((v - 2.0) / 0.5).atan() / PI
    });
    let h = HalfCauchy::new(s(0.5)).sample(&[N], &mut rng()).unwrap();
    assert!(h.as_slice().iter().all(|&v| v >= 0.0));
    assert_ks("HalfCauchy", h.as_slice(), |v| 2.0 * (v / 0.5).atan() / PI);
}

#[test]
fn beta() {
    for (a, b) in [(2.0, 5.0), (0.5, 0.5), (3.0, 1.5)] {
        let x = Beta::standard(s(a), s(b)).sample(&[N], &mut rng()).unwrap();
        let ab: f64 = a + b;
        assert_moments(
            &format!("Beta({a}, {b})"),
            x.as_slice(),
            a / ab,
            a * b / (ab * ab * (ab + 1.0)),
            0.02,
        );
    }
    let arcsine = Beta::standard(s(0.5), s(0.5))
        .sample(&[N], &mut rng())
        .unwrap();
    assert_ks("Beta(0.5, 0.5)", arcsine.as_slice(), |v| {
        2.0 / PI * v.sqrt().asin()
    });

    let bounded = Beta::new(s(2.0), s(3.0), s(-1.0), s(2.0))
        .sample(&[N], &mut rng())
        .unwrap();
    assert!(
        bounded
            .as_slice()
            .iter()
            .all(|&v| (-1.0..=2.0).contains(&v))
    );
    assert_moments(
        "Beta(2, 3) on [-1, 2]",
        bounded.as_slice(),
        -1.0 + 3.0 * 0.4,
        9.0 * 6.0 / (25.0 * 6.0),
        0.02,
    );
}

#[test]
fn chi_squared() {
    for k in [1i64, 3, 10] {
        let x = ChiSquared::new(IntTensor::scalar(k))
            .sample(&[N], &mut rng())
            .unwrap();
        assert_moments(
            &format!("ChiSquared({k})"),
            x.as_slice(),
            k as f64,
            2.0 * k as f64,
            0.03,
        );
    }
}

#[test]
fn exponential() {
    let x = Exponential::new(s(2.0)).sample(&[N], &mut rng()).unwrap();
    assert_moments("Exponential", x.as_slice(), 0.5, 0.25, 0.03);
    assert_ks("Exponential", x.as_slice(), |v| 1.0 - (-2.0 * v).exp());
}

#[test]
fn gamma_including_shape_below_one() {
    for (theta, k) in [
        (2.0, 0.2),
        (2.0, 0.5),
        (1.0, 0.9),
        (0.5, 1.0),
        (1.5, 3.0),
        (1.0, 20.0),
    ] {
        let x = Gamma::new(s(theta), s(k)).sample(&[N], &mut rng()).unwrap();
        assert_moments(
            &format!("Gamma(theta {theta}, k {k})"),
            x.as_slice(),
            k * theta,
            k * theta * theta,
            0.04,
        );
    }
}

#[test]
fn inverse_gamma() {
    let (a, b): (f64, f64) = (6.0, 2.0);
    let x = InverseGamma::new(s(a), s(b))
        .sample(&[N], &mut rng())
        .unwrap();
    assert_moments(
        "InverseGamma",
        x.as_slice(),
        b / (a - 1.0),
        b * b / ((a - 1.0).powi(2) * (a - 2.0)),
        0.05,
    );
}

#[test]
fn laplace() {
    let x = Laplace::new(s(1.0), s(0.5))
        .sample(&[N], &mut rng())
        .unwrap();
    assert_moments("Laplace", x.as_slice(), 1.0, 2.0 * 0.25, 0.03);
    assert_ks("Laplace", x.as_slice(), |v| {
        if v < 1.0 {
            0.5 * ((v - 1.0) / 0.5).exp()
        } else {
            1.0 - 0.5 * (-(v - 1.0) / 0.5).exp()
        }
    });
}

#[test]
fn logistic() {
    let x = Logistic::new(s(1.0), s(0.5))
        .sample(&[N], &mut rng())
        .unwrap();
    assert_moments("Logistic", x.as_slice(), 1.0, 0.25 * PI * PI / 3.0, 0.03);
}

#[test]
fn log_normal() {
    let (mu, sigma): (f64, f64) = (0.0, 0.5);
    let x = LogNormal::new(s(mu), s(sigma))
        .sample(&[N], &mut rng())
        .unwrap();
    let var = ((sigma * sigma).exp() - 1.0) * (2.0 * mu + sigma * sigma).exp();
    assert_moments(
        "LogNormal",
        x.as_slice(),
        (mu + sigma * sigma / 2.0).exp(),
        var,
        0.04,
    );
}

#[test]
fn pareto() {
    let (l, a): (f64, f64) = (1.0, 5.0);
    let x = Pareto::new(s(l), s(a)).sample(&[N], &mut rng()).unwrap();
    assert!(x.as_slice().iter().all(|&v| v >= l));
    assert_moments(
        "Pareto",
        x.as_slice(),
        a * l / (a - 1.0),
        l * l * a / ((a - 1.0).powi(2) * (a - 2.0)),
        0.06,
    );
}

#[test]
fn student_t() {
    let v = 6.0;
    let x = StudentT::new(IntTensor::scalar(6))
        .sample(&[N], &mut rng())
        .unwrap();
    assert_moments("StudentT", x.as_slice(), 0.0, v / (v - 2.0), 0.05);
}

#[test]
fn triangular() {
    for (a, b, c) in [(0.0, 1.0, 0.25), (-1.0, 2.0, -1.0), (-1.0, 2.0, 2.0)] {
        let x = Triangular::new(s(a), s(b), s(c))
            .sample(&[N], &mut rng())
            .unwrap();
        let var = (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0;
        assert_moments(
            &format!("Triangular({a}, {b}, {c})"),
            x.as_slice(),
            (a + b + c) / 3.0,
            var,
            0.02,
        );
    }
}

#[test]
fn uniform() {
    let x = Uniform::new(s(-1.0), s(2.0))
        .sample(&[N], &mut rng())
        .unwrap();
    assert_moments("Uniform", x.as_slice(), 0.5, 9.0 / 12.0, 0.02);
    assert_ks("Uniform", x.as_slice(), |v| {
        ((v + 1.0) / 3.0).clamp(0.0, 1.0)
    });
}

#[test]
fn multivariate_gaussian() {
    let mu = DoubleTensor::vector(vec![0.5, -1.0]);
    let cov = DoubleTensor::new(&[2, 2], vec![2.0, 0.6, 0.6, 1.0]).unwrap();
    let x = MultivariateGaussian::new(mu, cov)
        .sample(&[N, 2], &mut rng())
        .unwrap();
    let a: Vec<f64> = x.as_slice().iter().step_by(2).copied().collect();
    let b: Vec<f64> = x.as_slice().iter().skip(1).step_by(2).copied().collect();
    assert_moments("MVG x0", &a, 0.5, 2.0, 0.02);
    assert_moments("MVG x1", &b, -1.0, 1.0, 0.02);
    let (ma, mb) = (moments(&a).0, moments(&b).0);
    let cov01 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - ma) * (y - mb))
        .sum::<f64>()
        / (N as f64 - 1.0);
    assert!((cov01 - 0.6).abs() < 0.02, "MVG covariance {cov01}");
}

#[test]
fn dirichlet_with_concentration_below_one() {
    let c = [0.5, 2.0, 3.0];
    let total: f64 = c.iter().sum();
    let d = Dirichlet::new(DoubleTensor::vector(c.to_vec()));
    let mut r = rng();
    let draws: Vec<DoubleTensor> = (0..N / 4).map(|_| d.sample(&mut r).unwrap()).collect();
    for (i, &ci) in c.iter().enumerate() {
        let values: Vec<f64> = draws.iter().map(|x| x.as_slice()[i]).collect();
        assert_moments(
            &format!("Dirichlet component {i}"),
            &values,
            ci / total,
            ci * (total - ci) / (total * total * (total + 1.0)),
            0.04,
        );
    }
    assert!(draws.iter().all(|x| (x.sum() - 1.0).abs() < 1e-12));
}

#[test]
fn bernoulli() {
    let x = Bernoulli::new(s(0.3)).sample(&[N], &mut rng()).unwrap();
    let values: Vec<f64> = x
        .as_slice()
        .iter()
        .map(|&b| f64::from(u8::from(b)))
        .collect();
    assert_moments("Bernoulli", &values, 0.3, 0.21, 0.02);
}

#[test]
fn binomial() {
    let x = Binomial::new(s(0.3), IntTensor::scalar(10))
        .sample(&[N], &mut rng())
        .unwrap();
    assert_moments("Binomial", &ints_as_f64(&x), 3.0, 2.1, 0.02);
}

#[test]
fn geometric() {
    let p = 0.25;
    let x = Geometric::new(s(p)).sample(&[N], &mut rng()).unwrap();
    assert!(x.as_slice().iter().all(|&k| k >= 1));
    assert_moments(
        "Geometric",
        &ints_as_f64(&x),
        1.0 / p,
        (1.0 - p) / (p * p),
        0.04,
    );
}

#[test]
fn poisson_with_broadcast_mu() {
    let x = Poisson::new(DoubleTensor::new(&[2, 1], vec![1.0, 700.0]).unwrap())
        .sample(&[2, N / 2], &mut rng())
        .unwrap();
    let values = ints_as_f64(&x);
    assert_moments("Poisson row 0 (mu 1)", &values[..N / 2], 1.0, 1.0, 0.03);
    assert_moments(
        "Poisson row 1 (mu 700)",
        &values[N / 2..],
        700.0,
        700.0,
        0.03,
    );
}

#[test]
fn uniform_int_with_negative_range() {
    let x = UniformInt::new(IntTensor::scalar(-3), IntTensor::scalar(2))
        .sample(&[N], &mut rng())
        .unwrap();
    assert!(x.as_slice().iter().all(|&k| (-3..2).contains(&k)));
    let mut counts = [0usize; 5];
    for &k in x.as_slice() {
        counts[(k + 3) as usize] += 1;
    }
    for count in counts {
        assert!(
            (count as f64 - N as f64 / 5.0).abs() < 5.0 * (N as f64 * 0.2 * 0.8).sqrt(),
            "counts {counts:?}"
        );
    }
    assert_moments("UniformInt", &ints_as_f64(&x), -1.0, 2.0, 0.02);
}

#[test]
fn multinomial() {
    let p = [0.2, 0.5, 0.3];
    let d = Multinomial::new(IntTensor::scalar(10), DoubleTensor::vector(p.to_vec())).unwrap();
    let x = d.sample(&[N / 4, 3], &mut rng()).unwrap();
    let rows: Vec<&[i64]> = x.as_slice().chunks(3).collect();
    assert!(rows.iter().all(|r| r.iter().sum::<i64>() == 10));
    for (i, &pi) in p.iter().enumerate() {
        let values: Vec<f64> = rows.iter().map(|r| r[i] as f64).collect();
        assert_moments(
            &format!("Multinomial category {i}"),
            &values,
            10.0 * pi,
            10.0 * pi * (1.0 - pi),
            0.04,
        );
    }
}

#[test]
fn categorical() {
    let d = Categorical::new(vec![s(1.0), s(2.0), s(3.0)]).unwrap();
    let x = d.sample(&[N], &mut rng()).unwrap();
    for (category, expected) in [(0, 1.0 / 6.0), (1, 2.0 / 6.0), (2, 3.0 / 6.0)] {
        let frequency = x.as_slice().iter().filter(|&&c| c == category).count() as f64 / N as f64;
        assert!(
            (frequency - expected).abs() < 5.0 * (expected * (1.0 - expected) / N as f64).sqrt(),
            "category {category}: {frequency}"
        );
    }
}

#[test]
fn kde() {
    let samples = vec![0.0, 1.0, 5.0];
    let kde = Kde::new(samples.clone(), 0.5).unwrap();
    let x = kde.sample(N, &mut rng());
    let mean = samples.iter().sum::<f64>() / 3.0;
    let population_var = samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 3.0;
    assert_moments("KDE", x.as_slice(), mean, population_var + 0.25, 0.02);
}

#[test]
fn seeded_samples_are_reproducible() {
    let d = Gamma::new(s(1.0), s(0.5));
    assert_eq!(
        d.sample(&[100], &mut KeanuRng::seed_from_u64(7)).unwrap(),
        d.sample(&[100], &mut KeanuRng::seed_from_u64(7)).unwrap()
    );
}

#[test]
fn invalid_parameters_are_errors_not_panics() {
    let mut r = rng();
    assert!(Gamma::new(s(-1.0), s(1.0)).sample(&[3], &mut r).is_err());
    assert!(Cauchy::new(s(0.0), s(0.0)).sample(&[3], &mut r).is_err());
    assert!(Poisson::new(s(-2.0)).sample(&[3], &mut r).is_err());
    assert!(Laplace::new(s(0.0), s(-1.0)).sample(&[3], &mut r).is_err());
    assert!(
        Triangular::new(s(0.0), s(1.0), s(2.0))
            .sample(&[3], &mut r)
            .is_err()
    );
    assert!(
        Dirichlet::new(DoubleTensor::vector(vec![0.5, 0.5]))
            .log_prob(&DoubleTensor::vector(vec![0.2, 0.2]))
            .is_err()
    );
}
