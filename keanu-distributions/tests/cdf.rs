//! Log CDFs and log survival functions: against numerical integration of the density (or summation of the mass),
//! complementarity (logaddexp(log_cdf, log_ccdf) = 0), tail stability, inverse CDFs and gradients against finite
//! differences.

use keanu_distributions::*;
use keanu_tensor::{DoubleTensor, IntTensor};

const H: f64 = 1e-6;

fn s(v: f64) -> DoubleTensor {
    DoubleTensor::scalar(v)
}

fn v(values: &[f64]) -> DoubleTensor {
    DoubleTensor::vector(values.to_vec())
}

fn logaddexp(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if m == f64::NEG_INFINITY {
        m
    } else {
        m + ((a - m).exp() + (b - m).exp()).ln()
    }
}

/// ∫ₐᵇ f by composite Simpson's rule, over ln |x| when the interval doesn't contain 0 (so that heavy tails and
/// integrable singularities at 0 are resolved).
fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    if a > 0.0 {
        return simpson_linear(|u| f(u.exp()) * u.exp(), a.ln(), b.ln());
    }
    if b < 0.0 {
        return simpson_linear(|u| f(-u.exp()) * u.exp(), (-b).ln(), (-a).ln());
    }
    simpson_linear(f, a, b)
}

fn simpson_linear(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let n = 4000;
    let h = (b - a) / n as f64;
    let mut total = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        total += if i % 2 == 1 { 4.0 } else { 2.0 } * f(x);
    }
    total * h / 3.0
}

/// A continuous distribution as scalar functions of x.
struct Continuous<'f> {
    name: &'static str,
    log_pdf: &'f dyn Fn(f64) -> f64,
    log_cdf: &'f dyn Fn(f64) -> f64,
    log_ccdf: &'f dyn Fn(f64) -> f64,
}

impl Continuous<'_> {
    /// `points` are increasing; the first must have a CDF below `first_cdf_below` (the left tail, or the bound).
    fn check(&self, points: &[f64], first_cdf_below: f64) {
        let name = self.name;
        assert!(
            (self.log_cdf)(points[0]).exp() <= first_cdf_below,
            "{name}: CDF at {} is {}",
            points[0],
            (self.log_cdf)(points[0]).exp()
        );
        for w in points.windows(2) {
            let (a, b) = (w[0], w[1]);
            let integral = simpson(|x| (self.log_pdf)(x).exp(), a, b);
            let difference = (self.log_cdf)(b).exp() - (self.log_cdf)(a).exp();
            assert!(
                (integral - difference).abs() < 1e-8 * integral.abs().max(1e-3),
                "{name}: ∫ pdf over [{a}, {b}] = {integral}, CDF difference {difference}"
            );
            let survival = (self.log_ccdf)(a).exp() - (self.log_ccdf)(b).exp();
            assert!(
                (integral - survival).abs() < 1e-8 * integral.abs().max(1e-3),
                "{name}: ∫ pdf over [{a}, {b}] = {integral}, survival difference {survival}"
            );
        }
        for &x in points {
            let total = logaddexp((self.log_cdf)(x), (self.log_ccdf)(x));
            assert!(total.abs() < 1e-12, "{name}: logaddexp at {x} is {total}");
        }
    }
}

/// Checks `analytic` (element-wise, summed for a scalar parameter) against central differences of the summed
/// `f` as `at` is perturbed element-wise.
fn check_gradient(
    name: &str,
    at: &DoubleTensor,
    analytic: &DoubleTensor,
    f: impl Fn(&DoubleTensor) -> f64,
) {
    let analytic: Vec<f64> = if analytic.len() != at.len() && at.len() == 1 {
        vec![analytic.sum()]
    } else {
        analytic.as_slice().to_vec()
    };
    assert_eq!(analytic.len(), at.len(), "{name}");
    for i in 0..at.len() {
        let mut up = at.as_slice().to_vec();
        let mut down = up.clone();
        up[i] += H;
        down[i] -= H;
        let numeric = (f(&DoubleTensor::new(at.shape(), up).unwrap())
            - f(&DoubleTensor::new(at.shape(), down).unwrap()))
            / (2.0 * H);
        assert!(
            (numeric - analytic[i]).abs() < 2e-5 * numeric.abs().max(1.0),
            "{name}[{i}]: analytic {}, numeric {numeric}",
            analytic[i]
        );
    }
}

fn sum(t: Result<DoubleTensor>) -> f64 {
    t.unwrap().sum()
}

fn at(t: Result<DoubleTensor>) -> f64 {
    t.unwrap().as_slice()[0]
}

#[test]
fn gaussian() {
    let d = Gaussian::new(s(0.4), s(1.3));
    Continuous {
        name: "gaussian",
        log_pdf: &|x| at(d.log_prob(&s(x))),
        log_cdf: &|x| at(d.log_cdf(&s(x))),
        log_ccdf: &|x| at(d.log_ccdf(&s(x))),
    }
    .check(&[-8.0, -2.0, 0.4, 1.0, 3.5, 9.0], 1e-9);
    // stable far into both tails
    let lower = at(Gaussian::new(s(0.0), s(1.0)).log_cdf(&s(-40.0)));
    assert!((lower + 804.608_442_013_754_2).abs() < 1e-9, "{lower}");
    let upper = at(Gaussian::new(s(0.0), s(1.0)).log_ccdf(&s(40.0)));
    assert!((upper - lower).abs() < 1e-12);
    assert!(at(Gaussian::new(s(0.0), s(1.0)).log_cdf(&s(40.0))).abs() < 1e-300);

    let (x, mu, sigma) = (v(&[-3.0, 0.3, 2.5]), s(0.4), s(1.3));
    for upper in [false, true] {
        let f = |d: &Gaussian<'_>, x: &DoubleTensor| {
            if upper { d.log_ccdf(x) } else { d.log_cdf(x) }
        };
        let g = if upper {
            Gaussian::new(&mu, &sigma).d_log_ccdf(&x)
        } else {
            Gaussian::new(&mu, &sigma).d_log_cdf(&x)
        }
        .unwrap();
        check_gradient("x", &x, &g.x, |x| sum(f(&Gaussian::new(&mu, &sigma), x)));
        check_gradient("mu", &mu, &g.mu, |p| sum(f(&Gaussian::new(p, &sigma), &x)));
        check_gradient("sigma", &sigma, &g.sigma, |p| {
            sum(f(&Gaussian::new(&mu, p), &x))
        });
    }
}

#[test]
fn log_normal() {
    let d = LogNormal::new(s(0.2), s(0.7));
    Continuous {
        name: "log normal",
        log_pdf: &|x| at(d.log_prob(&s(x))),
        log_cdf: &|x| at(d.log_cdf(&s(x))),
        log_ccdf: &|x| at(d.log_ccdf(&s(x))),
    }
    .check(&[1e-3, 0.5, 1.0, 2.0, 5.0, 30.0], 1e-15);
    assert_eq!(at(d.log_cdf(&s(-1.0))), f64::NEG_INFINITY);
    assert_eq!(at(d.log_ccdf(&s(-1.0))), 0.0);

    let (x, mu, sigma) = (v(&[0.3, 1.1, 4.0]), s(0.2), s(0.7));
    for upper in [false, true] {
        let f = |d: &LogNormal<'_>, x: &DoubleTensor| {
            if upper { d.log_ccdf(x) } else { d.log_cdf(x) }
        };
        let d = LogNormal::new(&mu, &sigma);
        let g = if upper {
            d.d_log_ccdf(&x)
        } else {
            d.d_log_cdf(&x)
        }
        .unwrap();
        check_gradient("x", &x, &g.x, |x| sum(f(&LogNormal::new(&mu, &sigma), x)));
        check_gradient("mu", &mu, &g.mu, |p| sum(f(&LogNormal::new(p, &sigma), &x)));
        check_gradient("sigma", &sigma, &g.sigma, |p| {
            sum(f(&LogNormal::new(&mu, p), &x))
        });
    }
}

#[test]
fn exponential() {
    let d = Exponential::new(s(1.7));
    Continuous {
        name: "exponential",
        log_pdf: &|x| at(d.log_prob(&s(x))),
        log_cdf: &|x| at(d.log_cdf(&s(x))),
        log_ccdf: &|x| at(d.log_ccdf(&s(x))),
    }
    .check(&[0.0, 0.1, 0.5, 2.0, 10.0], 0.0);
    assert_eq!(at(d.log_ccdf(&s(100.0))), -170.0);
    assert!((at(d.log_cdf(&s(1e-12))) - (1.7e-12f64).ln()).abs() < 1e-9);

    let (x, rate) = (v(&[0.2, 1.1, 3.0]), s(1.7));
    for upper in [false, true] {
        let f = |d: &Exponential<'_>, x: &DoubleTensor| {
            if upper { d.log_ccdf(x) } else { d.log_cdf(x) }
        };
        let d = Exponential::new(&rate);
        let g = if upper {
            d.d_log_ccdf(&x)
        } else {
            d.d_log_cdf(&x)
        }
        .unwrap();
        check_gradient("x", &x, &g.x, |x| sum(f(&Exponential::new(&rate), x)));
        check_gradient("rate", &rate, &g.rate, |p| sum(f(&Exponential::new(p), &x)));
    }
}

#[test]
fn gamma() {
    for (theta, k) in [(1.5, 2.2), (0.8, 0.6), (2.0, 30.0)] {
        let d = Gamma::new(s(theta), s(k));
        let mean = theta * k;
        let first = if k < 1.0 { 1e-9 } else { 1e-6 * mean };
        Continuous {
            name: "gamma",
            log_pdf: &|x| at(d.log_prob(&s(x))),
            log_cdf: &|x| at(d.log_cdf(&s(x))),
            log_ccdf: &|x| at(d.log_ccdf(&s(x))),
        }
        .check(&[first, 0.3 * mean, mean, 2.0 * mean, 5.0 * mean], 1e-3);
        let (x, th, kk) = (v(&[0.3 * mean, mean, 3.0 * mean]), s(theta), s(k));
        for upper in [false, true] {
            let f = |d: &Gamma<'_>, x: &DoubleTensor| {
                if upper { d.log_ccdf(x) } else { d.log_cdf(x) }
            };
            let d = Gamma::new(&th, &kk);
            let g = if upper {
                d.d_log_ccdf(&x)
            } else {
                d.d_log_cdf(&x)
            }
            .unwrap();
            check_gradient("x", &x, &g.x, |x| sum(f(&Gamma::new(&th, &kk), x)));
            check_gradient("theta", &th, &g.theta, |p| sum(f(&Gamma::new(p, &kk), &x)));
            check_gradient("k", &kk, &g.k, |p| sum(f(&Gamma::new(&th, p), &x)));
        }
    }
}

#[test]
fn beta() {
    for (a, b, min, max) in [(2.5, 1.7, 0.0, 1.0), (0.7, 3.0, -1.0, 2.0)] {
        let d = Beta::new(s(a), s(b), s(min), s(max));
        let range = max - min;
        let points: Vec<f64> = [0.0, 0.05, 0.3, 0.6, 0.95, 1.0]
            .iter()
            .map(|y| min + y * range)
            .collect();
        // the density is infinite at the bound for a < 1: integrate from just inside it
        let mut inner = points.clone();
        inner[0] = min + 1e-12 * range;
        inner[5] = max - 1e-12 * range;
        Continuous {
            name: "beta",
            log_pdf: &|x| at(d.log_prob(&s(x))),
            log_cdf: &|x| at(d.log_cdf(&s(x))),
            log_ccdf: &|x| at(d.log_ccdf(&s(x))),
        }
        .check(&inner[1..5], 1.0);
        assert_eq!(at(d.log_cdf(&s(points[0]))), f64::NEG_INFINITY);
        assert_eq!(at(d.log_ccdf(&s(points[5]))), f64::NEG_INFINITY);
        assert_eq!(at(d.log_cdf(&s(points[5]))), 0.0);

        let x = v(&[min + 0.2 * range, min + 0.55 * range, min + 0.9 * range]);
        let (aa, bb, lo, hi) = (s(a), s(b), s(min), s(max));
        for upper in [false, true] {
            let f = |d: &Beta<'_>, x: &DoubleTensor| {
                if upper { d.log_ccdf(x) } else { d.log_cdf(x) }
            };
            let d = Beta::new(&aa, &bb, &lo, &hi);
            let g = if upper {
                d.d_log_ccdf(&x)
            } else {
                d.d_log_cdf(&x)
            }
            .unwrap();
            check_gradient("x", &x, &g.x, |x| sum(f(&Beta::new(&aa, &bb, &lo, &hi), x)));
            check_gradient("alpha", &aa, &g.alpha, |p| {
                sum(f(&Beta::new(p, &bb, &lo, &hi), &x))
            });
            check_gradient("beta", &bb, &g.beta, |p| {
                sum(f(&Beta::new(&aa, p, &lo, &hi), &x))
            });
        }
    }
}

#[test]
fn student_t() {
    for dof in [1, 4, 30] {
        let d = StudentT::new(IntTensor::scalar(dof));
        let far = if dof == 1 { -1e7 } else { -200.0 };
        Continuous {
            name: "student t",
            log_pdf: &|x| at(d.log_prob(&s(x))),
            log_cdf: &|x| at(d.log_cdf(&s(x))),
            log_ccdf: &|x| at(d.log_ccdf(&s(x))),
        }
        .check(&[far, -3.0, -0.5, 0.0, 0.7, 4.0], 1e-6);
        let x = v(&[-2.0, 0.1, 1.5]);
        let g = d.d_log_cdf(&x).unwrap();
        check_gradient("x", &x, &g.x, |x| sum(d.log_cdf(x)));
        let g = d.d_log_ccdf(&x).unwrap();
        check_gradient("x", &x, &g.x, |x| sum(d.log_ccdf(x)));
    }
    // symmetric, and the 1 dof case is the Cauchy
    let t1 = StudentT::new(IntTensor::scalar(1));
    let cauchy = Cauchy::new(s(0.0), s(1.0));
    for x in [-30.0, -1.0, 0.0, 2.0] {
        assert!((at(t1.log_cdf(&s(x))) - at(cauchy.log_cdf(&s(x)))).abs() < 1e-12);
        assert!((at(t1.log_cdf(&s(x))) - at(t1.log_ccdf(&s(-x)))).abs() < 1e-14);
    }
}

#[test]
fn cauchy() {
    let d = Cauchy::new(s(0.5), s(0.8));
    Continuous {
        name: "cauchy",
        log_pdf: &|x| at(d.log_prob(&s(x))),
        log_cdf: &|x| at(d.log_cdf(&s(x))),
        log_ccdf: &|x| at(d.log_ccdf(&s(x))),
    }
    .check(&[-1e8, -3.0, 0.5, 1.0, 6.0], 1e-8);
    // the far tail keeps its relative precision: F(x) ≈ s / (π |x - l|)
    let tail = at(d.log_cdf(&s(-1e12)));
    assert!((tail - (0.8 / (std::f64::consts::PI * 1e12)).ln()).abs() < 1e-9);

    let (x, l, sc) = (v(&[-1.0, 0.7, 4.0]), s(0.5), s(0.8));
    for upper in [false, true] {
        let f = |d: &Cauchy<'_>, x: &DoubleTensor| {
            if upper { d.log_ccdf(x) } else { d.log_cdf(x) }
        };
        let d = Cauchy::new(&l, &sc);
        let g = if upper {
            d.d_log_ccdf(&x)
        } else {
            d.d_log_cdf(&x)
        }
        .unwrap();
        check_gradient("x", &x, &g.x, |x| sum(f(&Cauchy::new(&l, &sc), x)));
        check_gradient("location", &l, &g.location, |p| {
            sum(f(&Cauchy::new(p, &sc), &x))
        });
        check_gradient("scale", &sc, &g.scale, |p| sum(f(&Cauchy::new(&l, p), &x)));
    }
}

#[test]
fn laplace_and_logistic() {
    let d = Laplace::new(s(1.1), s(0.6));
    Continuous {
        name: "laplace",
        log_pdf: &|x| at(d.log_prob(&s(x))),
        log_cdf: &|x| at(d.log_cdf(&s(x))),
        log_ccdf: &|x| at(d.log_ccdf(&s(x))),
    }
    .check(&[-12.0, 0.0, 1.1, 2.0, 15.0], 1e-7);
    assert_eq!(at(d.log_ccdf(&s(601.1))), -1000.0 - std::f64::consts::LN_2);
    let d = Logistic::new(s(-0.4), s(1.3));
    Continuous {
        name: "logistic",
        log_pdf: &|x| at(d.log_prob(&s(x))),
        log_cdf: &|x| at(d.log_cdf(&s(x))),
        log_ccdf: &|x| at(d.log_ccdf(&s(x))),
    }
    .check(&[-30.0, -2.0, -0.4, 1.0, 30.0], 1e-9);
    assert!((at(d.log_cdf(&s(-1300.4))) + 1000.0).abs() < 1e-9);

    let (x, m, sc) = (v(&[0.3, 1.7, 2.9]), s(1.1), s(0.6));
    for upper in [false, true] {
        let f = |d: &Laplace<'_>, x: &DoubleTensor| {
            if upper { d.log_ccdf(x) } else { d.log_cdf(x) }
        };
        let d = Laplace::new(&m, &sc);
        let g = if upper {
            d.d_log_ccdf(&x)
        } else {
            d.d_log_cdf(&x)
        }
        .unwrap();
        check_gradient("x", &x, &g.x, |x| sum(f(&Laplace::new(&m, &sc), x)));
        check_gradient("mu", &m, &g.mu, |p| sum(f(&Laplace::new(p, &sc), &x)));
        check_gradient("beta", &sc, &g.beta, |p| sum(f(&Laplace::new(&m, p), &x)));

        let f = |d: &Logistic<'_>, x: &DoubleTensor| {
            if upper { d.log_ccdf(x) } else { d.log_cdf(x) }
        };
        let d = Logistic::new(&m, &sc);
        let g = if upper {
            d.d_log_ccdf(&x)
        } else {
            d.d_log_cdf(&x)
        }
        .unwrap();
        check_gradient("x", &x, &g.x, |x| sum(f(&Logistic::new(&m, &sc), x)));
        check_gradient("mu", &m, &g.mu, |p| sum(f(&Logistic::new(p, &sc), &x)));
        check_gradient("s", &sc, &g.s, |p| sum(f(&Logistic::new(&m, p), &x)));
    }
}

#[test]
fn uniform() {
    let d = Uniform::new(s(-1.0), s(3.0));
    Continuous {
        name: "uniform",
        log_pdf: &|x| at(d.log_prob(&s(x))),
        log_cdf: &|x| at(d.log_cdf(&s(x))),
        log_ccdf: &|x| at(d.log_ccdf(&s(x))),
    }
    .check(&[-1.0, 0.0, 2.0, 2.999_999], 0.0);
    assert_eq!(at(d.log_cdf(&s(-2.0))), f64::NEG_INFINITY);
    assert_eq!(at(d.log_cdf(&s(5.0))), 0.0);
    assert_eq!(at(d.log_ccdf(&s(5.0))), f64::NEG_INFINITY);

    let (x, lo, hi) = (v(&[-0.5, 1.0, 2.5]), s(-1.0), s(3.0));
    for upper in [false, true] {
        let f = |d: &Uniform<'_>, x: &DoubleTensor| {
            if upper { d.log_ccdf(x) } else { d.log_cdf(x) }
        };
        let d = Uniform::new(&lo, &hi);
        let g = if upper {
            d.d_log_ccdf(&x)
        } else {
            d.d_log_cdf(&x)
        }
        .unwrap();
        check_gradient("x", &x, &g.x, |x| sum(f(&Uniform::new(&lo, &hi), x)));
        check_gradient("x_min", &lo, &g.x_min, |p| {
            sum(f(&Uniform::new(p, &hi), &x))
        });
        check_gradient("x_max", &hi, &g.x_max, |p| {
            sum(f(&Uniform::new(&lo, p), &x))
        });
    }
}

/// Checks a discrete distribution's CDF and survival against summing its mass from 0.
fn check_discrete(
    name: &str,
    log_pmf: impl Fn(i64) -> f64,
    log_cdf: impl Fn(i64) -> f64,
    log_ccdf: impl Fn(i64) -> f64,
    max: i64,
) {
    assert_eq!(log_cdf(-1), f64::NEG_INFINITY, "{name}");
    assert_eq!(log_ccdf(-1), 0.0, "{name}");
    let mut cumulative = 0.0;
    for k in 0..=max {
        cumulative += log_pmf(k).exp();
        let cdf = log_cdf(k).exp();
        assert!(
            (cdf - cumulative).abs() < 1e-12,
            "{name}: CDF at {k} is {cdf}, summed {cumulative}"
        );
        let survival = log_ccdf(k).exp();
        assert!(
            (survival - (1.0 - cumulative)).abs() < 1e-12,
            "{name}: survival at {k} is {survival}, summed {}",
            1.0 - cumulative
        );
        let total = logaddexp(log_cdf(k), log_ccdf(k));
        assert!(total.abs() < 1e-12, "{name}: logaddexp at {k} is {total}");
    }
}

#[test]
fn poisson() {
    for mu in [0.3, 4.5, 60.0] {
        let d = Poisson::new(s(mu));
        let i = |k: i64| IntTensor::scalar(k);
        check_discrete(
            "poisson",
            |k| at(d.log_prob(&i(k))),
            |k| at(d.log_cdf(&i(k))),
            |k| at(d.log_ccdf(&i(k))),
            (3.0 * mu) as i64 + 10,
        );
        let k = IntTensor::vector(vec![0, (mu as i64) / 2, mu as i64 + 3]);
        let m = s(mu);
        let g = Poisson::new(&m).d_log_cdf(&k).unwrap();
        check_gradient("mu", &m, &g.mu, |p| sum(Poisson::new(p).log_cdf(&k)));
        let g = Poisson::new(&m).d_log_ccdf(&k).unwrap();
        check_gradient("mu", &m, &g.mu, |p| sum(Poisson::new(p).log_ccdf(&k)));
    }
    // accurate in the upper tail: P(X > 60 | mu = 1) = e^-1 Σₙ₌₆₁ 1 / n! (about 1e-84)
    let tail = at(Poisson::new(s(1.0)).log_ccdf(&IntTensor::scalar(60)));
    let mut term = 1.0;
    let mut series = 0.0;
    for n in 62..90 {
        series += term;
        term /= n as f64;
    }
    let expected = -1.0 - keanu_tensor::special::ln_gamma(62.0) + f64::ln(series);
    assert!(
        (tail - expected).abs() < 1e-10 * expected.abs(),
        "{tail} vs {expected}"
    );
}

#[test]
fn binomial() {
    for (p, n) in [(0.3, 12), (0.97, 40), (0.001, 1000)] {
        let d = Binomial::new(s(p), IntTensor::scalar(n));
        let i = |k: i64| IntTensor::scalar(k);
        check_discrete(
            "binomial",
            |k| at(d.log_prob(&i(k))),
            |k| at(d.log_cdf(&i(k))),
            |k| at(d.log_ccdf(&i(k))),
            n,
        );
        assert_eq!(at(d.log_cdf(&i(n))), 0.0);
        assert_eq!(at(d.log_ccdf(&i(n))), f64::NEG_INFINITY);
        let mean = (p * n as f64) as i64;
        let k = IntTensor::vector(vec![0, mean, (mean + 2).min(n - 1)]);
        let pp = s(p);
        let nn = IntTensor::scalar(n);
        let g = Binomial::new(&pp, &nn).d_log_cdf(&k).unwrap();
        check_gradient("p", &pp, &g.p, |q| sum(Binomial::new(q, &nn).log_cdf(&k)));
        let g = Binomial::new(&pp, &nn).d_log_ccdf(&k).unwrap();
        check_gradient("p", &pp, &g.p, |q| sum(Binomial::new(q, &nn).log_ccdf(&k)));
    }
}

#[test]
fn inverse_cdfs_invert_the_cdfs() {
    let p = v(&[1e-300, 1e-10, 0.2, 0.5, 0.9]);
    let check = |name: &str,
                 inverse: Result<DoubleTensor>,
                 inverse_ccdf: Result<DoubleTensor>,
                 log_cdf: &dyn Fn(&DoubleTensor) -> Result<DoubleTensor>,
                 log_ccdf: &dyn Fn(&DoubleTensor) -> Result<DoubleTensor>| {
        let x = inverse.unwrap();
        let back = log_cdf(&x).unwrap();
        let y = inverse_ccdf.unwrap();
        let back_ccdf = log_ccdf(&y).unwrap();
        for i in 0..p.len() {
            let expected = p.as_slice()[i].ln();
            let relative = 1e-9 * expected.abs().max(1.0);
            assert!(
                (back.as_slice()[i] - expected).abs() < relative,
                "{name}: log_cdf(inverse_cdf({})) = {}",
                p.as_slice()[i],
                back.as_slice()[i]
            );
            assert!(
                (back_ccdf.as_slice()[i] - expected).abs() < relative,
                "{name}: log_ccdf(inverse_ccdf({})) = {}",
                p.as_slice()[i],
                back_ccdf.as_slice()[i]
            );
        }
    };
    let d = Gaussian::new(s(1.0), s(2.0));
    check(
        "gaussian",
        d.inverse_cdf(&p),
        d.inverse_ccdf(&p),
        &|x| d.log_cdf(x),
        &|x| d.log_ccdf(x),
    );
    let d = LogNormal::new(s(0.5), s(0.4));
    check(
        "log normal",
        d.inverse_cdf(&p),
        d.inverse_ccdf(&p),
        &|x| d.log_cdf(x),
        &|x| d.log_ccdf(x),
    );
    let d = Cauchy::new(s(-1.0), s(0.3));
    check(
        "cauchy",
        d.inverse_cdf(&p),
        d.inverse_ccdf(&p),
        &|x| d.log_cdf(x),
        &|x| d.log_ccdf(x),
    );
    let d = Laplace::new(s(2.0), s(0.5));
    check(
        "laplace",
        d.inverse_cdf(&p),
        d.inverse_ccdf(&p),
        &|x| d.log_cdf(x),
        &|x| d.log_ccdf(x),
    );
    let d = Logistic::new(s(0.0), s(1.5));
    check(
        "logistic",
        d.inverse_cdf(&p),
        d.inverse_ccdf(&p),
        &|x| d.log_cdf(x),
        &|x| d.log_ccdf(x),
    );
    let d = Exponential::new(s(3.0));
    check(
        "exponential",
        d.inverse_cdf(&p),
        d.inverse_ccdf(&p),
        &|x| d.log_cdf(x),
        &|x| d.log_ccdf(x),
    );
    // the uniform's inverse loses relative precision at tiny p near a non-zero bound: check moderate p only
    let d = Uniform::new(s(-1.0), s(3.0));
    let q = v(&[0.2, 0.5, 0.9]);
    let x = d.inverse_cdf(&q).unwrap();
    for (x, expected) in x.as_slice().iter().zip([-0.2, 1.0, 2.6]) {
        assert!((x - expected).abs() < 1e-12);
    }
    let y = d.inverse_ccdf(&q).unwrap();
    assert!((y.as_slice()[0] - 2.2).abs() < 1e-12);
}

#[test]
fn weibull() {
    for (alpha, beta) in [(1.8, 1.4), (0.7, 2.0)] {
        let d = Weibull::new(s(alpha), s(beta));
        Continuous {
            name: "weibull",
            log_pdf: &|x| at(d.log_prob(&s(x))),
            log_cdf: &|x| at(d.log_cdf(&s(x))),
            log_ccdf: &|x| at(d.log_ccdf(&s(x))),
        }
        .check(&[1e-9, 0.3, 1.0, 2.5, 8.0], 1e-5);
        let (x, a, b) = (v(&[0.4, 1.3, 3.1]), s(alpha), s(beta));
        for upper in [false, true] {
            let f = |d: &Weibull<'_>, x: &DoubleTensor| {
                if upper { d.log_ccdf(x) } else { d.log_cdf(x) }
            };
            let d = Weibull::new(&a, &b);
            let g = if upper {
                d.d_log_ccdf(&x)
            } else {
                d.d_log_cdf(&x)
            }
            .unwrap();
            check_gradient("x", &x, &g.x, |x| sum(f(&Weibull::new(&a, &b), x)));
            check_gradient("alpha", &a, &g.alpha, |p| sum(f(&Weibull::new(p, &b), &x)));
            check_gradient("beta", &b, &g.beta, |p| sum(f(&Weibull::new(&a, p), &x)));
        }
    }
    let p = v(&[1e-300, 1e-10, 0.2, 0.5, 0.9]);
    let d = Weibull::new(s(1.5), s(2.0));
    let x = d.inverse_cdf(&p).unwrap();
    let back = d.log_cdf(&x).unwrap();
    let y = d.inverse_ccdf(&p).unwrap();
    let back_ccdf = d.log_ccdf(&y).unwrap();
    for i in 0..p.len() {
        let expected = p.as_slice()[i].ln();
        assert!((back.as_slice()[i] - expected).abs() < 1e-9 * expected.abs().max(1.0));
        assert!((back_ccdf.as_slice()[i] - expected).abs() < 1e-9 * expected.abs().max(1.0));
    }
}

#[test]
fn negative_binomial() {
    for (mu, alpha) in [(4.2, 1.7), (0.5, 30.0), (20.0, 0.4)] {
        let d = NegativeBinomial::new(s(mu), s(alpha));
        let i = |k: i64| IntTensor::scalar(k);
        check_discrete(
            "negative binomial",
            |k| at(d.log_prob(&i(k))),
            |k| at(d.log_cdf(&i(k))),
            |k| at(d.log_ccdf(&i(k))),
            (mu * 6.0) as i64 + 20,
        );
        let k = IntTensor::vector(vec![0, (mu as i64) / 2, mu as i64 + 3]);
        let (m, a) = (s(mu), s(alpha));
        for upper in [false, true] {
            let f = |d: &NegativeBinomial<'_>, k: &IntTensor| {
                if upper { d.log_ccdf(k) } else { d.log_cdf(k) }
            };
            let d = NegativeBinomial::new(&m, &a);
            let g = if upper {
                d.d_log_ccdf(&k)
            } else {
                d.d_log_cdf(&k)
            }
            .unwrap();
            check_gradient("mu", &m, &g.mu, |p| {
                sum(f(&NegativeBinomial::new(p, &a), &k))
            });
            check_gradient("alpha", &a, &g.alpha, |p| {
                sum(f(&NegativeBinomial::new(&m, p), &k))
            });
        }
    }
}

/// Asserts a log CDF in a far tail (where the probability underflows) matches its mpmath value to 1e-10 relative.
fn assert_tail(name: &str, actual: f64, expected: f64) {
    assert!(
        ((actual - expected) / expected).abs() < 1e-10,
        "{name}: {actual} vs mpmath {expected}"
    );
}

/// Asserts every element of a gradient is finite.
fn assert_finite(name: &str, gradient: &DoubleTensor) {
    assert!(
        gradient.as_slice().iter().all(|g| g.is_finite()),
        "{name}: {:?}",
        gradient.as_slice()
    );
}

#[test]
fn log_cdfs_of_incomplete_gamma_distributions_in_the_far_tails() {
    // reference values ln P(a, x) and ln Q(a, x) from mpmath (50 digits); the probabilities underflow, and the
    // log CDFs used to be -infinity with NaN gradients
    let mu = s(1.0);
    let k = IntTensor::scalar(200);
    let poisson = Poisson::new(&mu);
    assert_tail(
        "Poisson(1) ccdf at 200",
        at(poisson.log_ccdf(&k)),
        -869.5303294330408,
    );
    let g = poisson.d_log_ccdf(&k).unwrap();
    assert_finite("Poisson ccdf mu", &g.mu);
    check_gradient("mu", &mu, &g.mu, |p| sum(Poisson::new(p).log_ccdf(&k)));

    let mu = s(1000.0);
    let k = IntTensor::scalar(10);
    let poisson = Poisson::new(&mu);
    assert_tail(
        "Poisson(1000) cdf at 10",
        at(poisson.log_cdf(&k)),
        -946.0168196296328,
    );
    let g = poisson.d_log_cdf(&k).unwrap();
    assert_finite("Poisson cdf mu", &g.mu);
    check_gradient("mu", &mu, &g.mu, |p| sum(Poisson::new(p).log_cdf(&k)));

    for (theta, k, x, upper, expected) in [
        (1.0, 2.0, 800.0, true, -793.3141390529316),
        (0.5, 300.0, 0.5, false, -1415.9025221917655),
    ] {
        let (th, kk, xx) = (s(theta), s(k), s(x));
        let f = |d: &Gamma<'_>, x: &DoubleTensor| {
            if upper { d.log_ccdf(x) } else { d.log_cdf(x) }
        };
        let d = Gamma::new(&th, &kk);
        assert_tail("Gamma", at(f(&d, &xx)), expected);
        let g = if upper {
            d.d_log_ccdf(&xx)
        } else {
            d.d_log_cdf(&xx)
        }
        .unwrap();
        for (name, gradient) in [("x", &g.x), ("theta", &g.theta), ("k", &g.k)] {
            assert_finite(name, gradient);
        }
        check_gradient("x", &xx, &g.x, |x| sum(f(&Gamma::new(&th, &kk), x)));
        check_gradient("theta", &th, &g.theta, |p| sum(f(&Gamma::new(p, &kk), &xx)));
        check_gradient("k", &kk, &g.k, |p| sum(f(&Gamma::new(&th, p), &xx)));
    }
}

#[test]
fn log_cdfs_of_incomplete_beta_distributions_in_the_far_tails() {
    // reference values from mpmath (50 digits) of ln I_x(a, b) for the parameters each CDF maps to
    for (mu, alpha, k, upper, expected) in [
        (2.0, 3.0, 1000, true, -905.0999862937989),
        (1e4, 200.0, 0, false, -786.3651265448651),
    ] {
        let (m, a, kk) = (s(mu), s(alpha), IntTensor::scalar(k));
        let f = |d: &NegativeBinomial<'_>, k: &IntTensor| {
            if upper { d.log_ccdf(k) } else { d.log_cdf(k) }
        };
        let d = NegativeBinomial::new(&m, &a);
        assert_tail("negative binomial", at(f(&d, &kk)), expected);
        let g = if upper {
            d.d_log_ccdf(&kk)
        } else {
            d.d_log_cdf(&kk)
        }
        .unwrap();
        assert_finite("mu", &g.mu);
        assert_finite("alpha", &g.alpha);
        check_gradient("mu", &m, &g.mu, |p| {
            sum(f(&NegativeBinomial::new(p, &a), &kk))
        });
        check_gradient("alpha", &a, &g.alpha, |p| {
            sum(f(&NegativeBinomial::new(&m, p), &kk))
        });
    }

    for (a, b, x, upper, expected) in [
        (2.0, 3.0, 1e-200, false, -919.2422777283903),
        (300.0, 2.0, 0.01, false, -1375.853962309922),
        (2.0, 300.0, 0.99, true, -1375.8539623099218),
    ] {
        let (aa, bb, lo, hi, xx) = (s(a), s(b), s(0.0), s(1.0), s(x));
        let f = |d: &Beta<'_>, x: &DoubleTensor| {
            if upper { d.log_ccdf(x) } else { d.log_cdf(x) }
        };
        let d = Beta::new(&aa, &bb, &lo, &hi);
        assert_tail("Beta", at(f(&d, &xx)), expected);
        let g = if upper {
            d.d_log_ccdf(&xx)
        } else {
            d.d_log_cdf(&xx)
        }
        .unwrap();
        for (name, gradient) in [("x", &g.x), ("alpha", &g.alpha), ("beta", &g.beta)] {
            assert_finite(name, gradient);
        }
        if x > 1e-100 {
            check_gradient("x", &xx, &g.x, |x| {
                sum(f(&Beta::new(&aa, &bb, &lo, &hi), x))
            });
        }
        check_gradient("alpha", &aa, &g.alpha, |p| {
            sum(f(&Beta::new(p, &bb, &lo, &hi), &xx))
        });
        check_gradient("beta", &bb, &g.beta, |p| {
            sum(f(&Beta::new(&aa, p, &lo, &hi), &xx))
        });
    }

    for (p, k, upper, expected) in [
        (0.001, 900, true, -5904.271493662422),
        (0.999, 100, false, -5895.1664025790915),
    ] {
        let (pp, nn, kk) = (s(p), IntTensor::scalar(1000), IntTensor::scalar(k));
        let f = |d: &Binomial<'_>, k: &IntTensor| {
            if upper { d.log_ccdf(k) } else { d.log_cdf(k) }
        };
        let d = Binomial::new(&pp, &nn);
        assert_tail("binomial", at(f(&d, &kk)), expected);
        let g = if upper {
            d.d_log_ccdf(&kk)
        } else {
            d.d_log_cdf(&kk)
        }
        .unwrap();
        assert_finite("p", &g.p);
        check_gradient("p", &pp, &g.p, |q| sum(f(&Binomial::new(q, &nn), &kk)));
    }

    let t = StudentT::new(IntTensor::scalar(5));
    let x = s(-1e70);
    assert_tail("Student t", at(t.log_cdf(&x)), -803.6545263121025);
    assert_tail("Student t", at(t.log_ccdf(&s(1e70))), -803.6545263121025);
    // d/dx ln P(T <= x) ~ -v/x in the lower tail (P ~ |x|^-v)
    let g = t.d_log_cdf(&x).unwrap().x.as_slice()[0];
    assert!((g - 5e-70).abs() < 1e-6 * 5e-70, "{g}");
}
