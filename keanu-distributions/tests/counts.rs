//! The beta-binomial and the logit and log parameterisations: their masses sum to 1 and match closed forms, they
//! stay finite at extreme logits, their gradients (and the beta-binomial's tail derivatives) match finite
//! differences, and their draws have the right moments.

use keanu_distributions::{
    BernoulliLogit, BetaBinomial, Binomial, BinomialLogit, CategoricalLogit, KeanuRng, Poisson,
    PoissonLog,
};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};

fn s(v: f64) -> DoubleTensor {
    DoubleTensor::scalar(v)
}

fn ks(n: i64) -> IntTensor {
    IntTensor::vector((0..=n).collect())
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{what}: {actual} vs {expected}"
    );
}

fn total(log_probs: &DoubleTensor) -> f64 {
    log_probs.as_slice().iter().map(|l| l.exp()).sum()
}

fn ln_factorial(k: i64) -> f64 {
    (1..=k).map(|i| (i as f64).ln()).sum()
}

#[test]
fn masses_sum_to_one() {
    for (n, a, b) in [
        (12, 0.7, 2.5),
        (5, 1.0, 1.0),
        (30, 40.0, 3.0),
        (0, 2.0, 2.0),
    ] {
        let d = BetaBinomial::new(IntTensor::scalar(n), s(a), s(b));
        close(
            total(&d.log_prob(&ks(n)).unwrap()),
            1.0,
            1e-12,
            "beta-binomial",
        );
    }
    for eta in [-3.0, 0.4, 5.0] {
        let d = BinomialLogit::new(s(eta), IntTensor::scalar(9));
        close(
            total(&d.log_prob(&ks(9)).unwrap()),
            1.0,
            1e-12,
            "binomial logit",
        );
        let p = PoissonLog::new(s(eta));
        close(
            total(&p.log_prob(&ks(400)).unwrap()),
            1.0,
            1e-12,
            "poisson log",
        );
        let b = BernoulliLogit::new(s(eta));
        close(
            total(&b.log_prob(&BoolTensor::vector(vec![true, false])).unwrap()),
            1.0,
            1e-14,
            "bernoulli logit",
        );
    }
    let c = CategoricalLogit::new([s(0.3), s(-1.2), s(2.0), s(f64::NEG_INFINITY)]).unwrap();
    let log_probs = c.log_prob(&ks(3)).unwrap();
    close(total(&log_probs), 1.0, 1e-14, "categorical logit");
    assert_eq!(log_probs.as_slice()[3], f64::NEG_INFINITY);
    assert_eq!(
        c.log_prob(&IntTensor::vector(vec![-1, 4]))
            .unwrap()
            .as_slice(),
        &[f64::NEG_INFINITY; 2]
    );
}

#[test]
fn the_beta_binomial_matches_its_closed_form_and_limits() {
    // P(k) = C(n, k) B(k + α, n − k + β) / B(α, β), from factorials and Γ at integer and half-integer points
    let (n, a, b) = (6, 2.0, 3.0);
    let d = BetaBinomial::new(IntTensor::scalar(n), s(a), s(b));
    let log_probs = d.log_prob(&ks(n)).unwrap();
    for k in 0..=n {
        // B(x, y) = (x − 1)!(y − 1)!/(x + y − 1)! for whole x, y
        let beta =
            |x: i64, y: i64| ln_factorial(x - 1) + ln_factorial(y - 1) - ln_factorial(x + y - 1);
        let expected = ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
            + beta(k + 2, n - k + 3)
            - beta(2, 3);
        close(
            log_probs.as_slice()[k as usize],
            expected,
            1e-12,
            "closed form",
        );
    }
    // α = β = 1 is uniform on 0..=n
    let uniform = BetaBinomial::new(IntTensor::scalar(n), s(1.0), s(1.0));
    for lp in uniform.log_prob(&ks(n)).unwrap().as_slice() {
        close(*lp, -(7f64).ln(), 1e-12, "uniform");
    }
    // large shapes with p = α/(α + β) approach the binomial
    let wide = BetaBinomial::new(IntTensor::scalar(n), s(3e6), s(7e6));
    let binomial = Binomial::new(s(0.3), IntTensor::scalar(n));
    let (bb, bin) = (
        wide.log_prob(&ks(n)).unwrap(),
        binomial.log_prob(&ks(n)).unwrap(),
    );
    for (x, y) in bb.as_slice().iter().zip(bin.as_slice()) {
        close(*x, *y, 1e-5, "binomial limit");
    }
    // outside the support and with invalid shapes the mass is 0
    let outside = d.log_prob(&IntTensor::vector(vec![-1, 7])).unwrap();
    assert_eq!(outside.as_slice(), &[f64::NEG_INFINITY; 2]);
    let invalid = BetaBinomial::new(IntTensor::scalar(n), s(-1.0), s(2.0));
    assert_eq!(
        invalid.log_prob(&IntTensor::scalar(2)).unwrap().as_slice(),
        &[f64::NEG_INFINITY]
    );
}

#[test]
fn the_beta_binomial_tails_are_sums_of_the_mass() {
    let (n, a, b) = (10, 1.3, 0.6);
    let d = BetaBinomial::new(IntTensor::scalar(n), s(a), s(b));
    let points = IntTensor::vector((-1..=11).collect());
    let mass: Vec<f64> = d
        .log_prob(&ks(n))
        .unwrap()
        .as_slice()
        .iter()
        .map(|l| l.exp())
        .collect();
    let cdf = d.log_cdf(&points).unwrap();
    let ccdf = d.log_ccdf(&points).unwrap();
    for (i, k) in (-1..=11).enumerate() {
        let below: f64 = mass.iter().take((k + 1).clamp(0, 11) as usize).sum();
        close(cdf.as_slice()[i].exp(), below, 1e-12, "cdf");
        close(ccdf.as_slice()[i].exp(), 1.0 - below, 1e-12, "survival");
    }
    // the derivatives, first and second, match finite differences in α and β
    let h = 1e-5;
    for k in [0, 3, 8] {
        for upper in [false, true] {
            let tail = |a: f64, b: f64| BetaBinomial::log_tail(k, n, a, b, upper);
            let at = tail(a, b);
            let da = (tail(a + h, b).log_probability - tail(a - h, b).log_probability) / (2.0 * h);
            let db = (tail(a, b + h).log_probability - tail(a, b - h).log_probability) / (2.0 * h);
            close(at.gradient[0], da, 1e-7, "d alpha");
            close(at.gradient[1], db, 1e-7, "d beta");
            let daa = (tail(a + h, b).gradient[0] - tail(a - h, b).gradient[0]) / (2.0 * h);
            let dab = (tail(a, b + h).gradient[0] - tail(a, b - h).gradient[0]) / (2.0 * h);
            let dbb = (tail(a, b + h).gradient[1] - tail(a, b - h).gradient[1]) / (2.0 * h);
            close(at.hessian[0], daa, 1e-6, "d² alpha");
            close(at.hessian[1], dab, 1e-6, "d² alpha beta");
            close(at.hessian[2], dbb, 1e-6, "d² beta");
            // and the tensor gradient is the same
            let g = if upper {
                d.d_log_ccdf(&IntTensor::scalar(k)).unwrap()
            } else {
                d.d_log_cdf(&IntTensor::scalar(k)).unwrap()
            };
            assert_eq!(g.alpha.as_slice()[0], at.gradient[0]);
        }
    }
}

#[test]
fn extreme_logits_stay_finite_and_accurate() {
    let t = BoolTensor::vector(vec![true, false]);
    // ln σ(700) = −ln(1 + e^−700) ≈ −e^−700, ln σ(−700) = −700 − ln(1 + e^−700)
    let up = BernoulliLogit::new(s(700.0)).log_prob(&t).unwrap();
    assert_eq!(up.as_slice()[1], -700.0);
    assert!(up.as_slice()[0] <= 0.0 && up.as_slice()[0] > -1e-300);
    let down = BernoulliLogit::new(s(-700.0)).log_prob(&t).unwrap();
    assert_eq!(down.as_slice()[0], -700.0);
    let g = BernoulliLogit::new(s(-700.0)).d_log_prob(&t).unwrap();
    assert_eq!(g.logit.as_slice(), &[1.0, -(-700f64).exp()]);

    // k ln σ(η) + (n − k) ln σ(−η): exactly (k − n)·700 at η = 700
    let n = IntTensor::scalar(10);
    let k = IntTensor::vector(vec![0, 3, 10]);
    let high = BinomialLogit::new(s(700.0), n.clone())
        .log_prob(&k)
        .unwrap();
    let ln_choose = |k: i64| ln_factorial(10) - ln_factorial(k) - ln_factorial(10 - k);
    close(
        high.as_slice()[0],
        -7000.0,
        1e-14,
        "binomial logit at 700, k = 0",
    );
    close(high.as_slice()[1], ln_choose(3) - 4900.0, 1e-14, "k = 3");
    assert!(high.as_slice()[2].abs() < 1e-14, "{}", high.as_slice()[2]);
    let low = BinomialLogit::new(s(-700.0), n.clone())
        .log_prob(&k)
        .unwrap();
    close(
        low.as_slice()[1],
        ln_choose(3) - 2100.0,
        1e-14,
        "binomial logit at −700",
    );
    let g = BinomialLogit::new(s(700.0), n).d_log_prob(&k).unwrap();
    assert_eq!(g.logit.as_slice(), &[-10.0, -7.0, 10.0 * (-700f64).exp()]);

    // kη − e^η − ln k!
    let counts = IntTensor::vector(vec![0, 4]);
    let big = PoissonLog::new(s(700.0)).log_prob(&counts).unwrap();
    assert!(big.as_slice().iter().all(|l| l.is_finite()));
    close(
        big.as_slice()[0],
        -(700f64).exp(),
        1e-14,
        "poisson log at 700",
    );
    let small = PoissonLog::new(s(-700.0)).log_prob(&counts).unwrap();
    close(small.as_slice()[0], -(-700f64).exp(), 1e-14, "k = 0");
    close(
        small.as_slice()[1],
        -2800.0 - ln_factorial(4),
        1e-14,
        "poisson log at −700",
    );
    // and it is the Poisson of e^η where both are representable
    let moderate = PoissonLog::new(s(1.7)).log_prob(&counts).unwrap();
    let poisson = Poisson::new(s(1.7f64.exp())).log_prob(&counts).unwrap();
    for (x, y) in moderate.as_slice().iter().zip(poisson.as_slice()) {
        close(*x, *y, 1e-12, "poisson");
    }

    // η_x − logsumexp(η)
    let c = CategoricalLogit::new([s(700.0), s(-700.0), s(0.0)]).unwrap();
    let lp = c.log_prob(&ks(2)).unwrap();
    close(lp.as_slice()[0], 0.0, 1e-14, "the dominant category");
    close(lp.as_slice()[1], -1400.0, 1e-14, "a negligible category");
    close(lp.as_slice()[2], -700.0, 1e-14, "the middle category");
}

#[test]
fn gradients_match_finite_differences() {
    let h = 1e-6;
    let fd = |f: &dyn Fn(f64) -> DoubleTensor, x: f64| -> Vec<f64> {
        let (p, m) = (f(x + h), f(x - h));
        p.as_slice()
            .iter()
            .zip(m.as_slice())
            .map(|(p, m)| (p - m) / (2.0 * h))
            .collect()
    };
    let check = |actual: &DoubleTensor, expected: Vec<f64>, what: &str| {
        for (a, e) in actual.as_slice().iter().zip(&expected) {
            close(*a, *e, 1e-6, what);
        }
    };
    let n = IntTensor::scalar(7);
    let k = IntTensor::vector(vec![0, 2, 5, 7]);
    let (a, b) = (1.6, 0.8);
    let g = BetaBinomial::new(&n, s(a), s(b)).d_log_prob(&k).unwrap();
    check(
        &g.alpha,
        fd(
            &|a| BetaBinomial::new(&n, s(a), s(b)).log_prob(&k).unwrap(),
            a,
        ),
        "beta-binomial alpha",
    );
    check(
        &g.beta,
        fd(
            &|b| BetaBinomial::new(&n, s(a), s(b)).log_prob(&k).unwrap(),
            b,
        ),
        "beta-binomial beta",
    );
    for eta in [-2.5, 0.3, 4.0] {
        let g = BinomialLogit::new(s(eta), &n).d_log_prob(&k).unwrap();
        check(
            &g.logit,
            fd(&|e| BinomialLogit::new(s(e), &n).log_prob(&k).unwrap(), eta),
            "binomial logit",
        );
        let g = PoissonLog::new(s(eta)).d_log_prob(&k).unwrap();
        check(
            &g.log_mu,
            fd(&|e| PoissonLog::new(s(e)).log_prob(&k).unwrap(), eta),
            "poisson log",
        );
        let x = BoolTensor::vector(vec![true, false]);
        let g = BernoulliLogit::new(s(eta)).d_log_prob(&x).unwrap();
        check(
            &g.logit,
            fd(&|e| BernoulliLogit::new(s(e)).log_prob(&x).unwrap(), eta),
            "bernoulli logit",
        );
    }
    let logits = [0.4, -0.9, 1.5];
    let categories = ks(2);
    let g = CategoricalLogit::new(logits.map(s))
        .unwrap()
        .d_log_prob(&categories)
        .unwrap();
    for j in 0..3 {
        let f = |x: f64| {
            let mut l = logits;
            l[j] = x;
            CategoricalLogit::new(l.map(s))
                .unwrap()
                .log_prob(&categories)
                .unwrap()
        };
        check(&g[j], fd(&f, logits[j]), "categorical logit");
    }
}

fn moments(values: &IntTensor) -> (f64, f64) {
    let v: Vec<f64> = values.as_slice().iter().map(|&k| k as f64).collect();
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    (mean, var)
}

#[test]
fn draws_have_the_right_moments() {
    let mut rng = KeanuRng::seed_from_u64(11);
    let draws = 40_000;
    // mean nα/(α + β), variance nαβ(α + β + n)/((α + β)²(α + β + 1))
    let (n, a, b) = (20, 2.0, 5.0);
    let d = BetaBinomial::new(IntTensor::scalar(n), s(a), s(b));
    let x = d.sample(&[draws], &mut rng).unwrap();
    assert!(x.as_slice().iter().all(|&k| (0..=n).contains(&k)));
    let (mean, var) = moments(&x);
    let nf = n as f64;
    close(mean, nf * a / (a + b), 0.01, "beta-binomial mean");
    let expected = nf * a * b * (a + b + nf) / ((a + b).powi(2) * (a + b + 1.0));
    close(var, expected, 0.03, "beta-binomial variance");

    let eta: f64 = -0.7;
    let p = 1.0 / (1.0 + (-eta).exp());
    let (mean, var) = moments(
        &BinomialLogit::new(s(eta), IntTensor::scalar(12))
            .sample(&[draws], &mut rng)
            .unwrap(),
    );
    close(mean, 12.0 * p, 0.01, "binomial logit mean");
    close(var, 12.0 * p * (1.0 - p), 0.03, "binomial logit variance");

    let (mean, var) = moments(&PoissonLog::new(s(1.2)).sample(&[draws], &mut rng).unwrap());
    close(mean, 1.2f64.exp(), 0.01, "poisson log mean");
    close(var, 1.2f64.exp(), 0.03, "poisson log variance");

    let flips = BernoulliLogit::new(s(eta))
        .sample(&[draws], &mut rng)
        .unwrap();
    let share = flips.as_slice().iter().filter(|&&f| f).count() as f64 / draws as f64;
    close(share, p, 0.01, "bernoulli logit");

    let logits = [0.4, -0.9, 1.5];
    let c = CategoricalLogit::new(logits.map(s)).unwrap();
    let categories = c.sample(&[draws], &mut rng).unwrap();
    let z: f64 = logits.iter().map(|l| l.exp()).sum();
    for (j, l) in logits.iter().enumerate() {
        let share = categories
            .as_slice()
            .iter()
            .filter(|&&k| k == j as i64)
            .count() as f64
            / draws as f64;
        close(share, l.exp() / z, 0.01, "categorical logit");
    }
}

#[test]
fn invalid_parameters_are_sampling_errors() {
    let mut rng = KeanuRng::seed_from_u64(1);
    assert!(
        BetaBinomial::new(IntTensor::scalar(3), s(0.0), s(1.0))
            .sample(&[], &mut rng)
            .is_err()
    );
    assert!(
        BetaBinomial::new(IntTensor::scalar(-1), s(1.0), s(1.0))
            .sample(&[], &mut rng)
            .is_err()
    );
    assert!(
        BernoulliLogit::new(s(f64::NAN))
            .sample(&[], &mut rng)
            .is_err()
    );
    assert!(PoissonLog::new(s(800.0)).sample(&[], &mut rng).is_err());
    assert!(
        CategoricalLogit::new([s(f64::NEG_INFINITY), s(f64::NEG_INFINITY)])
            .unwrap()
            .sample(&[], &mut rng)
            .is_err()
    );
    assert!(CategoricalLogit::new(Vec::<DoubleTensor>::new()).is_err());
}
