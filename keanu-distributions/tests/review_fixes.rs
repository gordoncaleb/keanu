//! Regression tests for the 2026-11-04 code review fixes.

use keanu_distributions::*;
use keanu_tensor::{DoubleTensor, IntTensor};

mod common;
use common::{assert_close, s, v};

fn all_distinct(values: &[f64]) -> bool {
    values
        .iter()
        .enumerate()
        .all(|(i, a)| values[i + 1..].iter().all(|b| a != b))
}

#[test]
fn multinomial_log_prob_is_minus_infinity_outside_the_support() {
    let multinomial = Multinomial::new(IntTensor::scalar(5), v(&[0.5, 0.5])).unwrap();
    // counts summing to less than n, a negative count, a count above n
    for x in [vec![1, 1], vec![-1, 6], vec![6, -1], vec![3, 3]] {
        let lp = multinomial.log_prob(&IntTensor::vector(x.clone())).unwrap();
        assert_eq!(lp.as_slice(), &[f64::NEG_INFINITY], "x = {x:?}");
    }
    // valid: C(5, 2) / 2^5
    let lp = multinomial
        .log_prob(&IntTensor::vector(vec![2, 3]))
        .unwrap();
    assert_close(lp.as_slice()[0], (10.0f64 / 32.0).ln(), 1e-12, "value");
    // per batch element
    let batched = Multinomial::new(IntTensor::vector(vec![5, 2]), v(&[0.5, 0.5])).unwrap();
    let x = IntTensor::new(&[2, 2], vec![2, 3, 1, 0]).unwrap();
    let lp = batched.log_prob(&x).unwrap();
    assert_close(lp.as_slice()[0], (10.0f64 / 32.0).ln(), 1e-12, "value");
    assert_eq!(lp.as_slice()[1], f64::NEG_INFINITY);
    // the wrong number of categories is an error, not a broadcast
    assert!(
        multinomial
            .log_prob(&IntTensor::vector(vec![1, 1, 3]))
            .is_err()
    );
    assert!(
        multinomial
            .log_prob(&IntTensor::new(&[2, 1], vec![5, 5]).unwrap())
            .is_err()
    );
}

#[test]
fn logistic_is_stable_far_in_the_tail() {
    // m = (mu - x) / s = 800: lp = m - ln s - 2 ln(1 + e^m) = -800 - ln 0.01
    let logistic = Logistic::new(s(0.0), s(0.01));
    let x = s(-8.0);
    let lp = logistic.log_prob(&x).unwrap().as_slice()[0];
    assert_close(lp, -800.0 - 0.01f64.ln(), 1e-9, "value");
    let g = logistic.d_log_prob(&x).unwrap();
    // d/dx = (2 sigmoid(m) - 1) / s, d/ds = (2 sigmoid(m) m - m - 1) / s
    assert_close(g.x.as_slice()[0], 100.0, 1e-9, "value");
    assert_close(g.mu.as_slice()[0], -100.0, 1e-9, "value");
    assert_close(g.s.as_slice()[0], 79_900.0, 1e-6, "value");
    // and the other tail
    let lp = logistic.log_prob(&s(8.0)).unwrap().as_slice()[0];
    assert_close(lp, -800.0 - 0.01f64.ln(), 1e-9, "value");
    let g = logistic.d_log_prob(&s(8.0)).unwrap();
    assert_close(g.x.as_slice()[0], -100.0, 1e-9, "value");
    assert_close(g.s.as_slice()[0], 79_900.0, 1e-6, "value");
}

#[test]
fn multivariate_gaussian_gradients_have_the_parameters_shapes() {
    let x = DoubleTensor::new(&[3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let mu = DoubleTensor::new(&[1, 2], vec![0.0, 0.0]).unwrap();
    let covariance = DoubleTensor::new(&[1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let g = MultivariateGaussian::new(mu.clone(), covariance)
        .d_log_prob(&x)
        .unwrap();
    assert_eq!(g.x.shape(), &[3, 2]);
    assert_eq!(g.mu.shape(), &[1, 2]);
    assert_eq!(g.covariance.shape(), &[1, 2, 2]);
    // sum over the rows of x - mu, and 0.5 (sum of outer products - 3 I)
    assert_eq!(g.mu.as_slice(), &[2.0, 2.0]);
    assert_eq!(g.covariance.as_slice(), &[-0.5, 0.5, 0.5, -0.5]);

    // x broadcast against a batch of means
    let g = MultivariateGaussian::new(
        x.clone(),
        DoubleTensor::new(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
    )
    .d_log_prob(&mu)
    .unwrap();
    assert_eq!(g.x.shape(), &[1, 2]);
    assert_eq!(g.x.as_slice(), &[2.0, 2.0]);
    assert_eq!(g.mu.shape(), &[3, 2]);
}

#[test]
fn samplers_draw_independently_for_each_broadcast_element() {
    let three = v(&[1.0, 1.0, 1.0]);
    let seeded = || KeanuRng::seed_from_u64(11);
    let check = |name: &str, small: DoubleTensor, full: DoubleTensor| {
        assert_eq!(small.shape(), &[3], "{name}");
        assert!(all_distinct(small.as_slice()), "{name}: {small:?}");
        // the requested shape being the broadcast shape uses the same stream
        assert_eq!(small, full, "{name}");
    };
    macro_rules! continuous {
        ($name:expr, $d:expr) => {{
            let d = $d;
            check(
                $name,
                d.sample(&[], &mut seeded()).unwrap(),
                d.sample(&[3], &mut seeded()).unwrap(),
            );
        }};
    }
    continuous!("gaussian", Gaussian::new(s(0.0), three.clone()));
    continuous!("half gaussian", HalfGaussian::new(three.clone()));
    continuous!("log normal", LogNormal::new(s(0.0), three.clone()));
    continuous!("cauchy", Cauchy::new(s(0.0), three.clone()));
    continuous!("half cauchy", HalfCauchy::new(three.clone()));
    continuous!("beta", Beta::standard(s(2.0), three.clone()));
    continuous!(
        "beta (bounds only)",
        Beta::new(s(2.0), s(3.0), v(&[0.0, 0.0, 0.0]), s(1.0))
    );
    continuous!("exponential", Exponential::new(three.clone()));
    continuous!("logistic", Logistic::new(s(0.0), three.clone()));
    continuous!("pareto", Pareto::new(three.clone(), s(2.0)));
    continuous!("student t", StudentT::new(IntTensor::vector(vec![3, 3, 3])));
    continuous!(
        "triangular",
        Triangular::new(v(&[0.0, 0.0, 0.0]), s(1.0), s(0.5))
    );
    continuous!("uniform", Uniform::new(v(&[0.0, 0.0, 0.0]), s(1.0)));

    let twenty = DoubleTensor::full(&[20], 0.5);
    let bernoulli = Bernoulli::new(twenty.clone());
    let small = bernoulli.sample(&[], &mut seeded()).unwrap();
    assert_eq!(small.shape(), &[20]);
    assert!(!small.all() && small.as_slice().iter().any(|&b| b));
    assert_eq!(small, bernoulli.sample(&[20], &mut seeded()).unwrap());

    let geometric = Geometric::new(DoubleTensor::full(&[20], 0.01));
    let small = geometric.sample(&[], &mut seeded()).unwrap().to_double();
    assert_eq!(small.shape(), &[20]);
    assert!(small.as_slice().iter().any(|&k| k != small.as_slice()[0]));
    assert_eq!(
        small,
        geometric.sample(&[20], &mut seeded()).unwrap().to_double()
    );

    let uniform_int = UniformInt::new(IntTensor::vector(vec![0; 20]), IntTensor::scalar(1000));
    let small = uniform_int.sample(&[], &mut seeded()).unwrap().to_double();
    assert_eq!(small.shape(), &[20]);
    assert!(small.as_slice().iter().any(|&k| k != small.as_slice()[0]));
    assert_eq!(
        small,
        uniform_int
            .sample(&[20], &mut seeded())
            .unwrap()
            .to_double()
    );

    let categorical = Categorical::new(vec![twenty.clone(), s(0.5)]).unwrap();
    let small = categorical.sample(&[], &mut seeded()).unwrap();
    assert_eq!(small.shape(), &[20]);
    assert!(small.as_slice().iter().any(|&k| k != small.as_slice()[0]));
    assert_eq!(small, categorical.sample(&[20], &mut seeded()).unwrap());
}

#[test]
fn kde_is_stable_far_from_the_samples() {
    let kde = Kde::new(vec![0.0, 1.0], 0.1).unwrap();
    let x = s(5.0);
    // ln(sum_i N((x - s_i) / h) / (n h)), dominated by the sample at 1: -(40^2)/2 - ln(0.2) - ln sqrt(2 pi)
    let expected = -800.0 + (1.0 + (-450.0f64).exp()).ln()
        - 0.2f64.ln()
        - 0.5 * (2.0 * std::f64::consts::PI).ln();
    assert_close(kde.log_prob(&x).as_slice()[0], expected, 1e-9, "value");
    // d/dx = -(x - 1) / h^2
    assert_close(
        kde.d_log_prob(&x).unwrap().x.as_slice()[0],
        -400.0,
        1e-9,
        "value",
    );

    // near the samples it agrees with ln of the pdf
    let near = v(&[-0.1, 0.4, 0.9]);
    for (lp, pdf) in kde
        .log_prob(&near)
        .as_slice()
        .iter()
        .zip(kde.pdf(&near).as_slice())
    {
        assert_close(*lp, pdf.ln(), 1e-12, "value");
    }
}

#[test]
fn kde_rejects_a_bandwidth_that_is_not_positive_and_finite() {
    for bandwidth in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(Kde::new(vec![0.0, 1.0], bandwidth).is_err(), "{bandwidth}");
    }
    assert!(Kde::with_scotts_bandwidth(vec![2.0, 2.0, 2.0]).is_err());
    assert!(Kde::with_scotts_bandwidth(vec![2.0]).is_err());
    assert!(Kde::with_scotts_bandwidth(vec![]).is_err());
    assert!(Kde::with_scotts_bandwidth(vec![1.0, 2.0, 4.0]).is_ok());
}

#[test]
fn dirichlet_log_prob_at_the_boundary_and_outside_the_support() {
    let dirichlet = Dirichlet::new(v(&[1.0, 2.0]));
    // (1 - 1) ln 0 is 0: lp = ln Γ(3) - ln Γ(1) - ln Γ(2) = ln 2
    assert_close(
        dirichlet.log_prob(&v(&[0.0, 1.0])).unwrap(),
        2f64.ln(),
        1e-12,
        "value",
    );
    assert_eq!(
        dirichlet.log_prob(&v(&[-0.5, 1.5])).unwrap(),
        f64::NEG_INFINITY
    );
}

#[test]
fn poisson_with_zero_mean() {
    let poisson = Poisson::new(s(0.0));
    let lp = poisson.log_prob(&IntTensor::vector(vec![0, 1])).unwrap();
    // ln P(0) = 0 (up to ln Γ(1) rounding), not NaN
    assert_close(lp.as_slice()[0], 0.0, 1e-12, "value");
    assert_eq!(lp.as_slice()[1], f64::NEG_INFINITY);
}

#[test]
fn binomial_gradient_at_the_ends_of_p() {
    let k = IntTensor::vector(vec![0, 3]);
    let at_zero = Binomial::new(s(0.0), IntTensor::scalar(3))
        .d_log_prob(&k)
        .unwrap();
    // k = 0: 0 - 3 / 1
    assert_eq!(at_zero.p.as_slice()[0], -3.0);
    let at_one = Binomial::new(s(1.0), IntTensor::scalar(3))
        .d_log_prob(&k)
        .unwrap();
    // k = n: 3 / 1 - 0
    assert_eq!(at_one.p.as_slice()[1], 3.0);
}

#[test]
fn invalid_parameters_only_affect_their_own_elements() {
    let pareto = Pareto::new(v(&[1.0, -1.0, 1.0]), v(&[1.0, 1.0, f64::NAN]));
    let lp = pareto.log_prob(&s(2.0)).unwrap();
    assert_eq!(lp.shape(), &[3]);
    assert_close(lp.as_slice()[0], -2.0 * 2f64.ln(), 1e-12, "value");
    assert_eq!(lp.as_slice()[1], f64::NEG_INFINITY);
    assert_eq!(lp.as_slice()[2], f64::NEG_INFINITY);

    let geometric = Geometric::new(v(&[0.5, 1.5]));
    let lp = geometric.log_prob(&IntTensor::scalar(1)).unwrap();
    assert_eq!(lp.shape(), &[2]);
    assert_close(lp.as_slice()[0], 0.5f64.ln(), 1e-12, "value");
    assert_eq!(lp.as_slice()[1], f64::NEG_INFINITY);
}

#[test]
fn pareto_support_includes_the_location() {
    // at x = x_m: ln(alpha) + alpha ln(x_m) - (alpha + 1) ln(x_m) = ln 2 for x_m = 1, alpha = 2
    let lp = Pareto::new(s(1.0), s(2.0)).log_prob(&s(1.0)).unwrap();
    assert_close(lp.as_slice()[0], 2f64.ln(), 1e-12, "value");
    let below = Pareto::new(s(1.0), s(2.0)).log_prob(&s(0.999)).unwrap();
    assert_eq!(below.as_slice()[0], f64::NEG_INFINITY);
}

/// A batch of symmetric positive definite n × n matrices, B Bᵀ + n I with B from a fixed sequence.
fn spd(batch: usize, n: usize) -> DoubleTensor {
    let data: Vec<f64> = (0..batch * n * n)
        .map(|i| ((i * 7919 % 1000) as f64 / 500.0) - 1.0)
        .collect();
    let b = DoubleTensor::new(&[batch, n, n], data).unwrap();
    let eye = &DoubleTensor::eye(n).unwrap() * n as f64;
    b.matmul_transposed(&b, false, true)
        .unwrap()
        .add(&eye)
        .unwrap()
}

#[test]
fn multivariate_gaussian_by_triangular_solves_matches_the_explicit_inverse() {
    // log_prob and d_log_prob solve with the Cholesky factor; check against the textbook formulas with Σ⁻¹,
    // batched, below and above the size where the solves switch to faer
    for n in [3, 100] {
        let covariance = spd(2, n);
        let mu = DoubleTensor::new(&[n], (0..n).map(|i| i as f64 / n as f64).collect()).unwrap();
        let x = DoubleTensor::new(
            &[2, n],
            (0..2 * n).map(|i| ((i * 31 % 17) as f64) / 8.0).collect(),
        )
        .unwrap();
        let d = MultivariateGaussian::new(mu.clone(), covariance.clone());

        let inverse = covariance.inverse().unwrap();
        let diff = x.sub(&mu).unwrap().reshape(&[2, n as isize, 1]).unwrap();
        let a = inverse.matmul(&diff).unwrap();
        let quadratic = diff.mul(&a).unwrap().sum();
        let log_det: f64 = covariance
            .determinant()
            .unwrap()
            .as_slice()
            .iter()
            .map(|d| d.ln())
            .sum();
        let expected =
            -0.5 * (quadratic + log_det + 2.0 * n as f64 * (2.0 * std::f64::consts::PI).ln());
        let actual = d.log_prob(&x).unwrap();
        assert_close(actual, expected, 1e-9 * expected.abs(), "value");

        let g = d.d_log_prob(&x).unwrap();
        let expected_mu = a.reshape(&[2, n as isize]).unwrap();
        let outer = a.matmul_transposed(&a, false, true).unwrap();
        let expected_cov = &outer.sub(&inverse).unwrap() * 0.5;
        for (actual, expected) in [
            (&g.mu, expected_mu.sum_axes(&[0]).unwrap()),
            (&g.x, expected_mu.neg()),
            (&g.covariance, expected_cov),
        ] {
            assert_eq!(actual.shape(), expected.shape());
            for (a, e) in actual.as_slice().iter().zip(expected.as_slice()) {
                assert_close(*a, *e, 1e-12, "value");
            }
        }
    }

    // a covariance that is not positive definite: log_prob is -infinity, the gradient an error, as before
    let not_pd = DoubleTensor::new(&[2, 2], vec![1.0, 2.0, 2.0, 1.0]).unwrap();
    let d = MultivariateGaussian::new(v(&[0.0, 0.0]), not_pd);
    assert_eq!(d.log_prob(&v(&[0.0, 0.0])).unwrap(), f64::NEG_INFINITY);
    assert!(d.d_log_prob(&v(&[0.0, 0.0])).is_err());
}
