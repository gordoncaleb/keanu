//! Finite difference checks of every gradient field at interior points of each distribution's support.

use keanu_distributions::*;
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};

mod common;
use common::{s, v};

const H: f64 = 1e-6;

/// Checks `analytic` against central differences of the summed log prob as `which` is perturbed element-wise.
/// `log_prob` takes the perturbed tensor; `analytic` must have the shape of the broadcast log prob (element-wise
/// gradients), and is summed down to the parameter's shape for scalar parameters.
fn check(
    name: &str,
    at: &DoubleTensor,
    analytic: &DoubleTensor,
    log_prob: impl Fn(&DoubleTensor) -> f64,
) {
    // element-wise gradients of a scalar parameter are summed over x
    let analytic: Vec<f64> = if analytic.len() != at.len() && at.len() == 1 {
        vec![analytic.sum()]
    } else {
        analytic.as_slice().to_vec()
    };
    assert_eq!(
        analytic.len(),
        at.len(),
        "{name}: gradient has {} values for {} inputs",
        analytic.len(),
        at.len()
    );
    for i in 0..at.len() {
        let mut up = at.as_slice().to_vec();
        let mut down = up.clone();
        up[i] += H;
        down[i] -= H;
        let numeric = (log_prob(&DoubleTensor::new(at.shape(), up).unwrap())
            - log_prob(&DoubleTensor::new(at.shape(), down).unwrap()))
            / (2.0 * H);
        assert!(
            (numeric - analytic[i]).abs() < 1e-5 * numeric.abs().max(1.0),
            "{name}[{i}]: analytic {}, numeric {numeric}",
            analytic[i]
        );
    }
}

fn total(t: Result<DoubleTensor>) -> f64 {
    t.unwrap().sum()
}

#[test]
fn gaussian() {
    let (x, mu, sigma) = (v(&[-1.2, 0.3, 2.5]), s(0.4), s(1.3));
    let g = Gaussian::new(mu.clone(), sigma.clone())
        .d_log_prob(&x)
        .unwrap();
    check("x", &x, &g.x, |x| {
        total(Gaussian::new(mu.clone(), sigma.clone()).log_prob(x))
    });
    check("mu", &mu, &g.mu, |m| {
        total(Gaussian::new(m.clone(), sigma.clone()).log_prob(&x))
    });
    check("sigma", &sigma, &g.sigma, |sg| {
        total(Gaussian::new(mu.clone(), sg.clone()).log_prob(&x))
    });
}

#[test]
fn half_gaussian_and_half_cauchy() {
    let (x, scale) = (v(&[0.2, 1.1, 3.0]), s(1.4));
    let g = HalfGaussian::new(scale.clone()).d_log_prob(&x).unwrap();
    check("half gaussian x", &x, &g.x, |x| {
        total(HalfGaussian::new(scale.clone()).log_prob(x))
    });
    check("half gaussian sigma", &scale, &g.sigma, |p| {
        total(HalfGaussian::new(p.clone()).log_prob(&x))
    });
    let g = HalfCauchy::new(scale.clone()).d_log_prob(&x).unwrap();
    check("half cauchy x", &x, &g.x, |x| {
        total(HalfCauchy::new(scale.clone()).log_prob(x))
    });
    check("half cauchy scale", &scale, &g.scale, |p| {
        total(HalfCauchy::new(p.clone()).log_prob(&x))
    });
}

#[test]
fn cauchy() {
    let (x, l, sc) = (v(&[-1.0, 0.7, 4.0]), s(0.5), s(0.8));
    let g = Cauchy::new(l.clone(), sc.clone()).d_log_prob(&x).unwrap();
    check("x", &x, &g.x, |x| {
        total(Cauchy::new(l.clone(), sc.clone()).log_prob(x))
    });
    check("location", &l, &g.location, |p| {
        total(Cauchy::new(p.clone(), sc.clone()).log_prob(&x))
    });
    check("scale", &sc, &g.scale, |p| {
        total(Cauchy::new(l.clone(), p.clone()).log_prob(&x))
    });
}

#[test]
fn beta_with_and_without_bounds() {
    for (min, max) in [(0.0, 1.0), (-1.0, 2.0)] {
        let x = v(&[
            min + 0.2 * (max - min),
            min + 0.55 * (max - min),
            min + 0.9 * (max - min),
        ]);
        let (a, b, lo, hi) = (s(2.5), s(1.7), s(min), s(max));
        let g = Beta::new(a.clone(), b.clone(), lo.clone(), hi.clone())
            .d_log_prob(&x)
            .unwrap();
        check("x", &x, &g.x, |x| {
            total(Beta::new(a.clone(), b.clone(), lo.clone(), hi.clone()).log_prob(x))
        });
        check("alpha", &a, &g.alpha, |p| {
            total(Beta::new(p.clone(), b.clone(), lo.clone(), hi.clone()).log_prob(&x))
        });
        check("beta", &b, &g.beta, |p| {
            total(Beta::new(a.clone(), p.clone(), lo.clone(), hi.clone()).log_prob(&x))
        });
    }
}

#[test]
fn chi_squared_and_student_t() {
    let x = v(&[0.3, 1.5, 4.0]);
    let g = ChiSquared::new(IntTensor::scalar(3))
        .d_log_prob(&x)
        .unwrap();
    check("chi squared x", &x, &g.x, |x| {
        total(ChiSquared::new(IntTensor::scalar(3)).log_prob(x))
    });
    let t = v(&[-2.0, 0.1, 1.5]);
    let g = StudentT::new(IntTensor::scalar(4)).d_log_prob(&t).unwrap();
    check("student t x", &t, &g.x, |t| {
        total(StudentT::new(IntTensor::scalar(4)).log_prob(t))
    });
}

#[test]
fn exponential() {
    let (x, rate) = (v(&[0.1, 0.9, 3.0]), s(1.7));
    let g = Exponential::new(rate.clone()).d_log_prob(&x).unwrap();
    check("x", &x, &g.x, |x| {
        total(Exponential::new(rate.clone()).log_prob(x))
    });
    check("rate", &rate, &g.rate, |p| {
        total(Exponential::new(p.clone()).log_prob(&x))
    });
}

#[test]
fn gamma_and_inverse_gamma() {
    let x = v(&[0.4, 1.2, 3.3]);
    let (theta, k) = (s(1.5), s(2.2));
    let g = Gamma::new(theta.clone(), k.clone()).d_log_prob(&x).unwrap();
    check("gamma x", &x, &g.x, |x| {
        total(Gamma::new(theta.clone(), k.clone()).log_prob(x))
    });
    check("gamma theta", &theta, &g.theta, |p| {
        total(Gamma::new(p.clone(), k.clone()).log_prob(&x))
    });
    check("gamma k", &k, &g.k, |p| {
        total(Gamma::new(theta.clone(), p.clone()).log_prob(&x))
    });
    let (a, b) = (s(2.5), s(1.3));
    let g = InverseGamma::new(a.clone(), b.clone())
        .d_log_prob(&x)
        .unwrap();
    check("inverse gamma x", &x, &g.x, |x| {
        total(InverseGamma::new(a.clone(), b.clone()).log_prob(x))
    });
    check("inverse gamma alpha", &a, &g.alpha, |p| {
        total(InverseGamma::new(p.clone(), b.clone()).log_prob(&x))
    });
    check("inverse gamma beta", &b, &g.beta, |p| {
        total(InverseGamma::new(a.clone(), p.clone()).log_prob(&x))
    });
}

#[test]
fn laplace_logistic_and_log_normal() {
    let x = v(&[0.3, 1.7, 2.9]);
    let (m, sc) = (s(1.1), s(0.6));
    let g = Laplace::new(m.clone(), sc.clone()).d_log_prob(&x).unwrap();
    check("laplace x", &x, &g.x, |x| {
        total(Laplace::new(m.clone(), sc.clone()).log_prob(x))
    });
    check("laplace mu", &m, &g.mu, |p| {
        total(Laplace::new(p.clone(), sc.clone()).log_prob(&x))
    });
    check("laplace beta", &sc, &g.beta, |p| {
        total(Laplace::new(m.clone(), p.clone()).log_prob(&x))
    });
    let g = Logistic::new(m.clone(), sc.clone()).d_log_prob(&x).unwrap();
    check("logistic x", &x, &g.x, |x| {
        total(Logistic::new(m.clone(), sc.clone()).log_prob(x))
    });
    check("logistic mu", &m, &g.mu, |p| {
        total(Logistic::new(p.clone(), sc.clone()).log_prob(&x))
    });
    check("logistic s", &sc, &g.s, |p| {
        total(Logistic::new(m.clone(), p.clone()).log_prob(&x))
    });
    let g = LogNormal::new(m.clone(), sc.clone())
        .d_log_prob(&x)
        .unwrap();
    check("log normal x", &x, &g.x, |x| {
        total(LogNormal::new(m.clone(), sc.clone()).log_prob(x))
    });
    check("log normal mu", &m, &g.mu, |p| {
        total(LogNormal::new(p.clone(), sc.clone()).log_prob(&x))
    });
    check("log normal sigma", &sc, &g.sigma, |p| {
        total(LogNormal::new(m.clone(), p.clone()).log_prob(&x))
    });
}

#[test]
fn pareto() {
    let (x, l, a) = (v(&[1.5, 2.2, 6.0]), s(1.2), s(2.5));
    let g = Pareto::new(l.clone(), a.clone()).d_log_prob(&x).unwrap();
    check("x", &x, &g.x, |x| {
        total(Pareto::new(l.clone(), a.clone()).log_prob(x))
    });
    check("location", &l, &g.location, |p| {
        total(Pareto::new(p.clone(), a.clone()).log_prob(&x))
    });
    check("scale", &a, &g.scale, |p| {
        total(Pareto::new(l.clone(), p.clone()).log_prob(&x))
    });
}

#[test]
fn triangular() {
    let (x, a, b, c) = (v(&[1.3, 2.9, 3.7]), s(1.0), s(4.0), s(2.5));
    let g = Triangular::new(a.clone(), b.clone(), c.clone())
        .d_log_prob(&x)
        .unwrap();
    check("x", &x, &g.x, |x| {
        total(Triangular::new(a.clone(), b.clone(), c.clone()).log_prob(x))
    });
    check("x_min", &a, &g.x_min, |p| {
        total(Triangular::new(p.clone(), b.clone(), c.clone()).log_prob(&x))
    });
    check("x_max", &b, &g.x_max, |p| {
        total(Triangular::new(a.clone(), p.clone(), c.clone()).log_prob(&x))
    });
    check("c", &c, &g.c, |p| {
        total(Triangular::new(a.clone(), b.clone(), p.clone()).log_prob(&x))
    });
}

#[test]
fn multivariate_gaussian() {
    let mu = v(&[0.5, -1.0]);
    let cov = DoubleTensor::new(&[2, 2], vec![2.0, 0.3, 0.3, 1.0]).unwrap();
    let x = DoubleTensor::new(&[3, 2], vec![0.0, 0.0, 1.0, 1.0, -1.0, 2.0]).unwrap();
    let g = MultivariateGaussian::new(mu.clone(), cov.clone())
        .d_log_prob(&x)
        .unwrap();
    check("x", &x, &g.x, |x| {
        MultivariateGaussian::new(mu.clone(), cov.clone())
            .log_prob(x)
            .unwrap()
    });
    check("mu", &mu, &g.mu, |m| {
        MultivariateGaussian::new(m.clone(), cov.clone())
            .log_prob(&x)
            .unwrap()
    });
    // perturbing one off-diagonal element breaks symmetry; compare against the symmetric perturbation of both
    let analytic_sym = g
        .covariance
        .add(&g.covariance.transpose().unwrap())
        .unwrap();
    for (i, j) in [(0usize, 0usize), (0, 1), (1, 1)] {
        let perturbed = |delta: f64| {
            let mut c = cov.as_slice().to_vec();
            c[i * 2 + j] += delta;
            if i != j {
                c[j * 2 + i] += delta;
            }
            MultivariateGaussian::new(mu.clone(), DoubleTensor::new(&[2, 2], c).unwrap())
                .log_prob(&x)
                .unwrap()
        };
        let numeric = (perturbed(H) - perturbed(-H)) / (2.0 * H);
        let analytic = if i == j {
            g.covariance.get(&[i, i]).unwrap()
        } else {
            analytic_sym.get(&[i, j]).unwrap()
        };
        assert!(
            (numeric - analytic).abs() < 1e-5 * numeric.abs().max(1.0),
            "covariance[{i},{j}]: {analytic} vs {numeric}"
        );
    }
}

#[test]
fn dirichlet() {
    let c = v(&[0.7, 2.0, 3.5]);
    let x = v(&[0.2, 0.3, 0.5]);
    let g = Dirichlet::new(c.clone()).d_log_prob(&x).unwrap();
    check("concentration", &c, &g.concentration, |p| {
        Dirichlet::new(p.clone()).log_prob(&x).unwrap()
    });
    // x is constrained to the simplex, so check the directional derivative along a direction that keeps the sum
    let direction = [1.0, -1.0, 0.0];
    let along = |t: f64| {
        Dirichlet::new(c.clone())
            .log_prob(&v(&[0.2 + t, 0.3 - t, 0.5]))
            .unwrap()
    };
    let numeric = (along(H) - along(-H)) / (2.0 * H);
    let analytic: f64 =
        g.x.as_slice()
            .iter()
            .zip(direction)
            .map(|(g, d)| g * d)
            .sum();
    assert!(
        (numeric - analytic).abs() < 1e-5 * numeric.abs().max(1.0),
        "x: {analytic} vs {numeric}"
    );
}

#[test]
fn discrete_parameter_gradients() {
    let k = IntTensor::vector(vec![0, 2, 5]);
    let p = s(0.35);
    let g = Binomial::new(p.clone(), IntTensor::scalar(6))
        .d_log_prob(&k)
        .unwrap();
    check("binomial p", &p, &g.p, |q| {
        total(Binomial::new(q.clone(), IntTensor::scalar(6)).log_prob(&k))
    });

    let k1 = IntTensor::vector(vec![1, 2, 7]);
    let g = Geometric::new(p.clone()).d_log_prob(&k1).unwrap();
    check("geometric p", &p, &g.p, |q| {
        total(Geometric::new(q.clone()).log_prob(&k1))
    });

    let mu = s(2.4);
    let g = Poisson::new(mu.clone()).d_log_prob(&k).unwrap();
    check("poisson mu", &mu, &g.mu, |q| {
        total(Poisson::new(q.clone()).log_prob(&k))
    });

    let b = BoolTensor::vector(vec![true, false, true]);
    let g = Bernoulli::new(p.clone()).d_log_prob(&b).unwrap();
    check("bernoulli p", &p, &g.prob_true, |q| {
        total(Bernoulli::new(q.clone()).log_prob(&b))
    });

    let probabilities = v(&[0.2, 0.5, 0.3]);
    let counts = IntTensor::vector(vec![1, 3, 1]);
    let g = Multinomial::new(IntTensor::scalar(5), probabilities.clone())
        .unwrap()
        .d_log_prob(&counts)
        .unwrap();
    check("multinomial p", &probabilities, &g.p, |q| {
        total(
            Multinomial::new(IntTensor::scalar(5), q.clone())
                .unwrap()
                .log_prob(&counts),
        )
    });
}

#[test]
fn kde() {
    let kde = Kde::new(vec![0.0, 1.0, 5.0, 5.5], 0.7).unwrap();
    let x = v(&[-0.5, 0.8, 3.0, 5.2]);
    let g = kde.d_log_prob(&x).unwrap();
    check("x", &x, &g.x, |x| kde.log_prob(x).sum());
}

#[test]
fn negative_binomial() {
    let k = IntTensor::vector(vec![0, 3, 11]);
    let (mu, alpha) = (s(4.2), s(1.7));
    let g = NegativeBinomial::new(mu.clone(), alpha.clone())
        .d_log_prob(&k)
        .unwrap();
    check("mu", &mu, &g.mu, |p| {
        total(NegativeBinomial::new(p.clone(), alpha.clone()).log_prob(&k))
    });
    check("alpha", &alpha, &g.alpha, |p| {
        total(NegativeBinomial::new(mu.clone(), p.clone()).log_prob(&k))
    });
}

#[test]
fn weibull() {
    let x = v(&[0.4, 1.3, 3.1]);
    let (a, b) = (s(1.8), s(1.4));
    let g = Weibull::new(a.clone(), b.clone()).d_log_prob(&x).unwrap();
    check("x", &x, &g.x, |x| {
        total(Weibull::new(a.clone(), b.clone()).log_prob(x))
    });
    check("alpha", &a, &g.alpha, |p| {
        total(Weibull::new(p.clone(), b.clone()).log_prob(&x))
    });
    check("beta", &b, &g.beta, |p| {
        total(Weibull::new(a.clone(), p.clone()).log_prob(&x))
    });
}

#[test]
fn von_mises() {
    let x = v(&[-2.5, 0.4, 3.0]);
    for kappa in [0.5, 4.0, 80.0] {
        let (mu, k) = (s(0.9), s(kappa));
        let g = VonMises::new(mu.clone(), k.clone()).d_log_prob(&x).unwrap();
        check("x", &x, &g.x, |x| {
            total(VonMises::new(mu.clone(), k.clone()).log_prob(x))
        });
        check("mu", &mu, &g.mu, |p| {
            total(VonMises::new(p.clone(), k.clone()).log_prob(&x))
        });
        check("kappa", &k, &g.kappa, |p| {
            total(VonMises::new(mu.clone(), p.clone()).log_prob(&x))
        });
    }
}

#[test]
fn ordered_logistic() {
    let k = IntTensor::vector(vec![0, 1, 2, 3]);
    let (eta, cutpoints) = (s(0.4), v(&[-1.2, 0.3, 1.9]));
    let g = OrderedLogistic::new(eta.clone(), cutpoints.clone())
        .d_log_prob(&k)
        .unwrap();
    check("eta", &eta, &g.eta, |p| {
        total(OrderedLogistic::new(p.clone(), cutpoints.clone()).log_prob(&k))
    });
    // the cutpoints' gradient has a row per value element: sum them for the total log prob
    let summed = g.cutpoints.sum_axes(&[0]).unwrap();
    check("cutpoints", &cutpoints, &summed, |p| {
        total(OrderedLogistic::new(eta.clone(), p.clone()).log_prob(&k))
    });
}
