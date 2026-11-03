//! Node log probabilities and log prob gradients, checked against the distributions and finite differences.

use keanu_distributions as d;
use keanu_distributions::KeanuRng;
use keanu_graph::{
    DoubleVar, Graph, GraphBuilder, NodeId, State, joint_log_prob, joint_log_prob_gradient,
    log_likelihood, log_prior, log_prob_gradient, node_log_prob,
};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};

const A: f64 = 0.3;

fn v(data: &[f64]) -> DoubleTensor {
    DoubleTensor::vector(data.to_vec())
}

fn evaluate(graph: &Graph, state: &mut State) {
    let all: Vec<NodeId> = graph.ids().collect();
    state
        .lazy_eval(graph, &all, &mut KeanuRng::seed_from_u64(3))
        .unwrap();
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{what}: {actual} != {expected}"
    );
}

/// Builds `x ~ distribution(parameters(a))` with a scalar latent `a`, and checks d log p(x) / da against a central
/// finite difference (moving `a` and cascading to the parameters).
fn check_parameter_gradient(name: &str, build: impl Fn(&GraphBuilder, DoubleVar) -> NodeId) {
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 10.0);
    b.set_value(a, DoubleTensor::scalar(A));
    let x = build(&b, a);
    let a = a.id();
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state);

    let gradient = log_prob_gradient(&graph, &state, &[x], &[a]).unwrap();
    let analytic = gradient
        .get(a)
        .unwrap_or_else(|| panic!("{name}: no gradient"));
    assert_eq!(analytic.shape(), &[] as &[usize], "{name}");

    let h = 1e-6;
    let at = |value: f64| {
        let mut s = state.clone();
        s.set_and_cascade(&graph, a, DoubleTensor::scalar(value))
            .unwrap();
        node_log_prob(&graph, &s, x).unwrap()
    };
    let numeric = (at(A + h) - at(A - h)) / (2.0 * h);
    assert_close(analytic.as_slice()[0], numeric, 1e-5, name);
}

/// Checks d log p(x) / dx for an unobserved node against finite differences of each element.
fn check_value_gradient(name: &str, build: impl Fn(&GraphBuilder) -> NodeId) {
    let b = GraphBuilder::new();
    let x = build(&b);
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state);
    let gradient = log_prob_gradient(&graph, &state, &[x], &[x]).unwrap();
    let analytic = gradient
        .get(x)
        .unwrap_or_else(|| panic!("{name}: no gradient"))
        .clone();
    let value = state.f64(x).unwrap().clone();
    assert_eq!(analytic.shape(), value.shape(), "{name}");
    let h = 1e-6;
    for i in 0..value.len() {
        let at = |delta: f64| {
            let mut data = value.as_slice().to_vec();
            data[i] += delta;
            let mut s = state.clone();
            s.set(&graph, x, DoubleTensor::new(value.shape(), data).unwrap())
                .unwrap();
            node_log_prob(&graph, &s, x).unwrap()
        };
        let numeric = (at(h) - at(-h)) / (2.0 * h);
        assert_close(
            analytic.as_slice()[i],
            numeric,
            1e-5,
            &format!("{name}[{i}]"),
        );
    }
}

#[test]
fn parameter_gradients_through_deterministic_parents_match_finite_differences() {
    let c3 = || v(&[0.5, -1.0, 2.0]);
    let observe = |b: &GraphBuilder, x: DoubleVar, value: DoubleTensor| {
        b.observe(x, value);
        x.id()
    };
    check_parameter_gradient("gaussian", |b, a| {
        let x = b.gaussian(a * 2.0 + c3(), a.exp());
        observe(b, x, v(&[0.1, 0.4, 1.3]))
    });
    check_parameter_gradient("half_gaussian", |b, a| {
        let x = b.half_gaussian_with_shape(&[3], a.exp() + 0.5);
        observe(b, x, v(&[0.1, 0.4, 1.3]))
    });
    check_parameter_gradient("cauchy", |b, a| {
        let x = b.cauchy(a + c3(), a * a + 1.0);
        observe(b, x, v(&[0.1, 0.4, 1.3]))
    });
    check_parameter_gradient("half_cauchy", |b, a| {
        let x = b.half_cauchy_with_shape(&[3], a.exp());
        observe(b, x, v(&[0.1, 0.4, 1.3]))
    });
    check_parameter_gradient("beta", |b, a| {
        let x = b.beta_with_shape(&[3], a.exp() + 1.0, a * a + 2.0);
        observe(b, x, v(&[0.1, 0.4, 0.9]))
    });
    check_parameter_gradient("dirichlet", |b, a| {
        let x = b.dirichlet((a * v(&[1.0, 2.0, 3.0])).exp());
        observe(b, x, v(&[0.2, 0.3, 0.5]))
    });
    check_parameter_gradient("exponential", |b, a| {
        let x = b.exponential_with_shape(&[3], a.exp());
        observe(b, x, v(&[0.1, 0.4, 1.3]))
    });
    check_parameter_gradient("gamma", |b, a| {
        let x = b.gamma(a.exp() * c3().abs(), a * a + 1.5);
        observe(b, x, v(&[0.1, 0.4, 1.3]))
    });
    check_parameter_gradient("inverse_gamma", |b, a| {
        let x = b.inverse_gamma_with_shape(&[3], a.exp() + 1.0, (-a).exp());
        observe(b, x, v(&[0.1, 0.4, 1.3]))
    });
    check_parameter_gradient("laplace", |b, a| {
        let x = b.laplace(a + c3(), a.exp());
        observe(b, x, v(&[0.1, 0.4, 1.3]))
    });
    check_parameter_gradient("logistic", |b, a| {
        let x = b.logistic(a + c3(), a.exp());
        observe(b, x, v(&[0.1, 0.4, 1.3]))
    });
    check_parameter_gradient("log_normal", |b, a| {
        let x = b.log_normal(a + c3(), a.exp());
        observe(b, x, v(&[0.1, 0.4, 1.3]))
    });
    check_parameter_gradient("multivariate_gaussian", |b, a| {
        let covariance = DoubleTensor::new(&[2, 2], vec![2.0, 0.3, 0.3, 1.0]).unwrap();
        let x = b.multivariate_gaussian(a * v(&[1.0, -1.0]), a.exp() * covariance);
        observe(b, x, v(&[0.4, -0.7]))
    });
    check_parameter_gradient("pareto", |b, a| {
        let x = b.pareto_with_shape(&[3], a.exp() * 0.5, a * a + 1.0);
        observe(b, x, v(&[1.1, 1.4, 2.3]))
    });
    check_parameter_gradient("triangular", |b, a| {
        let x = b.triangular_with_shape(&[3], a - 2.0, a + 3.0, a * 0.5);
        observe(b, x, v(&[-1.0, 0.4, 2.0]))
    });

    // discrete distributions: gradients with respect to their continuous parameters
    check_parameter_gradient("bernoulli", |b, a| {
        let x = b.bernoulli((a + c3()).sigmoid());
        b.observe(x, BoolTensor::vector(vec![true, false, true]));
        x.id()
    });
    check_parameter_gradient("binomial", |b, a| {
        let x = b.binomial((a + c3()).sigmoid(), 5);
        b.observe(x, IntTensor::vector(vec![1, 3, 5]));
        x.id()
    });
    check_parameter_gradient("geometric", |b, a| {
        let x = b.geometric((a + c3()).sigmoid());
        b.observe(x, IntTensor::vector(vec![1, 3, 5]));
        x.id()
    });
    check_parameter_gradient("poisson", |b, a| {
        let x = b.poisson((a + c3()).exp());
        b.observe(x, IntTensor::vector(vec![0, 3, 5]));
        x.id()
    });
    check_parameter_gradient("multinomial", |b, a| {
        let x = b.multinomial(5, (a * v(&[1.0, 2.0, 3.0])).exp());
        b.observe(x, IntTensor::vector(vec![1, 3, 1]));
        x.id()
    });
}

#[test]
fn value_gradients_match_finite_differences() {
    let set = |b: &GraphBuilder, x: DoubleVar, value: DoubleTensor| {
        b.set_value(x, value);
        x.id()
    };
    let positive = || v(&[0.2, 0.9, 2.1]);
    check_value_gradient("gaussian", |b| {
        let x = b.gaussian(v(&[0.0, 1.0, -1.0]), 2.0);
        set(b, x, v(&[0.3, -0.4, 1.2]))
    });
    check_value_gradient("half_gaussian", |b| {
        set(b, b.half_gaussian_with_shape(&[3], 1.5), positive())
    });
    check_value_gradient("cauchy", |b| {
        set(b, b.cauchy_with_shape(&[3], 0.5, 2.0), v(&[0.3, -0.4, 1.2]))
    });
    check_value_gradient("half_cauchy", |b| {
        set(b, b.half_cauchy_with_shape(&[3], 1.5), positive())
    });
    check_value_gradient("beta", |b| {
        set(b, b.beta_with_shape(&[3], 2.0, 3.0), v(&[0.2, 0.5, 0.7]))
    });
    check_value_gradient("chi_squared", |b| {
        set(b, b.chi_squared_with_shape(&[3], 3), positive())
    });
    check_value_gradient("exponential", |b| {
        set(b, b.exponential_with_shape(&[3], 1.5), positive())
    });
    check_value_gradient("gamma", |b| {
        set(b, b.gamma_with_shape(&[3], 2.0, 1.5), positive())
    });
    check_value_gradient("inverse_gamma", |b| {
        set(b, b.inverse_gamma_with_shape(&[3], 2.0, 1.5), positive())
    });
    check_value_gradient("laplace", |b| {
        set(
            b,
            b.laplace_with_shape(&[3], 0.5, 2.0),
            v(&[0.3, -0.4, 1.2]),
        )
    });
    check_value_gradient("logistic", |b| {
        set(
            b,
            b.logistic_with_shape(&[3], 0.5, 2.0),
            v(&[0.3, -0.4, 1.2]),
        )
    });
    check_value_gradient("log_normal", |b| {
        set(b, b.log_normal_with_shape(&[3], 0.5, 2.0), positive())
    });
    check_value_gradient("multivariate_gaussian", |b| {
        let covariance = DoubleTensor::new(&[2, 2], vec![2.0, 0.3, 0.3, 1.0]).unwrap();
        set(
            b,
            b.multivariate_gaussian(v(&[0.5, -1.0]), covariance),
            v(&[0.4, -0.7]),
        )
    });
    check_value_gradient("pareto", |b| {
        set(b, b.pareto_with_shape(&[3], 0.5, 2.0), v(&[0.6, 1.4, 2.3]))
    });
    check_value_gradient("student_t", |b| {
        set(b, b.student_t_with_shape(&[3], 4), v(&[0.3, -0.4, 1.2]))
    });
    check_value_gradient("triangular", |b| {
        set(
            b,
            b.triangular_with_shape(&[3], -2.0, 3.0, 0.5),
            v(&[-1.0, 0.2, 2.0]),
        )
    });
    check_value_gradient("kde", |b| {
        let kde = d::Kde::new(vec![-1.0, 0.0, 0.5, 2.0], 0.8).unwrap();
        set(b, b.kde(kde, &[3]), v(&[0.3, -0.4, 1.2]))
    });
}

#[test]
fn uniform_and_discrete_nodes_contribute_no_gradient_with_respect_to_themselves() {
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    b.set_value(a, DoubleTensor::scalar(A));
    let uniform = b.uniform(a - 1.0, a + 1.0);
    b.set_value(uniform, DoubleTensor::scalar(0.1));
    let categorical = b.categorical(&[], &[a.exp(), (a * 2.0).exp()]);
    b.observe(categorical, IntTensor::scalar(1));
    let uniform_int = b.uniform_int(0, 5);
    b.set_value(uniform_int, IntTensor::scalar(2));
    let (a, uniform, categorical, uniform_int) =
        (a.id(), uniform.id(), categorical.id(), uniform_int.id());
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state);

    // inside its support the uniform's gradient with respect to its value is zero, and its bounds get none (as Java)
    let g = log_prob_gradient(&graph, &state, &[uniform], &[a, uniform]).unwrap();
    assert_eq!(g.get(uniform).unwrap().as_slice(), &[0.0]);
    assert!(g.get(a).is_none());

    // categorical and uniform int have no gradient at all; integer latents are not valid targets
    let g = log_prob_gradient(
        &graph,
        &state,
        &[categorical, uniform_int],
        &[a, uniform_int],
    )
    .unwrap();
    assert!(g.is_empty());
}

#[test]
fn node_log_prob_is_the_summed_element_wise_log_prob_of_the_distribution() {
    let b = GraphBuilder::new();
    let mu = v(&[0.0, 1.0, -1.0]);
    let x = b.gaussian(mu.clone(), 2.0);
    let xv = v(&[0.3, -0.4, 1.2]);
    b.observe(x, xv.clone());
    let p = b.poisson(v(&[0.5, 2.0, 4.0]));
    let pv = IntTensor::vector(vec![0, 3, 7]);
    b.observe(p, pv.clone());
    let concentration = v(&[0.5, 2.0, 3.0]);
    let dir = b.dirichlet(concentration.clone());
    let dv = v(&[0.2, 0.3, 0.5]);
    b.set_value(dir, dv.clone());
    let covariance = DoubleTensor::new(&[2, 2], vec![2.0, 0.3, 0.3, 1.0]).unwrap();
    let mvg = b.multivariate_gaussian(v(&[0.5, -1.0]), covariance.clone());
    let mv = v(&[0.4, -0.7]);
    b.set_value(mvg, mv.clone());
    let half = b.half_gaussian_with_shape(&[3], 1.5);
    b.set_value(half, v(&[0.2, 0.9, 2.1]));
    let (x, p, dir, mvg, half) = (x.id(), p.id(), dir.id(), mvg.id(), half.id());
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state);

    let gaussian = d::Gaussian::new(mu, DoubleTensor::scalar(2.0))
        .log_prob(&xv)
        .unwrap()
        .sum();
    assert_close(
        node_log_prob(&graph, &state, x).unwrap(),
        gaussian,
        1e-14,
        "gaussian",
    );
    let poisson = d::Poisson::new(v(&[0.5, 2.0, 4.0]))
        .log_prob(&pv)
        .unwrap()
        .sum();
    assert_close(
        node_log_prob(&graph, &state, p).unwrap(),
        poisson,
        1e-14,
        "poisson",
    );
    let dirichlet = d::Dirichlet::new(concentration).log_prob(&dv).unwrap();
    assert_close(
        node_log_prob(&graph, &state, dir).unwrap(),
        dirichlet,
        1e-14,
        "dirichlet",
    );
    let half_gaussian = d::HalfGaussian::new(DoubleTensor::scalar(1.5))
        .log_prob(&v(&[0.2, 0.9, 2.1]))
        .unwrap()
        .sum();
    assert_close(
        node_log_prob(&graph, &state, half).unwrap(),
        half_gaussian,
        1e-14,
        "half_gaussian",
    );
    let multivariate = d::MultivariateGaussian::new(v(&[0.5, -1.0]), covariance)
        .log_prob(&mv)
        .unwrap();
    assert_close(
        node_log_prob(&graph, &state, mvg).unwrap(),
        multivariate,
        1e-14,
        "multivariate_gaussian",
    );

    // the totals split into likelihood (observed) and prior (latent)
    let joint = joint_log_prob(&graph, &state).unwrap();
    assert_close(
        log_likelihood(&graph, &state).unwrap() + log_prior(&graph, &state).unwrap(),
        joint,
        1e-14,
        "joint",
    );
}

#[test]
fn joint_log_prob_gradient_of_a_multi_node_model_matches_finite_differences() {
    // shared parents (a feeds the mean, the noise and a second node), broadcasting parameters ([] and [3] into
    // [2, 3]), an observed node and a latent node downstream of other latents
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    b.set_value(a, DoubleTensor::scalar(0.4));
    let m = b.gaussian_with_shape(&[3], 0.0, 2.0);
    b.set_value(m, v(&[0.1, -0.5, 0.8]));
    let mean = a * m + 1.0;
    let y = b.gaussian_with_shape(&[2, 3], mean, a.exp());
    b.observe(
        y,
        DoubleTensor::new(&[2, 3], vec![1.2, 0.7, 1.5, 0.9, 1.1, 1.0]).unwrap(),
    );
    let z = b.gamma(a.exp(), m * m + 1.0);
    b.set_value(z, v(&[0.5, 1.5, 2.5]));
    let (a, m, z) = (a.id(), m.id(), z.id());
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state);

    let gradient = joint_log_prob_gradient(&graph, &state).unwrap();
    assert_eq!(gradient.len(), 3);
    let h = 1e-6;
    for latent in [a, m, z] {
        let value = state.f64(latent).unwrap().clone();
        let analytic = gradient.get(latent).unwrap();
        assert_eq!(analytic.shape(), value.shape());
        for i in 0..value.len() {
            let at = |delta: f64| {
                let mut data = value.as_slice().to_vec();
                data[i] += delta;
                let mut s = state.clone();
                s.set_and_cascade(
                    &graph,
                    latent,
                    DoubleTensor::new(value.shape(), data).unwrap(),
                )
                .unwrap();
                joint_log_prob(&graph, &s).unwrap()
            };
            let numeric = (at(h) - at(-h)) / (2.0 * h);
            assert_close(
                analytic.as_slice()[i],
                numeric,
                1e-5,
                &format!("node #{}[{i}]", latent.index()),
            );
        }
    }
}
