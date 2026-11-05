//! Every distribution reports its support, resolving parameter-dependent bounds from the current state.

use keanu_distributions::Kde;
use keanu_graph::{
    CustomDistribution, DType, DoubleVar, GraphBuilder, GraphError, NodeId, Support, Value,
    node_support,
};
use keanu_tensor::DoubleTensor;

mod common;
use common::{evaluate, v};

/// The support of the node `build` returns, in a freshly built and evaluated graph.
fn support_of(build: impl FnOnce(&GraphBuilder) -> NodeId) -> Support {
    let b = GraphBuilder::new();
    let id = build(&b);
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 3);
    node_support(&graph, &state, id).unwrap()
}

type Build = fn(&GraphBuilder) -> NodeId;

#[test]
fn every_built_in_reports_its_documented_support() {
    let positive = || Support::greater_than(0.0);
    let cases: Vec<(&str, Build, Support)> = vec![
        ("gaussian", |b| b.gaussian(0.0, 1.0).id(), Support::Real),
        ("half_gaussian", |b| b.half_gaussian(1.0).id(), positive()),
        ("cauchy", |b| b.cauchy(0.0, 1.0).id(), Support::Real),
        ("half_cauchy", |b| b.half_cauchy(1.0).id(), positive()),
        (
            "beta",
            |b| b.beta(2.0, 3.0).id(),
            Support::interval(0.0, 1.0),
        ),
        ("chi_squared", |b| b.chi_squared(3).id(), positive()),
        (
            "dirichlet",
            |b| b.dirichlet(v(&[1.0, 2.0, 3.0])).id(),
            Support::Simplex,
        ),
        ("exponential", |b| b.exponential(1.5).id(), positive()),
        ("gamma", |b| b.gamma(1.0, 2.0).id(), positive()),
        (
            "inverse_gamma",
            |b| b.inverse_gamma(2.0, 1.0).id(),
            positive(),
        ),
        ("laplace", |b| b.laplace(0.0, 1.0).id(), Support::Real),
        ("logistic", |b| b.logistic(0.0, 1.0).id(), Support::Real),
        ("log_normal", |b| b.log_normal(0.0, 1.0).id(), positive()),
        (
            "lkj_cholesky",
            |b| b.lkj_cholesky(3, 2.0).id(),
            Support::CorrelationCholesky,
        ),
        (
            "multivariate_gaussian",
            |b| {
                b.multivariate_gaussian(v(&[0.0, 0.0]), DoubleTensor::eye(2).unwrap())
                    .id()
            },
            Support::Real,
        ),
        (
            "pareto",
            |b| b.pareto(1.5, 2.0).id(),
            Support::greater_than(1.5),
        ),
        ("student_t", |b| b.student_t(3).id(), Support::Real),
        (
            "triangular",
            |b| b.triangular(-1.0, 4.0, 1.0).id(),
            Support::interval(-1.0, 4.0),
        ),
        (
            "uniform",
            |b| b.uniform(2.0, 5.0).id(),
            Support::interval(2.0, 5.0),
        ),
        (
            "kde",
            |b| b.kde(Kde::new(vec![0.0, 1.0, 2.0], 0.5).unwrap(), &[]).id(),
            Support::Real,
        ),
        ("bernoulli", |b| b.bernoulli(0.3).id(), Support::Discrete),
        ("binomial", |b| b.binomial(0.3, 10).id(), Support::Discrete),
        (
            "categorical",
            |b| {
                let p = [b.constant(0.2), b.constant(0.8)];
                b.categorical(&[], &p).id()
            },
            Support::Discrete,
        ),
        ("geometric", |b| b.geometric(0.3).id(), Support::Discrete),
        (
            "multinomial",
            |b| b.multinomial(5, v(&[0.2, 0.8])).id(),
            Support::Discrete,
        ),
        ("poisson", |b| b.poisson(2.0).id(), Support::Discrete),
        (
            "negative_binomial",
            |b| b.negative_binomial(2.0, 1.5).id(),
            Support::Discrete,
        ),
        (
            "ordered_logistic",
            |b| b.ordered_logistic(0.0, v(&[-1.0, 1.0])).id(),
            Support::Discrete,
        ),
        ("weibull", |b| b.weibull(1.5, 2.0).id(), positive()),
        (
            "von_mises",
            |b| b.von_mises(0.0, 2.0).id(),
            Support::Circular,
        ),
        (
            "uniform_int",
            |b| b.uniform_int(0, 5).id(),
            Support::Discrete,
        ),
    ];
    for (name, build, expected) in cases {
        assert_eq!(support_of(build), expected, "{name}");
    }
}

#[test]
fn parameter_dependent_bounds_follow_the_state() {
    let b = GraphBuilder::new();
    let lower = b.gaussian(0.0, 1.0);
    b.set_value(lower, DoubleTensor::scalar(2.0));
    let x = b.uniform(lower, 5.0);
    let (lower, x) = (lower.id(), x.id());
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 3);
    assert_eq!(
        node_support(&graph, &state, x).unwrap(),
        Support::interval(2.0, 5.0)
    );

    state
        .set_and_cascade(&graph, lower, DoubleTensor::scalar(-3.0))
        .unwrap();
    assert_eq!(
        node_support(&graph, &state, x).unwrap(),
        Support::interval(-3.0, 5.0)
    );
}

#[test]
fn tensor_parameters_give_element_wise_bounds() {
    let support = support_of(|b| b.pareto(v(&[1.0, 2.0]), 3.0).id());
    assert_eq!(support, Support::GreaterThan(v(&[1.0, 2.0])));
}

#[derive(Debug)]
struct Flat;

impl CustomDistribution for Flat {
    fn log_prob(&self, _: &Value, _: &[&Value]) -> Result<f64, String> {
        Ok(0.0)
    }
}

#[derive(Debug)]
struct PositiveCustom;

impl CustomDistribution for PositiveCustom {
    fn log_prob(&self, _: &Value, _: &[&Value]) -> Result<f64, String> {
        Ok(0.0)
    }

    fn support(&self, parameters: &[&Value]) -> Result<Support, String> {
        let lower = parameters[0]
            .as_f64()
            .ok_or("the lower bound must be f64")?;
        Ok(Support::GreaterThan(lower.clone()))
    }
}

#[derive(Debug)]
struct Counts;

impl CustomDistribution for Counts {
    fn value_type(&self) -> DType {
        DType::I64
    }

    fn log_prob(&self, _: &Value, _: &[&Value]) -> Result<f64, String> {
        Ok(0.0)
    }
}

#[derive(Debug)]
struct Failing;

impl CustomDistribution for Failing {
    fn log_prob(&self, _: &Value, _: &[&Value]) -> Result<f64, String> {
        Ok(0.0)
    }

    fn support(&self, _: &[&Value]) -> Result<Support, String> {
        Err("no support today".into())
    }
}

/// A graph with one custom node holding the value 1.0 (or 1), whose parameter is a constant 0.5.
fn custom_support(distribution: impl CustomDistribution + 'static) -> keanu_graph::Result<Support> {
    let b = GraphBuilder::new();
    let p = b.constant(0.5).id();
    let id = if distribution.value_type() == DType::F64 {
        let x: DoubleVar = b.custom(distribution, &[], &[p]);
        b.set_value(x, DoubleTensor::scalar(1.0));
        x.id()
    } else {
        let x: keanu_graph::IntVar = b.custom(distribution, &[], &[p]);
        b.set_value(x, keanu_tensor::IntTensor::scalar(1));
        x.id()
    };
    let (graph, state) = b.build().unwrap();
    node_support(&graph, &state, id)
}

#[test]
fn custom_distribution_defaults_to_real() {
    assert_eq!(custom_support(Flat).unwrap(), Support::Real);
}

#[test]
fn custom_distribution_with_non_f64_values_defaults_to_discrete() {
    assert_eq!(custom_support(Counts).unwrap(), Support::Discrete);
}

#[test]
fn custom_distribution_can_declare_its_support() {
    assert_eq!(
        custom_support(PositiveCustom).unwrap(),
        Support::greater_than(0.5)
    );
}

#[test]
fn custom_support_errors_are_reported() {
    match custom_support(Failing) {
        Err(GraphError::Custom(message)) => assert_eq!(message, "no support today"),
        other => panic!("expected a custom error, got {other:?}"),
    }
}

#[test]
fn non_probabilistic_and_unknown_nodes_are_errors() {
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    let doubled = (a * 2.0).id();
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 3);
    assert!(node_support(&graph, &state, doubled).is_err());
    assert!(matches!(
        node_support(&graph, &state, NodeId::from_index(999)),
        Err(GraphError::UnknownNode(_))
    ));
}

#[test]
fn missing_parameter_values_are_errors() {
    let b = GraphBuilder::new();
    let lower = b.gaussian(0.0, 1.0);
    let x = b.uniform(lower, 5.0).id();
    let (graph, state) = b.build().unwrap();
    assert!(matches!(
        node_support(&graph, &state, x),
        Err(GraphError::MissingValue(_))
    ));
}
