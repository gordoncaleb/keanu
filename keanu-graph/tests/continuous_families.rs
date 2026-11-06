//! The continuous families written as one scalar formula (skew normal, ex-Gaussian, Gumbel, generalized extreme
//! value, generalized Pareto, inverse Gaussian, Kumaraswamy, logit-normal, half Student-t, asymmetric Laplace,
//! log-logistic, Rice, generalized gamma) as graph nodes: their element-wise terms, the log-density op, mixtures,
//! truncation and censoring (values, sampling and bounds), and gradients through the graph. Their Hessians are in
//! `hessian.rs`, their supports in `support.rs`, saving in keanu-io's round trip.

use keanu_distributions as d;
use keanu_distributions::KeanuRng;
use keanu_graph::op::DistributionKind;
use keanu_graph::{
    DoubleVar, GraphBuilder, Model, NodeId, Value, node_log_prob, node_log_prob_terms,
};
use std::sync::Arc;

mod common;
use common::{evaluate, s, v};

/// Each kind with three values in its support and scalar parameters.
fn cases() -> Vec<(DistributionKind, Vec<f64>, Vec<f64>)> {
    use DistributionKind as K;
    vec![
        (K::SkewNormal, vec![-0.5, 0.4, 2.0], vec![0.2, 0.8, 1.5]),
        (K::ExGaussian, vec![-0.2, 1.0, 2.5], vec![0.3, 0.6, 1.4]),
        (K::Gumbel, vec![-1.0, 0.5, 3.0], vec![0.1, 1.2]),
        (
            K::GeneralizedExtremeValue,
            vec![-1.5, 0.7, 3.0],
            vec![0.2, 1.1, 0.15],
        ),
        (
            K::GeneralizedPareto,
            vec![0.5, 1.0, 3.0],
            vec![0.1, 0.8, 0.2],
        ),
        (K::InverseGaussian, vec![0.4, 0.9, 2.8], vec![1.3, 2.5]),
        (K::Kumaraswamy, vec![0.2, 0.4, 0.7], vec![2.2, 3.1]),
        (K::LogitNormal, vec![0.2, 0.6, 0.85], vec![0.3, 0.9]),
        (K::HalfStudentT, vec![0.3, 0.7, 2.5], vec![4.5, 1.2]),
        (
            K::AsymmetricLaplace,
            vec![-1.0, 0.9, 2.0],
            vec![0.1, 1.1, 0.3],
        ),
        (K::LogLogistic, vec![0.5, 1.1, 2.6], vec![1.4, 3.0]),
        (K::Rice, vec![0.4, 1.0, 2.2], vec![1.2, 0.7]),
        (
            K::GeneralizedGamma,
            vec![0.5, 1.2, 2.4],
            vec![1.3, 2.1, 1.6],
        ),
    ]
}

/// The element-wise log density from keanu-distributions.
fn expected_terms(kind: &DistributionKind, x: &[f64], p: &[f64]) -> Vec<f64> {
    use DistributionKind as K;
    let at = |&x: &f64| match kind {
        K::SkewNormal => d::SkewNormal::log_prob_derivatives(x, [p[0], p[1], p[2]]).value,
        K::ExGaussian => d::ExGaussian::log_prob_derivatives(x, [p[0], p[1], p[2]]).value,
        K::Gumbel => d::Gumbel::log_prob_derivatives(x, [p[0], p[1]]).value,
        K::GeneralizedExtremeValue => {
            d::GeneralizedExtremeValue::log_prob_derivatives(x, [p[0], p[1], p[2]]).value
        }
        K::GeneralizedPareto => {
            d::GeneralizedPareto::log_prob_derivatives(x, [p[0], p[1], p[2]]).value
        }
        K::InverseGaussian => d::InverseGaussian::log_prob_derivatives(x, [p[0], p[1]]).value,
        K::Kumaraswamy => d::Kumaraswamy::log_prob_derivatives(x, [p[0], p[1]]).value,
        K::LogitNormal => d::LogitNormal::log_prob_derivatives(x, [p[0], p[1]]).value,
        K::HalfStudentT => d::HalfStudentT::log_prob_derivatives(x, [p[0], p[1]]).value,
        K::AsymmetricLaplace => {
            d::AsymmetricLaplace::log_prob_derivatives(x, [p[0], p[1], p[2]]).value
        }
        K::LogLogistic => d::LogLogistic::log_prob_derivatives(x, [p[0], p[1]]).value,
        K::Rice => d::Rice::log_prob_derivatives(x, [p[0], p[1]]).value,
        K::GeneralizedGamma => {
            d::GeneralizedGamma::log_prob_derivatives(x, [p[0], p[1], p[2]]).value
        }
        _ => unreachable!(),
    };
    x.iter().map(at).collect()
}

fn constants(b: &GraphBuilder, values: &[f64]) -> Vec<NodeId> {
    values.iter().map(|&p| b.scalar(p).id()).collect()
}

#[test]
fn terms_and_the_log_density_op_match_the_distribution() {
    for (kind, x, p) in cases() {
        let name = kind.name();
        let b = GraphBuilder::new();
        let params = constants(&b, &p);
        let node = b
            .add_distribution(kind.clone(), Some(&[3]), &params)
            .unwrap();
        b.observe_of(node, Value::F64(v(&x))).unwrap();
        let density = b.log_density(kind.clone(), b.constant(v(&x)), &params).id();
        let (graph, mut state) = b.build().unwrap();
        evaluate(&graph, &mut state, 1);
        let expected = expected_terms(&kind, &x, &p);
        let terms = node_log_prob_terms(&graph, &state, node).unwrap();
        assert_eq!(terms.as_slice(), expected.as_slice(), "{name}");
        assert_eq!(
            state.f64(density).unwrap().as_slice(),
            expected.as_slice(),
            "{name}"
        );
        let total: f64 = expected.iter().sum();
        assert!(
            (node_log_prob(&graph, &state, node).unwrap() - total).abs() < 1e-12,
            "{name}"
        );
    }
}

#[test]
fn each_family_can_be_a_mixture_component() {
    for (kind, x, p) in cases() {
        let name = kind.name();
        let b = GraphBuilder::new();
        // two components: the parameters, and the first parameter shifted
        let shifted: Vec<f64> = p
            .iter()
            .enumerate()
            .map(|(i, &q)| if i == 0 { q * 1.2 } else { q })
            .collect();
        let params: Vec<NodeId> = p
            .iter()
            .zip(&shifted)
            .map(|(&a, &c)| b.constant(v(&[a, c])).id())
            .collect();
        let y: DoubleVar<'_> = b.mixture(kind.clone(), &params, b.constant(v(&[0.3, 0.7])));
        b.observe(y, s(x[0]));
        let y = y.id();
        let (graph, mut state) = b.build().unwrap();
        evaluate(&graph, &mut state, 1);
        let first = expected_terms(&kind, &x[..1], &p)[0];
        let second = expected_terms(&kind, &x[..1], &shifted)[0];
        let expected = (0.3 * first.exp() + 0.7 * second.exp()).ln();
        let actual = node_log_prob(&graph, &state, y).unwrap();
        assert!(
            (actual - expected).abs() < 1e-12,
            "{name}: {actual} vs {expected}"
        );
    }
}

#[test]
fn truncated_and_censored_families_sample_inside_their_bounds() {
    use DistributionKind as K;
    let kinds = [
        // by inverting the CDF
        (K::Gumbel, vec![0.1, 1.2], (-0.4, 1.6)),
        (
            K::GeneralizedExtremeValue,
            vec![0.2, 1.1, 0.15],
            (-0.4, 1.6),
        ),
        (K::GeneralizedPareto, vec![0.1, 0.8, -0.2], (0.5, 2.5)),
        (K::Kumaraswamy, vec![2.2, 3.1], (0.2, 0.7)),
        (K::LogitNormal, vec![0.3, 0.9], (0.2, 0.7)),
        (K::AsymmetricLaplace, vec![0.1, 1.1, 0.3], (-0.4, 1.6)),
        (K::LogLogistic, vec![1.4, 3.0], (0.5, 2.5)),
        // by rejection
        (K::SkewNormal, vec![0.2, 0.8, 1.5], (-0.4, 1.6)),
        (K::ExGaussian, vec![0.3, 0.6, 1.4], (-0.4, 1.6)),
        (K::InverseGaussian, vec![1.3, 2.5], (0.5, 2.5)),
        (K::HalfStudentT, vec![4.5, 1.2], (0.5, 2.5)),
        (K::GeneralizedGamma, vec![1.3, 2.1, 1.6], (0.5, 2.5)),
    ];
    for (kind, p, (lower, upper)) in kinds {
        let name = kind.name();
        let b = GraphBuilder::new();
        let params = constants(&b, &p);
        let truncated = b
            .add_distribution(
                DistributionKind::Truncated(Arc::new(kind.clone())),
                Some(&[500]),
                &[params.clone(), constants(&b, &[lower, upper])].concat(),
            )
            .unwrap();
        let censored = b
            .add_distribution(
                DistributionKind::Censored(Arc::new(kind.clone())),
                Some(&[500]),
                &[params, constants(&b, &[lower, upper])].concat(),
            )
            .unwrap();
        let (graph, mut state) = b.build().unwrap();
        evaluate(&graph, &mut state, 7);
        for node in [truncated, censored] {
            let values = state.f64(node).unwrap();
            assert!(
                values.as_slice().iter().all(|&x| x >= lower && x <= upper),
                "{name}: a draw outside [{lower}, {upper}]"
            );
            assert!(
                node_log_prob(&graph, &state, node).unwrap().is_finite(),
                "{name}"
            );
        }
        // some censored draws are at the bounds, and some truncated ones inside
        let censored = state.f64(censored).unwrap();
        assert!(
            censored
                .as_slice()
                .iter()
                .any(|&x| x == lower || x == upper),
            "{name}"
        );
    }
}

#[test]
fn truncation_renormalises_by_the_cdf() {
    let b = GraphBuilder::new();
    let params = constants(&b, &[0.2, 1.1, -0.15]);
    let bounds = constants(&b, &[-0.4, 1.6]);
    let kind = DistributionKind::Truncated(Arc::new(DistributionKind::GeneralizedExtremeValue));
    let node = b
        .add_distribution(kind, None, &[params, bounds].concat())
        .unwrap();
    b.observe_of(node, Value::F64(s(0.5))).unwrap();
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 1);
    let gev = d::GeneralizedExtremeValue::new(s(0.2), s(1.1), s(-0.15));
    let cdf = |x: f64| gev.log_cdf(&s(x)).unwrap().as_slice()[0].exp();
    let expected = gev.log_prob(&s(0.5)).unwrap().as_slice()[0] - (cdf(1.6) - cdf(-0.4)).ln();
    let actual = node_log_prob(&graph, &state, node).unwrap();
    assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");
}

#[test]
fn gradients_through_the_graph_match_the_distribution() {
    // a latent GEV whose parameters are latents: the model's gradient is the distribution's, plus the priors'
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0);
    let xi = b.gaussian(0.0, 1.0);
    b.set_value(mu, s(0.2));
    b.set_value(xi, s(0.1));
    let x = b.generalized_extreme_value(mu, 1.3, xi);
    b.set_value(x, s(0.9));
    let (mu, xi, x) = (mu.id(), xi.id(), x.id());
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 1);
    let model = Model::new(graph, &state).unwrap();
    let gradient = model.log_prob_gradient(&state).unwrap();
    let g = d::GeneralizedExtremeValue::log_prob_derivatives(0.9, [0.2, 1.3, 0.1]).gradient;
    let of = |id| gradient.get(id).unwrap().as_slice()[0];
    assert!((of(x) - g[0]).abs() < 1e-12);
    assert!((of(mu) - (g[1] - 0.2)).abs() < 1e-12);
    assert!((of(xi) - (g[3] - 0.1)).abs() < 1e-12);
}

#[test]
fn a_latent_with_a_parameter_dependent_support_is_sampled_inside_it() {
    // forward samples of a bounded GEV (ξ < 0) and generalized Pareto stay below their upper ends
    let b = GraphBuilder::new();
    let gev = b.generalized_extreme_value_with_shape(&[1000], 0.0, 1.0, -0.5);
    let gpd = b.generalized_pareto_with_shape(&[1000], 1.0, 1.0, -0.5);
    let (gev, gpd) = (gev.id(), gpd.id());
    let (graph, mut state) = b.build().unwrap();
    let mut rng = KeanuRng::seed_from_u64(3);
    state.lazy_eval(&graph, &[gev, gpd], &mut rng).unwrap();
    assert!(state.f64(gev).unwrap().as_slice().iter().all(|&x| x < 2.0));
    assert!(
        state
            .f64(gpd)
            .unwrap()
            .as_slice()
            .iter()
            .all(|&x| (1.0..3.0).contains(&x))
    );
}
