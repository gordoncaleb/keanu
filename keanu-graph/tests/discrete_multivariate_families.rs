//! The discrete and multivariate families of the distribution tail as graph nodes: the element-wise ones (Skellam,
//! hypergeometric) as terms, log-density ops and mixture components; the supports, finite supports and value
//! shapes of every kind; and gradients through the graph in unconstrained space. Their Hessians are in
//! `hessian.rs`, their transforms in `transform.rs`, saving in keanu-io's round trip.

use keanu_distributions as d;
use keanu_distributions::KeanuRng;
use keanu_graph::op::DistributionKind;
use keanu_graph::{
    GraphBuilder, IntVar, Model, NodeId, Parameterisation, Space, Support, Value, node_log_prob,
    node_log_prob_terms, node_support,
};
use keanu_tensor::{DoubleTensor, IntTensor};

mod common;
use common::{evaluate, s, t, v};

fn ints(values: &[i64]) -> IntTensor {
    IntTensor::vector(values.to_vec())
}

#[test]
fn skellam_and_hypergeometric_terms_log_densities_and_mixtures() {
    let b = GraphBuilder::new();
    let (mu1, mu2) = (b.scalar(1.7), b.scalar(0.6));
    let skellam = b.skellam_with_shape(&[3], mu1, mu2);
    b.observe(skellam, ints(&[-1, 0, 4]));
    let density = b
        .log_density(
            DistributionKind::Skellam,
            b.int_constant(ints(&[-1, 0, 4])),
            &[mu1.id(), mu2.id()],
        )
        .id();
    let hyper = b.hyper_geometric_with_shape(&[3], 20, 7, 5);
    b.observe(hyper, ints(&[0, 2, 5]));
    // a two-component Skellam mixture
    let mixture: IntVar<'_> = b.mixture(
        DistributionKind::Skellam,
        &[
            b.constant(v(&[1.7, 3.0])).id(),
            b.constant(v(&[0.6, 0.6])).id(),
        ],
        b.constant(v(&[0.3, 0.7])),
    );
    b.observe(mixture, IntTensor::scalar(2));
    let (skellam, hyper, mixture) = (skellam.id(), hyper.id(), mixture.id());
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 1);

    let expected = d::Skellam::new(s(1.7), s(0.6))
        .log_prob(&ints(&[-1, 0, 4]))
        .unwrap();
    assert_eq!(
        node_log_prob_terms(&graph, &state, skellam).unwrap(),
        expected
    );
    assert_eq!(state.f64(density).unwrap(), &expected);
    let expected = d::HyperGeometric::new(
        IntTensor::scalar(20),
        IntTensor::scalar(7),
        IntTensor::scalar(5),
    )
    .log_prob(&ints(&[0, 2, 5]))
    .unwrap();
    assert_eq!(
        node_log_prob_terms(&graph, &state, hyper).unwrap(),
        expected
    );
    let p = |m1: f64| {
        d::Skellam::new(s(m1), s(0.6))
            .log_prob(&IntTensor::scalar(2))
            .unwrap()
            .as_slice()[0]
            .exp()
    };
    let expected = (0.3 * p(1.7) + 0.7 * p(3.0)).ln();
    let actual = node_log_prob(&graph, &state, mixture).unwrap();
    assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");
}

#[test]
fn supports_and_finite_supports() {
    let b = GraphBuilder::new();
    let scale = b.constant(t(&[2, 2], &[1.0, 0.2, 0.2, 2.0]));
    let factor = b.constant(t(&[2, 2], &[1.0, 0.0, 0.2, 1.4]));
    let mu = b.constant(v(&[0.0, 0.6, 0.8]));
    let nodes = [
        (b.wishart(4.0, scale).id(), Support::CovarianceMatrix),
        (
            b.inverse_wishart(4.0, scale).id(),
            Support::CovarianceMatrix,
        ),
        (
            b.wishart_cholesky(4.0, factor).id(),
            Support::CovarianceCholesky,
        ),
        (
            b.inverse_wishart_cholesky(4.0, factor).id(),
            Support::CovarianceCholesky,
        ),
        (b.von_mises_fisher(mu, 2.0).id(), Support::UnitVector),
        (
            b.multivariate_student_t(4.0, b.constant(v(&[0.0, 1.0])), scale)
                .id(),
            Support::Real,
        ),
        (
            b.matrix_normal(
                b.constant(t(&[2, 1], &[0.0, 1.0])),
                scale,
                b.constant(t(&[1, 1], &[1.0])),
            )
            .id(),
            Support::Real,
        ),
        (b.skellam(1.0, 2.0).id(), Support::Discrete),
        (
            b.plackett_luce(b.constant(v(&[1.0, 2.0]))).id(),
            Support::Discrete,
        ),
    ];
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 2);
    for (id, support) in nodes {
        assert_eq!(node_support(&graph, &state, id).unwrap(), support);
    }

    // finitely many values: the hypergeometric's range per element, the ordered probit's categories
    let int = |values: &[i64]| Value::I64(ints(values));
    let range = DistributionKind::HyperGeometric
        .finite_support(&[&int(&[20, 10]), &int(&[7, 8]), &int(&[5, 9])], &[2])
        .unwrap();
    assert_eq!(range, Some(vec![0..=5, 7..=8]));
    let cutpoints = Value::F64(v(&[-1.0, 0.0, 1.0]));
    let categories = DistributionKind::OrderedProbit
        .finite_support(&[&Value::F64(s(0.3)), &cutpoints], &[2])
        .unwrap();
    assert_eq!(categories, Some(vec![0..=3, 0..=3]));
    for kind in [
        DistributionKind::OrderedProbit,
        DistributionKind::HyperGeometric,
    ] {
        assert!(kind.has_finite_support());
    }
    // the elements of these can't be changed one at a time, and the Skellam's values are unbounded
    for kind in [
        DistributionKind::DirichletMultinomial,
        DistributionKind::PlackettLuce,
        DistributionKind::Skellam,
    ] {
        assert!(!kind.has_finite_support());
    }
}

#[test]
fn value_shapes_follow_the_events_and_batches() {
    let b = GraphBuilder::new();
    let scale = b.constant(t(&[2, 2], &[1.0, 0.2, 0.2, 2.0]));
    // a batch of three Wisharts with one ν each
    let w = b.wishart_with_shape(&[3, 2, 2], b.constant(v(&[4.0, 5.0, 6.0])), scale);
    assert_eq!(w.shape().as_slice(), &[3, 2, 2]);
    let w = b.wishart(b.constant(v(&[4.0, 5.0])), scale);
    assert_eq!(w.shape().as_slice(), &[2, 2, 2]);
    let t_node = b.multivariate_student_t(4.0, b.constant(t(&[4, 2], &[0.0; 8])), scale);
    assert_eq!(t_node.shape().as_slice(), &[4, 2]);
    let m = b.matrix_normal(
        b.constant(DoubleTensor::zeros(&[2, 3])),
        scale,
        b.constant(DoubleTensor::eye(3).unwrap()),
    );
    assert_eq!(m.shape().as_slice(), &[2, 3]);
    let dm = b.dirichlet_multinomial(
        b.int_constant(ints(&[3, 4])),
        b.constant(v(&[1.0, 2.0, 3.0])),
    );
    assert_eq!(dm.shape().as_slice(), &[2, 3]);
    b.build().unwrap();

    // mismatched events are build errors
    for build in [
        |b: &GraphBuilder| {
            b.wishart(4.0, b.constant(t(&[2, 3], &[0.0; 6])));
        },
        |b: &GraphBuilder| {
            let scale = b.constant(DoubleTensor::eye(3).unwrap());
            b.multivariate_student_t(4.0, b.constant(v(&[0.0, 1.0])), scale);
        },
        |b: &GraphBuilder| {
            b.von_mises_fisher(b.constant(v(&[1.0])), 1.0);
        },
        |b: &GraphBuilder| {
            let scale = b.constant(DoubleTensor::eye(2).unwrap());
            b.wishart_with_shape(&[3, 3], 4.0, scale);
        },
    ] {
        let b = GraphBuilder::new();
        build(&b);
        assert!(b.build().is_err());
    }
}

/// Central differences of the model's log probability (with the Jacobian) in unconstrained space against its
/// gradient.
fn check_unconstrained_gradient(model: &Model, state: &mut keanu_graph::State) {
    let param = Parameterisation::with_space(model, state, Space::Unconstrained).unwrap();
    let z = param.from_state(state).unwrap();
    let (_, gradient) = param.log_prob_and_gradient(&z, true, state).unwrap();
    let gradient = gradient.unwrap();
    for i in 0..z.len() {
        let h = 1e-6;
        let at = |delta: f64, state: &mut keanu_graph::State| {
            let mut moved = z.clone();
            moved[i] += delta;
            param.log_prob_and_gradient(&moved, true, state).unwrap().0
        };
        let fd = (at(h, state) - at(-h, state)) / (2.0 * h);
        assert!(
            (gradient[i] - fd).abs() <= 1e-5 * fd.abs().max(1.0),
            "coordinate {i}: {} vs {fd}",
            gradient[i]
        );
    }
}

#[test]
fn gradients_through_the_transforms_match_finite_differences() {
    // a Wishart covariance of a multivariate normal, an LKJ-free Cholesky-factor prior and a von Mises-Fisher
    // direction, all latent
    let b = GraphBuilder::new();
    let scale = b.constant(t(&[2, 2], &[1.0, 0.3, 0.3, 1.5]));
    let covariance = b.inverse_wishart(5.0, scale);
    b.set_value(covariance, t(&[2, 2], &[0.8, 0.1, 0.1, 1.2]));
    let y = b.multivariate_gaussian(b.constant(v(&[0.0, 0.0])), covariance);
    b.observe(y, v(&[0.4, -1.1]));
    let factor = b.wishart_cholesky(4.0, b.constant(t(&[2, 2], &[1.0, 0.0, 0.2, 0.9])));
    b.set_value(factor, t(&[2, 2], &[1.2, 0.0, -0.3, 0.7]));
    let z = b.multivariate_student_t_cholesky(5.0, b.constant(v(&[0.0, 0.0])), factor);
    b.observe(z, v(&[0.2, 0.5]));
    let direction = b.von_mises_fisher(b.constant(v(&[0.0, 0.6, 0.8])), 3.0);
    b.set_value(direction, v(&[0.0, 1.0, 0.0]));
    let w = b.von_mises_fisher(direction, 5.0);
    b.observe(w, v(&[0.6, 0.0, 0.8]));
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 3);
    let model = Model::new(graph, &state).unwrap();
    check_unconstrained_gradient(&model, &mut state);
}

#[test]
fn a_latent_ranking_and_counts_sample_from_their_priors() {
    let b = GraphBuilder::new();
    let ranking = b.plackett_luce(b.constant(v(&[1.0, 2.0, 3.0])));
    let counts = b.dirichlet_multinomial_with_shape(&[4, 3], 6, b.constant(v(&[1.0, 2.0, 3.0])));
    let probit = b.ordered_probit_with_shape(&[5], 0.3, b.constant(v(&[-0.5, 0.5])));
    let ids: Vec<NodeId> = vec![ranking.id(), counts.id(), probit.id()];
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &ids, &mut KeanuRng::seed_from_u64(4))
        .unwrap();
    let mut sorted = state.i64(ids[0]).unwrap().as_slice().to_vec();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2]);
    assert!(
        state
            .i64(ids[1])
            .unwrap()
            .as_slice()
            .chunks(3)
            .all(|c| c.iter().sum::<i64>() == 6)
    );
    assert!(
        state
            .i64(ids[2])
            .unwrap()
            .as_slice()
            .iter()
            .all(|&k| (0..3).contains(&k))
    );
    for &id in &ids {
        assert!(node_log_prob(&graph, &state, id).unwrap().is_finite());
    }
}
