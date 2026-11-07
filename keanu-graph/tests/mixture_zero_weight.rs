//! The Hessian of a mixture's log density with respect to its weights where a weight is 0 or subnormal: with
//! L = ln(w g + (1 − w) l), d²L/dw² = −(g − l)² / p², finite at w = 0, not NaN or 0.

use keanu_graph::op::DistributionKind as K;
use keanu_graph::{
    DoubleVar, GraphBuilder, Model, NodeId, Parameterisation, Space, node_log_prob,
    node_log_prob_terms,
};
use keanu_tensor::DoubleTensor;
use keanu_tensor::special::ln_gamma;
use std::f64::consts::PI;

const X: f64 = 1.1;

fn gamma_pdf(x: f64, theta: f64, k: f64) -> f64 {
    ((k - 1.0) * x.ln() - x / theta - k * theta.ln() - ln_gamma(k)).exp()
}

fn log_normal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x.ln() - mu) / sigma;
    (-x.ln() - sigma.ln() - 0.5 * (2.0 * PI).ln() - 0.5 * z * z).exp()
}

fn gaussian_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    (-0.5 * z * z).exp() / (sigma * (2.0 * PI).sqrt())
}

/// The Hessian with respect to w ~ N(0, 1) of a model with a two-component mixture observed at [`X`], weighted
/// (w, 1 − w), with `mixture` making the mixture node from the weights.
fn hessian_wrt_weight(
    w: f64,
    mixture: impl for<'b> Fn(&'b GraphBuilder, DoubleVar<'b>) -> DoubleVar<'b>,
) -> f64 {
    let b = GraphBuilder::new();
    let weight = b.gaussian(0.0, 1.0);
    b.set_value(weight, DoubleTensor::scalar(w));
    let weights = b.stack(-1, &[weight, -weight + 1.0]);
    let y = mixture(&b, weights);
    b.observe(y, DoubleTensor::scalar(X));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let param = Parameterisation::with_space(&model, &state, Space::Constrained).unwrap();
    let z = param.from_state(&state).unwrap();
    assert_eq!(z, vec![w]);
    let hessian = param.hessian(&z, false, &mut state).unwrap();
    hessian.as_slice()[0]
}

#[test]
fn a_heterogeneous_mixtures_weight_hessian_is_finite_at_a_zero_weight() {
    let (g, l) = (gamma_pdf(X, 1.3, 2.2), log_normal_pdf(X, 0.4, 0.7));
    for w in [0.0, 1e-300, 0.3] {
        let actual = hessian_wrt_weight(w, |b, weights| {
            let ids: Vec<NodeId> = [1.3, 2.2, 0.4, 0.7]
                .iter()
                .map(|&v| b.scalar(v).id())
                .collect();
            b.mixture_of(&[(K::Gamma, &ids[..2]), (K::LogNormal, &ids[2..])], weights)
        });
        let p = w * g + (1.0 - w) * l;
        let expected = -1.0 - (g - l).powi(2) / (p * p);
        assert!(
            (actual - expected).abs() <= 1e-12 * expected.abs(),
            "w = {w}: {actual} vs {expected}"
        );
    }
}

#[test]
fn a_mixtures_weight_hessian_is_finite_at_a_zero_weight() {
    let (g, l) = (gaussian_pdf(X, 0.0, 1.0), gaussian_pdf(X, 2.0, 0.5));
    for w in [0.0, 1e-300, 0.3] {
        let actual = hessian_wrt_weight(w, |b, weights| {
            let mu = b.constant(DoubleTensor::vector(vec![0.0, 2.0]));
            let sigma = b.constant(DoubleTensor::vector(vec![1.0, 0.5]));
            b.mixture(K::Gaussian, &[mu.id(), sigma.id()], weights)
        });
        let p = w * g + (1.0 - w) * l;
        let expected = -1.0 - (g - l).powi(2) / (p * p);
        assert!(
            (actual - expected).abs() <= 1e-12 * expected.abs(),
            "w = {w}: {actual} vs {expected}"
        );
    }
}

#[test]
fn a_zero_weight_masks_a_singular_component_at_the_boundary() {
    for heterogeneous in [false, true] {
        let b = GraphBuilder::new();
        let beta = b.gaussian(2.0, 1.0);
        b.set_value(beta, DoubleTensor::scalar(2.0));
        let weights = b.constant(DoubleTensor::vector(vec![1.0, 0.0]));
        let y: DoubleVar = if heterogeneous {
            b.mixture_of(
                &[
                    (K::Beta, &[b.scalar(1.0).id(), beta.id()]),
                    (K::Beta, &[b.scalar(0.5).id(), beta.id()]),
                ],
                weights,
            )
        } else {
            let alpha = b.constant(DoubleTensor::vector(vec![1.0, 0.5]));
            b.mixture(K::Beta, &[alpha.id(), beta.id()], weights)
        };
        b.observe(y, DoubleTensor::scalar(0.0));
        let (y, beta) = (y.id(), beta.id());
        let (graph, mut state) = b.build().unwrap();
        let expected = 2.0_f64.ln();
        let total = node_log_prob(&graph, &state, y).unwrap();
        let pointwise = node_log_prob_terms(&graph, &state, y).unwrap();
        assert!((total - expected).abs() < 1e-12, "log probability: {total}");
        assert!((pointwise.as_slice()[0] - expected).abs() < 1e-12);

        let model = Model::new(graph, &state).unwrap();
        let (_, gradient) = model.log_prob_and_gradient(&state).unwrap();
        let gradient = gradient.unwrap();
        assert!((gradient.get(beta).unwrap().as_slice()[0] - 0.5).abs() < 1e-12);
        let param = Parameterisation::with_space(&model, &state, Space::Constrained).unwrap();
        let hessian = param.hessian(&[2.0], false, &mut state).unwrap();
        assert!((hessian.as_slice()[0] + 1.25).abs() < 1e-12);
    }
}

#[test]
fn fixed_zero_weights_cannot_poison_the_hessian_of_component_parameters() {
    for heterogeneous in [false, true] {
        let b = GraphBuilder::new();
        let mu = b.gaussian(0.0, 1.0);
        b.set_value(mu, DoubleTensor::scalar(0.0));
        let weights = b.constant(DoubleTensor::vector(vec![1.0, 0.0]));
        let other_mu = b.scalar(40.0);
        let sigma = b.scalar(1.0);
        let y: DoubleVar = if heterogeneous {
            b.mixture_of(
                &[
                    (K::Gaussian, &[mu.id(), sigma.id()]),
                    (K::Gaussian, &[other_mu.id(), sigma.id()]),
                ],
                weights,
            )
        } else {
            let means = b.stack(-1, &[mu, other_mu]);
            b.mixture(K::Gaussian, &[means.id(), sigma.id()], weights)
        };
        b.observe(y, DoubleTensor::scalar(40.0));
        let (graph, mut state) = b.build().unwrap();
        let model = Model::new(graph, &state).unwrap();
        let param = Parameterisation::with_space(&model, &state, Space::Constrained).unwrap();
        let hessian = param.hessian(&[0.0], false, &mut state).unwrap();
        assert_eq!(hessian.as_slice(), &[-2.0], "heterogeneous={heterogeneous}");
    }
}
