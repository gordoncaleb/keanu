//! The LKJ Cholesky node: its shape rules, prior samples, log probability and gradients through the graph, and
//! Hamiltonian Monte Carlo in the unconstrained space of its correlation Cholesky bijector recovering the known
//! marginal variance of each correlation.

use keanu_distributions::{KeanuRng, LkjCholesky};
use keanu_graph::op::DistributionKind;
use keanu_graph::{GraphBuilder, Model, NodeId, Parameterisation, State, Support, node_support};
use keanu_tensor::DoubleTensor;

mod common;
use common::evaluate;

/// Asserts each K × K matrix of `l` is a correlation Cholesky factor.
fn assert_correlation_cholesky(l: &DoubleTensor, k: usize) {
    for matrix in l.as_slice().chunks(k * k) {
        for (i, row) in matrix.chunks(k).enumerate() {
            assert!(row[i] > 0.0, "diagonal of {row:?}");
            assert!(row[i + 1..].iter().all(|&x| x == 0.0), "upper of {row:?}");
            let length: f64 = row.iter().map(|x| x * x).sum();
            assert!((length - 1.0).abs() < 1e-12, "row length {length}");
        }
    }
}

#[test]
fn the_node_has_a_k_by_k_value_sampled_from_the_prior() {
    let b = GraphBuilder::new();
    let l = b.lkj_cholesky(4, 1.5);
    let batch = b.lkj_cholesky_with_shape(&[2, 3, 3], DoubleTensor::vector(vec![1.0, 3.0]));
    let (l, batch) = (l.id(), batch.id());
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 5);
    let value = state.f64(l).unwrap();
    assert_eq!(value.shape(), &[4, 4]);
    assert_correlation_cholesky(value, 4);
    let value = state.f64(batch).unwrap();
    assert_eq!(value.shape(), &[2, 3, 3]);
    assert_correlation_cholesky(value, 3);
    assert_eq!(
        node_support(&graph, &state, l).unwrap(),
        Support::CorrelationCholesky
    );
}

#[test]
fn shapes_that_are_not_square_matrices_or_a_mismatched_eta_are_rejected() {
    let b = GraphBuilder::new();
    let eta = b.constant(2.0).id();
    let etas = b.constant(DoubleTensor::vector(vec![1.0, 2.0, 3.0])).id();
    for (shape, eta) in [
        (None, eta),
        (Some(&[3][..]), eta),
        (Some(&[2, 3]), eta),
        (Some(&[0, 0]), eta),
        (Some(&[2, 2, 2]), etas),
    ] {
        assert!(
            b.add_distribution(DistributionKind::LkjCholesky, shape, &[eta])
                .is_err(),
            "{shape:?}"
        );
    }
    assert!(
        b.add_distribution(DistributionKind::LkjCholesky, Some(&[3, 2, 2]), &[etas])
            .is_ok()
    );
}

#[test]
fn log_prob_and_gradients_through_the_graph_with_a_latent_eta() {
    let b = GraphBuilder::new();
    let eta = b.gamma(1.0, 2.0).label("eta");
    let l = b.lkj_cholesky(3, eta).label("l");
    let (eta, l) = (eta.id(), l.id());
    let (graph, mut state) = b.build().unwrap();
    // L from partial correlations (0.3; −0.4, 0.5): row 2 is (−0.4, 0.5·√(1 − 0.16), √(1 − 0.16 − 0.21))
    let factor = DoubleTensor::new(
        &[3, 3],
        vec![
            1.0,
            0.0,
            0.0,
            0.3,
            (1.0f64 - 0.09).sqrt(),
            0.0,
            -0.4,
            0.5 * 0.84f64.sqrt(),
            (1.0f64 - 0.16 - 0.21).sqrt(),
        ],
    )
    .unwrap();
    state.set(&graph, eta, DoubleTensor::scalar(1.8)).unwrap();
    state.set(&graph, l, factor.clone()).unwrap();
    let model = Model::new(graph, &state).unwrap();

    // Gamma(theta 1, k 2) at 1.8: ln(1.8) − 1.8
    let lkj = LkjCholesky::new(DoubleTensor::scalar(1.8));
    let expected = lkj.log_prob(&factor).unwrap() + 1.8f64.ln() - 1.8;
    let (log_prob, gradient) = model.log_prob_and_gradient(&state).unwrap();
    assert!(
        (log_prob - expected).abs() < 1e-12,
        "{log_prob} vs {expected}"
    );

    let gradient = gradient.unwrap();
    let g = lkj.d_log_prob(&factor).unwrap();
    assert_eq!(gradient.get(l).unwrap(), &g.x);
    // d/dη of the Gamma term is 1 / η − 1
    let d_eta = gradient.get(eta).unwrap().as_slice()[0];
    let expected = g.eta.as_slice()[0] + 1.0 / 1.8 - 1.0;
    assert!((d_eta - expected).abs() < 1e-12, "{d_eta} vs {expected}");
    // and the whole thing against a central difference in η
    let h = 1e-6;
    let at = |value: f64| {
        let mut moved = state.clone();
        moved
            .set(model.graph(), eta, DoubleTensor::scalar(value))
            .unwrap();
        model.log_prob(&moved).unwrap()
    };
    let numeric = (at(1.8 + h) - at(1.8 - h)) / (2.0 * h);
    assert!((d_eta - numeric).abs() < 1e-6, "{d_eta} vs {numeric}");
}

/// Draws from the unconstrained density of `param` (with the Jacobian) by Hamiltonian Monte Carlo with a unit mass
/// matrix and a jittered number of leapfrog steps, returning the constrained values of `node` after warm-up.
fn hmc(
    param: &Parameterisation<'_>,
    state: &mut State,
    node: NodeId,
    step: f64,
    iterations: usize,
    rng: &mut KeanuRng,
) -> Vec<DoubleTensor> {
    let n = param.dimension();
    let mut z = param.from_state(state).unwrap();
    let log_prob_and_gradient = |z: &[f64], state: &mut State| {
        let (log_prob, gradient) = param.log_prob_and_gradient(z, true, state).unwrap();
        (log_prob, gradient.unwrap())
    };
    let (mut log_prob, mut gradient) = log_prob_and_gradient(&z, state);
    let warm_up = iterations / 10;
    let mut draws = Vec::with_capacity(iterations - warm_up);
    let mut accepted = 0;
    for iteration in 0..iterations {
        let momentum: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
        let kinetic = |p: &[f64]| p.iter().map(|p| p * p).sum::<f64>() / 2.0;
        let (mut z_new, mut p, mut g) = (z.clone(), momentum.clone(), gradient.clone());
        let mut lp = log_prob;
        for _ in 0..10 + rng.next_below(10) {
            p.iter_mut().zip(&g).for_each(|(p, g)| *p += step / 2.0 * g);
            z_new.iter_mut().zip(&p).for_each(|(z, p)| *z += step * p);
            (lp, g) = log_prob_and_gradient(&z_new, state);
            p.iter_mut().zip(&g).for_each(|(p, g)| *p += step / 2.0 * g);
        }
        let log_accept = lp - kinetic(&p) - (log_prob - kinetic(&momentum));
        if rng.next_f64().ln() < log_accept {
            (z, log_prob, gradient) = (z_new, lp, g);
            accepted += 1;
        }
        if iteration >= warm_up {
            param.to_state(&z, state).unwrap();
            draws.push(state.f64(node).unwrap().clone());
        }
    }
    let rate = accepted as f64 / iterations as f64;
    assert!(rate > 0.6, "acceptance rate {rate}");
    draws
}

#[test]
fn hmc_on_lkj_eta_2_k_3_recovers_the_marginal_correlation_variance() {
    // Lewandowski, Kurowicka & Joe (2009), section 3.2: under LKJ(η) in dimension K each off-diagonal correlation
    // r has (r + 1) / 2 ~ Beta(η − 1 + K / 2, η − 1 + K / 2), so Var(r) = 1 / (2η + K − 1): 1 / 6 for η = 2, K = 3.
    // Sampling the node alone in the unconstrained space of its bijector, with the Jacobian, must reproduce it,
    // which checks the density, its gradient, the bijector and its log-Jacobian together.
    //
    // A small HMC driver over the parameterisation stands in for NUTS until NUTS runs in unconstrained space (U4).
    let (eta, k) = (2.0, 3);
    let b = GraphBuilder::new();
    let l = b.lkj_cholesky(k, eta).id();
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 1);
    let model = Model::new(graph, &state).unwrap();
    let param = Parameterisation::new(&model, &state).unwrap();
    assert_eq!(param.dimension(), 3);

    let mut rng = KeanuRng::seed_from_u64(2024);
    let draws = hmc(&param, &mut state, l, 0.25, 12_000, &mut rng);
    let variance = 1.0 / (2.0 * eta + k as f64 - 1.0);
    let n = draws.len() as f64;
    for (i, j) in [(1, 0), (2, 0), (2, 1)] {
        let correlations: Vec<f64> = draws
            .iter()
            .map(|l| {
                let l = l.as_slice();
                (0..k).map(|m| l[i * k + m] * l[j * k + m]).sum()
            })
            .collect();
        let mean = correlations.iter().sum::<f64>() / n;
        let second = correlations.iter().map(|r| r * r).sum::<f64>() / n;
        // Monte Carlo error: sd(r²) ≈ 0.19 here, and HMC draws keep an effective sample size of several thousand,
        // so the standard error of each moment is below 0.004; the tolerances are about 4 of them
        assert!(mean.abs() < 0.02, "r[{i},{j}]: mean {mean}");
        assert!(
            (second - variance).abs() < 0.015,
            "r[{i},{j}]: variance {second} vs {variance}"
        );
    }
}
