//! Hilbert space approximate GPs: the approximate covariance matches the exact kernel's (AE6 for the exponentiated
//! quadratic, likewise the Matérn kernels, and 2-D inputs), the function at the inputs and its prediction share the
//! weights, gradients and Hessian-vector products in the hyperparameters and weights match finite differences, and
//! unsupported kernels and malformed inputs are errors.

use keanu_distributions::KeanuRng;
use keanu_graph::gp::{Hsgp, HsgpDomain, Kernel};
use keanu_graph::{
    DoubleVar, GraphBuilder, GraphError, Model, NodeId, Parameterisation, Space, State,
};
use keanu_tensor::DoubleTensor;

mod common;
use common::t;

/// Evenly spaced points on [−1, 1].
fn grid(n: usize) -> DoubleTensor {
    DoubleTensor::vector(
        (0..n)
            .map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64)
            .collect(),
    )
}

/// The largest absolute difference between the exact covariance of `kernel` at `x` and its HSGP approximation with
/// `m` basis functions per dimension and extension factor `c`.
fn covariance_error(
    kernel: impl for<'b> Fn(&'b GraphBuilder) -> Kernel<'b>,
    x: &DoubleTensor,
    m: usize,
    c: f64,
) -> f64 {
    let b = GraphBuilder::new();
    let kernel = kernel(&b);
    let prior = Hsgp::new(&kernel, m, c).unwrap().prior(x.clone());
    let approximate = prior.covariance(x.clone(), x.clone()).id();
    let exact = kernel.covariance(x.clone(), x.clone()).id();
    drop((prior, kernel));
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(
            &graph,
            &[approximate, exact],
            &mut KeanuRng::seed_from_u64(1),
        )
        .unwrap();
    let (a, e) = (state.f64(approximate).unwrap(), state.f64(exact).unwrap());
    assert_eq!(a.shape(), e.shape());
    a.as_slice()
        .iter()
        .zip(e.as_slice())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f64::max)
}

#[test]
fn forty_basis_functions_approximate_the_exponentiated_quadratic_to_1e_3() {
    // AE6: lengthscale 0.3 on [−1, 1]. Riutort-Mayol et al.'s rule m ≥ 1.75 c / (ℓ / S) asks for m ≥ 12 at c = 2,
    // which puts the box's edges a full range from the data. At c = 1.5 the edges are close enough to distort the
    // covariance at the ends of the range by 4e-3, whatever m is.
    let error = covariance_error(|b| Kernel::exp_quad(b, 1.0, 0.3), &grid(101), 40, 2.0);
    assert!(error < 1e-3, "error {error}");
    // the error scales with the amplitude squared
    let scaled = covariance_error(|b| Kernel::exp_quad(b, 2.0, 0.3), &grid(101), 40, 2.0);
    assert!(scaled < 4e-3, "error {scaled}");
    let near_edges = covariance_error(|b| Kernel::exp_quad(b, 1.0, 0.3), &grid(101), 40, 1.5);
    assert!(near_edges > 1e-3, "error {near_edges}");
}

#[test]
fn matern_kernels_are_approximated_too_and_improve_with_more_basis_functions() {
    let x = grid(101);
    // the spectra decay as ω^−(2ν+1), so the rougher kernels need more basis functions
    let matern52 = covariance_error(|b| Kernel::matern52(b, 1.0, 0.3), &x, 40, 2.0);
    assert!(matern52 < 1e-3, "Matérn 5/2: {matern52}");
    let matern32 = covariance_error(|b| Kernel::matern32(b, 1.0, 0.3), &x, 80, 2.0);
    assert!(matern32 < 1e-3, "Matérn 3/2: {matern32}");
    let matern12 = |m| covariance_error(|b| Kernel::matern12(b, 1.0, 0.3), &x, m, 2.0);
    let (coarse, fine) = (matern12(40), matern12(200));
    assert!(fine < 0.5 * coarse, "Matérn 1/2: {coarse} then {fine}");
    assert!(fine < 0.02, "Matérn 1/2: {fine}");
}

#[test]
fn the_general_matern_kernel_is_approximated_and_matches_the_half_integer_spectra() {
    let x = grid(101);
    // ν between 3/2 and 5/2 needs a number of basis functions between theirs
    let error = covariance_error(|b| Kernel::matern(b, 2.0, 1.0, 0.3), &x, 60, 2.0);
    assert!(error < 1e-3, "Matérn 2: {error}");
    // the same spectral density as the closed forms: the approximations agree to rounding
    for (k, nu) in [0.5, 1.5, 2.5].into_iter().enumerate() {
        let b = GraphBuilder::new();
        let general = Hsgp::new(&Kernel::matern(&b, nu, 1.2, 0.4), 30, 1.5).unwrap();
        let closed = [Kernel::matern12, Kernel::matern32, Kernel::matern52][k](&b, 1.2, 0.4);
        let closed = Hsgp::new(&closed, 30, 1.5).unwrap();
        let (general, closed) = (general.prior(x.clone()), closed.prior(x.clone()));
        let ids = [
            general.covariance(x.clone(), x.clone()).id(),
            closed.covariance(x.clone(), x.clone()).id(),
        ];
        drop((general, closed));
        let (graph, mut state) = b.build().unwrap();
        state
            .lazy_eval(&graph, &ids, &mut KeanuRng::seed_from_u64(1))
            .unwrap();
        let (a, c) = (state.f64(ids[0]).unwrap(), state.f64(ids[1]).unwrap());
        for (a, c) in a.as_slice().iter().zip(c.as_slice()) {
            assert!((a - c).abs() < 1e-12, "ν = {nu}: {a} vs {c}");
        }
    }
}

#[test]
fn two_dimensional_inputs_use_a_tensor_grid_of_basis_functions() {
    // a 7 × 7 grid on [−1, 1]²
    let side = grid(7);
    let mut points = Vec::new();
    for &u in side.as_slice() {
        for &v in side.as_slice() {
            points.extend([u, v]);
        }
    }
    let x = t(&[49, 2], &points);
    // the box's edges are a range away from the data, where the boundary's error is about exp(−2 / ℓ²) for an
    // exponentiated quadratic: 4e-6 for ℓ = 0.4 (but 0.04 for ℓ = 0.8)
    let ard = DoubleTensor::vector(vec![0.3, 0.4]);
    let error = covariance_error(|b| Kernel::exp_quad(b, 1.0, ard.clone()), &x, 20, 2.0);
    assert!(error < 1e-3, "exp_quad ARD: {error}");
    let error = covariance_error(|b| Kernel::matern52(b, 1.0, 0.3), &x, 40, 2.0);
    assert!(error < 1e-3, "Matérn 5/2: {error}");

    let b = GraphBuilder::new();
    let prior = Hsgp::new(&Kernel::exp_quad(&b, 1.0, ard), 20, 2.0)
        .unwrap()
        .prior(x);
    assert_eq!(prior.beta.shape(), [400]);
    assert_eq!(prior.f.shape(), [49]);
    let domain = prior.domain().unwrap();
    assert_eq!(domain.centre(), [0.0, 0.0]);
    assert_eq!(domain.half_width(), [2.0, 2.0]);
}

#[test]
fn the_prediction_uses_the_same_weights() {
    let x = DoubleTensor::vector(vec![0.0, 0.5, 1.5, 3.0]);
    let b = GraphBuilder::new();
    let length_scale = b.half_gaussian(1.0);
    b.set_value(length_scale, DoubleTensor::scalar(0.8));
    let prior = Hsgp::new(&Kernel::matern32(&b, 1.3, length_scale), 15, 1.5)
        .unwrap()
        .prior(x.clone());
    let again = prior.predict(x.clone());
    let new = prior.predict(DoubleTensor::vector(vec![-0.75, 3.75]));
    let ids = [prior.f.id(), again.id(), new.id(), prior.beta.id()];
    let phi = prior.basis(&x).unwrap();
    drop(prior);
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &ids, &mut KeanuRng::seed_from_u64(4))
        .unwrap();
    let f = state.f64(ids[0]).unwrap().clone();
    assert_eq!(&f, state.f64(ids[1]).unwrap());
    assert_eq!(state.f64(ids[2]).unwrap().shape(), &[2]);
    assert_eq!(phi.shape(), &[4, 15]);
    // the domain is [−0.75, 3.75], where every basis function vanishes
    for &edge in state.f64(ids[2]).unwrap().as_slice() {
        assert!(edge.abs() < 1e-12, "{edge}");
    }
    // a new draw of the weights changes the function
    let beta = state.f64(ids[3]).unwrap().clone();
    state
        .set_and_cascade(
            &graph,
            ids[3],
            beta.mul(&DoubleTensor::scalar(2.0)).unwrap(),
        )
        .unwrap();
    let doubled = state.f64(ids[0]).unwrap();
    for (a, b) in f.as_slice().iter().zip(doubled.as_slice()) {
        assert!((2.0 * a - b).abs() < 1e-12);
    }
}

#[test]
fn an_explicit_domain_replaces_the_one_from_the_inputs() {
    let b = GraphBuilder::new();
    let domain = HsgpDomain::new(vec![1.0], vec![5.0]).unwrap();
    let prior = Hsgp::new(&Kernel::exp_quad(&b, 1.0, 1.0), 10, 1.5)
        .unwrap()
        .with_domain(domain.clone())
        .prior(DoubleTensor::vector(vec![0.0, 1.0]));
    assert_eq!(prior.domain(), Some(&domain));
    // beyond the inputs' range, inside the domain
    prior.predict(DoubleTensor::vector(vec![5.5]));
    drop(prior);
    b.build().unwrap();
    let from_inputs =
        HsgpDomain::from_inputs(&t(&[3, 2], &[0.0, 1.0, 2.0, 1.0, 4.0, 3.0]), 1.5).unwrap();
    assert_eq!(from_inputs.centre(), [2.0, 2.0]);
    assert_eq!(from_inputs.half_width(), [3.0, 1.5]);
}

/// A model with latent hyperparameters (amplitude, length scale) and weights, observed through a Gaussian
/// likelihood, with `kernel` built from the two hyperparameters.
fn hsgp_model(
    kernel: impl for<'b> Fn(&'b GraphBuilder, DoubleVar<'b>, DoubleVar<'b>) -> Kernel<'b>,
) -> (Model, State) {
    let x = DoubleTensor::vector(vec![-1.0, -0.3, 0.2, 0.9, 1.4]);
    let y = DoubleTensor::vector(vec![0.3, -0.5, 0.1, 0.8, -0.2]);
    let b = GraphBuilder::new();
    let amplitude = b.half_gaussian(2.0);
    let length_scale = b.half_gaussian(2.0);
    b.set_value(amplitude, DoubleTensor::scalar(1.2));
    b.set_value(length_scale, DoubleTensor::scalar(0.7));
    let prior = Hsgp::new(&kernel(&b, amplitude, length_scale), 6, 1.5)
        .unwrap()
        .prior(x);
    let weights: Vec<f64> = (0..6)
        .map(|i| 0.4 * ((i as f64) * 1.7 + 0.3).sin())
        .collect();
    b.set_value(prior.beta, DoubleTensor::vector(weights));
    let observed = b.gaussian(prior.f, 0.4);
    b.observe(observed, y);
    drop(prior);
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(2))
        .unwrap();
    let model = Model::new(graph, &state).unwrap();
    (model, state)
}

#[test]
fn gradients_and_hessian_vector_products_match_finite_differences() {
    type KernelFn = for<'b> fn(&'b GraphBuilder, DoubleVar<'b>, DoubleVar<'b>) -> Kernel<'b>;
    let kernels: [(&str, KernelFn); 5] = [
        ("exp_quad", |b, a, l| Kernel::exp_quad(b, a, l)),
        ("matern12", |b, a, l| Kernel::matern12(b, a, l)),
        ("matern32", |b, a, l| Kernel::matern32(b, a, l)),
        ("matern52", |b, a, l| Kernel::matern52(b, a, l)),
        ("matern 1.8", |b, a, l| Kernel::matern(b, 1.8, a, l)),
    ];
    for (name, kernel) in kernels {
        let (model, mut state) = hsgp_model(kernel);
        let param = Parameterisation::with_space(&model, &state, Space::Unconstrained).unwrap();
        let z = param.from_state(&state).unwrap();
        let n = z.len();
        assert_eq!(n, 8, "{name}");
        let mut gradient = |z: &[f64]| {
            let (_, g) = param.log_prob_and_gradient(z, true, &mut state).unwrap();
            g.unwrap()
        };
        let g = gradient(&z);
        let v: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.7 + 0.3).sin()).collect();
        let h = 1e-5;
        let shifted = |s: f64| -> Vec<f64> { z.iter().zip(&v).map(|(z, v)| z + s * v).collect() };
        let (plus, minus) = (gradient(&shifted(h)), gradient(&shifted(-h)));
        let hv = param.hvp(&z, &v, true, &mut state).unwrap();
        for i in 0..n {
            let fd = (plus[i] - minus[i]) / (2.0 * h);
            assert!(
                (hv[i] - fd).abs() <= 1e-6 * fd.abs().max(1.0),
                "{name}: (Hv)[{i}] = {} vs {fd}",
                hv[i]
            );
        }
        // the gradient against differences of the log probability
        for (j, &gj) in g.iter().enumerate() {
            let mut at = |s: f64| {
                let mut p = z.clone();
                p[j] += s;
                param.log_prob(&p, true, &mut state).unwrap()
            };
            let fd = (at(h) - at(-h)) / (2.0 * h);
            assert!(
                (gj - fd).abs() <= 1e-6 * fd.abs().max(1.0),
                "{name}: gradient[{j}] = {gj} vs {fd}"
            );
        }
    }
}

fn hsgp_error(kernel: impl for<'b> Fn(&'b GraphBuilder) -> Kernel<'b>, m: usize, c: f64) -> String {
    let b = GraphBuilder::new();
    match Hsgp::new(&kernel(&b), m, c) {
        Err(GraphError::InvalidNode { op, message }) => {
            assert_eq!(op, "hsgp");
            message
        }
        other => panic!("expected an error, got {other:?}"),
    }
}

fn build_error(f: impl FnOnce(&GraphBuilder)) -> String {
    let b = GraphBuilder::new();
    f(&b);
    match b.build() {
        Err(GraphError::InvalidNode { message, .. }) => message,
        Err(e) => panic!("expected an invalid node, got {e}"),
        Ok(_) => panic!("expected the build to fail"),
    }
}

#[test]
fn kernels_without_a_spectral_density_and_bad_settings_are_errors() {
    let message = hsgp_error(|b| Kernel::periodic(b, 1.0, 1.0, 2.0), 10, 1.5);
    assert!(
        message.contains("periodic") && message.contains("spectral density"),
        "{message}"
    );
    let message = hsgp_error(|b| Kernel::rational_quadratic(b, 1.0, 1.0, 2.0), 10, 1.5);
    assert!(message.contains("rational quadratic"), "{message}");
    let message = hsgp_error(|b| Kernel::linear(b, 1.0, 1.0, 0.0), 10, 1.5);
    assert!(message.contains("linear"), "{message}");
    let message = hsgp_error(
        |b| Kernel::exp_quad(b, 1.0, 1.0) + Kernel::matern12(b, 1.0, 1.0),
        10,
        1.5,
    );
    assert!(
        message.contains("sum") && message.contains("one HSGP per term"),
        "{message}"
    );
    let message = hsgp_error(
        |b| Kernel::exp_quad(b, 1.0, 1.0) * Kernel::matern12(b, 1.0, 1.0),
        10,
        1.5,
    );
    assert!(message.contains("product"), "{message}");
    assert!(hsgp_error(|b| Kernel::exp_quad(b, 1.0, 1.0), 0, 1.5).contains("basis function"));
    for c in [0.9, f64::NAN, f64::INFINITY] {
        assert!(hsgp_error(|b| Kernel::exp_quad(b, 1.0, 1.0), 10, c).contains("at least 1"));
    }
    assert!(HsgpDomain::new(vec![0.0], vec![0.0]).is_err());
    assert!(HsgpDomain::new(vec![0.0, 1.0], vec![1.0]).is_err());
    assert!(HsgpDomain::new(vec![f64::NAN], vec![1.0]).is_err());
}

#[test]
fn malformed_inputs_are_build_errors() {
    fn hsgp(b: &GraphBuilder, l: DoubleTensor) -> Hsgp<'_> {
        Hsgp::new(&Kernel::exp_quad(b, 1.0, l), 5, 1.5).unwrap()
    }
    let message = build_error(|b| {
        hsgp(b, DoubleTensor::scalar(1.0)).prior(DoubleTensor::zeros(&[2, 2, 2]));
    });
    assert!(message.contains("vector [n] or a matrix"), "{message}");
    let message = build_error(|b| {
        hsgp(b, DoubleTensor::vector(vec![1.0, 2.0, 3.0])).prior(t(&[2, 2], &[0.0, 0.0, 1.0, 1.0]));
    });
    assert!(message.contains("length scale"), "{message}");
    let message = build_error(|b| {
        hsgp(b, DoubleTensor::scalar(1.0)).prior(DoubleTensor::vector(vec![1.0, 1.0]));
    });
    assert!(message.contains("no spread"), "{message}");
    let message = build_error(|b| {
        hsgp(b, DoubleTensor::scalar(1.0)).prior(DoubleTensor::vector(vec![]));
    });
    assert!(message.contains("no points"), "{message}");
    let message = build_error(|b| {
        let prior = hsgp(b, DoubleTensor::scalar(1.0)).prior(DoubleTensor::vector(vec![0.0, 1.0]));
        // the domain is [−0.25, 1.25]
        prior.predict(DoubleTensor::vector(vec![1.3]));
    });
    assert!(message.contains("outside the domain"), "{message}");
    let message = build_error(|b| {
        let prior = hsgp(b, DoubleTensor::scalar(1.0)).prior(DoubleTensor::vector(vec![0.0, 1.0]));
        prior.predict(t(&[1, 2], &[0.5, 0.5]));
    });
    assert!(message.contains("dimensions"), "{message}");
    let message = build_error(|b| {
        let kernel = Kernel::exp_quad(b, 1.0, 1.0);
        Hsgp::new(&kernel, 2000, 1.5)
            .unwrap()
            .prior(t(&[2, 2], &[0.0, 0.0, 1.0, 1.0]));
    });
    assert!(message.contains("more than"), "{message}");
    let message = build_error(|b| {
        let kernel = Kernel::exp_quad(b, 1.0, 1.0);
        Hsgp::new(&kernel, 5, 1.5)
            .unwrap()
            .with_domain(HsgpDomain::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap())
            .prior(DoubleTensor::vector(vec![0.0, 0.5]));
    });
    assert!(message.contains("dimensions"), "{message}");
}
