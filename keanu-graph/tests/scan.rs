//! The general scan: values, first and second derivatives, edge lengths and the rules on bodies.

use keanu_distributions::KeanuRng;
use keanu_graph::op::{Op, ScanBody};
use keanu_graph::{
    DoubleVar, Graph, GraphBuilder, GraphError, Model, NodeId, Parameterisation, State,
    forward_partials, reverse_partials,
};
use keanu_tensor::DoubleTensor;

fn t(shape: &[usize], data: &[f64]) -> DoubleTensor {
    DoubleTensor::new(shape, data.to_vec()).unwrap()
}

fn rng() -> KeanuRng {
    KeanuRng::seed_from_u64(7)
}

fn evaluated(b: GraphBuilder) -> (Graph, State) {
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng()).unwrap();
    (graph, state)
}

fn assert_scan_error(error: Option<GraphError>, contains: &str) {
    match error {
        Some(GraphError::InvalidNode {
            op: "scan",
            message,
        }) if message.contains(contains) => {}
        other => panic!("expected a scan error containing {contains:?}, got {other:?}"),
    }
}

#[test]
fn a_cumulative_sum_scan_equals_cumsum() {
    let b = GraphBuilder::new();
    let xs = b.constant(t(&[4, 2], &[1.0, -2.0, 0.5, 3.0, 2.25, 1.0, -4.0, 0.125]));
    let zero = b.constant(DoubleTensor::zeros(&[2]));
    let scanned = b.scan(zero, &[xs], &[], |_, sum, x, _| sum + x[0]);
    let cumsum = xs.cumsum(0);
    assert_eq!(scanned.shape(), vec![4, 2]);
    let (scanned, cumsum) = (scanned.id(), cumsum.id());
    let (graph, state) = evaluated(b);
    assert!(matches!(graph[scanned].op(), Op::Scan(_)));
    assert_eq!(graph[scanned].op().name(), "scan");
    assert_eq!(state.f64(scanned), state.f64(cumsum));
}

/// The AR(1) model x_t = φ x_{t-1} + ε_t with latent φ and ε, observed through a Gaussian likelihood, built with a
/// scan or unrolled. Returns the builder's (φ, ε, x) ids.
fn ar1(b: &GraphBuilder, scan: bool, length: usize) -> (NodeId, NodeId, NodeId) {
    let phi = b.gaussian(0.0, 1.0).label("phi");
    let eps = b.gaussian_with_shape(&[length], 0.0, 1.0).label("eps");
    b.set_value(phi, DoubleTensor::scalar(0.7));
    b.set_value(
        eps,
        DoubleTensor::vector((0..length).map(|i| (i as f64 * 0.9).sin()).collect()),
    );
    let x0 = b.scalar(0.3);
    let x = if scan {
        b.scan(x0, &[eps], &[phi], |_, x, eps, c| c[0] * x + eps[0])
    } else {
        let mut x = x0;
        let mut items = Vec::new();
        for i in 0..length {
            x = phi * x + eps.take(&[i]);
            items.push(x);
        }
        b.stack(0, &items)
    };
    let data: Vec<f64> = (0..length).map(|i| (i as f64 * 0.4).cos()).collect();
    let y = b.gaussian(x, 0.5);
    b.observe(y, DoubleTensor::vector(data));
    (phi.id(), eps.id(), x.id())
}

#[test]
fn ar1_scan_matches_the_unrolled_model_exactly() {
    let build = |scan| {
        let b = GraphBuilder::new();
        let ids = ar1(&b, scan, 6);
        let (graph, state) = evaluated(b);
        let model = Model::new(graph, &state).unwrap();
        (model, state, ids)
    };
    let (scan_model, scan_state, (phi_s, eps_s, x_s)) = build(true);
    let (unrolled, unrolled_state, (phi_u, eps_u, x_u)) = build(false);
    assert_eq!(scan_state.f64(x_s), unrolled_state.f64(x_u));
    assert_eq!(
        scan_model.log_prob(&scan_state).unwrap(),
        unrolled.log_prob(&unrolled_state).unwrap()
    );
    let gs = scan_model.log_prob_gradient(&scan_state).unwrap();
    let gu = unrolled.log_prob_gradient(&unrolled_state).unwrap();
    assert_eq!(gs.get(phi_s), gu.get(phi_u));
    assert_eq!(gs.get(eps_s), gu.get(eps_u));
}

/// A nonlinear scan over a [2] carry with two per-step inputs and two constants, and a scalar of its output.
struct Nonlinear {
    init: NodeId,
    xs: NodeId,
    ys: NodeId,
    scale: NodeId,
    shift: NodeId,
    out: NodeId,
    f: NodeId,
}

fn nonlinear() -> (Graph, State, Nonlinear) {
    let b = GraphBuilder::new();
    let init = b.constant(t(&[2], &[0.3, -0.4]));
    let xs = b.constant(t(&[3, 2], &[0.5, -0.1, 0.2, 0.9, -0.6, 0.4]));
    let ys = b.constant(t(&[3], &[1.1, 0.7, -0.3]));
    let scale = b.constant(t(&[2, 2], &[0.8, -0.3, 0.2, 0.6]));
    let shift = b.scalar(0.25);
    let out = b.scan(init, &[xs, ys], &[scale, shift], |body, h, step, c| {
        // h' = tanh(W h + x_t) * y_t + shift * sin(h)
        let wh = c[0].matmul(h.reshape(&[2, 1])).reshape(&[2]);
        let _ = body;
        (wh + step[0]).tanh() * step[1] + c[1] * h.sin()
    });
    let f = (out.sin() * out).sum();
    let ids = Nonlinear {
        init: init.id(),
        xs: xs.id(),
        ys: ys.id(),
        scale: scale.id(),
        shift: shift.id(),
        out: out.id(),
        f: f.id(),
    };
    let (graph, state) = evaluated(b);
    (graph, state, ids)
}

fn value_of(graph: &Graph, state: &State, of: NodeId) -> f64 {
    let mut state = state.clone();
    state.eval(graph, &[of], &mut rng()).unwrap();
    state.f64(of).unwrap().as_slice()[0]
}

#[test]
fn gradients_match_finite_differences_and_forward_mode() {
    let (graph, state, n) = nonlinear();
    let wrt = [n.init, n.xs, n.ys, n.scale, n.shift];
    let reverse = reverse_partials(&graph, &state, n.f, &wrt).unwrap();
    for &w in &wrt {
        let gradient = reverse.get(w).unwrap();
        let x = state.f64(w).unwrap().clone();
        assert_eq!(gradient.shape(), x.shape());
        for i in 0..x.len() {
            let h = 1e-6;
            let at = |delta: f64| {
                let mut data = x.as_slice().to_vec();
                data[i] += delta;
                let mut moved = state.clone();
                moved
                    .set(&graph, w, DoubleTensor::new(x.shape(), data).unwrap())
                    .unwrap();
                value_of(&graph, &moved, n.f)
            };
            let fd = (at(h) - at(-h)) / (2.0 * h);
            let exact = gradient.as_slice()[i];
            assert!(
                (exact - fd).abs() < 1e-7 * fd.abs().max(1.0),
                "∂f/∂{w:?}[{i}] = {exact}, finite differences give {fd}"
            );
        }
        // forward mode agrees with reverse mode
        let forward = forward_partials(&graph, &state, w, &[n.f]).unwrap();
        let forward = forward.get(n.f).unwrap();
        for (a, b) in forward.as_slice().iter().zip(gradient.as_slice()) {
            assert!((a - b).abs() < 1e-12 * a.abs().max(1.0), "{a} vs {b}");
        }
    }
    // the full Jacobian of the stacked output in both modes
    let reverse = reverse_partials(&graph, &state, n.out, &[n.xs]).unwrap();
    let forward = forward_partials(&graph, &state, n.xs, &[n.out]).unwrap();
    let (r, f) = (reverse.get(n.xs).unwrap(), forward.get(n.out).unwrap());
    assert_eq!(r.shape(), &[3, 2, 3, 2]);
    assert_eq!(r.shape(), f.shape());
    for (a, b) in r.as_slice().iter().zip(f.as_slice()) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
    // a step's output doesn't depend on later inputs
    assert_eq!(r.as_slice()[2], 0.0);
}

/// The nonlinear scan with latent init, per-step inputs and constants, observed through a Gaussian likelihood.
fn nonlinear_model(length: usize) -> (Model, State) {
    let b = GraphBuilder::new();
    let latent = |shape: &[usize], data: Vec<f64>| {
        let x = b.gaussian_with_shape(shape, 0.0, 1.0);
        b.set_value(x, DoubleTensor::new(shape, data).unwrap());
        x
    };
    let init = latent(&[2], vec![0.3, -0.4]);
    let xs = latent(
        &[length, 2],
        (0..2 * length).map(|i| (i as f64 * 0.7).sin()).collect(),
    );
    let scale = latent(&[2, 2], vec![0.8, -0.3, 0.2, 0.6]);
    let shift = latent(&[], vec![0.25]);
    let out = b.scan(init, &[xs], &[scale, shift], |_, h, x, c| {
        let wh = c[0].matmul(h.reshape(&[2, 1])).reshape(&[2]);
        (wh + x[0]).tanh() + c[1] * h.sin() * h
    });
    let y = b.gaussian(out, 0.8);
    b.observe(
        y,
        DoubleTensor::new(
            &[length, 2],
            (0..2 * length).map(|i| (i as f64 * 0.3).cos()).collect(),
        )
        .unwrap(),
    );
    let (graph, state) = evaluated(b);
    let model = Model::new(graph, &state).unwrap();
    (model, state)
}

#[test]
fn hessian_vector_products_match_finite_differences_of_the_gradient() {
    let (model, mut state) = nonlinear_model(4);
    let param = Parameterisation::new(&model, &state).unwrap();
    let z = param.from_state(&state).unwrap();
    let n = z.len();
    assert_eq!(n, 2 + 8 + 4 + 1);
    let gradient = |z: &[f64], state: &mut State| {
        param
            .log_prob_and_gradient(z, false, state)
            .unwrap()
            .1
            .unwrap()
    };
    for k in 0..3 {
        let v: Vec<f64> = (0..n)
            .map(|i| ((i * (k + 2)) as f64 * 0.37 + 0.1).sin())
            .collect();
        let hv = param.hvp(&z, &v, false, &mut state).unwrap();
        let h = 1e-5;
        let plus: Vec<f64> = z.iter().zip(&v).map(|(z, v)| z + h * v).collect();
        let minus: Vec<f64> = z.iter().zip(&v).map(|(z, v)| z - h * v).collect();
        let (gp, gm) = (gradient(&plus, &mut state), gradient(&minus, &mut state));
        for i in 0..n {
            let fd = (gp[i] - gm[i]) / (2.0 * h);
            assert!(
                (hv[i] - fd).abs() < 1e-6 * fd.abs().max(1.0),
                "(Hv)[{i}] = {}, finite differences give {fd}",
                hv[i]
            );
        }
    }
    // the full Hessian is symmetric
    let hessian = param.hessian(&z, false, &mut state).unwrap();
    let h = hessian.as_slice();
    for i in 0..n {
        for j in 0..i {
            assert!((h[i * n + j] - h[j * n + i]).abs() < 1e-10);
        }
    }
}

#[test]
fn second_order_ar1_scan_matches_the_unrolled_model() {
    let hessian = |scan| {
        let b = GraphBuilder::new();
        ar1(&b, scan, 5);
        let (graph, mut state) = evaluated(b);
        let model = Model::new(graph, &state).unwrap();
        let param = Parameterisation::new(&model, &state).unwrap();
        let z = param.from_state(&state).unwrap();
        param.hessian(&z, false, &mut state).unwrap()
    };
    let (s, u) = (hessian(true), hessian(false));
    for (a, b) in s.as_slice().iter().zip(u.as_slice()) {
        assert!((a - b).abs() < 1e-12 * a.abs().max(1.0), "{a} vs {b}");
    }
}

#[test]
fn zero_steps_give_an_empty_output() {
    let b = GraphBuilder::new();
    let init = b.gaussian_with_shape(&[3], 0.0, 1.0);
    let empty = b.scan_steps(0, init, &[], |_, x, _| x * 2.0);
    let xs = b.constant(DoubleTensor::zeros(&[0, 3]));
    let also_empty = b.scan(init, &[xs], &[], |_, x, step, _| x + step[0]);
    assert_eq!(empty.shape(), vec![0, 3]);
    let f = empty.sum() + also_empty.sum() + init.sum();
    let (init, empty, f) = (init.id(), empty.id(), f.id());
    let (graph, state) = evaluated(b);
    assert_eq!(state.f64(empty).unwrap().shape(), &[0, 3]);
    let gradient = reverse_partials(&graph, &state, f, &[init]).unwrap();
    assert_eq!(gradient.get(init).unwrap().as_slice(), &[1.0; 3]);
}

#[test]
fn ten_thousand_steps_evaluate_and_differentiate_without_overflowing_the_stack() {
    let length = 10_000;
    let b = GraphBuilder::new();
    let (phi, eps, x) = ar1(&b, true, length);
    let (graph, state) = evaluated(b);
    assert_eq!(state.f64(x).unwrap().shape(), &[length]);
    let model = Model::new(graph, &state).unwrap();
    let gradient = model.log_prob_gradient(&state).unwrap();
    assert!(gradient.get(phi).unwrap().as_slice()[0].is_finite());
    assert_eq!(gradient.get(eps).unwrap().len(), length);
    let param = Parameterisation::new(&model, &state).unwrap();
    let z = param.from_state(&state).unwrap();
    let v = vec![1.0; z.len()];
    let mut state = state;
    let hv = param.hvp(&z, &v, false, &mut state).unwrap();
    assert!(hv.iter().all(|h| h.is_finite()));
}

#[test]
fn bodies_with_a_distribution_are_rejected() {
    let b = GraphBuilder::new();
    b.scan_steps(3, b.scalar(0.0), &[], |body, x, _| {
        x + body.gaussian(0.0, 1.0)
    });
    assert_scan_error(b.error(), "probabilistic");
    assert!(b.build().is_err());
}

#[test]
fn bodies_mixing_in_an_outer_node_are_rejected() {
    // using an outer handle in the body doesn't compile (see GraphBuilder::scan's compile_fail example); mixing one
    // through a handle of the wrong builder is caught when building
    let b = GraphBuilder::new();
    let outer = (0..5).map(|i| b.scalar(i as f64)).last().unwrap();
    let outer_id = outer.id();
    b.scan_steps(3, b.scalar(1.0), &[], |body, x, _| {
        x * body.double_var(outer_id)
    });
    // the body only has the carry placeholder, so the id is unknown there (ids are per builder: an id that happens to
    // be one of the body's own refers to that body node)
    assert_scan_error(b.error(), "not a node");

    let b = GraphBuilder::new();
    let outer = b.scalar(2.0);
    b.scan_steps(3, b.scalar(1.0), &[], |body, x, _| {
        // a handle of the outer builder, smuggled in with a shorter lifetime, is from the wrong builder
        let outer: DoubleVar<'_> = smuggle(outer, body);
        x * outer
    });
    assert_scan_error(b.error(), "different GraphBuilder");
}

/// Re-wraps an outer handle's id as a handle of the body builder by way of a node mixing the two, without any
/// unsafe code: the body records the wrong-builder error when the outer handle is used in its operation.
fn smuggle<'s>(outer: DoubleVar<'_>, body: &'s GraphBuilder) -> DoubleVar<'s> {
    let carry_like = body.scalar(1.0);
    let mixed = keanu_graph::Var::id(&mix(carry_like, outer));
    body.double_var(mixed)
}

fn mix<'a>(a: DoubleVar<'a>, b: DoubleVar<'a>) -> DoubleVar<'a> {
    a * b
}

#[test]
fn invalid_bodies_and_inputs_are_rejected() {
    // the output must have the carry's shape
    let b = GraphBuilder::new();
    b.scan_steps(2, b.constant(DoubleTensor::zeros(&[2])), &[], |_, x, _| {
        x.sum()
    });
    assert_scan_error(b.error(), "carry");
    // per-step inputs must agree on the number of steps
    let b = GraphBuilder::new();
    let (xs, ys) = (
        b.constant(DoubleTensor::zeros(&[3])),
        b.constant(DoubleTensor::zeros(&[4])),
    );
    b.scan(b.scalar(0.0), &[xs, ys], &[], |_, x, s, _| x + s[0] + s[1]);
    assert_scan_error(b.error(), "per-step input 1");
    // scan needs per-step inputs of rank >= 1
    let b = GraphBuilder::new();
    b.scan(b.scalar(0.0), &[], &[], |_, x, _, _| x);
    assert_scan_error(b.error(), "scan_steps");
    let b = GraphBuilder::new();
    b.scan(b.scalar(0.0), &[b.scalar(1.0)], &[], |_, x, _, _| x);
    assert_scan_error(b.error(), "dimension");
    // an observation in the body
    let b = GraphBuilder::new();
    b.scan_steps(2, b.scalar(0.0), &[], |body, x, _| {
        let flag = body.bool_scalar(true);
        body.observe(flag, keanu_tensor::BoolTensor::scalar(true));
        x
    });
    assert_scan_error(b.error(), "observed");
    // an error inside the body is reported as a scan error
    let b = GraphBuilder::new();
    b.scan_steps(2, b.constant(DoubleTensor::zeros(&[2])), &[], |_, x, _| {
        x + x.reshape(&[3])
    });
    assert_scan_error(b.error(), "in the body");
    // ScanBody::new checks its placeholders
    let body = GraphBuilder::new();
    let carry = body.double_proxy(&[]);
    let c = body.scalar(1.0);
    let out = carry + c;
    let (carry, c, out) = (carry.id(), c.id(), out.id());
    let (graph, _) = body.build().unwrap();
    assert!(ScanBody::new(graph.clone(), 2, carry, vec![], vec![], out).is_ok());
    assert!(ScanBody::new(graph.clone(), 2, carry, vec![], vec![c], out).is_err());
    assert!(ScanBody::new(graph.clone(), 2, carry, vec![carry], vec![], out).is_err());
    assert!(ScanBody::new(graph, 2, carry, vec![], vec![], NodeId::from_index(99)).is_err());
}

#[test]
fn several_carries_pack_into_one_tensor_and_scans_nest() {
    // Fibonacci: carry [a, b] -> [b, a + b]
    let b = GraphBuilder::new();
    let fib = b.scan_steps(6, b.constant(t(&[2], &[0.0, 1.0])), &[], |body, c, _| {
        let (x, y) = (c.slice("0"), c.slice("1"));
        body.stack(0, &[y, x + y])
    });
    // a nested scan: each step adds the cumulative sum of a constant vector's elements (computed by an inner scan)
    let v = b.constant(t(&[3], &[1.0, 2.0, 3.0]));
    let nested = b.scan_steps(2, b.scalar(0.0), &[v], |body, x, c| {
        let inner = body.scan(body.scalar(0.0), &[c[0]], &[], |_, s, e, _| s + e[0]);
        x + inner.sum()
    });
    let (fib, nested) = (fib.id(), nested.id());
    let (_, state) = evaluated(b);
    assert_eq!(
        state.f64(fib).unwrap().as_slice(),
        &[1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 5.0, 5.0, 8.0, 8.0, 13.0]
    );
    // inner cumsum [1, 3, 6] sums to 10
    assert_eq!(state.f64(nested).unwrap().as_slice(), &[10.0, 20.0]);
}
