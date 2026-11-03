#![allow(clippy::cloned_ref_to_slice_refs)] // clones keep the test tables readable
//! Finite-difference checks of forward and reverse mode auto diff, plus traversal rules.

use keanu_distributions::KeanuRng;
use keanu_graph::{
    DType, DoubleVar, Graph, GraphBuilder, GraphError, NodeId, State, Value, forward_partials,
    reverse_partials, reverse_partials_from,
};
use keanu_tensor::DoubleTensor;

/// Builds a graph from input values (as latents) and a function of them, then checks that the forward and reverse
/// partials of the output with respect to every input match central finite differences.
fn check(
    inputs: &[DoubleTensor],
    build: impl for<'b> Fn(&'b GraphBuilder, &[DoubleVar<'b>]) -> DoubleVar<'b>,
) {
    check_with_tolerance(inputs, 1e-6, build)
}

fn check_with_tolerance(
    inputs: &[DoubleTensor],
    tolerance: f64,
    build: impl for<'b> Fn(&'b GraphBuilder, &[DoubleVar<'b>]) -> DoubleVar<'b>,
) {
    let g = GraphBuilder::new();
    let vars: Vec<DoubleVar<'_>> = inputs
        .iter()
        .map(|v| {
            let latent = g.uniform_with_shape(v.shape(), -1000.0, 1000.0);
            g.set_value(latent, v.clone());
            latent
        })
        .collect();
    let output = build(&g, &vars).id();
    let ids: Vec<NodeId> = vars.iter().map(|v| v.id()).collect();
    let (graph, mut state) = g.build().expect("graph builds");
    state
        .lazy_eval(&graph, &[output], &mut KeanuRng::seed_from_u64(1))
        .unwrap();

    for (k, (&input, value)) in ids.iter().zip(inputs).enumerate() {
        let numeric = finite_difference(&graph, &state, input, value, output);
        let forward = forward_partials(&graph, &state, input, &[output])
            .unwrap()
            .get(output)
            .unwrap()
            .clone();
        let reverse = reverse_partials(&graph, &state, output, &[input])
            .unwrap()
            .get(input)
            .unwrap()
            .clone();
        for (mode, partial) in [("forward", &forward), ("reverse", &reverse)] {
            assert_eq!(
                partial.shape(),
                numeric.shape(),
                "{mode} partial wrt input {k} has the wrong shape"
            );
            for (i, (&a, &e)) in partial
                .as_slice()
                .iter()
                .zip(numeric.as_slice())
                .enumerate()
            {
                assert!(
                    (a - e).abs() <= tolerance * e.abs().max(1.0),
                    "{mode} partial wrt input {k}, element {i}: {a} != finite difference {e}"
                );
            }
        }
    }
}

/// ∂output/∂input shaped [output shape..., input shape...] by central differences.
fn finite_difference(
    graph: &Graph,
    state: &State,
    input: NodeId,
    value: &DoubleTensor,
    output: NodeId,
) -> DoubleTensor {
    let h = 1e-6;
    let out_shape = graph.node(output).shape().to_vec();
    let n_out: usize = out_shape.iter().product();
    let n_in = value.len();
    let mut data = vec![0.0; n_out * n_in];
    for j in 0..n_in {
        let at = |delta: f64| -> Vec<f64> {
            let mut s = state.clone();
            let mut v = value.as_slice().to_vec();
            v[j] += delta;
            s.set_and_cascade(graph, input, DoubleTensor::new(value.shape(), v).unwrap())
                .unwrap();
            s.f64(output).unwrap().as_slice().to_vec()
        };
        let (up, down) = (at(h), at(-h));
        for i in 0..n_out {
            data[i * n_in + j] = (up[i] - down[i]) / (2.0 * h);
        }
    }
    let shape: Vec<usize> = out_shape.iter().chain(value.shape()).copied().collect();
    DoubleTensor::new(&shape, data).unwrap()
}

fn t(shape: &[usize], data: &[f64]) -> DoubleTensor {
    DoubleTensor::new(shape, data.to_vec()).unwrap()
}

fn range(low: f64, high: f64, shape: &[usize]) -> DoubleTensor {
    let n: usize = shape.iter().product();
    t(
        shape,
        &(0..n)
            .map(|i| low + (high - low) * (i + 1) as f64 / (n + 1) as f64)
            .collect::<Vec<_>>(),
    )
}

fn spd(n: usize) -> DoubleTensor {
    let a = DoubleTensor::arange(1.0, (n * n + 1) as f64)
        .reshape(&[n as isize, n as isize])
        .unwrap()
        .sin();
    a.matmul_transposed(&a, false, true)
        .unwrap()
        .add(&(DoubleTensor::eye(n) * n as f64))
        .unwrap()
}

#[test]
fn composite_graphs_from_the_golden_vectors() {
    // cases where Java's answers were wrong before the Java fixes in this phase; finite differences decide
    check(
        &[range(0.5, 1.5, &[3]), DoubleTensor::scalar(0.7)],
        |_, v| {
            let shared = (v[0] * v[1]).sin();
            (shared * shared + shared.exp()).sum()
        },
    );
    check(&[range(-1.0, 1.0, &[3]), spd(3)], |_, v| {
        let x = v[0].reshape(&[3, 1]);
        x.transpose().matmul(v[1].inverse()).matmul(x).sum()
    });
    check(
        &[DoubleTensor::scalar(1.25), range(0.5, 1.5, &[2, 2])],
        |_, v| v[0].maximum(v[1]),
    );
    // the same node as both operands
    check(&[range(0.5, 1.5, &[3])], |_, v| {
        (v[0] * v[0] + v[0] / v[0].exp()).sum()
    });
}

#[test]
fn unary_ops() {
    let positive = range(0.2, 2.5, &[2, 3]);
    let unit = range(-0.8, 0.8, &[2, 3]);
    let signed = t(&[2, 3], &[-1.7, -0.3, 0.4, 1.1, 2.2, -2.5]);
    check(&[signed.clone()], |_, v| v[0].abs());
    check(&[signed.clone()], |_, v| -v[0]);
    check(&[positive.clone()], |_, v| v[0].reciprocal());
    check(&[signed.clone()], |_, v| v[0].exp());
    check(&[signed.clone()], |_, v| v[0].exp2());
    check(&[signed.clone()], |_, v| v[0].exp_m1());
    check(&[positive.clone()], |_, v| v[0].ln());
    check(&[positive.clone()], |_, v| v[0].ln_1p());
    check(&[positive.clone()], |_, v| v[0].log2());
    check(&[positive.clone()], |_, v| v[0].log10());
    check(&[positive.clone()], |_, v| v[0].sqrt());
    check(&[signed.clone()], |_, v| v[0].sin());
    check(&[signed.clone()], |_, v| v[0].cos());
    check(&[unit.clone()], |_, v| v[0].tan());
    check(&[unit.clone()], |_, v| v[0].asin());
    check(&[unit.clone()], |_, v| v[0].acos());
    check(&[signed.clone()], |_, v| v[0].atan());
    check(&[signed.clone()], |_, v| v[0].sinh());
    check(&[signed.clone()], |_, v| v[0].cosh());
    check(&[signed.clone()], |_, v| v[0].tanh());
    check(&[signed.clone()], |_, v| v[0].asinh());
    check(&[range(1.2, 3.0, &[2, 3])], |_, v| v[0].acosh());
    check(&[unit.clone()], |_, v| v[0].atanh());
    check(&[signed.clone()], |_, v| v[0].sigmoid());
    check(&[positive.clone()], |_, v| v[0].ln_gamma());
    check(&[positive.clone()], |_, v| v[0].digamma());
    check_with_tolerance(&[positive], 1e-5, |_, v| v[0].trigamma());
}

#[test]
fn binary_ops_with_broadcasting() {
    let shapes: [(&[usize], &[usize]); 4] = [
        (&[2, 3], &[2, 3]),
        (&[2, 3], &[3]),
        (&[], &[2, 2]),
        (&[2, 1], &[1, 3]),
    ];
    for (ls, rs) in shapes {
        let l = range(0.3, 2.0, ls);
        let r = range(0.4, 1.7, rs).map(|x| 2.1 - x);
        check(&[l.clone(), r.clone()], |_, v| v[0] + v[1]);
        check(&[l.clone(), r.clone()], |_, v| v[0] - v[1]);
        check(&[l.clone(), r.clone()], |_, v| v[0] * v[1]);
        check(&[l.clone(), r.clone()], |_, v| v[0] / v[1]);
        check(&[l.clone(), r.clone()], |_, v| v[0].pow(v[1]));
        check(&[l.clone(), r.clone()], |_, v| v[0].atan2(v[1]));
        check(&[l.clone(), r.clone()], |_, v| v[0].log_add_exp(v[1]));
        check(&[l.clone(), r.clone()], |_, v| v[0].log_add_exp2(v[1]));
        check(&[l.clone(), r.clone()], |_, v| v[0].safe_ln_times(v[1]));
        check(&[l.clone(), r.clone() * 1.37], |_, v| v[0].maximum(v[1]));
        check(&[l.clone(), r.clone() * 1.37], |_, v| v[0].minimum(v[1]));
    }
    // scalar operands on either side
    check(&[range(0.3, 2.0, &[3])], |_, v| v[0] * 3.0 - 1.5);
    check(&[range(0.3, 2.0, &[3])], |_, v| 2.0 / v[0]);
}

#[test]
fn reductions_and_scans() {
    let cube = range(-1.5, 2.5, &[2, 3, 4]);
    check(&[cube.clone()], |_, v| v[0].sum());
    check(&[cube.clone()], |_, v| v[0].sum_axes(&[0, 2]));
    check(&[cube.clone()], |_, v| v[0].sum_axes(&[-1]));
    check(&[cube.clone()], |_, v| v[0].mean());
    check(&[cube.clone()], |_, v| v[0].mean_axes(&[1]));
    check(&[range(0.5, 1.5, &[2, 3])], |_, v| v[0].product());
    check(&[range(0.5, 1.5, &[2, 3])], |_, v| v[0].product_axes(&[1]));
    // products and cumulative products are exact where elements are zero
    check(&[t(&[2, 2], &[1.5, 0.0, -2.0, 3.0])], |_, v| v[0].product());
    check(&[t(&[4], &[1.5, 0.0, -2.0, 3.0])], |_, v| v[0].cumprod(0));
    check(&[cube.clone()], |_, v| v[0].cumsum(1));
    check(&[range(-1.5, 2.5, &[2, 3])], |_, v| v[0].cumprod(-1));
    check(&[cube.clone()], |_, v| v[0].max());
    check(&[cube.clone()], |_, v| v[0].min());
    check(&[cube.clone()], |_, v| v[0].std_dev());
    check(&[range(-1.0, 3.0, &[2, 3])], |_, v| v[0].standardize());
}

#[test]
fn shape_ops() {
    let cube = range(-1.5, 2.5, &[2, 3, 4]);
    check(&[cube.clone()], |_, v| v[0].reshape(&[4, 6]));
    check(&[cube.clone()], |_, v| v[0].reshape(&[-1]));
    check(&[cube.clone()], |_, v| v[0].permute(&[2, 0, 1]));
    check(&[range(-1.0, 1.0, &[2, 3])], |_, v| v[0].transpose());
    check(&[cube.clone()], |_, v| v[0].slice("1, ::-1, 1:3"));
    check(&[cube.clone()], |_, v| v[0].slice("..., ::2"));
    check(&[range(-1.0, 1.0, &[3])], |_, v| v[0].broadcast_to(&[2, 3]));
    check(&[range(-1.0, 1.0, &[2, 1])], |_, v| {
        v[0].broadcast_to(&[2, 3])
    });
    check(&[cube.clone()], |_, v| v[0].take(&[1, 2, 3]));
    check(&[range(-1.0, 1.0, &[3])], |_, v| v[0].diag());
    check(&[range(-1.0, 1.0, &[2, 3, 3])], |_, v| v[0].diag_part());
    check(&[range(-1.0, 1.0, &[2, 3])], |_, v| v[0].diag_part());
    check(&[range(-1.0, 1.0, &[3, 4])], |_, v| v[0].tril(0));
    check(&[range(-1.0, 1.0, &[3, 4])], |_, v| v[0].tril(-1));
    check(&[range(-1.0, 1.0, &[3, 4])], |_, v| v[0].triu(1));
    check(&[range(-1.0, 1.0, &[6])], |_, v| {
        v[0].fill_triangular(true, true)
    });
    check(&[range(-1.0, 1.0, &[6])], |_, v| {
        v[0].fill_triangular(false, true)
    });
    check(&[range(-1.0, 1.0, &[3, 3])], |_, v| {
        v[0].triangle_part(true)
    });
    check(&[range(-1.0, 1.0, &[3, 3])], |_, v| {
        v[0].triangle_part(false)
    });
    check(
        &[range(-1.0, 1.0, &[2, 3]), range(0.0, 2.0, &[1, 3])],
        |g, v| g.concat(0, &[v[0], v[1], v[0]]),
    );
    check(
        &[range(-1.0, 1.0, &[2, 3]), range(0.0, 2.0, &[2, 1])],
        |g, v| g.concat(-1, &[v[0], v[1]]),
    );
}

#[test]
fn linear_algebra() {
    check(
        &[range(-1.0, 1.0, &[2, 3]), range(-1.0, 2.0, &[3, 4])],
        |_, v| v[0].matmul(v[1]),
    );
    check(
        &[range(-1.0, 1.0, &[2, 2, 3]), range(-1.0, 2.0, &[3, 2])],
        |_, v| v[0].matmul(v[1]),
    );
    check(
        &[range(-1.0, 1.0, &[2, 3]), range(-1.0, 2.0, &[2, 2, 3, 2])],
        |_, v| v[0].matmul(v[1]),
    );
    check(
        &[range(-1.0, 1.0, &[3, 2]), range(-1.0, 2.0, &[4, 3])],
        |_, v| v[0].matmul_transposed(v[1], true, true),
    );
    check(
        &[range(-1.0, 1.0, &[3, 2]), range(-1.0, 2.0, &[3, 4])],
        |_, v| v[0].matmul_transposed(v[1], true, false),
    );
    let a = range(-1.5, 2.5, &[2, 3, 4]);
    check(&[a.clone(), range(-1.0, 2.0, &[4, 3])], |_, v| {
        v[0].tensor_multiply(v[1], &[1, 2], &[1, 0])
    });
    check(&[a.clone(), range(-1.0, 2.0, &[4, 2])], |_, v| {
        v[0].tensor_multiply(v[1], &[-1], &[0])
    });
    check(&[a, range(-1.0, 2.0, &[3, 2])], |_, v| {
        v[0].tensor_multiply(v[1], &[1], &[0])
    });
    check(&[spd(3)], |_, v| v[0].inverse());
    check(
        &[DoubleTensor::concat(
            0,
            &[
                &spd(3).reshape(&[1, 3, 3]).unwrap(),
                &(spd(3) * 0.5).reshape(&[1, 3, 3]).unwrap(),
            ],
        )
        .unwrap()],
        |_, v| v[0].inverse(),
    );
    check(&[spd(3)], |_, v| v[0].determinant());
    check(
        &[range(-1.0, 1.5, &[2, 2, 2])
            .add(&(DoubleTensor::eye(2) * 2.0))
            .unwrap()],
        |_, v| v[0].determinant(),
    );
    // Cholesky reads only the lower triangle, so perturbing the upper triangle changes nothing
    check(&[spd(4)], |_, v| v[0].cholesky());
    check(&[range(-0.3, 0.4, &[3, 3])], |_, v| {
        let x = v[0];
        (x.matmul_transposed(x, false, true) + DoubleTensor::eye(3) * 3.0).cholesky()
    });
    check(&[spd(3).cholesky().unwrap()], |_, v| {
        v[0].cholesky_inverse()
    });
    check(&[spd(3)], |_, v| v[0].cholesky().cholesky_inverse());
}

#[test]
fn selection_proxies_and_lambdas() {
    check(
        &[
            t(&[2, 2], &[-1.0, 0.5, 2.0, -0.2]),
            range(0.5, 2.0, &[2, 2]),
        ],
        |g, v| g.where_(v[0].gt(g.scalar(0.0)), v[0] * v[1], v[1].exp()),
    );
    // broadcasting branches
    check(
        &[DoubleTensor::scalar(0.7), range(0.5, 2.0, &[3])],
        |g, v| g.where_(v[1].gt(g.scalar(1.0)), v[0], v[1]),
    );
    check(
        &[DoubleTensor::scalar(0.7), DoubleTensor::scalar(1.3)],
        |g, v| {
            let condition = g.bool_scalar(true);
            g.cpt(&[condition.id()], &[(vec![1], v[0] * 2.0)], v[1])
        },
    );
    check(
        &[DoubleTensor::scalar(0.7), DoubleTensor::scalar(1.3)],
        |g, v| {
            let condition = g.bool_scalar(false);
            g.cpt(&[condition.id()], &[(vec![1], v[0] * 2.0)], v[1].exp())
        },
    );
    check(&[range(0.5, 2.0, &[3])], |g, v| {
        let target = v[0] * 2.0;
        let proxy = g.double_proxy(&[3]);
        g.connect(proxy, target);
        proxy.sin()
    });
    check(
        &[range(0.5, 2.0, &[3]), DoubleTensor::scalar(1.5)],
        |g, v| {
            let id = g.lambda_with_jacobian(
                "scaled_square",
                &[v[0].id(), v[1].id()],
                &[3],
                |values| {
                    let (x, a) = (
                        values[0].as_f64().unwrap(),
                        values[1].as_f64().unwrap().as_slice()[0],
                    );
                    Ok(Value::F64(x.map(|x| a * x * x)))
                },
                |values| {
                    let (x, a) = (
                        values[0].as_f64().unwrap(),
                        values[1].as_f64().unwrap().as_slice()[0],
                    );
                    let dx = x.map(|x| 2.0 * a * x).diag().map_err(|e| e.to_string())?;
                    let da = x.map(|x| x * x);
                    Ok(vec![Some(dx), Some(da)])
                },
            );
            g.double_var(id) + 0.0
        },
    );
}

fn build_one<'b>(g: &'b GraphBuilder, value: DoubleTensor) -> DoubleVar<'b> {
    let latent = g.uniform_with_shape(value.shape(), -1000.0, 1000.0);
    g.set_value(latent, value);
    latent
}

fn evaluated(g: GraphBuilder, outputs: &[NodeId]) -> (Graph, State) {
    let (graph, mut state) = g.build().unwrap();
    state
        .lazy_eval(&graph, outputs, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    (graph, state)
}

#[test]
fn non_differentiable_ops_have_no_partials() {
    let g = GraphBuilder::new();
    let x = build_one(&g, t(&[3], &[-1.7, 0.3, 2.2]));
    let ops: Vec<NodeId> = vec![
        x.floor().id(),
        x.ceil().id(),
        x.round().id(),
        x.sign().id(),
        x.gt_mask(g.scalar(0.0)).id(),
        x.replace_nan(0.0).id(),
        (x.floor() * 2.0).id(),
        x.to_int().to_double().id(),
    ];
    let x = x.id();
    let (graph, state) = evaluated(g, &ops);
    for &op in &ops {
        let forward = forward_partials(&graph, &state, x, &[op]).unwrap();
        assert!(
            matches!(forward.get(op), Err(GraphError::NoPartial { .. })),
            "forward through {op:?}"
        );
        let reverse = reverse_partials(&graph, &state, op, &[x]).unwrap();
        assert!(
            matches!(reverse.get(x), Err(GraphError::NoPartial { .. })),
            "reverse through {op:?}"
        );
    }
}

#[test]
fn unrelated_nodes_have_no_partials_and_probabilistic_nodes_stop_gradients() {
    let g = GraphBuilder::new();
    let x = build_one(&g, DoubleTensor::scalar(0.5));
    let y = build_one(&g, DoubleTensor::scalar(1.5));
    let unrelated = (y * 2.0).id();
    // a probabilistic node in the middle: gradients do not flow through its value to its parameters
    let z = g.gaussian(x, 1.0);
    let after = (z * 3.0).id();
    let direct = (x * 3.0).id();
    let (xid, zid) = (x.id(), z.id());
    let (graph, state) = evaluated(g, &[unrelated, after, direct]);

    assert!(
        reverse_partials(&graph, &state, unrelated, &[xid])
            .unwrap()
            .get(xid)
            .is_err()
    );
    assert!(
        forward_partials(&graph, &state, xid, &[unrelated])
            .unwrap()
            .get(unrelated)
            .is_err()
    );
    assert!(
        reverse_partials(&graph, &state, after, &[xid])
            .unwrap()
            .get(xid)
            .is_err()
    );
    assert_eq!(
        reverse_partials(&graph, &state, after, &[zid])
            .unwrap()
            .get(zid)
            .unwrap()
            .as_slice(),
        &[3.0]
    );
    assert_eq!(
        reverse_partials(&graph, &state, direct, &[xid])
            .unwrap()
            .get(xid)
            .unwrap()
            .as_slice(),
        &[3.0]
    );
}

#[test]
fn observed_nodes() {
    let g = GraphBuilder::new();
    let x = build_one(&g, DoubleTensor::scalar(0.5));
    let z = g.gaussian(x, 1.0);
    g.observe(z, DoubleTensor::scalar(0.2));
    let (xid, zid) = (x.id(), z.id());
    let (graph, state) = evaluated(g, &[zid]);
    // reverse mode from an observed node gives nothing, as in Java
    assert!(
        reverse_partials(&graph, &state, zid, &[xid])
            .unwrap()
            .get(xid)
            .is_err()
    );
    // forward mode with respect to an observed node gives nothing
    assert!(
        forward_partials(&graph, &state, zid, &[zid])
            .unwrap()
            .get(zid)
            .is_err()
    );
}

#[test]
fn reverse_from_an_upstream_partial_is_the_chain_rule() {
    let g = GraphBuilder::new();
    let x = build_one(&g, range(0.5, 1.5, &[3]));
    let y = (x * x).sin();
    let total = y.sum();
    let (xid, yid, tid) = (x.id(), y.id(), total.id());
    let (graph, state) = evaluated(g, &[tid]);

    let upstream = DoubleTensor::ones(&[3]);
    let chained = reverse_partials_from(&graph, &state, yid, &upstream, &[xid])
        .unwrap()
        .get(xid)
        .unwrap()
        .clone();
    let direct = reverse_partials(&graph, &state, tid, &[xid])
        .unwrap()
        .get(xid)
        .unwrap()
        .clone();
    assert_eq!(chained, direct);

    // an outer shape is kept: [2, 3] upstream gives [2, 3] partials
    let upstream = DoubleTensor::new(&[2, 3], vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0]).unwrap();
    let p = reverse_partials_from(&graph, &state, yid, &upstream, &[xid])
        .unwrap()
        .get(xid)
        .unwrap()
        .clone();
    assert_eq!(p.shape(), &[2, 3]);
    assert!(reverse_partials_from(&graph, &state, yid, &DoubleTensor::ones(&[2]), &[xid]).is_err());
}

#[test]
fn integer_and_bool_nodes_are_not_differentiable() {
    let g = GraphBuilder::new();
    let i = g.int_scalar(3);
    let out = (i * 2).to_double();
    let (iid, oid) = (i.id(), out.id());
    let (graph, state) = evaluated(g, &[oid]);
    assert!(matches!(
        forward_partials(&graph, &state, iid, &[oid]),
        Err(GraphError::NotDifferentiable(_))
    ));
    let _ = DType::F64;
}

#[test]
fn values_must_be_evaluated_first() {
    let g = GraphBuilder::new();
    let x = build_one(&g, DoubleTensor::scalar(0.5));
    let y = x.exp();
    let (xid, yid) = (x.id(), y.id());
    let (graph, state) = g.build().unwrap();
    assert!(matches!(
        reverse_partials(&graph, &state, yid, &[xid]),
        Err(GraphError::MissingValue(_))
    ));
}
