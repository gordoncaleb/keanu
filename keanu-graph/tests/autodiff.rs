#![allow(clippy::cloned_ref_to_slice_refs)] // clones keep the test tables readable
//! Finite-difference checks of forward and reverse mode auto diff, plus traversal rules.

use keanu_distributions::KeanuRng;
use keanu_graph::{
    DType, DoubleVar, Graph, GraphBuilder, GraphError, NodeId, State, Value, forward_partials,
    reverse_partials, reverse_partials_from,
};
use keanu_tensor::DoubleTensor;

mod common;
use common::t;

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
    let out_shape = graph[output].shape().to_vec();
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
        .unwrap()
        .reshape(&[n as isize, n as isize])
        .unwrap()
        .sin();
    a.matmul_transposed(&a, false, true)
        .unwrap()
        .add(&(DoubleTensor::eye(n).unwrap() * n as f64))
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
fn special_function_ops() {
    let signed = t(&[2, 3], &[-1.7, -0.3, 0.4, 1.1, 2.2, -2.5]);
    let probability = t(&[2, 3], &[0.01, 0.2, 0.45, 0.5, 0.8, 0.97]);
    let unit = range(-0.9, 0.9, &[2, 3]);
    let negative = t(&[2, 3], &[-0.01, -0.3, -0.69, -1.0, -4.0, -20.0]);
    check(&[signed.clone()], |_, v| v[0].erf());
    check(&[signed.clone()], |_, v| v[0].erfc());
    check(&[unit], |_, v| v[0].erf_inv());
    check(&[signed.clone()], |_, v| v[0].log_ndtr());
    // deep in the lower tail, where Φ underflows but ln Φ and its derivative don't
    check(&[t(&[3], &[-40.0, -15.0, 6.0])], |_, v| v[0].log_ndtr());
    check(&[probability.clone()], |_, v| v[0].ndtri());
    check(&[t(&[2], &[1e-6, 0.999])], |_, v| v[0].ndtri());
    check(&[t(&[4], &[-40.0, -1.0, 2.0, 40.0])], |_, v| {
        v[0].softplus()
    });
    check(&[signed], |_, v| v[0].softplus());
    check(&[probability], |_, v| v[0].logit());
    check(&[negative], |_, v| v[0].log1mexp());
    // scalars take the scalar fast path
    let scalar = DoubleTensor::scalar;
    check(&[scalar(-0.4)], |_, v| v[0].erf());
    check(&[scalar(-0.4)], |_, v| v[0].erfc());
    check(&[scalar(0.6)], |_, v| v[0].erf_inv());
    check(&[scalar(-12.0)], |_, v| v[0].log_ndtr());
    check(&[scalar(0.3)], |_, v| v[0].ndtri());
    check(&[scalar(-0.4)], |_, v| v[0].softplus());
    check(&[scalar(0.3)], |_, v| v[0].logit());
    check(&[scalar(-0.4)], |_, v| v[0].log1mexp());
    let shapes: [(&[usize], &[usize]); 3] = [(&[2, 3], &[2, 3]), (&[2, 3], &[3]), (&[], &[2, 2])];
    for (ls, rs) in shapes {
        let a = range(0.3, 4.0, ls);
        let b = range(0.5, 25.0, rs);
        check(&[a.clone(), b.clone()], |_, v| v[0].log_beta(v[1]));
        check(&[a, b], |_, v| v[1].log_beta(v[0]));
    }
    check(&[DoubleTensor::scalar(2.5)], |_, v| v[0].log_beta(3.0));
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

/// A [2, 3, 4] cube with one zero in some lanes and two in others, along every axis.
fn cube_with_zeros() -> DoubleTensor {
    let mut data: Vec<f64> = range(-1.5, 2.5, &[2, 3, 4])
        .as_slice()
        .iter()
        .map(|&x| x + 0.05)
        .collect();
    // [0, 0, 1] alone in its lane along each axis; [1, 2, 0] and [1, 2, 3] share a lane along the last axis,
    // [1, 0, 2] and [1, 2, 2] share one along axis 1, [0, 1, 3] and [1, 1, 3] share one along axis 0
    for (i, j, k) in [
        (0, 0, 1),
        (1, 2, 0),
        (1, 2, 3),
        (1, 0, 2),
        (1, 2, 2),
        (0, 1, 3),
        (1, 1, 3),
    ] {
        data[i * 12 + j * 4 + k] = 0.0;
    }
    t(&[2, 3, 4], &data)
}

#[test]
fn products_and_scans_are_exact_around_zeros() {
    let zeros = cube_with_zeros();
    let one_zero = t(&[5], &[1.5, -0.5, 0.0, 2.0, 1.25]);
    let two_zeros = t(&[5], &[1.5, 0.0, -2.0, 0.0, 1.25]);
    let adjacent = t(&[4], &[0.0, 0.0, 3.0, -1.0]);
    for x in [&one_zero, &two_zeros, &adjacent] {
        check(&[x.clone()], |_, v| v[0].product());
        check(&[x.clone()], |_, v| v[0].product_axes(&[0]));
        check(&[x.clone()], |_, v| v[0].cumprod(0));
        check(&[x.clone()], |_, v| v[0].cumsum(-1));
    }
    check(&[zeros.clone()], |_, v| v[0].product());
    for axes in [
        &[0][..],
        &[1],
        &[2],
        &[-1],
        &[0, 2],
        &[2, 0],
        &[1, 2],
        &[0, 1, 2],
    ] {
        check(&[zeros.clone()], |_, v| v[0].product_axes(axes));
    }
    for axis in [0, 1, 2, -1, -3] {
        check(&[zeros.clone()], |_, v| v[0].cumprod(axis));
        check(&[zeros.clone()], |_, v| v[0].cumsum(axis));
    }
    // with every element zero
    check(&[DoubleTensor::zeros(&[2, 3])], |_, v| {
        v[0].product_axes(&[1])
    });
    check(&[DoubleTensor::zeros(&[2, 3])], |_, v| v[0].cumprod(1));
}

#[test]
fn products_and_scans_with_broadcasting_and_several_paths() {
    let zeros = cube_with_zeros();
    // the input reaches the op through a broadcast, so its partial sums over the broadcast copies
    check(&[t(&[4], &[1.5, 0.0, -2.0, 0.0])], |_, v| {
        v[0].broadcast_to(&[3, 4]).cumprod(1)
    });
    check(&[t(&[3, 1], &[1.5, 0.0, -2.0])], |_, v| {
        v[0].broadcast_to(&[3, 4]).product_axes(&[0])
    });
    // the op's result broadcast against another input, and the input used on more than one path
    check(&[zeros.clone(), range(0.5, 1.5, &[2, 3])], |_, v| {
        v[0].product_axes(&[2]) * v[1]
    });
    check(&[zeros.clone(), range(0.5, 1.5, &[4])], |_, v| {
        (v[0].cumprod(2) * v[1] + v[0].cumsum(0)).product_axes(&[1])
    });
    check(
        &[range(0.5, 1.5, &[3, 4]), range(-1.0, 1.0, &[4])],
        |_, v| (v[0] * v[1]).cumsum(0).cumprod(1).product(),
    );
}

#[test]
fn products_and_scans_of_scalars_and_empty_tensors() {
    check(&[DoubleTensor::scalar(1.75)], |_, v| v[0].product());
    check(&[DoubleTensor::scalar(0.0)], |_, v| v[0].product());
    check(&[range(0.5, 1.5, &[2, 3])], |_, v| v[0].product_axes(&[]));
    check(&[range(0.5, 1.5, &[1, 3])], |_, v| v[0].cumprod(0));
    check(&[DoubleTensor::zeros(&[0])], |_, v| v[0].product());
    check(&[DoubleTensor::zeros(&[2, 0])], |_, v| {
        v[0].product_axes(&[1])
    });
    check(&[DoubleTensor::zeros(&[2, 0])], |_, v| {
        v[0].product_axes(&[0])
    });
    check(&[DoubleTensor::zeros(&[0, 3])], |_, v| v[0].cumprod(0));
    check(&[DoubleTensor::zeros(&[0, 3])], |_, v| v[0].cumsum(1));
}

#[test]
fn tensor_multiply_contractions() {
    let a = range(-1.5, 2.5, &[2, 3, 4]);
    // contracting nothing (an outer product), everything (an inner product), and axes in a different order
    check(&[range(-1.0, 1.0, &[2]), range(0.5, 2.0, &[3])], |_, v| {
        v[0].tensor_multiply(v[1], &[], &[])
    });
    check(&[a.clone(), range(-1.0, 2.0, &[4, 3, 2])], |_, v| {
        v[0].tensor_multiply(v[1], &[0, 1, 2], &[2, 1, 0])
    });
    check(&[a.clone(), range(-1.0, 2.0, &[3, 5, 2])], |_, v| {
        v[0].tensor_multiply(v[1], &[0, 1], &[2, 0])
    });
    check(&[a.clone(), range(-1.0, 2.0, &[4, 2, 5])], |_, v| {
        v[0].tensor_multiply(v[1], &[2, -3], &[0, 1])
    });
    // a scalar operand, and the same node as both operands
    check(
        &[DoubleTensor::scalar(1.5), range(-1.0, 2.0, &[2, 3])],
        |_, v| v[0].tensor_multiply(v[1], &[], &[]),
    );
    check(&[range(-1.0, 2.0, &[3, 3])], |_, v| {
        v[0].tensor_multiply(v[0], &[1], &[0])
    });
    // operands that are themselves broadcast
    check(
        &[range(-1.0, 2.0, &[3]), range(0.5, 1.5, &[3, 2])],
        |_, v| v[0].broadcast_to(&[2, 3]).tensor_multiply(v[1], &[1], &[0]),
    );
    // an empty contracted dimension: every result is zero, as is every partial
    check(
        &[DoubleTensor::zeros(&[2, 0]), DoubleTensor::zeros(&[0, 3])],
        |_, v| v[0].tensor_multiply(v[1], &[1], &[0]),
    );
}

/// The exact Jacobian [n_out, n_in] of a function that is affine in each element of its input: column j is
/// f(x with x_j = 1) - f(x with x_j = 0).
fn affine_jacobian(x: &DoubleTensor, f: impl Fn(&DoubleTensor) -> DoubleTensor) -> Vec<f64> {
    let n_in = x.len();
    let columns: Vec<Vec<f64>> = (0..n_in)
        .map(|j| {
            let at = |v: f64| {
                let mut data = x.as_slice().to_vec();
                data[j] = v;
                f(&DoubleTensor::new(x.shape(), data).unwrap()).into_vec()
            };
            at(1.0).iter().zip(at(0.0)).map(|(a, b)| a - b).collect()
        })
        .collect();
    let n_out = columns.first().map_or(0, Vec::len);
    (0..n_out * n_in)
        .map(|k| columns[k % n_in][k / n_in])
        .collect()
}

#[test]
fn closed_form_partials_match_the_exact_jacobian() {
    // products, scans and contractions are affine in each element, so differencing at 1 and 0 gives the exact
    // Jacobian; the closed forms must agree to rounding, including around zeros
    let zeros = cube_with_zeros();
    let other = range(-1.0, 2.0, &[4, 3]);
    type Case<'a> = (
        Box<dyn Fn(&DoubleTensor) -> DoubleTensor + 'a>,
        Box<dyn for<'b> Fn(&'b GraphBuilder, DoubleVar<'b>) -> DoubleVar<'b> + 'a>,
    );
    let cases: Vec<Case<'_>> = vec![
        (
            Box::new(|x| DoubleTensor::scalar(x.product())),
            Box::new(|_, v| v.product()),
        ),
        (
            Box::new(|x| x.product_axes(&[0, 2]).unwrap()),
            Box::new(|_, v| v.product_axes(&[0, 2])),
        ),
        (
            Box::new(|x| x.cumprod(1).unwrap()),
            Box::new(|_, v| v.cumprod(1)),
        ),
        (
            Box::new(|x| x.cumprod(2).unwrap()),
            Box::new(|_, v| v.cumprod(2)),
        ),
        (
            Box::new(|x| x.cumsum(0).unwrap()),
            Box::new(|_, v| v.cumsum(0)),
        ),
        (
            Box::new(|x| x.tensor_multiply(&other, &[2, 1], &[0, 1]).unwrap()),
            Box::new(|g, v| {
                let o = g.constant(other.clone());
                v.tensor_multiply(o, &[2, 1], &[0, 1])
            }),
        ),
    ];
    for (k, (f, build)) in cases.iter().enumerate() {
        let expected = affine_jacobian(&zeros, f);
        let g = GraphBuilder::new();
        let x = build_one(&g, zeros.clone());
        let out = build(&g, x);
        let (xid, oid) = (x.id(), out.id());
        let (graph, state) = evaluated(g, &[oid]);
        let forward = forward_partials(&graph, &state, xid, &[oid]).unwrap();
        let reverse = reverse_partials(&graph, &state, oid, &[xid]).unwrap();
        for (mode, partial) in [("forward", forward.get(oid)), ("reverse", reverse.get(xid))] {
            let partial = partial.unwrap();
            assert_eq!(partial.len(), expected.len(), "case {k} {mode}");
            for (i, (&a, &e)) in partial.as_slice().iter().zip(&expected).enumerate() {
                assert!(
                    (a - e).abs() <= 1e-12 * e.abs().max(1.0),
                    "case {k} {mode} element {i}: {a} != {e}"
                );
            }
        }
    }
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
            .add(&(DoubleTensor::eye(2).unwrap() * 2.0))
            .unwrap()],
        |_, v| v[0].determinant(),
    );
    // Cholesky reads only the lower triangle, so perturbing the upper triangle changes nothing
    check(&[spd(4)], |_, v| v[0].cholesky());
    check(&[range(-0.3, 0.4, &[3, 3])], |_, v| {
        let x = v[0];
        (x.matmul_transposed(x, false, true) + DoubleTensor::eye(3).unwrap() * 3.0).cholesky()
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
    // a predicate smaller than the output, broadcasting with each branch
    for threshold in [-100.0, 100.0] {
        check(
            &[range(0.5, 2.0, &[2, 3]), range(-1.0, 1.0, &[3])],
            move |g, v| g.where_(v[1].sum().lt(g.scalar(threshold)), v[0], v[1] * 2.0),
        );
    }
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

#[test]
fn set_with_mask_replace_nan_and_multiplexer_pass_gradients() {
    // set_with_mask(x, x > 0, 0) * x at x = -1 keeps x, so the derivative of x * x is -2 (not -1)
    check(&[DoubleTensor::scalar(-1.0)], |_, v| {
        v[0].set_with_mask(v[0].gt_mask(0.0), 0.0) * v[0]
    });
    // the operand's gradient where the mask is 0, the (scalar) set value's where it is 1
    check(
        &[t(&[3], &[-1.0, 0.5, 2.0]), DoubleTensor::scalar(1.5)],
        |_, v| (v[0].set_with_mask(v[0].gt_mask(0.0), v[1]) * v[0]).sin(),
    );
    check(
        &[t(&[2, 2], &[-1.0, 0.5, 2.0, -0.3]), t(&[1, 1], &[0.8])],
        |_, v| v[0].set_with_mask(v[0].lt_mask(0.0), v[1]).exp().sum(),
    );
    // replace_nan passes the gradient where the input wasn't NaN, and none where it was replaced
    check(&[t(&[3], &[-1.0, 2.0, 0.5])], |_, v| {
        v[0].ln().replace_nan(5.0) * v[0]
    });
    // the multiplexer passes the selected option's gradient (and zero for the others)
    for selector in [0, 1] {
        check(
            &[DoubleTensor::scalar(0.7), range(0.5, 2.0, &[1])],
            move |g, v| {
                let options = [v[0].broadcast_to(&[1]) * 3.0, v[1].exp()];
                g.multiplexer(g.int_scalar(selector), &options) * v[1]
            },
        );
    }
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

#[test]
fn log_sum_exp_and_softmax_gradients() {
    let cube = t(
        &[2, 3, 2],
        &[
            0.3, -1.2, 2.0, 0.7, -0.4, 1.1, 0.9, -2.0, 0.2, 1.6, -0.8, 0.05,
        ],
    );
    check(&[cube.clone()], |_, v| v[0].log_sum_exp());
    for axes in [vec![0], vec![1], vec![-1], vec![0, 2], vec![]] {
        check(&[cube.clone()], |_, v| v[0].log_sum_exp_axes(&axes));
    }
    for axis in [0, 1, 2, -1] {
        check(&[cube.clone()], |_, v| v[0].softmax(axis));
        check(&[cube.clone()], |_, v| v[0].log_softmax(axis));
        // a non-identity cotangent into each rule
        check(&[cube.clone()], |_, v| {
            (v[0].softmax(axis) * cube.clone()).sum()
        });
        check(&[cube.clone()], |_, v| {
            (v[0].log_softmax(axis) * cube.clone()).exp().sum()
        });
    }
    // large values: the shifted forms stay finite and the gradients are the softmax
    check(&[t(&[3], &[1000.0, 1000.5, 999.0])], |_, v| {
        v[0].log_sum_exp()
    });
}

#[test]
fn log_sum_exp_gradient_is_softmax_and_never_nan() {
    let g = GraphBuilder::new();
    let x = g.uniform_with_shape(&[2], -1000.0, 1000.0);
    g.set_value(x, t(&[2], &[1000.0, 1000.0]));
    let lse = x.log_sum_exp();
    let soft = x.softmax(0);
    let (x, lse, soft) = (x.id(), lse.id(), soft.id());
    let (graph, mut state) = g.build().unwrap();
    state
        .lazy_eval(&graph, &[lse, soft], &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let value = state.f64(lse).unwrap().as_slice()[0];
    assert!((value - (1000.0 + std::f64::consts::LN_2)).abs() < 1e-12);
    let grad = reverse_partials(&graph, &state, lse, &[x]).unwrap();
    assert_eq!(grad.get(x).unwrap().as_slice(), &[0.5, 0.5]);
    assert_eq!(state.f64(soft).unwrap().as_slice(), &[0.5, 0.5]);

    // every element -inf: the value is -inf and the gradients are finite (uniform)
    state
        .set_and_cascade(&graph, x, t(&[2], &[f64::NEG_INFINITY; 2]))
        .unwrap();
    assert_eq!(state.f64(lse).unwrap().as_slice(), &[f64::NEG_INFINITY]);
    let reverse = reverse_partials(&graph, &state, lse, &[x]).unwrap();
    assert_eq!(reverse.get(x).unwrap().as_slice(), &[0.5, 0.5]);
    let forward = forward_partials(&graph, &state, x, &[lse, soft]).unwrap();
    assert_eq!(forward.get(lse).unwrap().as_slice(), &[0.5, 0.5]);
    assert!(
        forward
            .get(soft)
            .unwrap()
            .as_slice()
            .iter()
            .all(|v| v.is_finite())
    );
}

/// Non-symmetric, well-conditioned matrices [batch..., n, n]: smooth values plus a dominant diagonal.
fn general(batch: &[usize], n: usize, seed: f64) -> DoubleTensor {
    let shape: Vec<usize> = batch.iter().copied().chain([n, n]).collect();
    let len: usize = shape.iter().product();
    let values: Vec<f64> = (0..len).map(|i| (seed + 1.3 * i as f64).sin()).collect();
    t(&shape, &values)
        .add(&(DoubleTensor::eye(n).unwrap() * (n as f64 + 1.0)))
        .unwrap()
}

#[test]
fn solve_triangular_matches_the_tensor_function_and_finite_differences() {
    // the unread triangle is far from zero: the solve must ignore it and give it no gradient
    let t_matrix = general(&[2], 3, 0.4);
    let rhs = range(-1.0, 1.5, &[2, 3, 2]);
    for lower in [true, false] {
        for transpose in [false, true] {
            let g = GraphBuilder::new();
            let x = g
                .constant(t_matrix.clone())
                .solve_triangular(rhs.clone(), lower, transpose)
                .id();
            let (graph, mut state) = g.build().unwrap();
            state
                .lazy_eval(&graph, &[x], &mut KeanuRng::seed_from_u64(1))
                .unwrap();
            let expected = t_matrix.solve_triangular(&rhs, lower, transpose).unwrap();
            assert_eq!(state.f64(x).unwrap(), &expected);

            check(&[t_matrix.clone(), rhs.clone()], |_, v| {
                v[0].solve_triangular(v[1], lower, transpose)
            });
            // a vector right-hand side against a batch of matrices
            check(&[t_matrix.clone(), range(-0.5, 2.0, &[3])], |_, v| {
                v[0].solve_triangular(v[1], lower, transpose)
            });
            // one matrix against a batch of right-hand sides, and a non-identity cotangent
            check(&[general(&[], 3, 1.1), rhs.clone()], |_, v| {
                (v[0].solve_triangular(v[1], lower, transpose) * rhs.clone())
                    .sin()
                    .sum()
            });
        }
    }
}

#[test]
fn solve_satisfies_the_system_and_its_gradients_match_finite_differences() {
    let a = general(&[2], 4, 0.2);
    let rhs = range(-1.0, 1.0, &[2, 4, 3]);
    let g = GraphBuilder::new();
    let x = g.constant(a.clone()).solve(rhs.clone()).id();
    let (graph, mut state) = g.build().unwrap();
    state
        .lazy_eval(&graph, &[x], &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let product = a.matmul(state.f64(x).unwrap()).unwrap();
    for (p, r) in product.as_slice().iter().zip(rhs.as_slice()) {
        assert!((p - r).abs() < 1e-12);
    }

    check(&[a.clone(), rhs.clone()], |_, v| v[0].solve(v[1]));
    check(&[a.clone(), range(-0.5, 2.0, &[4])], |_, v| {
        v[0].solve(v[1])
    });
    check(&[general(&[], 4, 0.9), rhs.clone()], |_, v| {
        (v[0].solve(v[1]) * rhs.clone()).sin().sum()
    });
    // broadcasting a batch of matrices against one right-hand side
    check(&[a, range(-1.0, 1.0, &[4, 1])], |_, v| {
        v[0].solve(v[1]).sum()
    });
}

#[test]
fn singular_systems_are_errors_not_panics() {
    let g = GraphBuilder::new();
    let a = g.uniform_with_shape(&[2, 2], -10.0, 10.0);
    g.set_value(a, t(&[2, 2], &[1.0, 2.0, 3.0, 4.0]));
    let x = a.solve(DoubleTensor::ones(&[2, 1])).id();
    let lower = a
        .solve_triangular(DoubleTensor::ones(&[2]), true, false)
        .id();
    let a = a.id();
    let (graph, mut state) = g.build().unwrap();
    state
        .lazy_eval(&graph, &[x, lower], &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    assert!(state.f64(x).is_some());
    let singular = state.set_and_cascade(&graph, a, t(&[2, 2], &[1.0, 2.0, 2.0, 4.0]));
    assert!(singular.is_err(), "a singular solve must fail evaluation");
    let zero_diagonal = state.set_and_cascade(&graph, a, t(&[2, 2], &[0.0, 0.0, 3.0, 4.0]));
    assert!(zero_diagonal.is_err());
}

#[test]
fn slogdet_values_and_gradients() {
    // the second matrix has a negative determinant
    let a = general(&[2], 3, 0.7);
    let mut data = a.as_slice().to_vec();
    for value in &mut data[9..12] {
        *value = -*value;
    }
    let a = t(&[2, 3, 3], &data);
    let g = GraphBuilder::new();
    let (sign, log_abs) = g.constant(a.clone()).slogdet();
    let (sign, log_abs) = (sign.id(), log_abs.id());
    let (graph, mut state) = g.build().unwrap();
    state
        .lazy_eval(&graph, &[sign, log_abs], &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let det = a.determinant().unwrap();
    assert_eq!(state.f64(sign).unwrap().as_slice(), &[1.0, -1.0]);
    for (l, d) in state
        .f64(log_abs)
        .unwrap()
        .as_slice()
        .iter()
        .zip(det.as_slice())
    {
        assert!((l - d.abs().ln()).abs() < 1e-12);
    }
    check(&[a.clone()], |_, v| v[0].log_abs_determinant());
    check(&[general(&[], 4, 0.1)], |_, v| v[0].log_abs_determinant());
    // sign · exp(log |det|) is the determinant, whose gradient is checked elsewhere
    check(&[a], |_, v| {
        let (sign, log_abs) = v[0].slogdet();
        sign * log_abs.exp() - v[0].determinant()
    });
}

#[test]
fn eigh_reconstructs_the_matrix_and_its_gradients_match_finite_differences() {
    // distinct eigenvalues; the upper triangle is not read
    let a = general(&[2], 3, 0.3);
    let g = GraphBuilder::new();
    let (values, vectors) = g.constant(a.clone()).eigh();
    let rebuilt = (vectors * values.reshape(&[2, 1, 3])).matmul_transposed(vectors, false, true);
    let (values, rebuilt) = (values.id(), rebuilt.id());
    let (graph, mut state) = g.build().unwrap();
    state
        .lazy_eval(&graph, &[values, rebuilt], &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    for lane in state.f64(values).unwrap().as_slice().chunks(3) {
        assert!(lane[0] < lane[1] && lane[1] < lane[2]);
    }
    let symmetric = a
        .tril(0)
        .unwrap()
        .add(&a.tril(-1).unwrap().transpose().unwrap())
        .unwrap();
    for (r, s) in state
        .f64(rebuilt)
        .unwrap()
        .as_slice()
        .iter()
        .zip(symmetric.as_slice())
    {
        assert!((r - s).abs() < 1e-10);
    }

    check(&[a.clone()], |_, v| v[0].eigvalsh());
    check(&[a.clone()], |_, v| v[0].eigh_vectors());
    check(&[general(&[], 4, 2.0)], |_, v| {
        let (values, vectors) = v[0].eigh();
        (vectors.sin() * values.reshape(&[1, 4])).sum()
    });
    // Q Λ Qᵀ is the symmetric matrix read from the lower triangle, so its gradient is that triangle's identity map
    check(&[a], |_, v| {
        let (values, vectors) = v[0].eigh();
        (vectors * values.reshape(&[2, 1, 3])).matmul_transposed(vectors, false, true)
    });
}
