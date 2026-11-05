//! Indexing by tensor: gather, its adjoint scatter-add, and stack, including their gradients and the hierarchical
//! `mu[group]` pattern where the index is itself a node whose value can change.

use keanu_distributions::KeanuRng;
use keanu_graph::{
    DoubleVar, Graph, GraphBuilder, IntVar, NodeId, State, forward_partials, reverse_partials,
};
use keanu_tensor::{DoubleTensor, IntTensor};

mod common;
use common::t;

fn rng() -> KeanuRng {
    KeanuRng::seed_from_u64(1)
}

/// A latent of the given value, so it can be differentiated with respect to.
fn latent<'b>(b: &'b GraphBuilder, value: &DoubleTensor) -> DoubleVar<'b> {
    let x = b.uniform_with_shape(value.shape(), -1000.0, 1000.0);
    b.set_value(x, value.clone());
    x
}

/// ∂output/∂input shaped [output shape..., input shape...] by central differences.
fn finite_difference(graph: &Graph, state: &State, input: NodeId, output: NodeId) -> DoubleTensor {
    let h = 1e-6;
    let value = state.f64(input).unwrap().clone();
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

/// Forward and reverse partials of `output` with respect to each input match finite differences.
fn check_gradients(graph: &Graph, state: &State, inputs: &[NodeId], output: NodeId) {
    for (k, &input) in inputs.iter().enumerate() {
        let numeric = finite_difference(graph, state, input, output);
        let forward = forward_partials(graph, state, input, &[output])
            .unwrap()
            .get(output)
            .unwrap()
            .clone();
        let reverse = reverse_partials(graph, state, output, &[input])
            .unwrap()
            .get(input)
            .unwrap()
            .clone();
        for (mode, partial) in [("forward", &forward), ("reverse", &reverse)] {
            assert_eq!(partial.shape(), numeric.shape(), "{mode} wrt input {k}");
            for (i, (&a, &e)) in partial
                .as_slice()
                .iter()
                .zip(numeric.as_slice())
                .enumerate()
            {
                assert!(
                    (a - e).abs() <= 1e-6 * e.abs().max(1.0),
                    "{mode} partial wrt input {k}, element {i}: {a} != finite difference {e}"
                );
            }
        }
    }
}

/// Builds a graph of latents with the given values and a function of them, evaluates it and checks its gradients.
fn check(
    inputs: &[DoubleTensor],
    build: impl for<'b> Fn(&'b GraphBuilder, &[DoubleVar<'b>]) -> DoubleVar<'b>,
) {
    let b = GraphBuilder::new();
    let vars: Vec<_> = inputs.iter().map(|v| latent(&b, v)).collect();
    let output = build(&b, &vars).id();
    let ids: Vec<NodeId> = vars.iter().map(|v| v.id()).collect();
    let (graph, mut state) = b.build().unwrap();
    state.lazy_eval(&graph, &[output], &mut rng()).unwrap();
    check_gradients(&graph, &state, &ids, output);
}

#[test]
fn gather_selects_by_index() {
    let b = GraphBuilder::new();
    let mu = b.constant(DoubleTensor::vector(vec![10.0, 20.0, 30.0]));
    let group = b.int_constant(IntTensor::vector(vec![0, 2, 2, 1]));
    let picked = mu.gather(0, group);
    assert_eq!(picked.shape(), vec![4]);
    let picked = picked.id();
    let (graph, mut state) = b.build().unwrap();
    state.lazy_eval(&graph, &[picked], &mut rng()).unwrap();
    assert_eq!(
        state.f64(picked).unwrap().as_slice(),
        &[10.0, 30.0, 30.0, 20.0]
    );
}

#[test]
fn gather_on_an_inner_axis_inserts_the_index_shape() {
    let b = GraphBuilder::new();
    let x = b.constant(t(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let picked = x.gather(1, IntTensor::vector(vec![2, 0, 2, 1]));
    assert_eq!(picked.shape(), vec![2, 4]);
    let matrix_index = x.gather(-1, IntTensor::new(&[2, 2], vec![0, 1, 1, 2]).unwrap());
    assert_eq!(matrix_index.shape(), vec![2, 2, 2]);
    let (picked, matrix_index) = (picked.id(), matrix_index.id());
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &[picked, matrix_index], &mut rng())
        .unwrap();
    assert_eq!(
        state.f64(picked).unwrap().as_slice(),
        &[3.0, 1.0, 3.0, 2.0, 6.0, 4.0, 6.0, 5.0]
    );
}

#[test]
fn gather_applies_to_integers_and_booleans() {
    let b = GraphBuilder::new();
    let idx = IntTensor::vector(vec![1, 1, 0]);
    let ints = b
        .int_constant(IntTensor::vector(vec![5, 7]))
        .gather(0, idx.clone());
    let bools = b
        .bool_constant(keanu_tensor::BoolTensor::vector(vec![true, false]))
        .gather(0, idx);
    let (ints, bools) = (ints.id(), bools.id());
    let (graph, mut state) = b.build().unwrap();
    state.lazy_eval(&graph, &[ints, bools], &mut rng()).unwrap();
    assert_eq!(state.i64(ints).unwrap().as_slice(), &[7, 7, 5]);
    assert_eq!(
        state.value(bools).unwrap().as_bool().unwrap().as_slice(),
        &[false, false, true]
    );
}

#[test]
fn repeated_indices_accumulate_gradients() {
    let idx = IntTensor::vector(vec![0, 2, 2, 1, 2]);
    check(&[t(&[3], &[0.5, -1.0, 2.0])], |_, v| {
        v[0].gather(0, idx.clone())
    });
    // a non-linear use, so the reverse cotangents differ per output element
    check(&[t(&[3], &[0.5, -1.0, 2.0])], |_, v| {
        (v[0].gather(0, idx.clone()) * DoubleTensor::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]))
            .exp()
            .sum()
    });
    check(&[t(&[2, 3], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])], |_, v| {
        v[0].gather(1, IntTensor::new(&[2, 2], vec![2, 2, 0, 2]).unwrap())
            .sin()
    });
    check(&[t(&[3, 2], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])], |_, v| {
        v[0].gather(0, IntTensor::scalar(1))
    });

    // the gradient of a sum of gathered values counts each index
    let b = GraphBuilder::new();
    let mu = latent(&b, &t(&[3], &[0.5, -1.0, 2.0]));
    let total = mu.gather(0, idx.clone()).sum();
    let (mu, total) = (mu.id(), total.id());
    let (graph, mut state) = b.build().unwrap();
    state.lazy_eval(&graph, &[total], &mut rng()).unwrap();
    let grad = reverse_partials(&graph, &state, total, &[mu]).unwrap();
    assert_eq!(grad.get(mu).unwrap().as_slice(), &[1.0, 1.0, 3.0]);
}

#[test]
fn scatter_add_is_gathers_adjoint_and_differentiable() {
    let b = GraphBuilder::new();
    let u = b.constant(DoubleTensor::vector(vec![1.0, 2.0, 3.0, 4.0]));
    let summed = u.scatter_add(0, IntTensor::vector(vec![0, 2, 2, 1]), 3);
    assert_eq!(summed.shape(), vec![3]);
    let summed = summed.id();
    let (graph, mut state) = b.build().unwrap();
    state.lazy_eval(&graph, &[summed], &mut rng()).unwrap();
    assert_eq!(state.f64(summed).unwrap().as_slice(), &[1.0, 4.0, 5.0]);

    check(&[t(&[4], &[0.1, -0.2, 0.3, 0.4])], |_, v| {
        v[0].scatter_add(0, IntTensor::vector(vec![0, 2, 2, 1]), 3)
    });
    check(&[t(&[2, 3], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])], |_, v| {
        (v[0].scatter_add(1, IntTensor::vector(vec![1, 1, 0]), 4) * 2.0).exp()
    });
    // gather of a scatter-add, so each rule sees a non-identity cotangent
    check(&[t(&[3], &[0.1, 0.2, 0.3])], |_, v| {
        let idx = IntTensor::vector(vec![1, 1, 0]);
        v[0].ln_1p()
            .scatter_add(0, idx.clone(), 2)
            .gather(0, idx)
            .cos()
    });
}

#[test]
fn out_of_range_indices_are_evaluation_errors() {
    for bad in [-1, 3, i64::MIN, i64::MAX] {
        let b = GraphBuilder::new();
        let mu = b.constant(DoubleTensor::vector(vec![1.0, 2.0, 3.0]));
        let picked = mu.gather(0, IntTensor::vector(vec![0, bad]));
        let summed = mu.scatter_add(0, IntTensor::vector(vec![0, 1, bad]), 3);
        let (picked, summed) = (picked.id(), summed.id());
        let (graph, mut state) = b.build().unwrap();
        assert!(
            state.lazy_eval(&graph, &[picked], &mut rng()).is_err(),
            "gather index {bad}"
        );
        assert!(
            state.lazy_eval(&graph, &[summed], &mut rng()).is_err(),
            "scatter-add index {bad}"
        );
    }

    // an index node set out of range fails the cascade rather than panicking
    let b = GraphBuilder::new();
    let mu = b.constant(DoubleTensor::vector(vec![1.0, 2.0, 3.0]));
    let group = b.uniform_int(0, 3);
    let picked = mu.gather(0, group);
    let (group, picked) = (group.id(), picked.id());
    let (graph, mut state) = b.build().unwrap();
    state.lazy_eval(&graph, &[picked], &mut rng()).unwrap();
    assert!(
        state
            .set_and_cascade(&graph, group, IntTensor::scalar(5))
            .is_err()
    );
}

#[test]
fn indices_must_be_integers_with_a_matching_shape() {
    let b = GraphBuilder::new();
    let mu = b.constant(DoubleTensor::vector(vec![1.0, 2.0, 3.0]));
    let not_int = b.add_node(keanu_graph::op::Op::Gather { axis: 0 }, &[mu.id(), mu.id()]);
    assert!(not_int.is_err());
    let bad_axis = b.add_node(
        keanu_graph::op::Op::Gather { axis: 1 },
        &[mu.id(), b.int_scalar(0).id()],
    );
    assert!(bad_axis.is_err());
    let wrong_shape = b.add_node(
        keanu_graph::op::Op::ScatterAdd { axis: 0, size: 3 },
        &[mu.id(), b.int_constant(IntTensor::vector(vec![0, 1])).id()],
    );
    assert!(wrong_shape.is_err());
    let bools = b.bool_constant(keanu_tensor::BoolTensor::vector(vec![true]));
    let bool_scatter = b.add_node(
        keanu_graph::op::Op::ScatterAdd { axis: 0, size: 3 },
        &[bools.id(), b.int_constant(IntTensor::vector(vec![0])).id()],
    );
    assert!(bool_scatter.is_err());
}

/// The hierarchical pattern: `mu[group]` where the group is a categorical latent. Changing the group's value
/// propagates through a cascade, and gradients follow the new index.
#[test]
fn a_changing_index_propagates_and_moves_the_gradient() {
    let b = GraphBuilder::new();
    let mu = latent(&b, &t(&[3], &[1.0, 2.0, 3.0]));
    let p = b.constant(1.0);
    let group = b.categorical(&[2], &[p, p, p]);
    b.set_value(group, IntTensor::vector(vec![0, 1]));
    let picked = mu.gather(0, group);
    let y = (picked * picked).sum();
    let (mu, group, picked, y) = (mu.id(), group.id(), picked.id(), y.id());
    let (graph, mut state) = b.build().unwrap();
    state.lazy_eval(&graph, &[y], &mut rng()).unwrap();
    assert_eq!(state.f64(picked).unwrap().as_slice(), &[1.0, 2.0]);
    let grad = |state: &State| {
        reverse_partials(&graph, state, y, &[mu])
            .unwrap()
            .get(mu)
            .unwrap()
            .as_slice()
            .to_vec()
    };
    assert_eq!(grad(&state), vec![2.0, 4.0, 0.0]);

    state
        .set_and_cascade(&graph, group, IntTensor::vector(vec![2, 2]))
        .unwrap();
    assert_eq!(state.f64(picked).unwrap().as_slice(), &[3.0, 3.0]);
    assert_eq!(state.f64(y).unwrap().as_slice(), &[18.0]);
    assert_eq!(grad(&state), vec![0.0, 0.0, 12.0]);
    let forward = forward_partials(&graph, &state, mu, &[y]).unwrap();
    assert_eq!(forward.get(y).unwrap().as_slice(), &[0.0, 0.0, 12.0]);
    check_gradients(&graph, &state, &[mu], y);
}

/// Gradients never flow into the index.
#[test]
fn indices_are_not_differentiable() {
    let b = GraphBuilder::new();
    let mu = latent(&b, &t(&[3], &[1.0, 2.0, 3.0]));
    let group: IntVar<'_> = b.uniform_int_with_shape(&[2], 0, 3);
    b.set_value(group, IntTensor::vector(vec![2, 0]));
    let y = mu.gather(0, group).sum();
    let (mu, group, y) = (mu.id(), group.id(), y.id());
    let (graph, mut state) = b.build().unwrap();
    state.lazy_eval(&graph, &[y], &mut rng()).unwrap();
    let grad = reverse_partials(&graph, &state, y, &[mu, group]).unwrap();
    assert_eq!(grad.get(mu).unwrap().as_slice(), &[1.0, 0.0, 1.0]);
    assert!(grad.get(group).is_err());
}

#[test]
fn empty_indices_give_empty_results() {
    let b = GraphBuilder::new();
    let mu = latent(&b, &t(&[3], &[1.0, 2.0, 3.0]));
    let picked = mu.gather(0, IntTensor::vector(vec![]));
    assert_eq!(picked.shape(), vec![0]);
    let total = picked.sum();
    let (mu, picked, total) = (mu.id(), picked.id(), total.id());
    let (graph, mut state) = b.build().unwrap();
    state.lazy_eval(&graph, &[total], &mut rng()).unwrap();
    assert!(state.f64(picked).unwrap().is_empty());
    assert_eq!(state.f64(total).unwrap().as_slice(), &[0.0]);
    let grad = reverse_partials(&graph, &state, total, &[mu]).unwrap();
    assert_eq!(grad.get(mu).unwrap().as_slice(), &[0.0, 0.0, 0.0]);
    let forward = forward_partials(&graph, &state, mu, &[picked]).unwrap();
    assert_eq!(forward.get(picked).unwrap().shape(), &[0, 3]);
}

#[test]
fn stack_joins_along_a_new_axis_and_splits_gradients() {
    let b = GraphBuilder::new();
    let parts: Vec<_> = (0..3)
        .map(|i| b.constant(DoubleTensor::vector(vec![i as f64, 10.0 + i as f64])))
        .collect();
    let rows = b.stack(0, &parts);
    let columns = b.stack(-1, &parts);
    assert_eq!(rows.shape(), vec![3, 2]);
    assert_eq!(columns.shape(), vec![2, 3]);
    let (rows, columns) = (rows.id(), columns.id());
    drop(parts);
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &[rows, columns], &mut rng())
        .unwrap();
    assert_eq!(
        state.f64(rows).unwrap().as_slice(),
        &[0.0, 10.0, 1.0, 11.0, 2.0, 12.0]
    );
    assert_eq!(
        state.f64(columns).unwrap().as_slice(),
        &[0.0, 1.0, 2.0, 10.0, 11.0, 12.0]
    );

    let inputs = [
        t(&[2], &[0.1, 0.2]),
        t(&[2], &[0.3, 0.4]),
        t(&[2], &[0.5, 0.6]),
    ];
    check(&inputs, |g, v| g.stack(0, &[v[0], v[1], v[2]]));
    check(&inputs, |g, v| {
        (g.stack(1, &[v[0], v[1], v[0], v[2]]) * DoubleTensor::vector(vec![1.0, 2.0, 3.0, 4.0]))
            .exp()
    });
    let ints = {
        let b = GraphBuilder::new();
        let s = b.stack(0, &[b.int_scalar(1), b.int_scalar(2)]);
        assert_eq!(s.shape(), vec![2]);
        b.build().is_ok()
    };
    assert!(ints);

    let b = GraphBuilder::new();
    let a = b.constant(DoubleTensor::vector(vec![1.0, 2.0]));
    let c = b.constant(DoubleTensor::vector(vec![1.0, 2.0, 3.0]));
    let _ = b.stack(0, &[a, c]);
    assert!(b.build().is_err(), "stacking different shapes is an error");
    let b = GraphBuilder::new();
    let _ = b.stack::<DoubleVar<'_>>(0, &[]);
    assert!(b.build().is_err(), "stacking nothing is an error");
}
