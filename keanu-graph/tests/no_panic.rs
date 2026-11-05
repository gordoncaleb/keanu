//! Fuzz style property test: building, evaluating and differentiating random graphs, including invalid shapes,
//! axes, slices and permutations, must return errors rather than panic.

use keanu_distributions::KeanuRng;
use keanu_graph::{DoubleVar, GraphBuilder, forward_partials, joint_log_prob, reverse_partials};
use keanu_tensor::{DoubleTensor, IntTensor};
use proptest::prelude::*;

#[derive(Debug, Clone)]
enum Step {
    Add(usize),
    Mul(usize),
    Div(usize),
    Pow(usize),
    Maximum(usize),
    MatMul(usize),
    Exp,
    Ln,
    Sqrt,
    Reshape(Vec<isize>),
    Permute(Vec<usize>),
    Transpose,
    Slice(String),
    SumAxes(Vec<isize>),
    CumSum(isize),
    BroadcastTo(Vec<usize>),
    Take(Vec<usize>),
    Gather(isize, Vec<i64>),
    ScatterAdd(isize, Vec<i64>, usize),
    Stack(isize, usize),
    Softmax(isize),
    LogSumExp(Vec<isize>),
    Tril(isize),
    Cholesky,
    Inverse,
    Determinant,
    SolveTriangular(usize, bool, bool),
    Solve(usize),
    Slogdet,
    Eigh,
    Standardize,
    Gaussian(usize),
    Scan(usize),
}

fn shape() -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(0usize..4, 0..4)
}

fn step() -> impl Strategy<Value = Step> {
    let slice = (
        prop::collection::vec(
            (
                -4isize..5,
                -4isize..5,
                prop::sample::select(vec![-2isize, -1, 1, 2, 0]),
            ),
            0..4,
        ),
        any::<bool>(),
    )
        .prop_map(|(dims, ellipsis)| {
            let mut parts: Vec<String> = dims
                .iter()
                .map(|(a, b, c)| format!("{a}:{b}:{c}"))
                .collect();
            if ellipsis {
                parts.insert(0, "...".into());
            }
            parts.join(",")
        });
    prop_oneof![
        (0usize..8).prop_map(Step::Add),
        (0usize..8).prop_map(Step::Mul),
        (0usize..8).prop_map(Step::Div),
        (0usize..8).prop_map(Step::Pow),
        (0usize..8).prop_map(Step::Maximum),
        (0usize..8).prop_map(Step::MatMul),
        Just(Step::Exp),
        Just(Step::Ln),
        Just(Step::Sqrt),
        prop::collection::vec(-2isize..5, 0..4).prop_map(Step::Reshape),
        prop::collection::vec(0usize..4, 0..4).prop_map(Step::Permute),
        Just(Step::Transpose),
        slice.prop_map(Step::Slice),
        prop::collection::vec(-4isize..4, 0..3).prop_map(Step::SumAxes),
        (-4isize..4).prop_map(Step::CumSum),
        shape().prop_map(Step::BroadcastTo),
        prop::collection::vec(0usize..4, 0..4).prop_map(Step::Take),
        (-3isize..3, prop::collection::vec(-1i64..4, 0..4)).prop_map(|(a, i)| Step::Gather(a, i)),
        (-3isize..3, prop::collection::vec(-1i64..4, 0..4), 0usize..4)
            .prop_map(|(a, i, n)| Step::ScatterAdd(a, i, n)),
        (-3isize..4, 0usize..8).prop_map(|(a, i)| Step::Stack(a, i)),
        (-4isize..4).prop_map(Step::Softmax),
        prop::collection::vec(-4isize..4, 0..3).prop_map(Step::LogSumExp),
        (-3isize..3).prop_map(Step::Tril),
        Just(Step::Cholesky),
        Just(Step::Inverse),
        Just(Step::Determinant),
        (0usize..8, any::<bool>(), any::<bool>())
            .prop_map(|(i, l, t)| Step::SolveTriangular(i, l, t)),
        (0usize..8).prop_map(Step::Solve),
        Just(Step::Slogdet),
        Just(Step::Eigh),
        Just(Step::Standardize),
        (0usize..8).prop_map(Step::Gaussian),
        (0usize..8).prop_map(Step::Scan),
    ]
}

fn tensor(shape: &[usize], seed: u64) -> DoubleTensor {
    let n: usize = shape.iter().product();
    DoubleTensor::new(
        shape,
        (0..n)
            .map(|i| ((i as u64 * 7 + seed) % 11) as f64 * 0.3 - 1.0)
            .collect(),
    )
    .unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 400, ..ProptestConfig::default() })]

    #[test]
    fn random_graphs_never_panic(inputs in prop::collection::vec(shape(), 1..4), steps in prop::collection::vec(step(), 0..12)) {
        let b = GraphBuilder::new();
        let mut nodes: Vec<DoubleVar<'_>> = inputs.iter().enumerate().map(|(i, s)| b.constant(tensor(s, i as u64))).collect();
        let latent = b.gaussian(0.0, 1.0);
        nodes.push(latent);
        for step in &steps {
            let last = *nodes.last().unwrap();
            let pick = |i: usize| nodes[i % nodes.len()];
            let next = match step {
                Step::Add(i) => last + pick(*i),
                Step::Mul(i) => last * pick(*i),
                Step::Div(i) => last / pick(*i),
                Step::Pow(i) => last.pow(pick(*i)),
                Step::Maximum(i) => last.maximum(pick(*i)),
                Step::MatMul(i) => last.matmul(pick(*i)),
                Step::Exp => last.exp(),
                Step::Ln => last.ln(),
                Step::Sqrt => last.sqrt(),
                Step::Reshape(s) => last.reshape(s),
                Step::Permute(p) => last.permute(p),
                Step::Transpose => last.transpose(),
                Step::Slice(s) => last.slice(s),
                Step::SumAxes(a) => last.sum_axes(a),
                Step::CumSum(a) => last.cumsum(*a),
                Step::BroadcastTo(s) => last.broadcast_to(s),
                Step::Take(i) => last.take(i),
                Step::Gather(a, i) => last.gather(*a, IntTensor::vector(i.clone())),
                Step::ScatterAdd(a, i, n) => last.scatter_add(*a, IntTensor::vector(i.clone()), *n),
                Step::Stack(a, i) => b.stack(*a, &[last, pick(*i)]),
                Step::Softmax(a) => last.softmax(*a).log_softmax(*a),
                Step::LogSumExp(a) => last.log_sum_exp_axes(a),
                Step::Tril(k) => last.tril(*k),
                Step::Cholesky => last.cholesky(),
                Step::Inverse => last.inverse(),
                Step::Determinant => last.determinant(),
                Step::SolveTriangular(i, l, t) => last.solve_triangular(pick(*i), *l, *t),
                Step::Solve(i) => last.solve(pick(*i)),
                Step::Slogdet => {
                    let (sign, log_abs) = last.slogdet();
                    sign * log_abs
                }
                Step::Eigh => {
                    let (values, vectors) = last.eigh();
                    vectors.matmul(values.reshape(&[-1, 1]))
                }
                Step::Standardize => last.standardize(),
                Step::Gaussian(i) => b.gaussian(last, pick(*i).abs() + 1.0),
                Step::Scan(i) => b.scan(pick(*i), &[last], &[last], |_, c, x, k| (c * x[0]).sin() + k[0].sum()),
            };
            nodes.push(next);
        }
        let output = nodes.last().unwrap().id();
        let latent = latent.id();
        // errors are fine; panics fail the test
        if let Ok((graph, mut state)) = b.build() {
            let mut rng = KeanuRng::seed_from_u64(1);
            let all: Vec<_> = graph.ids().collect();
            if state.lazy_eval(&graph, &all, &mut rng).is_ok() {
                let _ = joint_log_prob(&graph, &state);
                let _ = reverse_partials(&graph, &state, output, &[latent]);
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1000, ..ProptestConfig::default() })]

    /// Steps whose parameters are chosen to be valid for the current shape, so that evaluation and both auto diff
    /// modes are exercised on deep random graphs (the property above mostly exercises construction errors).
    #[test]
    fn random_valid_graphs_evaluate_and_differentiate_without_panicking(start in prop::collection::vec(1usize..4, 0..4), choices in prop::collection::vec((0usize..15, any::<u8>()), 0..16)) {
        let b = GraphBuilder::new();
        let latent = b.gaussian_with_shape(&start, 0.0, 1.0);
        let mut last: DoubleVar<'_> = latent;
        for &(choice, r) in &choices {
            let shape = last.shape().to_vec();
            let rank = shape.len();
            last = match choice {
                0 => last + 1.5,
                1 => last * last,
                2 => last / (last.abs() + 1.0),
                3 => (last.abs() + 0.5).pow(1.5),
                4 => last.maximum(b.constant(0.25)),
                5 => last.sin(),
                6 => (last.abs() + 1.0).ln(),
                7 if rank >= 2 => last.transpose(),
                8 if rank >= 1 => {
                    let mut p: Vec<usize> = (0..rank).collect();
                    p.rotate_left(r as usize % rank);
                    last.permute(&p)
                }
                9 => last.reshape(&[-1]),
                10 if rank >= 1 => last.sum_axes(&[(r as usize % rank) as isize]),
                11 if rank >= 1 => last.cumsum(-1),
                12 if rank >= 1 => last.slice("::-1"),
                13 => b.gaussian(last, 1.0),
                14 if rank >= 1 => b.scan(b.constant(DoubleTensor::zeros(&shape[1..])), &[last], &[latent], |_, c, x, k| (c + x[0]).sin() * k[0].sum()),
                _ => last.sigmoid(),
            };
        }
        let output = last.id();
        let latent = latent.id();
        let (graph, mut state) = b.build().expect("every step is valid for its shape");
        let mut rng = KeanuRng::seed_from_u64(2);
        let all: Vec<_> = graph.ids().collect();
        state.lazy_eval(&graph, &all, &mut rng).unwrap();
        joint_log_prob(&graph, &state).unwrap();
        // the output may depend on the latent only through a probabilistic node, so missing partials are fine
        let _ = reverse_partials(&graph, &state, output, &[latent]);
        let _ = forward_partials(&graph, &state, latent, &[output]);
    }
}
