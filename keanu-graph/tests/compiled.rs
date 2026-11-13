//! A `CompiledModel` gives exactly the general path's log probability and gradient
//! (`Parameterisation::log_prob_and_gradient_compiled` against `Parameterisation::log_prob_and_gradient`).

use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, GraphBuilder, Model, NodeId, Parameterisation, Space, State};
use keanu_tensor::{DoubleTensor, IntTensor};
use std::sync::Arc;

/// A model with nodes of many distributions and deterministic ops of every kind the compiled program handles:
/// element-wise unary and binary ops with broadcasting, a gather along constant indices, stacking, concatenation,
/// slicing and reshaping, ops it runs through their general rules (a matrix product, a sum, a `where` with a branch
/// that is −∞ where it isn't selected) and a factor. Latents are sampled from their priors.
fn model(seed: u64) -> (Arc<Graph>, State, Model, Vec<NodeId>) {
    let b = GraphBuilder::new();
    let a = b.gaussian_with_shape(&[3], 0.0, 1.0);
    let s = b.gamma(1.0, 2.0);
    let h = b.half_gaussian(s);
    let e = b.exponential(1.5);
    let bt = b.beta(2.0, 3.0);
    let ln = b.log_normal(0.0, 0.5);
    let c = b.cauchy(a, h);
    let dir = b.dirichlet(DoubleTensor::vector(vec![1.0, 2.0, 3.0]));
    let p = b.poisson(s);
    b.observe(p, IntTensor::scalar(3));
    let bin = b.binomial(bt, 10);
    b.observe(bin, IntTensor::scalar(4));
    // element-wise with broadcasting, a gather and data movement
    let picked = a.gather(0, b.int_constant(IntTensor::vector(vec![2, 0, 0, 1])));
    let stacked = b.stack(0, &[a, c]).slice("1:, :").reshape(&[3]);
    let joined = b.concat(0, &[picked, stacked]);
    let shifted = (joined * s - ln).exp().ln_1p() / (e + 1.0) + dir.take(&[1]);
    // general ops: a matrix product, a sum, and a where whose unselected branch is impossible
    let m = b.constant(
        DoubleTensor::new(&[2, 7], (0..14).map(|i| (i as f64 * 0.3).sin()).collect()).unwrap(),
    );
    let projected = m.matmul(shifted.reshape(&[7, 1])).reshape(&[2]);
    let total = shifted.sum();
    let guarded = b.where_(total.gt(-1e9), projected, (projected - 1e12).ln());
    let y = b.gaussian(guarded, e + total.abs() * 0.01 + 0.5);
    b.observe(y, DoubleTensor::vector(vec![0.5, -0.2]));
    // Student-t terms from the registers: scalar parameters, a location and scale broadcast over the value, and a
    // scale per element
    let nu = b.log_normal(1.5, 0.5);
    let t = b.location_scale_student_t(nu, shifted.slice(":3") * 0.5, s);
    b.observe(t, DoubleTensor::vector(vec![0.4, -1.1, 2.5]));
    let lt = b.location_scale_student_t_with_shape(&[2, 3], nu, a, h + 0.1);
    b.factor(-(lt * lt).sum() * 0.05);
    let per_element = b.location_scale_student_t(nu + e, a, (a * 0.3).exp());
    b.observe(per_element, DoubleTensor::vector(vec![0.1, 0.7, -0.6]));
    b.factor(-(a * a).sum() * 0.1);
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<_> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(seed))
        .unwrap();
    let graph = Arc::new(graph);
    let model = Model::new(graph.clone(), &state).unwrap();
    let latents = model.continuous_latents().to_vec();
    (graph, state, model, latents)
}

/// Compares the two paths at points around the state's.
fn assert_identical(param: &Parameterisation<'_>, state: &State, label: &str) {
    let mut compiled = param.compile(state).expect("the model compiles");
    let mut general_state = state.clone();
    let start = param.from_state(state).unwrap();
    for k in 0..8 {
        let z: Vec<f64> = start
            .iter()
            .enumerate()
            .map(|(i, &x)| x + 0.2 * k as f64 * (i as f64 * 1.3 + k as f64).sin())
            .collect();
        for jacobian in [true, false] {
            let general = param.log_prob_and_gradient(&z, jacobian, &mut general_state);
            let fast = param.log_prob_and_gradient_compiled(&mut compiled, &z, jacobian, state);
            // an invalid value (a Dirichlet's off the simplex, in constrained space) is an error on both
            let (Ok((lp, g)), Ok((clp, cg))) = (&general, &fast) else {
                assert_eq!(
                    format!("{:?}", fast.err()),
                    format!("{:?}", general.err()),
                    "{label} point {k}"
                );
                continue;
            };
            let (lp, g, clp, cg) = (*lp, g.clone(), *clp, cg.clone());
            // the allocation-free form gives the same numbers
            let mut into = vec![f64::NAN; z.len()];
            let (ilp, possible) = param
                .log_prob_and_gradient_compiled_into(&mut compiled, &z, jacobian, state, &mut into)
                .unwrap();
            assert_eq!(ilp.to_bits(), clp.to_bits(), "{label} point {k}: into");
            assert_eq!(possible, cg.is_some(), "{label} point {k}: into");
            if let Some(cg) = &cg {
                let cg: Vec<u64> = cg.iter().map(|x| x.to_bits()).collect();
                let into: Vec<u64> = into.iter().map(|x| x.to_bits()).collect();
                assert_eq!(into, cg, "{label} point {k}: into");
            }
            assert_eq!(
                clp.to_bits(),
                lp.to_bits(),
                "{label} point {k}: {clp} vs {lp}"
            );
            match (g, cg) {
                (Some(g), Some(cg)) => {
                    let g: Vec<u64> = g.iter().map(|x| x.to_bits()).collect();
                    let cg: Vec<u64> = cg.iter().map(|x| x.to_bits()).collect();
                    assert_eq!(cg, g, "{label} point {k}");
                }
                (None, None) => {}
                (g, cg) => panic!("{label} point {k}: {g:?} vs {cg:?}"),
            }
        }
    }
}

#[test]
fn the_compiled_model_matches_the_general_path_exactly() {
    for seed in 0..10 {
        let (_, state, model, _) = model(seed);
        for space in [Space::Unconstrained, Space::Constrained] {
            let param = Parameterisation::with_space(&model, &state, space).unwrap();
            assert_identical(&param, &state, &format!("seed {seed}, {space:?}"));
        }
    }
}

#[test]
fn a_compiled_block_reads_the_other_latents_from_the_state() {
    for seed in 0..5 {
        let (_, mut state, model, latents) = model(seed);
        let block = &latents[..3];
        let param =
            Parameterisation::for_latents(&model, &state, block, Space::Unconstrained).unwrap();
        assert_identical(&param, &state, &format!("seed {seed}, block"));
        // another step moves a latent outside the block: the compiled model follows the state
        let other = latents[4];
        let value = state.f64(other).unwrap().clone();
        let mut graph_state = state.clone();
        graph_state
            .set(model.graph(), other, value.map(|v| v * 0.9))
            .unwrap();
        graph_state.cascade(model.graph(), &[other]).unwrap();
        state = graph_state;
        assert_identical(&param, &state, &format!("seed {seed}, block after a move"));
    }
}

#[test]
fn an_impossible_point_is_impossible_on_both_paths() {
    let b = GraphBuilder::new();
    let x = b.uniform(0.0, 1.0);
    let y = b.gaussian(x, 1.0);
    b.observe(y, DoubleTensor::scalar(0.3));
    b.set_value(x, DoubleTensor::scalar(0.5));
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let param = Parameterisation::with_space(&model, &state, Space::Constrained).unwrap();
    let mut compiled = param.compile(&state).unwrap();
    let (lp, g) = param
        .log_prob_and_gradient_compiled(&mut compiled, &[1.5], true, &state)
        .unwrap();
    assert_eq!(lp, f64::NEG_INFINITY);
    assert!(g.is_none());
    let (lp, g) = param
        .log_prob_and_gradient(&[1.5], true, &mut state.clone())
        .unwrap();
    assert_eq!(lp, f64::NEG_INFINITY);
    assert!(g.is_none());
}
