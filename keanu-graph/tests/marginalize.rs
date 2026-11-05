//! Automatic marginalisation of discrete latents: the rewritten model's log probability, gradients and Hessians
//! against hand-marginalised models and brute-force enumeration, the supported dependency patterns, and the
//! rejected ones.

use keanu_distributions::KeanuRng;
use keanu_graph::op::DistributionKind;
use keanu_graph::{
    DoubleVar, GraphBuilder, GraphError, Model, NodeId, Parameterisation, State, marginalize,
};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};

mod common;
use common::{assert_all_close_rel, assert_close_rel, s, t, v};

/// Evaluates every node of the graph (sampling none: every latent has a value).
fn evaluate(graph: &keanu_graph::Graph, state: &mut State) {
    let all: Vec<NodeId> = graph.ids().collect();
    state
        .lazy_eval(graph, &all, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
}

/// ln Σ eˣ.
fn log_sum_exp(xs: &[f64]) -> f64 {
    let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
}

/// Draws from 0.4 N(-2, 1) + 0.6 N(3, 0.7).
fn mixture_data(count: usize, seed: u64) -> Vec<f64> {
    let mut r = KeanuRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            if r.next_f64() < 0.4 {
                -2.0 + r.next_gaussian()
            } else {
                3.0 + 0.7 * r.next_gaussian()
            }
        })
        .collect()
}

/// The latents of a two-component mixture with ordered means, component scales and Dirichlet weights.
struct Mixture {
    first: NodeId,
    gap: NodeId,
    sigmas: NodeId,
    weights: NodeId,
}

/// The mixture, written with `Mixture` or with an explicit categorical assignment per point (labelled
/// "assignment"); its graph and state, and its latents.
fn mixture_model(data: &[f64], explicit: bool) -> (keanu_graph::Graph, State, Mixture, NodeId) {
    let b = GraphBuilder::new();
    let first = b.gaussian_with_shape(&[1], 0.0, 10.0);
    let gap = b.half_gaussian_with_shape(&[1], 10.0);
    let means = b.concat(0, &[first, gap]).cumsum(0).label("means");
    let sigmas = b.half_gaussian_with_shape(&[2], 5.0);
    let weights = b.dirichlet(DoubleTensor::vector(vec![2.0, 2.0]));
    let n = data.len();
    let (y, z): (DoubleVar, Option<NodeId>) = if explicit {
        let z = b
            .categorical(&[n], &[weights.take(&[0]), weights.take(&[1])])
            .label("assignment");
        b.set_value(z, IntTensor::vector(vec![0; n]));
        (
            b.gaussian_with_shape(&[n], means.gather(0, z), sigmas.gather(0, z)),
            Some(z.id()),
        )
    } else {
        (
            b.mixture_with_shape(
                &[n],
                DistributionKind::Gaussian,
                &[means.id(), sigmas.id()],
                weights,
            ),
            None,
        )
    };
    let y = y.label("y");
    b.observe(y, DoubleTensor::vector(data.to_vec()));
    b.set_value(first, v(&[0.0]));
    b.set_value(gap, v(&[1.0]));
    b.set_value(sigmas, v(&[1.0, 1.0]));
    b.set_value(weights, v(&[0.5, 0.5]));
    let latents = Mixture {
        first: first.id(),
        gap: gap.id(),
        sigmas: sigmas.id(),
        weights: weights.id(),
    };
    let y = y.id();
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state);
    (graph, state, latents, z.unwrap_or(y))
}

/// Sets the mixture's latents to random values and recomputes the graph.
fn set_random(model: &Model, state: &mut State, latents: &Mixture, r: &mut KeanuRng) -> Vec<f64> {
    let w = 0.1 + 0.8 * r.next_f64();
    let values = vec![
        3.0 * r.next_gaussian(),
        0.2 + 4.0 * r.next_f64(),
        0.3 + 2.0 * r.next_f64(),
        0.3 + 2.0 * r.next_f64(),
        w,
    ];
    let graph = model.graph();
    for (id, value) in [
        (latents.first, v(&values[..1])),
        (latents.gap, v(&values[1..2])),
        (latents.sigmas, v(&values[2..4])),
        (latents.weights, v(&[w, 1.0 - w])),
    ] {
        state.set_and_cascade(graph, id, value).unwrap();
    }
    values
}

/// AE1, the log-probability half: the mixture with explicit assignments, marginalised automatically, has the same
/// log probability as the `Mixture` model at random parameter values, and the same gradients and Hessian-vector
/// products in the unconstrained space NUTS samples (the categorical's normalisation, ln Σ w, is constant on the
/// simplex, so it only changes the constrained gradient with respect to the weights, along (1, 1)).
#[test]
fn marginalised_assignments_match_the_mixture_model() {
    let data = mixture_data(40, 3);
    let (explicit, explicit_state, explicit_latents, z) = mixture_model(&data, true);
    let (graph, state, marginalized) = marginalize(&explicit, &explicit_state, &[z]).unwrap();
    // the latents keep their labels and map to nodes of the same shape; the assignment and the observation's
    // density are gone
    let map = |id: NodeId| marginalized.node(id).unwrap();
    let latents = Mixture {
        first: map(explicit_latents.first),
        gap: map(explicit_latents.gap),
        sigmas: map(explicit_latents.sigmas),
        weights: map(explicit_latents.weights),
    };
    assert_eq!(marginalized.node(z), None);
    assert_eq!(graph.find("assignment"), None);
    assert_eq!(
        graph.find("means"),
        marginalized.node(explicit.find("means").unwrap())
    );
    let y = graph.find("y").unwrap();
    assert!(
        !graph[y].is_probabilistic(),
        "the observation is a constant"
    );
    let record = &marginalized.latents()[0];
    assert_eq!(record.original(), z);
    assert_eq!(record.label(), Some("assignment"));
    assert_eq!(record.values(), 0..=1);
    assert_eq!(graph[record.log_weights()].shape(), &[40, 2]);
    assert!(graph[record.factor()].is_factor());

    let model = Model::new(graph, &state).unwrap();
    assert_eq!(model.discrete_latents(), Vec::<NodeId>::new());
    assert_eq!(model.observed(), &[] as &[NodeId]);
    let (mixture, mut mixture_state, mixture_latents, _) = mixture_model(&data, false);
    let mixture = Model::new(mixture, &mixture_state).unwrap();
    let mut state = state;
    let mut r = KeanuRng::seed_from_u64(9);
    for trial in 0..5 {
        let mut r2 = r.clone();
        set_random(&model, &mut state, &latents, &mut r);
        set_random(&mixture, &mut mixture_state, &mixture_latents, &mut r2);
        let what = format!("trial {trial}");
        assert_close_rel(
            model.log_prob(&state).unwrap(),
            mixture.log_prob(&mixture_state).unwrap(),
            1e-11,
            &what,
        );
        // the constrained gradient with respect to the means and scales
        let g = model.log_prob_gradient(&state).unwrap();
        let h = mixture.log_prob_gradient(&mixture_state).unwrap();
        for (a, b) in [
            (latents.first, mixture_latents.first),
            (latents.gap, mixture_latents.gap),
            (latents.sigmas, mixture_latents.sigmas),
        ] {
            assert_all_close_rel(
                g.get(a).unwrap().as_slice(),
                h.get(b).unwrap().as_slice(),
                1e-10,
                &what,
            );
        }
        // the unconstrained gradient and Hessian-vector products
        let p = Parameterisation::new(&model, &state).unwrap();
        let q = Parameterisation::new(&mixture, &mixture_state).unwrap();
        let x = p.from_state(&state).unwrap();
        assert_all_close_rel(&x, &q.from_state(&mixture_state).unwrap(), 1e-12, &what);
        let (lp, gp) = p.log_prob_and_gradient(&x, true, &mut state).unwrap();
        let (lq, gq) = q
            .log_prob_and_gradient(&x, true, &mut mixture_state)
            .unwrap();
        assert_close_rel(lp, lq, 1e-11, &what);
        assert_all_close_rel(&gp.unwrap(), &gq.unwrap(), 1e-9, &what);
        let direction: Vec<f64> = (0..x.len()).map(|_| r.next_gaussian()).collect();
        let hp = p.hvp(&x, &direction, true, &mut state).unwrap();
        let hq = q.hvp(&x, &direction, true, &mut mixture_state).unwrap();
        assert_all_close_rel(&hp, &hq, 1e-8, &what);
    }
}

/// A Poisson change point: the counts' rate switches from `early` to `late` at a switch year drawn uniformly.
/// Returns the graph, state, switch and the two rates.
fn change_point(counts: &[i64]) -> (keanu_graph::Graph, State, NodeId, NodeId, NodeId) {
    let b = GraphBuilder::new();
    let years = counts.len() as i64;
    let switch = b.uniform_int(0, years).label("switch");
    let early = b.exponential(0.5).label("early");
    let late = b.exponential(0.5).label("late");
    let year = b.int_constant(IntTensor::vector((0..years).collect()));
    let rate = b.where_(year.lt(switch), early, late);
    let observed = b.poisson(rate);
    b.observe(observed, IntTensor::vector(counts.to_vec()));
    b.set_value(switch, IntTensor::scalar(3));
    b.set_value(early, s(2.0));
    b.set_value(late, s(1.0));
    let (switch, early, late) = (switch.id(), early.id(), late.id());
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state);
    (graph, state, switch, early, late)
}

/// A scalar discrete-uniform change point, marginalised, has the log probability of brute-force enumeration over the
/// switch year, and its conditional probabilities are the enumeration's normalised terms.
#[test]
fn a_change_point_marginalises_to_the_enumerated_sum() {
    let counts = [4, 5, 3, 6, 4, 1, 0, 2, 1, 1, 0, 2];
    let (graph, state, switch, early, late) = change_point(&counts);
    let original = Model::new(graph.clone(), &state).unwrap();
    let (new_graph, new_state, marginalized) = marginalize(&graph, &state, &[switch]).unwrap();
    let model = Model::new(new_graph, &new_state).unwrap();
    let mut state = state;
    let mut new_state = new_state;
    let mut r = KeanuRng::seed_from_u64(4);
    for _ in 0..4 {
        let (e, l) = (0.2 + 6.0 * r.next_f64(), 0.2 + 6.0 * r.next_f64());
        for (g, st, a, b) in [
            (&graph, &mut state, early, late),
            (
                model.graph().as_ref(),
                &mut new_state,
                marginalized.node(early).unwrap(),
                marginalized.node(late).unwrap(),
            ),
        ] {
            st.set_and_cascade(g, a, s(e)).unwrap();
            st.set_and_cascade(g, b, s(l)).unwrap();
        }
        let terms: Vec<f64> = (0..counts.len() as i64)
            .map(|year| {
                state
                    .set_and_cascade(&graph, switch, IntTensor::scalar(year))
                    .unwrap();
                original.log_prob(&state).unwrap()
            })
            .collect();
        let total = log_sum_exp(&terms);
        assert_close_rel(
            model.log_prob(&new_state).unwrap(),
            total,
            1e-12,
            "log prob",
        );
        let p = marginalized.latents()[0].probabilities(&new_state).unwrap();
        let expected: Vec<f64> = terms.iter().map(|t| (t - total).exp()).collect();
        assert_all_close_rel(p.as_slice(), &expected, 1e-10, "conditional probabilities");
        assert_close_rel(p.as_slice().iter().sum(), 1.0, 1e-12, "sum");

        // the gradient against finite differences
        let gradient = model.log_prob_gradient(&new_state).unwrap();
        for id in [early, late] {
            let id = marginalized.node(id).unwrap();
            let h = 1e-6;
            let base = new_state.f64(id).unwrap().as_slice()[0];
            let mut at = |x: f64| {
                new_state.set_and_cascade(model.graph(), id, s(x)).unwrap();
                model.log_prob(&new_state).unwrap()
            };
            let fd = (at(base + h) - at(base - h)) / (2.0 * h);
            at(base);
            assert_close_rel(
                gradient.get(id).unwrap().as_slice()[0],
                fd,
                1e-6,
                "gradient",
            );
        }
    }
}

/// An occupancy model: sites occupied with probability ψ, each visit detecting the species at an occupied site with
/// probability pⱼ. The occupancy is a vector Bernoulli latent [N] and the detections an [N, J] Bernoulli child,
/// summed over its trailing (visit) axis.
#[test]
fn vector_bernoulli_occupancy_sums_the_trailing_axes() {
    let (n, j) = (6, 4);
    #[rustfmt::skip]
    let detections = [
        true, false, true, false,
        false, false, false, false,
        false, true, false, false,
        false, false, false, false,
        true, true, true, false,
        false, false, false, false,
    ];
    let b = GraphBuilder::new();
    let psi = b.beta(2.0, 2.0).label("psi");
    let p = b.beta_with_shape(&[j], 2.0, 2.0).label("p");
    let occupied = b.bernoulli_with_shape(&[n], psi).label("occupied");
    let detect = occupied.to_double_mask().reshape(&[n as isize, 1]) * p;
    let y = b.bernoulli_with_shape(&[n, j], detect).label("y");
    b.observe(y, BoolTensor::new(&[n, j], detections.to_vec()).unwrap());
    b.set_value(psi, s(0.6));
    b.set_value(p, v(&[0.3, 0.5, 0.7, 0.4]));
    b.set_value(occupied, BoolTensor::full(&[n], true));
    let occupied = occupied.id();
    let (graph, state) = b.build().unwrap();
    let (graph, state, marginalized) = marginalize(&graph, &state, &[occupied]).unwrap();
    let model = Model::new(graph, &state).unwrap();

    let (psi_value, p_values) = (0.6, [0.3, 0.5, 0.7, 0.4]);
    let beta22 = |x: f64| (6.0 * x * (1.0 - x)).ln();
    let mut expected = beta22(psi_value) + p_values.iter().map(|&x| beta22(x)).sum::<f64>();
    let mut posterior = Vec::new();
    for site in 0..n {
        let row = &detections[site * j..(site + 1) * j];
        let seen: f64 = row
            .iter()
            .zip(&p_values)
            .map(|(&d, &q): (&bool, &f64)| if d { q.ln() } else { (1.0 - q).ln() })
            .sum();
        let occupied_term = psi_value * seen.exp();
        let empty_term = if row.iter().any(|&d| d) {
            0.0
        } else {
            1.0 - psi_value
        };
        expected += (occupied_term + empty_term).ln();
        posterior.push(occupied_term / (occupied_term + empty_term));
    }
    assert_close_rel(model.log_prob(&state).unwrap(), expected, 1e-12, "log prob");
    let record = &marginalized.latents()[0];
    assert_eq!(record.dtype(), keanu_graph::DType::Bool);
    let probabilities = record.probabilities(&state).unwrap();
    assert_eq!(probabilities.shape(), &[n, 2]);
    for (site, &occupied) in posterior.iter().enumerate() {
        let row = &probabilities.as_slice()[2 * site..2 * site + 2];
        assert_close_rel(row[1], occupied, 1e-12, &format!("site {site}"));
        assert_close_rel(row[0] + row[1], 1.0, 1e-12, &format!("site {site} sum"));
    }
    // a draw of the occupancy is true wherever the species was detected
    let draw = record
        .draw(&probabilities, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let draw = draw.as_bool().unwrap().as_slice().to_vec();
    for site in [0, 2, 4] {
        assert!(draw[site], "site {site}");
    }
    // NUTS-ready: every latent is continuous, with gradients
    assert_eq!(model.latents().len(), 2);
    model.log_prob_gradient(&state).unwrap();
}

/// A vector latent whose children are reached through a matrix product, which combines its elements, is rejected
/// with an error naming the product.
#[test]
fn a_child_reached_through_a_matmul_is_rejected() {
    let b = GraphBuilder::new();
    let z = b
        .categorical(&[3], &[b.scalar(0.5), b.scalar(0.5)])
        .label("z");
    let design = b.constant(t(&[3, 3], &[1.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0, 1.0]));
    let mixed = design.matmul(z.to_double().reshape(&[3, 1])).reshape(&[3]);
    let y = b.gaussian(mixed, 1.0);
    b.observe(y, v(&[0.0, 1.0, 2.0]));
    let z = z.id();
    let (graph, state) = b.build().unwrap();
    let error = marginalize(&graph, &state, &[z]).unwrap_err();
    let GraphError::CannotMarginalize { latent, node, .. } = &error else {
        panic!("{error}");
    };
    assert_eq!(latent, "\"z\"");
    assert!(node.contains("(matmul)"), "{error}");
    assert!(
        error.to_string().starts_with("cannot marginalise \"z\": #"),
        "{error}"
    );
}

/// The patterns outside the supported ones are rejected, naming the node that blocks the rewrite.
#[test]
fn unsupported_patterns_are_rejected_naming_the_blocking_node() {
    let blocked = |graph: &keanu_graph::Graph, state: &State, z: NodeId, expected: &str| {
        let error = marginalize(graph, state, &[z]).unwrap_err();
        let GraphError::CannotMarginalize { node, .. } = &error else {
            panic!("{error}");
        };
        assert!(node.contains(expected), "expected {expected:?} in {error}");
    };

    // a continuous latent
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0).label("x");
    let y = b.gaussian(x, 1.0);
    b.observe(y, s(0.5));
    let x = x.id();
    let (graph, state) = b.build().unwrap();
    blocked(&graph, &state, x, "\"x\" (gaussian)");

    // an observed latent, a latent child, a factor, and a child broadcast across the latent's axes
    let b = GraphBuilder::new();
    let z = b.bernoulli_with_shape(&[3], 0.5).label("z");
    let observed_z = b.bernoulli(0.5).label("observed");
    b.observe(observed_z, BoolTensor::scalar(true));
    let mean = z.to_double_mask();
    let latent_child = b.gaussian(mean, 1.0).label("latent child");
    let _ = latent_child;
    let (z, observed_z) = (z.id(), observed_z.id());
    let (graph, state) = b.build().unwrap();
    blocked(&graph, &state, observed_z, "\"observed\"");
    blocked(&graph, &state, z, "\"latent child\" (gaussian)");

    let b = GraphBuilder::new();
    let z = b.bernoulli_with_shape(&[3], 0.5).label("z");
    b.factor(z.to_double_mask().sum()).label("penalty");
    let z = z.id();
    let (graph, state) = b.build().unwrap();
    blocked(&graph, &state, z, "\"penalty\" (factor)");

    let b = GraphBuilder::new();
    let z = b.bernoulli_with_shape(&[3], 0.5).label("z");
    let y = b
        .gaussian_with_shape(&[2, 3], z.to_double_mask(), 1.0)
        .label("y");
    b.observe(y, DoubleTensor::zeros(&[2, 3]));
    let z = z.id();
    let (graph, state) = b.build().unwrap();
    blocked(&graph, &state, z, "\"y\" (gaussian)");

    // a sum over the latent's own axis
    let b = GraphBuilder::new();
    let z = b.bernoulli_with_shape(&[3], 0.5).label("z");
    let y = b.gaussian(z.to_double_mask().sum().label("total"), 1.0);
    b.observe(y, s(1.0));
    let z = z.id();
    let (graph, state) = b.build().unwrap();
    blocked(&graph, &state, z, "\"total\" (reduce)");

    // a support set by another latent
    let b = GraphBuilder::new();
    let top = b.poisson(3.0).label("top");
    let z = b.uniform_int(b.int_scalar(0), top + 1).label("z");
    let y = b.gaussian(z.to_double(), 1.0);
    b.observe(y, s(1.0));
    let z = z.id();
    let (graph, state) = b.build().unwrap();
    blocked(&graph, &state, z, "\"top\" (poisson)");

    // a count with infinitely many values
    let b = GraphBuilder::new();
    let z = b.poisson(3.0).label("z");
    let y = b.gaussian(z.to_double(), 1.0);
    b.observe(y, s(1.0));
    let z = z.id();
    let (graph, state) = b.build().unwrap();
    let error = marginalize(&graph, &state, &[z]).unwrap_err();
    assert!(error.to_string().contains("infinitely many"), "{error}");
}

/// Two coins, each with its own tosses: marginalised one after another, the model's log probability is the product
/// of the two marginal likelihoods, the latents' records follow the later rewrite, and nodes that depend on a
/// latent but lead to no observation are dropped.
#[test]
fn several_latents_are_marginalised_one_after_another() {
    let b = GraphBuilder::new();
    let bias = b.uniform(0.5, 1.0).label("bias");
    let coins: Vec<NodeId> = (0..2)
        .map(|c| {
            let biased = b.bernoulli(0.3).label(&format!("biased {c}"));
            let p = b.where_(biased, bias, b.scalar(0.5));
            let tosses = b
                .bernoulli_with_shape(&[3], p)
                .label(&format!("tosses {c}"));
            b.observe(tosses, BoolTensor::vector(vec![true, c == 0, true]));
            // a deterministic node that depends on the coin but leads nowhere
            let _ = biased.to_double_mask().label(&format!("indicator {c}"));
            biased.id()
        })
        .collect();
    b.set_value(bias, s(0.8));
    let bias = bias.id();
    let (graph, state) = b.build().unwrap();
    let (new_graph, new_state, marginalized) = marginalize(&graph, &state, &coins).unwrap();
    for c in 0..2 {
        assert_eq!(new_graph.find(&format!("indicator {c}")), None);
        assert_eq!(new_graph.find(&format!("biased {c}")), None);
        let tosses = new_graph.find(&format!("tosses {c}")).unwrap();
        assert_eq!(
            Some(tosses),
            marginalized.node(graph.find(&format!("tosses {c}")).unwrap())
        );
    }
    let model = Model::new(new_graph, &new_state).unwrap();
    assert_eq!(model.latents(), &[marginalized.node(bias).unwrap()]);
    let marginal = |heads: i32| {
        let biased = 0.3 * 0.8f64.powi(heads) * 0.2f64.powi(3 - heads);
        let fair = 0.7 * 0.5f64.powi(3);
        (biased, fair)
    };
    let (b0, f0) = marginal(3);
    let (b1, f1) = marginal(2);
    let expected = 2f64.ln() + (b0 + f0).ln() + (b1 + f1).ln();
    assert_close_rel(
        model.log_prob(&new_state).unwrap(),
        expected,
        1e-12,
        "log prob",
    );
    for (record, (b, f)) in marginalized.latents().iter().zip([(b0, f0), (b1, f1)]) {
        assert!(model.factors().contains(&record.factor()));
        let p = record.probabilities(&new_state).unwrap();
        assert_close_rel(p.as_slice()[1], b / (b + f), 1e-12, record.name());
    }
    // repeating a latent is an error
    assert!(marginalize(&graph, &state, &[coins[0], coins[0]]).is_err());
}
