//! The parameterisation seam: the flat unconstrained layout of a model's continuous latents, mapping it to and from
//! a state, and the log probability and its gradient in unconstrained space with or without the Jacobian.

use keanu_distributions::KeanuRng;
use keanu_graph::{GraphBuilder, GraphError, Model, NodeId, Parameterisation, Space, State};
use keanu_tensor::DoubleTensor;

mod common;
use common::{assert_close_rel, evaluate, v};

fn model_of(build: impl FnOnce(&GraphBuilder)) -> (Model, State) {
    let b = GraphBuilder::new();
    build(&b);
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 11);
    let model = Model::new(graph, &state).unwrap();
    (model, state)
}

fn logistic(y: f64) -> f64 {
    1.0 / (1.0 + (-y).exp())
}

/// Gaussian, HalfGaussian, Beta (with a latent-dependent parameter) and Dirichlet latents, all observed through
/// likelihood terms so every coordinate has a non-trivial gradient.
fn mixed_model() -> (Model, State) {
    model_of(|b| {
        let mu = b.gaussian(0.0, 1.0).label("mu");
        let sigma = b.half_gaussian(2.0).label("sigma");
        let p = b.beta(sigma + 1.0, 3.0).label("p");
        let w = b.dirichlet(v(&[1.5, 2.0, 3.0])).label("w");
        let y = b.gaussian(mu, sigma);
        b.observe(y, DoubleTensor::scalar(0.7));
        let z = b.gaussian(mu * p, 0.5);
        b.observe(z, DoubleTensor::scalar(-0.2));
        let u = b.gaussian(w * 2.0, sigma);
        b.observe(u, v(&[0.4, 0.5, 1.1]));
    })
}

#[test]
fn without_the_jacobian_the_log_prob_is_the_constrained_log_prob_at_forward_z() {
    let (model, mut state) = mixed_model();
    let param = Parameterisation::new(&model, &state).unwrap();
    // mu (1) + sigma (1) + p (1) + w (2 free coordinates)
    assert_eq!(param.dimension(), 5);
    let z = [0.3, -0.4, 1.2, 0.5, -0.8];
    let (log_prob, gradient) = param.log_prob_and_gradient(&z, false, &mut state).unwrap();
    assert!(gradient.is_some());
    // log_prob_and_gradient leaves the state at forward(z)
    assert_eq!(log_prob, model.log_prob(&state).unwrap());

    let mu = model.find("mu").unwrap();
    let sigma = model.find("sigma").unwrap();
    let p = model.find("p").unwrap();
    let w = model.find("w").unwrap();
    assert_eq!(state.f64(mu).unwrap().as_slice(), &[0.3]);
    assert_close_rel(
        state.f64(sigma).unwrap().as_slice()[0],
        (-0.4f64).exp(),
        1e-15,
        "sigma",
    );
    assert_close_rel(
        state.f64(p).unwrap().as_slice()[0],
        logistic(1.2),
        1e-15,
        "p",
    );
    let w_value = state.f64(w).unwrap().as_slice();
    assert_close_rel(w_value.iter().sum::<f64>(), 1.0, 1e-14, "simplex sum");
    assert!(w_value.iter().all(|&x| x > 0.0));
}

#[test]
fn with_the_jacobian_it_adds_the_log_determinant_for_gamma_beta_and_dirichlet() {
    let (model, mut state) = model_of(|b| {
        b.gamma(1.5, 2.0).label("g");
        b.beta(2.0, 3.0).label("b");
        b.dirichlet(v(&[1.0, 2.0, 3.0, 4.0])).label("d");
    });
    let param = Parameterisation::new(&model, &state).unwrap();
    assert_eq!(param.dimension(), 5);
    let z = [0.7, -0.3, 0.2, -1.1, 0.6];
    let (without, _) = param.log_prob_and_gradient(&z, false, &mut state).unwrap();
    let (with, _) = param.log_prob_and_gradient(&z, true, &mut state).unwrap();

    // Gamma: x = exp(z), log J = z
    let gamma_jacobian = z[0];
    // Beta: x = logistic(z), log J = log x + log(1 - x)
    let x = logistic(z[1]);
    let beta_jacobian = x.ln() + (1.0 - x).ln();
    // Dirichlet: log J = Σ log xᵢ + ½ log K, with x the simplex value in the state
    let d = state.f64(model.find("d").unwrap()).unwrap();
    let dirichlet_jacobian = d.as_slice().iter().map(|x| x.ln()).sum::<f64>() + 0.5 * 4f64.ln();

    let expected = without + gamma_jacobian + beta_jacobian + dirichlet_jacobian;
    assert_close_rel(with, expected, 1e-12, "log prob with the Jacobian");
    assert_close_rel(
        param.log_det_jacobian(&z).unwrap(),
        gamma_jacobian + beta_jacobian + dirichlet_jacobian,
        1e-12,
        "log det Jacobian",
    );
}

fn finite_difference_gradient(
    param: &Parameterisation<'_>,
    state: &mut State,
    z: &[f64],
    jacobian: bool,
) -> Vec<f64> {
    let h = 1e-6;
    (0..z.len())
        .map(|i| {
            let mut up = z.to_vec();
            let mut down = z.to_vec();
            up[i] += h;
            down[i] -= h;
            let (f_up, _) = param.log_prob_and_gradient(&up, jacobian, state).unwrap();
            let (f_down, _) = param.log_prob_and_gradient(&down, jacobian, state).unwrap();
            (f_up - f_down) / (2.0 * h)
        })
        .collect()
}

#[test]
fn the_gradient_matches_finite_differences_in_z() {
    let (model, mut state) = mixed_model();
    let param = Parameterisation::new(&model, &state).unwrap();
    let mut rng = KeanuRng::seed_from_u64(5);
    for _ in 0..3 {
        let z = param.uniform_initial_point(&mut rng);
        for jacobian in [false, true] {
            let (_, gradient) = param
                .log_prob_and_gradient(&z, jacobian, &mut state)
                .unwrap();
            let gradient = gradient.unwrap();
            let numeric = finite_difference_gradient(&param, &mut state, &z, jacobian);
            assert_eq!(gradient.len(), numeric.len());
            for (i, (a, n)) in gradient.iter().zip(&numeric).enumerate() {
                assert_close_rel(
                    *a,
                    *n,
                    1e-6,
                    &format!("coordinate {i}, jacobian {jacobian}"),
                );
            }
        }
    }
}

#[test]
fn from_state_then_to_state_leaves_the_state_unchanged() {
    let (model, mut state) = mixed_model();
    let param = Parameterisation::new(&model, &state).unwrap();
    let before = state.clone();
    let z = param.from_state(&state).unwrap();
    assert_eq!(z.len(), param.dimension());
    param.to_state(&z, &mut state).unwrap();
    for id in model.graph().ids() {
        match (before.f64(id), state.f64(id)) {
            (Some(a), Some(b)) => {
                for (x, y) in a.as_slice().iter().zip(b.as_slice()) {
                    assert_close_rel(*y, *x, 1e-12, &format!("node {}", id.index()));
                }
            }
            (a, b) => assert_eq!(a.is_some(), b.is_some()),
        }
    }
    // and to_state then from_state recovers z
    let z2 = param.from_state(&state).unwrap();
    for (a, b) in z.iter().zip(&z2) {
        assert_close_rel(*b, *a, 1e-10, "z round trip");
    }
}

#[test]
fn from_state_rejects_values_outside_or_on_the_boundary_of_the_support() {
    for value in [5.0, 1.0, -0.5] {
        let b = GraphBuilder::new();
        let x = b.uniform(0.0, 1.0).label("x").id();
        let (graph, mut state) = b.build().unwrap();
        state.set(&graph, x, DoubleTensor::scalar(value)).unwrap();
        let model = Model::new(graph, &state).unwrap();
        let param = Parameterisation::new(&model, &state).unwrap();
        let error = param.from_state(&state).unwrap_err();
        assert!(matches!(error, GraphError::Transform(_)), "{error:?}");
        assert!(error.to_string().contains("\"x\""), "{error}");
    }
    let b = GraphBuilder::new();
    let s = b.half_gaussian(1.0).id();
    let (graph, mut state) = b.build().unwrap();
    state.set(&graph, s, DoubleTensor::scalar(0.0)).unwrap();
    let model = Model::new(graph, &state).unwrap();
    let param = Parameterisation::new(&model, &state).unwrap();
    assert!(param.from_state(&state).is_err());
}

#[test]
fn constrained_mode_reproduces_the_current_values_and_gradients() {
    let (model, mut state) = mixed_model();
    let param = Parameterisation::with_space(&model, &state, Space::Constrained).unwrap();
    assert_eq!(param.space(), Space::Constrained);
    assert_eq!(param.latents(), model.continuous_latents().as_slice());

    // the flat layout is every continuous latent's elements, in latent order, each row-major
    let expected: Vec<f64> = model
        .continuous_latents()
        .iter()
        .flat_map(|&id| state.f64(id).unwrap().as_slice().to_vec())
        .collect();
    assert_eq!(param.dimension(), expected.len());
    let x = param.from_state(&state).unwrap();
    assert_eq!(x, expected);

    // log prob and gradient are exactly the model's, with or without the (zero) Jacobian
    let (model_log_prob, model_gradient) = model.log_prob_and_gradient(&state).unwrap();
    let model_gradient = model_gradient.unwrap();
    let expected_gradient: Vec<f64> = model
        .continuous_latents()
        .iter()
        .flat_map(|&id| model_gradient.get(id).unwrap().as_slice().to_vec())
        .collect();
    for jacobian in [false, true] {
        let (log_prob, gradient) = param
            .log_prob_and_gradient(&x, jacobian, &mut state)
            .unwrap();
        assert_eq!(log_prob, model_log_prob);
        assert_eq!(gradient.unwrap(), expected_gradient);
    }
    assert_eq!(param.log_det_jacobian(&x).unwrap(), 0.0);

    // to_state sets the values as given, even outside the support, as the algorithms do today
    let mut moved = x.clone();
    moved[1] = -1.0; // sigma
    param.to_state(&moved, &mut state).unwrap();
    assert_eq!(
        state.f64(model.find("sigma").unwrap()).unwrap().as_slice(),
        &[-1.0]
    );
    assert_eq!(param.from_state(&state).unwrap(), moved);
    let (log_prob, gradient) = param
        .log_prob_and_gradient(&moved, true, &mut state)
        .unwrap();
    assert_eq!(log_prob, f64::NEG_INFINITY);
    assert!(gradient.is_none());
}

#[test]
fn a_uniform_bounded_by_another_latent_is_rejected_with_a_reparameterisation_hint() {
    let b = GraphBuilder::new();
    let lo = b.gaussian(0.0, 1.0).label("lo");
    b.uniform(lo, 5.0).label("x");
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 11);
    let model = Model::new(graph, &state).unwrap();
    let error = Parameterisation::new(&model, &state).unwrap_err();
    assert!(matches!(error, GraphError::Transform(_)), "{error:?}");
    let message = error.to_string();
    assert!(message.contains("\"x\""), "{message}");
    assert!(message.contains("\"lo\""), "{message}");
    assert!(message.contains("reparameteris"), "{message}");

    // through a deterministic node too
    let b = GraphBuilder::new();
    let lo = b.gaussian(0.0, 1.0).label("lo");
    b.uniform(-1.0, lo.exp() + 1.0).label("x");
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 11);
    let model = Model::new(graph, &state).unwrap();
    assert!(Parameterisation::new(&model, &state).is_err());

    // constrained mode needs no transforms, so it accepts the model
    assert!(Parameterisation::with_space(&model, &state, Space::Constrained).is_ok());
}

#[test]
fn latent_parameters_that_do_not_move_the_support_are_accepted() {
    // a hierarchical scale: the Gamma's support (x > 0) doesn't depend on its latent scale
    let (model, state) = model_of(|b| {
        let scale = b.half_gaussian(1.0);
        b.gamma(scale, 2.0);
    });
    assert_eq!(
        Parameterisation::new(&model, &state).unwrap().dimension(),
        2
    );
}

#[test]
fn parameter_dependent_bounds_are_fixed_at_their_values_in_the_state() {
    let (model, mut state) = model_of(|b| {
        let lo = b.constant(DoubleTensor::scalar(1.0)) * 2.0;
        b.uniform(lo, 6.0).label("x");
    });
    let param = Parameterisation::new(&model, &state).unwrap();
    for z in [-30.0, 0.0, 30.0] {
        param.to_state(&[z], &mut state).unwrap();
        let x = state.f64(model.find("x").unwrap()).unwrap().as_slice()[0];
        assert!((2.0..=6.0).contains(&x), "{x}");
    }
    param.to_state(&[0.0], &mut state).unwrap();
    assert_eq!(
        state.f64(model.find("x").unwrap()).unwrap().as_slice(),
        &[4.0]
    );
}

#[test]
fn a_model_without_continuous_latents_has_dimension_zero() {
    let (model, mut state) = model_of(|b| {
        let x = b.gaussian(0.0, 1.0);
        b.observe(x, DoubleTensor::scalar(0.3));
    });
    let param = Parameterisation::new(&model, &state).unwrap();
    assert_eq!(param.dimension(), 0);
    assert!(param.from_state(&state).unwrap().is_empty());
    param.to_state(&[], &mut state).unwrap();
    for jacobian in [false, true] {
        let (log_prob, gradient) = param
            .log_prob_and_gradient(&[], jacobian, &mut state)
            .unwrap();
        assert_eq!(log_prob, model.log_prob(&state).unwrap());
        assert_eq!(gradient.unwrap(), Vec::<f64>::new());
    }
    assert!(
        param
            .uniform_initial_point(&mut KeanuRng::seed_from_u64(1))
            .is_empty()
    );
}

#[test]
fn a_point_of_the_wrong_length_is_an_error() {
    let (model, mut state) = mixed_model();
    let param = Parameterisation::new(&model, &state).unwrap();
    assert!(param.to_state(&[0.0; 4], &mut state).is_err());
    assert!(
        param
            .log_prob_and_gradient(&[0.0; 6], true, &mut state)
            .is_err()
    );
    assert!(param.log_det_jacobian(&[]).is_err());
    assert!(param.to_constrained(&[0.0; 2]).is_err());
}

#[test]
fn the_uniform_initial_point_is_in_the_stan_interval_and_reproducible() {
    let (model, state) = mixed_model();
    let param = Parameterisation::new(&model, &state).unwrap();
    let a = param.uniform_initial_point(&mut KeanuRng::seed_from_u64(9));
    let b = param.uniform_initial_point(&mut KeanuRng::seed_from_u64(9));
    assert_eq!(a, b);
    assert_eq!(a.len(), param.dimension());
    assert!(a.iter().all(|z| (-2.0..2.0).contains(z)));
}

#[test]
fn the_layout_names_the_latent_of_each_coordinate() {
    let (model, state) = mixed_model();
    let param = Parameterisation::new(&model, &state).unwrap();
    let w = model.find("w").unwrap();
    let mu = model.find("mu").unwrap();
    assert_eq!(param.coordinate_latent(0), Some(mu));
    assert_eq!(param.coordinate_latent(3), Some(w));
    assert_eq!(param.coordinate_latent(4), Some(w));
    assert_eq!(param.coordinate_latent(5), None);
    assert_eq!(param.range(w), Some(3..5));
    assert_eq!(param.range(NodeId::from_index(1000)), None);
}

#[test]
fn the_likelihood_gradient_pulls_back_through_the_bijectors() {
    let (model, mut state) = mixed_model();
    let param = Parameterisation::new(&model, &state).unwrap();
    let z = [0.1, 0.2, -0.3, 0.4, 0.5];
    param.to_state(&z, &mut state).unwrap();
    let gradient = param
        .pull_back(&z, &model.log_likelihood_gradient(&state).unwrap())
        .unwrap();
    let h = 1e-6;
    for i in 0..z.len() {
        let mut up = z;
        let mut down = z;
        up[i] += h;
        down[i] -= h;
        param.to_state(&up, &mut state).unwrap();
        let f_up = model.log_likelihood(&state).unwrap();
        param.to_state(&down, &mut state).unwrap();
        let f_down = model.log_likelihood(&state).unwrap();
        assert_close_rel(gradient[i], (f_up - f_down) / (2.0 * h), 1e-6, "likelihood");
    }
}

#[test]
fn latent_mixture_weights_do_not_move_the_support() {
    // a latent two-component Gaussian mixture whose weights are a latent Dirichlet: perturbing the weights takes
    // them off the simplex, which must not count as moving the (weight-independent) support
    let (model, state) = model_of(|b| {
        let w = b.dirichlet(v(&[1.0, 1.0])).label("w");
        let (mu, sigma) = (b.constant(v(&[-2.0, 2.0])), b.scalar(1.0));
        let x: keanu_graph::DoubleVar = b.mixture(
            keanu_graph::op::DistributionKind::Gaussian,
            &[mu.id(), sigma.id()],
            w,
        );
        let y = b.gaussian(x, 0.5);
        b.observe(y, DoubleTensor::scalar(1.5));
    });
    assert_eq!(
        Parameterisation::new(&model, &state).unwrap().dimension(),
        2
    );
    // a hidden Markov model's chain probabilities don't move its support either
    let (model, state) = model_of(|b| {
        let initial = b.dirichlet(v(&[1.0, 1.0]));
        let transition = b.constant(DoubleTensor::new(&[2, 2], vec![0.9, 0.1, 0.2, 0.8]).unwrap());
        let (mu, sigma) = (b.constant(v(&[-1.0, 1.0])), b.scalar(1.0));
        let x: keanu_graph::DoubleVar = b.hidden_markov_model(
            3,
            keanu_graph::op::DistributionKind::Gaussian,
            &[mu.id(), sigma.id()],
            initial,
            transition,
        );
        let y = b.gaussian(x, 0.5);
        b.observe(y, v(&[0.3, -0.2, 1.1]));
    });
    assert_eq!(
        Parameterisation::new(&model, &state).unwrap().dimension(),
        4
    );
}

/// A block of the continuous latents: the layout holds only them, in the order given, the log probability is the
/// joint one, and the gradient is the full gradient's entries for the block.
#[test]
fn a_block_of_latents_has_the_joint_log_prob_and_its_partial_gradient() {
    let b = GraphBuilder::new();
    let k = b.uniform_int(0, 3).label("k");
    let mu = b.gaussian(0.0, 1.0).label("mu");
    let sigma = b.half_gaussian(2.0).label("sigma");
    let y = b.gaussian_with_shape(&[2], mu + k.to_double(), sigma);
    b.observe(y, v(&[0.4, 1.3]));
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 3);
    let model = Model::new(graph, &state).unwrap();
    let (mu, sigma) = (model.find("mu").unwrap(), model.find("sigma").unwrap());
    let k = model.find("k").unwrap();

    let block =
        Parameterisation::for_latents(&model, &state, &[sigma, mu], Space::Unconstrained).unwrap();
    assert_eq!(block.latents(), [sigma, mu]);
    assert_eq!(block.dimension(), 2);
    let z = [0.3, -0.2];
    let (log_prob, gradient) = block.log_prob_and_gradient(&z, false, &mut state).unwrap();
    assert_close_rel(log_prob, model.log_prob(&state).unwrap(), 1e-12, "log prob");
    let full = model.log_prob_gradient(&state).unwrap();
    let gradient = gradient.unwrap();
    // sigma = exp(z₀), so ∂/∂z₀ = sigma ∂/∂sigma
    let d_sigma = full.get(sigma).unwrap().to_scalar().unwrap() * 0.3f64.exp();
    assert_close_rel(gradient[0], d_sigma, 1e-12, "sigma");
    assert_close_rel(
        gradient[1],
        full.get(mu).unwrap().to_scalar().unwrap(),
        1e-12,
        "mu",
    );
    let (_, partial) = model.log_prob_and_gradient_wrt(&state, &[mu]).unwrap();
    let mu_gradient = partial.unwrap().get(mu).unwrap().to_scalar().unwrap();
    assert_close_rel(mu_gradient, gradient[1], 1e-12, "mu alone");

    // not continuous latents, repeats and an empty block are rejected
    for latents in [vec![k], vec![mu, mu], vec![]] {
        assert!(matches!(
            Parameterisation::for_latents(&model, &state, &latents, Space::Unconstrained),
            Err(GraphError::InvalidValue(_))
        ));
    }
    assert!(model.log_prob_and_gradient_wrt(&state, &[k]).is_err());
}

/// mu ~ N(0, 1), sigma ~ HalfGaussian(2), y ~ N(mu, sigma) observed (three values), and a factor −mu²·sigma / 2.
/// Returns the model, its state, and (mu, sigma, y, factor).
fn weighted_model() -> (Model, State, [NodeId; 4]) {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0);
    let sigma = b.half_gaussian(2.0);
    let y = b.gaussian_with_shape(&[3], mu, sigma);
    b.observe(y, v(&[0.4, -1.2, 2.0]));
    let factor = b.factor(mu * mu * sigma * -0.5);
    let ids = [mu.id(), sigma.id(), y.id(), factor.id()];
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 3);
    let model = Model::new(graph, &state).unwrap();
    (model, state, ids)
}

#[test]
fn weights_scale_each_term_and_its_derivatives() {
    let (model, mut state, [mu, sigma, y, factor]) = weighted_model();
    let plain = Parameterisation::new(&model, &state).unwrap();
    let weighted = Parameterisation::new(&model, &state)
        .unwrap()
        .with_weights([(y, 7.5), (factor, 2.0), (mu, 0.5)])
        .unwrap();
    assert_eq!(weighted.weight(y), 7.5);
    assert_eq!(weighted.weight(sigma), 1.0);
    let z = [0.3, -0.4];
    for jacobian in [false, true] {
        // the value: each term times its weight
        plain.to_state(&z, &mut state).unwrap();
        let graph = model.graph();
        let term = |id| keanu_graph::node_log_prob(graph, &state, id).unwrap();
        let (lp_mu, lp_y, lp_f) = (term(mu), term(y), term(factor));
        let unweighted = plain.log_prob(&z, jacobian, &mut state).unwrap();
        let value = weighted.log_prob(&z, jacobian, &mut state).unwrap();
        assert_close_rel(
            value,
            unweighted + 6.5 * lp_y + 1.0 * lp_f - 0.5 * lp_mu,
            1e-12,
            "weighted log prob",
        );
        let (with_gradient, gradient) = weighted
            .log_prob_and_gradient(&z, jacobian, &mut state)
            .unwrap();
        assert_close_rel(with_gradient, value, 1e-12, "with the gradient");
        let gradient = gradient.unwrap();
        // the gradient against finite differences of the weighted value
        let step = 1e-6;
        for i in 0..2 {
            let mut plus = z;
            plus[i] += step;
            let mut minus = z;
            minus[i] -= step;
            let fd = (weighted.log_prob(&plus, jacobian, &mut state).unwrap()
                - weighted.log_prob(&minus, jacobian, &mut state).unwrap())
                / (2.0 * step);
            assert!((gradient[i] - fd).abs() < 1e-5, "{} vs {fd}", gradient[i]);
        }
        // the Hessian against finite differences of the weighted gradient
        let hessian = weighted.hessian(&z, jacobian, &mut state).unwrap();
        let direction = [0.7, -1.3];
        let hvp = weighted.hvp(&z, &direction, jacobian, &mut state).unwrap();
        let grad_at = |z: &[f64], state: &mut State| {
            weighted
                .log_prob_and_gradient(z, jacobian, state)
                .unwrap()
                .1
                .unwrap()
        };
        let plus: Vec<f64> = z
            .iter()
            .zip(&direction)
            .map(|(z, d)| z + step * d)
            .collect();
        let minus: Vec<f64> = z
            .iter()
            .zip(&direction)
            .map(|(z, d)| z - step * d)
            .collect();
        let (gp, gm) = (grad_at(&plus, &mut state), grad_at(&minus, &mut state));
        for i in 0..2 {
            let fd = (gp[i] - gm[i]) / (2.0 * step);
            assert!((hvp[i] - fd).abs() < 1e-5, "{} vs {fd}", hvp[i]);
            let row: f64 = (0..2)
                .map(|j| hessian.as_slice()[i * 2 + j] * direction[j])
                .sum();
            assert_close_rel(row, hvp[i], 1e-10, "Hessian row");
        }
    }
}

#[test]
fn an_impossible_term_stays_impossible_whatever_its_weight() {
    let (model, mut state, [_, sigma, y, _]) = weighted_model();
    let param = Parameterisation::with_space(&model, &state, Space::Constrained)
        .unwrap()
        .with_weights([(sigma, 0.0), (y, 0.0)])
        .unwrap();
    // sigma = −1 is outside its support
    let log_prob = param.log_prob(&[0.0, -1.0], false, &mut state).unwrap();
    assert_eq!(log_prob, f64::NEG_INFINITY);
    // a weight of 0 drops a possible term
    let dropped = param.log_prob(&[0.0, 1.0], false, &mut state).unwrap();
    let mu_only = keanu_graph::node_log_prob(model.graph(), &state, model.latents()[0]).unwrap()
        + keanu_graph::node_log_prob(model.graph(), &state, model.factors()[0]).unwrap();
    assert_close_rel(dropped, mu_only, 1e-12, "dropped terms");
}

#[test]
fn invalid_weights_are_errors() {
    let (model, state, [mu, _, y, _]) = weighted_model();
    let mut param = Parameterisation::new(&model, &state).unwrap();
    for weight in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            param.set_weight(y, weight),
            Err(GraphError::InvalidValue(_))
        ));
    }
    // a deterministic node has no term to weigh
    let graph = model.graph();
    let deterministic = graph
        .ids()
        .find(|&id| id > mu && !graph[id].is_probabilistic() && !graph[id].is_factor())
        .unwrap();
    assert!(matches!(
        param.set_weight(deterministic, 2.0),
        Err(GraphError::InvalidValue(_))
    ));
    assert!(matches!(
        param.set_weight(NodeId::from_index(10_000), 2.0),
        Err(GraphError::UnknownNode(_))
    ));
    assert_eq!(param.weight(y), 1.0);
}
