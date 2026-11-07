//! A log density of +∞ is an impossible point: the model's log probability paths give −∞ (and no gradient) there,
//! and the bounded transforms never round onto a bound, where a density such as a Beta's with a shape below 1 is
//! +∞.

use keanu_graph::transform::{Bijector, Interval, LowerBound, UpperBound};
use keanu_graph::{GraphBuilder, Model, NodeId, Parameterisation, State};
use keanu_tensor::DoubleTensor;

fn beta_half_model() -> (Model, State, NodeId) {
    let b = GraphBuilder::new();
    let p = b.beta(0.5, 0.5);
    b.set_value(p, DoubleTensor::scalar(0.5));
    let p = p.id();
    let (graph, state) = b.build().unwrap();
    (Model::new(graph, &state).unwrap(), state, p)
}

#[test]
fn an_infinite_density_at_a_bound_is_impossible() {
    let (model, mut state, p) = beta_half_model();
    for bound in [0.0, 1.0] {
        model
            .set_and_cascade(&mut state, [(p, DoubleTensor::scalar(bound).into())])
            .unwrap();
        assert_eq!(model.log_prob(&state).unwrap(), f64::NEG_INFINITY);
        let (log_prob, gradient) = model.log_prob_and_gradient(&state).unwrap();
        assert_eq!(log_prob, f64::NEG_INFINITY);
        assert!(gradient.is_none());
        assert_eq!(
            model.local_log_prob(&state, &[p]).unwrap(),
            f64::NEG_INFINITY
        );
    }
}

#[test]
fn the_unconstrained_log_density_stays_finite_far_into_the_tails() {
    let (model, mut state, _) = beta_half_model();
    let param = Parameterisation::new(&model, &state).unwrap();
    for y in [-1e4, -800.0, -745.5, -40.0, 37.5, 40.0, 800.0, 1e4] {
        let (log_prob, gradient) = param.log_prob_and_gradient(&[y], true, &mut state).unwrap();
        assert!(log_prob.is_finite(), "y = {y}: log prob {log_prob}");
        let gradient = gradient.unwrap();
        assert!(gradient[0].is_finite(), "y = {y}: gradient {gradient:?}");
        let x = state.f64(model.latents()[0]).unwrap().to_scalar().unwrap();
        assert!(x > 0.0 && x < 1.0, "y = {y}: x = {x}");
        assert!(param.log_prob(&[y], true, &mut state).unwrap().is_finite());
    }
}

#[test]
fn bounded_transforms_stay_strictly_inside_their_bounds() {
    let shape: &[usize] = &[];
    let s = DoubleTensor::scalar;
    let interval = Interval::new(&s(0.0), &s(1.0), shape).unwrap();
    let shifted = Interval::new(&s(2.0), &s(3.0), shape).unwrap();
    let lower = LowerBound::new(&s(3.0), shape).unwrap();
    let upper = UpperBound::new(&s(-3.0), shape).unwrap();
    for y in [-1e4, -800.0, -40.0, 0.0, 40.0, 800.0, 1e4] {
        let x = |b: &dyn Bijector| b.forward(&[y]).unwrap().to_scalar().unwrap();
        let (i, j, l, u) = (x(&interval), x(&shifted), x(&lower), x(&upper));
        assert!(i > 0.0 && i < 1.0, "y = {y}: {i}");
        assert!(j > 2.0 && j < 3.0, "y = {y}: {j}");
        assert!(l > 3.0 && l.is_finite(), "y = {y}: {l}");
        assert!(u < -3.0 && u.is_finite(), "y = {y}: {u}");
    }
    // adjacent bounds leave no float strictly inside: the value is one of them, without panicking
    let one = 1.0f64;
    let tight = Interval::new(&s(one), &s(f64::from_bits(one.to_bits() + 1)), shape).unwrap();
    assert!(
        tight
            .forward(&[0.0])
            .unwrap()
            .to_scalar()
            .unwrap()
            .is_finite()
    );
}
