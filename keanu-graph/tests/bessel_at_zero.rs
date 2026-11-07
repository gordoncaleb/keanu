//! The Bessel ops' derivatives at x = 0: ln I₀(x) = x²/4 + O(x⁴) there, so its first derivative is 0 and its second
//! ½, not NaN.

use keanu_graph::{GraphBuilder, Model, Parameterisation, Space};
use keanu_tensor::DoubleTensor;

#[test]
fn the_log_bessel_i0_second_derivative_at_zero_is_one_half() {
    // x ~ N(0, 1) at 0 with the factor ln I₀(x): the log density's Hessian is −1 + ½
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    b.factor(x.log_bessel_i(0.0));
    b.set_value(x, DoubleTensor::scalar(0.0));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let param = Parameterisation::with_space(&model, &state, Space::Constrained).unwrap();
    let (log_prob, gradient) = param
        .log_prob_and_gradient(&[0.0], false, &mut state)
        .unwrap();
    assert!(log_prob.is_finite());
    assert_eq!(gradient.unwrap(), vec![0.0]);
    let hessian = param.hessian(&[0.0], false, &mut state).unwrap();
    assert_eq!(hessian.as_slice(), &[-0.5]);
    // and nearby, the same to first order
    let near = param.hessian(&[1e-6], false, &mut state).unwrap();
    assert!((near.as_slice()[0] + 0.5).abs() < 1e-9, "{near:?}");
}
