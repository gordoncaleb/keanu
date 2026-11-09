use keanu_distributions::{Binomial, BinomialLogit};
use keanu_tensor::{DoubleTensor, IntTensor};

#[test]
fn binomial_logit_tails_match_probability_form_and_survive_rounded_probabilities() {
    let n = IntTensor::scalar(5);
    let k = IntTensor::scalar(2);
    let logits = DoubleTensor::vector(vec![-1.5, 0.0, 1.5]);
    let probs = logits.map(|eta| 1.0 / (1.0 + (-eta).exp()));
    let logit = BinomialLogit::new(&logits, &n);
    let reference = Binomial::new(&probs, &n);
    for (actual, expected) in [
        (logit.log_cdf(&k).unwrap(), reference.log_cdf(&k).unwrap()),
        (logit.log_ccdf(&k).unwrap(), reference.log_ccdf(&k).unwrap()),
    ] {
        for (&actual, &expected) in actual.as_slice().iter().zip(expected.as_slice()) {
            assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");
        }
    }

    let extremes = BinomialLogit::new(DoubleTensor::vector(vec![-1000.0, 1000.0]), &n);
    let small = 10.0_f64.ln() - 3000.0;
    let lower = extremes.log_cdf(&k).unwrap();
    let upper = extremes.log_ccdf(&k).unwrap();
    assert_eq!(lower.as_slice()[0], 0.0);
    assert_eq!(upper.as_slice()[1], 0.0);
    assert!((lower.as_slice()[1] - small).abs() < 1e-12);
    assert!((upper.as_slice()[0] - small).abs() < 1e-12);
    assert_eq!(
        extremes.d_log_cdf(&k).unwrap().logit.as_slice(),
        &[0.0, -3.0]
    );
    assert_eq!(
        extremes.d_log_ccdf(&k).unwrap().logit.as_slice(),
        &[3.0, 0.0]
    );
}
