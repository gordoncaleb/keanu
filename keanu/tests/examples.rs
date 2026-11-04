//! The examples, checked against the results the Java examples' tests assert.
//!
//! The Java tests run short Metropolis–Hastings chains and pass for one particular seed; with a different random
//! number generator those chain lengths don't converge reliably (checked over many seeds), so these tests run
//! chains 20 times longer than the Java defaults and assert the same tolerances. Where Metropolis–Hastings with
//! prior proposals can't mix (the Challenger model's strongly correlated parameters), the posterior is checked with
//! NUTS and the Java-style run is only checked to complete.

#![allow(dead_code)]

#[path = "../examples/ab_testing.rs"]
mod ab_testing;
#[path = "../examples/challenger_disaster.rs"]
mod challenger_disaster;
#[path = "../examples/cheating_students.rs"]
mod cheating_students;
#[path = "../examples/coal_mining_disasters.rs"]
mod coal_mining_disasters;
#[path = "../examples/custom_distribution.rs"]
mod custom_distribution;
#[path = "../examples/text_messaging.rs"]
mod text_messaging;

const LONGER: usize = 20;

fn assert_close(actual: f64, expected: f64, within: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= within,
        "{what}: {actual} is not within {within} of {expected}"
    );
}

#[test]
fn coal_mining_disasters_switch_year() {
    // Java asserts exactly 1890 for its seed; the posterior's mode is at 1890 with neighbouring years close behind
    let year =
        coal_mining_disasters::run(1, coal_mining_disasters::DEFAULT_SAMPLES * LONGER).unwrap();
    assert!((1888..=1892).contains(&year), "switch year {year}");
}

#[test]
fn text_messaging_switch_point_and_rates() {
    let r = text_messaging::run(1, text_messaging::DEFAULT_SAMPLES * LONGER).unwrap();
    assert!(
        (41..=45).contains(&r.switch_point_mode),
        "switch point {}",
        r.switch_point_mode
    );
    assert_close(r.early_rate_mean, 18.0, 2.0, "early rate mean");
    assert_close(r.late_rate_mode, 23.0, 2.0, "late rate mode");
}

#[test]
fn ab_testing_posteriors() {
    let p = ab_testing::run(1, ab_testing::DEFAULT_SAMPLES * LONGER).unwrap();
    assert!((0.02..=0.07).contains(&p.p_a), "p_a {}", p.p_a);
    assert!((0.02..=0.07).contains(&p.p_b), "p_b {}", p.p_b);
}

#[test]
fn cheating_students_with_bernoulli() {
    let p = cheating_students::run_with_bernoulli(100, 35.0, 1, 2000 * LONGER).unwrap();
    assert_close(p, 0.2, 0.05, "probability of cheating");
}

#[test]
fn cheating_students_with_binomial() {
    let p = cheating_students::run_with_binomial(100, 35, 1, 100 * LONGER).unwrap();
    assert_close(p, 0.2, 0.05, "probability of cheating");
}

#[test]
fn challenger_disaster_beta_is_small_and_positive() {
    let p = challenger_disaster::run_with_nuts(1, 4000).unwrap();
    assert_close(p.beta, 0.2, 0.2, "beta");
    assert_close(p.alpha, -15.0, 15.0, "alpha");
    // the Java style run (Metropolis–Hastings with prior proposals) still completes
    challenger_disaster::run(1, challenger_disaster::DEFAULT_SAMPLES).unwrap();
}

/// The posterior means of the true rates, computed independently by integrating over a dense grid of
/// φ = (logit(a / (a + b)), ln(a + b)): E[θⱼ | y] = ∫ (a + yⱼ) / (a + b + n) p(φ | y) dφ, with
/// p(φ | y) ∝ a b (a + b)^-5/2 ∏ⱼ B(a + yⱼ, b + n - yⱼ) / B(a, b). They agree to 10⁻⁵ whether or not the hyperprior
/// is truncated at ln(a + b) = 20, as the example's is.
const EXACT_TRUE_RATE_MEANS: [f64; 10] = [
    0.44791, 0.46776, 0.48264, 0.51737, 0.56201, 0.47768, 0.46776, 0.49256, 0.53721, 0.49752,
];

#[test]
fn custom_distribution_posterior_matches_numerical_integration() {
    // Sampled in the reparameterised coordinates, the chains mix well. Over seeds 1 to 40, a rate's mean was off by
    // 0.0005 (root mean square) and at most 0.0022 with NUTS (10,000 samples), and by 0.003 and at most 0.0125 with
    // Metropolis–Hastings (40,000 samples, the first half dropped). The tolerances are about twice the largest
    // errors, and eight to ten times the typical ones.
    let r = custom_distribution::run(1, 40_000, 10_000).unwrap();
    for (j, (&exact, (&nuts, &mh))) in EXACT_TRUE_RATE_MEANS
        .iter()
        .zip(r.nuts.iter().zip(&r.metropolis_hastings))
        .enumerate()
    {
        assert_close(nuts, exact, 0.005, &format!("NUTS true rate {j}"));
        assert_close(
            mh,
            exact,
            0.025,
            &format!("Metropolis–Hastings true rate {j}"),
        );
    }
}

#[test]
fn custom_distribution_gradient_matches_finite_differences() {
    use keanu::prelude::*;
    // the hyperprior and the rate offsets, without the observations: the gradient with respect to φ includes the
    // offsets' gradient with respect to their parameter
    let b = GraphBuilder::new();
    let phi: DoubleVar = b.custom(custom_distribution::HyperPrior, &[2], &[]);
    let offsets: DoubleVar = b.custom(custom_distribution::RateOffsets, &[3], &[phi.id()]);
    let at_phi = vec![0.3, 2.5];
    let at_offsets = vec![-1.2, 0.4, 2.0];
    b.set_value(phi, DoubleTensor::vector(at_phi.clone()));
    b.set_value(offsets, DoubleTensor::vector(at_offsets.clone()));
    let (phi, offsets) = (phi.id(), offsets.id());
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let gradient = model.log_prob_gradient(&state).unwrap();
    for (node, at) in [(phi, &at_phi), (offsets, &at_offsets)] {
        let analytic = gradient.get(node).unwrap().as_slice().to_vec();
        for i in 0..at.len() {
            let h = 1e-6;
            let log_prob = |delta: f64| {
                let mut s = state.clone();
                let mut v = at.clone();
                v[i] += delta;
                s.set_and_cascade(model.graph(), node, DoubleTensor::vector(v))
                    .unwrap();
                model.log_prob(&s).unwrap()
            };
            let numeric = (log_prob(h) - log_prob(-h)) / (2.0 * h);
            assert_close(analytic[i], numeric, 1e-6, "custom gradient");
        }
    }

    // beyond the truncation the log prob is -infinity
    let mut s = state.clone();
    let beyond = custom_distribution::MAX_LN_SIZE + 1.0;
    s.set_and_cascade(model.graph(), phi, DoubleTensor::vector(vec![0.3, beyond]))
        .unwrap();
    assert_eq!(model.log_prob(&s).unwrap(), f64::NEG_INFINITY);
}

#[test]
fn custom_distribution_that_cannot_be_sampled_reports_an_error() {
    use keanu::prelude::*;
    let b = GraphBuilder::new();
    let hyper: DoubleVar = b.custom(custom_distribution::HyperPrior, &[2], &[]);
    let hyper = hyper.id();
    let (graph, mut state) = b.build().unwrap();
    let error = state
        .lazy_eval(&graph, &[hyper], &mut KeanuRng::seed_from_u64(1))
        .unwrap_err();
    assert!(error.to_string().contains("cannot be sampled"), "{error}");
}

#[test]
fn custom_node_type_must_match_the_handle() {
    use keanu::prelude::*;
    let b = GraphBuilder::new();
    let _wrong: IntVar = b.custom(custom_distribution::HyperPrior, &[2], &[]);
    assert!(b.build().is_err());
}
