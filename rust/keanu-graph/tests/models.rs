//! Rebuilds each model in golden-vectors/models.json with the builder and checks its log prob, log likelihood
//! and their gradients against the values recorded from the Java implementation.

use keanu_graph::{
    Graph, GraphBuilder, LogProbGradient, State, joint_log_prob, joint_log_prob_gradient,
    log_likelihood, log_likelihood_gradient,
};
use keanu_parity::{close, golden_dir, number, parse_tensor};
use keanu_tensor::{DoubleTensor, IntTensor};
use serde_json::Value as Json;

/// Models that can't be rebuilt, with the reason. Empty: every golden model is checked.
const SKIPPED: &[(&str, &str)] = &[];

const TOLERANCE: f64 = 1e-10;

type ModelBuilder = fn() -> (Graph, State);

fn tensor(shape: &[usize], data: &[f64]) -> DoubleTensor {
    DoubleTensor::new(shape, data.to_vec()).unwrap()
}

fn thermometers() -> (Graph, State) {
    let b = GraphBuilder::new();
    let temperature = b.uniform(20.0, 30.0).label("temperature");
    b.set_value(temperature, DoubleTensor::scalar(25.3));
    let one = b.gaussian(temperature, 2.5).label("thermometerOne");
    b.observe(one, DoubleTensor::scalar(25.0));
    let two = b.gaussian(temperature, 5.0).label("thermometerTwo");
    b.observe(two, DoubleTensor::scalar(30.0));
    b.build().unwrap()
}

fn hierarchical_gaussian() -> (Graph, State) {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 10.0).label("mu");
    b.set_value(mu, DoubleTensor::scalar(1.2));
    let sigma = b.half_gaussian(2.0).label("sigma");
    b.set_value(sigma, DoubleTensor::scalar(0.8));
    let group_means = b.gaussian_with_shape(&[3], mu, sigma).label("groupMeans");
    b.set_value(group_means, DoubleTensor::vector(vec![0.5, 1.5, 2.0]));
    let observations = b
        .gaussian_with_shape(&[2, 3], group_means, 1.0)
        .label("observations");
    b.observe(
        observations,
        tensor(&[2, 3], &[0.4, 1.9, 2.2, 0.7, 1.1, 2.6]),
    );
    b.build().unwrap()
}

fn linear_regression() -> (Graph, State) {
    let b = GraphBuilder::new();
    let x = b
        .constant(tensor(&[4, 2], &[1.0, 0.5, 2.0, -1.0, 3.0, 0.2, -0.5, 1.5]))
        .label("x");
    let weights = b.gaussian_with_shape(&[2, 1], 0.0, 1.0).label("weights");
    b.set_value(weights, tensor(&[2, 1], &[0.7, -1.1]));
    let intercept = b.gaussian(0.0, 5.0).label("intercept");
    b.set_value(intercept, DoubleTensor::scalar(0.3));
    let noise = b.exponential(2.0).label("noise");
    b.set_value(noise, DoubleTensor::scalar(0.6));
    let y_hat = x.matmul(weights) + intercept;
    let y = b.gaussian(y_hat, noise).label("y");
    b.observe(y, tensor(&[4, 1], &[0.1, 3.2, 1.5, -1.0]));
    b.build().unwrap()
}

fn beta_binomial() -> (Graph, State) {
    let b = GraphBuilder::new();
    let p = b.beta(2.0, 3.0).label("p");
    b.set_value(p, DoubleTensor::scalar(0.35));
    let successes = b.binomial_with_shape(&[4], p, 10).label("successes");
    b.observe(successes, IntTensor::vector(vec![3, 5, 2, 4]));
    b.build().unwrap()
}

fn gamma_poisson() -> (Graph, State) {
    let b = GraphBuilder::new();
    let rate = b.gamma(2.0, 1.5).label("rate");
    b.set_value(rate, DoubleTensor::scalar(2.4));
    let counts = b.poisson_with_shape(&[5], rate).label("counts");
    b.observe(counts, IntTensor::vector(vec![1, 4, 2, 0, 3]));
    let _total = counts.sum().label("total");
    b.build().unwrap()
}

/// x[t] ~ N(x[t-1], 0.5) with x[-1] = 0, y[t] ~ N(x[t], 1) observed; the previous x reaches each item through a
/// proxy, as Java's Sequence builds it.
fn random_walk_sequence() -> (Graph, State) {
    let b = GraphBuilder::new();
    let observations = [0.3, 0.9, 0.4, 1.6];
    let latents = [0.2, 0.7, 0.6, 1.4];
    let mut previous = b.scalar(0.0);
    for t in 0..observations.len() {
        let proxy = b.double_proxy(&[]);
        b.connect(proxy, previous);
        let x = b
            .gaussian(proxy, 0.5)
            .label(&format!("Sequence_Item_{t}.1.x"));
        b.set_value(x, DoubleTensor::scalar(latents[t]));
        let y = b.gaussian(x, 1.0).label(&format!("Sequence_Item_{t}.1.y"));
        b.observe(y, DoubleTensor::scalar(observations[t]));
        previous = x;
    }
    b.build().unwrap()
}

fn compare_gradients(
    name: &str,
    which: &str,
    graph: &Graph,
    gradient: &LogProbGradient,
    expected: &Json,
    problems: &mut Vec<String>,
) {
    let expected = expected.as_object().unwrap();
    for (label, tensor) in expected {
        let expected = parse_tensor(tensor).unwrap();
        let id = graph
            .find(label)
            .unwrap_or_else(|| panic!("{name}: no node labelled {label}"));
        match gradient.get(id) {
            None => problems.push(format!("{name}: {which} has no gradient for {label}")),
            Some(actual) => {
                if actual.shape() != expected.shape.as_slice() {
                    problems.push(format!(
                        "{name}: {which}[{label}] shape {:?} != {:?}",
                        actual.shape(),
                        expected.shape
                    ));
                } else if let Err(e) = close(actual.as_slice(), &expected.data, TOLERANCE) {
                    problems.push(format!("{name}: {which}[{label}]: {e}"));
                }
            }
        }
    }
    for (id, _) in gradient.iter() {
        let label = graph.node(id).label().unwrap_or("<unlabelled>");
        if !expected.contains_key(label) {
            problems.push(format!(
                "{name}: {which} has an unexpected gradient for {label}"
            ));
        }
    }
}

#[test]
fn every_golden_model() {
    let text = std::fs::read_to_string(golden_dir().join("models.json")).unwrap();
    let root: Json = serde_json::from_str(&text).unwrap();
    let models: [(&str, ModelBuilder); 6] = [
        ("thermometers", thermometers),
        ("hierarchicalGaussian", hierarchical_gaussian),
        ("linearRegression", linear_regression),
        ("betaBinomial", beta_binomial),
        ("gammaPoisson", gamma_poisson),
        ("randomWalkSequence", random_walk_sequence),
    ];

    let mut problems = Vec::new();
    let mut checked = 0;
    for case in root["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        if SKIPPED.iter().any(|(n, _)| *n == name) {
            continue;
        }
        let Some((_, build)) = models.iter().find(|(n, _)| *n == name) else {
            problems.push(format!("{name}: no Rust model"));
            continue;
        };
        checked += 1;
        let (graph, mut state) = build();
        // compute the deterministic nodes; every probabilistic node already has its recorded value
        let all: Vec<_> = graph.ids().collect();
        state
            .lazy_eval(
                &graph,
                &all,
                &mut keanu_distributions::KeanuRng::seed_from_u64(0),
            )
            .unwrap();
        let expected = &case["expected"];

        for (label, value) in expected["latentValues"].as_object().unwrap() {
            let id = graph
                .find(label)
                .unwrap_or_else(|| panic!("{name}: no node labelled {label}"));
            let value = parse_tensor(value).unwrap();
            assert_eq!(
                state.f64(id).unwrap().as_slice(),
                value.data.as_slice(),
                "{name}: latent {label}"
            );
        }

        for (which, actual) in [
            ("logProb", joint_log_prob(&graph, &state).unwrap()),
            ("logLikelihood", log_likelihood(&graph, &state).unwrap()),
        ] {
            if let Err(e) = close(&[actual], &[number(&expected[which])], TOLERANCE) {
                problems.push(format!("{name}: {which}: {e}"));
            }
        }
        compare_gradients(
            name,
            "dLogProb",
            &graph,
            &joint_log_prob_gradient(&graph, &state).unwrap(),
            &expected["dLogProb"],
            &mut problems,
        );
        compare_gradients(
            name,
            "dLogLikelihood",
            &graph,
            &log_likelihood_gradient(&graph, &state).unwrap(),
            &expected["dLogLikelihood"],
            &mut problems,
        );
    }
    println!("checked {checked} models ({} skipped)", SKIPPED.len());
    assert!(
        problems.is_empty(),
        "{} problems:\n{}",
        problems.len(),
        problems.join("\n")
    );
}
