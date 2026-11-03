//! Every model in golden-vectors/models.json, saved by the Java implementation, loads from both its protobuf and
//! JSON files and reproduces Java's log probability, log likelihood, gradients and the value of every labelled node.
//! Deterministic nodes are recomputed in Rust rather than taken from the saved values.

use keanu_distributions::KeanuRng;
use keanu_graph::op::Op;
use keanu_graph::{Model, State, Value};
use keanu_io::Loaded;
use keanu_parity::{close, golden_dir, number, parse_tensor};
use serde_json::Value as Json;
use std::collections::BTreeMap;

const TOLERANCE: f64 = 1e-10;

fn models() -> Vec<Json> {
    let text = std::fs::read_to_string(golden_dir().join("models.json")).unwrap();
    let root: Json = serde_json::from_str(&text).unwrap();
    root["cases"].as_array().unwrap().clone()
}

fn as_f64s(value: &Value) -> (Vec<usize>, Vec<f64>) {
    match value {
        Value::F64(t) => (t.shape().to_vec(), t.as_slice().to_vec()),
        Value::I64(t) => (
            t.shape().to_vec(),
            t.as_slice().iter().map(|&x| x as f64).collect(),
        ),
        Value::Bool(t) => (
            t.shape().to_vec(),
            t.as_slice()
                .iter()
                .map(|&x| f64::from(u8::from(x)))
                .collect(),
        ),
    }
}

/// Clears every deterministic, non-constant, unobserved value and recomputes it.
fn recompute(loaded: &Loaded) -> State {
    let mut state = loaded.state.clone();
    let graph = &loaded.graph;
    for id in graph.ids() {
        let node = graph.node(id);
        if !node.is_probabilistic()
            && !matches!(node.op(), Op::Constant(_))
            && !state.is_observed(id)
        {
            state.clear(id);
        }
    }
    let all: Vec<_> = graph.ids().collect();
    state
        .lazy_eval(graph, &all, &mut KeanuRng::seed_from_u64(0))
        .unwrap();
    state
}

fn check_tensor(
    problems: &mut Vec<String>,
    what: &str,
    actual: (Vec<usize>, Vec<f64>),
    expected: &Json,
) {
    let expected = parse_tensor(expected).unwrap();
    if actual.0 != expected.shape {
        problems.push(format!(
            "{what}: shape {:?}, expected {:?}",
            actual.0, expected.shape
        ));
    } else if let Err(e) = close(&actual.1, &expected.data, TOLERANCE) {
        problems.push(format!("{what}: {e}"));
    }
}

fn check(name: &str, format: &str, loaded: &Loaded, expected: &Json) -> Vec<String> {
    let mut problems = Vec::new();
    let graph = &loaded.graph;
    let state = recompute(loaded);
    let model = Model::new(graph.clone(), &state).unwrap();
    let what = |w: &str| format!("{name} ({format}) {w}");

    for (key, actual) in [
        ("logProb", model.log_prob(&state).unwrap()),
        ("logLikelihood", model.log_likelihood(&state).unwrap()),
    ] {
        if let Err(e) = close(&[actual], &[number(&expected[key])], TOLERANCE) {
            problems.push(format!("{}: {e}", what(key)));
        }
    }

    for (key, gradient) in [
        ("dLogProb", model.log_prob_gradient(&state).unwrap()),
        (
            "dLogLikelihood",
            model.log_likelihood_gradient(&state).unwrap(),
        ),
    ] {
        let expected: &serde_json::Map<String, Json> = expected[key].as_object().unwrap();
        let actual: BTreeMap<String, _> = gradient
            .iter()
            .map(|(id, g)| {
                (
                    graph.node(id).label().unwrap_or("?").to_owned(),
                    (g.shape().to_vec(), g.as_slice().to_vec()),
                )
            })
            .collect();
        let expected_keys: Vec<&String> = expected.keys().collect();
        let actual_keys: Vec<&String> = actual.keys().collect();
        if expected_keys != actual_keys {
            problems.push(format!(
                "{}: gradient keys {actual_keys:?}, expected {expected_keys:?}",
                what(key)
            ));
            continue;
        }
        for (label, value) in actual {
            check_tensor(
                &mut problems,
                &what(&format!("{key}[{label}]")),
                value,
                &expected[&label],
            );
        }
    }

    for key in ["latentValues", "nodeValues"] {
        for (label, value) in expected[key].as_object().unwrap() {
            match graph.find(label) {
                None => problems.push(format!("{}: no node labelled {label}", what(key))),
                Some(id) => check_tensor(
                    &mut problems,
                    &what(&format!("{key}[{label}]")),
                    as_f64s(state.value(id).unwrap()),
                    value,
                ),
            }
        }
    }
    problems
}

#[test]
fn every_golden_model_loads_and_matches_java() {
    let cases = models();
    assert!(
        cases.len() >= 7,
        "expected the golden models including the catalog"
    );
    let mut problems = Vec::new();
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let pb = std::fs::read(golden_dir().join(case["protobuf"].as_str().unwrap())).unwrap();
        let json =
            std::fs::read_to_string(golden_dir().join(case["json"].as_str().unwrap())).unwrap();
        for (format, loaded) in [
            ("protobuf", keanu_io::load_protobuf(&pb)),
            ("json", keanu_io::load_json(&json)),
        ] {
            match loaded {
                Ok(loaded) => {
                    assert_eq!(loaded.metadata.get("name").map(String::as_str), Some(name));
                    problems.extend(check(name, format, &loaded, &case["expected"]));
                }
                Err(e) => problems.push(format!("{name} ({format}): {e}")),
            }
        }
    }
    assert!(
        problems.is_empty(),
        "{} problems:\n{}",
        problems.len(),
        problems.join("\n")
    );
}
