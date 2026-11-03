//! The non-gradient optimizer (BOBYQA, translated from PRIMA) on the golden models, against Java's BOBYQA
//! (Apache Commons Math) results in golden-vectors/algorithms.json.
//!
//! Where Java's BOBYQA reached the optimum, Rust must reach the same one. Where it did not (it stops short, or
//! refuses problems with a single latent dimension), Rust must instead reach the optimum found by the better of
//! Java's gradient optimizers in the same file.

use keanu_algorithms::optimizer::{Bobyqa, NonGradientOptimizer};
use keanu_distributions::KeanuRng;
use keanu_graph::{Model, NodeId};
use keanu_parity::{golden_dir, number, parse_tensor};
use serde_json::Value as Json;

/// Cases with no optimum to agree on.
const SKIPPED: &[(&str, &str)] = &[
    (
        "hierarchicalGaussian/BOBYQA/MAP",
        "the posterior is unbounded (sigma -> 0 with the group means at mu)",
    ),
    (
        "hierarchicalGaussian/BOBYQA/radius0.5/range5/MAP",
        "the posterior is unbounded (sigma -> 0 with the group means at mu)",
    ),
];

/// Cases where Java's BOBYQA stopped at a non-optimal point, so Rust is checked against the gradient optimum.
const JAVA_STOPPED_SHORT: &[(&str, &str)] = &[
    (
        "linearRegression/BOBYQA/MAP",
        "with Keanu's default initial radius of 10 Java's initial interpolation points leave the support of `noise`",
    ),
    (
        "linearRegression/BOBYQA/MLE",
        "with Keanu's default initial radius of 10 Java's initial interpolation points leave the support of `noise`",
    ),
];

/// Latents that the fitness does not depend on, so any value is optimal.
fn identifiable(model: &str, fitness: &str, latent: &str) -> bool {
    !(model == "hierarchicalGaussian" && fitness == "MLE" && (latent == "mu" || latent == "sigma"))
}

/// The better (higher fitness) of Java's Adam and conjugate gradient results for a model: both stop on loose
/// criteria, and Adam in particular stops measurably short on some models.
fn reference_gradient_case<'a>(cases: &'a [Json], model: &str, fitness: &str) -> &'a Json {
    cases
        .iter()
        .filter(|c| c["model"] == model && c["fitness"] == fitness)
        .filter(|c| {
            c["algorithm"]["name"] == "Adam" || c["algorithm"]["name"] == "ConjugateGradient"
        })
        .filter(|c| c["result"].get("error").is_none())
        .max_by(|a, b| number(&a["result"]["fitness"]).total_cmp(&number(&b["result"]["fitness"])))
        .expect("a gradient optimizer reference for every model")
}

#[test]
fn bobyqa_reaches_the_same_optimum_as_java() {
    let text = std::fs::read_to_string(golden_dir().join("algorithms.json")).unwrap();
    let root: Json = serde_json::from_str(&text).unwrap();
    let cases = root["cases"].as_array().unwrap();

    let mut checked = 0;
    let mut failures = Vec::new();
    for case in cases.iter().filter(|c| c["algorithm"]["name"] == "BOBYQA") {
        let name = case["name"].as_str().unwrap();
        if SKIPPED.iter().any(|(n, _)| *n == name) {
            continue;
        }
        let model_name = case["model"].as_str().unwrap();
        let fitness_name = case["fitness"].as_str().unwrap();

        let java = &case["result"];
        let stopped_short = JAVA_STOPPED_SHORT.iter().any(|(n, _)| *n == name);
        let reference = if java.get("error").is_some() || stopped_short {
            &reference_gradient_case(cases, model_name, fitness_name)["result"]
        } else {
            java
        };

        let bytes = std::fs::read(golden_dir().join(format!("models/{model_name}.pb"))).unwrap();
        let loaded = keanu_io::load_protobuf(&bytes).unwrap();
        let graph = loaded.graph;
        let mut state = loaded.state;
        let all: Vec<NodeId> = graph.ids().collect();
        state
            .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(0))
            .unwrap();
        let model = Model::new(graph, &state).unwrap();

        let settings = &case["algorithm"];
        let mut bobyqa = Bobyqa::default();
        if let Some(r) = settings.get("initialTrustRegionRadius") {
            bobyqa.initial_trust_region_radius = number(r);
        }
        if let Some(r) = settings.get("boundsRange") {
            bobyqa.bounds_range = number(r);
        }
        let optimizer = NonGradientOptimizer::new(&model).algorithm(bobyqa);
        let result = if fitness_name == "MAP" {
            optimizer.max_a_posteriori(&mut state)
        } else {
            optimizer.max_likelihood(&mut state)
        };
        let result = match result {
            Ok(r) => r,
            Err(e) => {
                failures.push(format!("{name}: {e}"));
                continue;
            }
        };
        checked += 1;

        if stopped_short {
            let java_fitness = number(&java["fitness"]);
            if result.fitness < java_fitness {
                failures.push(format!(
                    "{name}: Rust fitness {} is below Java's non-optimal {java_fitness}",
                    result.fitness
                ));
            }
        }
        let expected_fitness = number(&reference["fitness"]);
        if (result.fitness - expected_fitness).abs() > 1e-7 * expected_fitness.abs().max(1.0) {
            failures.push(format!(
                "{name}: fitness {} != {expected_fitness}",
                result.fitness
            ));
        }
        for (label, expected) in reference["values"].as_object().unwrap() {
            if !identifiable(model_name, fitness_name, label) {
                continue;
            }
            let id = model
                .graph()
                .find(label)
                .unwrap_or_else(|| panic!("{name}: no node {label}"));
            let expected = parse_tensor(expected).unwrap();
            let actual = result.get(id).unwrap();
            for (a, e) in actual.as_slice().iter().zip(&expected.data) {
                if (a - e).abs() > 1e-4 * e.abs().max(1.0) {
                    failures.push(format!(
                        "{name}: {label} = {:?}, expected {:?}",
                        actual.as_slice(),
                        expected.data
                    ));
                    break;
                }
            }
        }
    }
    assert_eq!(checked, 22, "every case except the skipped ones is checked");
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
