//! NUTS and Metropolis–Hastings posteriors on the golden models agree with Java's (golden-vectors/samplers.json).
//!
//! The random streams differ, so for every latent element the Rust and Java posterior means must agree within
//! Z combined Monte Carlo standard errors, sd · sqrt(1 / ess_java + 1 / ess_rust), and likewise the standard
//! deviations (whose standard error is roughly sd · sqrt(1 / (2 ess))). The Rust chains use the same settings and
//! fixed seeds, so the test is deterministic.

use keanu_algorithms::mcmc::metropolis::MetropolisHastings;
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_algorithms::mcmc::{SamplingOptions, generate};
use keanu_algorithms::statistics::effective_sample_size;
use keanu_distributions::KeanuRng;
use keanu_graph::{Model, NodeId};
use keanu_parity::{golden_dir, parse_tensor};
use serde_json::Value as Json;

/// How many combined standard errors apart the posteriors may be.
const Z: f64 = 5.0;

/// Cases that can't be compared, with the reason.
const SKIPPED: &[(&str, &str)] = &[];

#[test]
fn sampler_posteriors_match_java() {
    let text = std::fs::read_to_string(golden_dir().join("samplers.json")).unwrap();
    let root: Json = serde_json::from_str(&text).unwrap();
    let cases = root["cases"].as_array().unwrap();
    assert!(!cases.is_empty());

    let mut failures = Vec::new();
    let mut compared = 0;
    for case in cases {
        let name = case["name"].as_str().unwrap();
        if SKIPPED.iter().any(|(n, _)| *n == name) {
            continue;
        }
        let model_name = case["model"].as_str().unwrap();
        let bytes = std::fs::read(golden_dir().join(format!("models/{model_name}.pb"))).unwrap();
        let loaded = keanu_io::load_protobuf(&bytes).unwrap();
        let (graph, mut state) = (loaded.graph, loaded.state);
        let mut rng = KeanuRng::seed_from_u64(1);
        let all: Vec<NodeId> = graph.ids().collect();
        state.lazy_eval(&graph, &all, &mut rng).unwrap();
        let model = Model::new(graph, &state).unwrap();

        let posterior = &case["result"]["posterior"];
        let labels: Vec<&str> = posterior
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        let nodes: Vec<NodeId> = labels
            .iter()
            .map(|l| {
                model
                    .find(l)
                    .unwrap_or_else(|| panic!("{name}: no node labelled {l}"))
            })
            .collect();

        let settings = &case["sampler"];
        let total = settings["samples"].as_u64().unwrap() as usize;
        let drop = settings["drop"].as_u64().unwrap() as usize;
        let options = SamplingOptions {
            drop,
            down_sample_interval: 1,
        };
        let samples = match settings["name"].as_str().unwrap() {
            "NUTS" => {
                let nuts = Nuts {
                    adapt_count: settings["adaptCount"].as_u64().unwrap() as usize,
                    ..Nuts::default()
                };
                let mut sampler = nuts.sampler(&model, &mut state, &nodes, &mut rng).unwrap();
                generate(&mut sampler, &mut state, total, options, &mut rng).unwrap()
            }
            "MetropolisHastings" => MetropolisHastings::default()
                .sample_with(&model, &mut state, &nodes, total, options, &mut rng)
                .unwrap(),
            other => panic!("unknown sampler {other}"),
        };

        for (label, &id) in labels.iter().zip(&nodes) {
            let expected = &posterior[*label];
            let java_mean = parse_tensor(&expected["mean"]).unwrap().data;
            let java_sd = parse_tensor(&expected["sd"]).unwrap().data;
            let java_ess = parse_tensor(&expected["ess"]).unwrap().data;
            let chain = samples.f64(id).unwrap();
            for e in 0..java_mean.len() {
                let index: Vec<usize> = unflatten(e, chain.as_slice()[0].shape());
                let xs = chain.element(&index).unwrap();
                let n = xs.len() as f64;
                let m = xs.iter().sum::<f64>() / n;
                let sd = (xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0)).sqrt();
                let ess = effective_sample_size(&xs).max(10.0);
                let error = (1.0 / java_ess[e].max(10.0) + 1.0 / ess).sqrt();
                compared += 1;
                if (m - java_mean[e]).abs() > Z * java_sd[e] * error {
                    failures.push(format!(
                        "{name} {label}[{e}]: mean {m} vs Java {} (sd {}, ess {ess:.0} / {:.0})",
                        java_mean[e], java_sd[e], java_ess[e]
                    ));
                }
                if (sd - java_sd[e]).abs() > Z * java_sd[e] * error / 2f64.sqrt() {
                    failures.push(format!(
                        "{name} {label}[{e}]: sd {sd} vs Java {}",
                        java_sd[e]
                    ));
                }
            }
        }
    }
    println!("compared {compared} posterior elements");
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn unflatten(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0; shape.len()];
    for d in (0..shape.len()).rev() {
        index[d] = flat % shape[d];
        flat /= shape[d];
    }
    index
}
