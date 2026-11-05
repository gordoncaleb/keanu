//! MAP and maximum likelihood on the golden models agree with Java's results (golden-vectors/algorithms.json).
//!
//! Optimizers reach the optimum by different paths, so results are compared with optimization tolerances. Adam is
//! an exact port of Java's update rule, so in constrained space (Java's) it is held to a tighter tolerance than
//! conjugate gradient, whose line search is independent of Java's; in unconstrained space (the default) Adam is held
//! to conjugate gradient's tolerance. L-BFGS (the default) is checked on every conjugate gradient case, to conjugate
//! gradient's tolerance and to use no more evaluations than conjugate gradient.

use keanu_algorithms::optimizer::{
    Adam, ConjugateGradient, Fitness, GradientOptimizer, Lbfgs, OptimizedResult,
};
use keanu_algorithms::{Result, Space};
use keanu_distributions::KeanuRng;
use keanu_graph::{Model, NodeId};
use keanu_parity::{golden_dir, parse_tensor};
use serde_json::Value as Json;

/// Cases whose Java result is not a meaningful optimum, with the reason.
const SKIPPED: &[(&str, &str)] = &[
    (
        "hierarchicalGaussian/Adam/MAP",
        "the posterior is unbounded (sigma -> 0 with the group means at mu); Java's Adam steps to a negative sigma",
    ),
    (
        "hierarchicalGaussian/ConjugateGradient/MAP",
        "the posterior is unbounded (sigma -> 0 with the group means at mu), so there is no optimum to agree on",
    ),
];

fn number(v: &Json) -> f64 {
    keanu_parity::number(v)
}

#[test]
fn optimizers_match_java() {
    let text = std::fs::read_to_string(golden_dir().join("algorithms.json")).unwrap();
    let root: Json = serde_json::from_str(&text).unwrap();
    let cases = root["cases"].as_array().unwrap();
    assert!(!cases.is_empty());

    let mut failures = Vec::new();
    let mut checked = 0;
    let mut lbfgs_checked = 0;
    for case in cases {
        let name = case["name"].as_str().unwrap();
        if SKIPPED.iter().any(|(n, _)| *n == name) {
            continue;
        }
        if case["algorithm"]["name"] == "BOBYQA" {
            // checked by golden_bobyqa.rs
            continue;
        }
        let model_name = case["model"].as_str().unwrap();
        let bytes = std::fs::read(golden_dir().join(format!("models/{model_name}.pb"))).unwrap();
        let loaded = keanu_io::load_protobuf(&bytes).unwrap();
        let graph = loaded.graph;
        let mut state = loaded.state;
        let all: Vec<NodeId> = graph.ids().collect();
        state
            .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(0))
            .unwrap();
        let model = Model::new(graph, &state).unwrap();
        let start = state.clone();

        let algorithm = case["algorithm"]["name"].as_str().unwrap();
        let mut optimizer = GradientOptimizer::new(&model);
        let (value_tolerance, fitness_tolerance) = match algorithm {
            "Adam" => {
                let max = case["algorithm"]["maxEvaluations"].as_u64().unwrap() as usize;
                optimizer = optimizer.algorithm(Adam::default().with_max_evaluations(max));
                // the unconstrained trajectory differs from Java's: it must reach the same optimum, at least as
                // well, to conjugate gradient's tolerance
                if case["result"].get("error").is_none() {
                    let fitness = match case["fitness"].as_str().unwrap() {
                        "MAP" => Fitness::LogProb,
                        _ => Fitness::LogLikelihood,
                    };
                    let result = optimizer.optimize(&mut state.clone(), fitness).unwrap();
                    let java_fitness = number(&case["result"]["fitness"]);
                    if result.fitness < java_fitness - 1e-7 * java_fitness.abs().max(1.0) {
                        failures.push(format!(
                            "{name} unconstrained: fitness {} vs Java {java_fitness}",
                            result.fitness
                        ));
                    }
                    for (label, value) in case["result"]["values"].as_object().unwrap() {
                        let java = parse_tensor(value).unwrap();
                        let rust = result.get(model.find(label).unwrap()).unwrap();
                        for (r, j) in rust.as_slice().iter().zip(&java.data) {
                            if (r - j).abs() > 2e-3 * j.abs().max(1.0) {
                                failures
                                    .push(format!("{name} unconstrained {label}: {r} vs Java {j}"));
                            }
                        }
                    }
                }
                // Adam is an exact port of Java's update rule, in the latents' own values
                optimizer = optimizer.with_space(Space::Constrained);
                (1e-6, 1e-9)
            }
            "ConjugateGradient" => {
                optimizer = optimizer.algorithm(ConjugateGradient::default());
                // both stop when the fitness changes by less than 1e-8, so the position is only known to ~1e-4
                (2e-3, 1e-7)
            }
            other => panic!("unknown algorithm {other}"),
        };
        let result = match case["fitness"].as_str().unwrap() {
            "MAP" => optimizer.max_a_posteriori(&mut state),
            _ => optimizer.max_likelihood(&mut state),
        };
        checked += 1;

        let evaluations = result.as_ref().ok().map(|r| r.evaluations);
        if let Some(evaluations) = evaluations {
            println!("{name}: used {evaluations} evaluations");
        }
        let expected = &case["result"];
        if let Some(p) = compare(&model, expected, result, value_tolerance, fitness_tolerance) {
            failures.push(format!("{name}: {p}"));
        }

        if algorithm == "ConjugateGradient" {
            let lbfgs = GradientOptimizer::new(&model).algorithm(Lbfgs::default());
            let result = match case["fitness"].as_str().unwrap() {
                "MAP" => lbfgs.max_a_posteriori(&mut start.clone()),
                _ => lbfgs.max_likelihood(&mut start.clone()),
            };
            if let Ok(r) = &result {
                println!("{name}: L-BFGS used {} evaluations", r.evaluations);
                if !r.converged {
                    failures.push(format!("{name} with L-BFGS: did not converge"));
                }
                // L-BFGS needs at most as many evaluations as conjugate gradient on every golden case (between a
                // half and a ninth of them; docs/benchmarks.md). The one documented exception is the benchmark's
                // warm start from the optimum, where L-BFGS's stricter convergence tests cost it a few more; no
                // golden case starts there.
                if let Some(cg) = evaluations
                    && r.evaluations > cg
                {
                    failures.push(format!(
                        "{name} with L-BFGS: {} evaluations, more than conjugate gradient's {cg}",
                        r.evaluations
                    ));
                }
            }
            // Java's converged flag is conjugate gradient's (which can run out of evaluations): L-BFGS has to
            // converge, and agree on the optimum
            let result = result.map(|mut r| {
                r.converged = expected
                    .get("converged")
                    .map_or(r.converged, |c| c.as_bool().unwrap());
                r
            });
            if let Some(p) = compare(&model, expected, result, value_tolerance, fitness_tolerance) {
                failures.push(format!("{name} with L-BFGS: {p}"));
            }
            lbfgs_checked += 1;
        }
    }
    println!(
        "checked {checked} of {} optimizer cases ({} skipped)",
        cases
            .iter()
            .filter(|c| c["algorithm"]["name"] != "BOBYQA")
            .count(),
        SKIPPED.len()
    );
    // BOBYQA cases are checked by golden_bobyqa.rs
    let gradient_cases = cases
        .iter()
        .filter(|c| c["algorithm"]["name"] != "BOBYQA")
        .count();
    assert_eq!(checked + SKIPPED.len(), gradient_cases);
    println!("checked L-BFGS on {lbfgs_checked} conjugate gradient cases");
    assert!(lbfgs_checked > 0);
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// What is wrong with an optimizer's result compared with Java's, if anything.
fn compare(
    model: &Model,
    expected: &Json,
    result: Result<OptimizedResult>,
    value_tolerance: f64,
    fitness_tolerance: f64,
) -> Option<String> {
    match (expected.get("error"), result) {
        (Some(_), Err(_)) => None,
        (Some(e), Ok(_)) => Some(format!("Java failed with {e} but Rust succeeded")),
        (None, Err(e)) => Some(format!("unexpected error {e}")),
        (None, Ok(result)) => {
            let mut problems = Vec::new();
            let java_fitness = number(&expected["fitness"]);
            if (result.fitness - java_fitness).abs()
                > fitness_tolerance * java_fitness.abs().max(1.0)
            {
                problems.push(format!("fitness {} vs Java {java_fitness}", result.fitness));
            }
            if let Some(converged) = expected.get("converged")
                && converged.as_bool().unwrap() != result.converged
            {
                problems.push(format!(
                    "converged {} vs Java {converged}",
                    result.converged
                ));
            }
            for (label, value) in expected["values"].as_object().unwrap() {
                let java = parse_tensor(value).unwrap();
                let id = model
                    .find(label)
                    .unwrap_or_else(|| panic!("no node {label}"));
                let rust = result
                    .get(id)
                    .unwrap_or_else(|| panic!("no result for {label}"));
                for (r, j) in rust.as_slice().iter().zip(&java.data) {
                    if (r - j).abs() > value_tolerance * j.abs().max(1.0) {
                        problems.push(format!("{label}: {r} vs Java {j}"));
                    }
                }
            }
            (!problems.is_empty()).then(|| problems.join("; "))
        }
    }
}
