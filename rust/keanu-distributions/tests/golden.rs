//! Checks every case in golden-vectors/distributions.json, recorded from the Java implementation.
//!
//! Every case must be checked or listed in `SKIPPED` with a reason, so coverage cannot silently shrink.
//! Gradients are compared only where the log prob is finite (they are unspecified outside the support).

use keanu_distributions::*;
use keanu_parity::{
    Arg, Case, DType, GoldenTensor, Outcome, close, load, number, parse_outcome, parse_tensor,
};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};
use serde_json::Value;
use std::collections::BTreeMap;

/// Cases that intentionally differ from Java, with the reason.
const SKIPPED: &[(&str, &str)] = &[];

const TOLERANCE: f64 = 1e-10;

fn f64_tensor(arg: &Arg) -> DoubleTensor {
    match arg {
        Arg::Tensor(GoldenTensor {
            dtype: DType::F64,
            shape,
            data,
        }) => DoubleTensor::new(shape, data.clone()).unwrap(),
        other => panic!("expected an f64 tensor, got {other:?}"),
    }
}
fn int_tensor(arg: &Arg) -> IntTensor {
    match arg {
        Arg::Tensor(GoldenTensor {
            dtype: DType::I32 | DType::I64,
            shape,
            data,
        }) => IntTensor::new(shape, data.iter().map(|&x| x as i64).collect()).unwrap(),
        other => panic!("expected an integer tensor, got {other:?}"),
    }
}
fn bool_tensor(arg: &Arg) -> BoolTensor {
    match arg {
        Arg::Tensor(GoldenTensor {
            dtype: DType::Bool,
            shape,
            data,
        }) => BoolTensor::new(shape, data.iter().map(|&x| x != 0.0).collect()).unwrap(),
        other => panic!("expected a bool tensor, got {other:?}"),
    }
}

/// The computed log prob and named gradients (Java's names), or None if the distribution is unknown.
struct Computed {
    log_prob: Result<DoubleTensor>,
    gradients: Option<Result<BTreeMap<&'static str, DoubleTensor>>>,
}

fn grads(pairs: Vec<(&'static str, DoubleTensor)>) -> BTreeMap<&'static str, DoubleTensor> {
    pairs.into_iter().collect()
}

fn run(case: &Case) -> Option<Computed> {
    let x = &case.args[0];
    let p = |name: &str| f64_tensor(case.param(name));
    let i = |name: &str| int_tensor(case.param(name));
    Some(match case.op.as_str() {
        "Gaussian" => {
            let d = Gaussian::new(p("mu"), p("sigma"));
            let x = f64_tensor(x);
            Computed {
                log_prob: d.log_prob(&x),
                gradients: Some(
                    d.d_log_prob(&x)
                        .map(|g| grads(vec![("X", g.x), ("MU", g.mu), ("SIGMA", g.sigma)])),
                ),
            }
        }
        "Beta" => {
            let d = Beta::new(p("alpha"), p("beta"), p("xMin"), p("xMax"));
            let x = f64_tensor(x);
            Computed {
                log_prob: d.log_prob(&x),
                gradients: Some(
                    d.d_log_prob(&x)
                        .map(|g| grads(vec![("X", g.x), ("A", g.alpha), ("B", g.beta)])),
                ),
            }
        }
        "Cauchy" => {
            let d = Cauchy::new(p("location"), p("scale"));
            let x = f64_tensor(x);
            Computed {
                log_prob: d.log_prob(&x),
                gradients: Some(
                    d.d_log_prob(&x)
                        .map(|g| grads(vec![("X", g.x), ("L", g.location), ("S", g.scale)])),
                ),
            }
        }
        "ChiSquared" => {
            let d = ChiSquared::new(i("k"));
            let x = f64_tensor(x);
            Computed {
                log_prob: d.log_prob(&x),
                gradients: Some(d.d_log_prob(&x).map(|g| grads(vec![("X", g.x)]))),
            }
        }
        "Dirichlet" => {
            let d = Dirichlet::new(p("concentration"));
            let x = f64_tensor(x);
            Computed {
                log_prob: d.log_prob(&x).map(DoubleTensor::scalar),
                gradients: Some(
                    d.d_log_prob(&x)
                        .map(|g| grads(vec![("X", g.x), ("C", g.concentration)])),
                ),
            }
        }
        "Exponential" => {
            let d = Exponential::new(p("rate"));
            let x = f64_tensor(x);
            Computed {
                log_prob: d.log_prob(&x),
                gradients: Some(
                    d.d_log_prob(&x)
                        .map(|g| grads(vec![("X", g.x), ("LAMBDA", g.rate)])),
                ),
            }
        }
        "Gamma" => {
            let d = Gamma::new(p("theta"), p("k"));
            let x = f64_tensor(x);
            Computed {
                log_prob: d.log_prob(&x),
                gradients: Some(
                    d.d_log_prob(&x)
                        .map(|g| grads(vec![("X", g.x), ("THETA", g.theta), ("K", g.k)])),
                ),
            }
        }
        "InverseGamma" => {
            let d = InverseGamma::new(p("alpha"), p("beta"));
            let x = f64_tensor(x);
            Computed {
                log_prob: d.log_prob(&x),
                gradients: Some(
                    d.d_log_prob(&x)
                        .map(|g| grads(vec![("X", g.x), ("A", g.alpha), ("B", g.beta)])),
                ),
            }
        }
        "Laplace" => {
            let d = Laplace::new(p("mu"), p("beta"));
            let x = f64_tensor(x);
            Computed {
                log_prob: d.log_prob(&x),
                gradients: Some(
                    d.d_log_prob(&x)
                        .map(|g| grads(vec![("X", g.x), ("MU", g.mu), ("BETA", g.beta)])),
                ),
            }
        }
        "Logistic" => {
            let d = Logistic::new(p("mu"), p("s"));
            let x = f64_tensor(x);
            Computed {
                log_prob: d.log_prob(&x),
                gradients: Some(
                    d.d_log_prob(&x)
                        .map(|g| grads(vec![("X", g.x), ("MU", g.mu), ("S", g.s)])),
                ),
            }
        }
        "LogNormal" => {
            let d = LogNormal::new(p("mu"), p("sigma"));
            let x = f64_tensor(x);
            Computed {
                log_prob: d.log_prob(&x),
                gradients: Some(
                    d.d_log_prob(&x)
                        .map(|g| grads(vec![("X", g.x), ("MU", g.mu), ("SIGMA", g.sigma)])),
                ),
            }
        }
        "MultivariateGaussian" => {
            let d = MultivariateGaussian::new(p("mu"), p("covariance"));
            let x = f64_tensor(x);
            Computed {
                log_prob: d.log_prob(&x).map(DoubleTensor::scalar),
                gradients: Some(
                    d.d_log_prob(&x).map(|g| {
                        grads(vec![("X", g.x), ("MU", g.mu), ("COVARIANCE", g.covariance)])
                    }),
                ),
            }
        }
        "Pareto" => {
            let d = Pareto::new(p("location"), p("scale"));
            let x = f64_tensor(x);
            Computed {
                log_prob: d.log_prob(&x),
                gradients: Some(
                    d.d_log_prob(&x)
                        .map(|g| grads(vec![("X", g.x), ("L", g.location), ("S", g.scale)])),
                ),
            }
        }
        "StudentT" => {
            let d = StudentT::new(i("v"));
            let x = f64_tensor(x);
            // Java names the derivative with respect to t "T"
            Computed {
                log_prob: d.log_prob(&x),
                gradients: Some(d.d_log_prob(&x).map(|g| grads(vec![("T", g.x)]))),
            }
        }
        "Triangular" => {
            let d = Triangular::new(p("xMin"), p("xMax"), p("c"));
            let x = f64_tensor(x);
            Computed {
                log_prob: d.log_prob(&x),
                gradients: Some(
                    d.d_log_prob(&x).map(|g| {
                        grads(vec![("X", g.x), ("A", g.x_min), ("B", g.x_max), ("C", g.c)])
                    }),
                ),
            }
        }
        // Uniform and UniformInt have no gradient (Java throws); there is no d_log_prob in Rust
        "Uniform" => Computed {
            log_prob: Uniform::new(p("xMin"), p("xMax")).log_prob(&f64_tensor(x)),
            gradients: None,
        },
        "UniformInt" => Computed {
            log_prob: UniformInt::new(i("xMin"), i("xMax")).log_prob(&int_tensor(x)),
            gradients: None,
        },
        "Bernoulli" => {
            let d = Bernoulli::new(p("probTrue"));
            let x = bool_tensor(x);
            Computed {
                log_prob: d.log_prob(&x),
                gradients: Some(
                    d.d_log_prob(&x)
                        .map(|g| grads(vec![("PROB_TRUE", g.prob_true)])),
                ),
            }
        }
        "Binomial" => {
            let d = Binomial::new(p("p"), i("n"));
            let x = int_tensor(x);
            Computed {
                log_prob: d.log_prob(&x),
                gradients: Some(d.d_log_prob(&x).map(|g| grads(vec![("P", g.p)]))),
            }
        }
        "Geometric" => {
            let d = Geometric::new(p("p"));
            let x = int_tensor(x);
            Computed {
                log_prob: d.log_prob(&x),
                gradients: Some(d.d_log_prob(&x).map(|g| grads(vec![("P", g.p)]))),
            }
        }
        "Multinomial" => {
            let d = Multinomial::new(i("n"), p("p")).unwrap();
            let x = int_tensor(x);
            Computed {
                log_prob: d.log_prob(&x),
                gradients: Some(d.d_log_prob(&x).map(|g| grads(vec![("P", g.p)]))),
            }
        }
        "Poisson" => {
            let d = Poisson::new(p("mu"));
            let x = int_tensor(x);
            Computed {
                log_prob: d.log_prob(&x),
                gradients: Some(d.d_log_prob(&x).map(|g| grads(vec![("MU", g.mu)]))),
            }
        }
        // vertex-only distributions: the recorded log prob is the total over x
        "HalfGaussian" => {
            let d = HalfGaussian::new(p("param"));
            let x = f64_tensor(x);
            Computed {
                log_prob: d.log_prob(&x).map(|t| DoubleTensor::scalar(t.sum())),
                gradients: Some(
                    d.d_log_prob(&x)
                        .map(|g| grads(vec![("X", g.x), ("PARAM", g.sigma)])),
                ),
            }
        }
        "HalfCauchy" => {
            let d = HalfCauchy::new(p("param"));
            let x = f64_tensor(x);
            Computed {
                log_prob: d.log_prob(&x).map(|t| DoubleTensor::scalar(t.sum())),
                gradients: Some(
                    d.d_log_prob(&x)
                        .map(|g| grads(vec![("X", g.x), ("PARAM", g.scale)])),
                ),
            }
        }
        _ => return None,
    })
}

/// The recorded log prob, as a tensor (a bare number for the vertex-only totals).
fn expected_log_prob(result: &Value) -> Outcome {
    match result.get("logProb") {
        Some(v) => parse_outcome(v),
        None => {
            let total = &result["totalLogProb"];
            match total.get("error") {
                Some(e) => Outcome::Error(e.as_str().unwrap().to_owned()),
                None => Outcome::Tensor(GoldenTensor {
                    dtype: DType::F64,
                    shape: vec![],
                    data: vec![number(total)],
                }),
            }
        }
    }
}

fn check_tensor(
    label: &str,
    computed: &DoubleTensor,
    expected: &GoldenTensor,
    finite: Option<&[bool]>,
) -> Option<String> {
    if computed.shape() != expected.shape.as_slice() {
        return Some(format!(
            "{label}: shape {:?}, expected {:?}",
            computed.shape(),
            expected.shape
        ));
    }
    let (mut a, mut e) = (computed.as_slice().to_vec(), expected.data.clone());
    if let Some(mask) = finite {
        // compare only where the log prob is finite
        if mask.len() == a.len() {
            a = a
                .iter()
                .zip(mask)
                .filter(|&(_, &m)| m)
                .map(|(&v, _)| v)
                .collect();
            e = e
                .iter()
                .zip(mask)
                .filter(|&(_, &m)| m)
                .map(|(&v, _)| v)
                .collect();
        } else if !mask.iter().all(|&m| m) {
            return None;
        }
    }
    close(&a, &e, TOLERANCE)
        .err()
        .map(|p| format!("{label}: {p}"))
}

#[test]
fn every_distribution_golden_vector() {
    let cases = load("distributions.json");
    let mut names = std::collections::HashSet::new();
    let (mut failures, mut unmapped, mut checked) = (Vec::new(), Vec::new(), 0);
    for case in &cases {
        assert!(
            names.insert(case.name.clone()),
            "duplicate case name {}",
            case.name
        );
        if SKIPPED.iter().any(|(n, _)| *n == case.name) {
            continue;
        }
        let Some(computed) = run(case) else {
            unmapped.push(case.name.clone());
            continue;
        };
        checked += 1;
        let mut problems = Vec::new();

        let finite: Vec<bool> = match (&expected_log_prob(&case.result), &computed.log_prob) {
            (Outcome::Error(_), Err(_)) => vec![],
            (Outcome::Error(e), Ok(_)) => {
                problems.push(format!(
                    "logProb: Java failed with {e} but Rust returned a value"
                ));
                vec![]
            }
            (Outcome::Tensor(_), Err(e)) => {
                problems.push(format!("logProb: unexpected error {e}"));
                vec![]
            }
            (Outcome::Tensor(expected), Ok(lp)) => {
                problems.extend(check_tensor("logProb", lp, expected, None));
                expected.data.iter().map(|v| v.is_finite()).collect()
            }
        };

        let recorded = case
            .result
            .get("dLogProb")
            .or_else(|| case.result.get("dTotalLogProb"))
            .unwrap();
        match (recorded.get("error"), computed.gradients) {
            (Some(_), None) => {} // no gradient on either side
            (Some(_), Some(Err(_))) => {}
            (Some(e), Some(Ok(_))) => problems.push(format!(
                "dLogProb: Java failed with {e} but Rust returned gradients"
            )),
            (None, None) => problems.push("dLogProb: Java has a gradient but Rust does not".into()),
            (None, Some(Err(e))) => problems.push(format!("dLogProb: unexpected error {e}")),
            (None, Some(Ok(gradients))) => {
                let expected = recorded.as_object().unwrap();
                let expected_names: Vec<&str> = expected.keys().map(String::as_str).collect();
                let computed_names: Vec<&str> = gradients.keys().copied().collect();
                let mut sorted = expected_names.clone();
                sorted.sort();
                if sorted != computed_names {
                    problems.push(format!(
                        "dLogProb: gradients {computed_names:?}, expected {expected_names:?}"
                    ));
                }
                for (name, value) in expected {
                    let Some(g) = gradients.get(name.as_str()) else {
                        continue;
                    };
                    let e = parse_tensor(value).unwrap();
                    problems.extend(check_tensor(&format!("d/d{name}"), g, &e, Some(&finite)));
                }
            }
        }
        if !problems.is_empty() {
            failures.push(format!("{}: {}", case.name, problems.join("; ")));
        }
    }
    println!(
        "checked {checked} of {} cases ({} skipped)",
        cases.len(),
        SKIPPED.len()
    );
    assert!(
        unmapped.is_empty(),
        "cases with no Rust mapping: {unmapped:?}"
    );
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
