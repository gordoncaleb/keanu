//! Checks keanu-graph against every case in golden-vectors/vertices.json, recorded from the Java implementation.
//!
//! Each case is rebuilt as a Rust graph whose inputs are probabilistic latents holding the recorded input values
//! (as in Java). Every case must either be checked or appear in `SKIPPED` with a reason.
//!
//! Node values are checked for every case, and the forward and reverse partials of the output with respect to each
//! input for every case that records them (an error in the vectors means Rust must fail too).

use keanu_distributions::KeanuRng;
use keanu_graph::{
    BoolVar, DoubleVar, Graph, GraphBuilder, IntVar, NodeId, State, Value, forward_partials,
    reverse_partials,
};
use keanu_parity::{Arg, Case, DType, GoldenTensor, Outcome, close, load, parse_outcome};
use keanu_tensor::{DoubleTensor, IntTensor};

/// Cases that cannot apply to the Rust implementation, with the reason.
const SKIPPED: &[(&str, &str)] = &[];

const TOLERANCE: f64 = 1e-12;

fn input_tensor(arg: &Arg) -> DoubleTensor {
    match arg {
        Arg::Tensor(GoldenTensor {
            dtype: DType::F64,
            shape,
            data,
        }) => DoubleTensor::new(shape, data.clone()).unwrap(),
        other => panic!("expected an f64 tensor input, got {other:?}"),
    }
}

/// A built case: the output node and the input nodes (for partials in step 2).
struct Built {
    output: NodeId,
    inputs: Vec<NodeId>,
}

fn d(v: DoubleVar<'_>) -> NodeId {
    v.id()
}
fn i(v: IntVar<'_>) -> NodeId {
    v.id()
}
fn b(v: BoolVar<'_>) -> NodeId {
    v.id()
}

/// The Java vertex graph for a case, with its inputs as latents holding the recorded values, or None if the case
/// name is not mapped.
fn build_case(case: &Case, g: &GraphBuilder) -> Option<Built> {
    let inputs: Vec<DoubleVar<'_>> = case
        .args
        .iter()
        .map(|arg| {
            let value = input_tensor(arg);
            let latent = g.uniform_with_shape(value.shape(), -1000.0, 1000.0);
            g.set_value(latent, value);
            latent
        })
        .collect();
    let output = build_output(case, g, &inputs)?;
    Some(Built {
        output,
        inputs: inputs.iter().map(|v| v.id()).collect(),
    })
}

fn build_output<'g>(case: &Case, g: &'g GraphBuilder, inputs: &[DoubleVar<'g>]) -> Option<NodeId> {
    let x = || inputs[0];
    let y = || inputs[1];

    // constants used by the value-only cases
    let p = || g.constant(DoubleTensor::new(&[2, 2], vec![-1.5, 0.0, 2.0, 3.5]).unwrap());
    let q = || g.constant(DoubleTensor::new(&[2, 2], vec![0.5, 0.0, 2.0, -1.0]).unwrap());
    let ii = || g.int_constant(IntTensor::new(&[2, 3], vec![-7, -3, 0, 5, 9, 2]).unwrap());
    let jj = || g.int_constant(IntTensor::vector(vec![2, -2, 3]));

    let name = case.name.as_str();
    if let Some(op) = name.strip_prefix("unary/") {
        let v = x();
        return Some(d(match op {
            "abs" => v.abs(),
            "unaryMinus" => -v,
            "reciprocal" => v.reciprocal(),
            "sign" => v.sign(),
            "exp" => v.exp(),
            "exp2" => v.exp2(),
            "expM1" => v.exp_m1(),
            "log" => v.ln(),
            "log1p" => v.ln_1p(),
            "log2" => v.log2(),
            "log10" => v.log10(),
            "sqrt" => v.sqrt(),
            "sin" => v.sin(),
            "cos" => v.cos(),
            "tan" => v.tan(),
            "asin" => v.asin(),
            "acos" => v.acos(),
            "atan" => v.atan(),
            "sinh" => v.sinh(),
            "cosh" => v.cosh(),
            "tanh" => v.tanh(),
            "asinh" => v.asinh(),
            "acosh" => v.acosh(),
            "atanh" => v.atanh(),
            "sigmoid" => v.sigmoid(),
            "floor" => v.floor(),
            "ceil" => v.ceil(),
            "round" => v.round(),
            "logGamma" => v.ln_gamma(),
            "digamma" => v.digamma(),
            "trigamma" => v.trigamma(),
            _ => return None,
        }));
    }
    if let Some(rest) = name.strip_prefix("binary/") {
        let op = rest.split('/').next().unwrap();
        let (a, c) = (x(), y());
        return Some(d(match op {
            "plus" => a + c,
            "minus" => a - c,
            "times" => a * c,
            "div" => a / c,
            "pow" => a.pow(c),
            "atan2" => a.atan2(c),
            "logAddExp" => a.log_add_exp(c),
            "logAddExp2" => a.log_add_exp2(c),
            "max" => a.maximum(c),
            "min" => a.minimum(c),
            _ => return None,
        }));
    }
    Some(match name {
        "reduce/sum" => d(x().sum()),
        "reduce/sum/dims[0,2]" => d(x().sum_axes(&[0, 2])),
        "reduce/sum/dims[-1]" => d(x().sum_axes(&[-1])),
        "reduce/product" => d(x().product()),
        "reduce/mean" => d(x().mean()),
        "reduce/mean/dims[1]" => d(x().mean_axes(&[1])),
        "reduce/max" => d(x().max()),
        "reduce/min" => d(x().min()),
        "reduce/standardDeviation" => d(x().std_dev()),
        "reduce/standardize" => d(x().standardize()),
        "reduce/cumSum/1" => d(x().cumsum(1)),
        "reduce/cumProd/-1" => d(x().cumprod(-1)),
        "reduce/cumProd/withZero" => d(x().cumprod(0)),

        "shape/reshape" => d(x().reshape(&[4, 6])),
        "shape/permute" => d(x().permute(&[2, 0, 1])),
        "shape/transpose" => d(x().transpose()),
        "shape/slice" => d(x().slice("1, ::-1, 1:3")),
        "shape/diag" => d(x().diag()),
        "shape/diagPart" => d(x().diag_part()),
        // Java's triLower(0) is tril(0); triUpper(1) is triu(1)
        "shape/triLower" => d(x().tril(0)),
        "shape/triUpper" => d(x().triu(1)),
        "shape/broadcast" => d(x().broadcast_to(&[2, 3])),
        "shape/fillTriangular" => d(x().fill_triangular(true, true)),

        "linalg/matrixMultiply" | "linalg/matrixMultiply/batch" => d(x().matmul(y())),
        "linalg/tensorMultiply" => d(x().tensor_multiply(y(), &[1, 2], &[1, 0])),
        "linalg/matrixInverse" => d(x().inverse()),
        "linalg/matrixDeterminant" => d(x().determinant()),
        "linalg/choleskyDecomposition" => d(x().cholesky()),
        // Java's choleskyInverse returns the lower triangle of the inverse plus the input's strict upper triangle
        // (zero for a Cholesky factor); building exactly that also makes the partials comparable
        "linalg/choleskyInverse" => d(x().cholesky_inverse().tril(0) + x().triu(1)),

        "graph/sharedSubexpression" => {
            let shared = (x() * y()).sin();
            d((shared * shared + shared.exp()).sum())
        }
        "graph/logSumExpLike" => d(x().exp().sum().ln() - x().max()),
        "graph/quadraticForm" => {
            let v = x().reshape(&[3, 1]);
            d(v.transpose().matmul(y().inverse()).matmul(v).sum())
        }
        "graph/withConstant" => {
            let c = g.constant(DoubleTensor::new(&[2, 2], vec![1.0, -2.0, 3.0, -4.0]).unwrap());
            d((x() * c).sigmoid().sum())
        }
        "control/if" => d(g.where_(x().gt(0.0), x() * y(), y().exp())),

        "bool/greaterThan" => b(p().gt(q())),
        "bool/greaterThanOrEqual" => b(p().ge(q())),
        "bool/lessThan" => b(p().lt(q())),
        "bool/lessThanOrEqual" => b(p().le(q())),
        "bool/elementwiseEquals" => b(p().elementwise_eq(q())),
        "bool/and" => b(p().gt(q()) & p().lt(3.0)),
        "bool/or" => b(p().gt(q()) | p().lt(0.0)),
        "bool/not" => b(!p().gt(q())),
        "bool/allTrue" => b(p().gt(-2.0).all()),
        "bool/anyFalse" => b((!p().gt(-2.0)).any()),
        "bool/toDoubleMask" => d(p().gt(q()).to_double_mask()),

        "integer/plus" => i(ii() + jj()),
        "integer/times" => i(ii() * jj()),
        "integer/div" => i(ii() / jj()),
        "integer/mod" => i(ii() % jj()),
        "integer/abs" => i(ii().abs()),
        "integer/sum" => i(ii().sum()),
        "integer/sumDim" => i(ii().sum_axes(&[1])),
        "integer/max" => i(ii().maximum(jj())),
        "integer/argMax" => i(ii().argmax_axis(1)),
        "integer/toDouble" => d(ii().to_double()),
        "double/toInteger" => i(p().to_int()),
        "generic/take" => d(p().take(&[1, 0])),
        "generic/where" => i(g.where_(
            p().gt(q()),
            g.int_constant(IntTensor::new(&[2, 2], vec![1, 2, 3, 4]).unwrap()),
            g.int_constant(IntTensor::new(&[2, 2], vec![-1, -2, -3, -4]).unwrap()),
        )),
        "generic/concat" => d(g.concat(0, &[p(), q()])),
        _ => return None,
    })
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
        other => panic!("unexpected value {other:?}"),
    }
}

/// Compares the forward and reverse partials of the output with respect to each input against the vectors.
/// Returns the number of partials checked.
fn check_partials(
    case: &Case,
    graph: &Graph,
    state: &State,
    output: NodeId,
    inputs: &[NodeId],
    failures: &mut Vec<String>,
) -> usize {
    let mut checked = 0;
    for (mode, key) in [
        ("forward", "forwardPartials"),
        ("reverse", "reversePartials"),
    ] {
        let Some(expected) = case.result.get(key).and_then(|v| v.as_array()) else {
            continue;
        };
        assert_eq!(
            expected.len(),
            inputs.len(),
            "{}: one {key} entry per input",
            case.name
        );
        for (k, (&input, expected)) in inputs.iter().zip(expected).enumerate() {
            checked += 1;
            let computed = if mode == "forward" {
                forward_partials(graph, state, input, &[output])
                    .and_then(|p| p.get(output).cloned())
            } else {
                reverse_partials(graph, state, output, &[input]).and_then(|p| p.get(input).cloned())
            };
            let problem = match (parse_outcome(expected), computed) {
                (Outcome::Error(_), Err(_)) => None,
                (Outcome::Error(e), Ok(_)) => {
                    Some(format!("Java failed with {e} but Rust returned a partial"))
                }
                (Outcome::Tensor(_), Err(e)) => Some(format!("unexpected error: {e}")),
                (Outcome::Tensor(expected), Ok(partial)) => {
                    if partial.shape() != expected.shape.as_slice() {
                        Some(format!(
                            "shape {:?}, expected {:?}",
                            partial.shape(),
                            expected.shape
                        ))
                    } else {
                        close(partial.as_slice(), &expected.data, PARTIAL_TOLERANCE).err()
                    }
                }
            };
            if let Some(p) = problem {
                failures.push(format!("{} {mode} partial wrt input {k}: {p}", case.name));
            }
        }
    }
    checked
}

const PARTIAL_TOLERANCE: f64 = 1e-10;

#[test]
fn every_vertex_golden_value() {
    let cases = load("vertices.json");
    let mut failures = Vec::new();
    let mut unmapped = Vec::new();
    let mut checked = 0;
    let mut partial_checks = 0;
    for case in &cases {
        if SKIPPED.iter().any(|(n, _)| *n == case.name) {
            continue;
        }
        let g = GraphBuilder::new();
        let Some(Built { output, inputs }) = build_case(case, &g) else {
            unmapped.push(case.name.clone());
            continue;
        };
        checked += 1;
        let expected = parse_outcome(&case.result["value"]);
        let built = g.build().and_then(|(graph, mut state)| {
            state.lazy_eval(&graph, &[output], &mut KeanuRng::seed_from_u64(1))?;
            Ok((graph, state))
        });
        if let Ok((graph, state)) = &built {
            partial_checks += check_partials(case, graph, state, output, &inputs, &mut failures);
        }
        let computed = built.map(|(_, state)| state.value(output).cloned().unwrap());
        let problem = match (expected, computed) {
            (Outcome::Error(_), Err(_)) => None,
            (Outcome::Error(e), Ok(_)) => {
                Some(format!("Java failed with {e} but Rust returned a value"))
            }
            (Outcome::Tensor(_), Err(e)) => Some(format!("unexpected error: {e}")),
            (Outcome::Tensor(expected), Ok(value)) => {
                let (shape, data) = as_f64s(&value);
                if shape != expected.shape {
                    Some(format!("shape {shape:?}, expected {:?}", expected.shape))
                } else {
                    close(&data, &expected.data, TOLERANCE).err()
                }
            }
        };
        if let Some(p) = problem {
            failures.push(format!("{}: {p}", case.name));
        }
    }
    println!(
        "checked {checked} of {} cases ({} skipped), {partial_checks} partials",
        cases.len(),
        SKIPPED.len()
    );
    assert!(
        unmapped.is_empty(),
        "cases with no Rust mapping:\n{}",
        unmapped.join("\n")
    );
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
