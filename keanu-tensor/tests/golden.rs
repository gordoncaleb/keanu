//! Checks keanu-tensor against every case in golden-vectors/tensors.json, recorded from the Java implementation.
//!
//! Every case must either be checked or appear in `SKIPPED` with a reason, so coverage cannot silently shrink.

use keanu_parity::{Arg, Case, DType, GoldenTensor, Outcome, close, load};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor, TensorError};

/// Cases that intentionally differ from Java, with the reason.
const SKIPPED: &[(&str, &str)] = &[
    (
        "i32/pow/overflow",
        "Java's 32-bit integers are 64-bit in Rust, so this no longer overflows",
    ),
    (
        "i32/sum/overflow",
        "Java's 32-bit integers are 64-bit in Rust, so this no longer overflows",
    ),
];

/// A value computed by the Rust implementation, in a form comparable to the recorded result.
enum Value {
    Tensor { shape: Vec<usize>, data: Vec<f64> },
}

type Computed = Result<Value, TensorError>;

fn f(t: DoubleTensor) -> Value {
    Value::Tensor {
        shape: t.shape().to_vec(),
        data: t.into_vec(),
    }
}
fn i(t: IntTensor) -> Value {
    Value::Tensor {
        shape: t.shape().to_vec(),
        data: t.into_vec().into_iter().map(|x| x as f64).collect(),
    }
}
fn b(t: BoolTensor) -> Value {
    Value::Tensor {
        shape: t.shape().to_vec(),
        data: t
            .into_vec()
            .into_iter()
            .map(|x| f64::from(u8::from(x)))
            .collect(),
    }
}
fn scalar(x: f64) -> Value {
    Value::Tensor {
        shape: vec![],
        data: vec![x],
    }
}

fn as_f64(arg: &Arg) -> DoubleTensor {
    match arg {
        Arg::Tensor(GoldenTensor {
            dtype: DType::F64,
            shape,
            data,
        }) => DoubleTensor::new(shape, data.clone()).unwrap(),
        other => panic!("expected an f64 tensor, got {other:?}"),
    }
}
fn as_int(arg: &Arg) -> IntTensor {
    match arg {
        Arg::Tensor(GoldenTensor {
            dtype: DType::I32 | DType::I64,
            shape,
            data,
        }) => IntTensor::new(shape, data.iter().map(|&x| x as i64).collect()).unwrap(),
        other => panic!("expected an integer tensor, got {other:?}"),
    }
}
fn as_bool(arg: &Arg) -> BoolTensor {
    match arg {
        Arg::Tensor(GoldenTensor {
            dtype: DType::Bool,
            shape,
            data,
        }) => BoolTensor::new(shape, data.iter().map(|&x| x != 0.0).collect()).unwrap(),
        other => panic!("expected a bool tensor, got {other:?}"),
    }
}
fn int(arg: &Arg) -> isize {
    match arg {
        Arg::Int(v) => *v as isize,
        other => panic!("expected an int, got {other:?}"),
    }
}
fn ints(arg: &Arg) -> Vec<isize> {
    match arg {
        Arg::Ints(v) => v.iter().map(|&x| x as isize).collect(),
        other => panic!("expected ints, got {other:?}"),
    }
}
fn uints(arg: &Arg) -> Vec<usize> {
    ints(arg).into_iter().map(|x| x as usize).collect()
}
fn boolean(arg: &Arg) -> bool {
    match arg {
        Arg::Bool(v) => *v,
        other => panic!("expected a bool, got {other:?}"),
    }
}
fn string(arg: &Arg) -> &str {
    match arg {
        Arg::Str(s) => s,
        other => panic!("expected a string, got {other:?}"),
    }
}
fn dtype(arg: &Arg) -> Option<DType> {
    match arg {
        Arg::Tensor(t) => Some(t.dtype),
        _ => None,
    }
}

const EXACT: f64 = 0.0;
const TRANSCENDENTAL: f64 = 1e-14;
const SPECIAL: f64 = 1e-12;
const LINALG: f64 = 1e-12;

/// Runs a case, returning the computed value and the tolerance to compare it with, or None if the op is unknown.
fn run(case: &Case) -> Option<(Computed, f64)> {
    let a = &case.args;
    let arity = a.len();
    let ty = dtype(&a[0]);
    let op = case.op.as_str();

    macro_rules! unary_f64 {
        ($method:ident, $tol:expr) => {
            (Ok(f(as_f64(&a[0]).$method())), $tol)
        };
    }
    macro_rules! binary_f64 {
        ($method:ident, $tol:expr) => {
            (as_f64(&a[0]).$method(&as_f64(&a[1])).map(f), $tol)
        };
    }
    macro_rules! compare {
        ($method:ident) => {
            match ty {
                Some(DType::F64) => (as_f64(&a[0]).$method(&as_f64(&a[1])).map(b), EXACT),
                _ => (as_int(&a[0]).$method(&as_int(&a[1])).map(b), EXACT),
            }
        };
    }
    macro_rules! numeric_binary {
        ($method:ident) => {
            match ty {
                Some(DType::F64) => (as_f64(&a[0]).$method(&as_f64(&a[1])).map(f), EXACT),
                _ => (as_int(&a[0]).$method(&as_int(&a[1])).map(i), EXACT),
            }
        };
    }

    Some(match (op, ty, arity) {
        // element-wise unary
        ("abs", Some(DType::F64), 1) => unary_f64!(abs, EXACT),
        ("abs", _, 1) => (Ok(i(as_int(&a[0]).abs())), EXACT),
        ("unaryMinus", Some(DType::F64), 1) => unary_f64!(neg, EXACT),
        ("unaryMinus", _, 1) => (Ok(i(as_int(&a[0]).neg())), EXACT),
        ("reciprocal", _, 1) => unary_f64!(reciprocal, EXACT),
        ("sign", _, 1) => unary_f64!(sign, EXACT),
        ("exp", _, 1) => unary_f64!(exp, TRANSCENDENTAL),
        ("exp2", _, 1) => unary_f64!(exp2, TRANSCENDENTAL),
        ("expM1", _, 1) => unary_f64!(exp_m1, TRANSCENDENTAL),
        ("log", _, 1) => unary_f64!(ln, TRANSCENDENTAL),
        ("log1p", _, 1) => unary_f64!(ln_1p, TRANSCENDENTAL),
        ("log2", _, 1) => unary_f64!(log2, TRANSCENDENTAL),
        ("log10", _, 1) => unary_f64!(log10, TRANSCENDENTAL),
        ("sqrt", _, 1) => unary_f64!(sqrt, EXACT),
        ("sin", _, 1) => unary_f64!(sin, TRANSCENDENTAL),
        ("cos", _, 1) => unary_f64!(cos, TRANSCENDENTAL),
        ("tan", _, 1) => unary_f64!(tan, TRANSCENDENTAL),
        ("asin", _, 1) => unary_f64!(asin, TRANSCENDENTAL),
        ("acos", _, 1) => unary_f64!(acos, TRANSCENDENTAL),
        ("atan", _, 1) => unary_f64!(atan, TRANSCENDENTAL),
        ("sinh", _, 1) => unary_f64!(sinh, TRANSCENDENTAL),
        ("cosh", _, 1) => unary_f64!(cosh, TRANSCENDENTAL),
        ("tanh", _, 1) => unary_f64!(tanh, TRANSCENDENTAL),
        ("asinh", _, 1) => unary_f64!(asinh, TRANSCENDENTAL),
        ("acosh", _, 1) => unary_f64!(acosh, TRANSCENDENTAL),
        ("atanh", _, 1) => unary_f64!(atanh, TRANSCENDENTAL),
        ("sigmoid", _, 1) => unary_f64!(sigmoid, TRANSCENDENTAL),
        ("floor", _, 1) => unary_f64!(floor, EXACT),
        ("ceil", _, 1) => unary_f64!(ceil, EXACT),
        ("round", _, 1) => unary_f64!(round, EXACT),
        ("logGamma", _, 1) => unary_f64!(ln_gamma, SPECIAL),
        ("digamma", _, 1) => unary_f64!(digamma, SPECIAL),
        ("trigamma", _, 1) => unary_f64!(trigamma, SPECIAL),

        // element-wise binary
        ("plus", _, 2) => numeric_binary!(add),
        ("minus", _, 2) => numeric_binary!(sub),
        ("times", _, 2) => numeric_binary!(mul),
        ("max", _, 2) => numeric_binary!(maximum),
        ("min", _, 2) => numeric_binary!(minimum),
        ("div", Some(DType::F64), 2) => binary_f64!(div, EXACT),
        ("div", _, 2) => (as_int(&a[0]).div(&as_int(&a[1])).map(i), EXACT),
        ("mod", _, 2) => (as_int(&a[0]).rem(&as_int(&a[1])).map(i), EXACT),
        ("pow", Some(DType::F64), 2) => binary_f64!(pow, TRANSCENDENTAL),
        ("pow", _, 2) => (as_int(&a[0]).pow(&as_int(&a[1])).map(i), EXACT),
        ("atan2", _, 2) => binary_f64!(atan2, TRANSCENDENTAL),
        ("reverseMinus", _, 2) => (as_f64(&a[1]).sub(&as_f64(&a[0])).map(f), EXACT),
        ("reverseDiv", _, 2) => (as_f64(&a[1]).div(&as_f64(&a[0])).map(f), EXACT),
        ("logAddExp", _, 2) => binary_f64!(log_add_exp, TRANSCENDENTAL),
        ("logAddExp2", _, 2) => binary_f64!(log_add_exp2, TRANSCENDENTAL),
        ("safeLogTimes", _, 2) => binary_f64!(safe_ln_times, TRANSCENDENTAL),
        ("greaterThanMask", _, 2) => binary_f64!(gt_mask, EXACT),
        ("greaterThanOrEqualToMask", _, 2) => binary_f64!(ge_mask, EXACT),
        ("lessThanMask", _, 2) => binary_f64!(lt_mask, EXACT),
        ("lessThanOrEqualToMask", _, 2) => binary_f64!(le_mask, EXACT),
        ("greaterThan", _, 2) => compare!(gt),
        ("greaterThanOrEqual", _, 2) => compare!(ge),
        ("lessThan", _, 2) => compare!(lt),
        ("lessThanOrEqual", _, 2) => compare!(le),
        ("elementwiseEquals", _, 2) => compare!(elementwise_eq),
        ("notEqualTo", _, 2) => compare!(elementwise_ne),

        // reductions over everything
        ("sum", Some(DType::F64), 1) => (Ok(scalar(as_f64(&a[0]).sum())), EXACT),
        ("sum", _, 1) => (Ok(scalar(as_int(&a[0]).sum() as f64)), EXACT),
        ("product", Some(DType::F64), 1) => (Ok(scalar(as_f64(&a[0]).product())), EXACT),
        ("product", _, 1) => (Ok(scalar(as_int(&a[0]).product() as f64)), EXACT),
        ("max", Some(DType::F64), 1) => (Ok(scalar(as_f64(&a[0]).max().unwrap())), EXACT),
        ("max", _, 1) => (Ok(scalar(as_int(&a[0]).max().unwrap() as f64)), EXACT),
        ("min", Some(DType::F64), 1) => (Ok(scalar(as_f64(&a[0]).min().unwrap())), EXACT),
        ("min", _, 1) => (Ok(scalar(as_int(&a[0]).min().unwrap() as f64)), EXACT),
        ("mean", _, 1) => (Ok(scalar(as_f64(&a[0]).mean())), 1e-15),
        ("standardDeviation", _, 1) => (Ok(scalar(as_f64(&a[0]).std_dev())), 1e-14),
        ("standardize", _, 1) => (Ok(f(as_f64(&a[0]).standardize())), 1e-14),
        ("argMax", Some(DType::F64), 1) => {
            (Ok(scalar(as_f64(&a[0]).argmax().unwrap() as f64)), EXACT)
        }
        ("argMax", _, 1) => (Ok(scalar(as_int(&a[0]).argmax().unwrap() as f64)), EXACT),
        ("argMin", Some(DType::F64), 1) => {
            (Ok(scalar(as_f64(&a[0]).argmin().unwrap() as f64)), EXACT)
        }
        ("argMin", _, 1) => (Ok(scalar(as_int(&a[0]).argmin().unwrap() as f64)), EXACT),
        ("nanArgMax", _, 1) => (
            Ok(scalar(as_f64(&a[0]).nan_argmax().unwrap() as f64)),
            EXACT,
        ),
        ("nanArgMin", _, 1) => (
            Ok(scalar(as_f64(&a[0]).nan_argmin().unwrap() as f64)),
            EXACT,
        ),

        // reductions and scans over axes
        ("sum", Some(DType::F64), 2) => (as_f64(&a[0]).sum_axes(&ints(&a[1])).map(f), EXACT),
        ("product", Some(DType::F64), 2) => {
            (as_f64(&a[0]).product_axes(&ints(&a[1])).map(f), EXACT)
        }
        ("mean", _, 2) => (as_f64(&a[0]).mean_axes(&ints(&a[1])).map(f), 1e-15),
        ("argMax", Some(DType::F64), 2) => (as_f64(&a[0]).argmax_axis(int(&a[1])).map(i), EXACT),
        ("argMin", Some(DType::F64), 2) => (as_f64(&a[0]).argmin_axis(int(&a[1])).map(i), EXACT),
        ("cumSum", Some(DType::F64), 2) => (as_f64(&a[0]).cumsum(int(&a[1])).map(f), EXACT),
        ("cumSum", _, 2) => (as_int(&a[0]).cumsum(int(&a[1])).map(i), EXACT),
        ("cumProd", Some(DType::F64), 2) => (as_f64(&a[0]).cumprod(int(&a[1])).map(f), EXACT),
        ("cumProd", _, 2) => (as_int(&a[0]).cumprod(int(&a[1])).map(i), EXACT),

        // shapes
        ("reshape", _, 2) => (as_f64(&a[0]).reshape(&ints(&a[1])).map(f), EXACT),
        ("permute", _, 2) => (as_f64(&a[0]).permute(&uints(&a[1])).map(f), EXACT),
        ("transpose", _, 1) => (as_f64(&a[0]).transpose().map(f), EXACT),
        ("broadcast", _, 2) => (as_f64(&a[0]).broadcast_to(&uints(&a[1])).map(f), EXACT),
        ("diag", _, 1) => (as_f64(&a[0]).diag().map(f), EXACT),
        ("diagPart", _, 1) => (as_f64(&a[0]).diag_part().map(f), EXACT),
        // Java's triLower(k) counts k in the opposite direction to numpy (and to its own triUpper)
        ("triLower", _, 2) => (as_f64(&a[0]).tril(-int(&a[1])).map(f), EXACT),
        ("triUpper", _, 2) => (as_f64(&a[0]).triu(int(&a[1])).map(f), EXACT),
        ("fillTriangular", _, 3) => (
            as_f64(&a[0])
                .fill_triangular(boolean(&a[1]), boolean(&a[2]))
                .map(f),
            EXACT,
        ),
        ("slice", _, 2) => (as_f64(&a[0]).slice(string(&a[1])).map(f), EXACT),

        // linear algebra
        ("matrixMultiply", _, 2) => (as_f64(&a[0]).matmul(&as_f64(&a[1])).map(f), LINALG),
        ("matrixMultiply", _, 4) => (
            as_f64(&a[0])
                .matmul_transposed(&as_f64(&a[1]), boolean(&a[2]), boolean(&a[3]))
                .map(f),
            LINALG,
        ),
        ("matrixInverse", _, 1) => (as_f64(&a[0]).inverse().map(f), LINALG),
        ("matrixDeterminant", _, 1) => (as_f64(&a[0]).determinant().map(f), LINALG),
        ("choleskyDecomposition", _, 1) => (as_f64(&a[0]).cholesky().map(f), LINALG),
        // Java returns only the lower triangle of the inverse; Rust returns the full symmetric inverse
        ("choleskyInverse", _, 1) => (
            as_f64(&a[0])
                .cholesky_inverse()
                .and_then(|t| t.tril(0))
                .map(f),
            LINALG,
        ),
        ("tensorMultiply", _, 4) => (
            as_f64(&a[0])
                .tensor_multiply(&as_f64(&a[1]), &ints(&a[2]), &ints(&a[3]))
                .map(f),
            LINALG,
        ),

        // casts
        ("toDouble", _, 1) => (Ok(f(as_int(&a[0]).to_double())), EXACT),
        ("toInteger", _, 1) => (Ok(i(as_f64(&a[0]).to_int())), EXACT),
        ("toDoubleMask", _, 1) => (Ok(f(as_bool(&a[0]).to_double_mask())), EXACT),
        ("toIntegerMask", _, 1) => (Ok(i(as_bool(&a[0]).to_int_mask())), EXACT),

        // boolean logic
        ("and", _, 2) => (as_bool(&a[0]).and(&as_bool(&a[1])).map(b), EXACT),
        ("or", _, 2) => (as_bool(&a[0]).or(&as_bool(&a[1])).map(b), EXACT),
        ("xor", _, 2) => (as_bool(&a[0]).xor(&as_bool(&a[1])).map(b), EXACT),
        ("not", _, 1) => (Ok(b(as_bool(&a[0]).not())), EXACT),
        ("allTrue", _, 1) => (Ok(scalar(f64::from(u8::from(as_bool(&a[0]).all())))), EXACT),
        ("allFalse", _, 1) => (
            Ok(scalar(f64::from(u8::from(!as_bool(&a[0]).any())))),
            EXACT,
        ),
        ("anyTrue", _, 1) => (Ok(scalar(f64::from(u8::from(as_bool(&a[0]).any())))), EXACT),
        ("anyFalse", _, 1) => (
            Ok(scalar(f64::from(u8::from(!as_bool(&a[0]).all())))),
            EXACT,
        ),

        _ => return None,
    })
}

#[test]
fn every_tensor_golden_vector() {
    let cases = load("tensors.json");
    let mut failures = Vec::new();
    let mut unmapped = Vec::new();
    let mut checked = 0;
    for case in &cases {
        if SKIPPED.iter().any(|(name, _)| *name == case.name) {
            continue;
        }
        let Some((computed, tolerance)) = run(case) else {
            unmapped.push(format!(
                "{} (op {}, {} args)",
                case.name,
                case.op,
                case.args.len()
            ));
            continue;
        };
        checked += 1;
        let problem = match (case.outcome(), computed) {
            (Outcome::Error(_), Err(_)) => None,
            (Outcome::Error(e), Ok(_)) => {
                Some(format!("Java failed with {e} but Rust returned a value"))
            }
            (Outcome::Tensor(_), Err(e)) => Some(format!("unexpected error: {e}")),
            (Outcome::Tensor(expected), Ok(Value::Tensor { shape, data })) => {
                if shape != expected.shape {
                    Some(format!("shape {shape:?}, expected {:?}", expected.shape))
                } else {
                    close(&data, &expected.data, tolerance).err()
                }
            }
        };
        if let Some(problem) = problem {
            failures.push(format!("{}: {problem}", case.name));
        }
    }
    println!(
        "checked {checked} of {} cases ({} skipped)",
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
