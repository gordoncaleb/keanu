//! Loads the golden vectors recorded from the Java implementation (see `fixtures/golden-vectors/README.md`)
//! so the Rust crates can be checked against them.

use serde_json::Value;
use std::path::{Path, PathBuf};

/// A tensor as recorded in the golden vectors. Values of every dtype are held as f64 (integers in the
/// vectors are exactly representable), together with the dtype so callers can tell them apart.
#[derive(Debug, Clone, PartialEq)]
pub struct GoldenTensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DType {
    F64,
    I32,
    I64,
    Bool,
}

/// One argument of a case: a tensor, or a plain value such as an axis, a slice string or a list of dimensions.
#[derive(Debug, Clone)]
pub enum Arg {
    Tensor(GoldenTensor),
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Ints(Vec<i64>),
}

/// The recorded result of a case.
#[derive(Debug, Clone)]
pub enum Outcome {
    Tensor(GoldenTensor),
    /// The Java implementation failed with this exception (simple class name).
    Error(String),
}

#[derive(Debug, Clone)]
pub struct Case {
    pub name: String,
    pub op: String,
    pub args: Vec<Arg>,
    /// Named parameters (distribution cases), in file order.
    pub params: Vec<(String, Arg)>,
    pub result: Value,
}

impl Case {
    /// The result interpreted as a single tensor or error (tensor cases).
    pub fn outcome(&self) -> Outcome {
        parse_outcome(&self.result)
    }

    /// A named parameter. Panics if it is missing.
    pub fn param(&self, name: &str) -> &Arg {
        self.params
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, a)| a)
            .unwrap_or_else(|| panic!("case {} has no parameter {name}", self.name))
    }
}

/// The frozen golden vectors (`keanu-parity/fixtures/golden-vectors`).
pub fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/golden-vectors")
}

/// Loads every case from one of the golden vector files, e.g. `"tensors.json"`.
pub fn load(file: &str) -> Vec<Case> {
    let path = golden_dir().join(file);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let root: Value = serde_json::from_str(&text).expect("golden vectors are valid JSON");
    assert_eq!(root["formatVersion"], 1, "unsupported golden vector format");
    root["cases"]
        .as_array()
        .expect("cases array")
        .iter()
        .map(|case| Case {
            name: case["name"].as_str().unwrap().to_owned(),
            op: case["op"].as_str().unwrap().to_owned(),
            args: case["args"]
                .as_array()
                .unwrap()
                .iter()
                .map(parse_arg)
                .collect(),
            params: case
                .get("params")
                .and_then(Value::as_object)
                .map(|m| m.iter().map(|(k, v)| (k.clone(), parse_arg(v))).collect())
                .unwrap_or_default(),
            result: case["result"].clone(),
        })
        .collect()
}

/// Decodes a JSON number or one of the strings "NaN", "Infinity", "-Infinity".
pub fn number(value: &Value) -> f64 {
    match value {
        Value::Number(n) => n.as_f64().unwrap(),
        Value::Bool(b) => f64::from(u8::from(*b)),
        Value::String(s) => match s.as_str() {
            "NaN" => f64::NAN,
            "Infinity" => f64::INFINITY,
            "-Infinity" => f64::NEG_INFINITY,
            other => panic!("unexpected number string {other}"),
        },
        other => panic!("not a number: {other}"),
    }
}

pub fn parse_tensor(value: &Value) -> Option<GoldenTensor> {
    let dtype = match value.get("dtype")?.as_str()? {
        "f64" => DType::F64,
        "i32" => DType::I32,
        "i64" => DType::I64,
        "bool" => DType::Bool,
        other => panic!("unknown dtype {other}"),
    };
    let shape = value["shape"]
        .as_array()?
        .iter()
        .map(|d| d.as_u64().unwrap() as usize)
        .collect();
    let data = value["data"].as_array()?.iter().map(number).collect();
    Some(GoldenTensor { dtype, shape, data })
}

pub fn parse_outcome(value: &Value) -> Outcome {
    if let Some(error) = value.get("error") {
        return Outcome::Error(error.as_str().unwrap().to_owned());
    }
    Outcome::Tensor(parse_tensor(value).unwrap_or_else(|| panic!("not a tensor result: {value}")))
}

fn parse_arg(value: &Value) -> Arg {
    if let Some(t) = parse_tensor(value) {
        return Arg::Tensor(t);
    }
    match value {
        Value::Bool(b) => Arg::Bool(*b),
        Value::String(s) => Arg::Str(s.clone()),
        Value::Number(n) if n.is_i64() => Arg::Int(n.as_i64().unwrap()),
        Value::Number(n) => Arg::Float(n.as_f64().unwrap()),
        Value::Array(items) => Arg::Ints(items.iter().map(|i| i.as_i64().unwrap()).collect()),
        other => panic!("unsupported argument {other}"),
    }
}

/// Compares two f64 slices: equal if both NaN, both the same infinity, or within the tolerance
/// (relative, with an absolute floor of `tolerance` near zero).
pub fn close(actual: &[f64], expected: &[f64], tolerance: f64) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "length {} != expected {}",
            actual.len(),
            expected.len()
        ));
    }
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let ok = (a.is_nan() && e.is_nan())
            || a == e
            || (a.is_finite() && e.is_finite() && (a - e).abs() <= tolerance * e.abs().max(1.0));
        if !ok {
            return Err(format!("element {i}: {a} != expected {e}"));
        }
    }
    Ok(())
}
