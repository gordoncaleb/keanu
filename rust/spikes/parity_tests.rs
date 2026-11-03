// Shared by both spikes: `T` is the spike's tensor type. Runs the subset of tensors.json the spike implements.
use keanu_parity::{close, load, Arg, Case, GoldenTensor, Outcome};

fn tensor(arg: &Arg) -> T {
    match arg {
        Arg::Tensor(GoldenTensor { shape, data, .. }) => T::new(shape, data.clone()).unwrap(),
        other => panic!("expected tensor, got {other:?}"),
    }
}

fn int(arg: &Arg) -> i64 {
    match arg {
        Arg::Int(i) => *i,
        other => panic!("expected int, got {other:?}"),
    }
}

fn ints(arg: &Arg) -> Vec<i64> {
    match arg {
        Arg::Ints(v) => v.clone(),
        other => panic!("expected ints, got {other:?}"),
    }
}

fn boolean(arg: &Arg) -> bool {
    match arg {
        Arg::Bool(b) => *b,
        other => panic!("expected bool, got {other:?}"),
    }
}

type Got = Result<(Vec<usize>, Vec<f64>), String>;

fn of(t: Result<T, impl std::fmt::Debug>) -> Got {
    t.map(|t| (t.shape().to_vec(), t.data().to_vec())).map_err(|e| format!("{e:?}"))
}

/// Runs a case if the spike supports it, returning None for unsupported ops.
fn run(case: &Case) -> Option<(Got, f64)> {
    let a = &case.args;
    let name = case.name.as_str();
    let only_f64 = a.iter().all(|x| !matches!(x, Arg::Tensor(t) if t.dtype != keanu_parity::DType::F64));
    if !only_f64 {
        return None;
    }
    let exact = 0.0;
    let transcendental = 1e-14;
    Some(match case.op.as_str() {
        "plus" if name.starts_with("binary/") => (of(tensor(&a[0]).plus(&tensor(&a[1]))), exact),
        "times" if name.starts_with("binary/") => (of(tensor(&a[0]).times(&tensor(&a[1]))), exact),
        "div" if name.starts_with("binary/") => (of(tensor(&a[0]).div(&tensor(&a[1]))), exact),
        "pow" if name.starts_with("binary/") => (of(tensor(&a[0]).pow(&tensor(&a[1]))), transcendental),
        "atan2" => (of(tensor(&a[0]).atan2(&tensor(&a[1]))), transcendental),
        "exp" => (Ok(map(&tensor(&a[0]), f64::exp)), transcendental),
        "log" => (Ok(map(&tensor(&a[0]), f64::ln)), transcendental),
        "sigmoid" => (Ok(map(&tensor(&a[0]), |x| 1.0 / (1.0 + (-x).exp()))), transcendental),
        "sum" if a.len() == 1 => (Ok(shape_data(tensor(&a[0]).sum())), 1e-15),
        "sum" => {
            let dims: Vec<isize> = ints(&a[1]).iter().map(|&d| d as isize).collect();
            (of(tensor(&a[0]).sum_dims(&dims)), 1e-15)
        }
        "cumSum" => (of(tensor(&a[0]).cum_sum(int(&a[1]) as isize)), 1e-15),
        "argMax" if a.len() == 2 => {
            let r = tensor(&a[0]).arg_max(int(&a[1]) as isize).map(|(s, v)| (s, v.iter().map(|&i| i as f64).collect()));
            (r.map_err(|e| format!("{e:?}")), exact)
        }
        "argMax" => {
            let flat = tensor(&a[0]).reshape(&[-1]).unwrap();
            let r = flat.arg_max(0).map(|(_, v)| (vec![], v.iter().map(|&i| i as f64).collect()));
            (r.map_err(|e| format!("{e:?}")), exact)
        }
        "reshape" => (of(tensor(&a[0]).reshape(&ints(&a[1]))), exact),
        "permute" => {
            let p: Vec<usize> = ints(&a[1]).iter().map(|&d| d as usize).collect();
            (of(tensor(&a[0]).permute(&p)), exact)
        }
        "slice" => match &a[1] {
            Arg::Str(spec) => (of(tensor(&a[0]).slice(spec)), exact),
            _ => return None,
        },
        "matrixMultiply" if a.len() == 2 => (of(tensor(&a[0]).matmul(&tensor(&a[1]), false, false)), 1e-12),
        "matrixMultiply" => (of(tensor(&a[0]).matmul(&tensor(&a[1]), boolean(&a[2]), boolean(&a[3]))), 1e-12),
        "choleskyDecomposition" => (of(tensor(&a[0]).cholesky()), 1e-12),
        _ => return None,
    })
}

fn map(t: &T, f: impl Fn(f64) -> f64) -> (Vec<usize>, Vec<f64>) {
    shape_data(t.map(f))
}

fn shape_data(t: T) -> (Vec<usize>, Vec<f64>) {
    (t.shape().to_vec(), t.data().to_vec())
}

#[test]
fn tensor_golden_vectors() {
    let cases = load("tensors.json");
    let mut checked = 0;
    let mut failures = Vec::new();
    for case in &cases {
        let Some((got, tolerance)) = run(case) else { continue };
        checked += 1;
        let problem = match (case.outcome(), got) {
            (Outcome::Error(_), Err(_)) => None,
            (Outcome::Error(e), Ok(_)) => Some(format!("expected error {e}, got a value")),
            (Outcome::Tensor(_), Err(e)) => Some(format!("unexpected error {e}")),
            (Outcome::Tensor(expected), Ok((shape, data))) => {
                if shape != expected.shape {
                    Some(format!("shape {shape:?} != expected {:?}", expected.shape))
                } else {
                    close(&data, &expected.data, tolerance).err()
                }
            }
        };
        if let Some(p) = problem {
            failures.push(format!("{}: {p}", case.name));
        }
    }
    println!("checked {checked} of {} tensor cases", cases.len());
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}
