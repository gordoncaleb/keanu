//! Saving and loading round trips.
//!
//! - Rust save -> Rust load reproduces every golden model and a model built in Rust with operations that have no
//!   single Java vertex (and so are saved as small Java subgraphs).
//! - Format stability: the Rust-written files and the values Rust computes for them are committed as fixtures in
//!   `tests/fixtures/rust-written`, and the test fails if saving produces different files. (Until the Java
//!   implementation was removed, a Java test also loaded these files; see the `java-final` tag.)
//!   This test fails if the fixtures are stale; run with `KEANU_REGENERATE=1` to update them.

use keanu_distributions::{Kde, KeanuRng};
use keanu_graph::op::Op;
use keanu_graph::{Graph, GraphBuilder, Model, State, Value};
use keanu_io::{Metadata, load_json, load_protobuf, save_json, save_protobuf};
use keanu_parity::{close, golden_dir};
use keanu_tensor::{DoubleTensor, IntTensor};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn recompute(graph: &Graph, state: &State) -> State {
    let mut state = state.clone();
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

/// The log prob and the value of every labelled node, the facts a round trip must preserve.
#[derive(Debug, PartialEq)]
struct Summary {
    log_prob: f64,
    values: BTreeMap<String, (Vec<usize>, Vec<f64>)>,
}

fn flatten(value: &Value) -> (Vec<usize>, Vec<f64>) {
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

fn summarise(graph: &Graph, state: &State) -> Summary {
    let state = recompute(graph, state);
    let model = Model::new(graph.clone(), &state).unwrap();
    let values = graph
        .ids()
        .filter_map(|id| {
            graph
                .node(id)
                .label()
                .map(|l| (l.to_owned(), flatten(state.value(id).unwrap())))
        })
        .collect();
    Summary {
        log_prob: model.log_prob(&state).unwrap(),
        values,
    }
}

fn assert_same(context: &str, actual: &Summary, expected: &Summary) {
    close(&[actual.log_prob], &[expected.log_prob], 1e-12)
        .unwrap_or_else(|e| panic!("{context} log prob: {e}"));
    assert_eq!(
        actual.values.keys().collect::<Vec<_>>(),
        expected.values.keys().collect::<Vec<_>>(),
        "{context} labels"
    );
    for (label, (shape, data)) in &actual.values {
        let (expected_shape, expected_data) = &expected.values[label];
        assert_eq!(shape, expected_shape, "{context} {label} shape");
        close(data, expected_data, 1e-12).unwrap_or_else(|e| panic!("{context} {label}: {e}"));
    }
}

/// A model using operations without a single Java vertex type, and Rust-only conveniences.
fn rust_built_model() -> (Graph, State) {
    let b = GraphBuilder::new();
    let a = b
        .constant(DoubleTensor::new(&[2, 3], vec![0.2, 0.5, 0.7, 0.3, 0.9, 0.4]).unwrap())
        .label("a");
    let row = b
        .constant(DoubleTensor::vector(vec![0.3, 0.6, 0.9]))
        .label("row");
    let square = b
        .constant(
            DoubleTensor::new(&[3, 3], vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0]).unwrap(),
        )
        .label("square");
    (-a).label("neg");
    a.reciprocal().label("reciprocal");
    a.sqrt().label("sqrt");
    a.transpose().label("transpose");
    a.maximum(row).label("broadcastMax");
    a.minimum(row).label("broadcastMin");
    a.slice("::-1, -1::-2").label("slice");
    a.slice("..., 1").label("sliceEllipsis");
    square
        .cholesky()
        .cholesky_inverse()
        .label("fullCholeskyInverse");
    a.tril(1).label("tril");
    a.sum_axes(&[-1]).label("sumAxes");
    let i = b.int_constant(IntTensor::vector(vec![-7, 3, 5])).label("i");
    (-i).label("intNeg");
    a.gt(row).to_double_mask().label("mask");

    let mu = b.gaussian(0.0, 1.0).label("mu");
    b.set_value(mu, DoubleTensor::scalar(0.3));
    let sigma = b.half_gaussian(1.5).label("sigma");
    b.set_value(sigma, DoubleTensor::scalar(0.8));
    let scale = b.half_cauchy(2.0).label("scale");
    b.set_value(scale, DoubleTensor::scalar(1.1));
    let kde = b
        .kde(Kde::new(vec![0.1, 0.4, 0.5, 1.2], 0.3).unwrap(), &[])
        .label("kde");
    b.set_value(kde, DoubleTensor::scalar(0.45));
    let y = b
        .gaussian_with_shape(&[2, 3], a * mu + 1.0, sigma)
        .label("y");
    b.observe(
        y,
        DoubleTensor::new(&[2, 3], vec![1.1, 1.4, 1.0, 1.3, 1.5, 0.9]).unwrap(),
    );
    let counts = b
        .poisson_with_shape(&[3], (row * scale).exp())
        .label("counts");
    b.observe(counts, IntTensor::vector(vec![1, 2, 4]));
    b.build().unwrap()
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rust-written")
}

/// (name, graph, state) of every model to round trip: the golden models (loaded from Java's files) and the
/// Rust-built model.
fn cases() -> Vec<(String, Graph, State)> {
    let text = std::fs::read_to_string(golden_dir().join("models.json")).unwrap();
    let root: serde_json::Value = serde_json::from_str(&text).unwrap();
    let mut cases: Vec<(String, Graph, State)> = root["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| {
            let pb = std::fs::read(golden_dir().join(case["protobuf"].as_str().unwrap())).unwrap();
            let loaded = load_protobuf(&pb).unwrap();
            (
                case["name"].as_str().unwrap().to_owned(),
                loaded.graph,
                loaded.state,
            )
        })
        .collect();
    let (graph, state) = rust_built_model();
    cases.push(("rustBuilt".to_owned(), graph, state));
    cases
}

#[test]
fn rust_save_then_rust_load_preserves_every_model() {
    for (name, graph, state) in cases() {
        let expected = summarise(&graph, &state);
        let metadata: Metadata = [("name".to_owned(), name.clone())].into_iter().collect();

        let pb = save_protobuf(&graph, &state, &metadata).unwrap();
        let reloaded = load_protobuf(&pb).unwrap();
        assert_eq!(reloaded.metadata, metadata);
        assert_same(
            &format!("{name} protobuf"),
            &summarise(&reloaded.graph, &reloaded.state),
            &expected,
        );

        let json = save_json(&graph, &state, &metadata).unwrap();
        let reloaded = load_json(&json).unwrap();
        assert_same(
            &format!("{name} json"),
            &summarise(&reloaded.graph, &reloaded.state),
            &expected,
        );

        // saving a loaded model gives the same file again (the format is stable)
        assert_eq!(
            save_protobuf(&reloaded.graph, &reloaded.state, &metadata).unwrap(),
            save_protobuf(
                &load_protobuf(&pb).unwrap().graph,
                &load_protobuf(&pb).unwrap().state,
                &metadata
            )
            .unwrap()
        );
    }
}

/// Writes (or checks) the Rust-written files and the values Rust computes for them, for the Java side to load.
#[test]
fn rust_written_fixtures_for_java_are_up_to_date() {
    let regenerate = std::env::var_os("KEANU_REGENERATE").is_some();
    let dir = fixtures();
    std::fs::create_dir_all(&dir).unwrap();
    let mut stale = Vec::new();
    let mut expected = serde_json::Map::new();
    for (name, graph, state) in cases() {
        let metadata: Metadata = [("name".to_owned(), name.clone())].into_iter().collect();
        let summary = summarise(&graph, &state);
        let mut values = serde_json::Map::new();
        for (label, (shape, data)) in &summary.values {
            values.insert(label.clone(), serde_json::json!({ "shape": shape, "data": data.iter().map(|d| if d.is_finite() { serde_json::json!(d) } else { serde_json::json!(d.to_string()) }).collect::<Vec<_>>() }));
        }
        expected.insert(
            name.clone(),
            serde_json::json!({ "logProb": summary.log_prob, "values": values }),
        );
        let files = [
            (
                format!("{name}.pb"),
                save_protobuf(&graph, &state, &metadata).unwrap(),
            ),
            (
                format!("{name}.json"),
                save_json(&graph, &state, &metadata).unwrap().into_bytes(),
            ),
        ];
        for (file, bytes) in files {
            let path = dir.join(&file);
            if std::fs::read(&path).ok().as_deref() != Some(bytes.as_slice()) {
                if regenerate {
                    std::fs::write(&path, &bytes).unwrap();
                } else {
                    stale.push(file);
                }
            }
        }
    }
    let expected_text =
        serde_json::to_string_pretty(&serde_json::Value::Object(expected)).unwrap() + "\n";
    let expected_path = dir.join("expected.json");
    if std::fs::read_to_string(&expected_path).ok().as_deref() != Some(expected_text.as_str()) {
        if regenerate {
            std::fs::write(&expected_path, expected_text).unwrap();
        } else {
            stale.push("expected.json".into());
        }
    }
    assert!(
        stale.is_empty(),
        "stale fixtures {stale:?}: run KEANU_REGENERATE=1 cargo test -p keanu-io --test round_trip"
    );
}

#[test]
fn nodes_without_a_saved_form_are_reported() {
    let b = GraphBuilder::new();
    let x = b.scalar(1.0);
    b.lambda(
        "double it",
        &[x.id()],
        keanu_graph::DType::F64,
        &[],
        |v: &[Value]| Ok(v[0].clone()),
    );
    let (graph, state) = b.build().unwrap();
    assert!(matches!(
        save_protobuf(&graph, &state, &Metadata::new()),
        Err(keanu_io::IoError::NotSaveable(_))
    ));
}

#[test]
fn unsupported_vertex_types_are_reported() {
    let json = r#"{"graph": {"vertices": [{"id": {"id": "[0]"}, "vertexType": "io.improbable.keanu.vertices.tensor.generic.nonprobabilistic.PrintVertex"}]}}"#;
    assert!(
        matches!(load_json(json), Err(keanu_io::IoError::UnsupportedVertex(t)) if t == "PrintVertex")
    );
}

#[cfg(feature = "dot")]
#[test]
fn dot_export_describes_nodes_edges_and_metadata() {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0).label("model.mu");
    b.set_value(mu, DoubleTensor::scalar(0.5));
    let sum = mu + 2.0;
    let _ = b.gaussian(sum, 1.0);
    let (graph, state) = b.build().unwrap();
    let metadata: Metadata = [("author".to_owned(), "keanu".to_owned())]
        .into_iter()
        .collect();

    let dot = keanu_io::to_dot(&graph, None, &metadata);
    assert!(dot.starts_with("digraph BayesianNetwork {\n// Model metadata:\n// author=keanu\n"));
    assert!(dot.ends_with('}'));
    assert!(dot.contains("<0> -> <2> [label=mu]"));
    assert!(dot.contains("<1> -> <2> [label=sigma]"));
    assert!(dot.contains("2[label=\"mu\"]"), "{dot}");
    assert!(dot.contains("[label=\"+\"]"));
    assert!(dot.contains("0[label=\"0.0\"]"));

    let with_values = keanu_io::to_dot(&graph, Some(&state), &Metadata::new());
    assert!(
        with_values.contains("2[label=\"0.5 (mu)\"]"),
        "{with_values}"
    );
}
