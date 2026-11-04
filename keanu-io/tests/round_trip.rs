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
            state.clear(id).unwrap();
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
        |v: &[&Value]| Ok(v[0].clone()),
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

const DOUBLE_CONSTANT: &str =
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex";

/// A model of a vector constant `[0]` followed by `vertex` (JSON).
fn with_vector_and(vertex: &str) -> String {
    format!(
        r#"{{"graph": {{"vertices": [
            {{"id": {{"id": "[0]"}}, "vertexType": "{DOUBLE_CONSTANT}", "shape": ["3"],
              "parameters": [{{"name": "constant", "doubleTensorParam": {{"shape": ["3"], "values": [1, 2, 3]}}}}]}},
            {vertex}
        ]}}}}"#
    )
}

fn is_decode_error(result: keanu_io::Result<keanu_io::Loaded>) -> bool {
    matches!(result, Err(keanu_io::IoError::Decode(_)))
}

#[test]
fn a_slice_dimension_outside_the_input_is_invalid() {
    for dimension in ["4611686018427387904", "1", "-1"] {
        let json = with_vector_and(&format!(
            r#"{{"id": {{"id": "[1]"}}, "vertexType": "io.improbable.keanu.vertices.tensor.generic.nonprobabilistic.operators.unary.SliceVertex",
                "parameters": [{{"name": "inputVertex", "parentVertex": {{"id": "[0]"}}}},
                               {{"name": "dimension", "longParam": "{dimension}"}},
                               {{"name": "index", "longParam": "0"}}]}}"#
        ));
        assert!(
            matches!(load_json(&json), Err(keanu_io::IoError::InvalidParameter { ref parameter, .. }) if parameter == "dimension"),
            "dimension {dimension}"
        );
    }
}

#[test]
fn slices_with_spaces_and_a_trailing_comma_save_and_load() {
    let b = GraphBuilder::new();
    let a = b.constant(DoubleTensor::new(&[2, 4], (0..8).map(f64::from).collect()).unwrap());
    a.slice(": , 1 : 3").label("spaced");
    a.slice("1,").label("trailing");
    let (graph, state) = b.build().unwrap();
    let json = save_json(&graph, &state, &Metadata::new()).unwrap();
    let loaded = load_json(&json).unwrap();
    let state = recompute(&loaded.graph, &loaded.state);
    let value = |label: &str| {
        let id = loaded.graph.find(label).unwrap();
        flatten(state.value(id).unwrap())
    };
    assert_eq!(value("spaced"), (vec![2, 2], vec![1.0, 2.0, 5.0, 6.0]));
    assert_eq!(value("trailing"), (vec![4], vec![4.0, 5.0, 6.0, 7.0]));
}

#[test]
fn a_tri_lower_k_that_cannot_be_negated_is_a_decode_error() {
    let tri_lower = |k: &str| {
        with_vector_and(&format!(
            r#"{{"id": {{"id": "[1]"}}, "vertexType": "io.improbable.keanu.vertices.tensor.generic.nonprobabilistic.operators.unary.TriLowerVertex",
                "parameters": [{{"name": "inputVertex", "parentVertex": {{"id": "[0]"}}}},
                               {{"name": "k", "longParam": "{k}"}}]}}"#
        ))
    };
    // used to overflow (a panic in debug builds)
    assert!(is_decode_error(load_json(&tri_lower(
        "-9223372036854775808"
    ))));
    // other values still load; tril needs rank >= 2 so the vector input fails in the graph, not decoding
    assert!(matches!(
        load_json(&tri_lower("9223372036854775807")),
        Err(keanu_io::IoError::Graph(_))
    ));
}

#[test]
fn shapes_with_too_many_elements_are_decode_errors() {
    let huge_constant = format!(
        r#"{{"graph": {{"vertices": [{{"id": {{"id": "[0]"}}, "vertexType": "{DOUBLE_CONSTANT}",
            "parameters": [{{"name": "constant", "doubleTensorParam": {{"shape": ["4294967296", "4294967296"], "values": []}}}}]}}]}}}}"#
    );
    assert!(is_decode_error(load_json(&huge_constant)));

    let huge_vertex_shape = with_vector_and(
        r#"{"id": {"id": "[1]"}, "vertexType": "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GaussianVertex",
            "shape": ["4294967296", "4294967296"],
            "parameters": [{"name": "mu", "parentVertex": {"id": "[0]"}}, {"name": "sigma", "parentVertex": {"id": "[0]"}}]}"#,
    );
    assert!(is_decode_error(load_json(&huge_vertex_shape)));

    let huge_broadcast = with_vector_and(
        r#"{"id": {"id": "[1]"}, "vertexType": "io.improbable.keanu.vertices.tensor.generic.nonprobabilistic.operators.unary.BroadcastVertex",
            "parameters": [{"name": "inputVertex", "parentVertex": {"id": "[0]"}},
                           {"name": "toShape", "longArrayParam": {"values": ["4294967296", "4294967296", "3"]}}]}"#,
    );
    assert!(is_decode_error(load_json(&huge_broadcast)));
}

#[test]
fn duplicate_or_missing_vertex_ids_are_decode_errors() {
    let duplicate = with_vector_and(&format!(
        r#"{{"id": {{"id": "[0]"}}, "vertexType": "{DOUBLE_CONSTANT}",
            "parameters": [{{"name": "constant", "doubleTensorParam": {{"values": [1]}}}}]}}"#
    ));
    assert!(is_decode_error(load_json(&duplicate)));

    let missing = with_vector_and(&format!(
        r#"{{"vertexType": "{DOUBLE_CONSTANT}", "parameters": [{{"name": "constant", "doubleTensorParam": {{"values": [1]}}}}]}}"#
    ));
    assert!(is_decode_error(load_json(&missing)));
}

#[test]
fn generic_stored_values_are_unsupported_values() {
    let json = format!(
        r#"{{"graph": {{
            "vertices": [{{"id": {{"id": "[0]"}}, "vertexType": "{DOUBLE_CONSTANT}",
                "parameters": [{{"name": "constant", "doubleTensorParam": {{"values": [1]}}}}]}}],
            "defaultState": [{{"id": {{"id": "[0]"}}, "value": {{"genericVal": {{"values": ["x"]}}}}}}]}}}}"#
    );
    assert!(matches!(
        load_json(&json),
        Err(keanu_io::IoError::UnsupportedValue(_))
    ));
}

#[test]
fn values_stored_by_label_are_applied_to_the_labelled_node() {
    let b = GraphBuilder::new();
    for i in 0..200 {
        b.gaussian(0.0, 1.0).label(&format!("x{i}"));
    }
    let (graph, state) = b.build().unwrap();
    let mut model: serde_json::Value =
        serde_json::from_str(&save_json(&graph, &state, &Metadata::new()).unwrap()).unwrap();
    model["graph"]["defaultState"] = (0..200)
        .map(|i| {
            serde_json::json!({
                "vertexLabel": format!("x{i}"),
                "value": {"doubleVal": {"values": [f64::from(i)]}},
                "isObserved": i % 2 == 0
            })
        })
        .collect();
    let loaded = load_json(&model.to_string()).unwrap();
    for id in loaded.graph.ids() {
        let Some(label) = loaded.graph.node(id).label() else {
            continue;
        };
        let i: u32 = label[1..].parse().unwrap();
        assert_eq!(
            loaded.state.value(id),
            Some(&Value::F64(DoubleTensor::scalar(f64::from(i))))
        );
        assert_eq!(loaded.state.is_observed(id), i % 2 == 0);
    }
}

/// The labels of a graph's nodes.
fn labels(graph: &Graph) -> Vec<Option<String>> {
    graph
        .ids()
        .map(|id| graph.node(id).label().map(str::to_owned))
        .collect()
}

#[test]
fn unlabelled_proxies_round_trip_without_an_invented_label() {
    let b = GraphBuilder::new();
    // the label the saver used to invent for the proxy (node 1)
    let x = b.scalar(1.5).label("proxy_1");
    let proxy = b.double_proxy(&[]);
    b.connect(proxy, x);
    (proxy * 2.0).label("doubled");
    let (graph, state) = b.build().unwrap();

    let reloaded =
        load_protobuf(&save_protobuf(&graph, &state, &Metadata::new()).unwrap()).unwrap();
    assert_eq!(labels(&reloaded.graph), labels(&graph));
    let reloaded = load_json(&save_json(&graph, &state, &Metadata::new()).unwrap()).unwrap();
    assert_eq!(labels(&reloaded.graph), labels(&graph));

    // files saved by earlier versions, with an invented label parameter, load without it
    let mut old: serde_json::Value =
        serde_json::from_str(&save_json(&graph, &state, &Metadata::new()).unwrap()).unwrap();
    let vertices = old["graph"]["vertices"].as_array_mut().unwrap();
    let proxy_vertex = vertices
        .iter_mut()
        .find(|v| {
            v["vertexType"]
                .as_str()
                .unwrap()
                .ends_with("DoubleProxyVertex")
        })
        .unwrap();
    for param in proxy_vertex["parameters"].as_array_mut().unwrap() {
        if param["name"] == "label" {
            param["stringParam"] = "proxy_7".into();
        }
    }
    let reloaded = load_json(&old.to_string()).unwrap();
    assert_eq!(labels(&reloaded.graph), labels(&graph));
}

/// Saves and loads a model `cycles` times, checking each load has the same number of nodes, values and log prob.
fn assert_save_load_is_idempotent(name: &str, graph: Graph, state: State, cycles: usize) {
    let metadata = Metadata::new();
    let first = load_protobuf(&save_protobuf(&graph, &state, &metadata).unwrap()).unwrap();
    let expected = summarise(&first.graph, &first.state);
    let nodes = first.graph.len();
    let (mut graph, mut state) = (first.graph, first.state);
    for cycle in 0..cycles {
        let loaded = if cycle % 2 == 0 {
            load_json(&save_json(&graph, &state, &metadata).unwrap()).unwrap()
        } else {
            load_protobuf(&save_protobuf(&graph, &state, &metadata).unwrap()).unwrap()
        };
        assert_eq!(
            loaded.graph.len(),
            nodes,
            "{name}: nodes after cycle {cycle}"
        );
        assert_same(
            &format!("{name} cycle {cycle}"),
            &summarise(&loaded.graph, &loaded.state),
            &expected,
        );
        (graph, state) = (loaded.graph, loaded.state);
    }
}

#[test]
fn saving_and_loading_repeatedly_does_not_grow_the_graph() {
    let b = GraphBuilder::new();
    let square = b
        .constant(
            DoubleTensor::new(&[3, 3], vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0]).unwrap(),
        )
        .label("square");
    square.cholesky().cholesky_inverse().label("inverse");
    let sigma = b.half_gaussian(1.5).label("sigma");
    b.set_value(sigma, DoubleTensor::scalar(0.8));
    let scale = b.half_cauchy(2.0).label("scale");
    b.set_value(scale, DoubleTensor::scalar(1.1));
    let (graph, state) = b.build().unwrap();

    let loaded = load_protobuf(&save_protobuf(&graph, &state, &Metadata::new()).unwrap()).unwrap();
    assert_eq!(loaded.graph.len(), graph.len());
    assert_save_load_is_idempotent("rust built", graph, state, 4);

    // and the models saved by Java (whose half distributions share a zero location constant, and whose cholesky
    // inverses are in Java's lower triangle form)
    for (name, graph, state) in cases() {
        assert_save_load_is_idempotent(&name, graph, state, 3);
    }
}

#[cfg(feature = "dot")]
#[test]
fn dot_labels_are_escaped() {
    let b = GraphBuilder::new();
    b.gaussian(0.0, 1.0).label("say \"hi\"\\\nbye");
    let (graph, _) = b.build().unwrap();
    let dot = keanu_io::to_dot(&graph, None, &Metadata::new());
    assert!(dot.contains(r#"2[label="say \"hi\"\\\nbye"]"#), "{dot}");
}
