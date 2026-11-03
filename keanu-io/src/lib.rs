//! Saving and loading [Keanu](https://github.com/gordoncaleb/keanu) models.
//!
//! The format is Keanu's protobuf schema (`proto/io/improbable/mir/*.proto`), shared with the Java implementation:
//! binary protobuf ([`load_protobuf`], [`save_protobuf`], feature `protobuf`) or its JSON mapping ([`load_json`],
//! [`save_json`], feature `json`). Vertex types are identified by their Java class names and parameters by the
//! Java parameter names, so files round-trip between the two implementations. [`to_dot`] (feature `dot`) exports a
//! Graphviz description of a graph.
//!
//! Not supported: `PrintVertex`, `GetBooleanIndexVertex`, lambda / CPT / categorical nodes (which Java can't save
//! either), and files written by older Java versions.

#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(any(feature = "protobuf", feature = "json"))]
mod classes;
mod error;
#[cfg(any(feature = "protobuf", feature = "json"))]
mod load;
#[cfg(any(feature = "protobuf", feature = "json"))]
mod save;

#[cfg(feature = "dot")]
mod dot;

/// The protobuf types of the saved model schema. Internal: the public API is the format level `load_*` / `save_*`
/// functions, so the generated types (and the prost version) can change without breaking users.
#[cfg(any(feature = "protobuf", feature = "json"))]
#[allow(
    clippy::all,
    missing_debug_implementations,
    missing_docs,
    dead_code,
    unreachable_pub
)]
pub(crate) mod proto {
    include!("generated/mir.rs");
    #[cfg(feature = "json")]
    include!("generated/mir.serde.rs");
}

#[cfg(feature = "dot")]
pub use dot::to_dot;
pub use error::{IoError, Result};
#[cfg(any(feature = "protobuf", feature = "json"))]
pub use load::Loaded;
#[cfg(any(feature = "protobuf", feature = "json"))]
use load::load_model;
#[cfg(any(feature = "protobuf", feature = "json"))]
use save::save_model;

use std::collections::BTreeMap;

/// Metadata saved with a model (e.g. a name or version).
pub type Metadata = BTreeMap<String, String>;

/// Loads a model saved as binary protobuf.
#[cfg(feature = "protobuf")]
pub fn load_protobuf(bytes: &[u8]) -> Result<Loaded> {
    let model = <proto::ProtoModel as prost::Message>::decode(bytes)
        .map_err(|e| IoError::Decode(e.to_string()))?;
    load_model(&model)
}

/// Saves a model as binary protobuf, with the values in `state` (observed values are saved as observations).
#[cfg(feature = "protobuf")]
pub fn save_protobuf(
    graph: &keanu_graph::Graph,
    state: &keanu_graph::State,
    metadata: &Metadata,
) -> Result<Vec<u8>> {
    Ok(prost::Message::encode_to_vec(&save_model(
        graph, state, metadata,
    )?))
}

/// Loads a model saved as JSON.
#[cfg(feature = "json")]
pub fn load_json(text: &str) -> Result<Loaded> {
    let model: proto::ProtoModel =
        serde_json::from_str(text).map_err(|e| IoError::Decode(e.to_string()))?;
    load_model(&model)
}

/// Saves a model as JSON, with the values in `state`.
#[cfg(feature = "json")]
pub fn save_json(
    graph: &keanu_graph::Graph,
    state: &keanu_graph::State,
    metadata: &Metadata,
) -> Result<String> {
    let model = save_model(graph, state, metadata)?;
    let mut json = serde_json::to_value(&model).map_err(|e| IoError::Decode(e.to_string()))?;
    json_non_finite::restore(&model, &mut json);
    serde_json::to_string_pretty(&json).map_err(|e| IoError::Decode(e.to_string()))
}

/// serde_json writes non-finite doubles as `null`, but the protobuf JSON mapping (and Java) write the strings
/// `"NaN"`, `"Infinity"` and `"-Infinity"`. The schema's only doubles are tensor values and double parameters, so
/// those positions are patched from the model after serialising.
#[cfg(feature = "json")]
mod json_non_finite {
    use crate::proto;
    use serde_json::Value;

    fn number(v: f64) -> Value {
        if v.is_nan() {
            "NaN".into()
        } else if v == f64::INFINITY {
            "Infinity".into()
        } else if v == f64::NEG_INFINITY {
            "-Infinity".into()
        } else {
            serde_json::json!(v)
        }
    }

    fn tensor(t: &proto::DoubleTensor, json: Option<&mut Value>) {
        if let Some(Value::Array(values)) = json.and_then(|j| j.get_mut("values")) {
            for (slot, &v) in values.iter_mut().zip(&t.values) {
                if !v.is_finite() {
                    *slot = number(v);
                }
            }
        }
    }

    pub(crate) fn restore(model: &proto::ProtoModel, json: &mut Value) {
        let Some(graph) = &model.graph else { return };
        let Some(json_graph) = json.get_mut("graph") else {
            return;
        };
        if let Some(Value::Array(vertices)) = json_graph.get_mut("vertices") {
            for (vertex, json_vertex) in graph.vertices.iter().zip(vertices.iter_mut()) {
                let Some(Value::Array(params)) = json_vertex.get_mut("parameters") else {
                    continue;
                };
                for (param, json_param) in vertex.parameters.iter().zip(params.iter_mut()) {
                    match &param.param {
                        Some(proto::named_param::Param::DoubleTensorParam(t)) => {
                            tensor(t, json_param.get_mut("doubleTensorParam"))
                        }
                        Some(proto::named_param::Param::DoubleParam(v)) if !v.is_finite() => {
                            json_param["doubleParam"] = number(*v);
                        }
                        _ => {}
                    }
                }
            }
        }
        if let Some(Value::Array(stored)) = json_graph.get_mut("defaultState") {
            for (value, json_value) in graph.default_state.iter().zip(stored.iter_mut()) {
                if let Some(proto::vertex_value::ValueType::DoubleVal(t)) =
                    value.value.as_ref().and_then(|v| v.value_type.as_ref())
                {
                    tensor(
                        t,
                        json_value
                            .get_mut("value")
                            .and_then(|v| v.get_mut("doubleVal")),
                    );
                }
            }
        }
    }
}
