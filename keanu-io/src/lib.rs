//! Saving and loading [Keanu](https://github.com/gordoncaleb/keanu) models.
//!
//! The format is Keanu's versioned protobuf schema (`proto/keanu/graph.proto`): binary protobuf ([`load_protobuf`],
//! [`save_protobuf`], feature `protobuf`) or its JSON mapping ([`load_json`], [`save_json`], feature `json`). Each
//! node is saved with its typed operation, in graph order, with its label; the state's values and observations and
//! a metadata map are saved alongside. [`to_dot`] (feature `dot`) exports a Graphviz description of a graph.
//!
//! Loading checks the model against [`LoadOptions`], whose defaults protect against untrusted files describing
//! enormous nodes or practically endless scans; [`load_protobuf_with`] and [`load_json_with`] take other limits.
//!
//! Every operation and distribution can be saved except lambda nodes and custom distributions, which return
//! [`IoError::NotSaveable`]. Files written by a newer version of the format return
//! [`IoError::UnsupportedVersion`].

#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(any(feature = "protobuf", feature = "json"))]
mod codes;
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
    include!("generated/keanu.rs");
    #[cfg(feature = "json")]
    include!("generated/keanu.serde.rs");
}

#[cfg(feature = "dot")]
pub use dot::to_dot;
pub use error::{IoError, Result};
#[cfg(any(feature = "protobuf", feature = "json"))]
use load::load_model;
#[cfg(any(feature = "protobuf", feature = "json"))]
pub use load::{LoadOptions, Loaded};
#[cfg(any(feature = "protobuf", feature = "json"))]
use save::save_model;

use std::collections::BTreeMap;

/// Metadata saved with a model (e.g. a name or version).
pub type Metadata = BTreeMap<String, String>;

/// Loads a model saved as binary protobuf, within the default [`LoadOptions`].
#[cfg(feature = "protobuf")]
pub fn load_protobuf(bytes: &[u8]) -> Result<Loaded> {
    load_protobuf_with(bytes, &LoadOptions::default())
}

/// Loads a model saved as binary protobuf, within the given limits.
#[cfg(feature = "protobuf")]
pub fn load_protobuf_with(bytes: &[u8], options: &LoadOptions) -> Result<Loaded> {
    let model = <proto::Model as prost::Message>::decode(bytes)
        .map_err(|e| IoError::Decode(e.to_string()))?;
    load_model(&model, options)
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

/// Loads a model saved as JSON, within the default [`LoadOptions`].
#[cfg(feature = "json")]
pub fn load_json(text: &str) -> Result<Loaded> {
    load_json_with(text, &LoadOptions::default())
}

/// Loads a model saved as JSON, within the given limits.
#[cfg(feature = "json")]
pub fn load_json_with(text: &str, options: &LoadOptions) -> Result<Loaded> {
    let model: proto::Model =
        serde_json::from_str(text).map_err(|e| IoError::Decode(e.to_string()))?;
    load_model(&model, options)
}

/// Saves a model as JSON, with the values in `state`.
#[cfg(feature = "json")]
pub fn save_json(
    graph: &keanu_graph::Graph,
    state: &keanu_graph::State,
    metadata: &Metadata,
) -> Result<String> {
    let model = save_model(graph, state, metadata)?;
    let mut json = serde_json::to_value(&model).map_err(encode_error)?;
    json_non_finite::restore(&model, &mut json);
    serde_json::to_string_pretty(&json).map_err(encode_error)
}

/// A failure to write JSON is an encoding error, not a decoding one.
#[cfg(feature = "json")]
fn encode_error(e: serde_json::Error) -> IoError {
    IoError::Encode(e.to_string())
}

/// serde_json writes non-finite doubles as `null`, but the protobuf JSON mapping writes the strings `"NaN"`,
/// `"Infinity"` and `"-Infinity"`. The doubles of the schema are patched from the model after serialising: tensor
/// values, the parameters of `approxEq` and `replaceNan` nodes, and KDE samples and bandwidths.
#[cfg(feature = "json")]
mod json_non_finite {
    use crate::proto;
    use crate::proto::node::Op;
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

    /// Patches a double field.
    fn double(v: f64, json: Option<&mut Value>) {
        if let Some(slot) = json.filter(|_| !v.is_finite()) {
            *slot = number(v);
        }
    }

    /// Patches a repeated double field.
    fn doubles(values: &[f64], json: Option<&mut Value>) {
        if let Some(Value::Array(slots)) = json {
            for (slot, &v) in slots.iter_mut().zip(values) {
                if !v.is_finite() {
                    *slot = number(v);
                }
            }
        }
    }

    fn tensor(t: &proto::Tensor, json: Option<&mut Value>) {
        if let Some(proto::tensor::Values::F64(v)) = &t.values {
            doubles(
                &v.values,
                json.and_then(|j| j.get_mut("f64"))
                    .and_then(|j| j.get_mut("values")),
            );
        }
    }

    pub(crate) fn restore(model: &proto::Model, json: &mut Value) {
        let Some(graph) = &model.graph else { return };
        let Some(json_graph) = json.get_mut("graph") else {
            return;
        };
        if let Some(Value::Array(nodes)) = json_graph.get_mut("nodes") {
            for (node, json_node) in graph.nodes.iter().zip(nodes.iter_mut()) {
                match &node.op {
                    Some(Op::Constant(t)) => tensor(t, json_node.get_mut("constant")),
                    Some(Op::ApproxEq(p)) => double(
                        p.epsilon,
                        json_node
                            .get_mut("approxEq")
                            .and_then(|j| j.get_mut("epsilon")),
                    ),
                    Some(Op::ReplaceNan(p)) => double(
                        p.value,
                        json_node
                            .get_mut("replaceNan")
                            .and_then(|j| j.get_mut("value")),
                    ),
                    Some(Op::Distribution(proto::Distribution { kde: Some(kde), .. })) => {
                        let json_kde = json_node
                            .get_mut("distribution")
                            .and_then(|j| j.get_mut("kde"));
                        if let Some(json_kde) = json_kde {
                            doubles(&kde.samples, json_kde.get_mut("samples"));
                            double(kde.bandwidth, json_kde.get_mut("bandwidth"));
                        }
                    }
                    _ => {}
                }
            }
        }
        if let Some(Value::Array(stored)) = json_graph.get_mut("values") {
            for (value, json_value) in graph.values.iter().zip(stored.iter_mut()) {
                if let Some(t) = &value.value {
                    tensor(t, json_value.get_mut("value"));
                }
            }
        }
    }
}

#[cfg(all(test, feature = "json"))]
mod tests {
    use super::*;

    #[test]
    fn json_serialisation_failures_are_encode_errors() {
        // maps with non-string keys can't be written as JSON
        let failure = serde_json::to_value(BTreeMap::from([(vec![1u8], 1)])).unwrap_err();
        let error = encode_error(failure);
        assert!(matches!(error, IoError::Encode(_)), "{error:?}");
        assert!(error.to_string().starts_with("cannot encode the model"));
    }
}
