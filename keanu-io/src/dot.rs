//! Graphviz DOT export.

use crate::Metadata;
use keanu_graph::op::{BinaryOp, DistributionKind, LogicOp, Op};
use keanu_graph::{Graph, NodeId, State, Value};
use std::fmt::Write;

/// A short description of a node: the binary operators and logic as symbols, distributions by name, and other
/// ops by their op name.
fn describe(op: &Op) -> String {
    match op {
        Op::Binary(BinaryOp::Add) => "+".into(),
        Op::Binary(BinaryOp::Sub) => "-".into(),
        Op::Binary(BinaryOp::Mul) => "*".into(),
        Op::Binary(BinaryOp::Div) => "/".into(),
        Op::Logic(LogicOp::And) => "AND".into(),
        Op::Logic(LogicOp::Or) => "OR".into(),
        Op::Logic(LogicOp::Xor) => "XOR".into(),
        Op::Unary(u) => format!("{u:?}"),
        Op::Binary(b) => format!("{b:?}"),
        Op::Distribution(d) => distribution_name(&d.kind),
        other => other.name().to_owned(),
    }
}

/// The name of a distribution (its `DistributionKind` variant), without formatting its data (a KDE's samples).
fn distribution_name(kind: &DistributionKind) -> String {
    match kind {
        DistributionKind::Kde(_) => "Kde".into(),
        DistributionKind::Custom(_) => "Custom".into(),
        DistributionKind::Truncated(_) => "Truncated".into(),
        DistributionKind::Censored(_) => "Censored".into(),
        DistributionKind::Mixture(_) => "Mixture".into(),
        // the other kinds are unit variants, which format as their name
        kind => format!("{kind:?}"),
    }
}

fn scalar_value(value: &Value) -> Option<String> {
    match value {
        Value::F64(t) if t.len() == 1 && t.rank() == 0 => Some(format!("{:?}", t.as_slice()[0])),
        Value::I64(t) if t.len() == 1 && t.rank() == 0 => Some(t.as_slice()[0].to_string()),
        Value::Bool(t) if t.len() == 1 && t.rank() == 0 => Some(t.as_slice()[0].to_string()),
        _ => None,
    }
}

/// A Graphviz DOT description of the graph. With `state`, scalar values are shown in the node labels; constants
/// always show their scalar value. Distribution edges are labelled with the parameter names.
pub fn to_dot(graph: &Graph, state: Option<&State>, metadata: &Metadata) -> String {
    let mut out = String::from("digraph BayesianNetwork {\n");
    if !metadata.is_empty() {
        out.push_str("// Model metadata:\n");
        for (key, value) in metadata {
            let _ = writeln!(out, "// {key}={value}");
        }
    }
    for child in graph.ids() {
        let node = &graph[child];
        let names = match node.op() {
            Op::Distribution(d) => d.kind.parameter_names().unwrap_or_default(),
            _ => Default::default(),
        };
        for (i, parent) in node.parents().iter().enumerate() {
            match names.get(i) {
                Some(name) => {
                    let _ = writeln!(
                        out,
                        "<{}> -> <{}> [label={name}]",
                        parent.index(),
                        child.index()
                    );
                }
                None => {
                    let _ = writeln!(out, "<{}> -> <{}>", parent.index(), child.index());
                }
            }
        }
    }
    for id in graph.ids() {
        let _ = writeln!(
            out,
            "{}[label=\"{}\"]",
            id.index(),
            escape(&node_label(graph, state, id))
        );
    }
    out.push('}');
    out
}

/// Escapes text for a double-quoted DOT string: quotes and backslashes are escaped, and line breaks become `\n`.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => {}
            c => out.push(c),
        }
    }
    out
}

fn node_label(graph: &Graph, state: Option<&State>, id: NodeId) -> String {
    let node = &graph[id];
    let name = node
        .label()
        .map(|l| l.rsplit('.').next().unwrap_or(l).to_owned());
    let value = match node.op() {
        Op::Constant(v) => scalar_value(v),
        _ => state.and_then(|s| s.value(id)).and_then(scalar_value),
    };
    match (value, name) {
        (Some(v), Some(n)) => format!("{v} ({n})"),
        (Some(v), None) if matches!(node.op(), Op::Constant(_)) => v,
        (Some(v), None) => format!("{v} ({})", describe(node.op())),
        (None, Some(n)) => n,
        (None, None) => describe(node.op()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use keanu_distributions::Kde;
    use std::sync::Arc;

    #[test]
    fn distribution_names_are_the_variant_names() {
        assert_eq!(distribution_name(&DistributionKind::Gaussian), "Gaussian");
        assert_eq!(
            distribution_name(&DistributionKind::MultivariateGaussian),
            "MultivariateGaussian"
        );
        assert_eq!(
            distribution_name(&DistributionKind::Categorical),
            "Categorical"
        );
        let kde = DistributionKind::Kde(Arc::new(Kde::new(vec![0.5; 10_000], 0.1).unwrap()));
        assert_eq!(distribution_name(&kde), "Kde");
        for (kind, name) in [
            (
                DistributionKind::Truncated(Arc::new(DistributionKind::Gaussian)),
                "Truncated",
            ),
            (
                DistributionKind::Censored(Arc::new(DistributionKind::Gaussian)),
                "Censored",
            ),
            (
                DistributionKind::Mixture(Arc::new(DistributionKind::Gaussian)),
                "Mixture",
            ),
            (DistributionKind::Weibull, "Weibull"),
            (DistributionKind::VonMises, "VonMises"),
            (DistributionKind::NegativeBinomial, "NegativeBinomial"),
            (DistributionKind::OrderedLogistic, "OrderedLogistic"),
        ] {
            assert_eq!(distribution_name(&kind), name);
        }
    }
}
