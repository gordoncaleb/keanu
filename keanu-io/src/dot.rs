//! Graphviz DOT export, equivalent to Java's `DotSaver` (but in a deterministic order).

use crate::Metadata;
use keanu_graph::op::{BinaryOp, DistributionKind, LogicOp, Op};
use keanu_graph::{Graph, NodeId, State, Value};
use std::fmt::Write;

/// The names of a distribution's parameters, in parent order, used to label the edges into it.
fn parameter_names(kind: &DistributionKind) -> &'static [&'static str] {
    use DistributionKind::*;
    match kind {
        Gaussian | LogNormal => &["mu", "sigma"],
        HalfGaussian => &["sigma"],
        Cauchy | Pareto => &["location", "scale"],
        HalfCauchy => &["scale"],
        Beta | InverseGamma => &["alpha", "beta"],
        ChiSquared => &["k"],
        Dirichlet => &["concentration"],
        Exponential => &["rate"],
        Gamma => &["theta", "k"],
        Laplace => &["mu", "beta"],
        Logistic => &["mu", "s"],
        MultivariateGaussian => &["mu", "covariance"],
        StudentT => &["v"],
        Triangular => &["xMin", "xMax", "c"],
        Uniform => &["xMin", "xMax"],
        Bernoulli => &["probTrue"],
        Binomial => &["p", "n"],
        Geometric => &["p"],
        Multinomial => &["n", "p"],
        Poisson => &["mu"],
        UniformInt => &["min", "max"],
        _ => &[],
    }
}

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
        Op::Distribution(d) => format!("{:?}", d.kind)
            .split('(')
            .next()
            .unwrap_or("Distribution")
            .to_owned(),
        other => other.name().to_owned(),
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

/// A Graphviz DOT description of the graph. With `state`, scalar values are shown in the node labels (as Java's
/// `DotSaver` with `saveValues`); constants always show their scalar value.
pub fn to_dot(graph: &Graph, state: Option<&State>, metadata: &Metadata) -> String {
    let mut out = String::from("digraph BayesianNetwork {\n");
    if !metadata.is_empty() {
        out.push_str("// Model metadata:\n");
        for (key, value) in metadata {
            let _ = writeln!(out, "// {key}={value}");
        }
    }
    for child in graph.ids() {
        let node = graph.node(child);
        let names = match node.op() {
            Op::Distribution(d) => parameter_names(&d.kind),
            _ => &[],
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
    let node = graph.node(id);
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
