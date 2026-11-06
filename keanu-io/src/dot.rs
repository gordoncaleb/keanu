//! Graphviz DOT export.

use crate::Metadata;
use keanu_graph::op::{BinaryOp, DistributionKind, LogicOp, OdeBody, Op, ScanBody};
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
        Op::Bessel { kind, order } => format!("{kind:?}({order})"),
        Op::Distribution(d) => distribution_name(&d.kind),
        Op::LogDensity(kind) => format!("log_density({})", distribution_name(kind)),
        other => other.name().to_owned(),
    }
}

/// The name of a distribution (its `DistributionKind` variant), without formatting its data (a KDE's samples).
fn distribution_name(kind: &DistributionKind) -> String {
    if let Some((nested, _)) = kind.nested() {
        // `Nested` variants are named as the kinds they tag
        return format!("{nested:?}");
    }
    match kind {
        DistributionKind::Kde(_) => "Kde".into(),
        DistributionKind::Icar(_) => "Icar".into(),
        DistributionKind::Car(_) => "Car".into(),
        DistributionKind::Custom(_) => "Custom".into(),
        DistributionKind::Simulator(_) => "Simulator".into(),
        DistributionKind::MixtureOf(_) => "MixtureOf".into(),
        DistributionKind::Transformed(..) => "Transformed".into(),
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
/// always show their scalar value. Distribution edges are labelled with the parameter names. A scan's or ODE's body
/// is drawn as a cluster subgraph (nodes named `"<scan>.<body node>"`, recursively for nested bodies), with a dashed
/// edge from the body's output to the scan or ODE node.
pub fn to_dot(graph: &Graph, state: Option<&State>, metadata: &Metadata) -> String {
    let mut out = String::from("digraph BayesianNetwork {\n");
    if !metadata.is_empty() {
        out.push_str("// Model metadata:\n");
        for (key, value) in metadata {
            let _ = writeln!(out, "// {key}={value}");
        }
    }
    write_graph(&mut out, graph, state, "", None);
    out.push('}');
    out
}

/// A node's DOT id: its index at the top level (`<i>` in edges), and the quoted, dotted path of indices inside scan
/// bodies.
fn dot_id(prefix: &str, id: NodeId, edge: bool) -> String {
    match (prefix, edge) {
        ("", true) => format!("<{}>", id.index()),
        ("", false) => id.index().to_string(),
        _ => format!("\"{prefix}{}\"", id.index()),
    }
}

/// A nested body: a scan's or an ODE's.
#[derive(Clone, Copy)]
enum Body<'a> {
    Scan(&'a ScanBody),
    Ode(&'a OdeBody),
}

/// The role of a body's input node: `carry`, `xs[i]` or `constants[i]` in a scan, `t`, `y` or `parameters[i]` in
/// an ODE.
fn input_name(body: Body<'_>, id: NodeId) -> Option<String> {
    match body {
        Body::Scan(body) => {
            if id == body.carry() {
                return Some("carry".into());
            }
            if let Some(i) = body.xs().iter().position(|&x| x == id) {
                return Some(format!("xs[{i}]"));
            }
            let i = body.constants().iter().position(|&c| c == id)?;
            Some(format!("constants[{i}]"))
        }
        Body::Ode(body) => {
            if id == body.time() {
                return Some("t".into());
            }
            if id == body.state() {
                return Some("y".into());
            }
            let i = body.parameters().iter().position(|&p| p == id)?;
            Some(format!("parameters[{i}]"))
        }
    }
}

/// Writes a graph's edges, node labels and scan and ODE body clusters, naming its nodes with `prefix`. In a body
/// (`body`), the inputs are labelled by their role.
fn write_graph(
    out: &mut String,
    graph: &Graph,
    state: Option<&State>,
    prefix: &str,
    body: Option<Body<'_>>,
) {
    for child in graph.ids() {
        let node = &graph[child];
        let names = match node.op() {
            Op::Distribution(d) => d.kind.parameter_names().unwrap_or_default(),
            _ => Default::default(),
        };
        let child_id = dot_id(prefix, child, true);
        for (i, parent) in node.parents().iter().enumerate() {
            let parent_id = dot_id(prefix, *parent, true);
            match names.get(i) {
                Some(name) => {
                    let _ = writeln!(out, "{parent_id} -> {child_id} [label={name}]");
                }
                None => {
                    let _ = writeln!(out, "{parent_id} -> {child_id}");
                }
            }
        }
    }
    for id in graph.ids() {
        let _ = writeln!(
            out,
            "{}[label=\"{}\"]",
            dot_id(prefix, id, false),
            escape(
                &body
                    .and_then(|b| input_name(b, id))
                    .unwrap_or_else(|| node_label(graph, state, id))
            )
        );
    }
    for id in graph.ids() {
        let (body, title, output) = match graph[id].op() {
            Op::Scan(body) => (
                body.graph(),
                format!("scan body ({} steps)", body.length()),
                (Body::Scan(body), body.output()),
            ),
            Op::Ode(body) => (
                body.graph(),
                "ODE body (dy/dt)".to_owned(),
                (Body::Ode(body), body.output()),
            ),
            _ => continue,
        };
        let inner = format!("{prefix}{}.", id.index());
        let _ = writeln!(out, "subgraph \"cluster_{inner}\" {{\nlabel=\"{title}\"");
        write_graph(out, body, None, &inner, Some(output.0));
        out.push_str("}\n");
        let _ = writeln!(
            out,
            "{} -> {} [style=dashed]",
            dot_id(&inner, output.1, true),
            dot_id(prefix, id, true)
        );
    }
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
            (
                DistributionKind::HiddenMarkovModel(Arc::new(DistributionKind::Poisson)),
                "HiddenMarkovModel",
            ),
            (
                DistributionKind::LinearGaussianStateSpace,
                "LinearGaussianStateSpace",
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
