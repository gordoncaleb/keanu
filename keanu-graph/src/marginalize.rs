//! Automatic marginalisation of discrete latents: a graph rewrite that sums a latent with finitely many values out
//! of the joint density, replacing it and the observations that depend on it by a factor.
//!
//! For a latent `z` of shape S taking the values k = lo..=hi, and its "children" (the observed probabilistic nodes
//! that depend on it through deterministic nodes), the rewrite builds, for each k, a copy of the deterministic
//! path from `z` to the children with `z` replaced by the constant k, evaluates each child's element-wise log
//! density there ([`Op::LogDensity`]), sums each child's density over its trailing axes to S and adds
//! log p(z = k). The K results are stacked along a new last axis (the latent's log weights, shape [S..., K]) and
//! the log-sum-exp over k becomes a [factor](Op::Factor):
//!
//! log p(y | θ) = Σₛ logsumexpₖ (log p(zₛ = k | θ) + Σ log p(yₛ | zₛ = k, θ)).
//!
//! The children become constants holding their observations, so their own log probabilities leave the joint log
//! probability (the factor holds them), and `z` and the path from it are removed. The softmax of the log weights
//! over k is the latent's conditional distribution given everything else, p(zₛ = k | θ, y), from which the latent
//! is recovered after sampling (see [`MarginalizedLatent::probabilities`] and [`MarginalizedLatent::draw`]).

use crate::error::{GraphError, Result};
use crate::graph::{Graph, Node, NodeId, Parents};
use crate::op::log_density::has_elementwise_density;
use crate::op::{BinaryOp, DistributionKind, Op, ReduceOp, UnaryOp};
use crate::rewrite::{Writer, check_distinct, check_state, mapped};
use crate::state::State;
use crate::value::{DType, Value};
use keanu_distributions::KeanuRng;
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor, Shape};
use std::collections::HashMap;
use std::ops::RangeInclusive;

/// The most values a marginalised latent may take: the rewrite copies the path from the latent to its children
/// once per value.
pub const MAX_MARGINALIZED_VALUES: usize = 1000;

/// The record of a [`marginalize`] rewrite: which node of the new graph each node of the original graph became, and
/// how to recover each marginalised latent.
#[derive(Debug, Clone)]
pub struct Marginalized {
    /// For each node of the original graph, its node in the new graph.
    mapping: Vec<Option<NodeId>>,
    latents: Vec<MarginalizedLatent>,
}

impl Marginalized {
    /// The node of the new graph that a node of the original graph became: the same operation on the corresponding
    /// parents, with the same label, or a constant holding the observation for an observed child of a marginalised
    /// latent. `None` for the marginalised latents, the deterministic nodes that depend on them (which have no
    /// single value any more) and ids that aren't nodes of the original graph.
    pub fn node(&self, original: NodeId) -> Option<NodeId> {
        self.mapping.get(original.index()).copied().flatten()
    }

    /// The marginalised latents, in the order they were marginalised.
    pub fn latents(&self) -> &[MarginalizedLatent] {
        &self.latents
    }

    /// The record of a marginalised latent, by its id in the original graph.
    pub fn latent(&self, original: NodeId) -> Option<&MarginalizedLatent> {
        self.latents.iter().find(|l| l.original == original)
    }
}

/// One marginalised latent: what it was and the nodes of the new graph that hold its conditional distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct MarginalizedLatent {
    original: NodeId,
    name: String,
    label: Option<String>,
    dtype: DType,
    shape: Shape,
    values: RangeInclusive<i64>,
    log_weights: NodeId,
    factor: NodeId,
}

impl MarginalizedLatent {
    /// The latent's id in the original graph.
    pub fn original(&self) -> NodeId {
        self.original
    }

    /// The latent as the original graph's [`Graph::describe`] names it: its label, quoted, or `#` and its index.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The latent's label in the original graph.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// The latent's value type: i64, or bool for a Bernoulli latent (whose values 0 and 1 are false and true).
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The latent's shape, S.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The values each element can take, lo..=hi (0..=1 for a Bernoulli latent). The k-th entry of the last axis of
    /// the [log weights](Self::log_weights) and [probabilities](Self::probabilities) is the value lo + k.
    pub fn values(&self) -> RangeInclusive<i64> {
        self.values.clone()
    }

    /// The number of values each element can take, K.
    pub fn value_count(&self) -> usize {
        (self.values.end() - self.values.start() + 1) as usize
    }

    /// The node of the new graph holding the unnormalised log conditional probabilities, shape [S..., K]: at
    /// [s, k], log p(zₛ = lo + k | θ) plus the log densities of the observations that depend on zₛ.
    pub fn log_weights(&self) -> NodeId {
        self.log_weights
    }

    /// The factor node of the new graph that holds the latent's marginal likelihood.
    pub fn factor(&self) -> NodeId {
        self.factor
    }

    /// The latent's conditional probabilities given the rest of the model at `state` (a state of the new graph whose
    /// [log weights](Self::log_weights) have been evaluated), shape [S..., K]: at [s, k], p(zₛ = lo + k | θ, y),
    /// which sums to 1 over k.
    ///
    /// # Errors
    ///
    /// [`GraphError::MissingValue`] if the log weights have no value in `state`.
    pub fn probabilities(&self, state: &State) -> Result<DoubleTensor> {
        let weights = state
            .value(self.log_weights)
            .ok_or(GraphError::MissingValue(self.log_weights))?
            .f64()?;
        let k = self.value_count();
        let mut out = weights.as_slice().to_vec();
        for row in out.chunks_mut(k) {
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = row.iter().map(|w| (w - max).exp()).sum();
            for w in row.iter_mut() {
                *w = (*w - max).exp() / total;
            }
        }
        Ok(DoubleTensor::new(weights.shape(), out)?)
    }

    /// Draws a value of the latent, element by element, from conditional `probabilities` (as
    /// [`probabilities`](Self::probabilities) returns them, shape [S..., K]): an i64 value of shape S, or a bool
    /// one for a Bernoulli latent.
    ///
    /// # Errors
    ///
    /// [`GraphError::InvalidValue`] if the probabilities have the wrong shape.
    pub fn draw(&self, probabilities: &DoubleTensor, rng: &mut KeanuRng) -> Result<Value> {
        let k = self.value_count();
        let mut expected = self.shape.to_vec();
        expected.push(k);
        if probabilities.shape() != expected.as_slice() {
            return Err(GraphError::InvalidValue(format!(
                "the probabilities of {} have shape {:?}, expected {expected:?}",
                self.name,
                probabilities.shape()
            )));
        }
        let lo = *self.values.start();
        let drawn: Vec<i64> = probabilities
            .as_slice()
            .chunks(k)
            .map(|row| {
                let u = rng.next_f64();
                let mut cumulative = 0.0;
                let mut chosen = None;
                for (i, &p) in row.iter().enumerate() {
                    if p > 0.0 {
                        chosen = Some(i);
                    }
                    cumulative += p;
                    if u < cumulative {
                        break;
                    }
                }
                lo + chosen.unwrap_or(0) as i64
            })
            .collect();
        Ok(match self.dtype {
            DType::Bool => Value::Bool(BoolTensor::new(
                &self.shape,
                drawn.into_iter().map(|v| v != 0).collect(),
            )?),
            _ => Value::I64(IntTensor::new(&self.shape, drawn)?),
        })
    }
}

/// Marginalises discrete latents out of a model: returns a new graph and state in which each of `latents` is summed
/// out of the joint density, and the [`Marginalized`] record of how the nodes correspond and how to recover the
/// latents. Several latents are marginalised one after another, in the order given.
///
/// A latent `z` of shape S can be marginalised when:
///
/// - it is unobserved, with a Bernoulli, categorical (or one of their logit forms) or element-wise discrete
///   distribution (a discrete uniform, binomial, beta-binomial, truncated count or zero-inflated binomial) whose
///   elements all take the same finitely many values (at most [`MAX_MARGINALIZED_VALUES`]), fixed rather than
///   depending on a latent: no parameter that sets the values (a count, a discrete uniform's or truncation's
///   bounds, also inside a zero-inflated or hurdle kind) may depend on a latent, except through an observed node;
/// - every probabilistic node that depends on it through deterministic nodes (a "child") is observed and has an
///   element-wise log density (see [`GraphBuilder::log_density`](crate::GraphBuilder::log_density)), and no factor
///   or observed deterministic node depends on it;
/// - for a vector (or higher rank) latent, every element of a child depends on at most one element of `z`, the one
///   at the same leading index: the child's shape starts with S (its trailing axes are summed), and the path from
///   `z` to it uses only element-wise ops (arithmetic, comparisons, `where_` and the other selections, casts and
///   element-wise functions) with broadcasting that keeps the leading axes aligned, reshapes and broadcasts that
///   keep the leading axes, reductions, softmaxes and cumulative sums over trailing axes, and `gather` with `z` (or
///   a node computed from it) as the indices along the first axis. A scalar latent may reach its children through
///   any deterministic ops, and its children's densities are summed completely.
///
/// Deterministic nodes that depend on a latent but lead to no child are dropped. Anything else is rejected with
/// [`GraphError::CannotMarginalize`], naming the node that blocks the rewrite.
///
/// In the new graph the children are constants holding their observations, with their labels; every other node
/// that doesn't depend on a marginalised latent is copied with its label, value and observation. The new graph's
/// joint log probability, gradients and Hessian-vector products are the original model's with the latents summed
/// out, so NUTS and the other gradient based algorithms sample it.
///
/// ```
/// use keanu_graph::{GraphBuilder, Model, marginalize};
/// use keanu_tensor::DoubleTensor;
///
/// // which of two coins, one fair and one biased, produced the tosses?
/// let b = GraphBuilder::new();
/// let biased = b.bernoulli(0.5).label("biased");
/// let p = b.where_(biased, b.scalar(0.9), b.scalar(0.5));
/// let tosses = b.bernoulli_with_shape(&[4], p);
/// b.observe(tosses, keanu_tensor::BoolTensor::vector(vec![true, true, true, false]));
/// let biased = biased.id();
/// let (graph, state) = b.build()?;
///
/// let (graph, state, marginalized) = marginalize(&graph, &state, &[biased])?;
/// let model = Model::new(graph, &state)?;
/// // p(tosses) = ½ · 0.9³ · 0.1 + ½ · 0.5⁴
/// let expected = (0.5 * 0.9f64.powi(3) * 0.1 + 0.5 * 0.5f64.powi(4)).ln();
/// assert!((model.log_prob(&state)? - expected).abs() < 1e-12);
/// let p_biased = marginalized.latents()[0].probabilities(&state)?;
/// assert!((p_biased.as_slice()[1] - 0.5 * 0.9f64.powi(3) * 0.1 / expected.exp()).abs() < 1e-12);
/// # Ok::<(), keanu_graph::GraphError>(())
/// ```
///
/// # Errors
///
/// [`GraphError::CannotMarginalize`] as above, [`GraphError::UnknownNode`] for an id that isn't a node of the
/// graph, [`GraphError::InvalidValue`] if `latents` is empty or repeats a latent or the state isn't the graph's,
/// and any error evaluating the latents' parameters or the new nodes.
pub fn marginalize(
    graph: &Graph,
    state: &State,
    latents: &[NodeId],
) -> Result<(Graph, State, Marginalized)> {
    graph.check_ids(latents)?;
    check_state(graph, state)?;
    if latents.is_empty() {
        return Err(GraphError::InvalidValue(
            "there are no latents to marginalise".into(),
        ));
    }
    check_distinct(graph, latents)?;
    let mut mapping: Vec<Option<NodeId>> = graph.ids().map(Some).collect();
    // for each node of the current graph, its node in the original graph
    let mut inverse: Vec<Option<NodeId>> = graph.ids().map(Some).collect();
    let mut records: Vec<MarginalizedLatent> = Vec::with_capacity(latents.len());
    let mut current: Option<(Graph, State)> = None;
    for &z in latents {
        let (g, s) = match &current {
            Some((g, s)) => (g, s),
            None => (graph, state),
        };
        let namer = Namer {
            original: graph,
            current: g,
            inverse: &inverse,
        };
        let Some(cz) = mapping[z.index()] else {
            let name = graph.describe(z);
            return Err(GraphError::CannotMarginalize {
                node: name.clone(),
                latent: name,
                reason: "depends on a latent marginalised before it".into(),
            });
        };
        let step = Step::run(g, s, cz, &namer)?;
        for m in mapping.iter_mut() {
            *m = m.and_then(|id| step.mapping[id.index()]);
        }
        let mut new_inverse = vec![None; step.graph.len()];
        for (old, new) in step.mapping.iter().enumerate() {
            if let Some(new) = new {
                new_inverse[new.index()] = inverse[old];
            }
        }
        inverse = new_inverse;
        for record in &mut records {
            let moved = |id: NodeId| {
                step.mapping[id.index()].ok_or_else(|| GraphError::CannotMarginalize {
                    latent: graph.describe(z),
                    node: record.name.clone(),
                    reason: "was marginalised before, and its factor depends on this latent".into(),
                })
            };
            let (log_weights, factor) = (moved(record.log_weights)?, moved(record.factor)?);
            record.log_weights = log_weights;
            record.factor = factor;
        }
        records.push(MarginalizedLatent {
            original: z,
            name: graph.describe(z),
            label: graph[z].label.clone(),
            dtype: graph[z].dtype,
            shape: graph[z].shape.clone(),
            values: step.values,
            log_weights: step.log_weights,
            factor: step.factor,
        });
        current = Some((step.graph, step.state));
    }
    let (graph, state) = current.expect("at least one latent was marginalised");
    Ok((
        graph,
        state,
        Marginalized {
            mapping,
            latents: records,
        },
    ))
}

/// Names nodes of the current (possibly already rewritten) graph in error messages, by their node in the original
/// graph where there is one.
struct Namer<'a> {
    original: &'a Graph,
    current: &'a Graph,
    inverse: &'a [Option<NodeId>],
}

impl Namer<'_> {
    fn name(&self, id: NodeId) -> String {
        match self.inverse.get(id.index()).copied().flatten() {
            Some(original) => self.original.describe(original),
            None => format!(
                "{} (made by marginalising an earlier latent)",
                self.current.describe(id)
            ),
        }
    }

    /// The node's name with its operation, or its distribution for a probabilistic node: `#12 (matmul)`.
    fn name_with_op(&self, id: NodeId) -> String {
        let node = &self.current[id];
        let op = match &node.op {
            Op::Distribution(d) => d.kind.name(),
            op => op.name(),
        };
        format!("{} ({op})", self.name(id))
    }
}

/// How log p(z = k) is computed on the graph.
enum Prior {
    /// With a log density node of the latent's kind at the constant k.
    Density,
    /// As ln pₖ − ln Σⱼ pⱼ from the categorical's probability parameters.
    Categorical,
    /// As ηₖ − logsumexpⱼ ηⱼ from the categorical's logit parameters.
    CategoricalLogit,
}

/// One latent marginalised out of one graph.
struct Step {
    graph: Graph,
    state: State,
    mapping: Vec<Option<NodeId>>,
    values: RangeInclusive<i64>,
    log_weights: NodeId,
    factor: NodeId,
}

/// `axis` (negative counting from the end) as an index into `rank` axes.
fn axis_index(axis: isize, rank: usize) -> Option<usize> {
    let a = if axis < 0 { axis + rank as isize } else { axis };
    usize::try_from(a).ok().filter(|&a| a < rank)
}

impl Step {
    fn run(graph: &Graph, state: &State, z: NodeId, namer: &Namer<'_>) -> Result<Self> {
        let latent = namer.name(z);
        let reject = |node: NodeId, reason: String| GraphError::CannotMarginalize {
            latent: latent.clone(),
            node: namer.name_with_op(node),
            reason,
        };
        let z_node = &graph[z];
        let Op::Distribution(distribution) = &z_node.op else {
            return Err(reject(z, "is not probabilistic".into()));
        };
        if state.is_observed(z) {
            return Err(reject(
                z,
                "is observed; only latents can be marginalised".into(),
            ));
        }
        let kind = &distribution.kind;
        let prior = match kind {
            DistributionKind::Categorical => Prior::Categorical,
            DistributionKind::CategoricalLogit => Prior::CategoricalLogit,
            k if k.value_type() != DType::F64 && has_elementwise_density(k) => Prior::Density,
            k => {
                return Err(reject(
                    z,
                    format!(
                        "has a {} distribution; only Bernoulli, categorical and element-wise discrete \
                         distributions with finitely many values (discrete uniform, binomial, beta-binomial, \
                         truncated counts) can be marginalised",
                        k.name()
                    ),
                ));
            }
        };
        let s: &[usize] = &z_node.shape;
        let r = s.len();
        let values = support(graph, state, z, kind, &reject, namer)?;
        let count = values.end() - values.start() + 1;
        if count as u64 > MAX_MARGINALIZED_VALUES as u64 {
            return Err(reject(
                z,
                format!(
                    "takes {count} values, more than the {MAX_MARGINALIZED_VALUES} that can be enumerated"
                ),
            ));
        }

        // the region: z and the deterministic nodes that depend on it; the children: the probabilistic nodes that do
        let n = graph.len();
        let mut region = vec![false; n];
        let mut child = vec![false; n];
        region[z.0] = true;
        for id in z.0 + 1..n {
            let node = &graph.nodes[id];
            if node.parents.iter().any(|p| region[p.0]) {
                if node.is_probabilistic() {
                    child[id] = true;
                } else {
                    region[id] = true;
                }
            }
        }
        // the deterministic nodes of the region that lead to a child (or are factors or observed, and rejected)
        let mut live = vec![false; n];
        for id in (z.0 + 1..n).rev() {
            if !region[id] {
                continue;
            }
            let node = &graph.nodes[id];
            if node.is_factor() {
                return Err(reject(
                    NodeId(id),
                    "depends on the latent; factors can't, since they have no density to sum over its values"
                        .into(),
                ));
            }
            if state.is_observed(NodeId(id)) {
                return Err(reject(
                    NodeId(id),
                    "is an observed deterministic node that depends on the latent".into(),
                ));
            }
            live[id] = graph.children[id].iter().any(|c| child[c.0] || live[c.0]);
        }
        let dependent = |p: &NodeId| region[p.0];
        let aligned = |shape: &[usize]| shape.len() >= r && &shape[..r] == s;

        // check the path and the children
        for id in z.0 + 1..n {
            let node = &graph.nodes[id];
            let me = NodeId(id);
            if child[id] {
                let Op::Distribution(d) = &node.op else {
                    unreachable!("children are probabilistic")
                };
                if !state.is_observed(me) {
                    return Err(reject(
                        me,
                        "is a latent that depends on the latent; only observed nodes can".into(),
                    ));
                }
                if !has_elementwise_density(&d.kind) {
                    return Err(reject(
                        me,
                        format!(
                            "depends on the latent, but the {} distribution has no element-wise log density",
                            d.kind.name()
                        ),
                    ));
                }
                if r > 0 {
                    if !aligned(&node.shape) {
                        return Err(reject(
                            me,
                            format!(
                                "has shape {:?}, which doesn't start with the latent's shape {s:?}",
                                node.shape.as_slice()
                            ),
                        ));
                    }
                    if node
                        .parents
                        .iter()
                        .filter(|p| dependent(p))
                        .any(|p| graph[*p].shape.len() != node.shape.len())
                    {
                        return Err(reject(
                            me,
                            format!(
                                "broadcasts a parameter that depends on the latent to shape {:?}, which moves \
                                 the latent's axes; reshape the parameter so that its leading axes are the \
                                 latent's",
                                node.shape.as_slice()
                            ),
                        ));
                    }
                }
                continue;
            }
            if !live[id] || r == 0 {
                continue;
            }
            check_aligned(graph, node, &dependent, r).map_err(|reason| reject(me, reason))?;
            if !aligned(&node.shape) {
                return Err(reject(
                    me,
                    format!(
                        "has shape {:?}, which doesn't start with the latent's shape {s:?}",
                        node.shape.as_slice()
                    ),
                ));
            }
        }

        // copy everything outside the region, with the children as constants
        let mut out = Writer::default();
        let mut mapping: Vec<Option<NodeId>> = vec![None; n];
        for id in 0..n {
            if region[id] {
                continue;
            }
            let node = &graph.nodes[id];
            let me = NodeId(id);
            mapping[id] = Some(if child[id] {
                let value = state.value(me).ok_or(GraphError::MissingValue(me))?.clone();
                out.constant(value, node.label.clone())
            } else {
                let parents = node
                    .parents
                    .iter()
                    .map(|&p| mapped(&mapping, p))
                    .collect::<Result<Parents>>()?;
                out.copy(
                    node,
                    parents,
                    state.value(me).cloned(),
                    state.is_observed(me),
                )
            });
        }

        // the prior's parameters, and for a categorical the log of their total
        let z_parameters = z_node
            .parents
            .iter()
            .map(|&p| mapped(&mapping, p))
            .collect::<Result<Vec<_>>>()?;
        let log_total = match prior {
            Prior::Categorical => {
                let mut total = z_parameters[0];
                for &p in &z_parameters[1..] {
                    total = out.add(Op::Binary(BinaryOp::Add), &[total, p])?;
                }
                Some(out.add(Op::Unary(UnaryOp::Ln), &[total])?)
            }
            Prior::CategoricalLogit => {
                let stacked = out.add(Op::Stack(-1), &z_parameters)?;
                Some(out.add(
                    Op::Reduce {
                        kind: ReduceOp::LogSumExp,
                        axes: Some(vec![-1]),
                    },
                    &[stacked],
                )?)
            }
            Prior::Density => None,
        };
        let children: Vec<NodeId> = (0..n).filter(|&id| child[id]).map(NodeId).collect();
        let path: Vec<NodeId> = (z.0 + 1..n).filter(|&id| live[id]).map(NodeId).collect();

        let mut terms = Vec::with_capacity(count as usize);
        for value in values.clone() {
            let k = out.constant(constant(z_node.dtype, s, value)?, None);
            let mut copies: HashMap<NodeId, NodeId> = HashMap::from([(z, k)]);
            let lookup = |copies: &HashMap<NodeId, NodeId>,
                          mapping: &[Option<NodeId>],
                          p: NodeId| match copies.get(&p) {
                Some(&c) => Ok(c),
                None => mapped(mapping, p),
            };
            for &id in &path {
                let node = &graph[id];
                let parents = node
                    .parents
                    .iter()
                    .map(|&p| lookup(&copies, &mapping, p))
                    .collect::<Result<Parents>>()?;
                let copy = out.push(
                    Node {
                        parents,
                        label: None,
                        ..node.clone()
                    },
                    None,
                    false,
                );
                copies.insert(id, copy);
            }
            let mut parts = Vec::with_capacity(children.len() + 1);
            for &c in &children {
                let node = &graph[c];
                let Op::Distribution(d) = &node.op else {
                    unreachable!("children are probabilistic")
                };
                let mut parents = vec![mapped(&mapping, c)?];
                for &p in &node.parents {
                    parents.push(lookup(&copies, &mapping, p)?);
                }
                let mut density = out.add(Op::LogDensity(d.kind.clone()), &parents)?;
                let rank = out.nodes[density.0].shape.len();
                if rank > r {
                    let axes = (r..rank).map(|a| a as isize).collect();
                    density = out.add(
                        Op::Reduce {
                            kind: ReduceOp::Sum,
                            axes: Some(axes),
                        },
                        &[density],
                    )?;
                }
                parts.push(density);
            }
            let mut log_prior = match (&prior, log_total) {
                (Prior::Categorical, Some(log_total)) => {
                    let p = z_parameters[(value - values.start()) as usize];
                    let ln = out.add(Op::Unary(UnaryOp::Ln), &[p])?;
                    out.add(Op::Binary(BinaryOp::Sub), &[ln, log_total])?
                }
                (Prior::CategoricalLogit, Some(log_total)) => {
                    let eta = z_parameters[(value - values.start()) as usize];
                    out.add(Op::Binary(BinaryOp::Sub), &[eta, log_total])?
                }
                _ => {
                    let mut parents = vec![k];
                    parents.extend_from_slice(&z_parameters);
                    out.add(Op::LogDensity(kind.clone()), &parents)?
                }
            };
            if out.nodes[log_prior.0].shape.as_slice() != s {
                log_prior = out.add(Op::Broadcast(s.to_vec()), &[log_prior])?;
            }
            let mut term = log_prior;
            for part in parts {
                term = out.add(Op::Binary(BinaryOp::Add), &[term, part])?;
            }
            terms.push(term);
        }
        let log_weights = out.add(Op::Stack(-1), &terms)?;
        let marginal = out.add(
            Op::Reduce {
                kind: ReduceOp::LogSumExp,
                axes: Some(vec![-1]),
            },
            &[log_weights],
        )?;
        let factor = out.add(Op::Factor, &[marginal])?;

        let (graph, state) = out.finish()?;
        Ok(Self {
            graph,
            state,
            mapping,
            values,
            log_weights,
            factor,
        })
    }
}

/// The values every element of the latent takes, the same for each element and fixed: not depending on any latent.
fn support(
    graph: &Graph,
    state: &State,
    z: NodeId,
    kind: &DistributionKind,
    reject: &dyn Fn(NodeId, String) -> GraphError,
    namer: &Namer<'_>,
) -> Result<RangeInclusive<i64>> {
    let node = &graph[z];
    match kind {
        DistributionKind::Bernoulli | DistributionKind::BernoulliLogit => return Ok(0..=1),
        DistributionKind::Categorical | DistributionKind::CategoricalLogit => {
            return Ok(0..=node.parents.len() as i64 - 1);
        }
        _ => {}
    }
    // the parameters whose values set the support, through nested kinds (a zero-inflated truncated count's bounds)
    let reads = kind.support_parameters(node.parents.len());
    for (&p, _) in node.parents.iter().zip(&reads).filter(|(_, read)| **read) {
        if let Some(l) = latent_ancestor(graph, state, p) {
            return Err(reject(
                l,
                format!(
                    "is a latent that the values of {} depend on; they must be fixed",
                    namer.name(z)
                ),
            ));
        }
    }
    let mut work = state.clone();
    work.lazy_eval(graph, &node.parents, &mut KeanuRng::seed_from_u64(0))?;
    let parameters = node
        .parents
        .iter()
        .map(|&p| work.value(p).ok_or(GraphError::MissingValue(p)))
        .collect::<Result<Vec<_>>>()?;
    let Some(ranges) = kind.finite_support(&parameters, &node.shape)? else {
        return Err(reject(z, "has infinitely many values".into()));
    };
    let Some(first) = ranges.first().cloned() else {
        return Err(reject(z, "has no elements".into()));
    };
    if ranges.iter().any(|r| *r != first) {
        return Err(reject(
            z,
            "has elements that take different values; all must take the same ones".into(),
        ));
    }
    if first.is_empty() {
        return Err(reject(z, "has no possible values".into()));
    }
    Ok(first)
}

/// The first latent (unobserved probabilistic node) among `id` and its ancestors, if any. The walk stops at
/// observed nodes, whose values are fixed whatever their parents' are.
fn latent_ancestor(graph: &Graph, state: &State, id: NodeId) -> Option<NodeId> {
    let mut seen = vec![false; id.0 + 1];
    let mut stack = vec![id];
    while let Some(n) = stack.pop() {
        if std::mem::replace(&mut seen[n.0], true) {
            continue;
        }
        if state.is_observed(n) {
            continue;
        }
        if graph[n].is_probabilistic() {
            return Some(n);
        }
        stack.extend(graph[n].parents.iter().copied());
    }
    None
}

/// The constant value k of the latent's type and shape.
fn constant(dtype: DType, shape: &[usize], k: i64) -> Result<Value> {
    Ok(match dtype {
        DType::Bool => Value::Bool(BoolTensor::full(shape, k != 0)),
        DType::I64 => Value::I64(IntTensor::full(shape, k)),
        DType::F64 => {
            return Err(GraphError::InvalidValue(
                "a continuous latent can't be marginalised".into(),
            ));
        }
    })
}

/// Why a deterministic node on the path from a vector latent of rank `r` to its children might combine different
/// elements of the latent, if it might: the ops that keep each element's leading index, as [`marginalize`]
/// documents, pass.
fn check_aligned(
    graph: &Graph,
    node: &Node,
    dependent: &dyn Fn(&NodeId) -> bool,
    r: usize,
) -> std::result::Result<(), String> {
    let rank = node.shape.len();
    let same_rank = |what: &str| -> std::result::Result<(), String> {
        match node
            .parents
            .iter()
            .filter(|p| dependent(p))
            .find(|p| graph[**p].shape.len() != rank)
        {
            Some(p) => Err(format!(
                "broadcasts {what} of shape {:?} that depends on the latent to shape {:?}, which moves the \
                 latent's axes; reshape it so that its leading axes are the latent's",
                graph[*p].shape.as_slice(),
                node.shape.as_slice()
            )),
            None => Ok(()),
        }
    };
    let trailing = |axis: isize, rank: usize| -> std::result::Result<(), String> {
        match axis_index(axis, rank) {
            Some(a) if a >= r => Ok(()),
            _ => Err(format!(
                "works along axis {axis}, one of the latent's axes, so it combines the latent's elements"
            )),
        }
    };
    let parents = &node.parents;
    match &node.op {
        Op::Unary(_)
        | Op::Binary(_)
        | Op::Compare(_)
        | Op::Mask(_)
        | Op::ApproxEq(_)
        | Op::Logic(_)
        | Op::Not
        | Op::Predicate(_)
        | Op::Cast(_)
        | Op::Where
        | Op::ReplaceNan(_)
        | Op::Bessel { .. }
        | Op::SetWithMask
        | Op::Assert(_)
        | Op::LogDensity(_)
        | Op::Broadcast(_)
        | Op::Proxy => same_rank("a parent"),
        Op::Multiplexer if !dependent(&parents[0]) => same_rank("an option"),
        Op::Reshape(_) => Ok(()),
        Op::Gather { axis } => {
            let input_rank = graph[parents[0]].shape.len();
            match (dependent(&parents[0]), dependent(&parents[1])) {
                (false, true) if axis_index(*axis, input_rank) == Some(0) => Ok(()),
                (false, true) => Err(format!(
                    "gathers by the latent along axis {axis}; only gathers along the first axis keep the latent's \
                     axes first"
                )),
                (true, false) => trailing(*axis, input_rank),
                _ => Err(
                    "gathers from a node that depends on the latent by indices that do too".into(),
                ),
            }
        }
        Op::Reduce {
            axes: Some(axes), ..
        } => {
            let input_rank = graph[parents[0]].shape.len();
            axes.iter().try_for_each(|&a| trailing(a, input_rank))
        }
        Op::Softmax { axis } | Op::LogSoftmax { axis } | Op::Cumulative { axis, .. } => {
            trailing(*axis, rank)
        }
        op => Err(format!(
            "depends on the latent through a {} op, which can combine the latent's elements; only element-wise \
             ops, gathers by the latent, reshapes and broadcasts, and reductions over trailing axes can be on the \
             path from a vector latent to its children",
            op.name()
        )),
    }
}
