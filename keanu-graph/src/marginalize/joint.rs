//! Joint enumeration for scalar finite latents that share likelihood terms.

use super::{MAX_MARGINALIZED_VALUES, Namer, constant, support};
use crate::error::{GraphError, Result};
use crate::graph::{Graph, NodeId, Parents};
use crate::op::log_density::{has_elementwise_density, has_joint_density};
use crate::op::{BinaryOp, DistributionKind, Op, Predicate, ReduceOp, UnaryOp};
use crate::rewrite::{Writer, check_distinct, check_state, mapped};
use crate::state::State;
use crate::value::DType;
use keanu_tensor::{DoubleTensor, special::log_sum_exp};

/// Nodes and assignments of a joint marginalization.
#[derive(Debug, Clone)]
pub struct JointMarginalized {
    mapping: Vec<Option<NodeId>>,
    latents: Vec<NodeId>,
    assignments: Vec<Vec<i64>>,
    log_weights: NodeId,
    factor: NodeId,
}

impl JointMarginalized {
    /// The corresponding node in the rewritten graph, if kept.
    pub fn node(&self, original: NodeId) -> Option<NodeId> {
        self.mapping.get(original.index()).copied().flatten()
    }

    /// The original latent IDs, in assignment column order.
    pub fn latents(&self) -> &[NodeId] {
        &self.latents
    }

    /// The assignments in weight order. Boolean values are encoded as 0 and 1.
    pub fn assignments(&self) -> &[Vec<i64>] {
        &self.assignments
    }

    /// The node holding one unnormalized joint log weight per assignment.
    pub fn log_weights(&self) -> NodeId {
        self.log_weights
    }

    /// The factor holding the marginal likelihood of the group.
    pub fn factor(&self) -> NodeId {
        self.factor
    }

    /// Conditional probabilities of the joint assignments in an evaluated rewritten state.
    pub fn probabilities(&self, state: &State) -> Result<DoubleTensor> {
        let weights = state
            .value(self.log_weights)
            .ok_or(GraphError::MissingValue(self.log_weights))?
            .f64()?;
        let shifted = weights
            - weights
                .as_slice()
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
        let normalizer = log_sum_exp(shifted.as_slice().iter().copied());
        if !normalizer.is_finite() {
            return Err(GraphError::InvalidValue(
                "joint assignments have no finite normalizer".into(),
            ));
        }
        Ok((&shifted - normalizer).exp())
    }
}

/// Jointly marginalizes finitely supported scalar latents, including latents that share
/// an observation or whose prior probabilities depend on another selected latent.
///
/// The Cartesian product is limited to [`MAX_MARGINALIZED_VALUES`] assignments. Supports must be fixed.
/// Dependent probabilistic nodes outside the group must be observed and have a graph log density.
/// A factor or observed deterministic node that depends on the group is an error.
/// The resulting graph contains a single log-sum-exp factor and preserves the joint assignment
/// probabilities for posterior recovery. For independent aligned tensor latents use [`super::marginalize`].
pub fn marginalize_joint(
    graph: &Graph,
    state: &State,
    latents: &[NodeId],
) -> Result<(Graph, State, JointMarginalized)> {
    graph.check_ids(latents)?;
    check_state(graph, state)?;
    check_distinct(graph, latents)?;
    if latents.is_empty() {
        return Err(GraphError::InvalidValue(
            "there are no latents to marginalise".into(),
        ));
    }
    let inverse: Vec<_> = graph.ids().map(Some).collect();
    let namer = Namer {
        original: graph,
        current: graph,
        inverse: &inverse,
    };
    let group = latents
        .iter()
        .map(|&id| graph.describe(id))
        .collect::<Vec<_>>()
        .join(", ");
    let reject = |node, reason: String| GraphError::CannotMarginalize {
        latent: group.clone(),
        node: namer.name_with_op(node),
        reason,
    };

    let mut selected = vec![None; graph.len()];
    let mut domains = Vec::with_capacity(latents.len());
    let mut count = 1_usize;
    for (i, &id) in latents.iter().enumerate() {
        let node = &graph[id];
        let Op::Distribution(distribution) = &node.op else {
            return Err(reject(id, "is not a latent".into()));
        };
        let kind = &distribution.kind;
        if node.dtype == DType::F64 || !node.shape.is_empty() || state.is_observed(id) {
            return Err(reject(
                id,
                "must be an unobserved scalar discrete node".into(),
            ));
        }
        if !has_elementwise_density(kind)
            && !matches!(
                kind,
                DistributionKind::Categorical | DistributionKind::CategoricalLogit
            )
        {
            return Err(reject(
                id,
                "has no supported finite discrete density".into(),
            ));
        }
        let domain = support(graph, state, id, kind, &reject, &namer)?;
        let size = i128::from(*domain.end()) - i128::from(*domain.start()) + 1;
        count = usize::try_from(size)
            .ok()
            .and_then(|size| count.checked_mul(size))
            .filter(|&count| count <= MAX_MARGINALIZED_VALUES)
            .ok_or_else(|| {
                reject(
                    id,
                    format!("the group exceeds {MAX_MARGINALIZED_VALUES} joint assignments"),
                )
            })?;
        domains.push(domain);
        selected[id.index()] = Some(i);
    }

    let mut region = vec![false; graph.len()];
    let mut children = Vec::new();
    for id in graph.ids() {
        let node = &graph[id];
        if selected[id.index()].is_some() {
            region[id.index()] = true;
        } else if node.parents.iter().any(|p| region[p.index()]) {
            match &node.op {
                Op::Distribution(d)
                    if state.is_observed(id)
                        && (has_elementwise_density(&d.kind) || has_joint_density(&d.kind)) =>
                {
                    children.push(id);
                }
                Op::Distribution(_) => {
                    return Err(reject(id, "a dependent node outside the group needs an observed value and a graph log density".into()));
                }
                _ if state.is_observed(id) || node.is_factor() => {
                    return Err(reject(
                        id,
                        "a factor or deterministic observation depends on the group".into(),
                    ));
                }
                _ => region[id.index()] = true,
            }
        }
    }

    // Keep only region nodes needed by the selected priors or observed likelihoods.
    let mut live = vec![false; graph.len()];
    for &id in latents.iter().chain(&children) {
        live[id.index()] = true;
    }
    for id in graph.ids().rev() {
        if live[id.index()] {
            for p in &graph[id].parents {
                live[p.index()] |= region[p.index()];
            }
        }
    }

    let mut out = Writer::default();
    let mut mapping = vec![None; graph.len()];
    for id in graph.ids().filter(|id| !region[id.index()]) {
        let node = &graph[id];
        let value = state.value(id).cloned();
        mapping[id.index()] = Some(if children.binary_search(&id).is_ok() {
            out.constant(
                value.ok_or(GraphError::MissingValue(id))?,
                node.label.clone(),
            )
        } else {
            let parents = node
                .parents
                .iter()
                .map(|&p| mapped(&mapping, p))
                .collect::<Result<Parents>>()?;
            out.copy(node, parents, value, state.is_observed(id))
        });
    }

    let active: Vec<_> = graph.ids().filter(|id| live[id.index()]).collect();
    let mut copies = mapping.clone();
    let impossible = out.constant(DoubleTensor::scalar(f64::NEG_INFINITY).into(), None);
    let mut assignments = Vec::with_capacity(count);
    let mut weights = Vec::with_capacity(count);
    for mut index in 0..count {
        let mut assignment = vec![0; latents.len()];
        for (value, domain) in assignment.iter_mut().zip(&domains).rev() {
            let size = (*domain.end() - *domain.start() + 1) as usize;
            *value = *domain.start() + (index % size) as i64;
            index /= size;
        }
        let mut terms = Vec::with_capacity(latents.len() + children.len());
        for &id in &active {
            let node = &graph[id];
            let parents = node
                .parents
                .iter()
                .map(|&p| mapped(&copies, p))
                .collect::<Result<Parents>>()?;
            if let Some(i) = selected[id.index()] {
                let Op::Distribution(d) = &node.op else {
                    unreachable!("validated latent")
                };
                let value = out.constant(constant(node.dtype, &[], assignment[i])?, None);
                copies[id.index()] = Some(value);
                terms.push(log_prior(
                    &mut out,
                    &d.kind,
                    value,
                    &parents,
                    (assignment[i] - *domains[i].start()) as usize,
                )?);
            } else if let Op::Distribution(d) = &node.op {
                let value = mapped(&copies, id)?;
                terms.push(log_density(&mut out, &d.kind, value, &parents)?);
            } else {
                let mut copy = node.clone();
                copy.label = None;
                copy.parents = parents;
                copies[id.index()] = Some(out.push(copy, None, false));
            }
        }
        let mut terms = terms.into_iter();
        let mut weight = terms.next().expect("at least one selected prior");
        for term in terms {
            weight = out.add(Op::Binary(BinaryOp::Add), &[weight, term])?;
        }
        // Match Model::log_prob: a nonfinite term makes this assignment impossible.
        let finite = out.add(Op::Predicate(Predicate::IsFinite), &[weight])?;
        weights.push(out.add(Op::Where, &[finite, weight, impossible])?);
        assignments.push(assignment);
    }
    let log_weights = out.add(Op::Stack(-1), &weights)?;
    let marginal = out.add(
        Op::Reduce {
            kind: ReduceOp::LogSumExp,
            axes: None,
        },
        &[log_weights],
    )?;
    let factor = out.add(Op::Factor, &[marginal])?;
    let (graph, state) = out.finish()?;
    Ok((
        graph,
        state,
        JointMarginalized {
            mapping,
            latents: latents.to_vec(),
            assignments,
            log_weights,
            factor,
        },
    ))
}

fn log_density(
    out: &mut Writer,
    kind: &DistributionKind,
    value: NodeId,
    parameters: &[NodeId],
) -> Result<NodeId> {
    let mut parents = vec![value];
    parents.extend_from_slice(parameters);
    let density = out.add(Op::LogDensity(kind.clone()), &parents)?;
    if out.nodes[density.index()].shape.is_empty() {
        Ok(density)
    } else {
        out.add(
            Op::Reduce {
                kind: ReduceOp::Sum,
                axes: None,
            },
            &[density],
        )
    }
}

fn log_prior(
    out: &mut Writer,
    kind: &DistributionKind,
    value: NodeId,
    parameters: &[NodeId],
    index: usize,
) -> Result<NodeId> {
    let (numerator, normalizer) = match kind {
        DistributionKind::Categorical => {
            let mut total = parameters[0];
            for &p in &parameters[1..] {
                total = out.add(Op::Binary(BinaryOp::Add), &[total, p])?;
            }
            (
                out.add(Op::Unary(UnaryOp::Ln), &[parameters[index]])?,
                out.add(Op::Unary(UnaryOp::Ln), &[total])?,
            )
        }
        DistributionKind::CategoricalLogit => {
            let logits = out.add(Op::Stack(-1), parameters)?;
            (
                parameters[index],
                out.add(
                    Op::Reduce {
                        kind: ReduceOp::LogSumExp,
                        axes: None,
                    },
                    &[logits],
                )?,
            )
        }
        _ => return log_density(out, kind, value, parameters),
    };
    out.add(Op::Binary(BinaryOp::Sub), &[numerator, normalizer])
}
