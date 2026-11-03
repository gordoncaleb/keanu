//! Log probabilities of probabilistic nodes and their gradients with respect to latent nodes.

use crate::autodiff::reverse_partials_from;
use crate::error::{GraphError, Result};
use crate::graph::{Graph, NodeId};
use crate::op::{DistributionKind, Op};
use crate::state::State;
use crate::value::{DType, Value};
use keanu_distributions as d;
use keanu_tensor::DoubleTensor;
use std::collections::HashMap;

fn value(state: &State, id: NodeId) -> Result<&Value> {
    state.value(id).ok_or(GraphError::MissingValue(id.0))
}

fn kind(graph: &Graph, id: NodeId) -> Result<&DistributionKind> {
    match graph.node(id).op() {
        Op::Distribution(op) => Ok(&op.kind),
        _ => Err(GraphError::InvalidValue(format!(
            "node #{} is not probabilistic, so it has no log probability",
            id.0
        ))),
    }
}

/// The log probability of a probabilistic node at its current value, given its parameters' current values
/// (summed over the elements of the node).
pub fn node_log_prob(graph: &Graph, state: &State, id: NodeId) -> Result<f64> {
    node_log_prob_at(graph, state, id, value(state, id)?)
}

/// The log probability that a probabilistic node takes the value `v`, given its parameters' current values
/// (summed over the elements of the node). The node's own current value is ignored.
pub fn node_log_prob_at(graph: &Graph, state: &State, id: NodeId, v: &Value) -> Result<f64> {
    use DistributionKind::*;
    let kind = kind(graph, id)?;
    let parents = graph.parents(id);
    let p = |i: usize| -> Result<DoubleTensor> { Ok(value(state, parents[i])?.f64()?.clone()) };
    let q = |i: usize| -> Result<keanu_tensor::IntTensor> {
        Ok(value(state, parents[i])?.i64()?.clone())
    };
    let total = |t: DoubleTensor| t.sum();
    Ok(match kind {
        Gaussian => total(d::Gaussian::new(p(0)?, p(1)?).log_prob(v.f64()?)?),
        HalfGaussian => total(d::HalfGaussian::new(p(0)?).log_prob(v.f64()?)?),
        Cauchy => total(d::Cauchy::new(p(0)?, p(1)?).log_prob(v.f64()?)?),
        HalfCauchy => total(d::HalfCauchy::new(p(0)?).log_prob(v.f64()?)?),
        Beta => total(d::Beta::standard(p(0)?, p(1)?).log_prob(v.f64()?)?),
        ChiSquared => total(d::ChiSquared::new(q(0)?).log_prob(v.f64()?)?),
        Dirichlet => d::Dirichlet::new(p(0)?).log_prob(v.f64()?)?,
        Exponential => total(d::Exponential::new(p(0)?).log_prob(v.f64()?)?),
        Gamma => total(d::Gamma::new(p(0)?, p(1)?).log_prob(v.f64()?)?),
        InverseGamma => total(d::InverseGamma::new(p(0)?, p(1)?).log_prob(v.f64()?)?),
        Laplace => total(d::Laplace::new(p(0)?, p(1)?).log_prob(v.f64()?)?),
        Logistic => total(d::Logistic::new(p(0)?, p(1)?).log_prob(v.f64()?)?),
        LogNormal => total(d::LogNormal::new(p(0)?, p(1)?).log_prob(v.f64()?)?),
        MultivariateGaussian => d::MultivariateGaussian::new(p(0)?, p(1)?).log_prob(v.f64()?)?,
        Pareto => total(d::Pareto::new(p(0)?, p(1)?).log_prob(v.f64()?)?),
        StudentT => total(d::StudentT::new(q(0)?).log_prob(v.f64()?)?),
        Triangular => total(d::Triangular::new(p(0)?, p(1)?, p(2)?).log_prob(v.f64()?)?),
        Uniform => total(d::Uniform::new(p(0)?, p(1)?).log_prob(v.f64()?)?),
        Kde(kde) => total(kde.log_prob(v.f64()?)),
        Bernoulli => total(d::Bernoulli::new(p(0)?).log_prob(v.bool()?)?),
        Binomial => total(d::Binomial::new(p(0)?, q(1)?).log_prob(v.i64()?)?),
        Categorical => {
            let probabilities = (0..parents.len()).map(p).collect::<Result<Vec<_>>>()?;
            total(d::Categorical::new(probabilities)?.log_prob(v.i64()?)?)
        }
        Geometric => total(d::Geometric::new(p(0)?).log_prob(v.i64()?)?),
        Multinomial => total(d::Multinomial::new(q(0)?, p(1)?)?.log_prob(v.i64()?)?),
        Poisson => total(d::Poisson::new(p(0)?).log_prob(v.i64()?)?),
        UniformInt => total(d::UniformInt::new(q(0)?, q(1)?).log_prob(v.i64()?)?),
    })
}

/// The total log probability of the given probabilistic nodes.
pub fn log_prob(graph: &Graph, state: &State, nodes: &[NodeId]) -> Result<f64> {
    nodes
        .iter()
        .map(|&id| node_log_prob(graph, state, id))
        .sum()
}

/// The joint log probability of every probabilistic node.
pub fn joint_log_prob(graph: &Graph, state: &State) -> Result<f64> {
    graph
        .probabilistic()
        .map(|id| node_log_prob(graph, state, id))
        .sum()
}

/// The log likelihood: the log probability of the observed probabilistic nodes.
pub fn log_likelihood(graph: &Graph, state: &State) -> Result<f64> {
    graph
        .probabilistic()
        .filter(|&id| state.is_observed(id))
        .map(|id| node_log_prob(graph, state, id))
        .sum()
}

/// The log prior: the log probability of the unobserved (latent) probabilistic nodes.
pub fn log_prior(graph: &Graph, state: &State) -> Result<f64> {
    graph
        .probabilistic()
        .filter(|&id| !state.is_observed(id))
        .map(|id| node_log_prob(graph, state, id))
        .sum()
}

/// The derivatives of one node's log probability with respect to its own value (`value`, None for non-f64
/// nodes and distributions without one) and each of its parents (`parameters`, None where there is no
/// derivative). Element-wise (shaped like the node's value) except where the distribution reduces itself
/// (multivariate Gaussian, Dirichlet).
#[derive(Debug, Clone)]
pub struct NodeLogProbGradient {
    pub value: Option<DoubleTensor>,
    pub parameters: Vec<Option<DoubleTensor>>,
}

/// ∂ log p(node) / ∂ value and ∂ log p(node) / ∂ parameter for each parameter, following the Java vertices'
/// `dLogProb`.
pub fn node_log_prob_gradient(
    graph: &Graph,
    state: &State,
    id: NodeId,
) -> Result<NodeLogProbGradient> {
    use DistributionKind::*;
    let kind = kind(graph, id)?;
    let parents = graph.parents(id);
    let p = |i: usize| -> Result<DoubleTensor> { Ok(value(state, parents[i])?.f64()?.clone()) };
    let q = |i: usize| -> Result<keanu_tensor::IntTensor> {
        Ok(value(state, parents[i])?.i64()?.clone())
    };
    let v = value(state, id)?;
    let some = Some;
    let (x, params): (Option<DoubleTensor>, Vec<Option<DoubleTensor>>) = match kind {
        Gaussian => {
            let g = d::Gaussian::new(p(0)?, p(1)?).d_log_prob(v.f64()?)?;
            (some(g.x), vec![some(g.mu), some(g.sigma)])
        }
        HalfGaussian => {
            let g = d::HalfGaussian::new(p(0)?).d_log_prob(v.f64()?)?;
            (some(g.x), vec![some(g.sigma)])
        }
        Cauchy => {
            let g = d::Cauchy::new(p(0)?, p(1)?).d_log_prob(v.f64()?)?;
            (some(g.x), vec![some(g.location), some(g.scale)])
        }
        HalfCauchy => {
            let g = d::HalfCauchy::new(p(0)?).d_log_prob(v.f64()?)?;
            (some(g.x), vec![some(g.scale)])
        }
        Beta => {
            let g = d::Beta::standard(p(0)?, p(1)?).d_log_prob(v.f64()?)?;
            (some(g.x), vec![some(g.alpha), some(g.beta)])
        }
        ChiSquared => {
            let g = d::ChiSquared::new(q(0)?).d_log_prob(v.f64()?)?;
            (some(g.x), vec![None])
        }
        Dirichlet => {
            let g = d::Dirichlet::new(p(0)?).d_log_prob(v.f64()?)?;
            (some(g.x), vec![some(g.concentration)])
        }
        Exponential => {
            let g = d::Exponential::new(p(0)?).d_log_prob(v.f64()?)?;
            (some(g.x), vec![some(g.rate)])
        }
        Gamma => {
            let g = d::Gamma::new(p(0)?, p(1)?).d_log_prob(v.f64()?)?;
            (some(g.x), vec![some(g.theta), some(g.k)])
        }
        InverseGamma => {
            let g = d::InverseGamma::new(p(0)?, p(1)?).d_log_prob(v.f64()?)?;
            (some(g.x), vec![some(g.alpha), some(g.beta)])
        }
        Laplace => {
            let g = d::Laplace::new(p(0)?, p(1)?).d_log_prob(v.f64()?)?;
            (some(g.x), vec![some(g.mu), some(g.beta)])
        }
        Logistic => {
            let g = d::Logistic::new(p(0)?, p(1)?).d_log_prob(v.f64()?)?;
            (some(g.x), vec![some(g.mu), some(g.s)])
        }
        LogNormal => {
            let g = d::LogNormal::new(p(0)?, p(1)?).d_log_prob(v.f64()?)?;
            (some(g.x), vec![some(g.mu), some(g.sigma)])
        }
        MultivariateGaussian => {
            let g = d::MultivariateGaussian::new(p(0)?, p(1)?).d_log_prob(v.f64()?)?;
            (some(g.x), vec![some(g.mu), some(g.covariance)])
        }
        Pareto => {
            let g = d::Pareto::new(p(0)?, p(1)?).d_log_prob(v.f64()?)?;
            (some(g.x), vec![some(g.location), some(g.scale)])
        }
        StudentT => {
            let g = d::StudentT::new(q(0)?).d_log_prob(v.f64()?)?;
            (some(g.x), vec![None])
        }
        Triangular => {
            let g = d::Triangular::new(p(0)?, p(1)?, p(2)?).d_log_prob(v.f64()?)?;
            (some(g.x), vec![some(g.x_min), some(g.x_max), some(g.c)])
        }
        Uniform => {
            // as UniformVertex: zero inside the support, -inf above x_max and +inf at or below x_min
            let x = v.f64()?;
            let (min, max) = (p(0)?, p(1)?);
            let dx = x
                .zip_with(&max, |x, max| if x > max { f64::NEG_INFINITY } else { 0.0 })?
                .zip_with(
                    &x.le(&min)?,
                    |d, below| if below { f64::INFINITY } else { d },
                )?;
            (some(dx), vec![None, None])
        }
        Kde(kde) => (some(kde.d_log_prob(v.f64()?)?.x), vec![]),
        Bernoulli => {
            let g = d::Bernoulli::new(p(0)?).d_log_prob(v.bool()?)?;
            (None, vec![some(g.prob_true)])
        }
        Binomial => {
            let g = d::Binomial::new(p(0)?, q(1)?).d_log_prob(v.i64()?)?;
            (None, vec![some(g.p), None])
        }
        Geometric => {
            let g = d::Geometric::new(p(0)?).d_log_prob(v.i64()?)?;
            (None, vec![some(g.p)])
        }
        Multinomial => {
            let g = d::Multinomial::new(q(0)?, p(1)?)?.d_log_prob(v.i64()?)?;
            (None, vec![None, some(g.p)])
        }
        Poisson => {
            let g = d::Poisson::new(p(0)?).d_log_prob(v.i64()?)?;
            (None, vec![some(g.mu)])
        }
        Categorical | UniformInt => (None, vec![None; parents.len()]),
    };
    Ok(NodeLogProbGradient {
        value: x,
        parameters: params,
    })
}

/// Sums a gradient over the dimensions that a tensor of `target` shape was broadcast along (as Java's
/// `AutoDiffBroadcast.correctForBroadcastPartialReverse`).
fn sum_to_shape(t: &DoubleTensor, target: &[usize]) -> Result<DoubleTensor> {
    if t.shape() == target {
        return Ok(t.clone());
    }
    let rank = t.rank();
    if target.len() > rank {
        return Ok(t.broadcast_to(target)?);
    }
    let offset = rank - target.len();
    let axes: Vec<isize> = (0..rank)
        .filter(|&i| i < offset || (target[i - offset] == 1 && t.shape()[i] != 1))
        .map(|i| i as isize)
        .collect();
    let summed = if axes.is_empty() {
        t.clone()
    } else {
        t.sum_axes(&axes)?
    };
    let shape: Vec<isize> = target.iter().map(|&d| d as isize).collect();
    Ok(summed.reshape(&shape)?)
}

/// The gradient of a joint log probability with respect to latent nodes, keyed by node.
#[derive(Debug, Clone, Default)]
pub struct LogProbGradient {
    gradients: HashMap<NodeId, DoubleTensor>,
}

impl LogProbGradient {
    /// The gradient with respect to a node, or None if the log probability does not depend on it.
    pub fn get(&self, node: NodeId) -> Option<&DoubleTensor> {
        self.gradients.get(&node)
    }

    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &DoubleTensor)> + '_ {
        self.gradients.iter().map(|(&id, t)| (id, t))
    }

    pub fn len(&self) -> usize {
        self.gradients.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gradients.is_empty()
    }

    pub fn into_map(self) -> HashMap<NodeId, DoubleTensor> {
        self.gradients
    }

    fn add(&mut self, id: NodeId, gradient: DoubleTensor) -> Result<()> {
        let summed = match self.gradients.remove(&id) {
            Some(existing) => existing.add(&gradient)?,
            None => gradient,
        };
        self.gradients.insert(id, summed);
        Ok(())
    }
}

/// True for nodes that log prob gradients can be taken with respect to: unobserved f64 probabilistic nodes.
fn is_latent_double(graph: &Graph, state: &State, id: NodeId) -> bool {
    let node = graph.node(id);
    node.is_probabilistic() && node.dtype() == DType::F64 && !state.is_observed(id)
}

/// The gradient of the total log probability of the probabilistic nodes `of` with respect to the latents in
/// `wrt` (as Java's `LogProbGradientCalculator`).
///
/// For each node in `of`: the derivative with respect to its own value is included if it is an unobserved f64
/// node, and the derivatives with respect to its parameters are propagated in reverse mode through
/// deterministic parents to the f64 latents in `wrt` (stopping at probabilistic nodes). Integer and boolean
/// latents in `wrt` are skipped.
pub fn log_prob_gradient(
    graph: &Graph,
    state: &State,
    of: &[NodeId],
    wrt: &[NodeId],
) -> Result<LogProbGradient> {
    let targets: Vec<NodeId> = wrt
        .iter()
        .copied()
        .filter(|&id| is_latent_double(graph, state, id))
        .collect();
    let mut result = LogProbGradient::default();
    for &node in of {
        let gradient = node_log_prob_gradient(graph, state, node)?;
        if is_latent_double(graph, state, node)
            && let Some(dx) = gradient.value
        {
            result.add(node, dx)?;
        }
        if targets.is_empty() {
            continue;
        }
        for (&parent, dp) in graph.parents(node).iter().zip(gradient.parameters) {
            let Some(dp) = dp else { continue };
            if graph.node(parent).dtype() != DType::F64 {
                continue;
            }
            let upstream = sum_to_shape(&dp, graph.node(parent).shape())?;
            let partials = reverse_partials_from(graph, state, parent, &upstream, &targets)?;
            for (latent, partial) in partials.into_map() {
                result.add(latent, partial)?;
            }
        }
    }
    Ok(result)
}

/// The gradient of the joint log probability of every probabilistic node with respect to every latent f64 node.
pub fn joint_log_prob_gradient(graph: &Graph, state: &State) -> Result<LogProbGradient> {
    let of: Vec<NodeId> = graph.probabilistic().collect();
    let wrt: Vec<NodeId> = of
        .iter()
        .copied()
        .filter(|&id| is_latent_double(graph, state, id))
        .collect();
    log_prob_gradient(graph, state, &of, &wrt)
}

/// The gradient of the log likelihood (observed nodes only) with respect to every latent f64 node.
pub fn log_likelihood_gradient(graph: &Graph, state: &State) -> Result<LogProbGradient> {
    let probabilistic: Vec<NodeId> = graph.probabilistic().collect();
    let of: Vec<NodeId> = probabilistic
        .iter()
        .copied()
        .filter(|&id| state.is_observed(id))
        .collect();
    let wrt: Vec<NodeId> = probabilistic
        .into_iter()
        .filter(|&id| is_latent_double(graph, state, id))
        .collect();
    log_prob_gradient(graph, state, &of, &wrt)
}
