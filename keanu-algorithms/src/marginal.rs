//! Recovering marginalised discrete latents after sampling: for each draw of a model rewritten by
//! [`marginalize`](keanu_graph::marginalize), the conditional distribution of each marginalised latent given the
//! draw, and draws of the latent from it.

use crate::error::{AlgorithmError, Result};
use crate::samples::NetworkSamples;
use crate::support::check_nodes;
use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, Marginalized, MarginalizedLatent, Model, NodeId, State, Value};
use keanu_tensor::DoubleTensor;

/// A marginalised latent's conditional distribution at each draw of a posterior, from [`recover_marginals`].
#[derive(Debug, Clone, PartialEq)]
pub struct LatentMarginals {
    latent: MarginalizedLatent,
    probabilities: Vec<DoubleTensor>,
}

impl LatentMarginals {
    /// The marginalised latent: its name in the original graph, shape, values and nodes in the rewritten one.
    pub fn latent(&self) -> &MarginalizedLatent {
        &self.latent
    }

    /// For each draw, the latent's conditional probabilities given the draw, shape [S..., K]: at [s, k],
    /// p(zₛ = lo + k | draw), which sums to 1 over k (see [`MarginalizedLatent::values`] for lo and K).
    pub fn probabilities(&self) -> &[DoubleTensor] {
        &self.probabilities
    }

    /// The number of draws.
    pub fn len(&self) -> usize {
        self.probabilities.len()
    }

    /// Whether there are no draws.
    pub fn is_empty(&self) -> bool {
        self.probabilities.is_empty()
    }

    /// The posterior probability of each value of each element, shape [S..., K]: the conditional probabilities
    /// averaged over the draws (a Rao–Blackwellised estimate, with less Monte Carlo error than counting draws of the
    /// latent).
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidArgument`] if there are no draws.
    pub fn mean(&self) -> Result<DoubleTensor> {
        let Some(first) = self.probabilities.first() else {
            return Err(AlgorithmError::InvalidArgument(
                "there are no draws to average".into(),
            ));
        };
        let mut total = vec![0.0; first.len()];
        for p in &self.probabilities {
            for (t, x) in total.iter_mut().zip(p.as_slice()) {
                *t += x;
            }
        }
        let count = self.probabilities.len() as f64;
        Ok(DoubleTensor::new(
            first.shape(),
            total.into_iter().map(|t| t / count).collect(),
        )?)
    }

    /// One draw of the latent for each posterior draw, from its conditional distribution given that draw: together
    /// with the posterior draws, draws from the joint posterior of the original model. Values are i64, or bool for
    /// a Bernoulli latent, of the latent's shape.
    ///
    /// # Errors
    ///
    /// Only if the probabilities don't have the latent's shape, which they always do as [`recover_marginals`]
    /// returns them.
    pub fn draw(&self, rng: &mut KeanuRng) -> Result<Vec<Value>> {
        self.probabilities
            .iter()
            .map(|p| Ok(self.latent.draw(p, rng)?))
            .collect()
    }
}

/// The conditional distribution of every latent `marginalized` records, at each draw in `samples`: for draws of the
/// rewritten model (from NUTS on it, say), p(z = k | draw) for each element of each marginalised latent z. The
/// draws are set in a copy of `state` (a state of the rewritten graph, such as the one [`marginalize`] returned),
/// and each latent's [log weights](MarginalizedLatent::log_weights) are recomputed and normalised.
///
/// For several chains, call it once per chain (`ChainSamples::chains`).
///
/// ```
/// use keanu_algorithms::marginal::recover_marginals;
/// use keanu_algorithms::NetworkSamples;
/// use keanu_graph::{GraphBuilder, Model, marginalize};
/// use keanu_tensor::{BoolTensor, DoubleTensor};
///
/// // is a coin biased (towards heads, by an unknown amount), given four tosses?
/// let b = GraphBuilder::new();
/// let bias = b.uniform(0.5, 1.0);
/// let biased = b.bernoulli(0.5);
/// let p = b.where_(biased, bias, b.scalar(0.5));
/// let tosses = b.bernoulli_with_shape(&[4], p);
/// b.observe(tosses, BoolTensor::vector(vec![true, true, true, false]));
/// b.set_value(bias, DoubleTensor::scalar(0.9));
/// let (bias, biased) = (bias.id(), biased.id());
/// let (graph, state) = b.build()?;
/// let (graph, mut state, marginalized) = marginalize(&graph, &state, &[biased])?;
/// let model = Model::new(graph, &state)?;
///
/// // two "draws" of the bias
/// let bias = marginalized.node(bias).unwrap();
/// let mut samples = NetworkSamples::new(model.graph(), &[bias])?;
/// for value in [0.9, 0.6] {
///     state.set_and_cascade(model.graph(), bias, DoubleTensor::scalar(value))?;
///     samples.record(&state, model.log_prob(&state)?)?;
/// }
/// let marginals = recover_marginals(&model, &state, &marginalized, &samples)?;
/// let p = marginals[0].probabilities();
/// let biased_given = |q: f64| {
///     let biased = q.powi(3) * (1.0 - q);
///     biased / (biased + 0.5f64.powi(4))
/// };
/// assert!((p[0].as_slice()[1] - biased_given(0.9)).abs() < 1e-12);
/// assert!((p[1].as_slice()[1] - biased_given(0.6)).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// [`marginalize`]: keanu_graph::marginalize
///
/// # Errors
///
/// [`AlgorithmError::InvalidArgument`] if `samples` doesn't record every latent the log weights depend on (naming
/// the missing ones), [`GraphError::UnknownNode`](keanu_graph::GraphError::UnknownNode) if a sampled node or a
/// recorded latent's log weights aren't nodes of the model's graph, and any error evaluating the log weights.
pub fn recover_marginals(
    model: &Model,
    state: &State,
    marginalized: &Marginalized,
    samples: &NetworkSamples,
) -> Result<Vec<LatentMarginals>> {
    let graph = model.graph();
    check_nodes(graph, samples.nodes())?;
    let targets: Vec<NodeId> = marginalized
        .latents()
        .iter()
        .map(MarginalizedLatent::log_weights)
        .collect();
    check_nodes(graph, &targets)?;
    let mut sampled = vec![false; graph.len()];
    for id in samples.nodes() {
        sampled[id.index()] = true;
    }
    let missing: Vec<String> = latent_ancestors(graph, state, &targets)
        .into_iter()
        .filter(|id| !sampled[id.index()])
        .map(|id| graph.describe(id))
        .collect();
    if !missing.is_empty() {
        return Err(AlgorithmError::InvalidArgument(format!(
            "the marginalised latents depend on latents without samples: {}",
            missing.join(", ")
        )));
    }

    let mut work = state.clone();
    // every latent the targets need has a value from the samples, so evaluating them samples nothing
    let mut rng = KeanuRng::seed_from_u64(0);
    let mut probabilities: Vec<Vec<DoubleTensor>> =
        vec![Vec::with_capacity(samples.len()); targets.len()];
    for sample in samples.iter() {
        for &id in samples.nodes() {
            if let Some(value) = sample.get(id) {
                work.set(graph, id, value)?;
            }
        }
        work.eval(graph, &targets, &mut rng)?;
        for (latent, p) in marginalized.latents().iter().zip(&mut probabilities) {
            p.push(latent.probabilities(&work)?);
        }
    }
    Ok(marginalized
        .latents()
        .iter()
        .cloned()
        .zip(probabilities)
        .map(|(latent, probabilities)| LatentMarginals {
            latent,
            probabilities,
        })
        .collect())
}

/// The latents (unobserved probabilistic nodes) among the ancestors of `targets`, up to the nearest probabilistic
/// ones, in id order.
fn latent_ancestors(graph: &Graph, state: &State, targets: &[NodeId]) -> Vec<NodeId> {
    let mut seen = vec![false; graph.len()];
    let mut stack = targets.to_vec();
    let mut latents = Vec::new();
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.index()], true) {
            continue;
        }
        let node = &graph[id];
        if node.is_probabilistic() {
            if !state.is_observed(id) {
                latents.push(id);
            }
        } else {
            stack.extend(node.parents().iter().copied());
        }
    }
    latents.sort();
    latents
}
