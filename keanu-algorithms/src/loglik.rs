//! Pointwise log-likelihood: the log likelihood of each observation at each posterior draw, the input to
//! leave-one-out cross-validation, WAIC and export to other tools.
//!
//! The "points" follow how PyMC and ArviZ label their `log_likelihood` variables: one per element of an observed
//! node whose distribution has one value per element (the continuous and count distributions, truncated, censored
//! and mixture distributions, the Bernoulli, the categorical and the ordered logistic), and one for the whole node
//! otherwise (a multivariate Gaussian, even a batch of vectors, a Dirichlet, a hidden Markov model, the linear
//! Gaussian state space model, a custom distribution). See [`node_log_prob_terms`].
//!
//! The values are computed from recorded draws, after sampling: each draw's latent values are set in a copy of the
//! state, the deterministic nodes are recomputed and the observed nodes' log probabilities evaluated. So it works
//! for the draws of any algorithm (NUTS, Metropolis–Hastings, the Laplace approximation, ADVI), as long as the
//! samples record every latent the observed nodes depend on.
//!
//! Factor nodes ([`GraphBuilder::factor`](keanu_graph::GraphBuilder::factor)) count in
//! [`Model::log_likelihood`], but they are not observations, so they have no points here: for a model with factors,
//! the points of a draw sum to its log likelihood less the factors' terms. Observed boolean constraints have no
//! points either.
//!
//! With the `parallel` feature the draws are evaluated on rayon's thread pool; the results are the same either way.
//!
//! ```
//! use keanu_algorithms::loglik::PointwiseLogLik;
//! use keanu_algorithms::mcmc::chains::sample_chains;
//! use keanu_algorithms::mcmc::nuts::Nuts;
//! use keanu_graph::{GraphBuilder, Model};
//! use keanu_tensor::DoubleTensor;
//!
//! let b = GraphBuilder::new();
//! let mu = b.gaussian(0.0, 5.0);
//! let y = b.gaussian_with_shape(&[3], mu, 1.0).label("y");
//! b.observe(y, DoubleTensor::vector(vec![0.4, 1.3, 0.9]));
//! let mu = mu.id();
//! let (graph, state) = b.build()?;
//! let model = Model::new(graph, &state)?;
//! let nuts = Nuts::default().with_adapt_count(100);
//! let chains = sample_chains(2, 1, &state, |state, rng| nuts.sample(&model, state, &[mu], 100, rng))?;
//!
//! let loglik = PointwiseLogLik::from_chains(&model, &state, &chains)?;
//! assert_eq!(loglik.names(), &["y[0]", "y[1]", "y[2]"]);
//! assert_eq!((loglik.num_chains(), loglik.draws(), loglik.len()), (2, 100, 3));
//! // every draw of every chain, chain by chain, for the second observation
//! assert_eq!(loglik.point(1).len(), 200);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{AlgorithmError, Result};
use crate::samples::{ChainSamples, NetworkSamples};
use crate::support::node_name;
use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, Model, NodeId, State, node_log_prob_terms, upstream_lambda_section};

/// One point of a pointwise log likelihood: an element of an observed node, or the whole node.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct LogLikPoint {
    /// The observed node.
    pub node: NodeId,
    /// The index of the element within the node; empty for a node with one point.
    pub index: Vec<usize>,
    /// The point's name: the node's label (or `#` and its index) followed by the index, as `y[3]` or `y[1, 2]`, or
    /// the bare name for a node with one point.
    pub name: String,
}

/// The log likelihood of each observation point at each draw of each chain. See the [module documentation](self).
#[derive(Debug, Clone, PartialEq)]
pub struct PointwiseLogLik {
    points: Vec<LogLikPoint>,
    names: Vec<String>,
    /// `[chain][draw][point]`.
    values: Vec<Vec<Vec<f64>>>,
}

impl PointwiseLogLik {
    /// The pointwise log likelihood of every draw of every chain.
    ///
    /// `state` gives the observations and any values the samples don't record (constants, values set while
    /// building); the draws' values replace the latents' values in a copy of it.
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidArgument`] if the model has no observed probabilistic nodes, if there are no
    /// draws, or if the samples don't record a latent an observed node depends on (naming the missing latents), and
    /// any error evaluating the model at a draw.
    pub fn from_chains(model: &Model, state: &State, samples: &ChainSamples) -> Result<Self> {
        let layout = Layout::new(model, state, &samples.chains()[0])?;
        let values = samples
            .chains()
            .iter()
            .map(|chain| layout.evaluate(model, state, chain))
            .collect::<Result<_>>()?;
        Ok(layout.finish(values))
    }

    /// The pointwise log likelihood of one chain's (or one approximation's) draws, as a single chain. Fails as
    /// [`from_chains`](Self::from_chains).
    pub fn from_samples(model: &Model, state: &State, samples: &NetworkSamples) -> Result<Self> {
        let layout = Layout::new(model, state, samples)?;
        let values = vec![layout.evaluate(model, state, samples)?];
        Ok(layout.finish(values))
    }

    /// The points, in order: observed nodes in topological order, each node's points in row-major order.
    pub fn points(&self) -> &[LogLikPoint] {
        &self.points
    }

    /// The name of each point (as [`LogLikPoint::name`]), in order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether there are no points (never true for a value this module returns).
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The number of chains.
    pub fn num_chains(&self) -> usize {
        self.values.len()
    }

    /// The number of draws in each chain.
    pub fn draws(&self) -> usize {
        self.values.first().map_or(0, Vec::len)
    }

    /// Every value, `[chain][draw][point]`.
    pub fn values(&self) -> &[Vec<Vec<f64>>] {
        &self.values
    }

    /// One chain's values, `[draw][point]`, or `None` past the last chain.
    pub fn chain(&self, chain: usize) -> Option<&[Vec<f64>]> {
        self.values.get(chain).map(Vec::as_slice)
    }

    /// One point's value at every draw of every chain, chain after chain (`num_chains · draws` values).
    ///
    /// # Panics
    ///
    /// If `point` is not less than [`len`](Self::len).
    pub fn point(&self, point: usize) -> Vec<f64> {
        assert!(point < self.len(), "point {point} of {}", self.len());
        self.values
            .iter()
            .flat_map(|chain| chain.iter().map(move |draw| draw[point]))
            .collect()
    }

    /// The sum over the points at each draw of each chain, `[chain][draw]`: the log likelihood of the observed
    /// nodes, without the factors' terms.
    pub fn totals(&self) -> Vec<Vec<f64>> {
        self.values
            .iter()
            .map(|chain| chain.iter().map(|draw| draw.iter().sum()).collect())
            .collect()
    }
}

/// What to evaluate at each draw: the observed nodes, their points, and the latents to set from the samples.
struct Layout {
    observed: Vec<NodeId>,
    points: Vec<LogLikPoint>,
    /// The observed nodes' parents, which each draw recomputes (back to the probabilistic nodes).
    parents: Vec<NodeId>,
    /// The latents the observed nodes depend on.
    latents: Vec<NodeId>,
}

impl Layout {
    /// The layout for samples like `first`, whose first draw gives the shape of each node's terms.
    fn new(model: &Model, state: &State, first: &NetworkSamples) -> Result<Self> {
        crate::support::check_likelihood(model)?;
        let graph = model.graph();
        let sampled = first.nodes();
        let observed: Vec<NodeId> = model
            .observed()
            .iter()
            .copied()
            .filter(|&id| graph[id].is_probabilistic())
            .collect();
        if observed.is_empty() {
            return Err(AlgorithmError::InvalidArgument(
                "the model has no observed probabilistic nodes, so it has no pointwise log likelihood".into(),
            ));
        }
        // the latents the observed nodes' parameters depend on, walking up through deterministic nodes and
        // stopping at probabilistic and observed ones
        let parents: Vec<NodeId> = observed
            .iter()
            .flat_map(|&id| graph[id].parents().iter().copied())
            .collect();
        let latents: Vec<NodeId> = upstream_lambda_section(graph, state, &parents)?
            .latent_and_observed()
            .iter()
            .copied()
            .filter(|&id| graph[id].is_probabilistic() && !state.is_observed(id))
            .collect();
        let missing: Vec<String> = latents
            .iter()
            .filter(|id| !sampled.contains(id))
            .map(|&id| graph.describe(id))
            .collect();
        if !missing.is_empty() {
            return Err(AlgorithmError::InvalidArgument(format!(
                "the observed nodes depend on latents the samples don't record: {}",
                missing.join(", ")
            )));
        }
        if first.is_empty() {
            return Err(AlgorithmError::InvalidArgument(
                "there are no draws to evaluate the log likelihood at".into(),
            ));
        }
        // the points of each node, from the shape of its log probability terms at the first draw (the shape
        // depends only on the nodes' shapes, so it is the same at every draw)
        let mut layout = Self {
            observed,
            points: Vec::new(),
            parents,
            latents,
        };
        let mut work = state.clone();
        layout.set_draw(model, &mut work, first, 0)?;
        for &id in &layout.observed {
            let shape = node_log_prob_terms(graph, &work, id)?.shape().to_vec();
            layout.points.extend(node_points(graph, id, &shape));
        }
        Ok(layout)
    }

    /// Sets the latents to their values at one draw and recomputes the deterministic nodes between them (and any
    /// other values) and the observed nodes.
    fn set_draw(
        &self,
        model: &Model,
        work: &mut State,
        samples: &NetworkSamples,
        index: usize,
    ) -> Result<()> {
        let graph = model.graph();
        let sample = samples
            .sample(index)
            .ok_or_else(|| AlgorithmError::InvalidArgument(format!("no draw {index}")))?;
        let values = self
            .latents
            .iter()
            .map(|&id| {
                sample.get(id).map(|value| (id, value)).ok_or_else(|| {
                    AlgorithmError::InvalidArgument(format!(
                        "draw {index} lacks {}",
                        graph.describe(id)
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for (id, value) in values {
            work.set(graph, id, value)?;
        }
        // every probabilistic node the observed nodes depend on has a value now, so nothing is sampled
        work.eval(graph, &self.parents, &mut KeanuRng::seed_from_u64(0))?;
        Ok(())
    }

    /// The values `[draw][point]` of one chain.
    fn evaluate(
        &self,
        model: &Model,
        state: &State,
        samples: &NetworkSamples,
    ) -> Result<Vec<Vec<f64>>> {
        let draw = |work: &mut State, index: usize| self.draw(model, work, samples, index);
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            (0..samples.len())
                .into_par_iter()
                .map_init(|| state.clone(), draw)
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut work = state.clone();
            (0..samples.len()).map(|i| draw(&mut work, i)).collect()
        }
    }

    /// The values of the points at one draw.
    fn draw(
        &self,
        model: &Model,
        work: &mut State,
        samples: &NetworkSamples,
        index: usize,
    ) -> Result<Vec<f64>> {
        let graph = model.graph();
        self.set_draw(model, work, samples, index)?;
        let mut row = Vec::with_capacity(self.points.len());
        for &id in &self.observed {
            row.extend_from_slice(node_log_prob_terms(graph, work, id)?.as_slice());
        }
        if row.len() != self.points.len() {
            return Err(AlgorithmError::InvalidArgument(format!(
                "draw {index} has {} log likelihood terms, expected {}",
                row.len(),
                self.points.len()
            )));
        }
        Ok(row)
    }

    fn finish(self, values: Vec<Vec<Vec<f64>>>) -> PointwiseLogLik {
        let names = self.points.iter().map(|p| p.name.clone()).collect();
        PointwiseLogLik {
            points: self.points,
            names,
            values,
        }
    }
}

/// The points of an observed node whose log probability terms have `shape`.
fn node_points(graph: &Graph, id: NodeId, shape: &[usize]) -> Vec<LogLikPoint> {
    let base = node_name(graph, id);
    if shape.is_empty() {
        return vec![LogLikPoint {
            node: id,
            index: Vec::new(),
            name: base,
        }];
    }
    let count: usize = shape.iter().product();
    (0..count)
        .map(|flat| {
            let mut index = vec![0; shape.len()];
            let mut rest = flat;
            for (axis, &dim) in shape.iter().enumerate().rev() {
                index[axis] = rest % dim;
                rest /= dim;
            }
            let joined: Vec<String> = index.iter().map(usize::to_string).collect();
            LogLikPoint {
                node: id,
                name: format!("{base}[{}]", joined.join(", ")),
                index,
            }
        })
        .collect()
}
