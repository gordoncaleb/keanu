//! The flat parameter space gradient based algorithms work in: a model's continuous latents laid out as one vector,
//! optionally mapped to unconstrained space through each latent's [`Bijector`].
//!
//! A [`Parameterisation`] is built from a [`Model`] and a [`State`]. It selects the model's continuous latents
//! ([`Model::continuous_latents`]), and in [`Space::Unconstrained`] (the default) resolves a bijector for each from
//! the support of its distribution ([`bijector_for`]). The flat vector `z` is the concatenation of each latent's
//! unconstrained coordinates, in latent order. In [`Space::Constrained`] every bijector is the identity, so `z` is
//! the concatenation of the latents' values, each row-major, which is the layout the gradient algorithms have always
//! used.
//!
//! ```
//! use keanu_graph::{GraphBuilder, Model, Parameterisation};
//! use keanu_tensor::DoubleTensor;
//!
//! let b = GraphBuilder::new();
//! let sigma = b.half_gaussian(1.0).label("sigma");
//! let y = b.gaussian(0.0, sigma);
//! b.observe(y, DoubleTensor::scalar(0.5));
//! let sigma = sigma.id();
//! let (graph, mut state) = b.build()?;
//! state.set(&graph, sigma, DoubleTensor::scalar(2.0))?;
//! let model = Model::new(graph, &state)?;
//!
//! let param = Parameterisation::new(&model, &state)?;
//! let z = param.from_state(&state)?; // log(2): sigma = exp(z)
//! assert!((z[0] - 2f64.ln()).abs() < 1e-12);
//!
//! // any real z is a valid point: here sigma = exp(-3)
//! let (log_prob, gradient) = param.log_prob_and_gradient(&[-3.0], true, &mut state)?;
//! assert!(log_prob.is_finite() && gradient.is_some());
//! assert!((state.f64(sigma).unwrap().as_slice()[0] - (-3f64).exp()).abs() < 1e-15);
//! # Ok::<(), keanu_graph::GraphError>(())
//! ```
//!
//! # Fixed bounds
//!
//! Bounds that are parameters of a distribution (a Uniform's limits, a Pareto's location) are read from the state
//! when the parameterisation is built and stay fixed. A bound that depends on another continuous latent would move
//! as that latent moves, which no fixed bijector can follow, so construction fails with an error naming both nodes
//! and suggesting a reparameterisation (for `x ~ Uniform(lo, hi)` with a latent `lo`, `u ~ Uniform(0, 1)` and
//! `x = lo + (hi − lo)·u`). Which parameters a support depends on is found by evaluating the support again with each
//! latent-dependent parameter changed: the parameters whose change moves the support are bounds.
//!
//! # Second order
//!
//! Hessian-vector products will be added as `hvp(z, v, jacobian)`, reusing this layout: the constrained tangent
//! of each latent is [`Bijector::forward_jvp`] of its slice of `v`, the model's Hessian-vector product is pulled back
//! with [`Bijector::forward_vjp`] as [`Parameterisation::pull_back`] does for the gradient, plus the curvature terms
//! of the bijectors themselves.

use crate::error::{GraphError, Result};
use crate::graph::{Graph, NodeId};
use crate::logprob::{LogProbGradient, node_support};
use crate::model::Model;
use crate::op::{DistributionKind, Op, Support};
use crate::state::State;
use crate::transform::{Bijector, Identity, bijector_for};
use crate::value::Value;
use keanu_distributions::KeanuRng;
use keanu_tensor::DoubleTensor;
use std::ops::Range;

/// The space a gradient based algorithm moves in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Space {
    /// Each latent is mapped to ℝⁿ through the bijector of its support, so every point is inside the support. The
    /// default.
    #[default]
    Unconstrained,
    /// Latents are used as they are (the identity everywhere): points can leave the support, where the log
    /// probability is −∞.
    Constrained,
}

/// One latent's place in the flat layout.
#[derive(Debug)]
struct Slot {
    id: NodeId,
    range: Range<usize>,
    bijector: Box<dyn Bijector>,
}

/// A model's continuous latents as one flat vector, in constrained or unconstrained [`Space`]. See the
/// [module documentation](self).
#[derive(Debug)]
pub struct Parameterisation<'m> {
    model: &'m Model,
    space: Space,
    latents: Vec<NodeId>,
    slots: Vec<Slot>,
    dimension: usize,
}

/// A node's label, quoted, or its index.
fn describe(graph: &Graph, id: NodeId) -> String {
    match graph[id].label() {
        Some(label) => format!("{label:?}"),
        None => format!("#{}", id.index()),
    }
}

/// For each node, a continuous latent its value depends on (itself, for a continuous latent), following
/// deterministic, unobserved nodes. Ids are a topological order, so one forward pass suffices.
fn latent_dependencies(model: &Model, state: &State) -> Vec<Option<NodeId>> {
    let graph = model.graph();
    let mut depends: Vec<Option<NodeId>> = vec![None; graph.len()];
    for &id in &model.continuous_latents() {
        depends[id.index()] = Some(id);
    }
    for id in graph.ids() {
        let node = &graph[id];
        if depends[id.index()].is_some() || node.is_probabilistic() || state.is_observed(id) {
            continue;
        }
        depends[id.index()] = node.parents().iter().find_map(|p| depends[p.index()]);
    }
    depends
}

/// The value changed in every element, so a support computed from it differs wherever it depends on it.
fn perturbed(value: &Value) -> Value {
    match value {
        Value::F64(t) => Value::F64(t.map(|x| x + 1.0 + x.abs())),
        Value::I64(t) => Value::I64(t.map(|x| x.wrapping_add(1))),
        Value::Bool(t) => Value::Bool(t.map(|x| !x)),
    }
}

/// Fails if the support of the latent `id` moves with a parameter that depends on another continuous latent.
fn check_fixed_support(
    graph: &Graph,
    state: &State,
    id: NodeId,
    kind: &DistributionKind,
    support: &Support,
    depends: &[Option<NodeId>],
) -> Result<()> {
    let parents = graph[id].parents();
    let values: Vec<&Value> = parents
        .iter()
        .map(|&p| state.value(p).ok_or(GraphError::MissingValue(p)))
        .collect::<Result<_>>()?;
    for (i, &parent) in parents.iter().enumerate() {
        let Some(latent) = depends[parent.index()] else {
            continue;
        };
        let changed = perturbed(values[i]);
        let mut probe = values.clone();
        probe[i] = &changed;
        let moves = match kind.support(&probe) {
            Ok(other) => &other != support,
            Err(_) => true,
        };
        if moves {
            return Err(GraphError::Transform(format!(
                "the support of {} depends on the latent {}, so its bounds would move during inference and no \
                 fixed transform can keep it inside them; reparameterise it, e.g. for x ~ Uniform(lo, hi) use \
                 u ~ Uniform(0, 1) and x = lo + (hi - lo) * u, or use Space::Constrained",
                describe(graph, id),
                describe(graph, latent),
            )));
        }
    }
    Ok(())
}

impl<'m> Parameterisation<'m> {
    /// The unconstrained parameterisation of `model`'s continuous latents, with bijectors resolved from their
    /// supports given the parameter values in `state`.
    ///
    /// # Errors
    ///
    /// [`GraphError::Transform`] if a latent's support has no transform (non-finite bounds, a discrete support) or
    /// depends on another continuous latent (see the [module documentation](self)), and
    /// [`GraphError::MissingValue`] if a latent's parameter has no value in `state`.
    pub fn new(model: &'m Model, state: &State) -> Result<Self> {
        Self::with_space(model, state, Space::Unconstrained)
    }

    /// The parameterisation of `model`'s continuous latents in `space`. [`Space::Constrained`] needs no supports, so
    /// it never fails.
    ///
    /// # Errors
    ///
    /// As [`Parameterisation::new`], for [`Space::Unconstrained`].
    pub fn with_space(model: &'m Model, state: &State, space: Space) -> Result<Self> {
        let graph = model.graph();
        let latents = model.continuous_latents();
        let depends = match space {
            Space::Unconstrained => latent_dependencies(model, state),
            Space::Constrained => Vec::new(),
        };
        let mut slots = Vec::with_capacity(latents.len());
        let mut offset = 0;
        for &id in &latents {
            let shape = graph[id].shape();
            let bijector: Box<dyn Bijector> = match space {
                Space::Constrained => Box::new(Identity::new(shape)),
                Space::Unconstrained => {
                    let support = node_support(graph, state, id)?;
                    if let Op::Distribution(op) = graph[id].op() {
                        check_fixed_support(graph, state, id, &op.kind, &support, &depends)?;
                    }
                    bijector_for(&support, shape).map_err(|e| in_latent(graph, id, e))?
                }
            };
            let len = bijector.unconstrained_len();
            slots.push(Slot {
                id,
                range: offset..offset + len,
                bijector,
            });
            offset += len;
        }
        Ok(Self {
            model,
            space,
            latents,
            slots,
            dimension: offset,
        })
    }

    /// The model.
    pub fn model(&self) -> &'m Model {
        self.model
    }

    /// The space of the flat vector.
    pub fn space(&self) -> Space {
        self.space
    }

    /// The continuous latents, in layout order (the order of [`Model::continuous_latents`]).
    pub fn latents(&self) -> &[NodeId] {
        &self.latents
    }

    /// The length of the flat vector. A simplex of K elements contributes K − 1 coordinates in unconstrained space,
    /// so this can be less than the latents' total size.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// The coordinates of a latent in the flat vector, or `None` if it is not one of the latents.
    pub fn range(&self, latent: NodeId) -> Option<Range<usize>> {
        self.slots
            .iter()
            .find(|s| s.id == latent)
            .map(|s| s.range.clone())
    }

    /// The latent a coordinate of the flat vector belongs to, or `None` past the end.
    pub fn coordinate_latent(&self, coordinate: usize) -> Option<NodeId> {
        self.slots
            .iter()
            .find(|s| s.range.contains(&coordinate))
            .map(|s| s.id)
    }

    fn check_len(&self, z: &[f64]) -> Result<()> {
        if z.len() == self.dimension {
            Ok(())
        } else {
            Err(GraphError::InvalidValue(format!(
                "a point has {} coordinates, expected the parameterisation's {}",
                z.len(),
                self.dimension
            )))
        }
    }

    /// The constrained value of each latent at `z`, in layout order. Fails if `z` has the wrong length.
    pub fn to_constrained(&self, z: &[f64]) -> Result<Vec<DoubleTensor>> {
        self.check_len(z)?;
        self.slots
            .iter()
            .map(|s| s.bijector.forward(&z[s.range.clone()]))
            .collect()
    }

    /// Sets each latent to its constrained value at `z` and recomputes their deterministic descendants. Fails if `z`
    /// has the wrong length.
    pub fn to_state(&self, z: &[f64], state: &mut State) -> Result<()> {
        let values = self.to_constrained(z)?;
        self.model.set_and_cascade(
            state,
            self.latents
                .iter()
                .copied()
                .zip(values.into_iter().map(Value::F64)),
        )
    }

    /// The point of the latents' current values: the inverse of [`to_state`](Self::to_state).
    ///
    /// # Errors
    ///
    /// [`GraphError::MissingValue`] if a latent has no value, and, in [`Space::Unconstrained`],
    /// [`GraphError::Transform`] naming the latent if its value is outside its support or on its boundary (where the
    /// unconstrained value would be infinite). Algorithms report the latter as an impossible start.
    pub fn from_state(&self, state: &State) -> Result<Vec<f64>> {
        let graph = self.model.graph();
        let mut z = Vec::with_capacity(self.dimension);
        for slot in &self.slots {
            let value = state
                .f64(slot.id)
                .ok_or(GraphError::MissingValue(slot.id))?;
            match self.space {
                // exactly the values, unchecked, as the algorithms have always read them
                Space::Constrained => z.extend_from_slice(value.as_slice()),
                Space::Unconstrained => z.extend(
                    slot.bijector
                        .inverse(value)
                        .map_err(|e| in_latent(graph, slot.id, e))?,
                ),
            }
        }
        Ok(z)
    }

    /// Σ log |det J| of the bijectors at `z`: what is added to the log probability with the Jacobian. Zero in
    /// [`Space::Constrained`]. Fails if `z` has the wrong length.
    pub fn log_det_jacobian(&self, z: &[f64]) -> Result<f64> {
        self.check_len(z)?;
        if self.space == Space::Constrained {
            return Ok(0.0);
        }
        self.slots
            .iter()
            .map(|s| s.bijector.log_det_jacobian(&z[s.range.clone()]))
            .sum()
    }

    /// The gradient with respect to `z` of a function whose gradient with respect to each latent's constrained value
    /// is `gradient` (a latent missing from it has a zero gradient), by the chain rule through each bijector. Use it
    /// for targets other than the joint log probability, such as [`Model::log_likelihood_gradient`]. Fails if `z`
    /// has the wrong length.
    pub fn pull_back(&self, z: &[f64], gradient: &LogProbGradient) -> Result<Vec<f64>> {
        self.check_len(z)?;
        let mut flat = Vec::with_capacity(self.dimension);
        for slot in &self.slots {
            let y = &z[slot.range.clone()];
            match gradient.get(slot.id) {
                Some(g) => flat.extend(slot.bijector.forward_vjp(y, g)?),
                None => flat.extend(std::iter::repeat_n(0.0, slot.range.len())),
            }
        }
        Ok(flat)
    }

    /// Sets the latents to their values at `z` (as [`to_state`](Self::to_state)) and returns the joint log
    /// probability there and, unless it is impossible (−∞), its gradient with respect to `z`.
    ///
    /// With `jacobian`, the log probability is that of `z`, a density over the flat vector:
    /// [`Model::log_prob`] at the constrained values plus [`log_det_jacobian`](Self::log_det_jacobian), as samplers
    /// and variational inference need. Without it, it is the model's log probability at the constrained values,
    /// whose maxima are the constrained maxima, as optimizers need. In [`Space::Constrained`] the Jacobian is zero
    /// and both are exactly [`Model::log_prob_and_gradient`].
    ///
    /// Fails if `z` has the wrong length or the model cannot be evaluated.
    pub fn log_prob_and_gradient(
        &self,
        z: &[f64],
        jacobian: bool,
        state: &mut State,
    ) -> Result<(f64, Option<Vec<f64>>)> {
        self.to_state(z, state)?;
        let (log_prob, gradient) = self.model.log_prob_and_gradient(state)?;
        let Some(gradient) = gradient else {
            return Ok((log_prob, None));
        };
        let mut flat = self.pull_back(z, &gradient)?;
        if !jacobian || self.space == Space::Constrained {
            return Ok((log_prob, Some(flat)));
        }
        let mut log_det = 0.0;
        for slot in &self.slots {
            let y = &z[slot.range.clone()];
            log_det += slot.bijector.log_det_jacobian(y)?;
            let jacobian_gradient = slot.bijector.log_det_jacobian_gradient(y)?;
            for (g, j) in flat[slot.range.clone()].iter_mut().zip(jacobian_gradient) {
                *g += j;
            }
        }
        Ok((log_prob + log_det, Some(flat)))
    }

    /// A random starting point with every coordinate drawn from Uniform(−2, 2), as Stan initialises its samplers
    /// and optimizers. Opt-in: algorithms start from the values in the state they are given. Apply it with
    /// [`to_state`](Self::to_state). Meant for [`Space::Unconstrained`]; in [`Space::Constrained`] the coordinates
    /// are the latents' values and can be outside their supports.
    pub fn uniform_initial_point(&self, rng: &mut KeanuRng) -> Vec<f64> {
        (0..self.dimension)
            .map(|_| rng.next_f64_between(-2.0, 2.0))
            .collect()
    }
}

/// Prefixes a transform error with the latent it is about.
fn in_latent(graph: &Graph, id: NodeId, error: GraphError) -> GraphError {
    match error {
        GraphError::Transform(message) => {
            GraphError::Transform(format!("latent {}: {message}", describe(graph, id)))
        }
        other => other,
    }
}
