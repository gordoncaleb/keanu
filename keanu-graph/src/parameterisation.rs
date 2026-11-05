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
//! [`Parameterisation::hvp`] is the exact product of the Hessian of the log probability (with or without the
//! Jacobian) with a direction `v`, and [`Parameterisation::hessian`] the whole Hessian. They reuse this layout: the
//! constrained tangent of each latent is [`Bijector::forward_jvp`] of its slice of `v`; the model's Hessian-vector
//! product in the constrained values is computed by forward-over-reverse differentiation (a forward sweep of the
//! tangents through the deterministic nodes, then a reverse sweep carrying each cotangent and its derivative, with
//! each op's and distribution's second derivatives); it is pulled back with [`Bijector::forward_vjp`] as
//! [`Parameterisation::pull_back`] does for the gradient; and the curvature of the bijectors is added:
//! [`Bijector::forward_vjp_jvp`] of the constrained gradient, and with the Jacobian
//! [`Bijector::log_det_jacobian_hvp`]. Nothing is approximated by finite differences.
//!
//! ```
//! use keanu_graph::{GraphBuilder, Model, Parameterisation};
//! use keanu_tensor::DoubleTensor;
//!
//! let b = GraphBuilder::new();
//! let mu = b.gaussian(0.0, 2.0);
//! let y = b.gaussian(mu, 0.5);
//! b.observe(y, DoubleTensor::scalar(1.0));
//! let mu = mu.id();
//! let (graph, mut state) = b.build()?;
//! state.set(&graph, mu, DoubleTensor::scalar(0.3))?;
//! let model = Model::new(graph, &state)?;
//! let param = Parameterisation::new(&model, &state)?;
//!
//! // log p = −mu²/8 − 2 (1 − mu)² + const, so the Hessian is −1/4 − 4
//! let hessian = param.hessian(&[0.3], true, &mut state)?;
//! assert!((hessian.as_slice()[0] + 4.25).abs() < 1e-12);
//! # Ok::<(), keanu_graph::GraphError>(())
//! ```
//!
//! Every op of the graph has a second-order rule except lambdas, and every distribution kind except custom
//! distributions. (A truncated or censored gamma, beta or negative binomial whose shape depends on the latents has
//! one: its CDF's second shape derivatives come from those of the regularised incomplete gamma and beta functions
//! in [`keanu_tensor::special`]; the linear Gaussian state space model differentiates its Kalman filter.) A model
//! whose log probability depends on the latents through a lambda or a custom distribution fails with
//! [`GraphError::NoSecondOrderRule`] naming the node. Second derivatives follow the
//! first-order gradients' conventions (the uniform has none with respect to its bounds, and the Laplace density
//! takes the almost-everywhere value at its kink).
//! Eigenvector derivatives assume distinct eigenvalues, as their first derivatives do.

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

/// For each node, one of `sources` (latents) its value depends on (itself, for a source), following
/// deterministic, unobserved nodes. Ids are a topological order, so one forward pass suffices.
fn latent_dependencies(model: &Model, state: &State, sources: &[NodeId]) -> Vec<Option<NodeId>> {
    let graph = model.graph();
    let mut depends: Vec<Option<NodeId>> = vec![None; graph.len()];
    for &id in sources {
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
                graph.describe(id),
                graph.describe(latent),
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
    /// [`GraphError::Transform`] if a latent's support has no transform (NaN bounds, a discrete support) or
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
        let latents = model.continuous_latents();
        Self::build(model, state, latents.clone(), space, &latents)
    }

    /// The parameterisation of some of `model`'s continuous latents, `latents`, in `space` (laid out in the order
    /// given), with every other latent held at its value in the state: for samplers that update these latents by
    /// gradients and the others by other steps (see `Nuts::block_sampler` in `keanu-algorithms`). The log
    /// probability and its gradient are the joint ones, with respect to these latents only.
    ///
    /// The other latents can change between uses (other steps update them), so in [`Space::Unconstrained`] a
    /// support bound that depends on any latent, continuous or discrete, inside the block or not, is rejected as
    /// [`Parameterisation::new`] rejects one that depends on a continuous latent.
    ///
    /// # Errors
    ///
    /// [`GraphError::InvalidValue`] if `latents` is empty, repeats a latent, or holds a node that is not a
    /// continuous latent of the model; otherwise as [`Parameterisation::with_space`].
    pub fn for_latents(
        model: &'m Model,
        state: &State,
        latents: &[NodeId],
        space: Space,
    ) -> Result<Self> {
        if latents.is_empty() {
            return Err(GraphError::InvalidValue(
                "a parameterisation needs at least one latent".into(),
            ));
        }
        let continuous = model.continuous_latents();
        let graph = model.graph();
        for (i, id) in latents.iter().enumerate() {
            if !continuous.contains(id) {
                return Err(GraphError::InvalidValue(format!(
                    "{} is not a continuous latent of the model",
                    graph.describe(*id)
                )));
            }
            if latents[..i].contains(id) {
                return Err(GraphError::InvalidValue(format!(
                    "{} is given twice",
                    graph.describe(*id)
                )));
            }
        }
        Self::build(model, state, latents.to_vec(), space, model.latents())
    }

    /// The parameterisation of `latents`, rejecting supports that move with any of `sources`.
    fn build(
        model: &'m Model,
        state: &State,
        latents: Vec<NodeId>,
        space: Space,
        sources: &[NodeId],
    ) -> Result<Self> {
        let graph = model.graph();
        let depends = match space {
            Space::Unconstrained => latent_dependencies(model, state, sources),
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

    /// The continuous latents, in layout order (the order of [`Model::continuous_latents`], or the order given to
    /// [`for_latents`](Self::for_latents)).
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

    /// [`to_state`](Self::to_state), returning false instead of failing where the cascade fails at an impossible
    /// point ([`GraphError::is_impossible_point`], such as a failed ODE solve). The state's deterministic values are
    /// then incomplete until the latents are set again.
    pub fn try_to_state(&self, z: &[f64], state: &mut State) -> Result<bool> {
        match self.to_state(z, state) {
            Ok(()) => Ok(true),
            Err(e) if e.is_impossible_point() => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Sets the latents to their values at `z` (as [`to_state`](Self::to_state)) and returns the joint log
    /// probability there: with `jacobian`, plus [`log_det_jacobian`](Self::log_det_jacobian), as
    /// [`log_prob_and_gradient`](Self::log_prob_and_gradient). −∞ where the model is impossible, including where
    /// evaluating it fails at an impossible point (as [`try_to_state`](Self::try_to_state)).
    pub fn log_prob(&self, z: &[f64], jacobian: bool, state: &mut State) -> Result<f64> {
        if !self.try_to_state(z, state)? {
            return Ok(f64::NEG_INFINITY);
        }
        let log_prob = self.model.log_prob(state)?;
        if jacobian && log_prob.is_finite() {
            Ok(log_prob + self.log_det_jacobian(z)?)
        } else {
            Ok(log_prob)
        }
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
    /// A point where evaluating the model fails at an impossible point ([`GraphError::is_impossible_point`], such as
    /// a failed ODE solve) gives (−∞, None), and the state's deterministic values are then incomplete until the
    /// latents are set again.
    ///
    /// Fails if `z` has the wrong length or the model cannot be evaluated.
    pub fn log_prob_and_gradient(
        &self,
        z: &[f64],
        jacobian: bool,
        state: &mut State,
    ) -> Result<(f64, Option<Vec<f64>>)> {
        if !self.try_to_state(z, state)? {
            return Ok((f64::NEG_INFINITY, None));
        }
        let (log_prob, gradient) = self.model.log_prob_and_gradient_of(state, &self.latents)?;
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

    /// Sets the latents to their values at `z` (as [`to_state`](Self::to_state)) and returns the exact product of
    /// the Hessian of the log probability at `z` (the target of
    /// [`log_prob_and_gradient`](Self::log_prob_and_gradient), with or without the Jacobian) with the direction `v`:
    /// the derivative of the gradient along `v`. See the [module documentation](self#second-order).
    ///
    /// At a point whose log probability is impossible (−∞) the result is meaningless (typically NaN).
    ///
    /// # Errors
    ///
    /// [`GraphError::InvalidValue`] if `z` or `v` has the wrong length, [`GraphError::NoSecondOrderRule`] naming the
    /// node if the model has a distribution or op on the path from a latent to the log probability without a
    /// second-order rule (a lambda, a custom distribution, or a truncated or censored distribution's shape parameter
    /// whose CDF has no closed-form second derivative; see the [module documentation](self#second-order)), and
    /// any error evaluating the model.
    pub fn hvp(&self, z: &[f64], v: &[f64], jacobian: bool, state: &mut State) -> Result<Vec<f64>> {
        self.check_len(z)?;
        if v.len() != self.dimension {
            return Err(GraphError::InvalidValue(format!(
                "a direction has {} coordinates, expected the parameterisation's {}",
                v.len(),
                self.dimension
            )));
        }
        let mut products = self.hessian_products(z, v, 1, jacobian, state)?;
        Ok(products.pop().unwrap_or_default())
    }

    /// Sets the latents to their values at `z` and returns the exact Hessian of the log probability at `z` (with or
    /// without the Jacobian, as [`hvp`](Self::hvp)), shaped `[n, n]` for the parameterisation's
    /// [`dimension`](Self::dimension) n. It is symmetric up to rounding.
    ///
    /// The n Hessian-vector products along the coordinate axes are computed together, as one forward and one
    /// reverse sweep whose tangents carry all n directions on their batch axis, rather than as n separate sweeps.
    ///
    /// # Errors
    ///
    /// As [`hvp`](Self::hvp).
    pub fn hessian(&self, z: &[f64], jacobian: bool, state: &mut State) -> Result<DoubleTensor> {
        self.check_len(z)?;
        let n = self.dimension;
        let mut identity = vec![0.0; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        let columns = self.hessian_products(z, &identity, n, jacobian, state)?;
        // column k is H eₖ
        let mut data = vec![0.0; n * n];
        for (k, column) in columns.iter().enumerate() {
            for (i, &h) in column.iter().enumerate() {
                data[i * n + k] = h;
            }
        }
        Ok(DoubleTensor::new(&[n, n], data)?)
    }

    /// The Hessian-vector products along `b` directions (row-major `[b, dimension]` in `directions`), after setting
    /// the state to `z`. By the chain rule through x = T(y) for each latent, with ∇ₓ and Hₓ the gradient and
    /// Hessian of the model's log probability in the constrained values:
    /// `H v = J_Tᵀ Hₓ (J_T v) + (∂/∂v J_Tᵀ) ∇ₓ + H_log|J| v`.
    fn hessian_products(
        &self,
        z: &[f64],
        directions: &[f64],
        b: usize,
        jacobian: bool,
        state: &mut State,
    ) -> Result<Vec<Vec<f64>>> {
        self.to_state(z, state)?;
        let n = self.dimension;
        let graph = self.model.graph();
        let direction =
            |k: usize, slot: &Slot| &directions[k * n + slot.range.start..k * n + slot.range.end];
        let mut tangents = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            let y = &z[slot.range.clone()];
            let shape = graph[slot.id].shape();
            let mut data = Vec::with_capacity(b * shape.iter().product::<usize>());
            for k in 0..b {
                let tangent = slot
                    .bijector
                    .forward_jvp(y, direction(k, slot))
                    .map_err(|e| in_latent(graph, slot.id, e))?;
                data.extend_from_slice(tangent.as_slice());
            }
            let batched: Vec<usize> = std::iter::once(b).chain(shape.iter().copied()).collect();
            tangents.push((slot.id, DoubleTensor::new(&batched, data)?));
        }
        let (gradient, hvps) = self.model.log_prob_gradient_and_hvp(state, tangents, b)?;
        let mut products = vec![vec![0.0; n]; b];
        let add = |row: &mut [f64], terms: Vec<f64>| {
            for (r, t) in row.iter_mut().zip(terms) {
                *r += t;
            }
        };
        for slot in &self.slots {
            let y = &z[slot.range.clone()];
            let shape = graph[slot.id].shape();
            let len: usize = shape.iter().product();
            let g = gradient.get(slot.id);
            let h = hvps.get(slot.id.index()).and_then(Option::as_ref);
            for (k, product) in products.iter_mut().enumerate() {
                let t = direction(k, slot);
                let row = &mut product[slot.range.clone()];
                let in_slot = |e| in_latent(graph, slot.id, e);
                if let Some(h) = h {
                    let hk =
                        DoubleTensor::new(shape, h.as_slice()[k * len..(k + 1) * len].to_vec())?;
                    add(row, slot.bijector.forward_vjp(y, &hk).map_err(in_slot)?);
                }
                if self.space == Space::Unconstrained {
                    if let Some(g) = g {
                        add(
                            row,
                            slot.bijector.forward_vjp_jvp(y, t, g).map_err(in_slot)?,
                        );
                    }
                    if jacobian {
                        add(
                            row,
                            slot.bijector.log_det_jacobian_hvp(y, t).map_err(in_slot)?,
                        );
                    }
                }
            }
        }
        Ok(products)
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
            GraphError::Transform(format!("latent {}: {message}", graph.describe(id)))
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GraphBuilder;
    use crate::transform::{
        Circular, CorrelationCholesky, Interval, LowerBound, Ordered, Simplex, UpperBound,
    };

    /// Replaces each latent's bijector, in layout order, recomputing the layout. The bijectors need not match the
    /// latents' supports: the log probability is just a function of the constrained values, and this exercises
    /// the second-order chain rule through every bijector with distributions that have a rule.
    fn with_bijectors(param: &mut Parameterisation<'_>, bijectors: Vec<Box<dyn Bijector>>) {
        let mut offset = 0;
        for (slot, bijector) in param.slots.iter_mut().zip(bijectors) {
            let len = bijector.unconstrained_len();
            slot.range = offset..offset + len;
            slot.bijector = bijector;
            offset += len;
        }
        param.dimension = offset;
    }

    #[test]
    fn the_hessian_through_every_bijector_matches_finite_differences() {
        let b = GraphBuilder::new();
        let eye = || DoubleTensor::eye(3).unwrap();
        let lower = b.gaussian(0.5, 1.0);
        let upper = b.gaussian(0.2, 1.5);
        let interval = b.gaussian_with_shape(&[2], 0.4, 1.0);
        let simplex =
            b.multivariate_gaussian(DoubleTensor::vector(vec![0.2, 0.3, 0.5]), eye() * 0.2);
        let ordered = b.multivariate_gaussian(DoubleTensor::vector(vec![-1.0, 0.0, 1.0]), eye());
        let cholesky = b.gaussian_with_shape(&[3, 3], 0.1, 1.0);
        let angle = b.gaussian(0.3, 2.0);
        let mean = lower * upper
            + interval.sum()
            + (simplex * ordered).sum()
            + cholesky.sum()
            + angle.sin();
        let y = b.gaussian(mean, 1.5);
        b.observe(y, DoubleTensor::scalar(0.4));
        let (graph, mut state) = b.build().unwrap();
        let all: Vec<NodeId> = graph.ids().collect();
        state
            .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(1))
            .unwrap();
        let model = Model::new(graph, &state).unwrap();
        let mut param = Parameterisation::new(&model, &state).unwrap();
        with_bijectors(
            &mut param,
            vec![
                Box::new(LowerBound::new(&DoubleTensor::scalar(0.0), &[]).unwrap()),
                Box::new(UpperBound::new(&DoubleTensor::scalar(1.0), &[]).unwrap()),
                Box::new(
                    Interval::new(
                        &DoubleTensor::scalar(-1.0),
                        &DoubleTensor::scalar(2.0),
                        &[2],
                    )
                    .unwrap(),
                ),
                Box::new(Simplex::new(&[3]).unwrap()),
                Box::new(Ordered::new(&[3]).unwrap()),
                Box::new(CorrelationCholesky::new(&[3, 3]).unwrap()),
                Box::new(Circular::new(&[])),
            ],
        );
        let n = param.dimension();
        assert_eq!(n, 1 + 1 + 2 + 2 + 3 + 3 + 2);
        let z: Vec<f64> = (0..n)
            .map(|i| 0.8 * ((i as f64) * 1.7 + 0.3).sin())
            .collect();
        for jacobian in [false, true] {
            let hessian = param.hessian(&z, jacobian, &mut state).unwrap();
            let h = hessian.as_slice();
            for j in 0..n {
                let step = 1e-5;
                let mut plus = z.clone();
                plus[j] += step;
                let mut minus = z.clone();
                minus[j] -= step;
                let gp = param
                    .log_prob_and_gradient(&plus, jacobian, &mut state)
                    .unwrap()
                    .1
                    .unwrap();
                let gm = param
                    .log_prob_and_gradient(&minus, jacobian, &mut state)
                    .unwrap()
                    .1
                    .unwrap();
                for i in 0..n {
                    let fd = (gp[i] - gm[i]) / (2.0 * step);
                    assert!(
                        (h[i * n + j] - fd).abs() <= 1e-6 * fd.abs().max(1.0),
                        "jacobian {jacobian}: H[{i}][{j}] = {} vs {fd}",
                        h[i * n + j]
                    );
                }
            }
        }
    }
}
