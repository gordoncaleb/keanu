//! The ODE op: the solution of dy/dt = f(t, y, θ) at given output times, where f is a nested, deterministic graph
//! (the body), integrated by the solvers of the [`ode`](crate::ode) module.
//!
//! The op's value is solved without sensitivities. Its derivatives integrate the forward sensitivities alongside the
//! state, with the body's Jacobian-vector products giving J S + ∂f/∂θ in one batched forward sweep per evaluation of
//! the right-hand side: forward mode along the parents' tangents (one sensitivity column per direction), and
//! reverse mode with one column per element of each parent that is not a constant, contracted with the cotangent at
//! the output times. Gradients therefore cost one extra solve of n·(1 + p) equations, for p the number of
//! parameter elements differentiated. There is no second-order rule.

use super::{Op, eval};
use crate::autodiff::forward_tangents;
use crate::error::{GraphError, Result};
use crate::graph::{Graph, NodeId};
use crate::ode::{self, OdeError, OdeOptions, OdeSystem, SensitivitySystem};
use crate::state::State;
use crate::value::{DType, Value};
use keanu_tensor::{DoubleTensor, Shape};
use std::fmt;

/// The body of an [`Op::Ode`]: a deterministic [`Graph`] computing dy/dt = f(t, y, θ), with input placeholders
/// and one output, and the solver options.
///
/// The placeholders are unconnected f64 proxy nodes of the body graph:
///
/// - the **time** t, a scalar;
/// - the **state** y, a vector of n elements;
/// - one **parameter** θⱼ for each parameter parent of the ODE node, of that parent's shape.
///
/// The **output** is dy/dt, an f64 vector of n elements. An ODE node's parents are the initial state y₀ (`[n]`), the
/// output times (`[T]`, finite, strictly increasing and after t₀ whenever the node is evaluated), the initial time
/// t₀ (a scalar) and the parameters, in that order, and its value is the state at each output time, shape `[T, n]`.
///
/// Bodies contain no probabilistic nodes or factors, and they can't refer to nodes outside the body: values from the
/// enclosing graph enter through the parameters. Build one with [`GraphBuilder::ode`](crate::GraphBuilder::ode),
/// or from a body graph with [`OdeBody::new`].
#[derive(Clone)]
pub struct OdeBody {
    graph: Graph,
    time: NodeId,
    state: NodeId,
    parameters: Vec<NodeId>,
    output: NodeId,
    options: OdeOptions,
    /// The body nodes that depend on the time or the state, recomputed at every evaluation of f; the others depend
    /// on the parameters only and are computed once per solve.
    dynamic: Vec<bool>,
}

fn invalid(message: impl Into<String>) -> GraphError {
    GraphError::InvalidNode {
        op: "ode",
        message: message.into(),
    }
}

/// The ODE node's parents, by role.
struct Parents<'a> {
    y0: &'a DoubleTensor,
    times: &'a DoubleTensor,
    t0: f64,
    parameters: &'a [&'a Value],
}

impl OdeBody {
    /// An ODE body over a body graph, given its placeholders, output and solver options (see [`OdeBody`]).
    ///
    /// Fails if a placeholder is not an unconnected f64 proxy or is listed twice, if the time is not a scalar or
    /// the state not a vector, if the body has a probabilistic node, a factor or another unconnected proxy, if the
    /// output is not an f64 vector of the state's shape, or if the options are invalid
    /// ([`OdeOptions::validate`]).
    pub fn new(
        graph: Graph,
        time: NodeId,
        state: NodeId,
        parameters: Vec<NodeId>,
        output: NodeId,
        options: OdeOptions,
    ) -> Result<Self> {
        options.validate().map_err(invalid)?;
        let inputs: Vec<NodeId> = [time, state]
            .into_iter()
            .chain(parameters.iter().copied())
            .collect();
        graph.check_ids(inputs.iter().chain([&output]))?;
        let mut is_input = vec![false; graph.len()];
        for &id in &inputs {
            let node = &graph[id];
            if !matches!(node.op, Op::Proxy) || !node.parents.is_empty() || node.dtype != DType::F64
            {
                return Err(invalid(format!(
                    "the body's input {id:?} must be an unconnected f64 proxy"
                )));
            }
            if std::mem::replace(&mut is_input[id.0], true) {
                return Err(invalid(format!("the body's input {id:?} is listed twice")));
            }
        }
        if !graph[time].shape().is_empty() {
            return Err(invalid(format!(
                "the body's time must be a scalar, got shape {:?}",
                graph[time].shape()
            )));
        }
        if graph[state].shape().len() != 1 {
            return Err(invalid(format!(
                "the body's state must be a vector, got shape {:?}",
                graph[state].shape()
            )));
        }
        for id in graph.ids() {
            let node = &graph[id];
            if node.is_probabilistic() {
                return Err(invalid(format!(
                    "the body's node {} is probabilistic: ODE bodies must be deterministic (draw random \
                     variables outside the ODE and pass them in as parameters)",
                    graph.describe(id)
                )));
            }
            if node.is_factor() {
                return Err(invalid(format!(
                    "the body's node {} is a factor: ODE bodies must be deterministic",
                    graph.describe(id)
                )));
            }
            if matches!(node.op, Op::Proxy) && node.parents.is_empty() && !is_input[id.0] {
                return Err(invalid(format!(
                    "the body's proxy {} is neither connected nor an input",
                    graph.describe(id)
                )));
            }
        }
        let (out, state_node) = (&graph[output], &graph[state]);
        if out.dtype != DType::F64 || out.shape != state_node.shape {
            return Err(invalid(format!(
                "the body's output (dy/dt) is {:?} of shape {:?}, but the state is f64 of shape {:?}",
                out.dtype,
                out.shape(),
                state_node.shape()
            )));
        }
        // ids are a topological order, so a node is dynamic if it is t or y, or any parent is
        let mut dynamic = vec![false; graph.len()];
        for id in graph.ids() {
            dynamic[id.0] =
                id == time || id == state || graph[id].parents.iter().any(|p| dynamic[p.0]);
        }
        Ok(Self {
            graph,
            time,
            state,
            parameters,
            output,
            options,
            dynamic,
        })
    }

    /// The body graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The time placeholder (a scalar).
    pub fn time(&self) -> NodeId {
        self.time
    }

    /// The state placeholder (a vector of n elements).
    pub fn state(&self) -> NodeId {
        self.state
    }

    /// The parameter placeholders, one per parameter parent.
    pub fn parameters(&self) -> &[NodeId] {
        &self.parameters
    }

    /// The output: dy/dt.
    pub fn output(&self) -> NodeId {
        self.output
    }

    /// The solver options.
    pub fn options(&self) -> &OdeOptions {
        &self.options
    }

    /// The state's dimension, n.
    pub fn dimension(&self) -> usize {
        self.graph[self.state].shape()[0]
    }

    /// The number of parents an ODE node with this body takes: the initial state, the output times, the initial
    /// time and the parameters.
    pub(crate) fn parent_count(&self) -> usize {
        3 + self.parameters.len()
    }

    /// The ODE node's type and shape given its parents': f64 `[T, n]`.
    pub(crate) fn output_type(&self, parents: &[(DType, &[usize])]) -> Result<(DType, Shape)> {
        if parents.len() != self.parent_count() {
            return Err(invalid(format!(
                "needs {} parents (the initial state, the output times, the initial time and {} parameters), \
                 got {}",
                self.parent_count(),
                self.parameters.len(),
                parents.len()
            )));
        }
        if parents.iter().any(|(t, _)| *t != DType::F64) {
            return Err(invalid(
                "the initial state, output times, initial time and parameters must be f64",
            ));
        }
        let n = self.dimension();
        if parents[0].1 != [n] {
            return Err(invalid(format!(
                "the initial state has shape {:?}, but the body's state has shape [{n}]",
                parents[0].1
            )));
        }
        let [length] = parents[1].1 else {
            return Err(invalid(format!(
                "the output times must be a vector, got shape {:?}",
                parents[1].1
            )));
        };
        if !parents[2].1.is_empty() {
            return Err(invalid(format!(
                "the initial time must be a scalar, got shape {:?}",
                parents[2].1
            )));
        }
        for (i, &p) in self.parameters.iter().enumerate() {
            let shape = parents[3 + i].1;
            if shape != self.graph[p].shape() {
                return Err(invalid(format!(
                    "parameter {i} has shape {shape:?}, but the body's parameter has shape {:?}",
                    self.graph[p].shape()
                )));
            }
        }
        let shape = Shape::from_slice(&[*length, n]);
        super::check_element_count(&shape).map_err(invalid)?;
        Ok((DType::F64, shape))
    }

    fn split<'a>(&self, parents: &'a [&'a Value]) -> Result<Parents<'a>> {
        if parents.len() != self.parent_count() {
            return Err(invalid(format!(
                "expected {} parents, got {}",
                self.parent_count(),
                parents.len()
            )));
        }
        let f64_of = |v: &'a Value| {
            v.as_f64().ok_or_else(|| {
                invalid("the initial state, output times, initial time and parameters must be f64")
            })
        };
        let t0 = f64_of(parents[2])?;
        if t0.len() != 1 {
            return Err(invalid("the initial time must be a scalar"));
        }
        Ok(Parents {
            y0: f64_of(parents[0])?,
            times: f64_of(parents[1])?,
            t0: t0.as_slice()[0],
            parameters: &parents[3..],
        })
    }

    /// Evaluates the ODE node: the state at each output time, `[T, n]`.
    ///
    /// Fails with [`GraphError::SolverFailed`] when the integration can't be completed at these values (which
    /// models treat as an impossible point), and with [`GraphError::InvalidNode`] if the output times are not
    /// finite, strictly increasing and after the initial time.
    pub(crate) fn eval(&self, parents: &[&Value]) -> Result<Value> {
        let p = self.split(parents)?;
        let mut system = Rhs::new(self, p.parameters, Vec::new(), 0)?;
        let solution = ode::solve(
            &mut system,
            p.t0,
            p.y0.as_slice(),
            p.times.as_slice(),
            &self.options,
        )
        .map_err(from_ode)?;
        let data: Vec<f64> = solution.states.into_iter().flatten().collect();
        Ok(Value::F64(DoubleTensor::new(
            &[p.times.len(), self.dimension()],
            data,
        )?))
    }

    /// The sensitivities ∂y(tᵢ)/∂ along `p` directions in the space of (y₀, t₀, θ): `y0` (n × p, row-major) and
    /// `t0` (p) give each direction's initial state and time components and `theta` the batched parameter tangents
    /// (`[p, θⱼ shape...]`). Returns S(tᵢ) (n × p) at each output time.
    fn sensitivities(
        &self,
        parents: &Parents<'_>,
        y0: &[f64],
        t0: &[f64],
        theta: Vec<(NodeId, DoubleTensor)>,
        p: usize,
    ) -> Result<Vec<Vec<f64>>> {
        let n = self.dimension();
        let mut system = Rhs::new(self, parents.parameters, theta, p)?;
        let mut s0 = y0.to_vec();
        if t0.iter().any(|&d| d != 0.0) {
            // y(t; t₀) with y(t₀) = y₀: ∂y/∂t₀ starts at −f(t₀, y₀)
            let mut f0 = vec![0.0; n];
            system.rhs(parents.t0, parents.y0.as_slice(), &mut f0)?;
            for i in 0..n {
                for k in 0..p {
                    s0[i * p + k] -= f0[i] * t0[k];
                }
            }
        }
        let solution = ode::solve_sensitivities(
            &mut system,
            parents.t0,
            parents.y0.as_slice(),
            &s0,
            parents.times.as_slice(),
            &self.options,
        )
        .map_err(from_ode)?;
        Ok(solution.sensitivities)
    }

    /// f(tᵢ, y(tᵢ)) at each output time, from the node's value: ∂y(tᵢ)/∂tᵢ.
    fn rates(&self, parents: &Parents<'_>, out: &DoubleTensor) -> Result<Vec<f64>> {
        let n = self.dimension();
        let mut system = Rhs::new(self, parents.parameters, Vec::new(), 0)?;
        let mut rates = vec![0.0; out.len()];
        for (i, &t) in parents.times.as_slice().iter().enumerate() {
            let range = i * n..(i + 1) * n;
            system.rhs(t, &out.as_slice()[range.clone()], &mut rates[range])?;
        }
        Ok(rates)
    }

    /// Forward mode: the tangent of the solution `[b, T, n]` from the parents' tangents (`[b, parent shape...]`).
    pub(crate) fn jvp(
        &self,
        parents: &[&Value],
        out: &DoubleTensor,
        t: &[Option<&DoubleTensor>],
        b: usize,
    ) -> Result<Option<DoubleTensor>> {
        if t.iter().all(Option::is_none) {
            return Ok(None);
        }
        let p = self.split(parents)?;
        let (n, length) = (self.dimension(), p.times.len());
        let mut result = vec![0.0; b * length * n];
        let integrated = t[0].is_some() || t[2].is_some() || t[3..].iter().any(Option::is_some);
        if integrated {
            // direction k is column k of the sensitivities
            let mut y0 = vec![0.0; n * b];
            if let Some(ty) = t[0] {
                for k in 0..b {
                    for i in 0..n {
                        y0[i * b + k] = ty.as_slice()[k * n + i];
                    }
                }
            }
            let t0 = t[2].map_or_else(|| vec![0.0; b], |tt| tt.as_slice().to_vec());
            let theta = self
                .parameters
                .iter()
                .zip(&t[3..])
                .filter_map(|(&id, tangent)| tangent.map(|tp| (id, tp.clone())))
                .collect();
            let sensitivities = self.sensitivities(&p, &y0, &t0, theta, b)?;
            for (i, s) in sensitivities.iter().enumerate() {
                for k in 0..b {
                    for r in 0..n {
                        result[(k * length + i) * n + r] = s[r * b + k];
                    }
                }
            }
        }
        if let Some(tt) = t[1] {
            let rates = self.rates(&p, out)?;
            for k in 0..b {
                for i in 0..length {
                    let dt = tt.as_slice()[k * length + i];
                    for r in 0..n {
                        result[(k * length + i) * n + r] += rates[i * n + r] * dt;
                    }
                }
            }
        }
        Ok(Some(DoubleTensor::new(&[b, length, n], result)?))
    }

    /// Reverse mode: the parents' cotangents from the solution's cotangent `g` (`[b, T, n]`), for the parents marked
    /// in `needed` (the others get none). One sensitivity column is integrated per element of the needed initial
    /// state, initial time and parameters.
    pub(crate) fn vjp(
        &self,
        parents: &[&Value],
        out: &DoubleTensor,
        g: &DoubleTensor,
        b: usize,
        needed: &[bool],
    ) -> Result<Vec<Option<DoubleTensor>>> {
        let p = self.split(parents)?;
        let (n, length) = (self.dimension(), p.times.len());
        let mut result: Vec<Option<DoubleTensor>> = vec![None; parents.len()];
        let g = g.as_slice();
        // the columns: the initial state's elements, the initial time, then each parameter's elements
        let mut offsets = Vec::with_capacity(parents.len());
        let mut columns = 0;
        for (j, value) in parents.iter().enumerate() {
            offsets.push(columns);
            if needed[j] && j != 1 {
                columns += value.shape().iter().product::<usize>();
            }
        }
        if columns > 0 {
            let mut y0 = vec![0.0; n * columns];
            if needed[0] {
                for i in 0..n {
                    y0[i * columns + offsets[0] + i] = 1.0;
                }
            }
            let mut t0 = vec![0.0; columns];
            if needed[2] {
                t0[offsets[2]] = 1.0;
            }
            let mut theta = Vec::new();
            for (j, &id) in self.parameters.iter().enumerate() {
                if !needed[3 + j] {
                    continue;
                }
                let shape = self.graph[id].shape();
                let size: usize = shape.iter().product();
                let mut seed = vec![0.0; columns * size];
                for m in 0..size {
                    seed[(offsets[3 + j] + m) * size + m] = 1.0;
                }
                let batched = crate::autodiff::batched_shape(columns, shape);
                theta.push((id, DoubleTensor::new(&batched, seed)?));
            }
            let sensitivities = self.sensitivities(&p, &y0, &t0, theta, columns)?;
            // the cotangent of column c: Σᵢ Σᵣ g[k, i, r] S(tᵢ)[r, c]
            let mut contracted = vec![0.0; b * columns];
            for (i, s) in sensitivities.iter().enumerate() {
                for k in 0..b {
                    let gi = &g[(k * length + i) * n..(k * length + i + 1) * n];
                    let row = &mut contracted[k * columns..(k + 1) * columns];
                    for (r, &gr) in gi.iter().enumerate() {
                        if gr != 0.0 {
                            for (c, x) in row.iter_mut().enumerate() {
                                *x += gr * s[r * columns + c];
                            }
                        }
                    }
                }
            }
            for (j, value) in parents.iter().enumerate() {
                if !needed[j] || j == 1 {
                    continue;
                }
                let size: usize = value.shape().iter().product();
                let mut data = Vec::with_capacity(b * size);
                for k in 0..b {
                    let start = k * columns + offsets[j];
                    data.extend_from_slice(&contracted[start..start + size]);
                }
                let shape = crate::autodiff::batched_shape(b, value.shape());
                result[j] = Some(DoubleTensor::new(&shape, data)?);
            }
        }
        if needed[1] {
            // ∂y(tᵢ)/∂tᵢ = f(tᵢ, y(tᵢ))
            let rates = self.rates(&p, out)?;
            let mut data = vec![0.0; b * length];
            for k in 0..b {
                for i in 0..length {
                    let gi = &g[(k * length + i) * n..(k * length + i + 1) * n];
                    data[k * length + i] = gi
                        .iter()
                        .zip(&rates[i * n..(i + 1) * n])
                        .map(|(a, f)| a * f)
                        .sum();
                }
            }
            result[1] = Some(DoubleTensor::new(&[b, length], data)?);
        }
        Ok(result)
    }
}

/// The error of an ODE solve as a graph error: a failed integration is [`GraphError::SolverFailed`], a malformed
/// problem an invalid node, and the body's own errors are passed on.
fn from_ode(e: OdeError<GraphError>) -> GraphError {
    match e {
        OdeError::InvalidProblem(message) => invalid(message),
        OdeError::SolverFailed(failed) => GraphError::SolverFailed(failed),
        OdeError::System(e) => e,
    }
}

/// The body as an [`OdeSystem`] and [`SensitivitySystem`] for given parameter values, with `p` sensitivity
/// columns whose parameter components are `theta`.
struct Rhs<'a> {
    body: &'a OdeBody,
    state: State,
    theta: Vec<(NodeId, DoubleTensor)>,
    p: usize,
}

impl<'a> Rhs<'a> {
    /// The body's state with the parameters set and every node that depends on the parameters alone computed.
    fn new(
        body: &'a OdeBody,
        parameters: &[&Value],
        theta: Vec<(NodeId, DoubleTensor)>,
        p: usize,
    ) -> Result<Self> {
        let graph = &body.graph;
        let mut values: Vec<Option<Value>> = vec![None; graph.len()];
        for (&id, &value) in body.parameters.iter().zip(parameters) {
            values[id.0] = Some(value.clone());
        }
        values[body.time.0] = Some(Value::F64(DoubleTensor::scalar(0.0)));
        values[body.state.0] = Some(Value::F64(DoubleTensor::zeros(&[body.dimension()])));
        let mut state = State {
            values,
            observed: vec![false; graph.len()],
        };
        for id in graph.ids() {
            if state.values[id.0].is_none() && !body.dynamic[id.0] {
                compute(graph, &mut state, id)?;
            }
        }
        Ok(Self {
            body,
            state,
            theta,
            p,
        })
    }

    /// Sets the time and state and recomputes the nodes that depend on them.
    fn at(&mut self, t: f64, y: &[f64]) -> Result<()> {
        let body = self.body;
        let n = body.dimension();
        self.state.values[body.time.0] = Some(Value::F64(DoubleTensor::scalar(t)));
        self.state.values[body.state.0] = Some(Value::F64(DoubleTensor::new(&[n], y.to_vec())?));
        for id in body.graph.ids() {
            if body.dynamic[id.0] && id != body.time && id != body.state {
                compute(&body.graph, &mut self.state, id)?;
            }
        }
        Ok(())
    }

    fn output(&self) -> Result<&[f64]> {
        self.state.values[self.body.output.0]
            .as_ref()
            .and_then(Value::as_f64)
            .map(DoubleTensor::as_slice)
            .ok_or_else(|| invalid("the body's output has no f64 value"))
    }

    /// The tangent of the output for batched seeds, `[b, n]` row-major, zero where the output doesn't depend on them.
    fn output_tangent(
        &self,
        seeds: Vec<(NodeId, DoubleTensor)>,
        b: usize,
        out: &mut Vec<f64>,
    ) -> Result<()> {
        let mut tangents = forward_tangents(&self.body.graph, &self.state, seeds, b)?;
        out.clear();
        match tangents[self.body.output.0].take() {
            Some(t) => out.extend_from_slice(t.as_slice()),
            None => out.resize(b * self.body.dimension(), 0.0),
        }
        Ok(())
    }
}

/// Computes one deterministic node of a body from its parents' values.
fn compute(graph: &Graph, state: &mut State, id: NodeId) -> Result<()> {
    let node = &graph[id];
    let value = match &node.op {
        Op::Constant(v) => v.clone(),
        op => {
            let inputs = node
                .parents
                .iter()
                .map(|p| {
                    state.values[p.0]
                        .as_ref()
                        .ok_or(GraphError::MissingValue(*p))
                })
                .collect::<Result<smallvec::SmallVec<[&Value; 4]>>>()?;
            match op {
                Op::Proxy => inputs[0].clone(),
                op => eval(op, &inputs)?,
            }
        }
    };
    state.values[id.0] = Some(value);
    Ok(())
}

impl OdeSystem for Rhs<'_> {
    type Error = GraphError;

    fn rhs(&mut self, t: f64, y: &[f64], dydt: &mut [f64]) -> Result<()> {
        self.at(t, y)?;
        dydt.copy_from_slice(self.output()?);
        Ok(())
    }

    fn jacobian(&mut self, t: f64, y: &[f64], dfdy: &mut [f64], dfdt: &mut [f64]) -> Result<()> {
        self.at(t, y)?;
        let n = self.body.dimension();
        // direction j < n is ∂/∂yⱼ, direction n is ∂/∂t
        let mut identity = vec![0.0; (n + 1) * n];
        for j in 0..n {
            identity[j * n + j] = 1.0;
        }
        let mut time = vec![0.0; n + 1];
        time[n] = 1.0;
        let seeds = vec![
            (self.body.state, DoubleTensor::new(&[n + 1, n], identity)?),
            (self.body.time, DoubleTensor::new(&[n + 1], time)?),
        ];
        let mut tangent = Vec::new();
        self.output_tangent(seeds, n + 1, &mut tangent)?;
        for i in 0..n {
            for j in 0..n {
                dfdy[i * n + j] = tangent[j * n + i];
            }
            dfdt[i] = tangent[n * n + i];
        }
        Ok(())
    }
}

impl SensitivitySystem for Rhs<'_> {
    fn sensitivity_rhs(&mut self, t: f64, y: &[f64], s: &[f64], ds: &mut [f64]) -> Result<()> {
        let (n, p) = (self.body.dimension(), self.p);
        if p == 0 {
            return Ok(());
        }
        self.at(t, y)?;
        // the seeds of the state are Sᵀ, [p, n]
        let mut st = vec![0.0; p * n];
        for i in 0..n {
            for k in 0..p {
                st[k * n + i] = s[i * p + k];
            }
        }
        let mut seeds = Vec::with_capacity(1 + self.theta.len());
        seeds.push((self.body.state, DoubleTensor::new(&[p, n], st)?));
        seeds.extend(self.theta.iter().cloned());
        let mut tangent = Vec::new();
        self.output_tangent(seeds, p, &mut tangent)?;
        for i in 0..n {
            for k in 0..p {
                ds[i * p + k] = tangent[k * n + i];
            }
        }
        Ok(())
    }
}

impl fmt::Debug for OdeBody {
    /// The placeholders and output, the options, and each body node's operation, parents, type, shape and label.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let nodes: Vec<String> = self
            .graph
            .ids()
            .map(|id| {
                let node = &self.graph[id];
                format!(
                    "{id:?} {:?} parents {:?} {:?}{:?} label {:?}",
                    node.op,
                    node.parents(),
                    node.dtype,
                    node.shape(),
                    node.label()
                )
            })
            .collect();
        f.debug_struct("OdeBody")
            .field("time", &self.time)
            .field("state", &self.state)
            .field("parameters", &self.parameters)
            .field("output", &self.output)
            .field("options", &self.options)
            .field("nodes", &nodes)
            .finish()
    }
}
