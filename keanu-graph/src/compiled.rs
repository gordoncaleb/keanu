//! The joint log probability and its gradient as a compiled program, for gradient-based samplers that evaluate it
//! many times at different values of the same latents.
//!
//! The general path ([`Parameterisation::log_prob_and_gradient`](crate::Parameterisation::log_prob_and_gradient))
//! sets the latents in the state, cascades to their deterministic descendants, then sweeps back through the graph,
//! building and dropping a tensor (or several) at every node every time. A [`CompiledModel`] plans that work once:
//!
//! - nodes that don't depend on the latents are read from the state as they are;
//! - the nodes that do are given registers, one `f64` per element, reused from one evaluation to the next.
//!   Element-wise unary and binary ops, gathers along constant indices and the ops that only move elements
//!   become one instruction per element. Every other op runs through its general rules on tensors made from the
//!   registers;
//! - each Gaussian or location-scale Student-t term is computed from the registers, element by element; every
//!   other probabilistic node's term and gradient parts come from its distribution's general code.
//!
//! The results are exactly the general path's: the same element functions and derivatives, the log probability
//! summed in the same order, and each node's cotangent accumulated in the same order, including the reverse
//! sweep's rules for which nodes are visited and for masked-out nonfinite values.

use crate::autodiff::{
    binary_derivatives_scalar, is_differentiable, masked_out, scalar_vjp, unary_derivative_scalar,
    vjp,
};
use crate::error::{GraphError, Result};
use crate::graph::{Graph, NodeId};
use crate::logprob::{
    LogProbGradient, Weights, scale_parts, sum_to_shape, term_and_parts, weight_of, weighted_term,
};
use crate::model::Model;
use crate::op::eval::{binary_f64, eval, unary_f64};
use crate::op::{BinaryOp, Op, UnaryOp};
use crate::state::State;
use crate::value::{DType, Value};
use keanu_tensor::DoubleTensor;

/// No register.
const NONE: usize = usize::MAX;

/// Where a binary op's operand element comes from for each output element.
#[derive(Clone, Debug)]
enum Operand {
    /// The parent has the output's shape: element k (registers from the offset).
    Same(usize),
    /// The parent is one element, broadcast (its register).
    Scalar(usize),
    /// Element `elements[k]` of the parent (registers from the offset).
    Map(usize, Vec<usize>),
}

impl Operand {
    #[inline]
    fn register(&self, k: usize) -> usize {
        match self {
            Operand::Same(at) => at + k,
            Operand::Scalar(at) => *at,
            Operand::Map(at, elements) => at + elements[k],
        }
    }

    #[inline]
    fn element(&self, k: usize) -> usize {
        match self {
            Operand::Same(_) => k,
            Operand::Scalar(_) => 0,
            Operand::Map(_, elements) => elements[k],
        }
    }
}

/// How a compiled node computes its elements (into the registers from its offset).
#[derive(Clone, Debug)]
enum Kernel {
    /// An element-wise unary op of a parent of the same shape (registers from `x`).
    Unary { op: UnaryOp, x: usize },
    Binary {
        op: BinaryOp,
        l: Operand,
        r: Operand,
    },
    /// Element k is a copy of register `from[k]`, which is element `sources[k].1` of parent `sources[k].0`.
    Copy {
        from: Vec<usize>,
        sources: Vec<(usize, usize)>,
    },
}

/// A probabilistic node's (or factor's) term of the log probability.
#[derive(Clone, Debug)]
enum Term {
    /// By its distribution's general code.
    General(NodeId),
    /// An element-wise family computed from the registers: the value and the parameters (in order), broadcast to
    /// the node's shape.
    Registers {
        node: NodeId,
        family: Family,
        x: Operand,
        parameters: Vec<Operand>,
    },
}

/// The families whose terms are computed from the registers.
#[derive(Clone, Copy, Debug)]
enum Family {
    /// A Gaussian; `single_scale` if sigma is one element (whose logarithm the family takes once).
    Gaussian { single_scale: bool },
    /// A location-scale Student-t.
    StudentT,
}

impl Family {
    /// The number of parameters.
    fn arity(self) -> usize {
        match self {
            Family::Gaussian { .. } => 2,
            Family::StudentT => 3,
        }
    }
}

/// A node that depends on the latents, computed in order.
#[derive(Clone, Debug)]
enum Item {
    /// Compiled into a kernel.
    Compiled { node: NodeId, kernel: usize },
    /// Computed by the op's general rules.
    General { node: NodeId },
}

/// The joint log probability of a model and its gradient with respect to some of its continuous latents, compiled
/// for one state's observed values and the values of the other latents. See the [module documentation](self).
///
/// Build one with [`CompiledModel::new`] and evaluate it with
/// [`Parameterisation::log_prob_and_gradient_compiled`](crate::Parameterisation::log_prob_and_gradient_compiled).
/// It reads every node that doesn't depend on the latents from the state it is given at each evaluation, so the
/// state must keep the values the model was compiled with for its observed nodes (the other latents may change).
/// It keeps its working memory between evaluations: one per thread.
#[derive(Debug)]
pub struct CompiledModel<'m> {
    model: &'m Model,
    latents: Vec<NodeId>,
    /// Each node's first register (NONE if it has none) and number of elements.
    offsets: Vec<usize>,
    lens: Vec<usize>,
    /// Whether each node depends on the latents (the latents included).
    dynamic: Vec<bool>,
    items: Vec<Item>,
    kernels: Vec<Kernel>,
    terms: Vec<Term>,
    /// The nodes that don't depend on the latents but are read by instructions, loaded into their registers at each
    /// evaluation.
    loads: Vec<NodeId>,
    // working memory
    registers: Vec<f64>,
    adjoints: Vec<f64>,
    /// Whether each node has received a cotangent in the current reverse pass.
    touched: Vec<bool>,
    /// The values of the dependent nodes as tensors, made when general code needs them.
    values: Vec<Option<Value>>,
    scratch: Vec<f64>,
    /// Each latent's derivative of its own term, in its registers, and whether it has one.
    direct: Vec<f64>,
    has_direct: Vec<bool>,
    /// Each latent's gradient, in its registers, and whether it has one.
    gradient: Vec<f64>,
    has_gradient: Vec<bool>,
}

impl<'m> CompiledModel<'m> {
    /// Compiles the model's joint log probability as a function of the continuous latents `latents` (the
    /// others held at their values in the state). None if the model has an observed deterministic node (whose
    /// observation must be checked against its computed value), or if a latent is not a continuous latent.
    pub fn new(model: &'m Model, latents: &[NodeId], state: &State) -> Option<Self> {
        let graph = model.graph();
        let n = graph.len();
        if model
            .observed()
            .iter()
            .any(|&id| !graph[id].is_probabilistic())
        {
            return None;
        }
        let mut dynamic = vec![false; n];
        for &id in latents {
            let node = graph.node(id)?;
            if !node.is_probabilistic() || node.dtype != DType::F64 || state.is_observed(id) {
                return None;
            }
            dynamic[id.0] = true;
        }
        // as the cascade: deterministic, unobserved descendants (ids are a topological order); and, as the
        // general path computes a deterministic node without a value when it is needed, those too
        for id in graph.ids() {
            let node = &graph[id];
            if !dynamic[id.0] && !node.is_probabilistic() && !state.is_observed(id) {
                dynamic[id.0] =
                    node.parents.iter().any(|p| dynamic[p.0]) || state.value(id).is_none();
            }
        }
        let mut compiled = Self {
            model,
            latents: latents.to_vec(),
            offsets: vec![NONE; n],
            lens: graph
                .ids()
                .map(|id| graph[id].shape.iter().product())
                .collect(),
            dynamic,
            items: Vec::new(),
            kernels: Vec::new(),
            terms: Vec::new(),
            loads: Vec::new(),
            registers: Vec::new(),
            adjoints: Vec::new(),
            touched: vec![false; n],
            values: vec![None; n],
            scratch: Vec::new(),
            direct: Vec::new(),
            has_direct: vec![false; latents.len()],
            gradient: Vec::new(),
            has_gradient: vec![false; latents.len()],
        };
        let mut registers = 0usize;
        let mut give = |offsets: &mut Vec<usize>, id: NodeId, len: usize| {
            if offsets[id.0] == NONE {
                offsets[id.0] = registers;
                registers += len;
            }
        };
        for id in graph.ids() {
            if compiled.dynamic[id.0] && graph[id].dtype == DType::F64 {
                give(&mut compiled.offsets, id, compiled.lens[id.0]);
            }
        }
        for id in graph.ids() {
            let node = &graph[id];
            if !compiled.dynamic[id.0] || node.is_probabilistic() {
                continue;
            }
            match compiled.compile_node(graph, id) {
                Some(kernel) => {
                    // the parents read by instructions need registers
                    for &p in &node.parents {
                        if graph[p].dtype == DType::F64 && !compiled.dynamic[p.0] {
                            if compiled.offsets[p.0] == NONE {
                                compiled.loads.push(p);
                            }
                            give(&mut compiled.offsets, p, compiled.lens[p.0]);
                        }
                    }
                    compiled.kernels.push(kernel);
                    compiled.items.push(Item::Compiled {
                        node: id,
                        kernel: compiled.kernels.len() - 1,
                    });
                }
                None => compiled.items.push(Item::General { node: id }),
            }
        }
        for &node in model.probabilistic_joint() {
            let term = compiled.compile_term(graph, node);
            if let Term::Registers { .. } = term {
                // the value and parameters are read from registers
                for p in std::iter::once(node).chain(graph[node].parents.iter().copied()) {
                    if !compiled.dynamic[p.0] {
                        if compiled.offsets[p.0] == NONE {
                            compiled.loads.push(p);
                        }
                        give(&mut compiled.offsets, p, compiled.lens[p.0]);
                    }
                }
            }
            compiled.terms.push(term);
        }
        // the operands' registers are known now
        compiled.resolve_registers(graph);
        compiled.registers = vec![0.0; registers];
        compiled.adjoints = vec![0.0; registers];
        compiled.direct = vec![0.0; registers];
        compiled.gradient = vec![0.0; registers];
        Some(compiled)
    }

    /// The latents the gradient is with respect to.
    pub fn latents(&self) -> &[NodeId] {
        &self.latents
    }

    /// The kernel computing a dependent deterministic node, with its operands' registers left to
    /// [`resolve_registers`](Self::resolve_registers); None if its op is not compiled.
    fn compile_node(&self, graph: &Graph, id: NodeId) -> Option<Kernel> {
        let node = &graph[id];
        if node.dtype != DType::F64 {
            return None;
        }
        let parents = &node.parents;
        // which element of a parent each output element reads, broadcasting
        let operand = |p: NodeId| -> Option<Operand> {
            if graph[p].shape() == node.shape() {
                return Some(Operand::Same(NONE));
            }
            if self.lens[p.0] == 1 {
                return Some(Operand::Scalar(NONE));
            }
            let positions = DoubleTensor::new(
                graph[p].shape(),
                (0..self.lens[p.0]).map(|i| i as f64).collect(),
            )
            .ok()?
            .broadcast_to(node.shape())
            .ok()?;
            Some(Operand::Map(
                NONE,
                positions.as_slice().iter().map(|&e| e as usize).collect(),
            ))
        };
        let f64_parent = |p: NodeId| graph[p].dtype == DType::F64;
        Some(match &node.op {
            Op::Unary(op) => {
                if matches!(
                    op,
                    UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round | UnaryOp::Sign
                ) || parents.len() != 1
                    || !f64_parent(parents[0])
                    || graph[parents[0]].shape() != node.shape()
                {
                    return None;
                }
                Kernel::Unary { op: *op, x: NONE }
            }
            Op::Binary(op) => {
                // the ops whose element derivatives are formed exactly as the tensor rules form them
                if !matches!(
                    op,
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow
                ) || parents.len() != 2
                    || !parents.iter().all(|&p| f64_parent(p))
                {
                    return None;
                }
                Kernel::Binary {
                    op: *op,
                    l: operand(parents[0])?,
                    r: operand(parents[1])?,
                }
            }
            Op::Gather { axis } => {
                // along constant indices only: which element goes where is then fixed
                let [x, indices] = parents[..] else {
                    return None;
                };
                if !f64_parent(x) {
                    return None;
                }
                let Op::Constant(Value::I64(indices)) = &graph[indices].op else {
                    return None;
                };
                let numbers = DoubleTensor::new(
                    graph[x].shape(),
                    (0..self.lens[x.0]).map(|i| i as f64).collect(),
                )
                .ok()?;
                let moved = eval(
                    &Op::Gather { axis: *axis },
                    &[&Value::F64(numbers), &Value::I64(indices.clone())],
                )
                .ok()?;
                copies(&moved, node.shape(), &[0])?
            }
            Op::Proxy
            | Op::Take(_)
            | Op::Stack(_)
            | Op::Concat(_)
            | Op::Slice(_)
            | Op::Reshape(_)
            | Op::Permute(_)
            | Op::Transpose
            | Op::Broadcast(_) => {
                if parents.is_empty() || !parents.iter().all(|&p| f64_parent(p)) {
                    return None;
                }
                // each parent's elements numbered after the previous parents', so that the moved numbers say both
                // which parent and which element
                let mut first = Vec::with_capacity(parents.len());
                let mut total = 0;
                let numbered: Vec<Value> = parents
                    .iter()
                    .map(|&p| {
                        first.push(total);
                        let t = DoubleTensor::new(
                            graph[p].shape(),
                            (total..total + self.lens[p.0]).map(|i| i as f64).collect(),
                        );
                        total += self.lens[p.0];
                        t.map(Value::F64)
                    })
                    .collect::<std::result::Result<_, _>>()
                    .ok()?;
                let moved = match &node.op {
                    Op::Proxy => numbered[0].clone(),
                    op => eval(op, &numbered.iter().collect::<Vec<_>>()).ok()?,
                };
                copies(&moved, node.shape(), &first)?
            }
            _ => return None,
        })
    }

    /// How a probabilistic node's term is computed: from the registers for a Gaussian or Student-t whose value and
    /// parameters are f64 and broadcast to the node's shape, else by its distribution's general code.
    fn compile_term(&self, graph: &Graph, node: NodeId) -> Term {
        let general = Term::General(node);
        let n = &graph[node];
        let Op::Distribution(distribution) = &n.op else {
            return general;
        };
        let family = match distribution.kind {
            crate::op::DistributionKind::Gaussian => Family::Gaussian {
                single_scale: n.parents.len() == 2 && self.lens[n.parents[1].0] == 1,
            },
            crate::op::DistributionKind::LocationScaleStudentT => Family::StudentT,
            _ => return general,
        };
        if n.dtype != DType::F64
            || n.parents.len() != family.arity()
            || !n.parents.iter().all(|&p| graph[p].dtype == DType::F64)
        {
            return general;
        }
        let operand = |p: NodeId| -> Option<Operand> {
            let shape = graph[p].shape();
            if shape == n.shape() {
                return Some(Operand::Same(NONE));
            }
            if self.lens[p.0] == 1 {
                return Some(Operand::Scalar(NONE));
            }
            // the broadcast shape must be the node's
            if keanu_tensor::broadcast_shape(shape, n.shape())
                .ok()?
                .as_slice()
                != n.shape()
            {
                return None;
            }
            let positions =
                DoubleTensor::new(shape, (0..self.lens[p.0]).map(|i| i as f64).collect())
                    .ok()?
                    .broadcast_to(n.shape())
                    .ok()?;
            Some(Operand::Map(
                NONE,
                positions.as_slice().iter().map(|&e| e as usize).collect(),
            ))
        };
        let Some(parameters) = n.parents.iter().map(|&p| operand(p)).collect() else {
            return general;
        };
        Term::Registers {
            node,
            family,
            x: Operand::Same(NONE),
            parameters,
        }
    }

    /// Fills in every kernel's (and register term's) operand registers from its parents' offsets.
    fn resolve_registers(&mut self, graph: &Graph) {
        for term in &mut self.terms {
            if let Term::Registers {
                node,
                x,
                parameters,
                ..
            } = term
            {
                let parents = &graph[*node].parents;
                let operands = std::iter::once(x).chain(parameters.iter_mut());
                for (operand, p) in
                    operands.zip(std::iter::once(*node).chain(parents.iter().copied()))
                {
                    match operand {
                        Operand::Same(at) | Operand::Scalar(at) | Operand::Map(at, _) => {
                            *at = self.offsets[p.0];
                        }
                    }
                }
            }
        }
        for item in &self.items {
            let Item::Compiled { node, kernel } = item else {
                continue;
            };
            let parents = &graph[*node].parents;
            let offsets = &self.offsets;
            let resolve = |operand: &mut Operand, p: NodeId| match operand {
                Operand::Same(at) | Operand::Scalar(at) | Operand::Map(at, _) => *at = offsets[p.0],
            };
            match &mut self.kernels[*kernel] {
                Kernel::Unary { x, .. } => *x = offsets[parents[0].0],
                Kernel::Binary { l, r, .. } => {
                    resolve(l, parents[0]);
                    resolve(r, parents[1]);
                }
                Kernel::Copy { from, sources } => {
                    for (f, &(parent, element)) in from.iter_mut().zip(sources.iter()) {
                        *f = offsets[parents[parent].0] + element;
                    }
                }
            }
        }
    }

    /// The joint log probability at the latents' constrained values `values` (one tensor per latent, in order),
    /// each node's term weighted by `weights`, and unless it is impossible (−∞), its gradient with respect to each
    /// latent (in order; None where it doesn't depend on it). Every other value is read from `state`, which isn't
    /// changed.
    pub(crate) fn log_prob_and_gradient(
        &mut self,
        values: &[DoubleTensor],
        state: &State,
        weights: Weights<'_>,
    ) -> Result<(f64, Option<Vec<Option<DoubleTensor>>>)> {
        self.load_latents(values)?;
        let Some(log_prob) = self.evaluate_loaded(state, weights)? else {
            return Ok((f64::NEG_INFINITY, None));
        };
        let graph = self.model.graph();
        let gradient = (0..self.latents.len())
            .map(|i| {
                self.latent_gradient(i)
                    .map(|g| DoubleTensor::new(graph[self.latents[i]].shape(), g.to_vec()))
                    .transpose()
            })
            .collect::<std::result::Result<_, _>>()?;
        Ok((log_prob, Some(gradient)))
    }

    /// The registers holding latent `i`'s (constrained) value, for the caller to fill in before
    /// [`evaluate_loaded`](Self::evaluate_loaded).
    pub(crate) fn latent_registers(&mut self, i: usize) -> &mut [f64] {
        let id = self.latents[i];
        let at = self.offsets[id.0];
        &mut self.registers[at..at + self.lens[id.0]]
    }

    /// The gradient with respect to latent `i` from the last [`evaluate_loaded`](Self::evaluate_loaded), or None
    /// where the log probability doesn't depend on it.
    pub(crate) fn latent_gradient(&self, i: usize) -> Option<&[f64]> {
        let id = self.latents[i];
        let at = self.offsets[id.0];
        self.has_gradient[i].then(|| &self.gradient[at..at + self.lens[id.0]])
    }

    /// The joint log probability at the latents' values in their registers (with each latent's gradient left for
    /// [`latent_gradient`](Self::latent_gradient)), or None where it is impossible (−∞, including where evaluating
    /// the model fails at an impossible point).
    pub(crate) fn evaluate_loaded(
        &mut self,
        state: &State,
        weights: Weights<'_>,
    ) -> Result<Option<f64>> {
        match self.evaluate(state, weights) {
            Err(e) if e.is_impossible_point() => Ok(None),
            Ok(Some(log_prob)) if log_prob.is_finite() => Ok(Some(log_prob)),
            Ok(_) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Copies the latents' values into their registers.
    fn load_latents(&mut self, values: &[DoubleTensor]) -> Result<()> {
        let graph = self.model.graph();
        if values.len() != self.latents.len() {
            return Err(GraphError::InvalidValue(format!(
                "expected {} latent values, got {}",
                self.latents.len(),
                values.len()
            )));
        }
        for (&id, value) in self.latents.iter().zip(values) {
            if value.shape() != graph[id].shape() {
                return Err(GraphError::InvalidValue(format!(
                    "{} has shape {:?}, not {:?}",
                    graph.describe(id),
                    graph[id].shape(),
                    value.shape()
                )));
            }
            let at = self.offsets[id.0];
            self.registers[at..at + value.len()].copy_from_slice(value.as_slice());
        }
        Ok(())
    }

    fn evaluate(&mut self, state: &State, weights: Weights<'_>) -> Result<Option<f64>> {
        let model = self.model;
        let graph = model.graph();
        // the forward pass
        for v in &mut self.values {
            *v = None;
        }
        for &id in &self.loads {
            let value = static_f64(state, id)?;
            let at = self.offsets[id.0];
            self.registers[at..at + value.len()].copy_from_slice(value.as_slice());
        }
        for i in 0..self.items.len() {
            match self.items[i].clone() {
                Item::Compiled { node, kernel } => {
                    let (out, len) = (self.offsets[node.0], self.lens[node.0]);
                    forward(
                        &self.kernels[kernel],
                        &mut self.registers,
                        out,
                        len,
                        &mut self.scratch,
                    );
                }
                Item::General { node } => {
                    self.materialize_parents(graph, node)?;
                    let value = {
                        let parents = self.parent_values(graph, state, node)?;
                        match &graph[node].op {
                            Op::Proxy if parents.is_empty() => {
                                return Err(GraphError::Proxy(format!(
                                    "proxy node {node:?} was never connected"
                                )));
                            }
                            op => eval(op, &parents)?,
                        }
                    };
                    if let (Value::F64(t), true) = (&value, self.offsets[node.0] != NONE) {
                        let at = self.offsets[node.0];
                        self.registers[at..at + t.len()].copy_from_slice(t.as_slice());
                    }
                    self.values[node.0] = Some(value);
                }
            }
        }

        // the terms, in order, with their gradient parts seeded into the cotangents
        self.adjoints.fill(0.0);
        self.touched.fill(false);
        let mut log_prob = 0.0;
        self.has_direct.fill(false);
        for t in 0..self.terms.len() {
            let term = match &self.terms[t] {
                Term::General(node) => self.general_term(graph, state, weights, *node)?,
                Term::Registers { .. } => self.register_term(graph, weights, t),
            };
            match term {
                Some(term) => log_prob += term,
                None => return Ok(None),
            }
        }

        // the reverse pass, from the last dependent node to the first
        for i in (0..self.items.len()).rev() {
            match self.items[i].clone() {
                Item::Compiled { node, kernel } => {
                    self.reverse_compiled(graph, node, kernel);
                }
                Item::General { node } => self.reverse_general(graph, state, node)?,
            }
        }

        // each latent's gradient: its own term's derivative, then the cotangent from the rest
        for i in 0..self.latents.len() {
            let id = self.latents[i];
            let range = self.offsets[id.0]..self.offsets[id.0] + self.lens[id.0];
            let (direct, swept) = (self.has_direct[i], self.touched[id.0]);
            self.has_gradient[i] = direct || swept;
            let gradient = &mut self.gradient[range.clone()];
            match (direct, swept) {
                (true, true) => {
                    for ((g, d), s) in gradient
                        .iter_mut()
                        .zip(&self.direct[range.clone()])
                        .zip(&self.adjoints[range])
                    {
                        *g = d + s;
                    }
                }
                (true, false) => gradient.copy_from_slice(&self.direct[range]),
                (false, true) => gradient.copy_from_slice(&self.adjoints[range]),
                (false, false) => {}
            }
        }
        Ok(Some(log_prob))
    }

    /// A term by its distribution's general code: its weighted value (None if it is impossible), with its gradient
    /// parts seeded into the cotangents.
    fn general_term(
        &mut self,
        graph: &Graph,
        state: &State,
        weights: Weights<'_>,
        node: NodeId,
    ) -> Result<Option<f64>> {
        self.materialize(graph, node)?;
        self.materialize_parents(graph, node)?;
        let (term, parts) = {
            let value = self.value_of(graph, state, node)?;
            let parents = self.parent_values(graph, state, node)?;
            term_and_parts(graph, node, value, &parents)?
        };
        if !term.is_finite() {
            return Ok(None);
        }
        let w = weight_of(weights, node);
        let Some(parts) = parts else {
            return Ok(Some(weighted_term(w, term)));
        };
        let (value_gradient, parameters) = scale_parts(parts, w);
        if let (Some(i), Some(dx)) = (self.latents.iter().position(|&l| l == node), value_gradient)
        {
            let at = self.offsets[node.0];
            self.direct[at..at + dx.len()].copy_from_slice(dx.as_slice());
            self.has_direct[i] = true;
        }
        for (&parent, dp) in graph[node].parents.iter().zip(parameters) {
            let Some(dp) = dp else { continue };
            if !self.seeds_into(graph, parent) {
                continue;
            }
            let seed = sum_to_shape(dp, graph[parent].shape())?;
            self.accumulate(parent, seed.as_slice());
        }
        Ok(Some(weighted_term(w, term)))
    }

    /// Whether a parameter's gradient part is seeded into its cotangent: as the general path, not for a constant or
    /// a non-f64 parent, and (as it can't lead to a latent) not for one that doesn't depend on the latents.
    fn seeds_into(&self, graph: &Graph, parent: NodeId) -> bool {
        let p = &graph[parent];
        p.dtype == DType::F64 && !matches!(p.op, Op::Constant(_)) && self.dynamic[parent.0]
    }

    /// A term from the registers, exactly as the family's code computes it element by element: the weighted value
    /// (None if impossible), with the gradient parts seeded.
    fn register_term(&mut self, graph: &Graph, weights: Weights<'_>, t: usize) -> Option<f64> {
        // taken out while its operands are read, and put back
        let term = std::mem::replace(&mut self.terms[t], Term::General(NodeId(0)));
        let Term::Registers {
            node,
            family,
            x,
            parameters,
        } = &term
        else {
            unreachable!("a register term")
        };
        let (node, family) = (*node, *family);
        let len = self.lens[node.0];
        let registers = &self.registers;
        // the per-element parts with respect to x and each parameter, one after another
        let mut parts = std::mem::take(&mut self.scratch);
        parts.clear();
        parts.resize((1 + parameters.len()) * len, 0.0);
        let mut total = 0.0;
        match family {
            Family::Gaussian { single_scale } => {
                let [mu, sigma] = &parameters[..] else {
                    unreachable!("a Gaussian's parameters")
                };
                let ln_single = single_scale.then(|| registers[sigma.register(0)].ln());
                for k in 0..len {
                    let (xv, m, s) = (
                        registers[x.register(k)],
                        registers[mu.register(k)],
                        registers[sigma.register(k)],
                    );
                    let ln_s = ln_single.unwrap_or_else(|| s.ln());
                    let (lp, [dx, dm, ds]) =
                        keanu_distributions::Gaussian::element_log_prob_and_gradient(
                            xv, m, s, ln_s,
                        );
                    total += lp;
                    parts[k] = dx;
                    parts[len + k] = dm;
                    parts[2 * len + k] = ds;
                }
            }
            Family::StudentT => {
                let [nu, mu, sigma] = &parameters[..] else {
                    unreachable!("a Student-t's parameters")
                };
                use keanu_distributions::LocationScaleStudentT as T;
                // with one nu and one sigma, the parts that depend only on them are computed once
                let common =
                    matches!((nu, sigma), (Operand::Scalar(_), Operand::Scalar(_))).then(|| {
                        T::element_common(registers[nu.register(0)], registers[sigma.register(0)])
                    });
                for k in 0..len {
                    let (xv, m) = (registers[x.register(k)], registers[mu.register(k)]);
                    let (lp, [dx, dn, dm, ds]) = match &common {
                        Some(common) => T::element_log_prob_and_gradient_with(xv, m, common),
                        None => T::element_log_prob_and_gradient(
                            xv,
                            registers[nu.register(k)],
                            m,
                            registers[sigma.register(k)],
                        ),
                    };
                    total += lp;
                    parts[k] = dx;
                    parts[len + k] = dn;
                    parts[2 * len + k] = dm;
                    parts[3 * len + k] = ds;
                }
            }
        }
        if !total.is_finite() {
            self.scratch = parts;
            self.terms[t] = term;
            return None;
        }
        let w = weight_of(weights, node);
        if w != 1.0 {
            for p in parts.iter_mut() {
                *p *= w;
            }
        }
        if let Some(i) = self.latents.iter().position(|&l| l == node) {
            let at = self.offsets[node.0];
            self.direct[at..at + len].copy_from_slice(&parts[..len]);
            self.has_direct[i] = true;
        }
        let mut seed = Vec::new();
        for (j, (operand, &parent)) in parameters.iter().zip(&graph[node].parents).enumerate() {
            if !self.seeds_into(graph, parent) {
                continue;
            }
            let part = &parts[(j + 1) * len..(j + 2) * len];
            match operand {
                // the part has the parameter's shape: seeded as it is
                Operand::Same(_) => self.accumulate(parent, part),
                // summed in order from zero, as summing it to the parameter's shape does
                operand => {
                    seed.clear();
                    seed.resize(self.lens[parent.0], 0.0);
                    for (k, &c) in part.iter().enumerate() {
                        seed[operand.element(k)] += c;
                    }
                    self.accumulate(parent, &seed);
                }
            }
        }
        self.scratch = parts;
        self.terms[t] = term;
        Some(weighted_term(w, total))
    }

    /// Adds a contribution to a dependent f64 node's cotangent (or sets it, if it has none yet).
    fn accumulate(&mut self, node: NodeId, contribution: &[f64]) {
        let at = self.offsets[node.0];
        let slot = &mut self.adjoints[at..at + contribution.len()];
        if std::mem::replace(&mut self.touched[node.0], true) {
            for (a, c) in slot.iter_mut().zip(contribution) {
                *a += c;
            }
        } else {
            slot.copy_from_slice(contribution);
        }
    }

    /// The reverse rule of a compiled node: as the general sweep, each parent's contribution is summed from zero
    /// over the output elements in order and then added to the parent's cotangent.
    fn reverse_compiled(&mut self, graph: &Graph, node: NodeId, kernel: usize) {
        if !self.touched[node.0] {
            return;
        }
        let parents = &graph[node].parents;
        let (out, len) = (self.offsets[node.0], self.lens[node.0]);
        // the contributions, parent by parent, in one scratch buffer
        let mut starts = smallvec::SmallVec::<[usize; 4]>::new();
        let mut total = 0;
        for &p in parents {
            starts.push(total);
            total += self.lens[p.0];
        }
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.clear();
        scratch.resize(total, 0.0);
        let registers = &self.registers;
        let g = &self.adjoints[out..out + len];
        let y = &registers[out..out + len];
        let masked = y.iter().any(|v| !v.is_finite()) && g.iter().all(|&g| g == 0.0);
        if !masked {
            match &self.kernels[kernel] {
                Kernel::Unary { op, x } => {
                    let x = &registers[*x..*x + len];
                    let scratch = &mut scratch[..len];
                    // one loop per op, with the derivative formed as the tensor rule forms it
                    macro_rules! apply {
                        ($d:expr) => {
                            for k in 0..len {
                                scratch[k] += g[k] * $d(x[k], y[k]);
                            }
                        };
                    }
                    match op {
                        UnaryOp::Neg => apply!(|_, _| -1.0),
                        UnaryOp::Exp => apply!(|_, y: f64| y),
                        UnaryOp::Ln => apply!(|x: f64, _| 1.0 / x),
                        UnaryOp::Ln1p => apply!(|x: f64, _| 1.0 / (1.0 + x)),
                        UnaryOp::Sqrt => apply!(|_, y: f64| 0.5 / y),
                        UnaryOp::Tanh => apply!(|_, y: f64| 1.0 - y * y),
                        UnaryOp::Sigmoid => apply!(|_, y: f64| y * (1.0 - y)),
                        UnaryOp::Reciprocal => apply!(|_, y: f64| -y * y),
                        op => apply!(|x, y| unary_derivative_scalar(*op, x, y).unwrap_or(f64::NAN)),
                    }
                }
                Kernel::Binary { op, l, r } => {
                    let (left, right) = scratch.split_at_mut(starts[1]);
                    match op {
                        // the general rules pass the cotangent through (or negate it) rather than multiplying
                        BinaryOp::Add => {
                            reverse_binary(registers, g, y, l, r, left, right, |g, _, _, _| (g, g))
                        }
                        BinaryOp::Sub => {
                            reverse_binary(registers, g, y, l, r, left, right, |g, _, _, _| (g, -g))
                        }
                        BinaryOp::Mul => {
                            reverse_binary(registers, g, y, l, r, left, right, |g, l, r, _| {
                                (g * r, g * l)
                            });
                        }
                        BinaryOp::Div => {
                            reverse_binary(registers, g, y, l, r, left, right, |g, l, r, _| {
                                (g * (1.0 / r), g * (-l / (r * r)))
                            })
                        }
                        op => reverse_binary(registers, g, y, l, r, left, right, |g, l, r, y| {
                            let (dl, dr) = binary_derivatives_scalar(*op, l, r, y)
                                .unwrap_or((f64::NAN, f64::NAN));
                            (g * dl, g * dr)
                        }),
                    }
                }
                Kernel::Copy { sources, .. } => {
                    for (k, &(parent, element)) in sources.iter().enumerate() {
                        scratch[starts[parent] + element] += g[k];
                    }
                }
            }
        }
        for (j, &p) in parents.iter().enumerate() {
            if self.dynamic[p.0] && self.offsets[p.0] != NONE && is_differentiable(graph, p) {
                let range = starts[j]..starts[j] + self.lens[p.0];
                self.accumulate(p, &scratch[range]);
            }
        }
        self.scratch = scratch;
    }

    /// The reverse rule of a node computed by its general rules, as the general sweep applies it.
    fn reverse_general(&mut self, graph: &Graph, state: &State, node: NodeId) -> Result<()> {
        if !self.touched[node.0] || self.offsets[node.0] == NONE || !is_differentiable(graph, node)
        {
            return Ok(());
        }
        let node_ref = &graph[node];
        let at = self.offsets[node.0];
        let cotangent = DoubleTensor::new(
            &crate::autodiff::batched_shape(1, node_ref.shape()),
            self.adjoints[at..at + self.lens[node.0]].to_vec(),
        )?;
        self.materialize_parents(graph, node)?;
        let contributions: Vec<Option<DoubleTensor>> = {
            let out = self.value_of(graph, state, node)?;
            if masked_out(&cotangent, out) {
                node_ref
                    .parents
                    .iter()
                    .map(|&p| Some(DoubleTensor::zeros(graph[p].shape())))
                    .collect()
            } else {
                let parents = self.parent_values(graph, state, node)?;
                match scalar_vjp(node_ref, &parents, out, &cotangent) {
                    Some(fast) => fast.into_iter().collect(),
                    None => vjp(graph, node, &parents, out, cotangent, 1)?,
                }
            }
        };
        for (&p, c) in node_ref.parents.iter().zip(contributions) {
            let Some(c) = c else { continue };
            if self.dynamic[p.0] && self.offsets[p.0] != NONE && is_differentiable(graph, p) {
                self.accumulate(p, c.as_slice());
            }
        }
        Ok(())
    }

    /// Makes a dependent f64 node's value a tensor from its registers, if it isn't one already.
    fn materialize(&mut self, graph: &Graph, node: NodeId) -> Result<()> {
        if self.dynamic[node.0] && self.values[node.0].is_none() && self.offsets[node.0] != NONE {
            let at = self.offsets[node.0];
            self.values[node.0] = Some(Value::F64(DoubleTensor::new(
                graph[node].shape(),
                self.registers[at..at + self.lens[node.0]].to_vec(),
            )?));
        }
        Ok(())
    }

    fn materialize_parents(&mut self, graph: &Graph, node: NodeId) -> Result<()> {
        for &p in &graph[node].parents {
            self.materialize(graph, p)?;
        }
        Ok(())
    }

    /// A node's value: its tensor if it depends on the latents (made by [`materialize`](Self::materialize) or
    /// by its general rules), else its value in the state.
    fn value_of<'a>(&'a self, graph: &Graph, state: &'a State, node: NodeId) -> Result<&'a Value> {
        if self.dynamic[node.0] {
            self.values[node.0].as_ref().ok_or_else(|| {
                GraphError::InvalidValue(format!("{} has no value", graph.describe(node)))
            })
        } else {
            state.value(node).ok_or(GraphError::MissingValue(node))
        }
    }

    fn parent_values<'a>(
        &'a self,
        graph: &Graph,
        state: &'a State,
        node: NodeId,
    ) -> Result<smallvec::SmallVec<[&'a Value; 4]>> {
        graph[node]
            .parents
            .iter()
            .map(|&p| self.value_of(graph, state, p))
            .collect()
    }
}

/// The copy kernel taking a moved tensor of element numbers to a node's elements: element k of the node is number
/// `moved[k]`, where each parent's numbers start at its entry of `first`.
fn copies(moved: &Value, shape: &[usize], first: &[usize]) -> Option<Kernel> {
    let Value::F64(moved) = moved else {
        return None;
    };
    if moved.shape() != shape {
        return None;
    }
    let sources: Vec<(usize, usize)> = moved
        .as_slice()
        .iter()
        .map(|&number| {
            let number = number as usize;
            let parent = first.iter().rposition(|&f| f <= number).unwrap_or(0);
            (parent, number - first[parent])
        })
        .collect();
    Some(Kernel::Copy {
        from: vec![NONE; sources.len()],
        sources,
    })
}

/// The reverse rule of a binary kernel: `contributions(g, l, r, y)` gives the two operands' contributions of one
/// output element, which are summed into `left` and `right` (each parent's elements) in output order.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn reverse_binary(
    registers: &[f64],
    g: &[f64],
    y: &[f64],
    l: &Operand,
    r: &Operand,
    left: &mut [f64],
    right: &mut [f64],
    contributions: impl Fn(f64, f64, f64, f64) -> (f64, f64),
) {
    let len = g.len();
    match (l, r) {
        (Operand::Same(a), Operand::Same(b)) => {
            let (lv, rv) = (&registers[*a..*a + len], &registers[*b..*b + len]);
            let (left, right) = (&mut left[..len], &mut right[..len]);
            for k in 0..len {
                let (cl, cr) = contributions(g[k], lv[k], rv[k], y[k]);
                left[k] += cl;
                right[k] += cr;
            }
        }
        (Operand::Same(a), Operand::Scalar(b)) => {
            let (lv, rv) = (&registers[*a..*a + len], registers[*b]);
            let left = &mut left[..len];
            let mut total = right[0];
            for k in 0..len {
                let (cl, cr) = contributions(g[k], lv[k], rv, y[k]);
                left[k] += cl;
                total += cr;
            }
            right[0] = total;
        }
        (Operand::Scalar(a), Operand::Same(b)) => {
            let (lv, rv) = (registers[*a], &registers[*b..*b + len]);
            let right = &mut right[..len];
            let mut total = left[0];
            for k in 0..len {
                let (cl, cr) = contributions(g[k], lv, rv[k], y[k]);
                total += cl;
                right[k] += cr;
            }
            left[0] = total;
        }
        _ => {
            for k in 0..len {
                let (cl, cr) = contributions(
                    g[k],
                    registers[l.register(k)],
                    registers[r.register(k)],
                    y[k],
                );
                left[l.element(k)] += cl;
                right[r.element(k)] += cr;
            }
        }
    }
}

/// Runs a kernel, writing the `len` registers from `out`. The results are formed in `buffer` (reading the
/// operands as slices, which lets the loops vectorise) and then copied into place.
fn forward(kernel: &Kernel, registers: &mut [f64], out: usize, len: usize, buffer: &mut Vec<f64>) {
    buffer.clear();
    buffer.resize(len, 0.0);
    let result = &mut buffer[..len];
    let input = &*registers;
    match kernel {
        Kernel::Unary { op, x } => {
            let x = &input[*x..*x + len];
            // one loop per op, so that each is inlined (and vectorised where it can be)
            macro_rules! apply {
                ($f:expr) => {
                    for (r, &x) in result.iter_mut().zip(x) {
                        *r = $f(x);
                    }
                };
            }
            use keanu_tensor::special;
            match op {
                UnaryOp::Neg => apply!(|x: f64| -x),
                UnaryOp::Abs => apply!(f64::abs),
                UnaryOp::Reciprocal => apply!(|x: f64| 1.0 / x),
                UnaryOp::Exp => apply!(f64::exp),
                UnaryOp::ExpM1 => apply!(f64::exp_m1),
                UnaryOp::Ln => apply!(f64::ln),
                UnaryOp::Ln1p => apply!(f64::ln_1p),
                UnaryOp::Sqrt => apply!(f64::sqrt),
                UnaryOp::Sin => apply!(f64::sin),
                UnaryOp::Cos => apply!(f64::cos),
                UnaryOp::Tanh => apply!(f64::tanh),
                UnaryOp::Sigmoid => apply!(|x: f64| 1.0 / (1.0 + (-x).exp())),
                UnaryOp::LnGamma => apply!(special::ln_gamma),
                UnaryOp::Softplus => apply!(special::softplus),
                UnaryOp::Logit => apply!(special::logit),
                UnaryOp::Log1mExp => apply!(special::log1mexp),
                op => apply!(|x| unary(*op, x)),
            }
        }
        Kernel::Binary { op, l, r } => match op {
            BinaryOp::Add => binary_kernel(input, l, r, result, |a, b| a + b),
            BinaryOp::Sub => binary_kernel(input, l, r, result, |a, b| a - b),
            BinaryOp::Mul => binary_kernel(input, l, r, result, |a, b| a * b),
            BinaryOp::Div => binary_kernel(input, l, r, result, |a, b| a / b),
            BinaryOp::Pow => binary_kernel(input, l, r, result, f64::powf),
            op => binary_kernel(input, l, r, result, |a, b| binary(*op, a, b)),
        },
        Kernel::Copy { from, .. } => {
            for (r, &f) in result.iter_mut().zip(from) {
                *r = input[f];
            }
        }
    }
    registers[out..out + len].copy_from_slice(result);
}

/// A binary kernel's results, by the element function `f`.
#[inline(always)]
fn binary_kernel(
    input: &[f64],
    l: &Operand,
    r: &Operand,
    result: &mut [f64],
    f: impl Fn(f64, f64) -> f64,
) {
    let len = result.len();
    match (l, r) {
        (Operand::Same(a), Operand::Same(b)) => {
            let (a, b) = (&input[*a..*a + len], &input[*b..*b + len]);
            for k in 0..len {
                result[k] = f(a[k], b[k]);
            }
        }
        (Operand::Same(a), Operand::Scalar(b)) => {
            let (a, b) = (&input[*a..*a + len], input[*b]);
            for k in 0..len {
                result[k] = f(a[k], b);
            }
        }
        (Operand::Scalar(a), Operand::Same(b)) => {
            let (a, b) = (input[*a], &input[*b..*b + len]);
            for k in 0..len {
                result[k] = f(a, b[k]);
            }
        }
        _ => {
            for k in 0..len {
                result[k] = f(input[l.register(k)], input[r.register(k)]);
            }
        }
    }
}

/// The value in the state of a node that doesn't depend on the latents, as f64.
fn static_f64(state: &State, id: NodeId) -> Result<&DoubleTensor> {
    state
        .value(id)
        .ok_or(GraphError::MissingValue(id))?
        .as_f64()
        .ok_or_else(|| GraphError::InvalidValue(format!("node {id:?} is not f64")))
}

fn unary(op: UnaryOp, x: f64) -> f64 {
    match op {
        UnaryOp::Neg => -x,
        UnaryOp::Exp => x.exp(),
        UnaryOp::Ln => x.ln(),
        UnaryOp::Sqrt => x.sqrt(),
        _ => unary_f64(op, &DoubleTensor::scalar(x)).as_slice()[0],
    }
}

fn binary(op: BinaryOp, l: f64, r: f64) -> f64 {
    match op {
        BinaryOp::Add => l + r,
        BinaryOp::Sub => l - r,
        BinaryOp::Mul => l * r,
        BinaryOp::Div => l / r,
        _ => binary_f64(op, &DoubleTensor::scalar(l), &DoubleTensor::scalar(r))
            .ok()
            .flatten()
            .map_or(f64::NAN, |t| t.as_slice()[0]),
    }
}

/// The gradient as a [`LogProbGradient`], for the parameterisation's pull back.
pub(crate) fn as_log_prob_gradient(
    latents: &[NodeId],
    gradient: Vec<Option<DoubleTensor>>,
    nodes: usize,
) -> Result<LogProbGradient> {
    let mut result = LogProbGradient::with_capacity(nodes);
    for (&id, g) in latents.iter().zip(gradient) {
        if let Some(g) = g {
            result.add(id, g)?;
        }
    }
    Ok(result)
}
