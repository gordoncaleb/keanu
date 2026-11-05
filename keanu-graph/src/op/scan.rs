//! The general scan: a loop that runs a nested, deterministic graph (the body) once per step, threading one carry
//! tensor through the steps and stacking the carries.
//!
//! The body is evaluated on its own [`State`] for each step. Its auto diff rules recompute each step's body state
//! from the stacked output (the carry into step t is output t − 1, or the initial carry) rather than storing the
//! states, so memory stays O(T) and the outer [`State`] and the vector-Jacobian product signature are unchanged.
//! Every loop over the steps is iterative, so long scans don't grow the call stack.

use super::{Op, eval};
use crate::autodiff::{accumulate, forward_tangents, reverse_sweep, second_order_sweep};
use crate::error::{GraphError, Result};
use crate::graph::{Graph, NodeId};
use crate::state::State;
use crate::value::{DType, Value};
use keanu_tensor::{DoubleTensor, Shape};
use std::fmt;

/// The body of a [`Op::Scan`]: a deterministic [`Graph`] with input placeholders and one output.
///
/// The placeholders are unconnected f64 proxy nodes of the body graph:
///
/// - the **carry**, holding the initial carry at the first step and the previous step's output after that;
/// - one **per-step input** for each `xs` parent of the scan node, holding element t (along axis 0) of that parent at
///   step t;
/// - one **constant** for each constant parent, holding that parent's value at every step.
///
/// The **output** is the next carry: an f64 node of the carry's shape. A scan node's parents are the initial carry,
/// the `xs` and the constants, in that order, and its value stacks the outputs of the `length` steps, with shape
/// `[length, carry shape...]`. Several carries are packed into one tensor (with `stack` and `slice`).
///
/// Bodies contain no probabilistic nodes, and they can't refer to nodes outside the body: values from the
/// enclosing graph enter through constants. Build one with [`GraphBuilder::scan`](crate::GraphBuilder::scan), or
/// from a body graph with [`ScanBody::new`].
#[derive(Clone)]
pub struct ScanBody {
    graph: Graph,
    length: usize,
    carry: NodeId,
    xs: Vec<NodeId>,
    constants: Vec<NodeId>,
    output: NodeId,
}

fn invalid(message: impl Into<String>) -> GraphError {
    GraphError::InvalidNode {
        op: "scan",
        message: message.into(),
    }
}

/// What a body node is fed at each step.
#[derive(Clone, Copy)]
enum Input {
    Carry,
    Step(usize),
    Constant(usize),
}

impl ScanBody {
    /// A scan body of `length` steps over a body graph, given its placeholders and output (see [`ScanBody`]).
    ///
    /// Fails if a placeholder is not an unconnected f64 proxy or is listed twice, if the body has a probabilistic
    /// node or another unconnected proxy, or if the output is not f64 with the carry's shape.
    pub fn new(
        graph: Graph,
        length: usize,
        carry: NodeId,
        xs: Vec<NodeId>,
        constants: Vec<NodeId>,
        output: NodeId,
    ) -> Result<Self> {
        let inputs: Vec<NodeId> = std::iter::once(carry)
            .chain(xs.iter().copied())
            .chain(constants.iter().copied())
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
        for id in graph.ids() {
            let node = &graph[id];
            if node.is_probabilistic() {
                return Err(invalid(format!(
                    "the body's node {} is probabilistic: scan bodies must be deterministic (draw random \
                     variables outside the scan and pass them in as per-step inputs or constants)",
                    crate::autodiff::describe(&graph, id)
                )));
            }
            if matches!(node.op, Op::Proxy) && node.parents.is_empty() && !is_input[id.0] {
                return Err(invalid(format!(
                    "the body's proxy {} is neither connected nor an input",
                    crate::autodiff::describe(&graph, id)
                )));
            }
        }
        let (out, carry_node) = (&graph[output], &graph[carry]);
        if out.dtype != DType::F64 || out.shape != carry_node.shape {
            return Err(invalid(format!(
                "the body's output is {:?} of shape {:?}, but the carry is f64 of shape {:?}",
                out.dtype,
                out.shape(),
                carry_node.shape()
            )));
        }
        let mut stacked = vec![length];
        stacked.extend_from_slice(carry_node.shape());
        if super::shape::element_count(&stacked).is_none() {
            return Err(invalid(format!(
                "{length} steps of a carry of shape {:?} is too many elements (at most {})",
                carry_node.shape(),
                super::shape::MAX_ELEMENTS
            )));
        }
        Ok(Self {
            graph,
            length,
            carry,
            xs,
            constants,
            output,
        })
    }

    /// The body graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The number of steps, T.
    pub fn length(&self) -> usize {
        self.length
    }

    /// The carry placeholder.
    pub fn carry(&self) -> NodeId {
        self.carry
    }

    /// The per-step input placeholders, one per `xs` parent.
    pub fn xs(&self) -> &[NodeId] {
        &self.xs
    }

    /// The constant placeholders, one per constant parent.
    pub fn constants(&self) -> &[NodeId] {
        &self.constants
    }

    /// The output: the next carry.
    pub fn output(&self) -> NodeId {
        self.output
    }

    /// The number of parents a scan node with this body takes: the initial carry, the `xs` and the constants.
    pub(crate) fn parent_count(&self) -> usize {
        1 + self.xs.len() + self.constants.len()
    }

    fn input(&self, id: NodeId) -> Option<Input> {
        if id == self.carry {
            return Some(Input::Carry);
        }
        if let Some(i) = self.xs.iter().position(|&x| x == id) {
            return Some(Input::Step(i));
        }
        self.constants
            .iter()
            .position(|&c| c == id)
            .map(Input::Constant)
    }

    fn is_input(&self) -> Vec<bool> {
        let mut is_input = vec![false; self.graph.len()];
        for id in std::iter::once(self.carry).chain(self.xs.iter().copied()) {
            is_input[id.0] = true;
        }
        for &id in &self.constants {
            is_input[id.0] = true;
        }
        is_input
    }

    fn shape_of(&self, id: NodeId) -> &[usize] {
        self.graph[id].shape()
    }

    /// The scan node's type and shape given its parents': f64 `[length, carry shape...]`.
    pub(crate) fn output_type(&self, parents: &[(DType, &[usize])]) -> Result<(DType, Shape)> {
        if parents.len() != self.parent_count() {
            return Err(invalid(format!(
                "needs {} parents (the initial carry, {} per-step inputs and {} constants), got {}",
                self.parent_count(),
                self.xs.len(),
                self.constants.len(),
                parents.len()
            )));
        }
        if parents.iter().any(|(t, _)| *t != DType::F64) {
            return Err(invalid(
                "the initial carry, per-step inputs and constants must be f64",
            ));
        }
        let carry = self.shape_of(self.carry);
        if parents[0].1 != carry {
            return Err(invalid(format!(
                "the initial carry has shape {:?}, but the body's carry has shape {carry:?}",
                parents[0].1
            )));
        }
        for (i, &x) in self.xs.iter().enumerate() {
            let shape = parents[1 + i].1;
            let step = self.shape_of(x);
            if shape.first() != Some(&self.length) || &shape[1..] != step {
                return Err(invalid(format!(
                    "per-step input {i} has shape {shape:?}, but {} steps of the body's input of shape {step:?} \
                     need shape [{}, ...{step:?}]",
                    self.length, self.length
                )));
            }
        }
        for (i, &c) in self.constants.iter().enumerate() {
            let shape = parents[1 + self.xs.len() + i].1;
            if shape != self.shape_of(c) {
                return Err(invalid(format!(
                    "constant {i} has shape {shape:?}, but the body's constant has shape {:?}",
                    self.shape_of(c)
                )));
            }
        }
        let mut shape = Shape::from_slice(&[self.length]);
        shape.extend_from_slice(carry);
        Ok((DType::F64, shape))
    }

    /// The body's state at one step: every body node evaluated from the step's carry, per-step inputs and constants.
    fn run(&self, carry: DoubleTensor, parents: &[&Value], step: usize) -> Result<State> {
        let n = self.graph.len();
        let mut values: Vec<Option<Value>> = vec![None; n];
        values[self.carry.0] = Some(Value::F64(carry));
        for (i, &x) in self.xs.iter().enumerate() {
            let xs = f64_of(parents[1 + i])?;
            values[x.0] = Some(Value::F64(take_step(
                xs,
                1,
                self.length,
                step,
                self.shape_of(x),
            )?));
        }
        for (i, &c) in self.constants.iter().enumerate() {
            values[c.0] = Some(parents[1 + self.xs.len() + i].clone());
        }
        for i in 0..n {
            if values[i].is_some() {
                continue;
            }
            let node = &self.graph.nodes[i];
            let value = match &node.op {
                Op::Constant(v) => v.clone(),
                op => {
                    let inputs = node
                        .parents
                        .iter()
                        .map(|p| values[p.0].as_ref().ok_or(GraphError::MissingValue(*p)))
                        .collect::<Result<smallvec::SmallVec<[&Value; 4]>>>()?;
                    match op {
                        Op::Proxy => inputs[0].clone(),
                        op => eval(op, &inputs)?,
                    }
                }
            };
            values[i] = Some(value);
        }
        Ok(State {
            values,
            observed: vec![false; n],
        })
    }

    /// The carry into a step: the initial carry at step 0, then the previous step's output, taken from the stacked
    /// output of the scan.
    fn carry_at(
        &self,
        parents: &[&Value],
        out: &DoubleTensor,
        step: usize,
    ) -> Result<DoubleTensor> {
        if step == 0 {
            Ok(f64_of(parents[0])?.clone())
        } else {
            take_step(out, 1, self.length, step - 1, self.shape_of(self.carry))
        }
    }

    /// The body's state at a step, recomputed from the scan's parents and its stacked output.
    fn state_at(&self, parents: &[&Value], out: &DoubleTensor, step: usize) -> Result<State> {
        self.run(self.carry_at(parents, out, step)?, parents, step)
    }

    /// Evaluates the scan: runs the body `length` times and stacks the outputs.
    pub(crate) fn eval(&self, parents: &[&Value]) -> Result<Value> {
        if parents.len() != self.parent_count() {
            return Err(invalid(format!(
                "expected {} parents, got {}",
                self.parent_count(),
                parents.len()
            )));
        }
        let carry_shape = self.shape_of(self.carry);
        let mut shape = vec![self.length];
        shape.extend_from_slice(carry_shape);
        let total = super::shape::element_count(&shape)
            .ok_or_else(|| invalid(format!("the output shape {shape:?} has too many elements")))?;
        let mut data = Vec::with_capacity(total);
        let mut carry = f64_of(parents[0])?.clone();
        for step in 0..self.length {
            let mut state = self.run(carry, parents, step)?;
            let Some(Value::F64(next)) = state.values[self.output.0].take() else {
                return Err(invalid("the body's output is not f64"));
            };
            data.extend_from_slice(next.as_slice());
            carry = next;
        }
        Ok(Value::F64(DoubleTensor::new(&shape, data)?))
    }

    /// The forward seeds of a step: the carry's tangent and the parents' tangents for this step, `[b, input...]`.
    fn step_seeds(
        &self,
        carry: Option<DoubleTensor>,
        t: &[Option<&DoubleTensor>],
        step: usize,
        b: usize,
    ) -> Result<Vec<(NodeId, DoubleTensor)>> {
        let mut seeds = Vec::new();
        if let Some(c) = carry {
            seeds.push((self.carry, c));
        }
        for (i, &x) in self.xs.iter().enumerate() {
            if let Some(tx) = t[1 + i] {
                let shape = crate::autodiff::batched_shape(b, self.shape_of(x));
                seeds.push((x, take_step(tx, b, self.length, step, &shape)?));
            }
        }
        for (i, &c) in self.constants.iter().enumerate() {
            if let Some(tc) = t[1 + self.xs.len() + i] {
                seeds.push((c, tc.clone()));
            }
        }
        Ok(seeds)
    }

    /// The batched output shape `[b, length, carry...]`.
    fn batched_out_shape(&self, b: usize) -> Vec<usize> {
        let mut shape = vec![b, self.length];
        shape.extend_from_slice(self.shape_of(self.carry));
        shape
    }

    /// Forward mode: the tangent of the stacked output `[b, length, carry...]` from the parents' tangents, pushing
    /// the carry's tangent through the body step by step.
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
        let m: usize = self.shape_of(self.carry).iter().product();
        let mut result = vec![0.0; b * self.length * m];
        let mut carry = t[0].cloned();
        for step in 0..self.length {
            let seeds = self.step_seeds(carry.take(), t, step, b)?;
            if seeds.is_empty() {
                continue;
            }
            let state = self.state_at(parents, out, step)?;
            let mut tangents = forward_tangents(&self.graph, &state, seeds, b)?;
            carry = tangents[self.output.0].take();
            if let Some(c) = &carry {
                put_step(&mut result, b, self.length, step, c.as_slice());
            }
        }
        Ok(Some(DoubleTensor::new(&self.batched_out_shape(b), result)?))
    }

    /// Reverse mode: the parents' cotangents from the stacked output's cotangent `[b, length, carry...]`, sweeping
    /// the steps backwards and recomputing each step's body state.
    pub(crate) fn vjp(
        &self,
        parents: &[&Value],
        out: &DoubleTensor,
        g: &DoubleTensor,
        b: usize,
    ) -> Result<Vec<Option<DoubleTensor>>> {
        let is_input = self.is_input();
        let carry_shape = crate::autodiff::batched_shape(b, self.shape_of(self.carry));
        let mut grads = Grads::new(self, parents, b);
        let mut carry: Option<DoubleTensor> = None;
        for step in (0..self.length).rev() {
            let mut seed = take_step(g, b, self.length, step, &carry_shape)?;
            if let Some(c) = carry.take() {
                seed = seed.add(&c)?;
            }
            let state = self.state_at(parents, out, step)?;
            reverse_sweep(
                &self.graph,
                &state,
                [(self.output, seed)],
                b,
                &is_input,
                |id, cotangent| {
                    match self.input(id) {
                        Some(Input::Carry) => carry = Some(cotangent),
                        Some(input) => grads.add(input, step, cotangent)?,
                        None => {}
                    }
                    Ok(())
                },
            )?;
        }
        grads.finish(carry)
    }

    /// The second-order rule: the derivative of [`vjp`](Self::vjp) for the cotangent `g` (`[1, length, carry...]`)
    /// along the parents' forward tangents `t` (`[b, parent...]`), holding `g` fixed. `yt`, the output's forward
    /// tangent, gives the carry's tangent into each step; it is recomputed if absent.
    ///
    /// This is the second-order sweep of the unrolled loop: backwards over the steps, each step's body carries the
    /// cotangent and its derivative, seeded with the step's slice of `g` plus what the next step sent back to its
    /// carry, and with no derivative at the scan's output.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn vjp_tangent(
        &self,
        parents: &[&Value],
        out: &DoubleTensor,
        g: &DoubleTensor,
        t: &[Option<&DoubleTensor>],
        yt: Option<&DoubleTensor>,
        b: usize,
    ) -> Result<Vec<Option<DoubleTensor>>> {
        if t.iter().all(Option::is_none) {
            return Ok(vec![None; parents.len()]);
        }
        let recomputed;
        let yt = match yt {
            Some(yt) => Some(yt),
            None => {
                recomputed = self.jvp(parents, out, t, b)?;
                recomputed.as_ref()
            }
        };
        let is_input = self.is_input();
        let carry = self.shape_of(self.carry);
        let (carry_1, carry_b) = (
            crate::autodiff::batched_shape(1, carry),
            crate::autodiff::batched_shape(b, carry),
        );
        let mut grads = Grads::new(self, parents, b);
        let (mut cotangent, mut derivative): (Option<DoubleTensor>, Option<DoubleTensor>) =
            (None, None);
        for step in (0..self.length).rev() {
            let carry_tangent = match (step, yt) {
                (0, _) => t[0].cloned(),
                (_, Some(yt)) => Some(take_step(yt, b, self.length, step - 1, &carry_b)?),
                (_, None) => None,
            };
            let seeds = self.step_seeds(carry_tangent, t, step, b)?;
            let state = self.state_at(parents, out, step)?;
            let tangents = forward_tangents(&self.graph, &state, seeds, b)?;
            let mut seed = take_step(g, 1, self.length, step, &carry_1)?;
            if let Some(c) = cotangent.take() {
                seed = seed.add(&c)?;
            }
            second_order_sweep(
                &self.graph,
                &state,
                &tangents,
                [(self.output, Some(seed), derivative.take())],
                b,
                &is_input,
                |id, c, d| {
                    match self.input(id) {
                        Some(Input::Carry) => (cotangent, derivative) = (c, d),
                        Some(input) => {
                            if let Some(d) = d {
                                grads.add(input, step, d)?;
                            }
                        }
                        None => {}
                    }
                    Ok(())
                },
            )?;
        }
        grads.finish(derivative)
    }
}

/// The parents' (co)tangents a backwards sweep over the steps accumulates: per-step inputs into their step's slot,
/// constants summed over the steps.
struct Grads<'a> {
    body: &'a ScanBody,
    b: usize,
    xs: Vec<Option<Vec<f64>>>,
    constants: Vec<Option<DoubleTensor>>,
    xs_shapes: Vec<Vec<usize>>,
}

impl<'a> Grads<'a> {
    fn new(body: &'a ScanBody, parents: &[&Value], b: usize) -> Self {
        Self {
            body,
            b,
            xs: vec![None; body.xs.len()],
            constants: vec![None; body.constants.len()],
            xs_shapes: (0..body.xs.len())
                .map(|i| {
                    let mut shape = vec![b];
                    shape.extend_from_slice(parents[1 + i].shape());
                    shape
                })
                .collect(),
        }
    }

    fn add(&mut self, input: Input, step: usize, contribution: DoubleTensor) -> Result<()> {
        match input {
            Input::Step(i) => {
                let len = self.xs_shapes[i].iter().product();
                let slot = self.xs[i].get_or_insert_with(|| vec![0.0; len]);
                put_step(
                    slot,
                    self.b,
                    self.body.length,
                    step,
                    contribution.as_slice(),
                );
            }
            Input::Constant(i) => accumulate(&mut self.constants[i], contribution)?,
            Input::Carry => {}
        }
        Ok(())
    }

    fn finish(self, carry: Option<DoubleTensor>) -> Result<Vec<Option<DoubleTensor>>> {
        let mut result = vec![carry];
        for (data, shape) in self.xs.into_iter().zip(&self.xs_shapes) {
            result.push(data.map(|d| DoubleTensor::new(shape, d)).transpose()?);
        }
        result.extend(self.constants);
        Ok(result)
    }
}

fn f64_of(value: &Value) -> Result<&DoubleTensor> {
    value
        .as_f64()
        .ok_or_else(|| invalid("the initial carry, per-step inputs and constants must be f64"))
}

/// Element `step` along axis 1 of a tensor of `b` blocks of `length` equal elements (`[b, length, rest...]`, or
/// `[length, rest...]` for `b` = 1), shaped `shape`.
fn take_step(
    x: &DoubleTensor,
    b: usize,
    length: usize,
    step: usize,
    shape: &[usize],
) -> Result<DoubleTensor> {
    let data = x.as_slice();
    let m = data.len() / (b * length).max(1);
    let mut out = Vec::with_capacity(b * m);
    for k in 0..b {
        let start = (k * length + step) * m;
        out.extend_from_slice(&data[start..start + m]);
    }
    Ok(DoubleTensor::new(shape, out)?)
}

/// Writes element `step` along axis 1 of `b` blocks of `length` equal elements.
fn put_step(dst: &mut [f64], b: usize, length: usize, step: usize, src: &[f64]) {
    let m = src.len() / b.max(1);
    for k in 0..b {
        let start = (k * length + step) * m;
        dst[start..start + m].copy_from_slice(&src[k * m..(k + 1) * m]);
    }
}

impl fmt::Debug for ScanBody {
    /// The length, the placeholders and output, and each body node's operation, parents, type, shape and label.
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
        f.debug_struct("ScanBody")
            .field("length", &self.length)
            .field("carry", &self.carry)
            .field("xs", &self.xs)
            .field("constants", &self.constants)
            .field("output", &self.output)
            .field("nodes", &nodes)
            .finish()
    }
}
