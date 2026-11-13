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
    /// The body nodes evaluated at each step, in order: every node that is neither an input nor a constant.
    step_nodes: Vec<NodeId>,
    /// The body compiled, if it is all scalars.
    tape: Option<super::scan_tape::ScalarTape>,
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
                    graph.describe(id)
                )));
            }
            if node.is_factor() {
                return Err(invalid(format!(
                    "the body's node {} is a factor: scan bodies must be deterministic (add factors outside the \
                     scan, on its output)",
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
        let step_nodes: Vec<NodeId> = graph
            .ids()
            .filter(|&id| !is_input[id.0] && !matches!(graph[id].op, Op::Constant(_)))
            .collect();
        let tape = super::scan_tape::ScalarTape::compile(&graph, &step_nodes);
        Ok(Self {
            graph,
            length,
            carry,
            xs,
            constants,
            output,
            step_nodes,
            tape,
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

    /// A body state holding the constants (the constant parents and the body's own constant nodes), which stay the
    /// same at every step: [`step_into`](Self::step_into) fills in the rest.
    fn fresh_state(&self, parents: &[&Value]) -> State {
        let n = self.graph.len();
        let mut values: Vec<Option<Value>> = vec![None; n];
        for (i, &c) in self.constants.iter().enumerate() {
            values[c.0] = Some(parents[1 + self.xs.len() + i].clone());
        }
        for id in self.graph.ids() {
            if let Op::Constant(v) = &self.graph[id].op {
                values[id.0] = Some(v.clone());
            }
        }
        State {
            values,
            observed: vec![false; n],
        }
    }

    /// Evaluates the body at one step into `state` (made by [`fresh_state`](Self::fresh_state) with the same
    /// parents): the step's carry and per-step inputs, then every other node from them and the constants.
    fn step_into(
        &self,
        state: &mut State,
        carry: DoubleTensor,
        parents: &[&Value],
        step: usize,
    ) -> Result<()> {
        let values = &mut state.values;
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
        for &id in &self.step_nodes {
            let node = &self.graph.nodes[id.0];
            let value = {
                let inputs = node
                    .parents
                    .iter()
                    .map(|p| values[p.0].as_ref().ok_or(GraphError::MissingValue(*p)))
                    .collect::<Result<smallvec::SmallVec<[&Value; 4]>>>()?;
                match &node.op {
                    Op::Proxy => inputs[0].clone(),
                    op => eval(op, &inputs)?,
                }
            };
            values[id.0] = Some(value);
        }
        Ok(())
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

    /// Recomputes the body's state at a step into `state`, from the scan's parents and its stacked output.
    fn state_at(
        &self,
        state: &mut State,
        parents: &[&Value],
        out: &DoubleTensor,
        step: usize,
    ) -> Result<()> {
        self.step_into(state, self.carry_at(parents, out, step)?, parents, step)
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
        if let Some(tape) = &self.tape {
            return Ok(Value::F64(DoubleTensor::new(
                &shape,
                self.tape_outputs(tape, parents)?,
            )?));
        }
        let mut data = Vec::with_capacity(total);
        let mut carry = f64_of(parents[0])?.clone();
        let mut state = self.fresh_state(parents);
        for step in 0..self.length {
            self.step_into(&mut state, carry, parents, step)?;
            // the output is recomputed at each step unless it is an input or a constant, which must stay
            let output = if self.step_nodes.contains(&self.output) {
                state.values[self.output.0].take()
            } else {
                state.values[self.output.0].clone()
            };
            let Some(Value::F64(next)) = output else {
                return Err(invalid("the body's output is not f64"));
            };
            data.extend_from_slice(next.as_slice());
            carry = next;
        }
        Ok(Value::F64(DoubleTensor::new(&shape, data)?))
    }

    /// The inputs' scalar values: the per-step inputs as slices of `length`, the constants as one value each.
    fn scalar_inputs<'v>(&self, parents: &[&'v Value]) -> Result<(Vec<&'v [f64]>, Vec<f64>)> {
        let xs = (0..self.xs.len())
            .map(|i| Ok(f64_of(parents[1 + i])?.as_slice()))
            .collect::<Result<Vec<_>>>()?;
        let constants = (0..self.constants.len())
            .map(|i| Ok(f64_of(parents[1 + self.xs.len() + i])?.as_slice()[0]))
            .collect::<Result<Vec<_>>>()?;
        Ok((xs, constants))
    }

    /// Fresh tape registers with the constant parents in place.
    fn tape_registers(&self, tape: &super::scan_tape::ScalarTape, constants: &[f64]) -> Vec<f64> {
        let mut registers = tape.registers();
        for (&c, &v) in self.constants.iter().zip(constants) {
            registers[c.0] = v;
        }
        registers
    }

    /// The stacked outputs of the steps, run on the tape.
    fn tape_outputs(
        &self,
        tape: &super::scan_tape::ScalarTape,
        parents: &[&Value],
    ) -> Result<Vec<f64>> {
        let (xs, constants) = self.scalar_inputs(parents)?;
        let mut registers = self.tape_registers(tape, &constants);
        let mut carry = f64_of(parents[0])?.as_slice()[0];
        let mut data = Vec::with_capacity(self.length);
        for step in 0..self.length {
            registers[self.carry.0] = carry;
            for (x, values) in self.xs.iter().zip(&xs) {
                registers[x.0] = values[step];
            }
            tape.forward(&mut registers);
            carry = registers[self.output.0];
            data.push(carry);
        }
        Ok(data)
    }

    /// Reverse mode on the tape (one cotangent): as [`vjp`](Self::vjp), sweeping the steps backwards and
    /// recomputing each step's registers.
    fn tape_vjp(
        &self,
        tape: &super::scan_tape::ScalarTape,
        parents: &[&Value],
        out: &DoubleTensor,
        g: &DoubleTensor,
    ) -> Result<Vec<Option<DoubleTensor>>> {
        let (xs, constants) = self.scalar_inputs(parents)?;
        let (out, g) = (out.as_slice(), g.as_slice());
        let mut registers = self.tape_registers(tape, &constants);
        let mut adjoints = vec![0.0; registers.len()];
        let mut touched = vec![false; registers.len()];
        let mut xs_grads: Vec<Option<Vec<f64>>> = vec![None; self.xs.len()];
        let mut constant_grads: Vec<Option<f64>> = vec![None; self.constants.len()];
        let mut carry: Option<f64> = None;
        let initial = f64_of(parents[0])?.as_slice()[0];
        for step in (0..self.length).rev() {
            registers[self.carry.0] = if step == 0 { initial } else { out[step - 1] };
            for (x, values) in self.xs.iter().zip(&xs) {
                registers[x.0] = values[step];
            }
            tape.forward(&mut registers);
            let seed = match carry.take() {
                Some(c) => g[step] + c,
                None => g[step],
            };
            tape.reverse(&registers, &mut adjoints, &mut touched, self.output.0, seed);
            // the inputs' cotangents, in decreasing id order as the general sweep emits them
            if std::mem::take(&mut touched[self.carry.0]) {
                carry = Some(std::mem::take(&mut adjoints[self.carry.0]));
            }
            for (i, x) in self.xs.iter().enumerate() {
                if std::mem::take(&mut touched[x.0]) {
                    let slot = xs_grads[i].get_or_insert_with(|| vec![0.0; self.length]);
                    slot[step] = std::mem::take(&mut adjoints[x.0]);
                }
            }
            for (i, c) in self.constants.iter().enumerate() {
                if std::mem::take(&mut touched[c.0]) {
                    let contribution = std::mem::take(&mut adjoints[c.0]);
                    constant_grads[i] = Some(match constant_grads[i] {
                        Some(total) => total + contribution,
                        None => contribution,
                    });
                }
            }
            // the body's own constants may have collected cotangents: they go nowhere
            for slot in touched.iter_mut() {
                *slot = false;
            }
        }
        let mut result = Vec::with_capacity(self.parent_count());
        result.push(carry.map(|c| DoubleTensor::from_fn(&[1], |_| c)));
        for grads in xs_grads {
            result.push(
                grads
                    .map(|d| DoubleTensor::new(&[1, self.length], d))
                    .transpose()?,
            );
        }
        for grads in constant_grads {
            result.push(grads.map(|c| DoubleTensor::from_fn(&[1], |_| c)));
        }
        Ok(result)
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
        let shape = self.batched_out_shape(b);
        let total = super::shape::element_count(&shape).ok_or_else(|| {
            invalid(format!(
                "the batched tangent of shape {shape:?} has too many elements (at most {})",
                super::shape::MAX_ELEMENTS
            ))
        })?;
        let mut result = vec![0.0; total];
        let mut carry = t[0].cloned();
        let mut state = self.fresh_state(parents);
        for step in 0..self.length {
            let seeds = self.step_seeds(carry.take(), t, step, b)?;
            if seeds.is_empty() {
                continue;
            }
            self.state_at(&mut state, parents, out, step)?;
            let mut tangents = forward_tangents(&self.graph, &state, seeds, b)?;
            carry = tangents[self.output.0].take();
            if let Some(c) = &carry {
                put_step(&mut result, b, self.length, step, c.as_slice());
            }
        }
        Ok(Some(DoubleTensor::new(&shape, result)?))
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
        if let (Some(tape), 1) = (&self.tape, b) {
            return self.tape_vjp(tape, parents, out, g);
        }
        let is_input = self.is_input();
        let carry_shape = crate::autodiff::batched_shape(b, self.shape_of(self.carry));
        let mut grads = Grads::new(self, parents, b);
        let mut carry: Option<DoubleTensor> = None;
        let mut state = self.fresh_state(parents);
        for step in (0..self.length).rev() {
            let mut seed = take_step(g, b, self.length, step, &carry_shape)?;
            if let Some(c) = carry.take() {
                seed = seed.add(&c)?;
            }
            self.state_at(&mut state, parents, out, step)?;
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
        let mut state = self.fresh_state(parents);
        for step in (0..self.length).rev() {
            let carry_tangent = match (step, yt) {
                (0, _) => t[0].cloned(),
                (_, Some(yt)) => Some(take_step(yt, b, self.length, step - 1, &carry_b)?),
                (_, None) => None,
            };
            let seeds = self.step_seeds(carry_tangent, t, step, b)?;
            self.state_at(&mut state, parents, out, step)?;
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
    if b * m != shape.iter().product::<usize>() {
        return Err(invalid(format!(
            "a step of {m} elements in {b} blocks doesn't fit shape {shape:?}"
        )));
    }
    // small steps (a scalar carry) are built in place, without allocating
    Ok(DoubleTensor::from_fn(shape, |i| {
        let (k, j) = (i / m.max(1), i % m.max(1));
        data[(k * length + step) * m + j]
    }))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GraphBuilder;
    use std::sync::Arc;

    /// The body of a long scan of a scalar carry, built but never evaluated.
    fn long_scan(length: usize) -> Arc<ScanBody> {
        let b = GraphBuilder::new();
        let x = b
            .scan_steps(length, b.scalar(1.0), &[], |_, x, _| x * 2.0)
            .id();
        let (graph, _) = b.build().unwrap();
        match graph[x].op() {
            Op::Scan(body) => body.clone(),
            op => panic!("expected a scan, got {op:?}"),
        }
    }

    #[test]
    fn the_scalar_tape_matches_the_general_path_exactly() {
        // a scalar body using the tape's fast ops, ops it evaluates through tensors, a body constant, a constant
        // parent and two per-step inputs, one of which the body ignores
        let b = GraphBuilder::new();
        let xs = b.constant(DoubleTensor::vector(
            (0..30).map(|i| (i as f64 * 0.37).sin()).collect(),
        ));
        let unused = b.constant(DoubleTensor::vector(vec![1.0; 30]));
        let rate = b.constant(DoubleTensor::scalar(0.8));
        let h = b
            .scan(b.scalar(0.3), &[xs, unused], &[rate], |b, h, x, c| {
                let shock = x[0];
                (h * c[0] + shock).tanh() + (h.exp() / 3.0).ln_1p()
                    - b.scalar(0.1) * (h * h + 1.0).sqrt()
            })
            .id();
        let (graph, state) = b.build().unwrap();
        let Op::Scan(body) = graph[h].op() else {
            panic!("expected a scan")
        };
        assert!(body.tape.is_some(), "the body should compile");
        let general = ScanBody {
            tape: None,
            ..(**body).clone()
        };
        let parents: Vec<&Value> = graph[h]
            .parents()
            .iter()
            .map(|&p| state.value(p).unwrap())
            .collect();
        let (Value::F64(fast), Value::F64(slow)) = (
            body.eval(&parents).unwrap(),
            general.eval(&parents).unwrap(),
        ) else {
            panic!("expected f64 outputs")
        };
        assert_eq!(fast.as_slice(), slow.as_slice());
        let g =
            DoubleTensor::new(&[1, 30], (0..30).map(|i| 1.0 + (i as f64).cos()).collect()).unwrap();
        let fast = body.vjp(&parents, &fast, &g, 1).unwrap();
        let slow = general.vjp(&parents, &slow, &g, 1).unwrap();
        assert_eq!(fast.len(), slow.len());
        for (f, s) in fast.iter().zip(&slow) {
            match (f, s) {
                (Some(f), Some(s)) => {
                    assert_eq!(f.shape(), s.shape());
                    assert_eq!(f.as_slice(), s.as_slice());
                }
                (None, None) => {}
                _ => panic!("one path has a cotangent the other lacks: {f:?} {s:?}"),
            }
        }
    }

    #[test]
    fn jvp_rejects_a_batched_tangent_too_large_to_allocate() {
        // 2^59 steps of a scalar carry is within the element cap, but the batched tangent [b, 2^59] is not: with 32
        // directions its element count overflows usize, and with 4 it is over the cap
        let body = long_scan(1 << 59);
        let init = Value::F64(DoubleTensor::scalar(1.0));
        let out = DoubleTensor::scalar(0.0);
        for b in [32, 4] {
            let tangent = DoubleTensor::zeros(&[b]);
            let result = body.jvp(&[&init], &out, &[Some(&tangent)], b);
            assert!(
                matches!(result, Err(GraphError::InvalidNode { op: "scan", .. })),
                "{b}: {result:?}"
            );
        }
    }
}
