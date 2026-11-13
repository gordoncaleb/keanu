//! A fast path for small nested bodies (the right-hand side of an ODE): the body compiled once into scalar
//! instructions over one `f64` register per element of each body node, evaluated, with batched forward tangents,
//! without building tensors or states.
//!
//! Element-wise unary and binary ops become one instruction per output element (with broadcasting), and the
//! ops that only move elements (`take`, `stack`, `concat`, `slice`, `reshape`, permutations and broadcasts)
//! become copies: which element goes where is found by applying the op itself to tensors of register numbers. The
//! values and tangents are those of the general path: the same element functions and derivatives, combined in the
//! same order, and a node has a tangent exactly when the general forward sweep gives it one.

use super::eval::{binary_f64, eval, unary_f64};
use super::{BinaryOp, Op, UnaryOp};
use crate::autodiff::{binary_derivatives_scalar, unary_derivative_scalar};
use crate::graph::{Graph, NodeId};
use crate::value::{DType, Value};
use keanu_tensor::DoubleTensor;

/// At most this many registers (elements over every body node): beyond it the general path is as good.
const MAX_REGISTERS: usize = 4096;

/// One element's computation, writing register `out`.
#[derive(Clone, Copy, Debug)]
enum Instruction {
    Unary {
        op: UnaryOp,
        x: usize,
        out: usize,
    },
    Binary {
        op: BinaryOp,
        l: usize,
        r: usize,
        out: usize,
    },
    Copy {
        x: usize,
        out: usize,
    },
}

/// The instructions computing one node's elements, and its parents (for which nodes have tangents).
#[derive(Clone, Debug)]
struct Step {
    node: usize,
    parents: Vec<usize>,
    instructions: std::ops::Range<usize>,
    /// Whether the node is a unary or binary op, whose tangent comes from its parents' tangents term by term (a
    /// parent without one contributes nothing), rather than a copy of elements.
    elementwise: bool,
}

/// A small body, compiled.
#[derive(Clone, Debug)]
pub(super) struct ElementTape {
    instructions: Vec<Instruction>,
    steps: Vec<Step>,
    /// The registers of the body's constant nodes, with their values.
    constants: Vec<(usize, f64)>,
    /// Each node's first register.
    offsets: Vec<usize>,
    registers: usize,
}

impl ElementTape {
    /// The tape of a body whose nodes are all f64, with at most [`MAX_REGISTERS`] elements in all, and are each an
    /// input (one of `inputs`, filled in by the caller), a constant, a differentiable element-wise unary or binary
    /// op, a connected proxy or an op that only moves elements. None otherwise.
    pub(super) fn compile(graph: &Graph, inputs: &[NodeId]) -> Option<Self> {
        let mut offsets = Vec::with_capacity(graph.len());
        let mut registers = 0usize;
        for id in graph.ids() {
            let node = &graph[id];
            if node.dtype != DType::F64 {
                return None;
            }
            offsets.push(registers);
            registers = registers.checked_add(node.shape.iter().product::<usize>())?;
            if registers > MAX_REGISTERS {
                return None;
            }
        }
        let numbers = |id: NodeId| -> Option<DoubleTensor> {
            let shape = graph[id].shape();
            let start = offsets[id.0];
            DoubleTensor::new(
                shape,
                (0..shape.iter().product::<usize>())
                    .map(|i| (start + i) as f64)
                    .collect(),
            )
            .ok()
        };
        let mut tape = Self {
            instructions: Vec::new(),
            steps: Vec::new(),
            constants: Vec::new(),
            offsets: offsets.clone(),
            registers,
        };
        for id in graph.ids() {
            if inputs.contains(&id) {
                continue;
            }
            let node = &graph[id];
            let out = offsets[id.0];
            let len: usize = node.shape.iter().product();
            let start = tape.instructions.len();
            let mut elementwise = false;
            match &node.op {
                Op::Constant(Value::F64(t)) => {
                    if t.shape() != node.shape() {
                        return None;
                    }
                    tape.constants
                        .extend(t.as_slice().iter().enumerate().map(|(i, &v)| (out + i, v)));
                    continue;
                }
                Op::Unary(op) => {
                    if matches!(
                        op,
                        UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round | UnaryOp::Sign
                    ) || node.parents.len() != 1
                        || graph[node.parents[0]].shape() != node.shape()
                    {
                        return None;
                    }
                    let x = offsets[node.parents[0].0];
                    tape.instructions
                        .extend((0..len).map(|i| Instruction::Unary {
                            op: *op,
                            x: x + i,
                            out: out + i,
                        }));
                    elementwise = true;
                }
                Op::Binary(op) => {
                    if matches!(op, BinaryOp::Rem) || node.parents.len() != 2 {
                        return None;
                    }
                    // each output element's operands: the parents' register numbers broadcast to the output
                    let l = numbers(node.parents[0])?.broadcast_to(node.shape()).ok()?;
                    let r = numbers(node.parents[1])?.broadcast_to(node.shape()).ok()?;
                    tape.instructions.extend(
                        l.as_slice()
                            .iter()
                            .zip(r.as_slice())
                            .enumerate()
                            .map(|(i, (&l, &r))| Instruction::Binary {
                                op: *op,
                                l: l as usize,
                                r: r as usize,
                                out: out + i,
                            }),
                    );
                    elementwise = true;
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
                    if matches!(node.op, Op::Proxy) && node.parents.len() != 1 {
                        return None;
                    }
                    let parents: Vec<Value> = node
                        .parents
                        .iter()
                        .map(|&p| numbers(p).map(Value::F64))
                        .collect::<Option<_>>()?;
                    let moved = match &node.op {
                        Op::Proxy => parents[0].clone(),
                        op => eval(op, &parents.iter().collect::<Vec<_>>()).ok()?,
                    };
                    let Value::F64(moved) = moved else {
                        return None;
                    };
                    if moved.shape() != node.shape() {
                        return None;
                    }
                    tape.instructions
                        .extend(moved.as_slice().iter().enumerate().map(|(i, &x)| {
                            Instruction::Copy {
                                x: x as usize,
                                out: out + i,
                            }
                        }));
                }
                _ => return None,
            }
            tape.steps.push(Step {
                node: id.0,
                parents: node.parents.iter().map(|p| p.0).collect(),
                instructions: start..tape.instructions.len(),
                elementwise,
            });
        }
        Some(tape)
    }

    /// Fresh registers, holding the body's constants.
    pub(super) fn registers(&self) -> Vec<f64> {
        let mut registers = vec![0.0; self.registers];
        for &(r, v) in &self.constants {
            registers[r] = v;
        }
        registers
    }

    /// The registers of a node's elements.
    pub(super) fn range(&self, graph: &Graph, id: NodeId) -> std::ops::Range<usize> {
        let start = self.offsets[id.0];
        start..start + graph[id].shape().iter().product::<usize>()
    }

    /// Computes every node from the inputs, which must already be in their registers.
    pub(super) fn forward(&self, registers: &mut [f64]) {
        for instruction in &self.instructions {
            match *instruction {
                Instruction::Unary { op, x, out } => registers[out] = unary(op, registers[x]),
                Instruction::Binary { op, l, r, out } => {
                    registers[out] = binary(op, registers[l], registers[r]);
                }
                Instruction::Copy { x, out } => registers[out] = registers[x],
            }
        }
    }

    /// Batched forward tangents along `b` directions, after [`forward`](Self::forward): `tangents` holds `b`
    /// values per register (register-major), with the seeded nodes' tangents in place and zeros elsewhere, and
    /// `has` marks the seeded nodes. Fills in every other node's tangent and marks the nodes that have one.
    pub(super) fn tangents(
        &self,
        registers: &[f64],
        tangents: &mut [f64],
        has: &mut [bool],
        b: usize,
    ) {
        for step in &self.steps {
            if has[step.node] {
                // seeded
                continue;
            }
            if !step.parents.iter().any(|&p| has[p]) {
                continue;
            }
            has[step.node] = true;
            for instruction in &self.instructions[step.instructions.clone()] {
                match *instruction {
                    Instruction::Unary { op, x, out } => {
                        let d = unary_derivative_scalar(op, registers[x], registers[out])
                            .unwrap_or(f64::NAN);
                        for k in 0..b {
                            tangents[out * b + k] = tangents[x * b + k] * d;
                        }
                    }
                    Instruction::Binary { op, l, r, out } => {
                        let (dl, dr) = binary_derivatives_scalar(
                            op,
                            registers[l],
                            registers[r],
                            registers[out],
                        )
                        .unwrap_or((f64::NAN, f64::NAN));
                        let (from_l, from_r) = (has[step.parents[0]], has[step.parents[1]]);
                        for k in 0..b {
                            let left = tangent_term(op, true, tangents[l * b + k], dl);
                            let right = tangent_term(op, false, tangents[r * b + k], dr);
                            tangents[out * b + k] = match (from_l, from_r) {
                                (true, true) => left + right,
                                (true, false) => left,
                                _ => right,
                            };
                        }
                    }
                    Instruction::Copy { x, out } => {
                        for k in 0..b {
                            tangents[out * b + k] = tangents[x * b + k];
                        }
                    }
                }
            }
            debug_assert!(
                step.elementwise
                    || step
                        .instructions
                        .clone()
                        .all(|i| matches!(self.instructions[i], Instruction::Copy { .. }))
            );
        }
    }
}

/// One operand's term of a binary op's tangent, as the general path forms it: the tangent itself for an operand
/// the op passes through (either side of a sum, the left of a difference), its negation for the right of a
/// difference, and the tangent times the derivative otherwise.
fn tangent_term(op: BinaryOp, left: bool, t: f64, d: f64) -> f64 {
    match (op, left) {
        (BinaryOp::Add, _) | (BinaryOp::Sub, true) => t,
        (BinaryOp::Sub, false) => -t,
        _ => t * d,
    }
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
