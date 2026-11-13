//! A fast path for scans whose body is all scalars: the body compiled once into a list of instructions over one
//! `f64` register per body node, evaluated and differentiated without building tensors or states.
//!
//! It computes exactly what the general path does (the same element functions and derivatives, in the same
//! order), including the reverse sweep's rules: only nodes that receive a cotangent are visited, a node's
//! cotangents are summed in decreasing order of the nodes they come from, and a node with a nonfinite value whose
//! cotangent is exactly zero (a masked-out branch) sends zeros to its parents rather than 0 × ∞.

use super::eval::{binary_f64, unary_f64};
use super::{BinaryOp, Op, UnaryOp};
use crate::autodiff::{binary_derivatives_scalar, unary_derivative_scalar};
use crate::graph::{Graph, NodeId};
use keanu_tensor::DoubleTensor;

/// One body node's computation, writing register `out`.
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

/// A scan body of scalars, compiled.
#[derive(Clone, Debug)]
pub(super) struct ScalarTape {
    instructions: Vec<Instruction>,
    /// The registers holding the body's constant nodes' values, and those values.
    constants: Vec<(usize, f64)>,
    registers: usize,
}

impl ScalarTape {
    /// The tape of a body if every node is an f64 scalar computed by a differentiable unary or binary op, a proxy
    /// or a constant, and every one of `step_nodes` is differentiable; None otherwise.
    pub(super) fn compile(graph: &Graph, step_nodes: &[NodeId]) -> Option<Self> {
        let mut constants = Vec::new();
        for id in graph.ids() {
            let node = &graph[id];
            if node.dtype != crate::value::DType::F64 || !node.shape.is_empty() {
                return None;
            }
            if let Op::Constant(value) = &node.op {
                let crate::value::Value::F64(t) = value else {
                    return None;
                };
                constants.push((id.0, *t.as_slice().first()?));
            }
        }
        let mut instructions = Vec::with_capacity(step_nodes.len());
        for &id in step_nodes {
            let node = &graph[id];
            let p = |i: usize| node.parents.get(i).map(|p| p.0);
            instructions.push(match &node.op {
                Op::Unary(op) => {
                    // the general reverse sweep stops at these (they aren't differentiable)
                    if matches!(
                        op,
                        UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round | UnaryOp::Sign
                    ) || node.parents.len() != 1
                    {
                        return None;
                    }
                    Instruction::Unary {
                        op: *op,
                        x: p(0)?,
                        out: id.0,
                    }
                }
                Op::Binary(op) => {
                    if matches!(op, BinaryOp::Rem) || node.parents.len() != 2 {
                        return None;
                    }
                    Instruction::Binary {
                        op: *op,
                        l: p(0)?,
                        r: p(1)?,
                        out: id.0,
                    }
                }
                Op::Proxy if node.parents.len() == 1 => Instruction::Copy {
                    x: p(0)?,
                    out: id.0,
                },
                _ => return None,
            });
        }
        Some(Self {
            instructions,
            constants,
            registers: graph.len(),
        })
    }

    /// Fresh registers, holding the body's constants.
    pub(super) fn registers(&self) -> Vec<f64> {
        let mut registers = vec![0.0; self.registers];
        for &(r, v) in &self.constants {
            registers[r] = v;
        }
        registers
    }

    /// Runs one step: the inputs must already be in their registers.
    pub(super) fn forward(&self, registers: &mut [f64]) {
        for instruction in &self.instructions {
            match *instruction {
                Instruction::Unary { op, x, out } => {
                    registers[out] = unary(op, registers[x]);
                }
                Instruction::Binary { op, l, r, out } => {
                    registers[out] = binary(op, registers[l], registers[r]);
                }
                Instruction::Copy { x, out } => registers[out] = registers[x],
            }
        }
    }

    /// The reverse sweep of one step, after [`forward`](Self::forward) at that step: from the cotangent `seed` at
    /// register `output`, accumulates cotangents into `adjoints` (with `touched` marking the registers that
    /// received one). Instructions write to non-input registers only, so the inputs' cotangents are left in
    /// `adjoints` for the caller; every other register's slot is cleared as it is visited.
    pub(super) fn reverse(
        &self,
        registers: &[f64],
        adjoints: &mut [f64],
        touched: &mut [bool],
        output: usize,
        seed: f64,
    ) {
        adjoints[output] = seed;
        touched[output] = true;
        for instruction in self.instructions.iter().rev() {
            let out = match *instruction {
                Instruction::Unary { out, .. }
                | Instruction::Binary { out, .. }
                | Instruction::Copy { out, .. } => out,
            };
            if !std::mem::take(&mut touched[out]) {
                continue;
            }
            let g = std::mem::take(&mut adjoints[out]);
            let y = registers[out];
            let masked = !y.is_finite() && g == 0.0;
            match *instruction {
                Instruction::Unary { op, x, .. } => {
                    let d = if masked {
                        0.0
                    } else {
                        g * unary_derivative_scalar(op, registers[x], y).unwrap_or(f64::NAN)
                    };
                    add(adjoints, touched, x, d);
                }
                Instruction::Binary { op, l, r, .. } => {
                    let (dl, dr) = if masked {
                        (0.0, 0.0)
                    } else {
                        let (dl, dr) = binary_derivatives_scalar(op, registers[l], registers[r], y)
                            .unwrap_or((f64::NAN, f64::NAN));
                        (g * dl, g * dr)
                    };
                    add(adjoints, touched, l, dl);
                    add(adjoints, touched, r, dr);
                }
                Instruction::Copy { x, .. } => {
                    add(adjoints, touched, x, if masked { 0.0 } else { g })
                }
            }
        }
    }
}

fn add(adjoints: &mut [f64], touched: &mut [bool], at: usize, value: f64) {
    if touched[at] {
        adjoints[at] += value;
    } else {
        adjoints[at] = value;
        touched[at] = true;
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
