//! Node creation infers shapes from the parents' shapes alone: it never evaluates an op, so building (or loading)
//! a graph whose shapes are enormous but under the element cap allocates nothing of their size.
//!
//! This binary's allocator refuses any single allocation over [`REFUSE_ABOVE`] bytes, so a regression (shape
//! inference allocating a value of a node's size) aborts the test instead of exhausting the machine's memory, and
//! records the largest request so the test can check that building stayed small.

use keanu_graph::GraphBuilder;
use keanu_graph::op::{CompareOp, Op, ReduceOp};
use keanu_tensor::IntTensor;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Allocations larger than this are refused (1 GiB).
const REFUSE_ABOVE: usize = 1 << 30;

static LARGEST: AtomicUsize = AtomicUsize::new(0);

struct Guard;

// SAFETY: forwards to the system allocator, refusing (returning null for) requests it chooses not to serve
unsafe impl GlobalAlloc for Guard {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        if layout.size() > REFUSE_ABOVE {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller upholds alloc's contract
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        if layout.size() > REFUSE_ABOVE {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller upholds alloc_zeroed's contract
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr was allocated by System with this layout
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LARGEST.fetch_max(new_size, Ordering::Relaxed);
        if new_size > REFUSE_ABOVE {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller upholds realloc's contract
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Guard = Guard;

#[test]
fn building_ops_on_huge_shapes_allocates_nothing_of_their_size() {
    // [2^34, 3] f64 is 384 GiB, within the element cap (2^60)
    let huge = [1usize << 34, 3];
    let b = GraphBuilder::new();
    let x = b.scalar(0.5).id();
    let wide = b.add_node(Op::Broadcast(huge.to_vec()), &[x]).unwrap();
    let row = b
        .add_node(
            Op::Constant(keanu_graph::Value::F64(keanu_tensor::DoubleTensor::vector(
                vec![1.0, 2.0, 3.0],
            ))),
            &[],
        )
        .unwrap();
    let exp = b
        .add_node(Op::Unary(keanu_graph::op::UnaryOp::Exp), &[wide])
        .unwrap();
    let sum = b
        .add_node(Op::Binary(keanu_graph::op::BinaryOp::Add), &[exp, row])
        .unwrap();
    let rows = b
        .add_node(
            Op::Reduce {
                kind: ReduceOp::Sum,
                axes: Some(vec![1]),
            },
            &[sum],
        )
        .unwrap();
    let total = b
        .add_node(
            Op::Reduce {
                kind: ReduceOp::Max,
                axes: None,
            },
            &[rows],
        )
        .unwrap();
    let indices = b
        .add_node(
            Op::Constant(keanu_graph::Value::I64(IntTensor::vector(vec![0, 2]))),
            &[],
        )
        .unwrap();
    let gathered = b.add_node(Op::Gather { axis: 1 }, &[sum, indices]).unwrap();
    let softmax = b.add_node(Op::Softmax { axis: 0 }, &[gathered]).unwrap();
    let cumsum = b
        .add_node(
            Op::Cumulative {
                kind: keanu_graph::op::CumulativeOp::CumSum,
                axis: 0,
            },
            &[softmax],
        )
        .unwrap();
    let positive = b
        .add_node(Op::Compare(CompareOp::Gt), &[cumsum, x])
        .unwrap();
    let chosen = b.add_node(Op::Where, &[positive, cumsum, softmax]).unwrap();
    let flat = b.add_node(Op::Reshape(vec![-1]), &[chosen]).unwrap();
    let sliced = b.add_node(Op::Slice("::2".into()), &[flat]).unwrap();
    let joined = b.add_node(Op::Concat(0), &[sliced, sliced]).unwrap();
    let stacked = b.add_node(Op::Stack(0), &[joined, joined]).unwrap();
    let transposed = b.add_node(Op::Transpose, &[stacked]).unwrap();
    let cast = b
        .add_node(Op::Cast(keanu_graph::DType::I64), &[transposed])
        .unwrap();
    assert_eq!(b.shape(total), Some(vec![]));
    assert_eq!(b.shape(cast), Some(vec![1 << 35, 2]));
    let (graph, _) = b.build().unwrap();
    assert_eq!(graph.len(), 19);
    let largest = LARGEST.load(Ordering::Relaxed);
    assert!(
        largest < 1 << 20,
        "building allocated {largest} bytes at once"
    );
}
