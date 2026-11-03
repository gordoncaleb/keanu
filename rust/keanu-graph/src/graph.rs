//! The immutable graph: an arena of nodes in creation (and therefore topological) order.

use crate::op::Op;
use crate::value::DType;
use keanu_tensor::Shape;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::fmt;

/// Identifies a node within its graph. Ids are assigned in creation order and a node's parents always have smaller
/// ids, so ascending id order is a topological order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub(crate) usize);

impl NodeId {
    /// The position of the node in its graph.
    pub fn index(self) -> usize {
        self.0
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

pub(crate) type Parents = SmallVec<[NodeId; 4]>;

/// One node: its operation, parents, value type and shape, and optional label.
#[derive(Debug, Clone)]
pub struct Node {
    pub(crate) op: Op,
    pub(crate) parents: Parents,
    pub(crate) dtype: DType,
    pub(crate) shape: Shape,
    pub(crate) label: Option<String>,
}

impl Node {
    pub fn op(&self) -> &Op {
        &self.op
    }
    pub fn parents(&self) -> &[NodeId] {
        &self.parents
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
    pub fn is_probabilistic(&self) -> bool {
        self.op.is_probabilistic()
    }
}

/// A computation graph. Immutable once built, and `Send + Sync`, so one graph can be shared by many [`State`]s
/// (chains, particles, optimizer runs).
///
/// [`State`]: crate::State
#[derive(Debug, Clone)]
pub struct Graph {
    pub(crate) nodes: Vec<Node>,
    pub(crate) children: Vec<Parents>,
    pub(crate) labels: HashMap<String, NodeId>,
}

impl Graph {
    pub(crate) fn new(nodes: Vec<Node>, labels: HashMap<String, NodeId>) -> Self {
        let mut children: Vec<Parents> = vec![Parents::new(); nodes.len()];
        for (i, node) in nodes.iter().enumerate() {
            for &p in &node.parents {
                if !children[p.0].contains(&NodeId(i)) {
                    children[p.0].push(NodeId(i));
                }
            }
        }
        Self {
            nodes,
            children,
            labels,
        }
    }

    /// The number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0]
    }

    /// Every node id in topological order.
    pub fn ids(&self) -> impl DoubleEndedIterator<Item = NodeId> + ExactSizeIterator + '_ {
        (0..self.nodes.len()).map(NodeId)
    }

    pub fn parents(&self, id: NodeId) -> &[NodeId] {
        &self.nodes[id.0].parents
    }

    pub fn children(&self, id: NodeId) -> &[NodeId] {
        &self.children[id.0]
    }

    /// The node with this label.
    pub fn find(&self, label: &str) -> Option<NodeId> {
        self.labels.get(label).copied()
    }

    /// The probabilistic nodes in topological order.
    pub fn probabilistic(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.ids().filter(|&id| self.node(id).is_probabilistic())
    }
}
