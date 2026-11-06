//! The immutable graph: an arena of nodes in creation (and therefore topological) order.

use crate::error::{GraphError, Result};
use crate::op::Op;
use crate::value::{DType, Value};
use keanu_tensor::Shape;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::fmt;
use std::ops::Index;
use std::sync::OnceLock;

/// Identifies a node within its graph. Ids are assigned in creation order and a node's parents always have smaller
/// ids, so ascending id order is a topological order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub(crate) usize);

impl NodeId {
    /// The position of the node in its graph.
    pub fn index(self) -> usize {
        self.0
    }

    /// The id of the node at a position (creation index) in a graph or builder. Nothing checks that the node exists
    /// until the id is used: methods taking ids return [`GraphError::UnknownNode`] (or `None` for lookups) for an
    /// id out of range.
    ///
    /// [`GraphError::UnknownNode`]: crate::GraphError::UnknownNode
    pub fn from_index(index: usize) -> Self {
        Self(index)
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
    /// The operation the node performs.
    pub fn op(&self) -> &Op {
        &self.op
    }
    /// The node's parents, in order.
    pub fn parents(&self) -> &[NodeId] {
        &self.parents
    }
    /// The element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    /// The shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Labels the node.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
    /// Whether the node is probabilistic.
    pub fn is_probabilistic(&self) -> bool {
        self.op.is_probabilistic()
    }
    /// Whether the node is a factor, a term of the joint log probability (see [`Op::Factor`]).
    pub fn is_factor(&self) -> bool {
        self.op.is_factor()
    }
    /// Whether the node is a simulator, a probabilistic node without a likelihood (see
    /// [`Simulator`](crate::op::Simulator)).
    pub fn is_simulator(&self) -> bool {
        self.op.is_simulator()
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
    /// Per node, auto diff's index map of a shape op, computed on first use (see [`Graph::index_map`]).
    index_maps: Vec<OnceLock<IndexMap>>,
}

/// For each output element of a shape op, the (parent, flat index) it copies, or `None` for a zero.
pub(crate) type IndexMap = Box<[Option<(usize, usize)>]>;

impl Graph {
    pub(crate) fn new(nodes: Vec<Node>, labels: HashMap<String, NodeId>) -> Self {
        let mut children: Vec<Parents> = vec![Parents::new(); nodes.len()];
        for (i, node) in nodes.iter().enumerate() {
            for &p in &node.parents {
                // children are pushed in increasing id order, so a duplicate parent (e.g. x * x) can only repeat the
                // last entry: checking it keeps this linear in fan-out, where `contains` was quadratic
                if children[p.0].last() != Some(&NodeId(i)) {
                    children[p.0].push(NodeId(i));
                }
            }
        }
        Self {
            index_maps: std::iter::repeat_with(OnceLock::new)
                .take(nodes.len())
                .collect(),
            nodes,
            children,
            labels,
        }
    }

    /// The index map of a shape op node (reshape, slice, concat, ...), given its parents' values: which parent
    /// element each output element copies. It depends only on the op and its parents' shapes, which are fixed, so
    /// it is computed once per node and cached.
    pub(crate) fn index_map(
        &self,
        id: NodeId,
        parents: &[&Value],
    ) -> Result<&[Option<(usize, usize)>]> {
        let cache = &self.index_maps[id.0];
        if let Some(map) = cache.get() {
            return Ok(map);
        }
        let map = crate::autodiff::index_map(&self.nodes[id.0].op, parents)?;
        Ok(cache.get_or_init(|| map.into_boxed_slice()))
    }

    /// The number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether there are no elements.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The node with this id, or `None` if it is not a node of this graph. `graph[id]` is the unchecked
    /// (panicking) form, for ids known to be the graph's own.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.0)
    }

    /// Whether the id is a node of this graph.
    pub fn contains(&self, id: NodeId) -> bool {
        id.0 < self.nodes.len()
    }

    /// [`GraphError::UnknownNode`] for the first id that isn't a node of this graph.
    pub(crate) fn check_ids<'a>(&self, ids: impl IntoIterator<Item = &'a NodeId>) -> Result<()> {
        match ids.into_iter().find(|id| !self.contains(**id)) {
            Some(&id) => Err(GraphError::UnknownNode(id)),
            None => Ok(()),
        }
    }

    /// Every node id in topological order.
    pub fn ids(&self) -> impl DoubleEndedIterator<Item = NodeId> + ExactSizeIterator + '_ {
        (0..self.nodes.len()).map(NodeId)
    }

    /// The node's parents, in order, or `None` if it is not a node of this graph.
    pub fn parents(&self, id: NodeId) -> Option<&[NodeId]> {
        self.node(id).map(Node::parents)
    }

    /// The nodes whose parents include this node, in id order, or `None` if it is not a node of this graph.
    pub fn children(&self, id: NodeId) -> Option<&[NodeId]> {
        self.children.get(id.0).map(|c| &c[..])
    }

    /// The node's children, for an id known to be a node of this graph.
    pub(crate) fn children_of(&self, id: NodeId) -> &[NodeId] {
        &self.children[id.0]
    }

    /// The node with this label.
    pub fn find(&self, label: &str) -> Option<NodeId> {
        self.labels.get(label).copied()
    }

    /// The probabilistic nodes in topological order.
    pub fn probabilistic(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.ids().filter(|&id| self[id].is_probabilistic())
    }

    /// The factor nodes (see [`Op::Factor`]) in topological order.
    pub fn factors(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.ids().filter(|&id| self[id].is_factor())
    }

    /// How error messages name a node: its label, quoted, or `#` and its index if it has none (or is not a node of
    /// this graph).
    ///
    /// ```
    /// use keanu_graph::GraphBuilder;
    ///
    /// let b = GraphBuilder::new();
    /// let mu = b.gaussian(0.0, 1.0).label("mu");
    /// let x = b.gaussian(mu, 1.0);
    /// let (mu, x) = (mu.id(), x.id());
    /// let (graph, _state) = b.build()?;
    /// assert_eq!(graph.describe(mu), "\"mu\"");
    /// assert_eq!(graph.describe(x), format!("#{}", x.index()));
    /// # Ok::<(), keanu_graph::GraphError>(())
    /// ```
    pub fn describe(&self, id: NodeId) -> String {
        match self.node(id).and_then(Node::label) {
            Some(label) => format!("{label:?}"),
            None => format!("#{}", id.0),
        }
    }
}

/// The node with this id, like slice indexing: for ids known to be nodes of this graph, such as those from
/// [`Graph::ids`], [`Graph::probabilistic`], a node's [`parents`](Node::parents) or the builder that made the graph.
/// [`Graph::node`] is the checked lookup.
///
/// # Panics
///
/// If the id is not a node of this graph.
impl Index<NodeId> for Graph {
    type Output = Node;

    fn index(&self, id: NodeId) -> &Node {
        match self.nodes.get(id.0) {
            Some(node) => node,
            None => panic!("{}", GraphError::UnknownNode(id)),
        }
    }
}
