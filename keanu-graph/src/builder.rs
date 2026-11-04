//! Building graphs with typed handles.

use crate::error::{GraphError, Result};
use crate::graph::{Graph, Node, NodeId, Parents};
use crate::op::{DistributionKind, DistributionOp, Lambda, Op, output_type};
use crate::state::State;
use crate::value::{DType, Value};
use crate::vars::{BoolVar, DoubleVar, IntVar, Var};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor, Shape};
use smallvec::SmallVec;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Default)]
struct Inner {
    nodes: Vec<Node>,
    error: Option<GraphError>,
    values: Vec<Option<Value>>,
    observed: Vec<bool>,
    sequences: usize,
}

/// Builds a [`Graph`] through typed handles ([`DoubleVar`], [`IntVar`], [`BoolVar`]) that support arithmetic
/// operators and methods for every other operation.
///
/// Operators cannot return errors, so the builder records the first invalid node (e.g. incompatible shapes) and
/// [`build`](Self::build) returns it; [`error`](Self::error) shows it earlier.
///
/// ```
/// use keanu_graph::GraphBuilder;
///
/// let b = GraphBuilder::new();
/// let mu = b.gaussian(0.0, 10.0).label("mu");
/// let x = b.gaussian_with_shape(&[3], mu, 1.0);
/// let y = (x * 2.0 + 1.0).exp().sum();
/// let (y, mu) = (y.id(), mu.id());
/// let (graph, _state) = b.build()?;
/// assert_eq!(graph.find("mu"), Some(mu));
/// assert_eq!(graph.node(y).shape(), &[] as &[usize]);
/// # Ok::<(), keanu_graph::GraphError>(())
/// ```
#[derive(Debug, Default)]
pub struct GraphBuilder {
    inner: RefCell<Inner>,
}

impl GraphBuilder {
    /// An empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// The first error recorded while building, if any.
    pub fn error(&self) -> Option<GraphError> {
        self.inner.borrow().error.clone()
    }

    pub(crate) fn record(&self, error: GraphError) {
        let mut inner = self.inner.borrow_mut();
        if inner.error.is_none() {
            inner.error = Some(error);
        }
    }

    fn push(&self, node: Node, value: Option<Value>) -> NodeId {
        let mut inner = self.inner.borrow_mut();
        let id = NodeId(inner.nodes.len());
        inner.nodes.push(node);
        inner.values.push(value);
        inner.observed.push(false);
        id
    }

    pub(crate) fn dtype_shape(&self, id: NodeId) -> (DType, Shape) {
        let inner = self.inner.borrow();
        let node = &inner.nodes[id.0];
        (node.dtype, node.shape.clone())
    }

    /// The shape of a node created so far, or `None` if there is no such node.
    pub fn shape(&self, id: NodeId) -> Option<Vec<usize>> {
        Some(self.inner.borrow().nodes.get(id.0)?.shape.to_vec())
    }

    /// [`GraphError::UnknownNode`] for the first id that isn't a node created so far.
    fn check_ids(&self, ids: &[NodeId]) -> Result<()> {
        let count = self.node_count();
        match ids.iter().find(|id| id.0 >= count) {
            Some(&id) => Err(GraphError::UnknownNode(id)),
            None => Ok(()),
        }
    }

    /// The handle if it belongs to this builder. A handle from another builder would silently refer to whichever
    /// node of this builder has the same id, so instead this records [`GraphError::WrongBuilder`] (failing
    /// [`build`](Self::build)) and returns a placeholder node of this builder so that building can continue.
    pub(crate) fn own<'b, V: Var<'b>>(&'b self, var: V) -> V {
        if !std::ptr::eq(var.builder(), self) {
            self.record(GraphError::WrongBuilder(var.id()));
        } else if var.id().0 >= self.node_count() {
            self.record(GraphError::UnknownNode(var.id()));
        } else {
            return var;
        }
        V::wrap(self, self.placeholder(V::DTYPE))
    }

    /// An error if a parent isn't a node of this builder.
    fn check_parents(&self, op: &'static str, parents: &[NodeId]) -> Result<()> {
        match parents.iter().find(|p| p.0 >= self.node_count()) {
            Some(p) => Err(GraphError::InvalidNode {
                op,
                message: format!("unknown parent {p:?}"),
            }),
            None => Ok(()),
        }
    }

    /// Calls `f` with the type and shape of each of `parents`, which must be nodes of this builder, to infer the
    /// type and shape of a new node. `f` must not change the builder.
    fn with_types<R>(&self, parents: &[NodeId], f: impl FnOnce(&[(DType, &[usize])]) -> R) -> R {
        let inner = self.inner.borrow();
        let types: SmallVec<[(DType, &[usize]); 4]> = parents
            .iter()
            .map(|p| (inner.nodes[p.0].dtype, inner.nodes[p.0].shape.as_slice()))
            .collect();
        f(&types)
    }

    /// Adds a deterministic node, inferring its type and shape. On error nothing is added.
    fn try_add(&self, op: Op, parents: &[NodeId]) -> Result<NodeId> {
        self.check_parents(op.name(), parents)?;
        let (dtype, shape) = self.with_types(parents, |types| output_type(&op, types))?;
        let value = match &op {
            Op::Constant(v) => Some(v.clone()),
            _ => None,
        };
        Ok(self.push(
            Node {
                op,
                parents: Parents::from_slice(parents),
                dtype,
                shape,
                label: None,
            },
            value,
        ))
    }

    /// Adds a deterministic node, inferring its type and shape. On error the error is recorded and a placeholder
    /// node of the expected type is returned so that building can continue.
    pub(crate) fn add(&self, op: Op, parents: &[NodeId], expected: DType) -> NodeId {
        let name = op.name();
        match self.try_add(op, parents) {
            Ok(id) => {
                let (dtype, _) = self.dtype_shape(id);
                if dtype != expected {
                    self.record(GraphError::InvalidNode {
                        op: name,
                        message: format!("produces {dtype:?}, expected {expected:?}"),
                    });
                }
                id
            }
            Err(e) => {
                self.record(e);
                self.placeholder(expected)
            }
        }
    }

    fn placeholder(&self, dtype: DType) -> NodeId {
        let value = Value::ones(dtype, &[]);
        self.push(
            Node {
                op: Op::Constant(value.clone()),
                parents: Parents::new(),
                dtype,
                shape: Shape::new(),
                label: None,
            },
            Some(value),
        )
    }

    pub(crate) fn set_label(&self, id: NodeId, label: &str) {
        self.inner.borrow_mut().nodes[id.0].label = Some(label.to_owned());
    }

    /// Builds the graph and its initial state (constants, and any values set or observed while building).
    /// Fails with the first error recorded while building, or if two nodes share a label.
    pub fn build(self) -> Result<(Graph, State)> {
        let inner = self.inner.into_inner();
        if let Some(e) = inner.error {
            return Err(e);
        }
        let mut labels = HashMap::new();
        for (i, node) in inner.nodes.iter().enumerate() {
            if let Some(label) = &node.label {
                if labels.insert(label.clone(), NodeId(i)).is_some() {
                    return Err(GraphError::DuplicateLabel(label.clone()));
                }
            }
        }
        let graph = Graph::new(inner.nodes, labels);
        let state = State {
            values: inner.values,
            observed: inner.observed,
        };
        Ok((graph, state))
    }

    /// The next sequence identifier: 1 for the first sequence built with this builder, then 2, ...
    pub(crate) fn next_sequence_identifier(&self) -> usize {
        let mut inner = self.inner.borrow_mut();
        inner.sequences += 1;
        inner.sequences
    }

    /// The number of nodes created so far.
    pub fn node_count(&self) -> usize {
        self.inner.borrow().nodes.len()
    }

    /// The type of a node created so far, or `None` if there is no such node.
    pub fn dtype(&self, id: NodeId) -> Option<DType> {
        Some(self.inner.borrow().nodes.get(id.0)?.dtype)
    }

    /// The value of a node created so far if it is a constant.
    pub fn constant_value(&self, id: NodeId) -> Option<Value> {
        match &self.inner.borrow().nodes.get(id.0)?.op {
            Op::Constant(value) => Some(value.clone()),
            _ => None,
        }
    }

    /// The label of a node created so far.
    pub fn label_of(&self, id: NodeId) -> Option<String> {
        self.inner.borrow().nodes.get(id.0)?.label.clone()
    }

    // ----- untyped construction, for loaders and other generic code -----
    //
    // These return errors immediately instead of recording them for `build`, and on error add no node.

    /// Runs `f` (which may record errors) and returns the error it recorded, if any, leaving an error recorded
    /// before the call in place so that `build` still fails with it.
    fn untyped(&self, f: impl FnOnce()) -> Result<()> {
        let pending = self.inner.borrow_mut().error.take();
        f();
        let mut inner = self.inner.borrow_mut();
        match std::mem::replace(&mut inner.error, pending) {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Adds a deterministic node with the given op and parents, inferring its type and shape.
    pub fn add_node(&self, op: Op, parents: &[NodeId]) -> Result<NodeId> {
        if matches!(op, Op::Distribution(_)) {
            return Err(GraphError::InvalidNode {
                op: op.name(),
                message: "use add_distribution for probabilistic nodes".into(),
            });
        }
        self.try_add(op, parents)
    }

    /// Adds a probabilistic node. `shape` is the requested shape (None for the parameters' broadcast shape).
    pub fn add_distribution(
        &self,
        kind: DistributionKind,
        shape: Option<&[usize]>,
        parameters: &[NodeId],
    ) -> Result<NodeId> {
        self.try_distribution(kind, shape, parameters)
    }

    /// Adds an unconnected proxy node of the given type and shape.
    pub fn add_proxy(&self, dtype: DType, shape: &[usize]) -> NodeId {
        self.proxy(dtype, shape)
    }

    /// Connects a proxy node to the (earlier) node it stands for.
    pub fn connect_ids(&self, proxy: NodeId, parent: NodeId) -> Result<()> {
        self.check_ids(&[proxy, parent])?;
        self.untyped(|| self.connect_node(proxy, parent))
    }

    /// Labels a node.
    pub fn set_label_of(&self, id: NodeId, label: &str) -> Result<()> {
        self.check_ids(&[id])?;
        self.set_label(id, label);
        Ok(())
    }

    /// Sets the initial value of a node, checking its type and shape.
    pub fn set_value_of(&self, id: NodeId, value: Value) -> Result<()> {
        self.check_ids(&[id])?;
        self.untyped(|| self.set_node_value(id, value))
    }

    /// Observes a node, checking its type and shape. Only probabilistic and boolean nodes can be observed.
    pub fn observe_of(&self, id: NodeId, value: Value) -> Result<()> {
        self.check_ids(&[id])?;
        self.untyped(|| self.observe_node(id, value))
    }

    // ----- constants -----

    /// A constant f64 node.
    pub fn constant(&self, value: impl Into<DoubleTensor>) -> DoubleVar<'_> {
        DoubleVar::new(
            self,
            self.add(Op::Constant(Value::F64(value.into())), &[], DType::F64),
        )
    }

    /// A handle for an existing f64 node, e.g. one returned by [`lambda`](Self::lambda). Records an error if the
    /// node doesn't exist or is not f64.
    pub fn double_var(&self, id: NodeId) -> DoubleVar<'_> {
        let Some(dtype) = self.dtype(id) else {
            self.record(GraphError::UnknownNode(id));
            return DoubleVar::new(self, self.placeholder(DType::F64));
        };
        if dtype != DType::F64 {
            self.record(GraphError::InvalidValue(format!(
                "node {id:?} is {dtype:?}, not f64"
            )));
        }
        DoubleVar::new(self, id)
    }

    /// A constant f64 scalar node.
    pub fn scalar(&self, value: f64) -> DoubleVar<'_> {
        self.constant(DoubleTensor::scalar(value))
    }

    /// A constant i64 node.
    pub fn int_constant(&self, value: impl Into<IntTensor>) -> IntVar<'_> {
        IntVar::new(
            self,
            self.add(Op::Constant(Value::I64(value.into())), &[], DType::I64),
        )
    }

    /// A constant i64 scalar node.
    pub fn int_scalar(&self, value: i64) -> IntVar<'_> {
        self.int_constant(IntTensor::scalar(value))
    }

    /// A constant bool node.
    pub fn bool_constant(&self, value: impl Into<BoolTensor>) -> BoolVar<'_> {
        BoolVar::new(
            self,
            self.add(Op::Constant(Value::Bool(value.into())), &[], DType::Bool),
        )
    }

    /// A constant bool scalar node.
    pub fn bool_scalar(&self, value: bool) -> BoolVar<'_> {
        self.bool_constant(BoolTensor::scalar(value))
    }

    // ----- initial values -----

    fn check_value(&self, id: NodeId, value: &Value) -> bool {
        let (dtype, shape) = self.dtype_shape(id);
        if value.dtype() != dtype || value.shape() != shape.as_slice() {
            self.record(GraphError::InvalidValue(format!(
                "node {id:?} holds {dtype:?} of shape {shape:?}, got {:?} of shape {:?}",
                value.dtype(),
                value.shape()
            )));
            return false;
        }
        true
    }

    /// Sets the initial value of a node in the built state.
    pub fn set_value<'b>(&'b self, var: impl Var<'b>, value: impl Into<Value>) {
        self.set_node_value(self.own(var).id(), value.into());
    }

    /// Sets the initial value of a node of this builder (unless it is observed), recording an error if the value
    /// has the wrong type or shape.
    fn set_node_value(&self, id: NodeId, value: Value) {
        if self.check_value(id, &value) {
            let mut inner = self.inner.borrow_mut();
            if !inner.observed[id.0] {
                inner.values[id.0] = Some(value);
            }
        }
    }

    /// Observes a node in the built state. Only probabilistic and boolean nodes can be observed.
    pub fn observe<'b>(&'b self, var: impl Var<'b>, value: impl Into<Value>) {
        self.observe_node(self.own(var).id(), value.into());
    }

    /// Observes a node of this builder, recording an error if it can't be observed or the value has the wrong
    /// type or shape.
    fn observe_node(&self, id: NodeId, value: Value) {
        let observable = {
            let inner = self.inner.borrow();
            let node = &inner.nodes[id.0];
            node.is_probabilistic() || node.dtype == DType::Bool
        };
        if !observable {
            self.record(GraphError::NotObservable(format!(
                "node {id:?} is deterministic and numeric, so it cannot be observed"
            )));
            return;
        }
        if self.check_value(id, &value) {
            let mut inner = self.inner.borrow_mut();
            inner.values[id.0] = Some(value);
            inner.observed[id.0] = true;
        }
    }

    // ----- proxies -----

    fn proxy(&self, dtype: DType, shape: &[usize]) -> NodeId {
        self.push(
            Node {
                op: Op::Proxy,
                parents: Parents::new(),
                dtype,
                shape: Shape::from_slice(shape),
                label: None,
            },
            None,
        )
    }

    /// A placeholder for a value that is connected later with [`connect`](Self::connect), e.g. the previous item of a
    /// sequence.
    pub fn double_proxy(&self, shape: &[usize]) -> DoubleVar<'_> {
        DoubleVar::new(self, self.proxy(DType::F64, shape))
    }

    /// An i64 proxy node, whose parent is connected later.
    pub fn int_proxy(&self, shape: &[usize]) -> IntVar<'_> {
        IntVar::new(self, self.proxy(DType::I64, shape))
    }

    /// A bool proxy node, whose parent is connected later.
    pub fn bool_proxy(&self, shape: &[usize]) -> BoolVar<'_> {
        BoolVar::new(self, self.proxy(DType::Bool, shape))
    }

    /// Connects a proxy to the node it stands for, which must already exist (have a smaller id) and have the same
    /// type and shape.
    pub fn connect<'b, V: Var<'b>>(&'b self, proxy: V, parent: V) {
        self.connect_node(self.own(proxy).id(), self.own(parent).id());
    }

    /// Connects a proxy node of this builder to a node of this builder, recording an error if it can't be.
    fn connect_node(&self, proxy: NodeId, parent: NodeId) {
        let error = {
            let inner = self.inner.borrow();
            let node = &inner.nodes[proxy.0];
            let target = &inner.nodes[parent.0];
            if !matches!(node.op, Op::Proxy) {
                Some(format!("node {proxy:?} is not a proxy"))
            } else if !node.parents.is_empty() {
                Some(format!("proxy {proxy:?} is already connected"))
            } else if parent >= proxy {
                Some(format!(
                    "proxy {proxy:?} can only be connected to an earlier node, not {parent:?}"
                ))
            } else if node.dtype != target.dtype || node.shape != target.shape {
                Some(format!(
                    "proxy {proxy:?} has type {:?} and shape {:?} but {parent:?} has {:?} and {:?}",
                    node.dtype, node.shape, target.dtype, target.shape
                ))
            } else {
                None
            }
        };
        match error {
            Some(message) => self.record(GraphError::Proxy(message)),
            None => self.inner.borrow_mut().nodes[proxy.0].parents.push(parent),
        }
    }

    // ----- lambdas, selection -----

    /// A node computed by a function of its parents' values, which must return a value of the declared type and
    /// shape.
    pub fn lambda(
        &self,
        name: &str,
        parents: &[NodeId],
        dtype: DType,
        shape: &[usize],
        function: impl Fn(&[&Value]) -> std::result::Result<Value, String> + Send + Sync + 'static,
    ) -> NodeId {
        let lambda = Lambda {
            name: name.to_owned(),
            function: Arc::new(function),
            jacobian: None,
            dtype,
            shape: shape.to_vec(),
        };
        self.add(Op::Lambda(lambda), parents, dtype)
    }

    /// A differentiable f64 lambda. `jacobian` returns, for each parent, ∂output/∂parent shaped
    /// `[output shape..., parent shape...]`, or `None` for a parent the output doesn't depend on differentiably.
    pub fn lambda_with_jacobian(
        &self,
        name: &str,
        parents: &[NodeId],
        shape: &[usize],
        function: impl Fn(&[&Value]) -> std::result::Result<Value, String> + Send + Sync + 'static,
        jacobian: impl Fn(
            &[&Value],
        )
            -> std::result::Result<Vec<Option<keanu_tensor::DoubleTensor>>, String>
        + Send
        + Sync
        + 'static,
    ) -> NodeId {
        let lambda = Lambda {
            name: name.to_owned(),
            function: Arc::new(function),
            jacobian: Some(Arc::new(jacobian)),
            dtype: DType::F64,
            shape: shape.to_vec(),
        };
        self.add(Op::Lambda(lambda), parents, DType::F64)
    }

    /// The option chosen by a scalar integer selector. Every option must have the same type and shape.
    pub fn multiplexer<'b, V: Var<'b>>(&'b self, selector: IntVar<'b>, options: &[V]) -> V {
        let mut parents = vec![self.own(selector).id()];
        parents.extend(options.iter().map(|&o| self.own(o).id()));
        V::wrap(self, self.add(Op::Multiplexer, &parents, V::DTYPE))
    }

    /// A conditional probability table: the outcome whose key matches the conditions' current values (booleans as
    /// 0 and 1), otherwise the default.
    pub fn cpt<'builder, V: Var<'builder>>(
        &'builder self,
        conditions: &[NodeId],
        cases: &[(Vec<i64>, V)],
        default: V,
    ) -> V {
        let mut parents = conditions.to_vec();
        parents.extend(cases.iter().map(|&(_, v)| self.own(v).id()));
        parents.push(self.own(default).id());
        let keys = cases.iter().map(|(k, _)| k.clone()).collect();
        V::wrap(
            self,
            self.add(
                Op::Cpt {
                    conditions: conditions.len(),
                    keys,
                },
                &parents,
                V::DTYPE,
            ),
        )
    }

    /// Selects element-wise from `then` where the predicate is true and from `otherwise` elsewhere, broadcasting.
    pub fn where_<'builder, V: Var<'builder>>(
        &'builder self,
        predicate: BoolVar<'builder>,
        then: V,
        otherwise: V,
    ) -> V {
        V::wrap(
            self,
            self.add(
                Op::Where,
                &[
                    self.own(predicate).id(),
                    self.own(then).id(),
                    self.own(otherwise).id(),
                ],
                V::DTYPE,
            ),
        )
    }

    /// Joins nodes along an existing axis.
    pub fn concat<'builder, V: Var<'builder>>(&'builder self, axis: isize, parts: &[V]) -> V {
        let parents: Vec<NodeId> = parts.iter().map(|&p| self.own(p).id()).collect();
        V::wrap(self, self.add(Op::Concat(axis), &parents, V::DTYPE))
    }

    // ----- distributions -----

    /// Adds a probabilistic node. On error nothing is added.
    fn try_distribution(
        &self,
        kind: DistributionKind,
        shape: Option<&[usize]>,
        parameters: &[NodeId],
    ) -> Result<NodeId> {
        self.check_parents("distribution", parameters)?;
        let dtype = kind.value_type();
        let shape = self.with_types(parameters, |types| {
            DistributionOp::value_shape(&kind, shape, types)
        })?;
        Ok(self.push(
            Node {
                op: Op::Distribution(DistributionOp {
                    kind,
                    shape: shape.clone(),
                }),
                parents: Parents::from_slice(parameters),
                dtype,
                shape,
                label: None,
            },
            None,
        ))
    }

    /// Adds a probabilistic node. On error the error is recorded and a placeholder node of the distribution's type
    /// is returned so that building can continue.
    pub(crate) fn distribution(
        &self,
        kind: DistributionKind,
        shape: Option<&[usize]>,
        parameters: &[NodeId],
    ) -> NodeId {
        let dtype = kind.value_type();
        self.try_distribution(kind, shape, parameters)
            .unwrap_or_else(|e| {
                self.record(e);
                self.placeholder(dtype)
            })
    }
}
