//! Building graphs with typed handles.

use crate::error::{GraphError, Result};
use crate::graph::{Graph, Node, NodeId, Parents};
use crate::op::{DistributionKind, DistributionOp, Lambda, Op, output_type};
use crate::state::State;
use crate::value::{DType, Value};
use crate::vars::{BoolVar, DoubleVar, IntVar, Var};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor, Shape};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Default)]
struct Inner {
    nodes: Vec<Node>,
    error: Option<GraphError>,
    values: Vec<Option<Value>>,
    observed: Vec<bool>,
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
    pub fn new() -> Self {
        Self::default()
    }

    /// The first error recorded while building, if any.
    pub fn error(&self) -> Option<GraphError> {
        self.inner.borrow().error.clone()
    }

    fn record(&self, error: GraphError) {
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

    /// The shape of a node created so far.
    pub fn shape(&self, id: NodeId) -> Vec<usize> {
        self.dtype_shape(id).1.to_vec()
    }

    /// Adds a deterministic node, inferring its type and shape. On error the error is recorded and a placeholder
    /// node of the expected type is returned so that building can continue.
    pub(crate) fn add(&self, op: Op, parents: &[NodeId], expected: DType) -> NodeId {
        let types: Vec<(DType, Shape)> = parents.iter().map(|&p| self.dtype_shape(p)).collect();
        let refs: Vec<(DType, &[usize])> = types.iter().map(|(t, s)| (*t, s.as_slice())).collect();
        match output_type(&op, &refs) {
            Ok((dtype, shape)) => {
                if dtype != expected {
                    self.record(GraphError::InvalidNode {
                        op: op.name(),
                        message: format!("produces {dtype:?}, expected {expected:?}"),
                    });
                }
                let value = match &op {
                    Op::Constant(v) => Some(v.clone()),
                    _ => None,
                };
                self.push(
                    Node {
                        op,
                        parents: Parents::from_slice(parents),
                        dtype,
                        shape,
                        label: None,
                    },
                    value,
                )
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
            if let Some(label) = &node.label
                && labels.insert(label.clone(), NodeId(i)).is_some()
            {
                return Err(GraphError::DuplicateLabel(label.clone()));
            }
        }
        let graph = Graph::new(inner.nodes, labels);
        let state = State {
            values: inner.values,
            observed: inner.observed,
        };
        Ok((graph, state))
    }

    // ----- constants -----

    pub fn constant(&self, value: impl Into<DoubleTensor>) -> DoubleVar<'_> {
        DoubleVar::new(
            self,
            self.add(Op::Constant(Value::F64(value.into())), &[], DType::F64),
        )
    }

    /// A handle for an existing f64 node, e.g. one returned by [`lambda`](Self::lambda). Records an error if the
    /// node is not f64.
    pub fn double_var(&self, id: NodeId) -> DoubleVar<'_> {
        let (dtype, _) = self.dtype_shape(id);
        if dtype != DType::F64 {
            self.record(GraphError::InvalidValue(format!(
                "node {id:?} is {dtype:?}, not f64"
            )));
        }
        DoubleVar::new(self, id)
    }

    pub fn scalar(&self, value: f64) -> DoubleVar<'_> {
        self.constant(DoubleTensor::scalar(value))
    }

    pub fn int_constant(&self, value: impl Into<IntTensor>) -> IntVar<'_> {
        IntVar::new(
            self,
            self.add(Op::Constant(Value::I64(value.into())), &[], DType::I64),
        )
    }

    pub fn int_scalar(&self, value: i64) -> IntVar<'_> {
        self.int_constant(IntTensor::scalar(value))
    }

    pub fn bool_constant(&self, value: impl Into<BoolTensor>) -> BoolVar<'_> {
        BoolVar::new(
            self,
            self.add(Op::Constant(Value::Bool(value.into())), &[], DType::Bool),
        )
    }

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
        let value = value.into();
        if self.check_value(var.id(), &value) {
            let mut inner = self.inner.borrow_mut();
            if !inner.observed[var.id().0] {
                inner.values[var.id().0] = Some(value);
            }
        }
    }

    /// Observes a node in the built state. Only probabilistic and boolean nodes can be observed.
    pub fn observe<'b>(&'b self, var: impl Var<'b>, value: impl Into<Value>) {
        let id = var.id();
        let value = value.into();
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

    pub fn int_proxy(&self, shape: &[usize]) -> IntVar<'_> {
        IntVar::new(self, self.proxy(DType::I64, shape))
    }

    pub fn bool_proxy(&self, shape: &[usize]) -> BoolVar<'_> {
        BoolVar::new(self, self.proxy(DType::Bool, shape))
    }

    /// Connects a proxy to the node it stands for, which must already exist (have a smaller id) and have the same
    /// type and shape.
    pub fn connect<'b, V: Var<'b>>(&'b self, proxy: V, parent: V) {
        let (proxy, parent) = (proxy.id(), parent.id());
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
        function: impl Fn(&[Value]) -> std::result::Result<Value, String> + Send + Sync + 'static,
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
        function: impl Fn(&[Value]) -> std::result::Result<Value, String> + Send + Sync + 'static,
        jacobian: impl Fn(
            &[Value],
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
        let mut parents = vec![selector.id()];
        parents.extend(options.iter().map(Var::id));
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
        parents.extend(cases.iter().map(|(_, v)| v.id()));
        parents.push(default.id());
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
                &[predicate.id(), then.id(), otherwise.id()],
                V::DTYPE,
            ),
        )
    }

    /// Joins nodes along an existing axis.
    pub fn concat<'builder, V: Var<'builder>>(&'builder self, axis: isize, parts: &[V]) -> V {
        let parents: Vec<NodeId> = parts.iter().map(Var::id).collect();
        V::wrap(self, self.add(Op::Concat(axis), &parents, V::DTYPE))
    }

    // ----- distributions -----

    pub(crate) fn distribution(
        &self,
        kind: DistributionKind,
        shape: Option<&[usize]>,
        parameters: &[NodeId],
    ) -> NodeId {
        let types: Vec<(DType, Shape)> = parameters.iter().map(|&p| self.dtype_shape(p)).collect();
        let refs: Vec<(DType, &[usize])> = types.iter().map(|(t, s)| (*t, s.as_slice())).collect();
        let dtype = kind.value_type();
        match DistributionOp::value_shape(&kind, shape, &refs) {
            Ok(shape) => self.push(
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
            ),
            Err(e) => {
                self.record(e);
                self.placeholder(dtype)
            }
        }
    }
}
