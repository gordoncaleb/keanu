//! Building graphs with typed handles.

use crate::error::{GraphError, Result};
use crate::graph::{Graph, Node, NodeId, Parents};
use crate::op::{DistributionKind, DistributionOp, Lambda, Op, output_type};
use crate::state::State;
use crate::value::{DType, Value};
use crate::vars::{BoolVar, DoubleVar, IntVar, IntoDoubleVar, Var};
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
/// assert_eq!(graph[y].shape(), &[] as &[usize]);
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

    /// Adds an unconnected proxy node of the given type and shape. Fails, adding nothing, if the shape has more
    /// elements than any node may have.
    pub fn add_proxy(&self, dtype: DType, shape: &[usize]) -> Result<NodeId> {
        check_shape("proxy", shape)?;
        Ok(self.proxy(dtype, shape))
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

    /// Adds an unconnected proxy node. If the shape has too many elements the error is recorded and a scalar proxy
    /// is added instead so that building can continue.
    pub(crate) fn proxy(&self, dtype: DType, shape: &[usize]) -> NodeId {
        let shape = match check_shape("proxy", shape) {
            Ok(()) => shape,
            Err(e) => {
                self.record(e);
                &[]
            }
        };
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

    // ----- factors and log densities -----

    /// A factor node: a term of the joint log probability, like PyMC's `Potential` or Stan's `target +=`. The node's
    /// value is the scalar sum of `expr`, and [`Model::log_prob`](crate::Model::log_prob) adds it to the log
    /// probabilities of the probabilistic nodes. A factor whose value is not finite (NaN or ±∞) makes the point
    /// impossible.
    ///
    /// Factors count in the log likelihood ([`Model::log_likelihood`](crate::Model::log_likelihood)) and not in the
    /// log prior. Their gradients and Hessian-vector products are those of `expr`, through the ordinary automatic
    /// differentiation, so every gradient based algorithm (NUTS, optimizers, variational inference, the Laplace
    /// approximation) and Metropolis–Hastings (which updates the log probability incrementally) respect them.
    /// Forward sampling, predictive sampling and the particle filter don't: a factor is a term of the density, not a
    /// generative step.
    ///
    /// ```
    /// use keanu_graph::{GraphBuilder, Model};
    /// use keanu_tensor::DoubleTensor;
    ///
    /// // a flat prior on [-10, 10) with a factor -x²/2 is a standard Gaussian, truncated far in its tails
    /// let b = GraphBuilder::new();
    /// let x = b.uniform(-10.0, 10.0);
    /// b.factor(x * x * -0.5).label("standard normal");
    /// let x = x.id();
    /// let (graph, mut state) = b.build()?;
    /// state.set_and_cascade(&graph, x, DoubleTensor::scalar(2.0))?;
    /// let model = Model::new(graph, &state)?;
    /// assert_eq!(model.log_prob(&state)?, -(20f64.ln()) - 2.0);
    /// # Ok::<(), keanu_graph::GraphError>(())
    /// ```
    pub fn factor<'b>(&'b self, expr: impl IntoDoubleVar<'b>) -> DoubleVar<'b> {
        let expr = expr.into_var(self);
        DoubleVar::new(self, self.add(Op::Factor, &[expr.id()], DType::F64))
    }

    /// The log density of each element of `value` under the distribution `kind` with the given parameters (node
    /// ids in the kind's parameter order), as a deterministic f64 node of the broadcast shape of the value and the
    /// parameters. Its sum is the log probability a node of `kind` with this value would have, so it can be used in
    /// a [`factor`](Self::factor), or to evaluate a density at a value that isn't a node of that distribution.
    ///
    /// `kind` must be element-wise: a distribution of one value per element with an element-wise density (the
    /// continuous and count distributions other than `dirichlet`, `multivariate_gaussian`, `lkj_cholesky`, `kde`,
    /// `categorical`, `multinomial` and `ordered_logistic`, and the `bernoulli`), or a truncated or censored one;
    /// anything else is an error, as is a value or parameter of the wrong type. The value has the kind's value type
    /// (f64, i64 for a count distribution, or bool for the Bernoulli). The node is differentiable, to second order,
    /// in its f64 parents, by the same rules as the distribution.
    ///
    /// ```
    /// use keanu_graph::GraphBuilder;
    /// use keanu_graph::op::DistributionKind;
    /// use keanu_tensor::IntTensor;
    ///
    /// let b = GraphBuilder::new();
    /// let counts = b.int_constant(IntTensor::vector(vec![0, 2]));
    /// let rate = b.scalar(1.5);
    /// let log_p = b.log_density(DistributionKind::Poisson, counts, &[rate.id()]);
    /// let log_p = log_p.id();
    /// let (graph, mut state) = b.build()?;
    /// state.eval(&graph, &[log_p], &mut keanu_distributions::KeanuRng::seed_from_u64(0))?;
    /// let values = state.f64(log_p).unwrap().as_slice().to_vec();
    /// assert!((values[0] + 1.5).abs() < 1e-12);
    /// assert!((values[1] - (2.0 * 1.5f64.ln() - 1.5 - 2f64.ln())).abs() < 1e-12);
    /// # Ok::<(), keanu_graph::GraphError>(())
    /// ```
    pub fn log_density<'b, V: Var<'b>>(
        &'b self,
        kind: DistributionKind,
        value: V,
        parameters: &[NodeId],
    ) -> DoubleVar<'b> {
        let mut parents = vec![self.own(value).id()];
        parents.extend_from_slice(parameters);
        DoubleVar::new(self, self.add(Op::LogDensity(kind), &parents, DType::F64))
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

    /// Joins nodes of the same shape along a new axis (`0..=rank`; -1 appends a last axis).
    pub fn stack<'builder, V: Var<'builder>>(&'builder self, axis: isize, parts: &[V]) -> V {
        let parents: Vec<NodeId> = parts.iter().map(|&p| self.own(p).id()).collect();
        V::wrap(self, self.add(Op::Stack(axis), &parents, V::DTYPE))
    }

    // ----- scan -----

    /// A loop over the steps of `xs` (along their first axis): the general scan.
    ///
    /// `body` builds the loop body on a fresh builder, given placeholders for the carry, for one step of each of
    /// `xs` (their shape without the first axis) and for each of `constants`, and returns the next carry, which must
    /// have the carry's shape. The scan's value stacks the carry after every step: shape `[T, init shape...]`, where
    /// T is the length of the first axis of every `xs` (they must agree; use [`scan_steps`](Self::scan_steps) for a
    /// loop without per-step inputs). Row t is `body(row t − 1, xs[t], constants)`, with row −1 being `init`.
    ///
    /// The body is deterministic: building it with a probabilistic node (`body.gaussian(...)`) or a factor fails. It
    /// sees only its placeholders: outer handles have a different lifetime from the body's, so using one in the body
    /// doesn't compile, and an outer node mixed in some other way (e.g. through its id) is reported when building.
    /// Pass outer values in as `constants`, and random variables drawn per step as `xs`. Several carries are packed
    /// into one tensor (e.g. with [`stack`](Self::stack) and [`slice`](DoubleVar::slice)). Values are f64.
    ///
    /// The graph holds the body once, so building is O(1) in T, and the scan is differentiable to every order the
    /// graph supports through `init`, `xs` and `constants`. Prefer an unrolled loop (one set of nodes per step, as
    /// [`Sequence`](crate::sequence::Sequence)) when each step needs its own random variables in the graph.
    ///
    /// ```
    /// use keanu_graph::GraphBuilder;
    /// use keanu_tensor::DoubleTensor;
    ///
    /// // x_t = φ x_{t-1} + ε_t
    /// let b = GraphBuilder::new();
    /// let phi = b.scalar(0.5);
    /// let eps = b.constant(DoubleTensor::vector(vec![1.0, 0.0, 2.0]));
    /// let x0 = b.scalar(4.0);
    /// let x = b.scan(x0, &[eps], &[phi], |_, x, eps, c| c[0] * x + eps[0]);
    /// let x = x.id();
    /// let (graph, mut state) = b.build()?;
    /// state.eval(&graph, &[x], &mut keanu_distributions::KeanuRng::seed_from_u64(0))?;
    /// assert_eq!(state.f64(x).unwrap().as_slice(), &[3.0, 1.5, 2.75]);
    /// # Ok::<(), keanu_graph::GraphError>(())
    /// ```
    ///
    /// ```compile_fail
    /// # use keanu_graph::GraphBuilder;
    /// let b = GraphBuilder::new();
    /// let outer = b.scalar(2.0);
    /// let xs = b.constant(keanu_tensor::DoubleTensor::vector(vec![1.0, 2.0]));
    /// // the body can't use an outer node directly: pass it as a constant
    /// b.scan(b.scalar(0.0), &[xs], &[], |_, carry, xs, _| carry * outer + xs[0]);
    /// ```
    pub fn scan<'b, F>(
        &'b self,
        init: DoubleVar<'b>,
        xs: &[DoubleVar<'b>],
        constants: &[DoubleVar<'b>],
        body: F,
    ) -> DoubleVar<'b>
    where
        F: for<'s> FnOnce(
            &'s GraphBuilder,
            DoubleVar<'s>,
            &[DoubleVar<'s>],
            &[DoubleVar<'s>],
        ) -> DoubleVar<'s>,
    {
        let xs: Vec<DoubleVar<'b>> = xs.iter().map(|&x| self.own(x)).collect();
        let Some(length) = xs
            .first()
            .map(|x| self.dtype_shape(x.id()).1.first().copied())
        else {
            self.record(GraphError::InvalidNode {
                op: "scan",
                message: "needs per-step inputs to take the number of steps from (use scan_steps)"
                    .into(),
            });
            return DoubleVar::new(self, self.placeholder(DType::F64));
        };
        let Some(length) = length else {
            self.record(GraphError::InvalidNode {
                op: "scan",
                message: "per-step inputs need at least one dimension (the steps)".into(),
            });
            return DoubleVar::new(self, self.placeholder(DType::F64));
        };
        self.scan_node(length, init, &xs, constants, body)
    }

    /// A loop of `length` steps without per-step inputs: as [`scan`](Self::scan), with `body` given the carry and
    /// the constants' placeholders. `length` may be 0, giving an empty `[0, init shape...]` value.
    ///
    /// ```
    /// use keanu_graph::GraphBuilder;
    ///
    /// // doubling: [2, 4, 8, 16]
    /// let b = GraphBuilder::new();
    /// let x = b.scan_steps(4, b.scalar(1.0), &[], |_, x, _| x * 2.0);
    /// assert_eq!(x.shape(), vec![4]);
    /// ```
    pub fn scan_steps<'b, F>(
        &'b self,
        length: usize,
        init: DoubleVar<'b>,
        constants: &[DoubleVar<'b>],
        body: F,
    ) -> DoubleVar<'b>
    where
        F: for<'s> FnOnce(&'s GraphBuilder, DoubleVar<'s>, &[DoubleVar<'s>]) -> DoubleVar<'s>,
    {
        self.scan_node(length, init, &[], constants, |b, carry, _, constants| {
            body(b, carry, constants)
        })
    }

    fn scan_node<'b, F>(
        &'b self,
        length: usize,
        init: DoubleVar<'b>,
        xs: &[DoubleVar<'b>],
        constants: &[DoubleVar<'b>],
        body: F,
    ) -> DoubleVar<'b>
    where
        F: for<'s> FnOnce(
            &'s GraphBuilder,
            DoubleVar<'s>,
            &[DoubleVar<'s>],
            &[DoubleVar<'s>],
        ) -> DoubleVar<'s>,
    {
        let init = self.own(init);
        let xs: Vec<DoubleVar<'b>> = xs.iter().map(|&x| self.own(x)).collect();
        let constants: Vec<DoubleVar<'b>> = constants.iter().map(|&c| self.own(c)).collect();
        let shape = |v: DoubleVar<'b>| self.dtype_shape(v.id()).1;
        let inner = GraphBuilder::new();
        let carry = inner.double_proxy(&shape(init));
        let step_inputs: Vec<DoubleVar<'_>> = xs
            .iter()
            .map(|&x| {
                let s = shape(x);
                inner.double_proxy(s.get(1..).unwrap_or_default())
            })
            .collect();
        let constant_inputs: Vec<DoubleVar<'_>> = constants
            .iter()
            .map(|&c| inner.double_proxy(&shape(c)))
            .collect();
        let output = inner.own(body(&inner, carry, &step_inputs, &constant_inputs));
        let ids = |vars: &[DoubleVar<'_>]| vars.iter().map(|v| v.id()).collect::<Vec<_>>();
        let (carry, step_inputs, constant_inputs, output) = (
            carry.id(),
            ids(&step_inputs),
            ids(&constant_inputs),
            output.id(),
        );
        let body = inner
            .build()
            .and_then(|(graph, state)| {
                check_body_state("scan", &graph, &state)?;
                crate::op::ScanBody::new(graph, length, carry, step_inputs, constant_inputs, output)
            })
            .map_err(|e| match e {
                e @ GraphError::InvalidNode { op: "scan", .. } => e,
                e => GraphError::InvalidNode {
                    op: "scan",
                    message: format!("in the body: {e}"),
                },
            });
        let body = match body {
            Ok(body) => body,
            Err(e) => {
                self.record(e);
                return DoubleVar::new(self, self.placeholder(DType::F64));
            }
        };
        let parents: Vec<NodeId> = std::iter::once(init)
            .chain(xs)
            .chain(constants)
            .map(|v| v.id())
            .collect();
        DoubleVar::new(
            self,
            self.add(Op::Scan(Arc::new(body)), &parents, DType::F64),
        )
    }

    /// The solution of the ordinary differential equation dy/dt = f(t, y, θ), y(t0) = y0, at each of the output
    /// `times`: an f64 node of shape `[T, n]` whose row i is y(tᵢ), for tᵢ the i-th output time.
    ///
    /// `y0` is the initial state, a vector of n elements; `times` a vector of T output times, which must be finite,
    /// strictly increasing and after `t0` (a scalar) whenever the node is evaluated; and `parameters` the values θ
    /// that f depends on, of any shapes. `rhs` builds f on a fresh builder, given placeholders for the time (a
    /// scalar), the state (`[n]`) and each parameter, and returns dy/dt, which must have the state's shape. As for
    /// [`scan`](Self::scan), the body is deterministic and sees only its placeholders: pass outer values in as
    /// parameters.
    ///
    /// `options` choose the solver ([`OdeSolver::Rk45`](crate::ode::OdeSolver::Rk45) for non-stiff problems,
    /// [`OdeSolver::Stiff`](crate::ode::OdeSolver::Stiff) for stiff ones), the tolerances and the step limit (see
    /// [`OdeOptions`](crate::ode::OdeOptions); the defaults are Stan's). A solve that fails at some parameter values
    /// evaluates to [`GraphError::SolverFailed`], which models and inference algorithms treat as an impossible
    /// point.
    ///
    /// The node is differentiable to first order in `y0`, `times`, `t0` and `parameters`, by forward sensitivities:
    /// a gradient integrates n·(1 + p) equations, for p the number of elements of the parents (other than the times
    /// and constants) it is taken through. It has no second-order rule, so Hessian-vector products (and the Laplace
    /// approximation) fail with [`GraphError::NoSecondOrderRule`].
    ///
    /// ```
    /// use keanu_graph::GraphBuilder;
    /// use keanu_graph::ode::OdeOptions;
    /// use keanu_tensor::DoubleTensor;
    ///
    /// // exponential decay dy/dt = -k y
    /// let b = GraphBuilder::new();
    /// let k = b.scalar(0.5);
    /// let times = b.constant(DoubleTensor::vector(vec![1.0, 2.0]));
    /// let y = b.ode(b.constant(DoubleTensor::vector(vec![3.0])), times, b.scalar(0.0), &[k], OdeOptions::default(),
    ///     |_, _t, y, theta| -(theta[0] * y));
    /// assert_eq!(y.shape(), vec![2, 1]);
    /// let y = y.id();
    /// let (graph, mut state) = b.build()?;
    /// state.eval(&graph, &[y], &mut keanu_distributions::KeanuRng::seed_from_u64(0))?;
    /// let solution = state.f64(y).unwrap().as_slice();
    /// assert!((solution[1] - 3.0 * (-1.0f64).exp()).abs() < 1e-5);
    /// # Ok::<(), keanu_graph::GraphError>(())
    /// ```
    pub fn ode<'b, F>(
        &'b self,
        y0: DoubleVar<'b>,
        times: DoubleVar<'b>,
        t0: DoubleVar<'b>,
        parameters: &[DoubleVar<'b>],
        options: crate::ode::OdeOptions,
        rhs: F,
    ) -> DoubleVar<'b>
    where
        F: for<'s> FnOnce(
            &'s GraphBuilder,
            DoubleVar<'s>,
            DoubleVar<'s>,
            &[DoubleVar<'s>],
        ) -> DoubleVar<'s>,
    {
        let (y0, times, t0) = (self.own(y0), self.own(times), self.own(t0));
        let parameters: Vec<DoubleVar<'b>> = parameters.iter().map(|&p| self.own(p)).collect();
        let shape = |v: DoubleVar<'b>| self.dtype_shape(v.id()).1;
        let inner = GraphBuilder::new();
        let time = inner.double_proxy(&[]);
        let state = inner.double_proxy(&shape(y0));
        let parameter_inputs: Vec<DoubleVar<'_>> = parameters
            .iter()
            .map(|&p| inner.double_proxy(&shape(p)))
            .collect();
        let output = inner.own(rhs(&inner, time, state, &parameter_inputs));
        let (time, state, output) = (time.id(), state.id(), output.id());
        let parameter_inputs: Vec<NodeId> = parameter_inputs.iter().map(|v| v.id()).collect();
        let body = inner
            .build()
            .and_then(|(graph, body_state)| {
                check_body_state("ode", &graph, &body_state)?;
                crate::op::OdeBody::new(graph, time, state, parameter_inputs, output, options)
            })
            .map_err(|e| match e {
                e @ GraphError::InvalidNode { op: "ode", .. } => e,
                e => GraphError::InvalidNode {
                    op: "ode",
                    message: format!("in the body: {e}"),
                },
            });
        let body = match body {
            Ok(body) => body,
            Err(e) => {
                self.record(e);
                return DoubleVar::new(self, self.placeholder(DType::F64));
            }
        };
        let parents: Vec<NodeId> = [y0, times, t0]
            .into_iter()
            .chain(parameters)
            .map(|v| v.id())
            .collect();
        DoubleVar::new(
            self,
            self.add(Op::Ode(Arc::new(body)), &parents, DType::F64),
        )
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
        check_shape(kind.name(), &shape)?;
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

/// An error if a node of `shape` would have more elements than any node may have (see
/// [`output_type`](crate::op::output_type), which applies the same cap to inferred shapes).
fn check_shape(op: &'static str, shape: &[usize]) -> Result<()> {
    crate::op::check_element_count(shape).map_err(|message| GraphError::InvalidNode { op, message })
}

/// A scan or ODE body is evaluated from its operations alone, so it can't hold observations or values set while
/// building.
fn check_body_state(op: &'static str, graph: &Graph, state: &State) -> Result<()> {
    for id in graph.ids() {
        let unchanged = match (&graph[id].op, state.value(id)) {
            _ if state.is_observed(id) => false,
            (Op::Constant(op), Some(value)) => same_bits(op, value),
            (_, value) => value.is_none(),
        };
        if !unchanged {
            return Err(GraphError::InvalidNode {
                op,
                message: format!(
                    "the body's node {id:?} is observed or has a value set: {op} bodies are computed from \
                     their inputs alone"
                ),
            });
        }
    }
    Ok(())
}

/// Whether two values are identical, NaNs included.
fn same_bits(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::F64(a), Value::F64(b)) => {
            a.shape() == b.shape()
                && a.as_slice()
                    .iter()
                    .zip(b.as_slice())
                    .all(|(x, y)| x.to_bits() == y.to_bits())
        }
        (a, b) => a == b,
    }
}
