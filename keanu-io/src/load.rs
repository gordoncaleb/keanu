//! Loading a saved model: each saved vertex (identified by its Java class name and named parameters) becomes a
//! node, or a few nodes where the Rust operation is defined differently.

use crate::Metadata;
use crate::classes::{self, SavedDistribution, SimpleOp};
use crate::error::{IoError, Result};
use crate::proto;
use crate::proto::named_param::Param;
use keanu_distributions::Kde;
use keanu_graph::op::{ArgOp, BinaryOp, DistributionKind, Op, ReduceOp, ScanOp};
use keanu_graph::{DType, Graph, GraphBuilder, NodeId, State, Value};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// A loaded model: its graph, the saved values (observations are observed), the saved metadata, and the node each
/// saved vertex id became.
#[derive(Debug)]
#[non_exhaustive]
pub struct Loaded {
    /// The graph.
    pub graph: Graph,
    /// The saved values and observations.
    pub state: State,
    /// The saved metadata.
    pub metadata: Metadata,
    /// Saved vertex id (e.g. `"[3]"`) to node.
    pub ids: HashMap<String, NodeId>,
}

/// A shape from the file: non-negative dimensions whose product (the element count) fits a `usize`.
pub(crate) fn shape_of(dims: &[i64]) -> Result<Vec<usize>> {
    let shape = dims
        .iter()
        .map(|&d| {
            usize::try_from(d).map_err(|_| IoError::Decode(format!("negative dimension {d}")))
        })
        .collect::<Result<Vec<usize>>>()?;
    checked_len(&shape)?;
    Ok(shape)
}

/// The element count of a shape, or a decode error if it overflows.
fn checked_len(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| IoError::Decode(format!("the shape {shape:?} has too many elements")))
}

pub(crate) fn double_tensor(t: &proto::DoubleTensor) -> Result<DoubleTensor> {
    DoubleTensor::new(&shape_of(&t.shape)?, t.values.clone())
        .map_err(|e| IoError::Decode(e.to_string()))
}

pub(crate) fn int_tensor(t: &proto::IntegerTensor) -> Result<IntTensor> {
    IntTensor::new(
        &shape_of(&t.shape)?,
        t.values.iter().map(|&v| i64::from(v)).collect(),
    )
    .map_err(|e| IoError::Decode(e.to_string()))
}

pub(crate) fn bool_tensor(t: &proto::BooleanTensor) -> Result<BoolTensor> {
    BoolTensor::new(&shape_of(&t.shape)?, t.values.clone())
        .map_err(|e| IoError::Decode(e.to_string()))
}

/// Builds a graph from a decoded model.
pub(crate) fn load_model(model: &proto::ProtoModel) -> Result<Loaded> {
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| IoError::Decode("the model has no graph".into()))?;
    let builder = GraphBuilder::new();
    let mut ids: HashMap<String, NodeId> = HashMap::new();

    let mut seen = HashSet::new();
    for vertex in &graph.vertices {
        let id = vertex_id(vertex)?;
        if !seen.insert(id) {
            return Err(IoError::Decode(format!(
                "more than one vertex has the id {id:?}"
            )));
        }
    }
    let helpers = SavedHelpers::find(graph);

    for vertex in &graph.vertices {
        let id = vertex_id(vertex)?;
        if helpers.skipped.contains(id) {
            continue;
        }
        let node = match helpers.cholesky_inverses.get(id) {
            Some(input) => {
                let input = *ids
                    .get(*input)
                    .ok_or_else(|| IoError::UnknownVertex((*input).to_owned()))?;
                builder.add_node(Op::CholeskyInverse, &[input])?
            }
            None => Loader {
                builder: &builder,
                vertex,
                ids: &ids,
                helpers: &helpers,
            }
            .load()?,
        };
        if !vertex.label.is_empty() {
            builder.set_label_of(node, &vertex.label);
        }
        ids.insert(id.to_owned(), node);
    }

    let mut labels: Option<HashMap<String, NodeId>> = None;
    for stored in &graph.default_state {
        let node = match stored.id.as_ref().filter(|id| !id.id.is_empty()) {
            // the value of a helper vertex that was not loaded (never an observation, see SavedHelpers)
            Some(id) if helpers.skipped.contains(id.id.as_str()) => continue,
            Some(id) => *ids
                .get(&id.id)
                .ok_or_else(|| IoError::UnknownVertex(id.id.clone()))?,
            None => {
                let labels = labels.get_or_insert_with(|| {
                    let mut labels = HashMap::new();
                    for n in (0..builder.node_count()).map(NodeId::from_index) {
                        if let Some(label) = builder.label_of(n) {
                            labels.entry(label).or_insert(n);
                        }
                    }
                    labels
                });
                *labels
                    .get(&stored.vertex_label)
                    .ok_or_else(|| IoError::UnknownVertex(stored.vertex_label.clone()))?
            }
        };
        let value = match stored.value.as_ref().and_then(|v| v.value_type.as_ref()) {
            Some(proto::vertex_value::ValueType::DoubleVal(t)) => Value::F64(double_tensor(t)?),
            Some(proto::vertex_value::ValueType::IntVal(t)) => Value::I64(int_tensor(t)?),
            Some(proto::vertex_value::ValueType::BoolVal(t)) => Value::Bool(bool_tensor(t)?),
            Some(proto::vertex_value::ValueType::GenericVal(_)) => {
                return Err(IoError::UnsupportedValue("generic (object)".into()));
            }
            None => return Err(IoError::Decode("a stored value has no value".into())),
        };
        if stored.is_observed {
            builder.observe_of(node, value)?;
        } else {
            builder.set_value_of(node, value)?;
        }
    }

    let metadata = model
        .metadata
        .as_ref()
        .map(|m| {
            m.metadata_info
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
        .unwrap_or_default();
    let (graph, state) = builder.build()?;
    Ok(Loaded {
        graph,
        state,
        metadata,
        ids,
    })
}

fn vertex_id(vertex: &proto::Vertex) -> Result<&str> {
    vertex
        .id
        .as_ref()
        .map(|id| id.id.as_str())
        .filter(|id| !id.is_empty())
        .ok_or_else(|| IoError::Decode(format!("a {} vertex has no id", vertex.vertex_type)))
}

fn simple_class(vertex: &proto::Vertex) -> &str {
    classes::simple_name(&vertex.vertex_type)
}

fn find_param<'v>(vertex: &'v proto::Vertex, name: &str) -> Option<&'v Param> {
    vertex
        .parameters
        .iter()
        .find(|p| p.name == name)
        .and_then(|p| p.param.as_ref())
}

fn parent_id<'v>(vertex: &'v proto::Vertex, name: &str) -> Option<&'v str> {
    match find_param(vertex, name) {
        Some(Param::ParentVertex(id)) => Some(id.id.as_str()),
        _ => None,
    }
}

fn int_param(vertex: &proto::Vertex, name: &str) -> Option<i64> {
    match find_param(vertex, name) {
        Some(Param::IntParam(v)) => Some(i64::from(*v)),
        Some(Param::LongParam(v)) => Some(*v),
        _ => None,
    }
}

/// Vertices that saving writes in place of a single node, which loading folds back so that saving and loading
/// repeatedly doesn't grow the graph:
///
/// - the zero location constant of a half Gaussian / half Cauchy (the Rust distributions have no location; Java
///   saved a shared zero constant too), and
/// - the subgraph `C + transpose(tril(C, -1))` with `C = CholeskyInverseVertex(tril(x))` that saving writes for a
///   Rust cholesky inverse (Java's vertex keeps only the lower triangle of the inverse).
///
/// Only unlabelled, unobserved vertices used by nothing else are folded.
#[derive(Default)]
struct SavedHelpers<'a> {
    /// Vertices not loaded.
    skipped: HashSet<&'a str>,
    /// Skipped zero constants that were the location of a half distribution.
    zero_locations: HashSet<&'a str>,
    /// AdditionVertex id -> the input of the saved cholesky inverse it completes.
    cholesky_inverses: HashMap<&'a str, &'a str>,
}

impl<'a> SavedHelpers<'a> {
    fn find(graph: &'a proto::Graph) -> Self {
        let mut by_id: HashMap<&str, &proto::Vertex> = HashMap::new();
        let mut uses: HashMap<&str, usize> = HashMap::new();
        for vertex in &graph.vertices {
            if let Some(id) = &vertex.id {
                by_id.insert(id.id.as_str(), vertex);
            }
            for param in vertex.parameters.iter().filter_map(|p| p.param.as_ref()) {
                match param {
                    Param::ParentVertex(id) => *uses.entry(id.id.as_str()).or_default() += 1,
                    Param::VertexArrayParam(array) => {
                        for id in &array.values {
                            *uses.entry(id.id.as_str()).or_default() += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
        let observed: HashSet<&str> = graph
            .default_state
            .iter()
            .filter(|s| s.is_observed)
            .filter_map(|s| s.id.as_ref().map(|id| id.id.as_str()))
            .collect();
        let uses_of = |id: &str| uses.get(id).copied().unwrap_or(0);
        // an unlabelled, unobserved vertex of the given class
        let helper = |id: &str, class: &str| -> Option<&'a proto::Vertex> {
            by_id.get(id).copied().filter(|v| {
                v.label.is_empty() && simple_class(v) == class && !observed.contains(id)
            })
        };

        let mut helpers = SavedHelpers::default();

        let mut location_uses: HashMap<&str, usize> = HashMap::new();
        for vertex in &graph.vertices {
            let location = match simple_class(vertex) {
                "HalfGaussianVertex" => parent_id(vertex, "mu"),
                "HalfCauchyVertex" => parent_id(vertex, "location"),
                _ => None,
            };
            if let Some(location) = location {
                *location_uses.entry(location).or_default() += 1;
            }
        }
        for (location, count) in location_uses {
            let is_zero = helper(location, "ConstantDoubleVertex").is_some_and(|v| {
                matches!(
                    find_param(v, "constant"),
                    Some(Param::DoubleTensorParam(t)) if t.values.iter().all(|&x| x == 0.0)
                )
            });
            if is_zero && uses_of(location) == count {
                helpers.skipped.insert(location);
                helpers.zero_locations.insert(location);
            }
        }

        for vertex in &graph.vertices {
            if simple_class(vertex) != "AdditionVertex" {
                continue;
            }
            let Some(add) = vertex.id.as_ref().map(|id| id.id.as_str()) else {
                continue;
            };
            let found = (|| {
                let c_id = parent_id(vertex, "left")?;
                let t_id = parent_id(vertex, "right")?;
                let t = helper(t_id, "PermuteVertex").filter(|_| uses_of(t_id) == 1)?;
                let rank = t.shape.len();
                if rank < 2 {
                    return None;
                }
                let permutation: Vec<i64> = match find_param(t, "rearrange")? {
                    Param::IntArrayParam(a) => a.values.iter().map(|&v| i64::from(v)).collect(),
                    Param::LongArrayParam(a) => a.values.clone(),
                    _ => return None,
                };
                let mut swap_last_two: Vec<i64> = (0..rank as i64).collect();
                swap_last_two.swap(rank - 1, rank - 2);
                if permutation != swap_last_two {
                    return None;
                }
                let s_id = parent_id(t, "inputVertex")?;
                let s = helper(s_id, "TriLowerVertex").filter(|_| uses_of(s_id) == 1)?;
                if int_param(s, "k")? != 1 || parent_id(s, "inputVertex")? != c_id {
                    return None;
                }
                let c = helper(c_id, "CholeskyInverseVertex").filter(|_| uses_of(c_id) == 2)?;
                let a_id = parent_id(c, "inputVertex")?;
                let a = helper(a_id, "TriLowerVertex").filter(|_| uses_of(a_id) == 1)?;
                if int_param(a, "k")? != 0 {
                    return None;
                }
                let x = parent_id(a, "inputVertex")?;
                Some(([a_id, c_id, s_id, t_id], x))
            })();
            if let Some((parts, x)) = found {
                helpers.skipped.extend(parts);
                helpers.cholesky_inverses.insert(add, x);
            }
        }
        helpers
    }
}

struct Loader<'a> {
    builder: &'a GraphBuilder,
    vertex: &'a proto::Vertex,
    ids: &'a HashMap<String, NodeId>,
    helpers: &'a SavedHelpers<'a>,
}

impl Loader<'_> {
    fn class(&self) -> &str {
        simple_class(self.vertex)
    }

    fn invalid(&self, parameter: &str, message: impl Into<String>) -> IoError {
        IoError::InvalidParameter {
            vertex: self.class().to_owned(),
            parameter: parameter.to_owned(),
            message: message.into(),
        }
    }

    fn param(&self, name: &str) -> Option<&Param> {
        find_param(self.vertex, name)
    }

    fn require(&self, name: &str) -> Result<&Param> {
        self.param(name)
            .ok_or_else(|| self.invalid(name, "missing"))
    }

    fn lookup(&self, id: &proto::VertexId) -> Result<NodeId> {
        self.ids
            .get(&id.id)
            .copied()
            .ok_or_else(|| IoError::UnknownVertex(id.id.clone()))
    }

    fn parent(&self, name: &str) -> Result<NodeId> {
        match self.require(name)? {
            Param::ParentVertex(id) => self.lookup(id),
            _ => Err(self.invalid(name, "expected a parent vertex")),
        }
    }

    fn parents(&self, name: &str) -> Result<Vec<NodeId>> {
        match self.require(name)? {
            Param::VertexArrayParam(array) => {
                array.values.iter().map(|id| self.lookup(id)).collect()
            }
            _ => Err(self.invalid(name, "expected an array of vertices")),
        }
    }

    fn int(&self, name: &str) -> Result<i64> {
        match self.require(name)? {
            Param::IntParam(v) => Ok(i64::from(*v)),
            Param::LongParam(v) => Ok(*v),
            _ => Err(self.invalid(name, "expected an integer")),
        }
    }

    fn optional_int(&self, name: &str) -> Result<Option<i64>> {
        if self.param(name).is_none() {
            Ok(None)
        } else {
            self.int(name).map(Some)
        }
    }

    fn double(&self, name: &str) -> Result<f64> {
        match self.require(name)? {
            Param::DoubleParam(v) => Ok(*v),
            _ => Err(self.invalid(name, "expected a double")),
        }
    }

    fn boolean(&self, name: &str) -> Result<bool> {
        match self.require(name)? {
            Param::BoolParam(v) => Ok(*v),
            _ => Err(self.invalid(name, "expected a boolean")),
        }
    }

    fn string(&self, name: &str) -> Result<String> {
        match self.param(name) {
            Some(Param::StringParam(v)) => Ok(v.clone()),
            None => Ok(String::new()),
            _ => Err(self.invalid(name, "expected a string")),
        }
    }

    fn longs(&self, name: &str) -> Result<Vec<i64>> {
        match self.require(name)? {
            Param::LongArrayParam(a) => Ok(a.values.clone()),
            Param::IntArrayParam(a) => Ok(a.values.iter().map(|&v| i64::from(v)).collect()),
            _ => Err(self.invalid(name, "expected an array of integers")),
        }
    }

    fn optional_longs(&self, name: &str) -> Result<Option<Vec<i64>>> {
        if self.param(name).is_none() {
            Ok(None)
        } else {
            self.longs(name).map(Some)
        }
    }

    fn booleans(&self, name: &str) -> Result<Vec<bool>> {
        match self.require(name)? {
            Param::BoolTensorParam(t) => Ok(t.values.clone()),
            _ => Err(self.invalid(name, "expected booleans")),
        }
    }

    fn shape(&self) -> Result<Vec<usize>> {
        shape_of(&self.vertex.shape)
    }

    fn node(&self, op: Op, parents: &[NodeId]) -> Result<NodeId> {
        Ok(self.builder.add_node(op, parents)?)
    }

    fn unary(&self, op: Op) -> Result<NodeId> {
        self.node(op, &[self.parent("inputVertex")?])
    }

    fn binary(&self, op: Op) -> Result<NodeId> {
        self.node(op, &[self.parent("left")?, self.parent("right")?])
    }

    fn distribution(&self, saved: &SavedDistribution) -> Result<NodeId> {
        if let Some(location) = saved.zero_location {
            // the half distributions are saved with a location parameter, which must be zero; a zero constant used
            // only as this location isn't loaded (see SavedHelpers)
            let folded = parent_id(self.vertex, location)
                .is_some_and(|id| self.helpers.zero_locations.contains(id));
            if !folded {
                let location_node = self.parent(location)?;
                let is_zero_constant = matches!(
                    self.builder.constant_value(location_node),
                    Some(Value::F64(t)) if t.as_slice().iter().all(|&v| v == 0.0)
                );
                if !is_zero_constant {
                    return Err(self.invalid(location, "must be the constant 0"));
                }
            }
        }
        let parents = saved
            .parameters
            .iter()
            .map(|p| self.parent(p))
            .collect::<Result<Vec<_>>>()?;
        let shape = self.shape()?;
        Ok(self
            .builder
            .add_distribution(saved.kind.clone(), Some(&shape), &parents)?)
    }

    /// numpy slice text for Java's StridedSliceVertex parameters.
    fn strided_slice(&self) -> Result<String> {
        let start = self.longs("start")?;
        let end = self.longs("end")?;
        let stride = self.longs("stride")?;
        let upper_bound = self.booleans("upperBoundStop")?;
        let drop = self.booleans("dropDimension")?;
        let ellipsis = self.optional_int("ellipsis")?;
        let n = start.len();
        if [end.len(), stride.len(), upper_bound.len(), drop.len()]
            .iter()
            .any(|&l| l != n)
        {
            return Err(self.invalid("start", "the slice parameters have different lengths"));
        }
        let mut items: Vec<String> = (0..n)
            .map(|i| {
                if drop[i] {
                    start[i].to_string()
                } else {
                    let stop = if upper_bound[i] {
                        String::new()
                    } else {
                        end[i].to_string()
                    };
                    format!("{}:{}:{}", start[i], stop, stride[i])
                }
            })
            .collect();
        if let Some(position) = ellipsis {
            let position =
                usize::try_from(position).map_err(|_| self.invalid("ellipsis", "negative"))?;
            items.insert(position.min(items.len()), "...".into());
        }
        if items.is_empty() {
            items.push("...".into());
        }
        Ok(items.join(", "))
    }

    fn load(&self) -> Result<NodeId> {
        let isize_of = |v: i64| v as isize;
        if let Some(simple) = SimpleOp::of_class(self.class()) {
            let parents = simple
                .parameters()
                .iter()
                .map(|p| self.parent(p))
                .collect::<Result<Vec<_>>>()?;
            return self.node(simple.op(), &parents);
        }
        if let Some(saved) = classes::distribution_of_class(self.class()) {
            return self.distribution(saved);
        }

        match self.class() {
            "ConstantDoubleVertex" => match self.require("constant")? {
                Param::DoubleTensorParam(t) => {
                    self.node(Op::Constant(Value::F64(double_tensor(t)?)), &[])
                }
                _ => Err(self.invalid("constant", "expected a double tensor")),
            },
            "ConstantIntegerVertex" => match self.require("constant")? {
                Param::IntTensorParam(t) => {
                    self.node(Op::Constant(Value::I64(int_tensor(t)?)), &[])
                }
                _ => Err(self.invalid("constant", "expected an integer tensor")),
            },
            "ConstantBooleanVertex" => match self.require("constant")? {
                Param::BoolTensorParam(t) => {
                    self.node(Op::Constant(Value::Bool(bool_tensor(t)?)), &[])
                }
                _ => Err(self.invalid("constant", "expected a boolean tensor")),
            },

            "NumericalEqualsVertex" => self.node(
                Op::ApproxEq(self.double("epsilon")?),
                &[self.parent("a")?, self.parent("b")?],
            ),
            "NotBinaryVertex" => self.unary(Op::Not),
            "AllTrueVertex" => self.unary(Op::All),
            "AnyTrueVertex" => self.unary(Op::Any),
            "AllFalseVertex" => {
                let any = self.unary(Op::Any)?;
                self.node(Op::Not, &[any])
            }
            "AnyFalseVertex" => {
                let all = self.unary(Op::All)?;
                self.node(Op::Not, &[all])
            }
            "CastNumberToDoubleVertex" => self.unary(Op::Cast(DType::F64)),
            "CastNumberToIntegerVertex" => self.unary(Op::Cast(DType::I64)),
            "CastNumberToBooleanVertex" | "CastToBooleanVertex" => {
                self.unary(Op::Cast(DType::Bool))
            }
            "BooleanToDoubleMaskVertex" => {
                self.node(Op::Cast(DType::F64), &[self.parent("inputName")?])
            }
            "BooleanToIntegerMaskVertex" => {
                self.node(Op::Cast(DType::I64), &[self.parent("inputName")?])
            }

            "SumVertex" => self.reduce(ReduceOp::Sum),
            "ProductVertex" => self.reduce(ReduceOp::Product),
            "MeanVertex" => self.reduce(ReduceOp::Mean),
            "MaxUnaryVertex" => self.unary(Op::Reduce {
                kind: ReduceOp::Max,
                axes: None,
            }),
            "MinUnaryVertex" => self.unary(Op::Reduce {
                kind: ReduceOp::Min,
                axes: None,
            }),
            "StandardDeviationVertex" => self.unary(Op::Reduce {
                kind: ReduceOp::StdDev,
                axes: None,
            }),
            "StandardizeVertex" => self.unary(Op::Standardize),
            "CumSumVertex" => self.unary(Op::Scan {
                kind: ScanOp::CumSum,
                axis: isize_of(self.int("requestedDimension")?),
            }),
            "CumProdVertex" => self.unary(Op::Scan {
                kind: ScanOp::CumProd,
                axis: isize_of(self.int("requestedDimension")?),
            }),
            "ArgMaxVertex" => self.arg(ArgOp::ArgMax),
            "ArgMinVertex" => self.arg(ArgOp::ArgMin),
            "NaNArgMaxVertex" => self.arg(ArgOp::NanArgMax),
            "NaNArgMinVertex" => self.arg(ArgOp::NanArgMin),

            "ReshapeVertex" => self.unary(Op::Reshape(
                self.longs("proposedShape")?
                    .into_iter()
                    .map(isize_of)
                    .collect(),
            )),
            "PermuteVertex" => self.unary(Op::Permute(self.unsigned("rearrange")?)),
            "BroadcastVertex" => {
                let to_shape = self.unsigned("toShape")?;
                checked_len(&to_shape)?;
                self.unary(Op::Broadcast(to_shape))
            }
            "SliceVertex" => {
                let input = self.parent("inputVertex")?;
                let rank = self.builder.shape(input).len();
                let dimension = usize::try_from(self.int("dimension")?)
                    .ok()
                    .filter(|&d| d < rank)
                    .ok_or_else(|| {
                        self.invalid(
                            "dimension",
                            format!("not a dimension of a rank {rank} input"),
                        )
                    })?;
                let mut items = vec![":".to_owned(); dimension];
                items.push(self.int("index")?.to_string());
                self.node(Op::Slice(items.join(", ")), &[input])
            }
            "StridedSliceVertex" => self.unary(Op::Slice(self.strided_slice()?)),
            "TakeVertex" => self.unary(Op::Take(self.unsigned("index")?)),
            "DiagVertex" => self.unary(Op::Diag),
            "DiagPartVertex" => self.unary(Op::DiagPart),
            // Java's TriLowerVertex counts k in the opposite direction to numpy's tril
            "TriLowerVertex" => self.unary(Op::Tril(-isize_of(self.int("k")?))),
            "TriUpperVertex" => self.unary(Op::Triu(isize_of(self.int("k")?))),
            "FillTriangularVertex" => self.unary(Op::FillTriangular {
                upper: self.boolean("fillUpper")?,
                lower: self.boolean("fillLower")?,
            }),
            "TrianglePartVertex" => self.unary(Op::TrianglePart {
                upper: self.boolean("upperPart")?,
            }),
            "ConcatenationVertex" | "IntegerConcatenationVertex" | "BooleanConcatenationVertex" => {
                self.node(
                    Op::Concat(isize_of(self.int("dimension")?)),
                    &self.parents("operands")?,
                )
            }

            "MatrixMultiplicationVertex" => self.binary(Op::MatMul {
                transpose_left: self.boolean("transposeLeft")?,
                transpose_right: self.boolean("transposeRight")?,
            }),
            "TensorMultiplicationVertex" => self.binary(Op::TensorMultiply {
                left_axes: self.longs("dimsLeft")?.into_iter().map(isize_of).collect(),
                right_axes: self.longs("dimsRight")?.into_iter().map(isize_of).collect(),
            }),
            "MatrixInverseVertex" => self.unary(Op::Inverse),
            "MatrixDeterminantVertex" => self.unary(Op::Determinant),
            "CholeskyDecompositionVertex" => self.unary(Op::Cholesky),
            "CholeskyInverseVertex" => {
                // Java returns the lower triangle of the inverse plus the input's strict upper triangle
                let input = self.parent("inputVertex")?;
                let inverse = self.node(Op::CholeskyInverse, &[input])?;
                let lower = self.node(Op::Tril(0), &[inverse])?;
                let upper = self.node(Op::Triu(1), &[input])?;
                self.node(Op::Binary(BinaryOp::Add), &[lower, upper])
            }

            "WhereVertex" => self.node(
                Op::Where,
                &[
                    self.parent("predicate")?,
                    self.parent("then")?,
                    self.parent("else")?,
                ],
            ),
            "MultiplexerVertex" => {
                let mut parents = vec![self.parent("selectorControlVertex")?];
                parents.extend(self.parents("selectVertices")?);
                self.node(Op::Multiplexer, &parents)
            }
            "SetWithMaskVertex" => self.node(
                Op::SetWithMask,
                &[
                    self.parent("operand")?,
                    self.parent("mask")?,
                    self.parent("setValue")?,
                ],
            ),
            "ReplaceNaNVertex" => self.unary(Op::ReplaceNan(self.double("REPLACE_WITH_VALUE")?)),
            "AssertVertex" => {
                let message = self.string("error")?;
                self.node(
                    Op::Assert((!message.is_empty()).then_some(message)),
                    &[self.parent("predicate")?],
                )
            }
            "DoubleProxyVertex" | "IntegerProxyVertex" | "BooleanProxyVertex" => {
                let parent = self.parent("parent")?;
                let dtype = self.builder.dtype(parent);
                let proxy = self.builder.add_proxy(dtype, &self.builder.shape(parent));
                self.builder.connect_ids(proxy, parent)?;
                // The proxy is labelled from the vertex label, which Java always wrote equal to this parameter.
                // Earlier Rust versions wrote an invented `proxy_<n>` here for unlabelled proxies, so it isn't used.
                self.string("label")?;
                Ok(proxy)
            }

            "KDEVertex" => {
                let samples = match self.require("samples")? {
                    Param::DoubleTensorParam(t) => double_tensor(t)?.into_vec(),
                    _ => return Err(self.invalid("samples", "expected a double tensor")),
                };
                let kde = Kde::new(samples, self.double("bandwidth")?)
                    .map_err(|e| self.invalid("samples", e.to_string()))?;
                Ok(self.builder.add_distribution(
                    DistributionKind::Kde(Arc::new(kde)),
                    Some(&self.shape()?),
                    &[],
                )?)
            }
            other => Err(IoError::UnsupportedVertex(other.to_owned())),
        }
    }

    fn unsigned(&self, name: &str) -> Result<Vec<usize>> {
        self.longs(name)?
            .into_iter()
            .map(|v| usize::try_from(v).map_err(|_| self.invalid(name, "negative value")))
            .collect()
    }

    fn reduce(&self, kind: ReduceOp) -> Result<NodeId> {
        let axes = self
            .optional_longs("overDimensions")?
            .map(|a| a.into_iter().map(|v| v as isize).collect());
        self.unary(Op::Reduce { kind, axes })
    }

    fn arg(&self, kind: ArgOp) -> Result<NodeId> {
        let axis = self.optional_int("axis")?.map(|a| a as isize);
        self.unary(Op::ArgReduce { kind, axis })
    }
}
