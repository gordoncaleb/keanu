//! Saving a graph in the saved model schema, with Java vertex class names and parameter names so the Java
//! implementation can load it.

use crate::Metadata;
use crate::classes::{self, SimpleOp};
use crate::error::{IoError, Result};
use crate::proto;
use crate::proto::named_param::Param;
use keanu_graph::op::{ArgOp, CompareOp, DistributionKind, Op, ReduceOp, ScanOp, UnaryOp};
use keanu_graph::{DType, Graph, NodeId, State, Value};
use keanu_tensor::ops::shape_ops::{SliceItem, parse_slice};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};

pub(crate) fn proto_double(t: &DoubleTensor) -> proto::DoubleTensor {
    proto::DoubleTensor {
        shape: t.shape().iter().map(|&d| d as i64).collect(),
        values: t.as_slice().to_vec(),
    }
}

pub(crate) fn proto_int(t: &IntTensor) -> Result<proto::IntegerTensor> {
    let values = t
        .as_slice()
        .iter()
        .map(|&v| {
            i32::try_from(v).map_err(|_| {
                IoError::NotSaveable(format!(
                    "integer {v} does not fit the format's 32-bit integers"
                ))
            })
        })
        .collect::<Result<_>>()?;
    Ok(proto::IntegerTensor {
        shape: t.shape().iter().map(|&d| d as i64).collect(),
        values,
    })
}

pub(crate) fn proto_bool(t: &BoolTensor) -> proto::BooleanTensor {
    proto::BooleanTensor {
        shape: t.shape().iter().map(|&d| d as i64).collect(),
        values: t.as_slice().to_vec(),
    }
}

fn proto_value(value: &Value) -> Result<proto::VertexValue> {
    use proto::vertex_value::ValueType;
    Ok(proto::VertexValue {
        value_type: Some(match value {
            Value::F64(t) => ValueType::DoubleVal(proto_double(t)),
            Value::I64(t) => ValueType::IntVal(proto_int(t)?),
            Value::Bool(t) => ValueType::BoolVal(proto_bool(t)),
            _ => {
                return Err(IoError::NotSaveable(
                    "a value of an unsupported type".into(),
                ));
            }
        }),
    })
}

fn java_class(simple: &str) -> Result<&'static str> {
    classes::java_class(simple)
        .ok_or_else(|| IoError::NotSaveable(format!("no saved type is recorded for {simple}")))
}

/// The class and parameters of a constant vertex, or None for an unsupported type.
fn constant_vertex(value: &Value) -> Option<(&'static str, Vec<(&'static str, P<'_>)>)> {
    let (class, constant) = match value {
        Value::F64(t) => ("ConstantDoubleVertex", P::DoubleTensor(t)),
        Value::I64(t) => ("ConstantIntegerVertex", P::IntTensor(t)),
        Value::Bool(t) => ("ConstantBooleanVertex", P::BoolTensor(t)),
        _ => return None,
    };
    Some((class, vec![("constant", constant)]))
}

/// One parameter of a saved vertex, borrowing from the graph where it can.
enum P<'a> {
    Parent(NodeId),
    /// A vertex already saved, by its index in the saved vertices.
    Saved(usize),
    Parents(&'a [NodeId]),
    Int(i64),
    Double(f64),
    Bool(bool),
    Str(&'a str),
    Longs(Vec<i64>),
    Ints(Vec<i64>),
    /// A vector.
    Doubles(&'a [f64]),
    /// A vector.
    Bools(Vec<bool>),
    DoubleTensor(&'a DoubleTensor),
    IntTensor(&'a IntTensor),
    BoolTensor(&'a BoolTensor),
}

struct Saver<'g> {
    graph: &'g Graph,
    vertices: Vec<proto::Vertex>,
    /// The index in `vertices` of each node's saved vertex, indexed by node.
    saved: Vec<Option<usize>>,
}

/// The saved id of the vertex at an index in the saved vertices.
fn vertex_id(index: usize) -> String {
    format!("[{index}]")
}

impl Saver<'_> {
    /// The id of the vertex a node was saved as.
    fn id(&self, node: NodeId) -> Result<proto::VertexId> {
        self.saved
            .get(node.index())
            .copied()
            .flatten()
            .map(|index| self.saved_id(index))
            .ok_or_else(|| {
                IoError::NotSaveable(format!("node {node:?} was not saved before its children"))
            })
    }

    /// The id of a saved vertex.
    fn saved_id(&self, index: usize) -> proto::VertexId {
        self.vertices[index].id.clone().unwrap_or_default()
    }

    fn param(&self, name: &str, p: P<'_>) -> Result<proto::NamedParam> {
        let param = match p {
            P::Parent(node) => Param::ParentVertex(self.id(node)?),
            P::Saved(index) => Param::ParentVertex(self.saved_id(index)),
            P::Parents(nodes) => Param::VertexArrayParam(proto::VertexArray {
                values: nodes.iter().map(|&n| self.id(n)).collect::<Result<_>>()?,
            }),
            P::Int(v) => Param::IntParam(
                i32::try_from(v)
                    .map_err(|_| IoError::NotSaveable(format!("{name} = {v} is too large")))?,
            ),
            P::Double(v) => Param::DoubleParam(v),
            P::Bool(v) => Param::BoolParam(v),
            P::Str(v) => Param::StringParam(v.to_owned()),
            P::Longs(v) => Param::LongArrayParam(proto::LongArray { values: v }),
            P::Ints(v) => Param::IntArrayParam(proto::IntArray {
                values: v
                    .into_iter()
                    .map(|x| {
                        i32::try_from(x)
                            .map_err(|_| IoError::NotSaveable(format!("{name} = {x} is too large")))
                    })
                    .collect::<Result<_>>()?,
            }),
            P::Doubles(values) => Param::DoubleTensorParam(proto::DoubleTensor {
                shape: vec![values.len() as i64],
                values: values.to_vec(),
            }),
            P::Bools(values) => Param::BoolTensorParam(proto::BooleanTensor {
                shape: vec![values.len() as i64],
                values,
            }),
            P::DoubleTensor(t) => Param::DoubleTensorParam(proto_double(t)),
            P::IntTensor(t) => Param::IntTensorParam(proto_int(t)?),
            P::BoolTensor(t) => Param::BoolTensorParam(proto_bool(t)),
        };
        Ok(proto::NamedParam {
            name: name.to_owned(),
            param: Some(param),
        })
    }

    /// Emits a vertex, returning its index in the saved vertices.
    fn emit(
        &mut self,
        class: &str,
        label: &str,
        shape: &[usize],
        params: Vec<(&str, P<'_>)>,
    ) -> Result<usize> {
        let mut parameters = params
            .into_iter()
            .map(|(name, p)| self.param(name, p))
            .collect::<Result<Vec<_>>>()?;
        // the Java saver writes parameters sorted by name
        parameters.sort_by(|a, b| a.name.cmp(&b.name));
        let index = self.vertices.len();
        self.vertices.push(proto::Vertex {
            id: Some(proto::VertexId {
                id: vertex_id(index),
            }),
            label: label.to_owned(),
            vertex_type: java_class(class)?.to_owned(),
            parameters,
            shape: shape.iter().map(|&d| d as i64).collect(),
            data_type: 0,
            metadata: Vec::new(),
        });
        Ok(index)
    }

    /// Emits an unlabelled helper constant.
    fn constant(&mut self, value: &Value) -> Result<usize> {
        let (class, params) = constant_vertex(value)
            .ok_or_else(|| IoError::NotSaveable("a constant of an unsupported type".into()))?;
        self.emit(class, "", value.shape(), params)
    }

    /// Saves a node, returning the index of its vertex in the saved vertices.
    fn save_node(&mut self, id: NodeId) -> Result<usize> {
        let graph = self.graph;
        let node = &graph[id];
        let label = node.label().unwrap_or("");
        let shape = node.shape();
        let parents = node.parents();
        let p = |i: usize| P::Parent(parents[i]);
        // the parents, with the given parameter names in order
        let named = |names: &[&'static str]| -> Vec<(&'static str, P)> {
            names
                .iter()
                .enumerate()
                .map(|(i, &n)| (n, P::Parent(parents[i])))
                .collect()
        };
        let input = || vec![("inputVertex", P::Parent(parents[0]))];
        let unsupported = |what: &str| {
            Err(IoError::NotSaveable(format!(
                "{what} (node {id:?}) has no saved form"
            )))
        };
        let input_dtype = parents.first().map(|&p| graph[p].dtype());

        if let Some((simple, class)) = SimpleOp::of(node.op()).and_then(|s| Some((s, s.class()?))) {
            return self.emit(class, label, shape, named(simple.parameters()));
        }
        let (class, params): (&str, Vec<(&str, P)>) = match node.op() {
            Op::Constant(value) => match constant_vertex(value) {
                Some(vertex) => vertex,
                None => return unsupported("a constant of an unsupported type"),
            },
            Op::Unary(op) => match op {
                UnaryOp::Neg => {
                    let minus_one = if node.dtype() == DType::I64 {
                        Value::I64(IntTensor::scalar(-1))
                    } else {
                        Value::F64(DoubleTensor::scalar(-1.0))
                    };
                    let c = self.constant(&minus_one)?;
                    (
                        "MultiplicationVertex",
                        vec![("left", p(0)), ("right", P::Saved(c))],
                    )
                }
                UnaryOp::Reciprocal => {
                    let c = self.constant(&Value::F64(DoubleTensor::scalar(1.0)))?;
                    (
                        "DivisionVertex",
                        vec![("left", P::Saved(c)), ("right", p(0))],
                    )
                }
                UnaryOp::Sqrt => {
                    let c = self.constant(&Value::F64(DoubleTensor::scalar(0.5)))?;
                    ("PowerVertex", vec![("left", p(0)), ("right", P::Saved(c))])
                }
                other => return Err(IoError::NotSaveable(format!("{other:?}"))),
            },
            Op::Mask(CompareOp::Eq | CompareOp::Ne) => return unsupported("an equality mask"),
            Op::ApproxEq(epsilon) => (
                "NumericalEqualsVertex",
                vec![("a", p(0)), ("b", p(1)), ("epsilon", P::Double(*epsilon))],
            ),
            Op::Not => ("NotBinaryVertex", input()),
            Op::All => ("AllTrueVertex", input()),
            Op::Any => ("AnyTrueVertex", input()),
            Op::Reduce { kind, axes } => {
                let class = match (kind, axes) {
                    (ReduceOp::Sum, _) => "SumVertex",
                    (ReduceOp::Product, _) => "ProductVertex",
                    (ReduceOp::Mean, _) => "MeanVertex",
                    (ReduceOp::Max, None) => "MaxUnaryVertex",
                    (ReduceOp::Min, None) => "MinUnaryVertex",
                    (ReduceOp::StdDev, None) => "StandardDeviationVertex",
                    _ => return unsupported("a max / min / standard deviation over some axes"),
                };
                let mut params = input();
                if let Some(axes) = axes {
                    params.push((
                        "overDimensions",
                        P::Ints(axes.iter().map(|&a| a as i64).collect()),
                    ));
                }
                (class, params)
            }
            Op::Standardize => ("StandardizeVertex", input()),
            Op::Scan { kind, axis } => (
                match kind {
                    ScanOp::CumSum => "CumSumVertex",
                    ScanOp::CumProd => "CumProdVertex",
                    _ => return unsupported("an unsupported scan"),
                },
                vec![
                    ("inputVertex", p(0)),
                    ("requestedDimension", P::Int(*axis as i64)),
                ],
            ),
            Op::ArgReduce { kind, axis } => {
                let class = match kind {
                    ArgOp::ArgMax => "ArgMaxVertex",
                    ArgOp::ArgMin => "ArgMinVertex",
                    ArgOp::NanArgMax => "NaNArgMaxVertex",
                    ArgOp::NanArgMin => "NaNArgMinVertex",
                    _ => return unsupported("an unsupported arg reduction"),
                };
                let mut params = input();
                if let Some(axis) = axis {
                    params.push(("axis", P::Int(*axis as i64)));
                }
                (class, params)
            }
            Op::Reshape(new_shape) => (
                "ReshapeVertex",
                vec![
                    ("inputVertex", p(0)),
                    (
                        "proposedShape",
                        P::Longs(new_shape.iter().map(|&d| d as i64).collect()),
                    ),
                ],
            ),
            Op::Permute(permutation) => (
                "PermuteVertex",
                vec![
                    ("inputVertex", p(0)),
                    (
                        "rearrange",
                        P::Ints(permutation.iter().map(|&d| d as i64).collect()),
                    ),
                ],
            ),
            Op::Transpose => {
                let rank = graph[parents[0]].shape().len();
                let mut permutation: Vec<i64> = (0..rank as i64).collect();
                permutation.swap(rank - 1, rank - 2);
                (
                    "PermuteVertex",
                    vec![("inputVertex", p(0)), ("rearrange", P::Ints(permutation))],
                )
            }
            Op::Broadcast(to) => (
                "BroadcastVertex",
                vec![
                    ("inputVertex", p(0)),
                    ("toShape", P::Longs(to.iter().map(|&d| d as i64).collect())),
                ],
            ),
            Op::Slice(spec) => {
                let s = strided_slice(spec).map_err(IoError::NotSaveable)?;
                let mut params = vec![
                    ("inputVertex", p(0)),
                    ("start", P::Longs(s.start)),
                    ("end", P::Longs(s.end)),
                    ("stride", P::Longs(s.stride)),
                    ("upperBoundStop", P::Bools(s.upper_bound_stop)),
                    ("dropDimension", P::Bools(s.drop_dimension)),
                ];
                if let Some(e) = s.ellipsis {
                    params.push(("ellipsis", P::Int(e as i64)));
                }
                ("StridedSliceVertex", params)
            }
            Op::Concat(axis) => (
                match node.dtype() {
                    DType::F64 => "ConcatenationVertex",
                    DType::I64 => "IntegerConcatenationVertex",
                    DType::Bool => "BooleanConcatenationVertex",
                    _ => return unsupported("a concatenation of an unsupported type"),
                },
                vec![
                    ("dimension", P::Int(*axis as i64)),
                    ("operands", P::Parents(parents)),
                ],
            ),
            Op::Take(index) => (
                "TakeVertex",
                vec![
                    ("inputVertex", p(0)),
                    ("index", P::Longs(index.iter().map(|&i| i as i64).collect())),
                ],
            ),
            Op::Diag => ("DiagVertex", input()),
            Op::DiagPart => ("DiagPartVertex", input()),
            // Java's TriLowerVertex counts k in the opposite direction to numpy's tril
            Op::Tril(k) => (
                "TriLowerVertex",
                vec![("inputVertex", p(0)), ("k", P::Int(-(*k as i64)))],
            ),
            Op::Triu(k) => (
                "TriUpperVertex",
                vec![("inputVertex", p(0)), ("k", P::Int(*k as i64))],
            ),
            Op::FillTriangular { upper, lower } => (
                "FillTriangularVertex",
                vec![
                    ("inputVertex", p(0)),
                    ("fillUpper", P::Bool(*upper)),
                    ("fillLower", P::Bool(*lower)),
                ],
            ),
            Op::TrianglePart { upper } => (
                "TrianglePartVertex",
                vec![("inputVertex", p(0)), ("upperPart", P::Bool(*upper))],
            ),
            Op::MatMul {
                transpose_left,
                transpose_right,
            } => (
                "MatrixMultiplicationVertex",
                vec![
                    ("left", p(0)),
                    ("right", p(1)),
                    ("transposeLeft", P::Bool(*transpose_left)),
                    ("transposeRight", P::Bool(*transpose_right)),
                ],
            ),
            Op::TensorMultiply {
                left_axes,
                right_axes,
            } => (
                "TensorMultiplicationVertex",
                vec![
                    ("left", p(0)),
                    ("right", p(1)),
                    (
                        "dimsLeft",
                        P::Ints(left_axes.iter().map(|&a| a as i64).collect()),
                    ),
                    (
                        "dimsRight",
                        P::Ints(right_axes.iter().map(|&a| a as i64).collect()),
                    ),
                ],
            ),
            Op::Inverse => ("MatrixInverseVertex", input()),
            Op::Determinant => ("MatrixDeterminantVertex", input()),
            Op::Cholesky => ("CholeskyDecompositionVertex", input()),
            Op::CholeskyInverse => {
                // Java's CholeskyInverseVertex gives the lower triangle of the inverse plus its input's strict upper
                // triangle, so: C = choleskyInverse(tril(L)); full = C + transpose(strictly lower part of C)
                let input_shape = graph[parents[0]].shape();
                let lower_input = self.emit(
                    "TriLowerVertex",
                    "",
                    input_shape,
                    vec![("inputVertex", p(0)), ("k", P::Int(0))],
                )?;
                let c = self.emit(
                    "CholeskyInverseVertex",
                    "",
                    shape,
                    vec![("inputVertex", P::Saved(lower_input))],
                )?;
                let strictly_lower = self.emit(
                    "TriLowerVertex",
                    "",
                    shape,
                    vec![("inputVertex", P::Saved(c)), ("k", P::Int(1))],
                )?;
                let rank = shape.len();
                let mut permutation: Vec<i64> = (0..rank as i64).collect();
                permutation.swap(rank - 1, rank - 2);
                let transposed = self.emit(
                    "PermuteVertex",
                    "",
                    shape,
                    vec![
                        ("inputVertex", P::Saved(strictly_lower)),
                        ("rearrange", P::Ints(permutation)),
                    ],
                )?;
                (
                    "AdditionVertex",
                    vec![("left", P::Saved(c)), ("right", P::Saved(transposed))],
                )
            }
            Op::Cast(to) => match (to, input_dtype) {
                (DType::F64, Some(DType::Bool)) => {
                    ("BooleanToDoubleMaskVertex", vec![("inputName", p(0))])
                }
                (DType::I64, Some(DType::Bool)) => {
                    ("BooleanToIntegerMaskVertex", vec![("inputName", p(0))])
                }
                (DType::F64, _) => ("CastNumberToDoubleVertex", input()),
                (DType::I64, _) => ("CastNumberToIntegerVertex", input()),
                (DType::Bool, Some(DType::Bool)) => ("CastToBooleanVertex", input()),
                (DType::Bool, _) => ("CastNumberToBooleanVertex", input()),
                _ => return unsupported("a cast to an unsupported type"),
            },
            Op::Where => (
                "WhereVertex",
                vec![("predicate", p(0)), ("then", p(1)), ("else", p(2))],
            ),
            Op::Multiplexer => (
                "MultiplexerVertex",
                vec![
                    ("selectorControlVertex", p(0)),
                    ("selectVertices", P::Parents(&parents[1..])),
                ],
            ),
            Op::SetWithMask => (
                "SetWithMaskVertex",
                vec![("operand", p(0)), ("mask", p(1)), ("setValue", p(2))],
            ),
            Op::ReplaceNan(v) => (
                "ReplaceNaNVertex",
                vec![("inputVertex", p(0)), ("REPLACE_WITH_VALUE", P::Double(*v))],
            ),
            Op::Assert(message) => (
                "AssertVertex",
                vec![
                    ("predicate", p(0)),
                    ("error", P::Str(message.as_deref().unwrap_or(""))),
                ],
            ),
            Op::Proxy => {
                if parents.is_empty() {
                    return unsupported("an unconnected proxy");
                }
                let class = match node.dtype() {
                    DType::F64 => "DoubleProxyVertex",
                    DType::I64 => "IntegerProxyVertex",
                    DType::Bool => "BooleanProxyVertex",
                    _ => return unsupported("a proxy of an unsupported type"),
                };
                // an unlabelled proxy is saved with an empty label: an invented one could collide with another
                // node's label
                (class, vec![("parent", p(0)), ("label", P::Str(label))])
            }
            Op::Distribution(distribution) => match &distribution.kind {
                DistributionKind::Kde(kde) => (
                    "KDEVertex",
                    vec![
                        ("samples", P::Doubles(&kde.samples)),
                        ("bandwidth", P::Double(kde.bandwidth)),
                    ],
                ),
                DistributionKind::Categorical => return unsupported("a categorical node"),
                kind => match classes::saved_distribution(kind) {
                    Some(saved) => {
                        let mut params = named(saved.parameters);
                        if let Some(location) = saved.zero_location {
                            let zero = self.constant(&Value::F64(DoubleTensor::scalar(0.0)))?;
                            params.push((location, P::Saved(zero)));
                        }
                        (saved.class, params)
                    }
                    None => return unsupported("this distribution"),
                },
            },
            Op::Lambda(_) => return unsupported("a lambda"),
            Op::Cpt { .. } => return unsupported("a conditional probability table"),
            _ => return unsupported(node.op().name()),
        };
        self.emit(class, label, shape, params)
    }
}

struct StridedSlice {
    start: Vec<i64>,
    end: Vec<i64>,
    stride: Vec<i64>,
    upper_bound_stop: Vec<bool>,
    drop_dimension: Vec<bool>,
    ellipsis: Option<usize>,
}

/// numpy slice text as the parameters of Java's StridedSliceVertex.
fn strided_slice(spec: &str) -> std::result::Result<StridedSlice, String> {
    let mut s = StridedSlice {
        start: vec![],
        end: vec![],
        stride: vec![],
        upper_bound_stop: vec![],
        drop_dimension: vec![],
        ellipsis: None,
    };
    let items = parse_slice(spec).map_err(|_| format!("cannot parse slice {spec:?}"))?;
    for item in items {
        match item {
            SliceItem::Ellipsis => s.ellipsis = Some(s.start.len()),
            SliceItem::Index(i) => {
                let i = i as i64;
                s.start.push(i);
                s.end.push(i + 1);
                s.stride.push(1);
                s.upper_bound_stop.push(false);
                s.drop_dimension.push(true);
            }
            SliceItem::Range { start, stop, step } => {
                let step = step.map_or(1, |v| v as i64);
                s.start
                    .push(start.map_or(if step > 0 { 0 } else { -1 }, |v| v as i64));
                s.end.push(stop.map_or(0, |v| v as i64));
                s.stride.push(step);
                s.upper_bound_stop.push(stop.is_none());
                s.drop_dimension.push(false);
            }
            _ => return Err(format!("cannot parse slice {spec:?}")),
        }
    }
    Ok(s)
}

/// The saved form of a graph and the values in a state.
pub(crate) fn save_model(
    graph: &Graph,
    state: &State,
    metadata: &Metadata,
) -> Result<proto::ProtoModel> {
    let mut saver = Saver {
        graph,
        vertices: Vec::with_capacity(graph.len()),
        saved: vec![None; graph.len()],
    };
    for id in graph.ids() {
        let saved = saver.save_node(id)?;
        saver.saved[id.index()] = Some(saved);
    }
    let mut default_state = Vec::new();
    for id in graph.ids() {
        if let Some(value) = state.value(id) {
            default_state.push(proto::StoredValue {
                value: Some(proto_value(value)?),
                vertex_label: String::new(),
                id: Some(saver.id(id)?),
                is_observed: state.is_observed(id),
            });
        }
    }
    Ok(proto::ProtoModel {
        graph: Some(proto::Graph {
            vertices: saver.vertices,
            default_state,
        }),
        metadata: (!metadata.is_empty()).then(|| proto::ModelMetadata {
            metadata_info: metadata
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }),
    })
}
