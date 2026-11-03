//! Saving a graph in the saved model schema, with Java vertex class names and parameter names so the Java
//! implementation can load it.

use crate::Metadata;
use crate::classes::JAVA_CLASSES;
use crate::error::{IoError, Result};
use crate::proto;
use crate::proto::named_param::Param;
use keanu_graph::op::{
    ArgOp, BinaryOp, CompareOp, DistributionKind, LogicOp, Op, Predicate, ReduceOp, ScanOp, UnaryOp,
};
use keanu_graph::{DType, Graph, NodeId, State, Value};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};
use std::collections::HashMap;

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
    JAVA_CLASSES
        .iter()
        .find(|(s, _)| *s == simple)
        .map(|(_, fq)| *fq)
        .ok_or_else(|| IoError::NotSaveable(format!("no saved type is recorded for {simple}")))
}

/// One parameter of a saved vertex.
enum P {
    Parent(NodeId),
    Saved(String),
    Parents(Vec<NodeId>),
    Int(i64),
    Double(f64),
    Bool(bool),
    Str(String),
    Longs(Vec<i64>),
    Ints(Vec<i64>),
    DoubleTensor(DoubleTensor),
    IntTensor(IntTensor),
    BoolTensor(BoolTensor),
}

struct Saver<'g> {
    graph: &'g Graph,
    vertices: Vec<proto::Vertex>,
    ids: HashMap<NodeId, String>,
}

impl Saver<'_> {
    fn next_id(&self) -> String {
        format!("[{}]", self.vertices.len())
    }

    fn id(&self, node: NodeId) -> Result<String> {
        self.ids.get(&node).cloned().ok_or_else(|| {
            IoError::NotSaveable(format!("node {node:?} was not saved before its children"))
        })
    }

    fn param(&self, name: &str, p: P) -> Result<proto::NamedParam> {
        let vertex_id = |id: String| proto::VertexId { id };
        let param = match p {
            P::Parent(node) => Param::ParentVertex(vertex_id(self.id(node)?)),
            P::Saved(id) => Param::ParentVertex(vertex_id(id)),
            P::Parents(nodes) => Param::VertexArrayParam(proto::VertexArray {
                values: nodes
                    .into_iter()
                    .map(|n| self.id(n).map(vertex_id))
                    .collect::<Result<_>>()?,
            }),
            P::Int(v) => Param::IntParam(
                i32::try_from(v)
                    .map_err(|_| IoError::NotSaveable(format!("{name} = {v} is too large")))?,
            ),
            P::Double(v) => Param::DoubleParam(v),
            P::Bool(v) => Param::BoolParam(v),
            P::Str(v) => Param::StringParam(v),
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
            P::DoubleTensor(t) => Param::DoubleTensorParam(proto_double(&t)),
            P::IntTensor(t) => Param::IntTensorParam(proto_int(&t)?),
            P::BoolTensor(t) => Param::BoolTensorParam(proto_bool(&t)),
        };
        Ok(proto::NamedParam {
            name: name.to_owned(),
            param: Some(param),
        })
    }

    /// Emits a vertex, returning its saved id.
    fn emit(
        &mut self,
        class: &str,
        label: &str,
        shape: &[usize],
        params: Vec<(&str, P)>,
    ) -> Result<String> {
        let mut parameters = params
            .into_iter()
            .map(|(name, p)| self.param(name, p))
            .collect::<Result<Vec<_>>>()?;
        // the Java saver writes parameters sorted by name
        parameters.sort_by(|a, b| a.name.cmp(&b.name));
        let id = self.next_id();
        self.vertices.push(proto::Vertex {
            id: Some(proto::VertexId { id: id.clone() }),
            label: label.to_owned(),
            vertex_type: java_class(class)?.to_owned(),
            parameters,
            shape: shape.iter().map(|&d| d as i64).collect(),
            data_type: 0,
            metadata: Vec::new(),
        });
        Ok(id)
    }

    /// Emits an unlabelled helper constant.
    fn constant(&mut self, value: Value) -> Result<String> {
        let shape = value.shape().to_vec();
        match value {
            Value::F64(t) => self.emit(
                "ConstantDoubleVertex",
                "",
                &shape,
                vec![("constant", P::DoubleTensor(t))],
            ),
            Value::I64(t) => self.emit(
                "ConstantIntegerVertex",
                "",
                &shape,
                vec![("constant", P::IntTensor(t))],
            ),
            Value::Bool(t) => self.emit(
                "ConstantBooleanVertex",
                "",
                &shape,
                vec![("constant", P::BoolTensor(t))],
            ),
            _ => Err(IoError::NotSaveable(
                "a constant of an unsupported type".into(),
            )),
        }
    }

    fn save_node(&mut self, id: NodeId) -> Result<String> {
        let node = self.graph.node(id);
        let label = node.label().unwrap_or("").to_owned();
        let shape = node.shape().to_vec();
        let parents = node.parents().to_vec();
        let p = |i: usize| P::Parent(parents[i]);
        let input = || vec![("inputVertex", P::Parent(parents[0]))];
        let left_right = || {
            vec![
                ("left", P::Parent(parents[0])),
                ("right", P::Parent(parents[1])),
            ]
        };
        let unsupported = |what: &str| {
            Err(IoError::NotSaveable(format!(
                "{what} (node {id:?}) has no saved form"
            )))
        };
        let input_dtype = parents.first().map(|&p| self.graph.node(p).dtype());

        let (class, params): (&str, Vec<(&str, P)>) = match node.op() {
            Op::Constant(value) => match value {
                Value::F64(t) => (
                    "ConstantDoubleVertex",
                    vec![("constant", P::DoubleTensor(t.clone()))],
                ),
                Value::I64(t) => (
                    "ConstantIntegerVertex",
                    vec![("constant", P::IntTensor(t.clone()))],
                ),
                Value::Bool(t) => (
                    "ConstantBooleanVertex",
                    vec![("constant", P::BoolTensor(t.clone()))],
                ),
                _ => return unsupported("a constant of an unsupported type"),
            },
            Op::Unary(op) => match op {
                UnaryOp::Neg => {
                    let minus_one = if node.dtype() == DType::I64 {
                        Value::I64(IntTensor::scalar(-1))
                    } else {
                        Value::F64(DoubleTensor::scalar(-1.0))
                    };
                    let c = self.constant(minus_one)?;
                    (
                        "MultiplicationVertex",
                        vec![("left", p(0)), ("right", P::Saved(c))],
                    )
                }
                UnaryOp::Reciprocal => {
                    let c = self.constant(Value::F64(DoubleTensor::scalar(1.0)))?;
                    (
                        "DivisionVertex",
                        vec![("left", P::Saved(c)), ("right", p(0))],
                    )
                }
                UnaryOp::Sqrt => {
                    let c = self.constant(Value::F64(DoubleTensor::scalar(0.5)))?;
                    ("PowerVertex", vec![("left", p(0)), ("right", P::Saved(c))])
                }
                other => (
                    unary_class(*other)
                        .ok_or_else(|| IoError::NotSaveable(format!("{other:?}")))?,
                    input(),
                ),
            },
            Op::Binary(op) => match op {
                BinaryOp::Max => ("MaxVertex", vec![("then", p(0)), ("else", p(1))]),
                BinaryOp::Min => ("MinVertex", vec![("then", p(0)), ("else", p(1))]),
                other => (
                    binary_class(*other)
                        .ok_or_else(|| IoError::NotSaveable(format!("{other:?}")))?,
                    left_right(),
                ),
            },
            Op::Compare(op) => match compare_class(*op) {
                Some(class) => (class, left_right()),
                None => return unsupported("an unsupported comparison"),
            },
            Op::Mask(op) => (
                match op {
                    CompareOp::Gt => "GreaterThanMaskVertex",
                    CompareOp::Ge => "GreaterThanOrEqualToMaskVertex",
                    CompareOp::Lt => "LessThanMaskVertex",
                    CompareOp::Le => "LessThanOrEqualToMaskVertex",
                    CompareOp::Eq | CompareOp::Ne => return unsupported("an equality mask"),
                    _ => return unsupported("a mask of an unsupported comparison"),
                },
                left_right(),
            ),
            Op::ApproxEq(epsilon) => (
                "NumericalEqualsVertex",
                vec![("a", p(0)), ("b", p(1)), ("epsilon", P::Double(*epsilon))],
            ),
            Op::Logic(op) => (
                match op {
                    LogicOp::And => "AndBinaryVertex",
                    LogicOp::Or => "OrBinaryVertex",
                    LogicOp::Xor => "XorBinaryVertex",
                    _ => return unsupported("an unsupported logical operation"),
                },
                left_right(),
            ),
            Op::Not => ("NotBinaryVertex", input()),
            Op::All => ("AllTrueVertex", input()),
            Op::Any => ("AnyTrueVertex", input()),
            Op::Predicate(predicate) => (
                match predicate {
                    Predicate::IsNan => "IsNaNVertex",
                    Predicate::NotNan => "NotNaNVertex",
                    Predicate::IsFinite => "IsFiniteVertex",
                    Predicate::IsInfinite => "IsInfiniteVertex",
                    Predicate::IsPositiveInfinity => "IsPositiveInfinityVertex",
                    Predicate::IsNegativeInfinity => "IsNegativeInfinityVertex",
                    _ => return unsupported("an unsupported predicate"),
                },
                input(),
            ),
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
                let rank = self.graph.node(parents[0]).shape().len();
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
                    (
                        "upperBoundStop",
                        P::BoolTensor(BoolTensor::vector(s.upper_bound_stop)),
                    ),
                    (
                        "dropDimension",
                        P::BoolTensor(BoolTensor::vector(s.drop_dimension)),
                    ),
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
                    ("operands", P::Parents(parents.clone())),
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
                let input_shape = self.graph.node(parents[0]).shape().to_vec();
                let lower_input = self.emit(
                    "TriLowerVertex",
                    "",
                    &input_shape,
                    vec![("inputVertex", p(0)), ("k", P::Int(0))],
                )?;
                let c = self.emit(
                    "CholeskyInverseVertex",
                    "",
                    &shape,
                    vec![("inputVertex", P::Saved(lower_input))],
                )?;
                let strictly_lower = self.emit(
                    "TriLowerVertex",
                    "",
                    &shape,
                    vec![("inputVertex", P::Saved(c.clone())), ("k", P::Int(1))],
                )?;
                let rank = shape.len();
                let mut permutation: Vec<i64> = (0..rank as i64).collect();
                permutation.swap(rank - 1, rank - 2);
                let transposed = self.emit(
                    "PermuteVertex",
                    "",
                    &shape,
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
                    ("selectVertices", P::Parents(parents[1..].to_vec())),
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
                    ("error", P::Str(message.clone().unwrap_or_default())),
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
                let proxy_label = if label.is_empty() {
                    format!("proxy_{}", id.index())
                } else {
                    label.clone()
                };
                (
                    class,
                    vec![("parent", p(0)), ("label", P::Str(proxy_label))],
                )
            }
            Op::Distribution(distribution) => {
                use DistributionKind as D;
                let named = |names: &[&'static str]| -> Vec<(&'static str, P)> {
                    names
                        .iter()
                        .enumerate()
                        .map(|(i, &n)| (n, P::Parent(parents[i])))
                        .collect()
                };
                match &distribution.kind {
                    D::Gaussian => ("GaussianVertex", named(&["mu", "sigma"])),
                    D::HalfGaussian => {
                        let zero = self.constant(Value::F64(DoubleTensor::scalar(0.0)))?;
                        (
                            "HalfGaussianVertex",
                            vec![("mu", P::Saved(zero)), ("sigma", p(0))],
                        )
                    }
                    D::Cauchy => ("CauchyVertex", named(&["location", "scale"])),
                    D::HalfCauchy => {
                        let zero = self.constant(Value::F64(DoubleTensor::scalar(0.0)))?;
                        (
                            "HalfCauchyVertex",
                            vec![("location", P::Saved(zero)), ("scale", p(0))],
                        )
                    }
                    D::Beta => ("BetaVertex", named(&["alpha", "beta"])),
                    D::ChiSquared => ("ChiSquaredVertex", named(&["k"])),
                    D::Dirichlet => ("DirichletVertex", named(&["concentration"])),
                    D::Exponential => ("ExponentialVertex", named(&["rate"])),
                    D::Gamma => ("GammaVertex", named(&["theta", "k"])),
                    D::InverseGamma => ("InverseGammaVertex", named(&["alpha", "beta"])),
                    D::Laplace => ("LaplaceVertex", named(&["mu", "beta"])),
                    D::Logistic => ("LogisticVertex", named(&["mu", "s"])),
                    D::LogNormal => ("LogNormalVertex", named(&["mu", "sigma"])),
                    D::MultivariateGaussian => {
                        ("MultivariateGaussianVertex", named(&["mu", "covariance"]))
                    }
                    D::Pareto => ("ParetoVertex", named(&["location", "scale"])),
                    D::StudentT => ("StudentTVertex", named(&["v"])),
                    D::Triangular => ("TriangularVertex", named(&["xMin", "xMax", "c"])),
                    D::Uniform => ("UniformVertex", named(&["xMin", "xMax"])),
                    D::Kde(kde) => (
                        "KDEVertex",
                        vec![
                            (
                                "samples",
                                P::DoubleTensor(DoubleTensor::vector(kde.samples.clone())),
                            ),
                            ("bandwidth", P::Double(kde.bandwidth)),
                        ],
                    ),
                    D::Bernoulli => ("BernoulliVertex", named(&["probTrue"])),
                    D::Binomial => ("BinomialVertex", named(&["p", "n"])),
                    D::Geometric => ("GeometricVertex", named(&["p"])),
                    D::Multinomial => ("MultinomialVertex", named(&["n", "p"])),
                    D::Poisson => ("PoissonVertex", named(&["mu"])),
                    D::UniformInt => ("UniformIntVertex", named(&["min", "max"])),
                    D::Categorical => return unsupported("a categorical node"),
                    _ => return unsupported("this distribution"),
                }
            }
            Op::Lambda(_) => return unsupported("a lambda"),
            Op::Cpt { .. } => return unsupported("a conditional probability table"),
            _ => return unsupported(node.op().name()),
        };
        self.emit(class, &label, &shape, params)
    }
}

fn unary_class(op: UnaryOp) -> Option<&'static str> {
    use UnaryOp::*;
    Some(match op {
        Abs => "AbsVertex",
        Sign => "SignVertex",
        Exp => "ExpVertex",
        Exp2 => "Exp2Vertex",
        ExpM1 => "ExpM1Vertex",
        Ln => "LogVertex",
        Ln1p => "Log1pVertex",
        Log2 => "Log2Vertex",
        Log10 => "Log10Vertex",
        Sin => "SinVertex",
        Cos => "CosVertex",
        Tan => "TanVertex",
        Asin => "ArcSinVertex",
        Acos => "ArcCosVertex",
        Atan => "ArcTanVertex",
        Sinh => "SinhVertex",
        Cosh => "CoshVertex",
        Tanh => "TanhVertex",
        Asinh => "ArcSinhVertex",
        Acosh => "ArcCoshVertex",
        Atanh => "ArcTanhVertex",
        Sigmoid => "SigmoidVertex",
        Floor => "FloorVertex",
        Ceil => "CeilVertex",
        Round => "RoundVertex",
        LnGamma => "LogGammaVertex",
        Digamma => "DigammaVertex",
        Trigamma => "TrigammaVertex",
        _ => return None,
    })
}

fn binary_class(op: BinaryOp) -> Option<&'static str> {
    use BinaryOp::*;
    Some(match op {
        Add => "AdditionVertex",
        Sub => "DifferenceVertex",
        Mul => "MultiplicationVertex",
        Div => "DivisionVertex",
        Pow => "PowerVertex",
        Rem => "ModVertex",
        Atan2 => "ArcTan2Vertex",
        LogAddExp => "LogAddExpVertex",
        LogAddExp2 => "LogAddExp2Vertex",
        SafeLnTimes => "SafeLogTimesVertex",
        Max => "MaxVertex",
        Min => "MinVertex",
        _ => return None,
    })
}

fn compare_class(op: CompareOp) -> Option<&'static str> {
    Some(match op {
        CompareOp::Gt => "GreaterThanVertex",
        CompareOp::Ge => "GreaterThanOrEqualVertex",
        CompareOp::Lt => "LessThanVertex",
        CompareOp::Le => "LessThanOrEqualVertex",
        CompareOp::Eq => "EqualsVertex",
        CompareOp::Ne => "NotEqualsVertex",
        _ => return None,
    })
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
    let parse = |t: &str| -> std::result::Result<Option<i64>, String> {
        if t.is_empty() {
            Ok(None)
        } else {
            t.parse()
                .map(Some)
                .map_err(|_| format!("cannot parse slice {spec:?}"))
        }
    };
    for item in spec.split(',').map(str::trim) {
        if item == "..." {
            s.ellipsis = Some(s.start.len());
            continue;
        }
        let parts: Vec<&str> = item.split(':').collect();
        match parts.as_slice() {
            [index] => {
                let i = parse(index)?.ok_or_else(|| format!("empty index in {spec:?}"))?;
                s.start.push(i);
                s.end.push(i + 1);
                s.stride.push(1);
                s.upper_bound_stop.push(false);
                s.drop_dimension.push(true);
            }
            [start, stop] | [start, stop, _] => {
                let step = if parts.len() == 3 {
                    parse(parts[2])?.unwrap_or(1)
                } else {
                    1
                };
                let first = parse(start)?.unwrap_or(if step > 0 { 0 } else { -1 });
                let end = parse(stop)?;
                s.start.push(first);
                s.end.push(end.unwrap_or(0));
                s.stride.push(step);
                s.upper_bound_stop.push(end.is_none());
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
        vertices: Vec::new(),
        ids: HashMap::new(),
    };
    for id in graph.ids() {
        let saved = saver.save_node(id)?;
        saver.ids.insert(id, saved);
    }
    let mut default_state = Vec::new();
    for id in graph.ids() {
        if let Some(value) = state.value(id) {
            default_state.push(proto::StoredValue {
                value: Some(proto_value(value)?),
                vertex_label: String::new(),
                id: Some(proto::VertexId { id: saver.id(id)? }),
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
