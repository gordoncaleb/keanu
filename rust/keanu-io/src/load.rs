//! Loading a saved model: each saved vertex (identified by its Java class name and named parameters) becomes a
//! node, or a few nodes where the Rust operation is defined differently.

use crate::Metadata;
use crate::error::{IoError, Result};
use crate::proto;
use crate::proto::named_param::Param;
use keanu_distributions::Kde;
use keanu_graph::op::{
    ArgOp, BinaryOp, CompareOp, DistributionKind, LogicOp, Op, Predicate, ReduceOp, ScanOp, UnaryOp,
};
use keanu_graph::{DType, Graph, GraphBuilder, NodeId, State, Value};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};
use std::collections::HashMap;
use std::sync::Arc;

/// A loaded model: its graph, the saved values (observations are observed), the saved metadata, and the node each
/// saved vertex id became.
#[derive(Debug)]
pub struct Loaded {
    pub graph: Graph,
    pub state: State,
    pub metadata: Metadata,
    /// Saved vertex id (e.g. `"[3]"`) to node.
    pub ids: HashMap<String, NodeId>,
}

pub(crate) fn shape_of(dims: &[i64]) -> Result<Vec<usize>> {
    dims.iter()
        .map(|&d| {
            usize::try_from(d).map_err(|_| IoError::Decode(format!("negative dimension {d}")))
        })
        .collect()
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
pub fn load_model(model: &proto::ProtoModel) -> Result<Loaded> {
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| IoError::Decode("the model has no graph".into()))?;
    let builder = GraphBuilder::new();
    let mut ids: HashMap<String, NodeId> = HashMap::new();

    for vertex in &graph.vertices {
        let id = vertex.id.as_ref().map(|v| v.id.clone()).unwrap_or_default();
        let node = Loader {
            builder: &builder,
            vertex,
            ids: &ids,
        }
        .load()?;
        if !vertex.label.is_empty() {
            builder.set_label_of(node, &vertex.label);
        }
        ids.insert(id, node);
    }

    for stored in &graph.default_state {
        let node = match stored.id.as_ref().filter(|id| !id.id.is_empty()) {
            Some(id) => *ids
                .get(&id.id)
                .ok_or_else(|| IoError::UnknownVertex(id.id.clone()))?,
            None => (0..builder.node_count())
                .map(keanu_graph_node_id)
                .find(|&n| builder.label_of(n).as_deref() == Some(stored.vertex_label.as_str()))
                .ok_or_else(|| IoError::UnknownVertex(stored.vertex_label.clone()))?,
        };
        let value = match stored.value.as_ref().and_then(|v| v.value_type.as_ref()) {
            Some(proto::vertex_value::ValueType::DoubleVal(t)) => Value::F64(double_tensor(t)?),
            Some(proto::vertex_value::ValueType::IntVal(t)) => Value::I64(int_tensor(t)?),
            Some(proto::vertex_value::ValueType::BoolVal(t)) => Value::Bool(bool_tensor(t)?),
            Some(proto::vertex_value::ValueType::GenericVal(_)) => {
                return Err(IoError::UnsupportedVertex("generic (object) values".into()));
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

/// NodeIds are created by the builder; the loader looks nodes up by creation index.
fn keanu_graph_node_id(index: usize) -> NodeId {
    NodeId::from_index(index)
}

struct Loader<'a> {
    builder: &'a GraphBuilder,
    vertex: &'a proto::Vertex,
    ids: &'a HashMap<String, NodeId>,
}

impl Loader<'_> {
    fn class(&self) -> &str {
        self.vertex
            .vertex_type
            .rsplit('.')
            .next()
            .unwrap_or(&self.vertex.vertex_type)
    }

    fn invalid(&self, parameter: &str, message: impl Into<String>) -> IoError {
        IoError::InvalidParameter {
            vertex: self.class().to_owned(),
            parameter: parameter.to_owned(),
            message: message.into(),
        }
    }

    fn param(&self, name: &str) -> Option<&Param> {
        self.vertex
            .parameters
            .iter()
            .find(|p| p.name == name)
            .and_then(|p| p.param.as_ref())
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

    fn distribution(&self, kind: DistributionKind, parameters: &[&str]) -> Result<NodeId> {
        let parents = parameters
            .iter()
            .map(|p| self.parent(p))
            .collect::<Result<Vec<_>>>()?;
        let shape = self.shape()?;
        Ok(self
            .builder
            .add_distribution(kind, Some(&shape), &parents)?)
    }

    /// The half distributions are saved with a location parameter, which must be zero.
    fn half_distribution(
        &self,
        kind: DistributionKind,
        location: &str,
        scale: &str,
    ) -> Result<NodeId> {
        let location_node = self.parent(location)?;
        let is_zero_constant = matches!(
            self.builder_constant(location_node),
            Some(Value::F64(t)) if t.as_slice().iter().all(|&v| v == 0.0)
        );
        if !is_zero_constant {
            return Err(self.invalid(location, "must be the constant 0"));
        }
        let parents = [self.parent(scale)?];
        let shape = self.shape()?;
        Ok(self
            .builder
            .add_distribution(kind, Some(&shape), &parents)?)
    }

    fn builder_constant(&self, id: NodeId) -> Option<Value> {
        self.builder.constant_value(id)
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
        use DistributionKind as D;
        let unary = |op: UnaryOp| self.unary(Op::Unary(op));
        let binary = |op: BinaryOp| self.binary(Op::Binary(op));
        let compare = |op: CompareOp| self.binary(Op::Compare(op));
        let mask = |op: CompareOp| self.binary(Op::Mask(op));
        let predicate = |p: Predicate| self.unary(Op::Predicate(p));
        let isize_of = |v: i64| v as isize;

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

            "AbsVertex" => unary(UnaryOp::Abs),
            "SignVertex" => unary(UnaryOp::Sign),
            "CeilVertex" => unary(UnaryOp::Ceil),
            "FloorVertex" => unary(UnaryOp::Floor),
            "RoundVertex" => unary(UnaryOp::Round),
            "ExpVertex" => unary(UnaryOp::Exp),
            "Exp2Vertex" => unary(UnaryOp::Exp2),
            "ExpM1Vertex" => unary(UnaryOp::ExpM1),
            "LogVertex" => unary(UnaryOp::Ln),
            "Log1pVertex" => unary(UnaryOp::Ln1p),
            "Log2Vertex" => unary(UnaryOp::Log2),
            "Log10Vertex" => unary(UnaryOp::Log10),
            "SinVertex" => unary(UnaryOp::Sin),
            "CosVertex" => unary(UnaryOp::Cos),
            "TanVertex" => unary(UnaryOp::Tan),
            "ArcSinVertex" => unary(UnaryOp::Asin),
            "ArcCosVertex" => unary(UnaryOp::Acos),
            "ArcTanVertex" => unary(UnaryOp::Atan),
            "SinhVertex" => unary(UnaryOp::Sinh),
            "CoshVertex" => unary(UnaryOp::Cosh),
            "TanhVertex" => unary(UnaryOp::Tanh),
            "ArcSinhVertex" => unary(UnaryOp::Asinh),
            "ArcCoshVertex" => unary(UnaryOp::Acosh),
            "ArcTanhVertex" => unary(UnaryOp::Atanh),
            "SigmoidVertex" => unary(UnaryOp::Sigmoid),
            "LogGammaVertex" => unary(UnaryOp::LnGamma),
            "DigammaVertex" => unary(UnaryOp::Digamma),
            "TrigammaVertex" => unary(UnaryOp::Trigamma),

            "AdditionVertex" => binary(BinaryOp::Add),
            "DifferenceVertex" => binary(BinaryOp::Sub),
            "MultiplicationVertex" => binary(BinaryOp::Mul),
            "DivisionVertex" => binary(BinaryOp::Div),
            "PowerVertex" => binary(BinaryOp::Pow),
            "ModVertex" => binary(BinaryOp::Rem),
            "ArcTan2Vertex" => binary(BinaryOp::Atan2),
            "LogAddExpVertex" => binary(BinaryOp::LogAddExp),
            "LogAddExp2Vertex" => binary(BinaryOp::LogAddExp2),
            "SafeLogTimesVertex" => binary(BinaryOp::SafeLnTimes),
            // saved as WhereVertex parameters; the predicate is derived again from then / else
            "MaxVertex" => self.node(
                Op::Binary(BinaryOp::Max),
                &[self.parent("then")?, self.parent("else")?],
            ),
            "MinVertex" => self.node(
                Op::Binary(BinaryOp::Min),
                &[self.parent("then")?, self.parent("else")?],
            ),

            "EqualsVertex" => compare(CompareOp::Eq),
            "NotEqualsVertex" => compare(CompareOp::Ne),
            "GreaterThanVertex" => compare(CompareOp::Gt),
            "GreaterThanOrEqualVertex" => compare(CompareOp::Ge),
            "LessThanVertex" => compare(CompareOp::Lt),
            "LessThanOrEqualVertex" => compare(CompareOp::Le),
            "NumericalEqualsVertex" => self.node(
                Op::ApproxEq(self.double("epsilon")?),
                &[self.parent("a")?, self.parent("b")?],
            ),
            "GreaterThanMaskVertex" => mask(CompareOp::Gt),
            "GreaterThanOrEqualToMaskVertex" => mask(CompareOp::Ge),
            "LessThanMaskVertex" => mask(CompareOp::Lt),
            "LessThanOrEqualToMaskVertex" => mask(CompareOp::Le),

            "AndBinaryVertex" => self.binary(Op::Logic(LogicOp::And)),
            "OrBinaryVertex" => self.binary(Op::Logic(LogicOp::Or)),
            "XorBinaryVertex" => self.binary(Op::Logic(LogicOp::Xor)),
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
            "IsNaNVertex" => predicate(Predicate::IsNan),
            "NotNaNVertex" => predicate(Predicate::NotNan),
            "IsFiniteVertex" => predicate(Predicate::IsFinite),
            "IsInfiniteVertex" => predicate(Predicate::IsInfinite),
            "IsPositiveInfinityVertex" => predicate(Predicate::IsPositiveInfinity),
            "IsNegativeInfinityVertex" => predicate(Predicate::IsNegativeInfinity),

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
            "BroadcastVertex" => self.unary(Op::Broadcast(self.unsigned("toShape")?)),
            "SliceVertex" => {
                let dimension = usize::try_from(self.int("dimension")?)
                    .map_err(|_| self.invalid("dimension", "negative"))?;
                let mut items = vec![":".to_owned(); dimension];
                items.push(self.int("index")?.to_string());
                self.unary(Op::Slice(items.join(", ")))
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
                let label = self.string("label")?;
                if !label.is_empty() {
                    self.builder.set_label_of(proxy, &label);
                }
                Ok(proxy)
            }

            "GaussianVertex" => self.distribution(D::Gaussian, &["mu", "sigma"]),
            "HalfGaussianVertex" => self.half_distribution(D::HalfGaussian, "mu", "sigma"),
            "CauchyVertex" => self.distribution(D::Cauchy, &["location", "scale"]),
            "HalfCauchyVertex" => self.half_distribution(D::HalfCauchy, "location", "scale"),
            "BetaVertex" => self.distribution(D::Beta, &["alpha", "beta"]),
            "ChiSquaredVertex" => self.distribution(D::ChiSquared, &["k"]),
            "DirichletVertex" => self.distribution(D::Dirichlet, &["concentration"]),
            "ExponentialVertex" => self.distribution(D::Exponential, &["rate"]),
            "GammaVertex" => self.distribution(D::Gamma, &["theta", "k"]),
            "InverseGammaVertex" => self.distribution(D::InverseGamma, &["alpha", "beta"]),
            "LaplaceVertex" => self.distribution(D::Laplace, &["mu", "beta"]),
            "LogisticVertex" => self.distribution(D::Logistic, &["mu", "s"]),
            "LogNormalVertex" => self.distribution(D::LogNormal, &["mu", "sigma"]),
            "MultivariateGaussianVertex" => {
                self.distribution(D::MultivariateGaussian, &["mu", "covariance"])
            }
            "ParetoVertex" => self.distribution(D::Pareto, &["location", "scale"]),
            "StudentTVertex" => self.distribution(D::StudentT, &["v"]),
            "TriangularVertex" => self.distribution(D::Triangular, &["xMin", "xMax", "c"]),
            "UniformVertex" => self.distribution(D::Uniform, &["xMin", "xMax"]),
            "KDEVertex" => {
                let samples = match self.require("samples")? {
                    Param::DoubleTensorParam(t) => double_tensor(t)?.into_vec(),
                    _ => return Err(self.invalid("samples", "expected a double tensor")),
                };
                let kde = Kde::new(samples, self.double("bandwidth")?)
                    .map_err(|e| self.invalid("samples", e.to_string()))?;
                Ok(self.builder.add_distribution(
                    D::Kde(Arc::new(kde)),
                    Some(&self.shape()?),
                    &[],
                )?)
            }
            "BernoulliVertex" => self.distribution(D::Bernoulli, &["probTrue"]),
            "BinomialVertex" => self.distribution(D::Binomial, &["p", "n"]),
            "GeometricVertex" => self.distribution(D::Geometric, &["p"]),
            "MultinomialVertex" => self.distribution(D::Multinomial, &["n", "p"]),
            "PoissonVertex" => self.distribution(D::Poisson, &["mu"]),
            "UniformIntVertex" => self.distribution(D::UniformInt, &["min", "max"]),

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
