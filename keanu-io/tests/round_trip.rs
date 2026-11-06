//! Saving and loading round trips.
//!
//! - Every operation and distribution kind saves and loads back as the same node, through protobuf and JSON.
//! - The golden models (converted from the files the Java implementation saved) load and reproduce the values Java
//!   computed, and saving a loaded golden model reproduces its file byte for byte (the format is stable).
//! - Values, observations, labels and metadata survive; invalid and newer files are rejected.

use keanu_distributions::{Kde, KeanuRng};
use keanu_graph::op::{
    ArgOp, BinaryOp, CompareOp, CumulativeOp, DistributionKind, LogicOp, Op, Predicate, ReduceOp,
    UnaryOp,
};
use keanu_graph::{CustomDistribution, DType, Graph, GraphBuilder, Model, NodeId, State, Value};
use keanu_io::{IoError, Loaded, Metadata, load_json, load_protobuf, save_json, save_protobuf};
use keanu_parity::{close, golden_dir, number, parse_tensor};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};
use serde_json::Value as Json;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

fn recompute(graph: &Graph, state: &State) -> State {
    let mut state = state.clone();
    for id in graph.ids() {
        let node = &graph[id];
        if !node.is_probabilistic()
            && !matches!(node.op(), Op::Constant(_))
            && !state.is_observed(id)
        {
            state.clear(id).unwrap();
        }
    }
    let all: Vec<_> = graph.ids().collect();
    state
        .lazy_eval(graph, &all, &mut KeanuRng::seed_from_u64(0))
        .unwrap();
    state
}

fn flatten(value: &Value) -> (Vec<usize>, Vec<f64>) {
    match value {
        Value::F64(t) => (t.shape().to_vec(), t.as_slice().to_vec()),
        Value::I64(t) => (
            t.shape().to_vec(),
            t.as_slice().iter().map(|&x| x as f64).collect(),
        ),
        Value::Bool(t) => (
            t.shape().to_vec(),
            t.as_slice()
                .iter()
                .map(|&x| f64::from(u8::from(x)))
                .collect(),
        ),
        other => panic!("unexpected value {other:?}"),
    }
}

/// Everything a round trip must preserve about a graph and state, exactly: each node's operation, parents, label,
/// type and shape, and its stored value and whether it is observed.
fn describe(graph: &Graph, state: &State) -> Vec<String> {
    graph
        .ids()
        .map(|id| {
            let node = &graph[id];
            format!(
                "{} {:?} parents {:?} label {:?} {:?}{:?} value {:?} observed {}",
                id.index(),
                node.op(),
                node.parents(),
                node.label(),
                node.dtype(),
                node.shape(),
                state.value(id),
                state.is_observed(id)
            )
        })
        .collect()
}

/// Saves and loads through protobuf and JSON, checking the loaded graph and state are the same as the saved ones.
fn assert_round_trips(context: &str, graph: &Graph, state: &State) {
    let metadata: Metadata = [("name".to_owned(), context.to_owned())].into();
    let expected = describe(graph, state);
    let pb = save_protobuf(graph, state, &metadata).unwrap_or_else(|e| panic!("{context}: {e}"));
    let json = save_json(graph, state, &metadata).unwrap_or_else(|e| panic!("{context}: {e}"));
    for (format, loaded) in [("protobuf", load_protobuf(&pb)), ("json", load_json(&json))] {
        let loaded = loaded.unwrap_or_else(|e| panic!("{context} ({format}): {e}"));
        assert_eq!(loaded.metadata, metadata, "{context} ({format})");
        assert_eq!(
            describe(&loaded.graph, &loaded.state),
            expected,
            "{context} ({format})"
        );
    }
}

// ----- every operation and distribution -----

fn constant(b: &GraphBuilder, value: Value) -> NodeId {
    b.add_node(Op::Constant(value), &[]).unwrap()
}

fn f64s(shape: &[usize], values: &[f64]) -> Value {
    Value::F64(DoubleTensor::new(shape, values.to_vec()).unwrap())
}

fn scalar(v: f64) -> Value {
    Value::F64(DoubleTensor::scalar(v))
}

fn int(v: i64) -> Value {
    Value::I64(IntTensor::scalar(v))
}

fn boolean(v: bool) -> Value {
    Value::Bool(BoolTensor::scalar(v))
}

fn matrix() -> Value {
    f64s(&[2, 3], &[0.2, 0.5, 0.7, 0.3, 0.9, 0.4])
}

fn square() -> Value {
    f64s(&[3, 3], &[4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0])
}

fn lower() -> Value {
    f64s(&[3, 3], &[2.0, 0.0, 0.0, 0.5, 1.5, 0.0, 0.2, 0.1, 1.2])
}

/// A graph whose last node applies `op` to constant parents with the given values.
fn op_case(op: Op, parents: &[Value]) -> (String, Graph, State) {
    let b = GraphBuilder::new();
    let ids: Vec<NodeId> = parents.iter().map(|v| constant(&b, v.clone())).collect();
    let name = format!("{op:?}");
    let id = b
        .add_node(op, &ids)
        .unwrap_or_else(|e| panic!("can't build {name}: {e}"));
    b.set_label_of(id, "x").unwrap();
    let (graph, state) = b.build().unwrap();
    (name, graph, state)
}

const UNARY: &[UnaryOp] = {
    use UnaryOp::*;
    &[
        Abs, Neg, Sign, Reciprocal, Exp, Exp2, ExpM1, Ln, Ln1p, Log2, Log10, Sqrt, Sin, Cos, Tan,
        Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Sigmoid, Floor, Ceil, Round,
        LnGamma, Digamma, Trigamma, Erf, Erfc, ErfInv, LogNdtr, Ndtri, Softplus, Logit, Log1mExp,
    ]
};

const BINARY: &[BinaryOp] = {
    use BinaryOp::*;
    &[
        Add,
        Sub,
        Mul,
        Div,
        Pow,
        Max,
        Min,
        Rem,
        Atan2,
        LogAddExp,
        LogAddExp2,
        SafeLnTimes,
        LogBeta,
    ]
};

const COMPARE: &[CompareOp] = {
    use CompareOp::*;
    &[Gt, Ge, Lt, Le, Eq, Ne]
};

const PREDICATES: &[Predicate] = {
    use Predicate::*;
    &[
        IsNan,
        NotNan,
        IsFinite,
        IsInfinite,
        IsPositiveInfinity,
        IsNegativeInfinity,
    ]
};

/// One case per variant of every deterministic operation (and per variant of their kinds).
fn op_cases() -> Vec<(String, Graph, State)> {
    let mut cases = vec![
        op_case(Op::Constant(matrix()), &[]),
        op_case(
            Op::Constant(f64s(&[3], &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY])),
            &[],
        ),
        op_case(
            Op::Constant(Value::F64(DoubleTensor::full(&[0, 2], 0.0))),
            &[],
        ),
        op_case(
            Op::Constant(Value::I64(IntTensor::vector(vec![i64::MIN, -1, i64::MAX]))),
            &[],
        ),
        op_case(
            Op::Constant(Value::Bool(BoolTensor::vector(vec![true, false]))),
            &[],
        ),
    ];
    for &op in UNARY {
        cases.push(op_case(Op::Unary(op), &[scalar(0.5)]));
    }
    for &op in BINARY {
        let parents = if op == BinaryOp::Rem {
            [int(7), int(3)]
        } else {
            [scalar(0.5), scalar(2.0)]
        };
        cases.push(op_case(Op::Binary(op), &parents));
    }
    for &op in COMPARE {
        cases.push(op_case(Op::Compare(op), &[scalar(0.5), scalar(2.0)]));
        cases.push(op_case(Op::Mask(op), &[scalar(0.5), scalar(2.0)]));
    }
    cases.push(op_case(Op::ApproxEq(1e-3), &[scalar(0.5), scalar(0.5)]));
    cases.push(op_case(
        Op::ApproxEq(f64::INFINITY),
        &[scalar(0.5), scalar(0.5)],
    ));
    for op in [LogicOp::And, LogicOp::Or, LogicOp::Xor] {
        cases.push(op_case(Op::Logic(op), &[boolean(true), boolean(false)]));
    }
    let bools = Value::Bool(BoolTensor::vector(vec![true, false]));
    for op in [Op::Not, Op::All, Op::Any] {
        cases.push(op_case(op, std::slice::from_ref(&bools)));
    }
    for &p in PREDICATES {
        cases.push(op_case(Op::Predicate(p), &[scalar(f64::NAN)]));
    }
    use ReduceOp::*;
    for kind in [Sum, Product, Mean, Max, Min, StdDev, LogSumExp] {
        cases.push(op_case(Op::Reduce { kind, axes: None }, &[matrix()]));
    }
    for (kind, axes) in [
        (Sum, vec![-1]),
        (Product, vec![0, 1]),
        (Mean, vec![]),
        (LogSumExp, vec![0]),
    ] {
        cases.push(op_case(
            Op::Reduce {
                kind,
                axes: Some(axes),
            },
            &[matrix()],
        ));
    }
    cases.push(op_case(Op::Standardize, &[matrix()]));
    cases.push(op_case(Op::Softmax { axis: -1 }, &[matrix()]));
    cases.push(op_case(Op::LogSoftmax { axis: 0 }, &[matrix()]));
    for kind in [CumulativeOp::CumSum, CumulativeOp::CumProd] {
        cases.push(op_case(Op::Cumulative { kind, axis: -1 }, &[matrix()]));
    }
    for kind in [
        ArgOp::ArgMax,
        ArgOp::ArgMin,
        ArgOp::NanArgMax,
        ArgOp::NanArgMin,
    ] {
        cases.push(op_case(Op::ArgReduce { kind, axis: None }, &[matrix()]));
    }
    cases.push(op_case(
        Op::ArgReduce {
            kind: ArgOp::ArgMax,
            axis: Some(-1),
        },
        &[matrix()],
    ));
    let vector = f64s(&[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    cases.extend([
        op_case(Op::Reshape(vec![3, -1]), &[matrix()]),
        op_case(Op::Permute(vec![1, 0]), &[matrix()]),
        op_case(Op::Transpose, &[matrix()]),
        op_case(Op::Broadcast(vec![4, 2, 3]), &[matrix()]),
        op_case(Op::Slice("::-1, ..., 1:".into()), &[matrix()]),
        op_case(Op::Concat(-1), &[matrix(), matrix()]),
        op_case(
            Op::Concat(0),
            &[
                Value::I64(IntTensor::vector(vec![1])),
                Value::I64(IntTensor::vector(vec![2, 3])),
            ],
        ),
        op_case(Op::Take(vec![1, 2]), &[matrix()]),
        op_case(
            Op::Gather { axis: -1 },
            &[matrix(), Value::I64(IntTensor::vector(vec![2, 0, 2]))],
        ),
        op_case(
            Op::ScatterAdd { axis: 0, size: 3 },
            &[matrix(), Value::I64(IntTensor::vector(vec![2, 2]))],
        ),
        op_case(Op::Stack(1), &[matrix(), matrix(), matrix()]),
        op_case(Op::Diag, &[f64s(&[3], &[1.0, 2.0, 3.0])]),
        op_case(Op::DiagPart, &[square()]),
        op_case(Op::Tril(-1), &[square()]),
        op_case(Op::Triu(1), &[square()]),
        op_case(
            Op::FillTriangular {
                upper: false,
                lower: true,
            },
            std::slice::from_ref(&vector),
        ),
        op_case(
            Op::FillTriangular {
                upper: true,
                lower: true,
            },
            std::slice::from_ref(&vector),
        ),
        op_case(Op::TrianglePart { upper: true }, &[square()]),
        op_case(
            Op::MatMul {
                transpose_left: true,
                transpose_right: false,
            },
            &[matrix(), matrix()],
        ),
        op_case(
            Op::TensorMultiply {
                left_axes: vec![1],
                right_axes: vec![0],
            },
            &[matrix(), square()],
        ),
        op_case(Op::Inverse, &[square()]),
        op_case(Op::Determinant, &[square()]),
        op_case(Op::Cholesky, &[square()]),
        op_case(Op::CholeskyInverse, &[lower()]),
        op_case(
            Op::SolveTriangular {
                lower: true,
                transpose: true,
            },
            &[lower(), f64s(&[3], &[1.0, -2.0, 0.5])],
        ),
        op_case(
            Op::SolveTriangular {
                lower: false,
                transpose: false,
            },
            &[square(), f64s(&[3, 2], &[0.2, 0.5, 0.7, 0.3, 0.9, 0.4])],
        ),
        op_case(Op::Solve, &[square(), f64s(&[3, 1], &[1.0, -2.0, 0.5])]),
        op_case(Op::LogAbsDeterminant, &[square()]),
        op_case(Op::DeterminantSign, &[square()]),
        op_case(Op::Eigvalsh, &[square()]),
        op_case(Op::EighVectors, &[square()]),
        op_case(Op::MatrixExp, &[square()]),
        op_case(Op::Where, &[boolean(true), scalar(1.0), scalar(2.0)]),
        op_case(Op::Multiplexer, &[int(1), scalar(1.0), scalar(2.0)]),
        op_case(
            Op::Cpt {
                conditions: 2,
                keys: vec![vec![0, 1], vec![1, i64::MAX]],
            },
            &[boolean(true), int(3), scalar(1.0), scalar(2.0), scalar(3.0)],
        ),
        op_case(
            Op::SetWithMask,
            &[
                f64s(&[2], &[1.0, 2.0]),
                f64s(&[2], &[1.0, 0.0]),
                scalar(5.0),
            ],
        ),
        op_case(Op::ReplaceNan(0.5), &[scalar(f64::NAN)]),
        op_case(Op::ReplaceNan(f64::NEG_INFINITY), &[scalar(f64::NAN)]),
        op_case(Op::Assert(Some("must hold".into())), &[boolean(true)]),
        op_case(Op::Assert(None), &[boolean(true)]),
        op_case(Op::Factor, &[matrix()]),
        op_case(
            Op::LogDensity(DistributionKind::Gaussian),
            &[matrix(), scalar(0.5), scalar(2.0)],
        ),
        op_case(
            Op::LogDensity(DistributionKind::Truncated(Arc::new(
                DistributionKind::Poisson,
            ))),
            &[int(3), scalar(2.0), scalar(1.0), scalar(f64::INFINITY)],
        ),
    ]);
    for (from, to) in [
        (scalar(0.5), DType::I64),
        (scalar(1.0), DType::Bool),
        (int(2), DType::F64),
        (boolean(true), DType::F64),
        (boolean(true), DType::I64),
        (boolean(true), DType::Bool),
    ] {
        cases.push(op_case(Op::Cast(to), &[from]));
    }
    // proxies: connected, and unconnected
    let b = GraphBuilder::new();
    let x = constant(&b, matrix());
    let proxy = b.add_proxy(DType::F64, &[2, 3]).unwrap();
    b.connect_ids(proxy, x).unwrap();
    b.add_proxy(DType::I64, &[4]).unwrap();
    let (graph, state) = b.build().unwrap();
    cases.push(("Proxy".into(), graph, state));
    cases.push(scan_case());
    cases.push(ode_case());
    cases
}

/// ODEs: Lotka–Volterra with the stiff solver and its own tolerances and a labelled body node, and a decay with
/// the default options, a matrix parameter and no output times.
fn ode_case() -> (String, Graph, State) {
    use keanu_graph::ode::{OdeOptions, OdeSolver};
    let b = GraphBuilder::new();
    let theta = b.constant(DoubleTensor::vector(vec![1.0, 0.5, 0.75, 0.25]));
    let times = b.constant(DoubleTensor::vector(vec![0.5, 1.0, 2.0]));
    let y0 = b.constant(DoubleTensor::vector(vec![2.0, 1.0]));
    let options = OdeOptions::new(OdeSolver::Stiff)
        .with_tolerances(1e-8, 1e-9)
        .with_max_steps(5000);
    let lv = b.ode(
        y0,
        times,
        b.scalar(0.0),
        &[theta],
        options,
        |body, _t, y, p| {
            let (prey, predator) = (y.take(&[0]), y.take(&[1]));
            let p = |i: usize| p[0].take(&[i]);
            let growth = (p(0) * prey - p(1) * prey * predator).label("growth");
            body.stack(0, &[growth, p(2) * prey * predator - p(3) * predator])
        },
    );
    b.set_label_of(lv.id(), "lv").unwrap();
    let rates = b.constant(DoubleTensor::new(&[2, 2], vec![-1.0, 0.0, 0.5, -2.0]).unwrap());
    b.ode(
        y0,
        b.constant(DoubleTensor::vector(vec![])),
        b.scalar(-1.0),
        &[rates],
        OdeOptions::default(),
        |_, t, y, p| p[0].matmul(y.reshape(&[2, 1])).reshape(&[2]) * t.cos(),
    );
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(0))
        .unwrap();
    ("Ode".into(), graph, state)
}

/// Scans: one with per-step inputs, constants and a labelled body holding a nested scan, and one of zero steps.
fn scan_case() -> (String, Graph, State) {
    let b = GraphBuilder::new();
    let xs = b.constant(DoubleTensor::new(&[3, 2], vec![0.5, -0.1, 0.2, 0.9, -0.6, 0.4]).unwrap());
    let w = b.constant(DoubleTensor::new(&[2, 2], vec![0.8, -0.3, 0.2, 0.6]).unwrap());
    let init = b.constant(DoubleTensor::vector(vec![0.3, -0.4]));
    let scan = b.scan(init, &[xs], &[w], |body, h, x, c| {
        let wh = c[0].matmul(h.reshape(&[2, 1])).reshape(&[2]).label("wh");
        let inner = body.scan_steps(2, h, &[], |_, g, _| g * 0.5);
        (wh + x[0]).tanh() + inner.sum_axes(&[0])
    });
    b.set_label_of(scan.id(), "x").unwrap();
    b.scan_steps(0, b.scalar(1.0), &[], |_, g, _| g.exp());
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(0))
        .unwrap();
    ("Scan".into(), graph, state)
}

/// The parameter values a distribution is built with in [`distribution_cases`]: scalars (or, for the multivariate
/// Gaussian, a vector mean and a matrix covariance), i64 where the distribution needs it.
fn distribution_parameters(kind: &DistributionKind) -> Vec<Value> {
    use DistributionKind::*;
    match kind {
        MultivariateGaussian => vec![
            f64s(&[2], &[0.0, 1.0]),
            f64s(&[2, 2], &[1.0, 0.2, 0.2, 2.0]),
        ],
        Dirichlet => vec![f64s(&[3], &[1.0, 2.0, 3.0])],
        Multinomial => vec![int(5), f64s(&[3], &[0.2, 0.3, 0.5])],
        Categorical => vec![scalar(0.2), scalar(0.3), scalar(0.5)],
        CategoricalLogit => vec![scalar(0.2), scalar(-0.3), scalar(0.5)],
        Kde(_) => vec![],
        // each component parameter with a component axis of length 2, then the weights
        Mixture(component) => {
            let mut parameters = with_two_states(component);
            parameters.push(f64s(&[2], &[0.3, 0.7]));
            parameters
        }
        // each emission parameter with a state axis of length 2, then the initial and transition probabilities
        HiddenMarkovModel(emission) => {
            let mut parameters = with_two_states(emission);
            parameters.push(f64s(&[2], &[0.4, 0.6]));
            parameters.push(f64s(&[2, 2], &[0.9, 0.1, 0.2, 0.8]));
            parameters
        }
        LinearGaussianStateSpace => vec![
            f64s(&[2], &[0.0, 1.0]),
            f64s(&[2, 2], &[1.0, 0.2, 0.2, 2.0]),
            f64s(&[2, 2], &[0.9, 0.1, 0.0, 0.8]),
            f64s(&[2, 2], &[0.1, 0.0, 0.0, 0.2]),
            f64s(&[1, 2], &[1.0, 0.5]),
            f64s(&[1, 1], &[0.3]),
        ],
        kind => kind
            .parameter_names()
            .unwrap()
            .iter()
            .map(|&name| match name {
                "k" if matches!(kind, ChiSquared) => int(3),
                "v" | "n" => int(3),
                "x_min" if matches!(kind, UniformInt) => int(1),
                "x_max" if matches!(kind, UniformInt) => int(5),
                "x_max" => scalar(2.0),
                "c" => scalar(1.0),
                "p" | "prob_true" => scalar(0.4),
                "cutpoints" => f64s(&[2], &[-0.5, 0.5]),
                "rho" => f64s(&[2], &[0.5, -0.2]),
                "alpha" if matches!(kind, Garch11) => scalar(0.2),
                "beta" if matches!(kind, Garch11) => scalar(0.3),
                "lower" => scalar(-1.0),
                "upper" => scalar(2.0),
                _ => scalar(0.5),
            })
            .collect(),
    }
}

/// A nested kind's base parameters, with each f64 scalar given a trailing axis of two values.
fn with_two_states(base: &DistributionKind) -> Vec<Value> {
    distribution_parameters(base)
        .into_iter()
        .map(|p| match p {
            Value::F64(t) if t.rank() == 0 => {
                let v = t.as_slice()[0];
                f64s(&[2], &[v, v + 0.25])
            }
            other => other,
        })
        .collect()
}

const DISTRIBUTIONS: &[&str] = &[
    "Gaussian",
    "HalfGaussian",
    "Cauchy",
    "HalfCauchy",
    "Beta",
    "ChiSquared",
    "Dirichlet",
    "Exponential",
    "Gamma",
    "InverseGamma",
    "Laplace",
    "Logistic",
    "LogNormal",
    "MultivariateGaussian",
    "Pareto",
    "StudentT",
    "Triangular",
    "Uniform",
    "Kde",
    "Bernoulli",
    "Binomial",
    "Categorical",
    "Geometric",
    "Multinomial",
    "Poisson",
    "UniformInt",
    "LkjCholesky",
    "Weibull",
    "VonMises",
    "NegativeBinomial",
    "OrderedLogistic",
    "Truncated(Gaussian)",
    "Censored(Poisson)",
    "Mixture(Gaussian)",
    "Mixture(Truncated(Gaussian))",
    "HiddenMarkovModel(Gaussian)",
    "HiddenMarkovModel(Poisson)",
    "LinearGaussianStateSpace",
    "ZeroSumNormal",
    "BetaBinomial",
    "BernoulliLogit",
    "BinomialLogit",
    "PoissonLog",
    "CategoricalLogit",
    "ZeroInflated(Poisson)",
    "ZeroInflated(BetaBinomial)",
    "Hurdle(NegativeBinomial)",
    "Hurdle(Gamma)",
    "GaussianRandomWalk",
    "Autoregressive",
    "Garch11",
];

fn distribution_kinds() -> Vec<DistributionKind> {
    use DistributionKind::*;
    vec![
        Gaussian,
        HalfGaussian,
        Cauchy,
        HalfCauchy,
        Beta,
        ChiSquared,
        Dirichlet,
        Exponential,
        Gamma,
        InverseGamma,
        Laplace,
        Logistic,
        LogNormal,
        MultivariateGaussian,
        Pareto,
        StudentT,
        Triangular,
        Uniform,
        Kde(Arc::new(
            keanu_distributions::Kde::new(vec![0.1, 0.4, f64::MAX], 0.3).unwrap(),
        )),
        Bernoulli,
        Binomial,
        Categorical,
        Geometric,
        Multinomial,
        Poisson,
        UniformInt,
        LkjCholesky,
        Weibull,
        VonMises,
        NegativeBinomial,
        OrderedLogistic,
        Truncated(Arc::new(Gaussian)),
        Censored(Arc::new(Poisson)),
        Mixture(Arc::new(Gaussian)),
        Mixture(Arc::new(Truncated(Arc::new(Gaussian)))),
        HiddenMarkovModel(Arc::new(Gaussian)),
        HiddenMarkovModel(Arc::new(Poisson)),
        LinearGaussianStateSpace,
        ZeroSumNormal,
        BetaBinomial,
        BernoulliLogit,
        BinomialLogit,
        PoissonLog,
        CategoricalLogit,
        ZeroInflated(Arc::new(Poisson)),
        ZeroInflated(Arc::new(BetaBinomial)),
        Hurdle(Arc::new(NegativeBinomial)),
        Hurdle(Arc::new(Gamma)),
        GaussianRandomWalk,
        Autoregressive,
        Garch11,
    ]
}

/// One case per distribution kind: a labelled node with a drawn value, and the same node observed with a larger
/// shape.
fn distribution_cases() -> Vec<(String, Graph, State)> {
    distribution_kinds()
        .into_iter()
        .map(|kind| {
            // a KDE by its name alone (its debug form lists its samples)
            let name = match kind {
                DistributionKind::Kde(_) => "Kde".to_owned(),
                _ => format!("{kind:?}"),
            };
            let b = GraphBuilder::new();
            let parents: Vec<NodeId> = distribution_parameters(&kind)
                .into_iter()
                .map(|v| constant(&b, v))
                .collect();
            // an LKJ node needs a [K, K] shape, a time series its number of steps
            let lkj = matches!(kind, DistributionKind::LkjCholesky);
            // and a zero-sum normal its vector length
            let zero_sum = matches!(kind, DistributionKind::ZeroSumNormal);
            let series = matches!(
                kind,
                DistributionKind::HiddenMarkovModel(_)
                    | DistributionKind::LinearGaussianStateSpace
                    | DistributionKind::GaussianRandomWalk
                    | DistributionKind::Autoregressive
                    | DistributionKind::Garch11
            );
            let requested = if lkj {
                Some(&[3, 3][..])
            } else if zero_sum {
                Some(&[4][..])
            } else if series {
                Some(&[3][..])
            } else {
                None
            };
            let latent = b
                .add_distribution(kind.clone(), requested, &parents)
                .unwrap_or_else(|e| panic!("can't build {name}: {e}"));
            b.set_label_of(latent, "latent").unwrap();
            let batch = b.shape(latent).unwrap();
            let shape: Vec<usize> = if lkj {
                vec![2, 3, 3]
            } else if zero_sum {
                vec![2, 4]
            } else if series {
                vec![5]
            } else if parents.is_empty() || batch.is_empty() {
                vec![2]
            } else {
                batch.clone()
            };
            let observed = b.add_distribution(kind, Some(&shape), &parents).unwrap();
            let (graph, mut state) = b.build().unwrap();
            let mut rng = KeanuRng::seed_from_u64(1);
            state
                .lazy_eval(&graph, &[latent, observed], &mut rng)
                .unwrap();
            let value = state.value(observed).unwrap().clone();
            state.observe(&graph, observed, value).unwrap();
            (name, graph, state)
        })
        .collect()
}

#[test]
fn every_operation_round_trips() {
    let cases = op_cases();
    // every op except lambdas, and every variant of the op kinds
    let names: BTreeSet<&str> = cases
        .iter()
        .flat_map(|(_, graph, _)| graph.ids().map(|id| graph[id].op().name()))
        .collect();
    let every_op = [
        "constant",
        "unary",
        "binary",
        "compare",
        "mask",
        "approx_eq",
        "logic",
        "not",
        "all",
        "any",
        "predicate",
        "reduce",
        "standardize",
        "softmax",
        "log_softmax",
        "cumulative",
        "arg_reduce",
        "reshape",
        "permute",
        "transpose",
        "broadcast",
        "slice",
        "concat",
        "take",
        "gather",
        "scatter_add",
        "stack",
        "diag",
        "diag_part",
        "tril",
        "triu",
        "fill_triangular",
        "triangle_part",
        "matmul",
        "tensor_multiply",
        "inverse",
        "determinant",
        "cholesky",
        "cholesky_inverse",
        "solve_triangular",
        "solve",
        "log_abs_determinant",
        "determinant_sign",
        "eigvalsh",
        "eigh_vectors",
        "matrix_exp",
        "cast",
        "where",
        "multiplexer",
        "cpt",
        "set_with_mask",
        "replace_nan",
        "proxy",
        "assert",
        "scan",
        "ode",
        "factor",
        "log_density",
    ];
    assert_eq!(names, every_op.into_iter().collect::<BTreeSet<_>>());
    for (name, graph, state) in &cases {
        assert_round_trips(name, graph, state);
    }
}

/// A model with a factor of log densities keeps its log probability through saving and loading.
#[test]
fn factors_and_log_densities_keep_their_log_probability() {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0);
    b.set_value(mu, DoubleTensor::scalar(0.3));
    let data = b.constant(DoubleTensor::vector(vec![0.5, -0.2, 1.4]));
    let log_p = b.log_density(
        DistributionKind::Gaussian,
        data,
        &[mu.id(), b.scalar(0.8).id()],
    );
    b.factor(log_p).label("likelihood");
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let expected = Model::new(graph.clone(), &state)
        .unwrap()
        .log_prob(&state)
        .unwrap();
    assert!(expected.is_finite());
    let bytes = save_protobuf(&graph, &state, &Metadata::new()).unwrap();
    let mut loaded = load_protobuf(&bytes).unwrap();
    let all: Vec<NodeId> = loaded.graph.ids().collect();
    loaded
        .state
        .lazy_eval(&loaded.graph, &all, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let model = Model::new(loaded.graph, &loaded.state).unwrap();
    assert_eq!(model.factors().len(), 1);
    assert_eq!(model.log_prob(&loaded.state).unwrap(), expected);
}

/// A structural time series with a regression, whose likelihood is a factor of a state-space log density, keeps
/// its log probability through saving and loading.
#[test]
fn a_state_space_log_density_keeps_its_log_probability() {
    use keanu_graph::sts::{Component, StructuralModel};
    let b = GraphBuilder::new();
    let y = [1.0, 1.8, 0.6, 2.2, 2.9, 1.4];
    let design = DoubleTensor::new(&[6, 1], vec![0.3, -0.1, 0.8, 0.2, -0.6, 0.4]).unwrap();
    StructuralModel::new(&b, &y)
        .with(Component::local_linear_trend())
        .with(Component::seasonal(2))
        .with(Component::regression(design))
        .build()
        .unwrap();
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let expected = Model::new(graph.clone(), &state)
        .unwrap()
        .log_prob(&state)
        .unwrap();
    assert!(expected.is_finite());
    let bytes = save_protobuf(&graph, &state, &Metadata::new()).unwrap();
    let mut loaded = load_protobuf(&bytes).unwrap();
    let all: Vec<NodeId> = loaded.graph.ids().collect();
    loaded
        .state
        .lazy_eval(&loaded.graph, &all, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let model = Model::new(loaded.graph, &loaded.state).unwrap();
    assert_eq!(model.factors().len(), 1);
    assert_eq!(model.log_prob(&loaded.state).unwrap(), expected);
}

/// A model rewritten by marginalising a discrete latent (whose factor holds Bernoulli log densities) keeps its log
/// probability and the latent's log weights through saving and loading.
#[test]
fn marginalised_models_keep_their_log_probability() {
    let b = GraphBuilder::new();
    let psi = b.beta(2.0, 2.0);
    let p = b.beta(2.0, 2.0);
    let occupied = b.bernoulli_with_shape(&[3], psi);
    let y = b.bernoulli_with_shape(&[3, 2], occupied.to_double_mask().reshape(&[3, 1]) * p);
    b.observe(
        y,
        BoolTensor::new(&[3, 2], vec![true, false, false, false, false, true]).unwrap(),
    );
    b.set_value(psi, DoubleTensor::scalar(0.4));
    b.set_value(p, DoubleTensor::scalar(0.7));
    let occupied = occupied.id();
    let (graph, state) = b.build().unwrap();
    let (graph, state, marginalized) =
        keanu_graph::marginalize(&graph, &state, &[occupied]).unwrap();
    let weights = marginalized.latents()[0].log_weights();
    let expected = Model::new(graph.clone(), &state)
        .unwrap()
        .log_prob(&state)
        .unwrap();
    assert!(expected.is_finite());
    let protobuf = save_protobuf(&graph, &state, &Metadata::new()).unwrap();
    let json = save_json(&graph, &state, &Metadata::new()).unwrap();
    for loaded in [load_protobuf(&protobuf).unwrap(), load_json(&json).unwrap()] {
        let loaded_state = recompute(&loaded.graph, &loaded.state);
        assert_eq!(loaded_state.value(weights), state.value(weights));
        let model = Model::new(loaded.graph, &loaded_state).unwrap();
        assert_eq!(model.log_prob(&loaded_state).unwrap(), expected);
    }
}

#[test]
fn every_distribution_kind_round_trips() {
    let cases = distribution_cases();
    let names: Vec<&str> = cases.iter().map(|(name, ..)| name.as_str()).collect();
    assert_eq!(names, DISTRIBUTIONS);
    for (name, graph, state) in &cases {
        assert_round_trips(name, graph, state);
    }
}

#[test]
fn nodes_without_a_saved_form_are_reported() {
    let b = GraphBuilder::new();
    let x = b.scalar(1.0);
    b.lambda("double it", &[x.id()], DType::F64, &[], |v: &[&Value]| {
        Ok(v[0].clone())
    });
    let (graph, state) = b.build().unwrap();
    let error = save_protobuf(&graph, &state, &Metadata::new()).unwrap_err();
    assert!(
        matches!(&error, IoError::NotSaveable(m) if m.contains("lambda")),
        "{error}"
    );

    #[derive(Debug)]
    struct Flat;
    impl CustomDistribution for Flat {
        fn log_prob(&self, _: &Value, _: &[&Value]) -> Result<f64, String> {
            Ok(0.0)
        }
    }
    let b = GraphBuilder::new();
    b.add_distribution(DistributionKind::Custom(Arc::new(Flat)), Some(&[]), &[])
        .unwrap();
    let (graph, state) = b.build().unwrap();
    for error in [
        save_protobuf(&graph, &state, &Metadata::new()).unwrap_err(),
        save_json(&graph, &state, &Metadata::new()).unwrap_err(),
    ] {
        assert!(
            matches!(&error, IoError::NotSaveable(m) if m.contains("custom distribution")),
            "{error}"
        );
    }
}

// ----- the golden models -----

fn golden_cases() -> Vec<Json> {
    let text = std::fs::read_to_string(golden_dir().join("models.json")).unwrap();
    let root: Json = serde_json::from_str(&text).unwrap();
    let cases = root["cases"].as_array().unwrap().clone();
    assert!(
        cases.len() >= 7,
        "expected the golden models including the catalog"
    );
    cases
}

/// The golden model files: (name, protobuf bytes, JSON text).
fn golden_files() -> Vec<(String, Vec<u8>, String)> {
    golden_cases()
        .iter()
        .map(|case| {
            let file = |key: &str| golden_dir().join(case[key].as_str().unwrap());
            (
                case["name"].as_str().unwrap().to_owned(),
                std::fs::read(file("protobuf")).unwrap(),
                std::fs::read_to_string(file("json")).unwrap(),
            )
        })
        .collect()
}

/// Checks a loaded golden model against the values Java computed: log prob, log likelihood, their gradients, and
/// the value of every labelled node (deterministic nodes are recomputed rather than taken from the file).
fn check_golden(context: &str, loaded: &Loaded, expected: &Json) -> Vec<String> {
    const TOLERANCE: f64 = 1e-10;
    let mut problems = Vec::new();
    let graph = &loaded.graph;
    let state = recompute(graph, &loaded.state);
    let model = Model::new(graph.clone(), &state).unwrap();
    fn check_tensor(
        problems: &mut Vec<String>,
        what: String,
        actual: (Vec<usize>, Vec<f64>),
        expected: &Json,
    ) {
        let expected = parse_tensor(expected).unwrap();
        if actual.0 != expected.shape {
            problems.push(format!(
                "{what}: shape {:?}, expected {:?}",
                actual.0, expected.shape
            ));
        } else if let Err(e) = close(&actual.1, &expected.data, TOLERANCE) {
            problems.push(format!("{what}: {e}"));
        }
    }
    for (key, actual) in [
        ("logProb", model.log_prob(&state).unwrap()),
        ("logLikelihood", model.log_likelihood(&state).unwrap()),
    ] {
        if let Err(e) = close(&[actual], &[number(&expected[key])], TOLERANCE) {
            problems.push(format!("{context} {key}: {e}"));
        }
    }
    for (key, gradient) in [
        ("dLogProb", model.log_prob_gradient(&state).unwrap()),
        (
            "dLogLikelihood",
            model.log_likelihood_gradient(&state).unwrap(),
        ),
    ] {
        let expected = expected[key].as_object().unwrap();
        let actual: BTreeMap<String, _> = gradient
            .iter()
            .map(|(id, g)| {
                (
                    graph[id].label().unwrap_or("?").to_owned(),
                    (g.shape().to_vec(), g.as_slice().to_vec()),
                )
            })
            .collect();
        let actual_keys: Vec<&String> = actual.keys().collect();
        let expected_keys: Vec<&String> = expected.keys().collect();
        if actual_keys != expected_keys {
            problems.push(format!(
                "{context} {key}: gradient keys {actual_keys:?}, expected {expected_keys:?}"
            ));
            continue;
        }
        for (label, value) in actual {
            check_tensor(
                &mut problems,
                format!("{context} {key}[{label}]"),
                value,
                &expected[&label],
            );
        }
    }
    for key in ["latentValues", "nodeValues"] {
        for (label, value) in expected[key].as_object().unwrap() {
            match graph.find(label) {
                None => problems.push(format!("{context} {key}: no node labelled {label}")),
                Some(id) => check_tensor(
                    &mut problems,
                    format!("{context} {key}[{label}]"),
                    flatten(state.value(id).unwrap()),
                    value,
                ),
            }
        }
    }
    problems
}

#[test]
fn golden_models_load_and_match_java() {
    let mut problems = Vec::new();
    for (case, (name, pb, json)) in golden_cases().iter().zip(golden_files()) {
        for (format, loaded) in [("protobuf", load_protobuf(&pb)), ("json", load_json(&json))] {
            let loaded = loaded.unwrap_or_else(|e| panic!("{name} ({format}): {e}"));
            assert_eq!(
                loaded.metadata.get("name").map(String::as_str),
                Some(name.as_str())
            );
            problems.extend(check_golden(
                &format!("{name} ({format})"),
                &loaded,
                &case["expected"],
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "{} problems:\n{}",
        problems.len(),
        problems.join("\n")
    );
}

/// The format is stable: saving a loaded golden model writes its file again, byte for byte.
#[test]
fn saving_a_golden_model_reproduces_its_file() {
    for (name, pb, json) in golden_files() {
        let loaded = load_protobuf(&pb).unwrap();
        assert!(
            save_protobuf(&loaded.graph, &loaded.state, &loaded.metadata).unwrap() == pb,
            "{name}: the protobuf file changed"
        );
        let loaded = load_json(&json).unwrap();
        assert_eq!(
            save_json(&loaded.graph, &loaded.state, &loaded.metadata).unwrap(),
            json,
            "{name}: the JSON file changed"
        );
        assert_round_trips(&name, &loaded.graph, &loaded.state);
    }
}

// ----- values, labels and metadata -----

/// A model with labels, observations, values set on latent and deterministic nodes, and metadata.
#[test]
fn observations_labels_values_and_metadata_survive() {
    let b = GraphBuilder::new();
    let a = b
        .constant(DoubleTensor::new(&[2, 3], vec![0.2, 0.5, 0.7, 0.3, 0.9, 0.4]).unwrap())
        .label("a");
    let mu = b.gaussian(0.0, 1.0).label("mu");
    b.set_value(mu, DoubleTensor::scalar(0.3));
    let sigma = b.half_gaussian(1.5).label("sigma");
    b.set_value(sigma, DoubleTensor::scalar(0.8));
    let kde = b
        .kde(Kde::new(vec![0.1, 0.4, 0.5, 1.2], 0.3).unwrap(), &[])
        .label("kde");
    b.set_value(kde, DoubleTensor::scalar(0.45));
    let mean = a * mu + 1.0;
    b.set_value(mean, DoubleTensor::full(&[2, 3], 2.0));
    let y = b.gaussian_with_shape(&[2, 3], mean, sigma).label("y");
    b.observe(
        y,
        DoubleTensor::new(&[2, 3], vec![1.1, 1.4, 1.0, 1.3, 1.5, 0.9]).unwrap(),
    );
    let flag = a.gt(0.5).label("flag");
    b.observe(flag, BoolTensor::new(&[2, 3], vec![false; 6]).unwrap());
    let counts = b.poisson_with_shape(&[3], b.scalar(2.0)).label("counts");
    b.observe(counts, IntTensor::vector(vec![1, 2, 4]));
    let (flag, mean) = (flag.id(), mean.id());
    let (graph, state) = b.build().unwrap();
    assert!(state.is_observed(flag) && state.value(mean).is_some());
    assert_round_trips("values", &graph, &state);

    let metadata: Metadata = [
        ("author".to_owned(), "keanu".to_owned()),
        ("version".to_owned(), "2".to_owned()),
    ]
    .into();
    let loaded = load_json(&save_json(&graph, &state, &metadata).unwrap()).unwrap();
    assert_eq!(loaded.metadata, metadata);
    let loaded = load_protobuf(&save_protobuf(&graph, &state, &metadata).unwrap()).unwrap();
    assert_eq!(loaded.metadata, metadata);
}

#[test]
fn slices_with_spaces_and_a_trailing_comma_save_and_load() {
    let b = GraphBuilder::new();
    let a = b.constant(DoubleTensor::new(&[2, 4], (0..8).map(f64::from).collect()).unwrap());
    a.slice(": , 1 : 3").label("spaced");
    a.slice("1,").label("trailing");
    let (graph, state) = b.build().unwrap();
    let json = save_json(&graph, &state, &Metadata::new()).unwrap();
    let loaded = load_json(&json).unwrap();
    let state = recompute(&loaded.graph, &loaded.state);
    let value = |label: &str| {
        let id = loaded.graph.find(label).unwrap();
        flatten(state.value(id).unwrap())
    };
    assert_eq!(value("spaced"), (vec![2, 2], vec![1.0, 2.0, 5.0, 6.0]));
    assert_eq!(value("trailing"), (vec![4], vec![4.0, 5.0, 6.0, 7.0]));
}

// ----- invalid and newer files -----

/// A saved model of a vector constant (node 0) as JSON.
fn saved_vector() -> Json {
    let b = GraphBuilder::new();
    b.constant(DoubleTensor::vector(vec![1.0, 2.0, 3.0]));
    let (graph, state) = b.build().unwrap();
    serde_json::from_str(&save_json(&graph, &state, &Metadata::new()).unwrap()).unwrap()
}

/// The saved vector model with another node.
fn with_node(node: Json) -> String {
    let mut model = saved_vector();
    model["graph"]["nodes"].as_array_mut().unwrap().push(node);
    model.to_string()
}

#[test]
fn files_with_a_newer_format_version_are_rejected() {
    let mut model = saved_vector();
    model["formatVersion"] = 2.into();
    assert_eq!(
        load_json(&model.to_string()).unwrap_err(),
        IoError::UnsupportedVersion {
            found: 2,
            supported: 1
        }
    );
    // the protobuf encoding starts with the version: field 1, varint
    let b = GraphBuilder::new();
    b.scalar(1.0);
    let (graph, state) = b.build().unwrap();
    let mut pb = save_protobuf(&graph, &state, &Metadata::new()).unwrap();
    assert_eq!(pb[..2], [0x08, 0x01]);
    pb[1] = 0x07;
    let error = load_protobuf(&pb).unwrap_err();
    assert_eq!(
        error,
        IoError::UnsupportedVersion {
            found: 7,
            supported: 1
        }
    );
    assert!(error.to_string().contains("format version 7"), "{error}");

    // a file without a version isn't a Keanu model
    let mut model = saved_vector();
    model.as_object_mut().unwrap().remove("formatVersion");
    assert!(matches!(
        load_json(&model.to_string()),
        Err(IoError::Decode(_))
    ));
}

fn is_invalid_node(result: keanu_io::Result<Loaded>, expected_index: usize) -> bool {
    matches!(result, Err(IoError::InvalidNode { index, .. }) if index == expected_index)
}

#[test]
fn invalid_nodes_are_reported_with_their_index() {
    // a parent that isn't an earlier node
    for parent in [1, 2, u32::MAX] {
        let json = with_node(serde_json::json!({"parents": [parent], "unary": "UNARY_OP_EXP"}));
        assert!(is_invalid_node(load_json(&json), 1), "parent {parent}");
    }
    // no operation, or an unspecified kind
    assert!(is_invalid_node(
        load_json(&with_node(serde_json::json!({"parents": [0]}))),
        1
    ));
    let unspecified =
        with_node(serde_json::json!({"parents": [0], "unary": "UNARY_OP_UNSPECIFIED"}));
    assert!(is_invalid_node(load_json(&unspecified), 1));
    let unspecified =
        with_node(serde_json::json!({"distribution": {"kind": "DISTRIBUTION_KIND_UNSPECIFIED"}}));
    assert!(is_invalid_node(load_json(&unspecified), 1));
    // a KDE without samples, or samples on another distribution
    let kde = with_node(serde_json::json!({"distribution": {"kind": "DISTRIBUTION_KIND_KDE"}}));
    assert!(is_invalid_node(load_json(&kde), 1));
    let samples = with_node(serde_json::json!({"parents": [0, 0],
        "distribution": {"kind": "DISTRIBUTION_KIND_GAUSSIAN", "kde": {"samples": [1.0], "bandwidth": 1.0}}}));
    assert!(is_invalid_node(load_json(&samples), 1));
    // a proxy with two parents
    let proxy = with_node(
        serde_json::json!({"parents": [0, 0], "proxy": {"dtype": "D_TYPE_F64", "shape": [3]}}),
    );
    assert!(is_invalid_node(load_json(&proxy), 1));
    // ops the graph rejects are graph errors
    let tril = with_node(serde_json::json!({"parents": [0], "tril": {"k": 1}}));
    assert!(matches!(load_json(&tril), Err(IoError::Graph(_))));
}

/// A saved model whose last node is a scan of two steps over a scalar carry with one constant: body nodes
/// [carry, constant, carry * constant].
fn saved_scan() -> Json {
    let b = GraphBuilder::new();
    let k = b.scalar(0.5);
    b.scan_steps(2, b.scalar(1.0), &[k], |_, x, c| x * c[0]);
    let (graph, state) = b.build().unwrap();
    serde_json::from_str(&save_json(&graph, &state, &Metadata::new()).unwrap()).unwrap()
}

#[test]
fn invalid_scan_bodies_are_rejected() {
    let scan_index = 2;
    let edit = |f: &dyn Fn(&mut Json)| {
        let mut model = saved_scan();
        f(&mut model["graph"]["nodes"][scan_index]["scan"]);
        load_json(&model.to_string())
    };
    assert!(edit(&|_| {}).is_ok());
    type Edit<'a> = &'a dyn Fn(&mut Json);
    let rejected: [(&str, Edit<'_>); 5] = [
        ("no body", &|scan| {
            scan.as_object_mut().unwrap().remove("body");
        }),
        ("stored values", &|scan| {
            scan["body"]["values"] =
                serde_json::json!([{"node": 2, "value": {"shape": [], "f64": {"values": [1.0]}}}]);
        }),
        ("an unknown output", &|scan| scan["output"] = 9.into()),
        ("a carry that isn't a proxy", &|scan| {
            scan["carry"] = 2.into()
        }),
        ("a distribution", &|scan| {
            scan["body"]["nodes"][2] = serde_json::json!({"parents": [0, 1],
                "distribution": {"kind": "DISTRIBUTION_KIND_GAUSSIAN"}});
        }),
    ];
    for (name, f) in rejected {
        let result = edit(f);
        assert!(is_invalid_node(result, scan_index), "{name}");
    }
    // the parents must match the body's inputs
    let mut model = saved_scan();
    model["graph"]["nodes"][scan_index]["parents"] = serde_json::json!([1]);
    assert!(matches!(
        load_json(&model.to_string()),
        Err(IoError::Graph(_))
    ));
}

#[test]
fn invalid_shapes_and_values_are_decode_errors() {
    let is_decode_error = |result: keanu_io::Result<Loaded>| {
        matches!(
            result,
            Err(IoError::Decode(_)) | Err(IoError::InvalidNode { .. })
        )
    };
    let huge = serde_json::json!(["4294967296", "4294967296"]);
    let huge_constant = with_node(serde_json::json!({"constant": {"shape": huge, "f64": {}}}));
    assert!(is_decode_error(load_json(&huge_constant)));
    let huge_distribution = with_node(serde_json::json!({"parents": [0, 0],
        "distribution": {"kind": "DISTRIBUTION_KIND_GAUSSIAN", "shape": huge}}));
    assert!(is_decode_error(load_json(&huge_distribution)));
    let huge_broadcast = with_node(serde_json::json!({"parents": [0],
        "broadcast": {"shape": ["4294967296", "4294967296", "3"]}}));
    assert!(is_decode_error(load_json(&huge_broadcast)));
    // values that don't fit their shape
    let mismatched =
        with_node(serde_json::json!({"constant": {"shape": [2], "f64": {"values": [1.0]}}}));
    assert!(is_decode_error(load_json(&mismatched)));
    let no_values = with_node(serde_json::json!({"constant": {"shape": [2]}}));
    assert!(is_decode_error(load_json(&no_values)));

    // a value stored for a node that doesn't exist
    let mut model = saved_vector();
    model["graph"]["values"] =
        serde_json::json!([{"node": 5, "value": {"shape": [], "f64": {"values": [1.0]}}}]);
    assert!(matches!(
        load_json(&model.to_string()),
        Err(IoError::Decode(_))
    ));
    // a stored value of the wrong type is a graph error
    model["graph"]["values"] = serde_json::json!([{"node": 0, "value": {"shape": [3], "bool": {"values": [true, true, true]}}}]);
    assert!(matches!(
        load_json(&model.to_string()),
        Err(IoError::Graph(_))
    ));
}

#[test]
fn malformed_cpt_condition_counts_are_errors_not_panics() {
    let is_error = |result: keanu_io::Result<Loaded>| {
        matches!(
            result,
            Err(IoError::Decode(_)) | Err(IoError::InvalidNode { .. }) | Err(IoError::Graph(_))
        )
    };
    let max = serde_json::json!("18446744073709551615");
    // more conditions than parents: a huge count (which used to overflow the arity check) and a small one
    for conditions in [max.clone(), 5.into()] {
        for parents in [serde_json::json!([]), serde_json::json!([0])] {
            let cpt = with_node(serde_json::json!({"parents": parents,
                "cpt": {"conditions": conditions, "keys": []}}));
            let result = load_json(&cpt).map(|_| ());
            assert!(
                matches!(result, Err(IoError::InvalidNode { index: 1, .. })),
                "conditions {conditions}, parents {parents}: {result:?}"
            );
        }
    }
    // a key without one value per condition
    let int = serde_json::json!({"constant": {"shape": [], "i64": {"values": ["1"]}}});
    let mut model = saved_vector();
    model["graph"]["nodes"].as_array_mut().unwrap().push(int);
    model["graph"]["nodes"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"parents": [1, 0, 0],
        "cpt": {"conditions": 1, "keys": [{"values": ["1", "2"]}]}}));
    assert!(is_error(load_json(&model.to_string())));
}

#[test]
fn shapes_too_large_to_allocate_are_rejected_when_loading() {
    let is_error = |result: keanu_io::Result<Loaded>| {
        matches!(
            result,
            Err(IoError::Decode(_)) | Err(IoError::InvalidNode { .. }) | Err(IoError::Graph(_))
        )
    };
    // 2^61 elements fit a usize, but not 8 bytes each in isize::MAX bytes
    let scalar = serde_json::json!({"constant": {"shape": [], "f64": {"values": [1.0]}}});
    let mut model = saved_vector();
    model["graph"]["nodes"].as_array_mut().unwrap().push(scalar);
    model["graph"]["nodes"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"parents": [1],
        "broadcast": {"shape": ["2305843009213693952"]}}));
    let result = load_json(&model.to_string());
    assert!(is_error(result), "broadcast");
    let proxy = with_node(
        serde_json::json!({"proxy": {"dtype": "D_TYPE_F64", "shape": ["2305843009213693952"]}}),
    );
    assert!(is_error(load_json(&proxy)), "proxy");
    // a scatter-add whose output would have 3 x 2^60 elements
    let mut model = saved_vector();
    model["graph"]["nodes"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"constant": {"shape": [1], "i64": {"values": ["0"]}}}));
    model["graph"]["nodes"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"parents": [0, 1],
        "scatterAdd": {"axis": "0", "size": "1152921504606846976"}}));
    assert!(is_error(load_json(&model.to_string())), "scatter-add");
    // a scan of 2^60 and 2^62 steps (8 bytes per step, as the carry is a scalar)
    for length in ["1152921504606846976", "4611686018427387904"] {
        let mut model = saved_scan();
        model["graph"]["nodes"][2]["scan"]["length"] = length.into();
        let result = load_json(&model.to_string());
        assert!(is_error(result), "scan length {length}");
    }
}

// ----- DOT -----

#[cfg(feature = "dot")]
#[test]
fn dot_export_describes_nodes_edges_and_metadata() {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0).label("model.mu");
    b.set_value(mu, DoubleTensor::scalar(0.5));
    let sum = mu + 2.0;
    let _ = b.gaussian(sum, 1.0);
    let (graph, state) = b.build().unwrap();
    let metadata: Metadata = [("author".to_owned(), "keanu".to_owned())]
        .into_iter()
        .collect();

    let dot = keanu_io::to_dot(&graph, None, &metadata);
    assert!(dot.starts_with("digraph BayesianNetwork {\n// Model metadata:\n// author=keanu\n"));
    assert!(dot.ends_with('}'));
    assert!(dot.contains("<0> -> <2> [label=mu]"));
    assert!(dot.contains("<1> -> <2> [label=sigma]"));
    assert!(dot.contains("2[label=\"mu\"]"), "{dot}");
    assert!(dot.contains("[label=\"+\"]"));
    assert!(dot.contains("0[label=\"0.0\"]"));
    assert!(dot.contains("[label=\"Gaussian\"]"), "{dot}");

    let with_values = keanu_io::to_dot(&graph, Some(&state), &Metadata::new());
    assert!(
        with_values.contains("2[label=\"0.5 (mu)\"]"),
        "{with_values}"
    );
}

#[cfg(feature = "dot")]
#[test]
fn dot_edges_of_nested_distributions_name_every_parameter() {
    let b = GraphBuilder::new();
    let (mu, sigma, lower, upper) = (b.scalar(0.0), b.scalar(1.0), b.scalar(-1.0), b.scalar(1.0));
    let ids = [mu.id(), sigma.id(), lower.id(), upper.id()];
    let truncated = DistributionKind::Truncated(Arc::new(DistributionKind::Gaussian));
    let weights = b.constant(DoubleTensor::vector(vec![0.4, 0.6]));
    let mut parents = ids.to_vec();
    parents.push(weights.id());
    let node = b
        .add_distribution(
            DistributionKind::Mixture(Arc::new(truncated)),
            None,
            &parents,
        )
        .unwrap();
    let (graph, _) = b.build().unwrap();
    let dot = keanu_io::to_dot(&graph, None, &Metadata::new());
    for (parent, name) in parents
        .iter()
        .zip(["mu", "sigma", "lower", "upper", "weights"])
    {
        let edge = format!("<{}> -> <{}> [label={name}]", parent.index(), node.index());
        assert!(dot.contains(&edge), "{edge} in {dot}");
    }
    assert!(dot.contains("[label=\"Mixture\"]"), "{dot}");
}

#[cfg(feature = "dot")]
#[test]
fn dot_labels_are_escaped() {
    let b = GraphBuilder::new();
    b.gaussian(0.0, 1.0).label("say \"hi\"\\\nbye");
    let (graph, _) = b.build().unwrap();
    let dot = keanu_io::to_dot(&graph, None, &Metadata::new());
    assert!(dot.contains(r#"2[label="say \"hi\"\\\nbye"]"#), "{dot}");
}

#[cfg(feature = "dot")]
#[test]
fn dot_draws_scan_bodies_as_clusters() {
    let b = GraphBuilder::new();
    let k = b.scalar(0.5);
    let x = b.scan_steps(2, b.scalar(1.0), &[k], |body, x, c| {
        let inner = body.scan_steps(3, x, &[], |_, y, _| y.exp());
        x * c[0] + inner.sum()
    });
    let x = x.id();
    let (graph, _) = b.build().unwrap();
    let dot = keanu_io::to_dot(&graph, None, &Metadata::new());
    let s = x.index();
    assert!(
        dot.contains(&format!(
            "subgraph \"cluster_{s}.\" {{\nlabel=\"scan body (2 steps)\""
        )),
        "{dot}"
    );
    // body edges and labels use the scan's prefix, and the output feeds the scan node
    assert!(dot.contains(&format!("\"{s}.0\" -> \"{s}.2\"")), "{dot}");
    assert!(
        dot.contains(&format!("\"{s}.1\"[label=\"constants[0]\"]")),
        "{dot}"
    );
    assert!(dot.contains(&format!("-> <{s}> [style=dashed]")), "{dot}");
    assert!(dot.contains(&format!("{s}[label=\"scan\"]")), "{dot}");
    // the nested scan's body is a cluster inside it
    assert!(
        dot.contains(&format!("subgraph \"cluster_{s}.2.\"")),
        "{dot}"
    );
    assert!(
        dot.contains(&format!("\"{s}.2.0\"[label=\"carry\"]")),
        "{dot}"
    );
}
