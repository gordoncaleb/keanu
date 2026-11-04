//! Java vertex classes, the stable type identifiers of the saved model format, and the tables of the vertices that
//! map one to one onto a Rust operation or distribution, shared by saving, loading and DOT export.

// the class names are only used by saving and loading, not by DOT export alone
#![cfg_attr(not(any(feature = "protobuf", feature = "json")), allow(dead_code))]

use keanu_graph::op::{BinaryOp, CompareOp, DistributionKind, LogicOp, Op, Predicate, UnaryOp};
use std::collections::HashMap;
use std::mem::Discriminant;
use std::sync::LazyLock;

/// A distribution saved as a Java vertex whose parameters are its parents.
#[derive(Debug)]
pub(crate) struct SavedDistribution {
    /// The distribution.
    pub(crate) kind: DistributionKind,
    /// The simple Java class name.
    pub(crate) class: &'static str,
    /// The Java parameter name of each parent, in parent order.
    pub(crate) parameters: &'static [&'static str],
    /// For the half distributions, Java's location parameter, which has no Rust parent and is saved as the
    /// constant 0.
    pub(crate) zero_location: Option<&'static str>,
}

const fn distribution(
    kind: DistributionKind,
    class: &'static str,
    parameters: &'static [&'static str],
) -> SavedDistribution {
    SavedDistribution {
        kind,
        class,
        parameters,
        zero_location: None,
    }
}

const fn half(
    kind: DistributionKind,
    class: &'static str,
    location: &'static str,
    parameters: &'static [&'static str],
) -> SavedDistribution {
    SavedDistribution {
        kind,
        class,
        parameters,
        zero_location: Some(location),
    }
}

/// Every distribution saved this way (KDE is saved with its samples, and categorical nodes have no saved form).
pub(crate) static DISTRIBUTIONS: &[SavedDistribution] = {
    use DistributionKind::*;
    &[
        distribution(Gaussian, "GaussianVertex", &["mu", "sigma"]),
        half(HalfGaussian, "HalfGaussianVertex", "mu", &["sigma"]),
        distribution(Cauchy, "CauchyVertex", &["location", "scale"]),
        half(HalfCauchy, "HalfCauchyVertex", "location", &["scale"]),
        distribution(Beta, "BetaVertex", &["alpha", "beta"]),
        distribution(ChiSquared, "ChiSquaredVertex", &["k"]),
        distribution(Dirichlet, "DirichletVertex", &["concentration"]),
        distribution(Exponential, "ExponentialVertex", &["rate"]),
        distribution(Gamma, "GammaVertex", &["theta", "k"]),
        distribution(InverseGamma, "InverseGammaVertex", &["alpha", "beta"]),
        distribution(Laplace, "LaplaceVertex", &["mu", "beta"]),
        distribution(Logistic, "LogisticVertex", &["mu", "s"]),
        distribution(LogNormal, "LogNormalVertex", &["mu", "sigma"]),
        distribution(
            MultivariateGaussian,
            "MultivariateGaussianVertex",
            &["mu", "covariance"],
        ),
        distribution(Pareto, "ParetoVertex", &["location", "scale"]),
        distribution(StudentT, "StudentTVertex", &["v"]),
        distribution(Triangular, "TriangularVertex", &["xMin", "xMax", "c"]),
        distribution(Uniform, "UniformVertex", &["xMin", "xMax"]),
        distribution(Bernoulli, "BernoulliVertex", &["probTrue"]),
        distribution(Binomial, "BinomialVertex", &["p", "n"]),
        distribution(Geometric, "GeometricVertex", &["p"]),
        distribution(Multinomial, "MultinomialVertex", &["n", "p"]),
        distribution(Poisson, "PoissonVertex", &["mu"]),
        distribution(UniformInt, "UniformIntVertex", &["min", "max"]),
    ]
};

/// The saved form of a distribution, if it is in [`DISTRIBUTIONS`].
pub(crate) fn saved_distribution(kind: &DistributionKind) -> Option<&'static SavedDistribution> {
    static BY_KIND: LazyLock<HashMap<Discriminant<DistributionKind>, &SavedDistribution>> =
        LazyLock::new(|| {
            DISTRIBUTIONS
                .iter()
                .map(|d| (std::mem::discriminant(&d.kind), d))
                .collect()
        });
    BY_KIND.get(&std::mem::discriminant(kind)).copied()
}

/// The distribution saved as a (simple) Java class, if it is in [`DISTRIBUTIONS`].
pub(crate) fn distribution_of_class(class: &str) -> Option<&'static SavedDistribution> {
    static BY_CLASS: LazyLock<HashMap<&str, &SavedDistribution>> =
        LazyLock::new(|| DISTRIBUTIONS.iter().map(|d| (d.class, d)).collect());
    BY_CLASS.get(class).copied()
}

/// An operation saved as a Java vertex whose only parameters are its parents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SimpleOp {
    Unary(UnaryOp),
    Binary(BinaryOp),
    Compare(CompareOp),
    Mask(CompareOp),
    Logic(LogicOp),
    Predicate(Predicate),
}

impl SimpleOp {
    /// The simple operation an op is, if any (it may still have no saved class).
    pub(crate) fn of(op: &Op) -> Option<Self> {
        Some(match op {
            Op::Unary(op) => Self::Unary(*op),
            Op::Binary(op) => Self::Binary(*op),
            Op::Compare(op) => Self::Compare(*op),
            Op::Mask(op) => Self::Mask(*op),
            Op::Logic(op) => Self::Logic(*op),
            Op::Predicate(p) => Self::Predicate(*p),
            _ => return None,
        })
    }

    /// The node's op.
    pub(crate) fn op(self) -> Op {
        match self {
            Self::Unary(op) => Op::Unary(op),
            Self::Binary(op) => Op::Binary(op),
            Self::Compare(op) => Op::Compare(op),
            Self::Mask(op) => Op::Mask(op),
            Self::Logic(op) => Op::Logic(op),
            Self::Predicate(p) => Op::Predicate(p),
        }
    }

    /// The Java parameter name of each parent, in parent order.
    pub(crate) fn parameters(self) -> &'static [&'static str] {
        match self {
            Self::Unary(_) | Self::Predicate(_) => &["inputVertex"],
            // saved as WhereVertex parameters; Java derives the predicate again from then / else
            Self::Binary(BinaryOp::Max | BinaryOp::Min) => &["then", "else"],
            _ => &["left", "right"],
        }
    }

    /// The simple Java class name, if the operation has one.
    pub(crate) fn class(self) -> Option<&'static str> {
        SIMPLE_OPS_BY_OP.get(&self).copied()
    }

    /// The operation saved as a (simple) Java class, if any.
    pub(crate) fn of_class(class: &str) -> Option<Self> {
        SIMPLE_OPS_BY_CLASS.get(class).copied()
    }
}

/// Every simple operation that has a Java class. (Negation, reciprocal and square root are saved as a binary
/// operation with a constant.)
pub(crate) const SIMPLE_OPS: &[(SimpleOp, &str)] = {
    use SimpleOp as S;
    &[
        (S::Unary(UnaryOp::Abs), "AbsVertex"),
        (S::Unary(UnaryOp::Sign), "SignVertex"),
        (S::Unary(UnaryOp::Exp), "ExpVertex"),
        (S::Unary(UnaryOp::Exp2), "Exp2Vertex"),
        (S::Unary(UnaryOp::ExpM1), "ExpM1Vertex"),
        (S::Unary(UnaryOp::Ln), "LogVertex"),
        (S::Unary(UnaryOp::Ln1p), "Log1pVertex"),
        (S::Unary(UnaryOp::Log2), "Log2Vertex"),
        (S::Unary(UnaryOp::Log10), "Log10Vertex"),
        (S::Unary(UnaryOp::Sin), "SinVertex"),
        (S::Unary(UnaryOp::Cos), "CosVertex"),
        (S::Unary(UnaryOp::Tan), "TanVertex"),
        (S::Unary(UnaryOp::Asin), "ArcSinVertex"),
        (S::Unary(UnaryOp::Acos), "ArcCosVertex"),
        (S::Unary(UnaryOp::Atan), "ArcTanVertex"),
        (S::Unary(UnaryOp::Sinh), "SinhVertex"),
        (S::Unary(UnaryOp::Cosh), "CoshVertex"),
        (S::Unary(UnaryOp::Tanh), "TanhVertex"),
        (S::Unary(UnaryOp::Asinh), "ArcSinhVertex"),
        (S::Unary(UnaryOp::Acosh), "ArcCoshVertex"),
        (S::Unary(UnaryOp::Atanh), "ArcTanhVertex"),
        (S::Unary(UnaryOp::Sigmoid), "SigmoidVertex"),
        (S::Unary(UnaryOp::Floor), "FloorVertex"),
        (S::Unary(UnaryOp::Ceil), "CeilVertex"),
        (S::Unary(UnaryOp::Round), "RoundVertex"),
        (S::Unary(UnaryOp::LnGamma), "LogGammaVertex"),
        (S::Unary(UnaryOp::Digamma), "DigammaVertex"),
        (S::Unary(UnaryOp::Trigamma), "TrigammaVertex"),
        (S::Binary(BinaryOp::Add), "AdditionVertex"),
        (S::Binary(BinaryOp::Sub), "DifferenceVertex"),
        (S::Binary(BinaryOp::Mul), "MultiplicationVertex"),
        (S::Binary(BinaryOp::Div), "DivisionVertex"),
        (S::Binary(BinaryOp::Pow), "PowerVertex"),
        (S::Binary(BinaryOp::Rem), "ModVertex"),
        (S::Binary(BinaryOp::Atan2), "ArcTan2Vertex"),
        (S::Binary(BinaryOp::LogAddExp), "LogAddExpVertex"),
        (S::Binary(BinaryOp::LogAddExp2), "LogAddExp2Vertex"),
        (S::Binary(BinaryOp::SafeLnTimes), "SafeLogTimesVertex"),
        (S::Binary(BinaryOp::Max), "MaxVertex"),
        (S::Binary(BinaryOp::Min), "MinVertex"),
        (S::Compare(CompareOp::Gt), "GreaterThanVertex"),
        (S::Compare(CompareOp::Ge), "GreaterThanOrEqualVertex"),
        (S::Compare(CompareOp::Lt), "LessThanVertex"),
        (S::Compare(CompareOp::Le), "LessThanOrEqualVertex"),
        (S::Compare(CompareOp::Eq), "EqualsVertex"),
        (S::Compare(CompareOp::Ne), "NotEqualsVertex"),
        (S::Mask(CompareOp::Gt), "GreaterThanMaskVertex"),
        (S::Mask(CompareOp::Ge), "GreaterThanOrEqualToMaskVertex"),
        (S::Mask(CompareOp::Lt), "LessThanMaskVertex"),
        (S::Mask(CompareOp::Le), "LessThanOrEqualToMaskVertex"),
        (S::Logic(LogicOp::And), "AndBinaryVertex"),
        (S::Logic(LogicOp::Or), "OrBinaryVertex"),
        (S::Logic(LogicOp::Xor), "XorBinaryVertex"),
        (S::Predicate(Predicate::IsNan), "IsNaNVertex"),
        (S::Predicate(Predicate::NotNan), "NotNaNVertex"),
        (S::Predicate(Predicate::IsFinite), "IsFiniteVertex"),
        (S::Predicate(Predicate::IsInfinite), "IsInfiniteVertex"),
        (
            S::Predicate(Predicate::IsPositiveInfinity),
            "IsPositiveInfinityVertex",
        ),
        (
            S::Predicate(Predicate::IsNegativeInfinity),
            "IsNegativeInfinityVertex",
        ),
    ]
};

static SIMPLE_OPS_BY_OP: LazyLock<HashMap<SimpleOp, &str>> =
    LazyLock::new(|| SIMPLE_OPS.iter().copied().collect());

static SIMPLE_OPS_BY_CLASS: LazyLock<HashMap<&str, SimpleOp>> =
    LazyLock::new(|| SIMPLE_OPS.iter().map(|&(op, class)| (class, op)).collect());

/// The simple name of a fully qualified Java class name.
pub(crate) fn simple_name(class: &str) -> &str {
    class.rsplit('.').next().unwrap_or(class)
}

/// The fully qualified Java class name of a simple one, if it is in [`JAVA_CLASSES`].
pub(crate) fn java_class(simple: &str) -> Option<&'static str> {
    JAVA_CLASSES
        .binary_search_by(|class| simple_name(class).cmp(simple))
        .ok()
        .map(|i| JAVA_CLASSES[i])
}

/// Fully qualified Java class names, sorted by simple name.
pub(crate) const JAVA_CLASSES: &[&str] = &[
    "io.improbable.keanu.vertices.tensor.number.operators.unary.AbsVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.binary.AdditionVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.AllFalseVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.AllTrueVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.AndBinaryVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.AnyFalseVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.AnyTrueVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcCosVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcCoshVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcSinVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcSinhVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.binary.ArcTan2Vertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcTanVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcTanhVertex",
    "io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.operators.unary.ArgMaxVertex",
    "io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.operators.unary.ArgMinVertex",
    "io.improbable.keanu.vertices.utility.AssertVertex",
    "io.improbable.keanu.vertices.tensor.bool.probabilistic.BernoulliVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.BetaVertex",
    "io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic.BinomialVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.multiple.BooleanConcatenationVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.BooleanProxyVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.multiple.BooleanToDoubleMaskVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.multiple.BooleanToIntegerMaskVertex",
    "io.improbable.keanu.vertices.tensor.BroadcastVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.CastNumberToBooleanVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.CastNumberToDoubleVertex",
    "io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.CastNumberToIntegerVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.CastToBooleanVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.CauchyVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.CeilVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.ChiSquaredVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.CholeskyDecompositionVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.CholeskyInverseVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.ConstantBooleanVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex",
    "io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.ConstantIntegerVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.CosVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.CoshVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.unary.CumProdVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.unary.CumSumVertex",
    "io.improbable.keanu.vertices.tensor.DiagPartVertex",
    "io.improbable.keanu.vertices.tensor.DiagVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.binary.DifferenceVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.DigammaVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.DirichletVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.binary.DivisionVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.DoubleProxyVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.compare.EqualsVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.Exp2Vertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ExpM1Vertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ExpVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.ExponentialVertex",
    "io.improbable.keanu.vertices.tensor.FillTriangularVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.FloorVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GammaVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GaussianVertex",
    "io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic.GeometricVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.binary.GreaterThanMaskVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.binary.GreaterThanOrEqualToMaskVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.compare.GreaterThanOrEqualVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.HalfCauchyVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.HalfGaussianVertex",
    "io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.operators.multiple.IntegerConcatenationVertex",
    "io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.IntegerProxyVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.InverseGammaVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.IsFiniteVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.IsInfiniteVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.IsNaNVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.IsNegativeInfinityVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.IsPositiveInfinityVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.KDEVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.LaplaceVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.binary.LessThanMaskVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.binary.LessThanOrEqualToMaskVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.compare.LessThanVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.Log10Vertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.Log1pVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.Log2Vertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.binary.LogAddExp2Vertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.binary.LogAddExpVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.LogGammaVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.LogNormalVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.LogVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.LogisticVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.MatrixDeterminantVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.MatrixInverseVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.binary.MatrixMultiplicationVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.unary.MaxUnaryVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.binary.MaxVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.MeanVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.unary.MinUnaryVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.binary.MinVertex",
    "io.improbable.keanu.vertices.tensor.number.fixed.operators.unary.ModVertex",
    "io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic.MultinomialVertex",
    "io.improbable.keanu.vertices.tensor.generic.nonprobabilistic.MultiplexerVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.binary.MultiplicationVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.MultivariateGaussianVertex",
    "io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.operators.unary.NaNArgMaxVertex",
    "io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.operators.unary.NaNArgMinVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.NotBinaryVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.NotNaNVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.compare.NumericalEqualsVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.OrBinaryVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.ParetoVertex",
    "io.improbable.keanu.vertices.tensor.PermuteVertex",
    "io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic.PoissonVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.binary.PowerVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.unary.ProductVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ReplaceNaNVertex",
    "io.improbable.keanu.vertices.tensor.ReshapeVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.RoundVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.binary.SafeLogTimesVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.ternary.SetWithMaskVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.SigmoidVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.unary.SignVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.SinVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.SinhVertex",
    "io.improbable.keanu.vertices.tensor.SliceVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.StandardDeviationVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.StandardizeVertex",
    "io.improbable.keanu.vertices.tensor.StridedSliceVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.StudentTVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.unary.SumVertex",
    "io.improbable.keanu.vertices.tensor.TakeVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.TanVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.TanhVertex",
    "io.improbable.keanu.vertices.tensor.number.operators.binary.TensorMultiplicationVertex",
    "io.improbable.keanu.vertices.tensor.TriLowerVertex",
    "io.improbable.keanu.vertices.tensor.TriUpperVertex",
    "io.improbable.keanu.vertices.tensor.TrianglePartVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.TriangularVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.operators.unary.TrigammaVertex",
    "io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic.UniformIntVertex",
    "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex",
    "io.improbable.keanu.vertices.tensor.WhereVertex",
    "io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.XorBinaryVertex",
];

/// The tables are complete and consistent: every entry has a Java class, saves as it with the table's parameter
/// names, loads back as the same operation with the same parents and (distributions) labels its DOT edges.
#[cfg(all(test, any(feature = "protobuf", feature = "json")))]
mod tests {
    use super::*;
    use crate::{Metadata, load::load_model, save::save_model};
    use keanu_graph::{Graph, GraphBuilder, NodeId, State, Value};
    use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};
    use std::collections::BTreeSet;

    /// Saves and loads a graph whose last node is the one under test, checking the saved vertex's class and
    /// parameter names, and returns the loaded graph.
    fn round_trip(graph: &Graph, state: &State, class: &str, parameters: &[&str]) -> Graph {
        let model = save_model(graph, state, &Metadata::new()).unwrap();
        let vertex = model.graph.as_ref().unwrap().vertices.last().unwrap();
        assert_eq!(vertex.vertex_type, java_class(class).unwrap(), "{class}");
        let saved: BTreeSet<&str> = vertex.parameters.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(saved, parameters.iter().copied().collect(), "{class}");
        load_model(&model).unwrap().graph
    }

    fn last(graph: &Graph) -> NodeId {
        graph.ids().last().unwrap()
    }

    #[test]
    fn java_classes_are_sorted_by_simple_name() {
        assert!(
            JAVA_CLASSES
                .windows(2)
                .all(|w| simple_name(w[0]) < simple_name(w[1]))
        );
        assert_eq!(
            java_class("GaussianVertex"),
            Some(
                "io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GaussianVertex"
            )
        );
        assert_eq!(java_class("NoSuchVertex"), None);
    }

    #[test]
    fn every_distribution_round_trips() {
        for saved in DISTRIBUTIONS {
            assert_eq!(
                distribution_of_class(saved.class).map(|d| d.class),
                Some(saved.class)
            );
            // find parameter types and shapes the distribution accepts: all scalars, or rank i + 1 for parameter i
            // (a multivariate Gaussian's vector mean and matrix covariance), each f64 or i64
            let n = saved.parameters.len();
            let built = (0..2)
                .flat_map(|ranked| (0..1 << n).map(move |ints| (ranked, ints)))
                .find_map(|(ranked, ints)| {
                    let b = GraphBuilder::new();
                    let parents: Vec<NodeId> = (0..n)
                        .map(|i| {
                            let shape = vec![1; ranked * (i + 1)];
                            let value = if ints & (1 << i) != 0 {
                                Value::I64(IntTensor::full(&shape, 1))
                            } else {
                                Value::F64(DoubleTensor::full(&shape, 1.0))
                            };
                            b.add_node(Op::Constant(value), &[]).unwrap()
                        })
                        .collect();
                    b.add_distribution(saved.kind.clone(), None, &parents)
                        .ok()?;
                    b.build().ok()
                });
            let (graph, state) = built.unwrap_or_else(|| panic!("can't build {}", saved.class));
            let mut parameters = saved.parameters.to_vec();
            parameters.extend(saved.zero_location);
            let loaded = round_trip(&graph, &state, saved.class, &parameters);
            let node = &loaded[last(&loaded)];
            let Op::Distribution(d) = node.op() else {
                panic!("{} loaded as {:?}", saved.class, node.op());
            };
            assert_eq!(
                std::mem::discriminant(&d.kind),
                std::mem::discriminant(&saved.kind)
            );
            assert_eq!(node.parents(), graph[last(&graph)].parents());
            #[cfg(feature = "dot")]
            {
                let dot = crate::to_dot(&graph, None, &Metadata::new());
                for name in saved.parameters {
                    assert!(dot.contains(&format!("[label={name}]")), "{dot}");
                }
            }
        }
    }

    #[test]
    fn every_simple_op_round_trips() {
        for &(simple, class) in SIMPLE_OPS {
            assert_eq!(SimpleOp::of_class(class), Some(simple));
            assert_eq!(simple.class(), Some(class));
            assert_eq!(SimpleOp::of(&simple.op()), Some(simple));
            // the first parent type the operation applies to
            let values = [
                Value::F64(DoubleTensor::scalar(0.5)),
                Value::I64(IntTensor::scalar(1)),
                Value::Bool(BoolTensor::scalar(true)),
            ];
            let built = values.into_iter().find_map(|value| {
                let b = GraphBuilder::new();
                let parents: Vec<NodeId> = simple
                    .parameters()
                    .iter()
                    .map(|_| b.add_node(Op::Constant(value.clone()), &[]).unwrap())
                    .collect();
                b.add_node(simple.op(), &parents).ok()?;
                b.build().ok().map(|built| (built, parents))
            });
            let ((graph, state), parents) = built.unwrap_or_else(|| panic!("can't build {class}"));
            let loaded = round_trip(&graph, &state, class, simple.parameters());
            let node = &loaded[last(&loaded)];
            assert_eq!(SimpleOp::of(node.op()), Some(simple), "{class}");
            assert_eq!(node.parents(), &parents[..]);
        }
    }
}
