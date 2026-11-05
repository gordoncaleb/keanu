//! The saved code of each enumerated operation kind: a table per enum pairing each Keanu variant with its schema
//! enum value. Adding a variant to one of Keanu's enums means adding its value to `proto/keanu/graph.proto` and a
//! row here.

use crate::error::{IoError, Result};
use crate::proto;
use keanu_distributions::Kde;
use keanu_graph::DType;
use keanu_graph::ode::OdeSolver;
use keanu_graph::op::{
    ArgOp, BinaryOp, CompareOp, CumulativeOp, DistributionKind, LogicOp, Nested, Predicate,
    ReduceOp, UnaryOp,
};
use std::sync::Arc;

/// A Keanu enum saved as a schema enum value.
pub(crate) trait Code: Sized {
    /// What the enum is, for error messages.
    const WHAT: &'static str;
    /// The schema value, or None for a variant without one.
    fn code(&self) -> Option<i32>;
    /// The variant of a schema value, or None for an unknown or unspecified one.
    fn of_code(code: i32) -> Option<Self>;
}

/// The schema value of a variant, or a not-saveable error.
pub(crate) fn encode<T: Code + std::fmt::Debug>(value: &T) -> Result<i32> {
    value
        .code()
        .ok_or_else(|| IoError::NotSaveable(format!("{} {value:?} has no saved form", T::WHAT)))
}

/// The variant of a schema value, or a decode error.
pub(crate) fn decode<T: Code>(code: i32) -> Result<T> {
    T::of_code(code).ok_or_else(|| IoError::Decode(format!("unknown {} {code}", T::WHAT)))
}

/// `Keanu enum => schema enum { KeanuVariant => SchemaVariant, ... }`, where `KeanuVariant` alone means the same
/// name in both.
macro_rules! codes {
    ($($what:literal: $rust:ty => $proto:ident { $($variant:ident $(=> $saved:ident)?),+ $(,)? })+) => {$(
        impl Code for $rust {
            const WHAT: &'static str = $what;

            #[allow(unreachable_patterns)]
            fn code(&self) -> Option<i32> {
                match self {
                    $(<$rust>::$variant => Some(codes!(@saved $proto $variant $($saved)?) as i32),)+
                    _ => None,
                }
            }

            fn of_code(code: i32) -> Option<Self> {
                match proto::$proto::try_from(code).ok()? {
                    $(codes!(@saved $proto $variant $($saved)?) => Some(<$rust>::$variant),)+
                    _ => None,
                }
            }
        }
    )+};
    (@saved $proto:ident $variant:ident) => { proto::$proto::$variant };
    (@saved $proto:ident $variant:ident $saved:ident) => { proto::$proto::$saved };
}

codes! {
    "unary op": UnaryOp => UnaryOp {
        Abs, Neg, Sign, Reciprocal, Exp, Exp2, ExpM1, Ln, Ln1p, Log2, Log10, Sqrt, Sin, Cos, Tan, Asin, Acos, Atan,
        Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Sigmoid, Floor, Ceil, Round, LnGamma, Digamma, Trigamma,
        Erf, Erfc, ErfInv, LogNdtr, Ndtri, Softplus, Logit, Log1mExp,
    }
    "binary op": BinaryOp => BinaryOp {
        Add, Sub, Mul, Div, Pow, Max, Min, Rem, Atan2, LogAddExp, LogAddExp2, SafeLnTimes, LogBeta,
    }
    "comparison": CompareOp => CompareOp { Gt, Ge, Lt, Le, Eq, Ne }
    "logic op": LogicOp => LogicOp { And, Or, Xor }
    "predicate": Predicate => Predicate {
        IsNan, NotNan, IsFinite, IsInfinite, IsPositiveInfinity, IsNegativeInfinity,
    }
    "reduction": ReduceOp => ReduceOp { Sum, Product, Mean, Max, Min, StdDev, LogSumExp }
    "cumulative op": CumulativeOp => CumulativeOp { CumSum => Sum, CumProd => Product }
    "arg reduction": ArgOp => ArgOp { ArgMax, ArgMin, NanArgMax, NanArgMin }
    "type": DType => DType { F64, I64, Bool }
    "ODE solver": OdeSolver => OdeSolver { Rk45, Stiff }
}

/// The schema value of each way of building a kind around another one (see [`DistributionKind::nested`]).
const NESTED_CODES: [(Nested, proto::DistributionKind); 6] = [
    (Nested::Truncated, proto::DistributionKind::Truncated),
    (Nested::Censored, proto::DistributionKind::Censored),
    (Nested::Mixture, proto::DistributionKind::Mixture),
    (
        Nested::HiddenMarkovModel,
        proto::DistributionKind::HiddenMarkovModel,
    ),
    (Nested::ZeroInflated, proto::DistributionKind::ZeroInflated),
    (Nested::Hurdle, proto::DistributionKind::Hurdle),
];

/// The schema value of a way of nesting, or None for one without a saved form.
fn nested_code(nested: Nested) -> Option<proto::DistributionKind> {
    NESTED_CODES
        .iter()
        .find(|(n, _)| *n == nested)
        .map(|(_, code)| *code)
}

/// The way of nesting a schema value saves, or None if it isn't a nested kind.
fn nested_of_code(code: proto::DistributionKind) -> Option<Nested> {
    NESTED_CODES
        .iter()
        .find(|(_, c)| *c == code)
        .map(|(nested, _)| *nested)
}

/// The distributions whose kind carries no data, by the same name in Keanu and the schema.
macro_rules! distribution_codes {
    ($($kind:ident),+ $(,)?) => {
        /// The saved type of a distribution: its schema kind, a KDE's samples and a nested kind's base (saved the
        /// same way, recursively). None for a custom distribution, or a kind built around one.
        pub(crate) fn distribution_type(kind: &DistributionKind) -> Option<proto::DistributionType> {
            let (code, kde, base) = match kind {
                $(DistributionKind::$kind => (proto::DistributionKind::$kind, None, None),)+
                DistributionKind::Kde(kde) => (
                    proto::DistributionKind::Kde,
                    Some(proto::Kde {
                        samples: kde.samples.clone(),
                        bandwidth: kde.bandwidth,
                    }),
                    None,
                ),
                _ => {
                    let (nested, base) = kind.nested()?;
                    (nested_code(nested)?, None, Some(base))
                }
            };
            let base = match base {
                Some(base) => Some(Box::new(distribution_type(base)?)),
                None => None,
            };
            Some(proto::DistributionType {
                kind: code as i32,
                kde,
                base,
            })
        }

        /// The distribution of a saved type, with a KDE built from its saved samples and a nested kind around its
        /// saved base.
        pub(crate) fn distribution_of_type(saved: &proto::DistributionType) -> Result<DistributionKind> {
            let code = saved.kind;
            let unknown = || IoError::Decode(format!("unknown distribution {code}"));
            let kind = proto::DistributionKind::try_from(code).map_err(|_| unknown())?;
            if saved.kde.is_some() && kind != proto::DistributionKind::Kde {
                return Err(IoError::Decode(format!(
                    "a {} distribution has KDE samples",
                    kind.as_str_name()
                )));
            }
            let nested = nested_of_code(kind);
            let base = match (&saved.base, nested.is_some()) {
                (Some(base), true) => Some(Arc::new(distribution_of_type(base)?)),
                (None, false) => None,
                (None, true) => {
                    return Err(IoError::Decode(format!(
                        "a {} distribution has no base distribution",
                        kind.as_str_name()
                    )));
                }
                (Some(_), false) => {
                    return Err(IoError::Decode(format!(
                        "a {} distribution has a base distribution",
                        kind.as_str_name()
                    )));
                }
            };
            Ok(match (kind, base) {
                $((proto::DistributionKind::$kind, _) => DistributionKind::$kind,)+
                (proto::DistributionKind::Kde, _) => {
                    let kde = saved.kde.as_ref().ok_or_else(|| IoError::Decode("a KDE has no samples".into()))?;
                    let kde = Kde::new(kde.samples.clone(), kde.bandwidth)
                        .map_err(|e| IoError::Decode(format!("invalid KDE: {e}")))?;
                    DistributionKind::Kde(Arc::new(kde))
                }
                (_, base) => match (nested, base) {
                    (Some(nested), Some(base)) => nested.wrap(base),
                    _ => return Err(unknown()),
                },
            })
        }
    };
}

distribution_codes!(
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
    LinearGaussianStateSpace,
    ZeroSumNormal,
    BetaBinomial,
    BernoulliLogit,
    BinomialLogit,
    PoissonLog,
    CategoricalLogit,
    GaussianRandomWalk,
    Autoregressive,
    Garch11,
);

#[cfg(test)]
mod tests {
    use super::*;

    /// Every value of the schema enum has a Keanu variant that saves as it, and 0 (unspecified) has none.
    fn every_code_has_a_variant<T: Code + std::fmt::Debug>(count: i32) {
        assert!(T::of_code(0).is_none(), "{}", T::WHAT);
        assert!(T::of_code(count + 1).is_none(), "{}", T::WHAT);
        for code in 1..=count {
            let variant = decode::<T>(code).unwrap();
            assert_eq!(encode(&variant).unwrap(), code, "{} {variant:?}", T::WHAT);
        }
    }

    /// The number of distribution kinds in the schema.
    const DISTRIBUTION_KINDS: i32 = 47;

    #[test]
    fn every_schema_value_has_a_variant() {
        every_code_has_a_variant::<UnaryOp>(39);
        every_code_has_a_variant::<BinaryOp>(13);
        every_code_has_a_variant::<CompareOp>(6);
        every_code_has_a_variant::<LogicOp>(3);
        every_code_has_a_variant::<Predicate>(6);
        every_code_has_a_variant::<ReduceOp>(7);
        every_code_has_a_variant::<CumulativeOp>(2);
        every_code_has_a_variant::<ArgOp>(4);
        every_code_has_a_variant::<DType>(3);
        let saved = |kind: i32, kde: Option<proto::Kde>, base: Option<proto::DistributionType>| {
            proto::DistributionType {
                kind,
                kde,
                base: base.map(Box::new),
            }
        };
        let gaussian = || saved(proto::DistributionKind::Gaussian as i32, None, None);
        let nested = NESTED_CODES.map(|(_, code)| code as i32);
        for code in 1..=DISTRIBUTION_KINDS {
            let kde = (code == proto::DistributionKind::Kde as i32).then(|| proto::Kde {
                samples: vec![1.0],
                bandwidth: 1.0,
            });
            let base = nested.contains(&code).then(gaussian);
            let original = saved(code, kde, base);
            let kind = distribution_of_type(&original).unwrap();
            assert_eq!(distribution_type(&kind), Some(original));
        }
        assert!(distribution_of_type(&saved(0, None, None)).is_err());
        assert!(distribution_of_type(&saved(DISTRIBUTION_KINDS + 1, None, None)).is_err());
        // a nested kind needs a base, and only a nested kind has one
        assert!(distribution_of_type(&saved(nested[0], None, None)).is_err());
        assert!(distribution_of_type(&saved(1, None, Some(gaussian()))).is_err());
        // nesting is recursive
        let deep = saved(
            nested[2],
            None,
            Some(saved(nested[0], None, Some(gaussian()))),
        );
        let kind = distribution_of_type(&deep).unwrap();
        assert!(
            matches!(&kind, DistributionKind::Mixture(base) if matches!(**base, DistributionKind::Truncated(_)))
        );
        assert_eq!(distribution_type(&kind), Some(deep));
    }
}
