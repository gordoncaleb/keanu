//! The saved code of each enumerated operation kind: a table per enum pairing each Keanu variant with its schema
//! enum value. Adding a variant to one of Keanu's enums means adding its value to `proto/keanu/graph.proto` and a
//! row here.

use crate::error::{IoError, Result};
use crate::proto;
use keanu_distributions::Kde;
use keanu_graph::DType;
use keanu_graph::op::{
    ArgOp, BinaryOp, CompareOp, DistributionKind, LogicOp, Predicate, ReduceOp, ScanOp, UnaryOp,
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
    "reduction": ReduceOp => ReduceOp { Sum, Product, Mean, Max, Min, StdDev }
    "cumulative op": ScanOp => CumulativeOp { CumSum => Sum, CumProd => Product }
    "arg reduction": ArgOp => ArgOp { ArgMax, ArgMin, NanArgMax, NanArgMin }
    "type": DType => DType { F64, I64, Bool }
}

/// The distributions whose kind carries no data, by the same name in Keanu and the schema.
macro_rules! distribution_codes {
    ($($kind:ident),+ $(,)?) => {
        /// The schema kind of a distribution (a KDE's samples are saved separately), or None for a custom one.
        pub(crate) fn distribution_code(kind: &DistributionKind) -> Option<proto::DistributionKind> {
            match kind {
                $(DistributionKind::$kind => Some(proto::DistributionKind::$kind),)+
                DistributionKind::Kde(_) => Some(proto::DistributionKind::Kde),
                _ => None,
            }
        }

        /// The distribution of a schema kind, with a KDE built from its saved samples.
        pub(crate) fn distribution_of_code(
            code: i32,
            kde: Option<&proto::Kde>,
        ) -> Result<DistributionKind> {
            let unknown = || IoError::Decode(format!("unknown distribution {code}"));
            let kind = proto::DistributionKind::try_from(code).map_err(|_| unknown())?;
            if kde.is_some() && kind != proto::DistributionKind::Kde {
                return Err(IoError::Decode(format!(
                    "a {} distribution has KDE samples",
                    kind.as_str_name()
                )));
            }
            Ok(match kind {
                $(proto::DistributionKind::$kind => DistributionKind::$kind,)+
                proto::DistributionKind::Kde => {
                    let kde = kde.ok_or_else(|| IoError::Decode("a KDE has no samples".into()))?;
                    let kde = Kde::new(kde.samples.clone(), kde.bandwidth)
                        .map_err(|e| IoError::Decode(format!("invalid KDE: {e}")))?;
                    DistributionKind::Kde(Arc::new(kde))
                }
                proto::DistributionKind::Unspecified => return Err(unknown()),
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

    #[test]
    fn every_schema_value_has_a_variant() {
        every_code_has_a_variant::<UnaryOp>(39);
        every_code_has_a_variant::<BinaryOp>(13);
        every_code_has_a_variant::<CompareOp>(6);
        every_code_has_a_variant::<LogicOp>(3);
        every_code_has_a_variant::<Predicate>(6);
        every_code_has_a_variant::<ReduceOp>(6);
        every_code_has_a_variant::<ScanOp>(2);
        every_code_has_a_variant::<ArgOp>(4);
        every_code_has_a_variant::<DType>(3);
        for code in 1..=27 {
            let kde = (code == proto::DistributionKind::Kde as i32).then(|| proto::Kde {
                samples: vec![1.0],
                bandwidth: 1.0,
            });
            let kind = distribution_of_code(code, kde.as_ref()).unwrap();
            assert_eq!(distribution_code(&kind).map(|k| k as i32), Some(code));
        }
        assert!(distribution_of_code(0, None).is_err());
        assert!(distribution_of_code(28, None).is_err());
    }
}
