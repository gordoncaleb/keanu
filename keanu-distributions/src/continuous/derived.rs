//! The families written as one scalar formula over [jets](crate::jet): each element's log density (and log CDF
//! and log survival function, where the family has them) is one generic function, from which the values, the
//! gradients and the second derivatives all follow exactly.

/// Defines a family of f64 parameters whose element-wise log density is the generic scalar function
/// `log_prob(x, [parameters])` (and, with `log_cdf`, whose log CDF or, if `upper`, log survival function is
/// `log_cdf(x, [parameters], upper)`): the struct, its gradient struct, `new`, `log_prob`, `d_log_prob` and
/// `log_prob_derivatives`, and with a CDF `log_cdf`, `log_ccdf`, `d_log_cdf`, `d_log_ccdf` and
/// `log_cdf_derivatives`. `inputs` is 1 + the number of parameters. The sampler (and any inverse CDF) is written
/// separately.
macro_rules! derived_distribution {
    (@cdf $name:ident, $gradient:ident, $n:literal, [$($param:ident),+]) => {};
    (@cdf $name:ident, $gradient:ident, $n:literal, [$($param:ident),+], $log_cdf:path) => {
            impl $name<'_> {
                /// The element-wise log CDF, ln P(X <= x), broadcasting x with the parameters.
                pub fn log_cdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
                    map((x, $(&*self.$param),+), |(x, $($param),+)| {
                        crate::jet::value(x, [$($param),+], |x, p| {
                            crate::continuous::derived::limit_or(x, false, |x| $log_cdf(x, p, false))
                        })
                    })
                }

                /// The element-wise log survival function, ln P(X > x), broadcasting x with the parameters.
                pub fn log_ccdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
                    map((x, $(&*self.$param),+), |(x, $($param),+)| {
                        crate::jet::value(x, [$($param),+], |x, p| {
                            crate::continuous::derived::limit_or(x, true, |x| $log_cdf(x, p, true))
                        })
                    })
                }

                /// The element-wise derivatives of [`Self::log_cdf`] with respect to x and each parameter.
                pub fn d_log_cdf(&self, x: &DoubleTensor) -> Result<$gradient> {
                    let [x, $($param),+] = map_n((x, $(&*self.$param),+), |(x, $($param),+)| {
                        Self::log_cdf_derivatives(x, [$($param),+], false).gradient
                    })?;
                    Ok($gradient { x, $($param),+ })
                }

                /// The element-wise derivatives of [`Self::log_ccdf`] with respect to x and each parameter.
                pub fn d_log_ccdf(&self, x: &DoubleTensor) -> Result<$gradient> {
                    let [x, $($param),+] = map_n((x, $(&*self.$param),+), |(x, $($param),+)| {
                        Self::log_cdf_derivatives(x, [$($param),+], true).gradient
                    })?;
                    Ok($gradient { x, $($param),+ })
                }

                /// The log CDF of one element (or, if `upper`, its log survival function) with its gradient and
                /// Hessian with respect to x (index 0) and each parameter (in order).
                pub fn log_cdf_derivatives(
                    x: f64,
                    parameters: [f64; $n - 1],
                    upper: bool,
                ) -> ElementDerivatives<$n> {
                    crate::jet::derivatives(x, parameters, |x, p| {
                        crate::continuous::derived::limit_or(x, upper, |x| $log_cdf(x, p, upper))
                    })
                }
            }

    };
    (
        $(#[$doc:meta])*
        $name:ident, $gradient:ident, inputs = $n:literal,
        ($($(#[$pdoc:meta])* $param:ident),+ $(,)?),
        log_prob = $log_prob:path
        $(, log_cdf = $log_cdf:path)? $(,)?
    ) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $name<'a> {
            $($(#[$pdoc])* pub $param: Param<'a>,)+
        }

        #[doc = concat!(
            "Derivatives of [`", stringify!($name), "::log_prob`] (and of its log CDF and log survival function, ",
            "where it has them)."
        )]
        #[derive(Debug, Clone, PartialEq)]
        pub struct $gradient {
            /// With respect to x.
            pub x: DoubleTensor,
            $(#[doc = concat!("With respect to `", stringify!($param), "`.")] pub $param: DoubleTensor,)+
        }

        impl<'a> $name<'a> {
            /// Creates the distribution from its parameters.
            pub fn new($($param: impl IntoParam<'a, f64>),+) -> Self {
                Self { $($param: $param.into_param()),+ }
            }

            /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the
            /// support, NaN for invalid parameters).
            pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
                map((x, $(&*self.$param),+), |(x, $($param),+)| {
                    crate::jet::value(x, [$($param),+], |x, p| $log_prob(x, p))
                })
            }

            /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
            pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<$gradient> {
                let [x, $($param),+] = map_n((x, $(&*self.$param),+), |(x, $($param),+)| {
                    Self::log_prob_derivatives(x, [$($param),+]).gradient
                })?;
                Ok($gradient { x, $($param),+ })
            }

            /// The log density of one element with its gradient and Hessian with respect to x (index 0) and each
            /// parameter (in order), computed exactly by forward-mode differentiation to second order.
            pub fn log_prob_derivatives(x: f64, parameters: [f64; $n - 1]) -> ElementDerivatives<$n> {
                crate::jet::derivatives(x, parameters, |x, p| $log_prob(x, p))
            }
        }

        derived_distribution!(@cdf $name, $gradient, $n, [$($param),+] $(, $log_cdf)?);
    };
}

use super::*;

/// ln p from ln p or ln(1 − p) computed directly, choosing whichever keeps its precision: `lower` (ln p) where
/// p < ½, else ln(1 − e^{ln(1 − p)}).
pub(super) fn log_from_either<const K: usize>(p: Jet<K>, complement: Jet<K>) -> Jet<K> {
    if p.v < 0.5 {
        p.ln()
    } else {
        (-complement).ln_1p()
    }
}

/// A log probability in jets from the log of one probability and of its complement: the first where it is below
/// ln ½, else ln(1 − e^{second}).
pub(super) fn log_complement<const K: usize>(
    log_p: Jet<K>,
    log_q: impl FnOnce() -> Jet<K>,
) -> Jet<K> {
    if log_p.v < -std::f64::consts::LN_2 {
        log_p
    } else {
        log_q().log1mexp()
    }
}

/// The log CDF (or, if `upper`, log survival function) at an infinite point, its limit, else `f(x)`.
pub(super) fn limit_or<const K: usize>(
    x: Jet<K>,
    upper: bool,
    f: impl FnOnce(Jet<K>) -> Jet<K>,
) -> Jet<K> {
    let (at_minus, at_plus) = if upper {
        (0.0, f64::NEG_INFINITY)
    } else {
        (f64::NEG_INFINITY, 0.0)
    };
    if x.v == f64::INFINITY {
        Jet::constant(at_plus)
    } else if x.v == f64::NEG_INFINITY {
        Jet::constant(at_minus)
    } else {
        f(x)
    }
}

/// The log CDF (or, if `upper`, log survival function) at a point below the support (`below`) or above it.
pub(super) fn outside<const K: usize>(below: bool, upper: bool) -> Jet<K> {
    Jet::constant(if below == upper {
        0.0
    } else {
        f64::NEG_INFINITY
    })
}

/// An error with `message` unless `valid` holds for the parameters at every element of their broadcast.
pub(super) fn check_all(
    parameters: &[&DoubleTensor],
    valid: impl Fn(&[f64]) -> bool,
    message: impl FnOnce() -> String,
) -> Result<()> {
    let mut ok = true;
    sample_each::<f64>(&[], parameters, |p| {
        ok &= valid(p);
        Ok(0.0)
    })?;
    if ok {
        Ok(())
    } else {
        invalid_parameter(message())
    }
}

/// Uniform draws in the open interval (0, 1), of the given shape.
pub(super) fn uniform_open(shape: &[usize], rng: &mut KeanuRng) -> DoubleTensor {
    let len = shape.iter().product();
    let data = (0..len).map(|_| rng.next_f64_non_zero()).collect();
    DoubleTensor::new(shape, data).expect("one value per element of the shape")
}
