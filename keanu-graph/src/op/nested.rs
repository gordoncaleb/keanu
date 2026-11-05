//! Kinds built around another kind: truncated, censored, mixture and hidden Markov model distributions.
//!
//! Each holds its inner kind (the base, component or emission) and takes the inner kind's parameters followed by
//! its own (see the wrappers module for the layout). [`DistributionKind::nested`] splits such a kind into a
//! [`Nested`] tag and the inner kind, and the tag's methods here are the one place that dispatches to the
//! wrappers, mixture and hmm modules, so the distribution table and the second-order rules delegate to them once.

use super::distribution::{DistributionKind, LogProbGradientParts};
use super::distribution_hvp::{self, LogProbHvpParts};
use super::wrappers::{self, Wrapper};
use super::{Support, hmm, mixture};
use crate::error::Result;
use crate::value::{DType, Value};
use keanu_distributions::KeanuRng;
use keanu_tensor::DoubleTensor;
use std::borrow::Cow;
use std::sync::Arc;

/// How a nested [`DistributionKind`] wraps its inner kind (see [`DistributionKind::nested`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Nested {
    /// [`DistributionKind::Truncated`]: the inner kind is the base.
    Truncated,
    /// [`DistributionKind::Censored`]: the inner kind is the base.
    Censored,
    /// [`DistributionKind::Mixture`]: the inner kind is the component.
    Mixture,
    /// [`DistributionKind::HiddenMarkovModel`]: the inner kind is the emission.
    HiddenMarkovModel,
}

/// The pattern matching every nested kind, for the exhaustive matches over [`DistributionKind`] (whose arm then
/// calls [`DistributionKind::nested_parts`]).
macro_rules! nested_kinds {
    () => {
        DistributionKind::Truncated(_)
            | DistributionKind::Censored(_)
            | DistributionKind::Mixture(_)
            | DistributionKind::HiddenMarkovModel(_)
    };
}
pub(crate) use nested_kinds;

impl DistributionKind {
    /// For a kind built around another one, how it wraps it and the inner kind (the base of a truncated or censored
    /// kind, the component of a mixture, the emission of a hidden Markov model); None for any other kind.
    ///
    /// ```
    /// use keanu_graph::op::{DistributionKind, Nested};
    /// use std::sync::Arc;
    ///
    /// let kind = DistributionKind::Truncated(Arc::new(DistributionKind::Gaussian));
    /// let (nested, inner) = kind.nested().unwrap();
    /// assert_eq!(nested, Nested::Truncated);
    /// assert!(matches!(**inner, DistributionKind::Gaussian));
    /// assert!(DistributionKind::Gaussian.nested().is_none());
    /// ```
    pub fn nested(&self) -> Option<(Nested, &Arc<DistributionKind>)> {
        Some(match self {
            DistributionKind::Truncated(inner) => (Nested::Truncated, inner),
            DistributionKind::Censored(inner) => (Nested::Censored, inner),
            DistributionKind::Mixture(inner) => (Nested::Mixture, inner),
            DistributionKind::HiddenMarkovModel(inner) => (Nested::HiddenMarkovModel, inner),
            _ => return None,
        })
    }

    /// [`Self::nested`] for a kind matched by `nested_kinds!()`.
    pub(crate) fn nested_parts(&self) -> (Nested, &DistributionKind) {
        let (nested, inner) = self
            .nested()
            .expect("only called on a kind matched as nested");
        (nested, inner)
    }
}

impl Nested {
    /// The kind wrapping `inner` this way.
    pub fn wrap(self, inner: Arc<DistributionKind>) -> DistributionKind {
        match self {
            Nested::Truncated => DistributionKind::Truncated(inner),
            Nested::Censored => DistributionKind::Censored(inner),
            Nested::Mixture => DistributionKind::Mixture(inner),
            Nested::HiddenMarkovModel => DistributionKind::HiddenMarkovModel(inner),
        }
    }

    /// The names of the parameters this wrapper adds after the inner kind's: `lower` and `upper` for a truncated or
    /// censored kind, `weights` for a mixture, `initial` and `transition` for a hidden Markov model.
    pub fn own_parameter_names(self) -> &'static [&'static str] {
        match self {
            Nested::Truncated | Nested::Censored => &["lower", "upper"],
            Nested::Mixture => &["weights"],
            Nested::HiddenMarkovModel => &["initial", "transition"],
        }
    }

    /// The truncated or censored wrapper, None for a mixture or a hidden Markov model.
    pub(crate) fn wrapper(self) -> Option<Wrapper> {
        match self {
            Nested::Truncated => Some(Wrapper::Truncated),
            Nested::Censored => Some(Wrapper::Censored),
            Nested::Mixture | Nested::HiddenMarkovModel => None,
        }
    }

    /// The kind's name in errors.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Nested::Truncated => "truncated",
            Nested::Censored => "censored",
            Nested::Mixture => "mixture",
            Nested::HiddenMarkovModel => "hidden_markov_model",
        }
    }

    /// The inner kind's parameter types followed by this wrapper's own (all f64), or None if the inner kind has a
    /// variable number of parameters.
    pub(crate) fn parameter_types(self, inner: &DistributionKind) -> Option<Cow<'static, [DType]>> {
        let mut types = inner.parameter_types()?.into_owned();
        types.extend(std::iter::repeat_n(
            DType::F64,
            self.own_parameter_names().len(),
        ));
        Some(Cow::Owned(types))
    }

    /// The inner kind's parameter names followed by this wrapper's own.
    pub(crate) fn parameter_names(
        self,
        inner: &DistributionKind,
    ) -> Option<Cow<'static, [&'static str]>> {
        let mut names = inner.parameter_names()?.into_owned();
        names.extend_from_slice(self.own_parameter_names());
        Some(Cow::Owned(names))
    }

    /// Whether the kind has one value per element with an element-wise density (a truncated or censored kind), so
    /// that it can itself be a mixture component or an emission.
    pub(crate) fn is_elementwise(self) -> bool {
        self.wrapper().is_some()
    }

    /// Checks that `inner` can be wrapped this way, returning the reason it can't.
    pub(crate) fn check_inner(self, inner: &DistributionKind) -> std::result::Result<(), String> {
        match self {
            Nested::Truncated | Nested::Censored if !inner.has_cdf() => Err(format!(
                "the base distribution ({}) has no CDF, so it cannot be truncated or censored",
                inner.name()
            )),
            Nested::Mixture if !inner.is_elementwise() => Err(format!(
                "the component distribution ({}) does not have one value per element with an element-wise \
                 density, so it cannot be mixed",
                inner.name()
            )),
            Nested::HiddenMarkovModel if !inner.is_elementwise() => Err(format!(
                "the emission distribution ({}) does not have one value per element with an element-wise \
                 density, so it cannot be a hidden Markov model's emission",
                inner.name()
            )),
            _ => Ok(()),
        }
    }

    /// The support given the parameters' values (see [`DistributionKind::support`]).
    pub(crate) fn support(
        self,
        inner: &DistributionKind,
        parameters: &[&Value],
    ) -> Result<Support> {
        match self {
            Nested::Truncated | Nested::Censored => wrappers::support(inner, parameters),
            Nested::Mixture => mixture::support(inner, parameters),
            Nested::HiddenMarkovModel => hmm::support(inner, parameters),
        }
    }

    /// Draws a value of the given shape given the parameters' values.
    pub(crate) fn sample(
        self,
        inner: &DistributionKind,
        shape: &[usize],
        parameters: &[&Value],
        rng: &mut KeanuRng,
    ) -> Result<Value> {
        match self {
            Nested::Truncated => wrappers::sample_truncated(inner, shape, parameters, rng),
            Nested::Censored => wrappers::sample_censored(inner, shape, parameters, rng),
            Nested::Mixture => mixture::sample(inner, shape, parameters, rng),
            Nested::HiddenMarkovModel => hmm::sample(inner, shape, parameters, rng),
        }
    }

    /// The log probability of `value` (summed over its elements) and, with `gradient`, its gradient parts.
    pub(crate) fn log_prob(
        self,
        inner: &DistributionKind,
        value: &Value,
        parameters: &[&Value],
        gradient: bool,
    ) -> Result<(f64, Option<LogProbGradientParts>)> {
        match self {
            Nested::Truncated => {
                wrappers::log_prob(Wrapper::Truncated, inner, value, parameters, gradient)
            }
            Nested::Censored => {
                wrappers::log_prob(Wrapper::Censored, inner, value, parameters, gradient)
            }
            Nested::Mixture => mixture::log_prob(inner, value, parameters, gradient),
            Nested::HiddenMarkovModel => hmm::log_prob(inner, value, parameters, gradient),
        }
    }

    /// The log density of each element of `value` and, with `gradient`, its element-wise gradient parts, for an
    /// [element-wise](Self::is_elementwise) wrapper; None for the others.
    pub(crate) fn elementwise(
        self,
        inner: &DistributionKind,
        value: &Value,
        parameters: &[&Value],
        gradient: bool,
    ) -> Option<Result<(DoubleTensor, Option<LogProbGradientParts>)>> {
        let wrapper = self.wrapper()?;
        Some(wrappers::elementwise(
            wrapper, inner, value, parameters, gradient,
        ))
    }

    /// The Hessian-vector products of the log density of `kind`, a kind matched by `nested_kinds!()`, as
    /// [`DistributionKind::log_prob_hvp`].
    pub(crate) fn log_prob_hvp(
        kind: &DistributionKind,
        value: &Value,
        parameters: &[&Value],
        value_tangent: Option<&DoubleTensor>,
        parameter_tangents: &[Option<&DoubleTensor>],
        b: usize,
    ) -> Result<LogProbHvpParts> {
        let (nested, inner) = kind.nested_parts();
        match nested {
            // element-wise, so contracted per element (see `distribution_hvp::elementwise`)
            Nested::Truncated | Nested::Censored => distribution_hvp::elementwise_hvp(
                kind,
                value,
                parameters,
                value_tangent,
                parameter_tangents,
                b,
            ),
            Nested::Mixture => mixture::log_prob_hvp(
                inner,
                value,
                parameters,
                value_tangent,
                parameter_tangents,
                b,
            ),
            Nested::HiddenMarkovModel => hmm::log_prob_hvp(
                inner,
                value,
                parameters,
                value_tangent,
                parameter_tangents,
                b,
            ),
        }
    }
}
