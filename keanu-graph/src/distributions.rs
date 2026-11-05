//! Probabilistic nodes: one constructor per distribution on [`GraphBuilder`].

use crate::builder::GraphBuilder;
use crate::graph::NodeId;
use crate::op::{CustomDistribution, DistributionKind};
use crate::vars::{BoolVar, DoubleVar, IntVar, IntoDoubleVar, IntoIntVar, Var};
use crate::{GraphError, Value};
use keanu_distributions::Kde;
use keanu_tensor::DoubleTensor;
use std::sync::Arc;

macro_rules! continuous {
    ($($(#[$doc:meta])* $name:ident, $with_shape:ident => $kind:ident ($($p:ident),+);)*) => {
        impl GraphBuilder {
            $(
                $(#[$doc])*
                /// The value's shape is the broadcast of the parameters' shapes.
                pub fn $name<'b>(&'b self, $($p: impl IntoDoubleVar<'b>),+) -> DoubleVar<'b> {
                    let ids = [$($p.into_var(self).id()),+];
                    DoubleVar::new(self, self.distribution(DistributionKind::$kind, None, &ids))
                }

                $(#[$doc])*
                /// The value's shape is `shape` broadcast with the parameters' shapes.
                pub fn $with_shape<'b>(&'b self, shape: &[usize], $($p: impl IntoDoubleVar<'b>),+) -> DoubleVar<'b> {
                    let ids = [$($p.into_var(self).id()),+];
                    DoubleVar::new(self, self.distribution(DistributionKind::$kind, Some(shape), &ids))
                }
            )*
        }
    };
}

continuous! {
    /// Gaussian(mu, sigma).
    gaussian, gaussian_with_shape => Gaussian(mu, sigma);
    /// |Gaussian(0, sigma)|.
    half_gaussian, half_gaussian_with_shape => HalfGaussian(sigma);
    cauchy, cauchy_with_shape => Cauchy(location, scale);
    half_cauchy, half_cauchy_with_shape => HalfCauchy(scale);
    /// Beta(alpha, beta) on [0, 1].
    beta, beta_with_shape => Beta(alpha, beta);
    /// Exponential with the given rate (mean 1 / rate).
    exponential, exponential_with_shape => Exponential(rate);
    /// Gamma with scale theta and shape k.
    gamma, gamma_with_shape => Gamma(theta, k);
    inverse_gamma, inverse_gamma_with_shape => InverseGamma(alpha, beta);
    laplace, laplace_with_shape => Laplace(mu, beta);
    logistic, logistic_with_shape => Logistic(mu, s);
    log_normal, log_normal_with_shape => LogNormal(mu, sigma);
    pareto, pareto_with_shape => Pareto(location, scale);
    triangular, triangular_with_shape => Triangular(x_min, x_max, c);
    /// Uniform on [x_min, x_max).
    uniform, uniform_with_shape => Uniform(x_min, x_max);
}

impl GraphBuilder {
    /// A chi-squared node with k degrees of freedom.
    pub fn chi_squared<'b>(&'b self, k: impl IntoIntVar<'b>) -> DoubleVar<'b> {
        let k = k.into_var(self).id();
        DoubleVar::new(
            self,
            self.distribution(DistributionKind::ChiSquared, None, &[k]),
        )
    }

    /// Like [`Self::chi_squared`], with values of the given shape (broadcast with the parameters' shapes).
    pub fn chi_squared_with_shape<'b>(
        &'b self,
        shape: &[usize],
        k: impl IntoIntVar<'b>,
    ) -> DoubleVar<'b> {
        let k = k.into_var(self).id();
        DoubleVar::new(
            self,
            self.distribution(DistributionKind::ChiSquared, Some(shape), &[k]),
        )
    }

    /// A Student's t node with v degrees of freedom.
    pub fn student_t<'b>(&'b self, v: impl IntoIntVar<'b>) -> DoubleVar<'b> {
        let v = v.into_var(self).id();
        DoubleVar::new(
            self,
            self.distribution(DistributionKind::StudentT, None, &[v]),
        )
    }

    /// Like [`Self::student_t`], with values of the given shape (broadcast with the parameters' shapes).
    pub fn student_t_with_shape<'b>(
        &'b self,
        shape: &[usize],
        v: impl IntoIntVar<'b>,
    ) -> DoubleVar<'b> {
        let v = v.into_var(self).id();
        DoubleVar::new(
            self,
            self.distribution(DistributionKind::StudentT, Some(shape), &[v]),
        )
    }

    /// A Dirichlet distribution; the value has the concentration's shape.
    pub fn dirichlet<'b>(&'b self, concentration: impl IntoDoubleVar<'b>) -> DoubleVar<'b> {
        let c = concentration.into_var(self).id();
        DoubleVar::new(
            self,
            self.distribution(DistributionKind::Dirichlet, None, &[c]),
        )
    }

    /// A multivariate Gaussian; the value's last dimension is the event dimension.
    pub fn multivariate_gaussian<'b>(
        &'b self,
        mu: impl IntoDoubleVar<'b>,
        covariance: impl IntoDoubleVar<'b>,
    ) -> DoubleVar<'b> {
        let ids = [mu.into_var(self).id(), covariance.into_var(self).id()];
        DoubleVar::new(
            self,
            self.distribution(DistributionKind::MultivariateGaussian, None, &ids),
        )
    }

    /// An LKJ(eta) prior over the lower triangular Cholesky factor of a `dimension` × `dimension` correlation matrix
    /// (see [`keanu_distributions::LkjCholesky`]): the value has shape [dimension, dimension], and its support is
    /// [`Support::CorrelationCholesky`](crate::Support::CorrelationCholesky). `eta` = 1 is uniform over correlation
    /// matrices and larger `eta` favours weaker correlations. The correlation matrix is `L Lᵀ`.
    pub fn lkj_cholesky<'b>(
        &'b self,
        dimension: usize,
        eta: impl IntoDoubleVar<'b>,
    ) -> DoubleVar<'b> {
        self.lkj_cholesky_with_shape(&[dimension, dimension], eta)
    }

    /// Like [`Self::lkj_cholesky`], with a batch of factors: `shape` is [..., K, K] and `eta` broadcasts to the batch
    /// shape [...].
    pub fn lkj_cholesky_with_shape<'b>(
        &'b self,
        shape: &[usize],
        eta: impl IntoDoubleVar<'b>,
    ) -> DoubleVar<'b> {
        let eta = eta.into_var(self).id();
        DoubleVar::new(
            self,
            self.distribution(DistributionKind::LkjCholesky, Some(shape), &[eta]),
        )
    }

    /// A kernel density estimate from fixed samples, with values of the given shape.
    pub fn kde(&self, kde: Kde, shape: &[usize]) -> DoubleVar<'_> {
        DoubleVar::new(
            self,
            self.distribution(DistributionKind::Kde(Arc::new(kde)), Some(shape), &[]),
        )
    }

    /// A node with a user-defined distribution (see [`CustomDistribution`]), a value of the given shape and the
    /// given nodes as its parameters. The handle type must match the distribution's value type.
    pub fn custom<'b, V: Var<'b>>(
        &'b self,
        distribution: impl CustomDistribution + 'static,
        shape: &[usize],
        parameters: &[NodeId],
    ) -> V {
        let kind = DistributionKind::Custom(Arc::new(distribution));
        if kind.value_type() != V::DTYPE {
            self.record(crate::GraphError::InvalidNode {
                op: "custom",
                message: format!(
                    "the distribution's values are {:?} but a {:?} node was requested",
                    kind.value_type(),
                    V::DTYPE
                ),
            });
        }
        V::wrap(self, self.distribution(kind, Some(shape), parameters))
    }

    /// True with the given probability.
    pub fn bernoulli<'b>(&'b self, probability: impl IntoDoubleVar<'b>) -> BoolVar<'b> {
        let p = probability.into_var(self).id();
        BoolVar::new(
            self,
            self.distribution(DistributionKind::Bernoulli, None, &[p]),
        )
    }

    /// Like [`Self::bernoulli`], with values of the given shape (broadcast with the parameters' shapes).
    pub fn bernoulli_with_shape<'b>(
        &'b self,
        shape: &[usize],
        probability: impl IntoDoubleVar<'b>,
    ) -> BoolVar<'b> {
        let p = probability.into_var(self).id();
        BoolVar::new(
            self,
            self.distribution(DistributionKind::Bernoulli, Some(shape), &[p]),
        )
    }

    /// A Binomial(p, n) node: successes out of n trials.
    pub fn binomial<'b>(&'b self, p: impl IntoDoubleVar<'b>, n: impl IntoIntVar<'b>) -> IntVar<'b> {
        let ids = [p.into_var(self).id(), n.into_var(self).id()];
        IntVar::new(
            self,
            self.distribution(DistributionKind::Binomial, None, &ids),
        )
    }

    /// Like [`Self::binomial`], with values of the given shape (broadcast with the parameters' shapes).
    pub fn binomial_with_shape<'b>(
        &'b self,
        shape: &[usize],
        p: impl IntoDoubleVar<'b>,
        n: impl IntoIntVar<'b>,
    ) -> IntVar<'b> {
        let ids = [p.into_var(self).id(), n.into_var(self).id()];
        IntVar::new(
            self,
            self.distribution(DistributionKind::Binomial, Some(shape), &ids),
        )
    }

    /// A category index in 0..probabilities.len(), with the given (unnormalised) probabilities.
    pub fn categorical<'b>(
        &'b self,
        shape: &[usize],
        probabilities: &[DoubleVar<'b>],
    ) -> IntVar<'b> {
        let ids: Vec<_> = probabilities.iter().map(|p| p.id()).collect();
        IntVar::new(
            self,
            self.distribution(DistributionKind::Categorical, Some(shape), &ids),
        )
    }

    /// A geometric node with success probability p.
    pub fn geometric<'b>(&'b self, p: impl IntoDoubleVar<'b>) -> IntVar<'b> {
        let p = p.into_var(self).id();
        IntVar::new(
            self,
            self.distribution(DistributionKind::Geometric, None, &[p]),
        )
    }

    /// Like [`Self::geometric`], with values of the given shape (broadcast with the parameters' shapes).
    pub fn geometric_with_shape<'b>(
        &'b self,
        shape: &[usize],
        p: impl IntoDoubleVar<'b>,
    ) -> IntVar<'b> {
        let p = p.into_var(self).id();
        IntVar::new(
            self,
            self.distribution(DistributionKind::Geometric, Some(shape), &[p]),
        )
    }

    /// A multinomial node: counts of each category out of n draws.
    pub fn multinomial<'b>(
        &'b self,
        n: impl IntoIntVar<'b>,
        p: impl IntoDoubleVar<'b>,
    ) -> IntVar<'b> {
        let ids = [n.into_var(self).id(), p.into_var(self).id()];
        IntVar::new(
            self,
            self.distribution(DistributionKind::Multinomial, None, &ids),
        )
    }

    /// A Poisson node with mean mu.
    pub fn poisson<'b>(&'b self, mu: impl IntoDoubleVar<'b>) -> IntVar<'b> {
        let mu = mu.into_var(self).id();
        IntVar::new(
            self,
            self.distribution(DistributionKind::Poisson, None, &[mu]),
        )
    }

    /// Like [`Self::poisson`], with values of the given shape (broadcast with the parameters' shapes).
    pub fn poisson_with_shape<'b>(
        &'b self,
        shape: &[usize],
        mu: impl IntoDoubleVar<'b>,
    ) -> IntVar<'b> {
        let mu = mu.into_var(self).id();
        IntVar::new(
            self,
            self.distribution(DistributionKind::Poisson, Some(shape), &[mu]),
        )
    }

    /// Uniform on the integers in [x_min, x_max).
    pub fn uniform_int<'b>(
        &'b self,
        x_min: impl IntoIntVar<'b>,
        x_max: impl IntoIntVar<'b>,
    ) -> IntVar<'b> {
        let ids = [x_min.into_var(self).id(), x_max.into_var(self).id()];
        IntVar::new(
            self,
            self.distribution(DistributionKind::UniformInt, None, &ids),
        )
    }

    /// Like [`Self::uniform_int`], with values of the given shape (broadcast with the parameters' shapes).
    pub fn uniform_int_with_shape<'b>(
        &'b self,
        shape: &[usize],
        x_min: impl IntoIntVar<'b>,
        x_max: impl IntoIntVar<'b>,
    ) -> IntVar<'b> {
        let ids = [x_min.into_var(self).id(), x_max.into_var(self).id()];
        IntVar::new(
            self,
            self.distribution(DistributionKind::UniformInt, Some(shape), &ids),
        )
    }
}

impl GraphBuilder {
    /// A truncated distribution: `base` (a univariate distribution kind with a CDF: Gaussian, LogNormal,
    /// Exponential, Gamma, Beta, StudentT, Cauchy, Laplace, Logistic, Uniform, Poisson or Binomial) with
    /// `parameters` (its parameter nodes, in its order), restricted to `[lower, upper]` and renormalised by the
    /// probability the base gives that range. Either bound may be infinite; for a discrete base both must be whole
    /// numbers (or infinite) and the range includes them.
    ///
    /// The node's support is the base's intersected with the bounds, so gradient based samplers keep a latent
    /// truncated node inside them. Values are drawn by inverting the base's CDF where it has a closed-form inverse
    /// (Gaussian, LogNormal, Exponential, Cauchy, Laplace, Logistic, Uniform) and by rejection otherwise, which
    /// fails with an error if the bounds hold too little of the base's probability.
    ///
    /// The handle type must match the base's value type ([`DoubleVar`] or [`IntVar`]). Building fails if the base
    /// has no CDF, the parameters don't match it, or constant bounds have `lower >= upper`; bounds that are not
    /// constant are checked when the node is evaluated.
    ///
    /// ```
    /// use keanu_graph::{DoubleVar, GraphBuilder, op::DistributionKind};
    ///
    /// let b = GraphBuilder::new();
    /// let (mu, sigma) = (b.scalar(0.0), b.scalar(1.0));
    /// // a half-normal, as a Gaussian truncated below at 0
    /// let x: DoubleVar = b.truncated(DistributionKind::Gaussian, &[mu.id(), sigma.id()], 0.0, f64::INFINITY);
    /// # let _ = x;
    /// b.build()?;
    /// # Ok::<(), keanu_graph::GraphError>(())
    /// ```
    pub fn truncated<'b, V: Var<'b>>(
        &'b self,
        base: DistributionKind,
        parameters: &[NodeId],
        lower: impl IntoDoubleVar<'b>,
        upper: impl IntoDoubleVar<'b>,
    ) -> V {
        let kind = DistributionKind::Truncated(Arc::new(base));
        self.wrapper(kind, parameters, lower.into_var(self), upper.into_var(self))
    }

    /// A censored distribution: `base` (a univariate distribution kind with a CDF, as for [`Self::truncated`]) with
    /// `parameters`, observed through bounds. A value strictly between the bounds has the base's density; a value
    /// equal to `lower` stands for every value at or below it, with probability P(X <= lower), and a value equal
    /// to `upper` for every value at or above it, with probability P(X >= upper). Values beyond the bounds are
    /// impossible. Either bound may be infinite (no censoring on that side); for a discrete base both must be
    /// whole numbers (or infinite).
    ///
    /// Censoring is for observed data, such as measurements clipped by an instrument's range. Values are drawn
    /// from the base and recorded as the nearer bound when beyond it. Errors are as for [`Self::truncated`].
    pub fn censored<'b, V: Var<'b>>(
        &'b self,
        base: DistributionKind,
        parameters: &[NodeId],
        lower: impl IntoDoubleVar<'b>,
        upper: impl IntoDoubleVar<'b>,
    ) -> V {
        let kind = DistributionKind::Censored(Arc::new(base));
        self.wrapper(kind, parameters, lower.into_var(self), upper.into_var(self))
    }

    fn wrapper<'b, V: Var<'b>>(
        &'b self,
        kind: DistributionKind,
        parameters: &[NodeId],
        lower: DoubleVar<'b>,
        upper: DoubleVar<'b>,
    ) -> V {
        let op = if matches!(kind, DistributionKind::Truncated(_)) {
            "truncated"
        } else {
            "censored"
        };
        if kind.value_type() != V::DTYPE {
            self.record(GraphError::InvalidNode {
                op,
                message: format!(
                    "the base distribution's values are {:?} but a {:?} node was requested",
                    kind.value_type(),
                    V::DTYPE
                ),
            });
        }
        if let (Some(Value::F64(l)), Some(Value::F64(u))) = (
            self.constant_value(lower.id()),
            self.constant_value(upper.id()),
        ) {
            if let Ok((l, u)) = broadcast_pair(&l, &u) {
                // NaN bounds fail too
                if l.as_slice()
                    .iter()
                    .zip(u.as_slice())
                    .any(|(l, u)| l >= u || l.is_nan() || u.is_nan())
                {
                    self.record(GraphError::InvalidNode {
                        op,
                        message: "the lower bound must be less than the upper bound".into(),
                    });
                }
            }
        }
        let mut ids = parameters.to_vec();
        ids.extend([lower.id(), upper.id()]);
        V::wrap(self, self.distribution(kind, None, &ids))
    }
}

/// Two tensors broadcast to their common shape.
fn broadcast_pair(
    a: &DoubleTensor,
    b: &DoubleTensor,
) -> keanu_tensor::Result<(DoubleTensor, DoubleTensor)> {
    let shape = keanu_tensor::broadcast_shape(a.shape(), b.shape())?;
    Ok((a.broadcast_to(&shape)?, b.broadcast_to(&shape)?))
}
