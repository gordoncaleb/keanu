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
    /// Weibull with shape alpha and scale beta.
    weibull, weibull_with_shape => Weibull(alpha, beta);
    /// Von Mises with mean direction mu and concentration kappa: an angle, mapped to unconstrained space with a
    /// circular transform by the gradient based algorithms. Samples are in [-π, π).
    von_mises, von_mises_with_shape => VonMises(mu, kappa);
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

    /// A zero-sum normal prior over vectors of `n` elements that sum to zero (see
    /// [`keanu_distributions::ZeroSumNormal`]): N(0, σ² I) restricted to the zero-sum subspace, as PyMC's
    /// `ZeroSumNormal`. Each element's marginal standard deviation is σ √((N − 1) / N). Its support is
    /// [`Support::SumToZero`](crate::Support::SumToZero), which the gradient based algorithms map to N − 1
    /// unconstrained coordinates with Stan's isometric `sum_to_zero_vector` transform. A typical use is group effects
    /// next to an intercept, which the constraint identifies.
    pub fn zero_sum_normal<'b>(&'b self, n: usize, sigma: impl IntoDoubleVar<'b>) -> DoubleVar<'b> {
        self.zero_sum_normal_with_shape(&[n], sigma)
    }

    /// Like [`Self::zero_sum_normal`], with a batch of vectors: `shape` is [..., N], each vector of the last axis
    /// sums to zero, and `sigma` broadcasts to the batch shape [...].
    pub fn zero_sum_normal_with_shape<'b>(
        &'b self,
        shape: &[usize],
        sigma: impl IntoDoubleVar<'b>,
    ) -> DoubleVar<'b> {
        let sigma = sigma.into_var(self).id();
        DoubleVar::new(
            self,
            self.distribution(DistributionKind::ZeroSumNormal, Some(shape), &[sigma]),
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

    /// A negative binomial node in mean–dispersion form: mean mu and variance mu + mu²/alpha (a Poisson whose rate
    /// is Gamma distributed). It approaches Poisson(mu) as alpha grows.
    pub fn negative_binomial<'b>(
        &'b self,
        mu: impl IntoDoubleVar<'b>,
        alpha: impl IntoDoubleVar<'b>,
    ) -> IntVar<'b> {
        let ids = [mu.into_var(self).id(), alpha.into_var(self).id()];
        IntVar::new(
            self,
            self.distribution(DistributionKind::NegativeBinomial, None, &ids),
        )
    }

    /// Like [`Self::negative_binomial`], with values of the given shape (broadcast with the parameters' shapes).
    pub fn negative_binomial_with_shape<'b>(
        &'b self,
        shape: &[usize],
        mu: impl IntoDoubleVar<'b>,
        alpha: impl IntoDoubleVar<'b>,
    ) -> IntVar<'b> {
        let ids = [mu.into_var(self).id(), alpha.into_var(self).id()];
        IntVar::new(
            self,
            self.distribution(DistributionKind::NegativeBinomial, Some(shape), &ids),
        )
    }

    /// An ordered logistic node (Stan's `ordered_logistic`): a category index in 0..K from a location eta and K - 1
    /// strictly increasing cutpoints along the cutpoints' last axis. The value's shape is the broadcast of eta's
    /// shape and the cutpoints' leading axes. Latent cutpoints are best given an ordered prior (the `Ordered`
    /// support), so gradient based samplers keep them increasing.
    pub fn ordered_logistic<'b>(
        &'b self,
        eta: impl IntoDoubleVar<'b>,
        cutpoints: impl IntoDoubleVar<'b>,
    ) -> IntVar<'b> {
        let ids = [eta.into_var(self).id(), cutpoints.into_var(self).id()];
        IntVar::new(
            self,
            self.distribution(DistributionKind::OrderedLogistic, None, &ids),
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
    /// A beta-binomial node: successes out of `n` trials whose success probability is Beta(`alpha`, `beta`)
    /// distributed (see [`keanu_distributions::BetaBinomial`]), an over-dispersed binomial with mean
    /// nα/(α + β). Its log CDF and survival function are finite sums, so it can be truncated and censored.
    pub fn beta_binomial<'b>(
        &'b self,
        n: impl IntoIntVar<'b>,
        alpha: impl IntoDoubleVar<'b>,
        beta: impl IntoDoubleVar<'b>,
    ) -> IntVar<'b> {
        self.beta_binomial_node(None, n, alpha, beta)
    }

    /// Like [`Self::beta_binomial`], with values of the given shape (broadcast with the parameters' shapes).
    pub fn beta_binomial_with_shape<'b>(
        &'b self,
        shape: &[usize],
        n: impl IntoIntVar<'b>,
        alpha: impl IntoDoubleVar<'b>,
        beta: impl IntoDoubleVar<'b>,
    ) -> IntVar<'b> {
        self.beta_binomial_node(Some(shape), n, alpha, beta)
    }

    fn beta_binomial_node<'b>(
        &'b self,
        shape: Option<&[usize]>,
        n: impl IntoIntVar<'b>,
        alpha: impl IntoDoubleVar<'b>,
        beta: impl IntoDoubleVar<'b>,
    ) -> IntVar<'b> {
        let ids = [
            n.into_var(self).id(),
            alpha.into_var(self).id(),
            beta.into_var(self).id(),
        ];
        IntVar::new(
            self,
            self.distribution(DistributionKind::BetaBinomial, shape, &ids),
        )
    }

    /// True with probability σ(`logit`), the Bernoulli parameterised by its log odds (Stan's `bernoulli_logit`):
    /// its log probability is computed as −softplus(∓logit), accurate for any logit, so a linear predictor can be
    /// passed directly.
    pub fn bernoulli_logit<'b>(&'b self, logit: impl IntoDoubleVar<'b>) -> BoolVar<'b> {
        let logit = logit.into_var(self).id();
        BoolVar::new(
            self,
            self.distribution(DistributionKind::BernoulliLogit, None, &[logit]),
        )
    }

    /// Like [`Self::bernoulli_logit`], with values of the given shape (broadcast with the parameter's shape).
    pub fn bernoulli_logit_with_shape<'b>(
        &'b self,
        shape: &[usize],
        logit: impl IntoDoubleVar<'b>,
    ) -> BoolVar<'b> {
        let logit = logit.into_var(self).id();
        BoolVar::new(
            self,
            self.distribution(DistributionKind::BernoulliLogit, Some(shape), &[logit]),
        )
    }

    /// Successes out of `n` trials with success probability σ(`logit`), the binomial parameterised by its log odds
    /// (Stan's `binomial_logit`), accurate for any logit.
    pub fn binomial_logit<'b>(
        &'b self,
        logit: impl IntoDoubleVar<'b>,
        n: impl IntoIntVar<'b>,
    ) -> IntVar<'b> {
        let ids = [logit.into_var(self).id(), n.into_var(self).id()];
        IntVar::new(
            self,
            self.distribution(DistributionKind::BinomialLogit, None, &ids),
        )
    }

    /// Like [`Self::binomial_logit`], with values of the given shape (broadcast with the parameters' shapes).
    pub fn binomial_logit_with_shape<'b>(
        &'b self,
        shape: &[usize],
        logit: impl IntoDoubleVar<'b>,
        n: impl IntoIntVar<'b>,
    ) -> IntVar<'b> {
        let ids = [logit.into_var(self).id(), n.into_var(self).id()];
        IntVar::new(
            self,
            self.distribution(DistributionKind::BinomialLogit, Some(shape), &ids),
        )
    }

    /// A Poisson node with mean e^`log_mu` (Stan's `poisson_log`), for a log-linear predictor: its log probability
    /// kη − e^η − ln k! takes no logarithm of the mean.
    pub fn poisson_log<'b>(&'b self, log_mu: impl IntoDoubleVar<'b>) -> IntVar<'b> {
        let log_mu = log_mu.into_var(self).id();
        IntVar::new(
            self,
            self.distribution(DistributionKind::PoissonLog, None, &[log_mu]),
        )
    }

    /// Like [`Self::poisson_log`], with values of the given shape (broadcast with the parameter's shape).
    pub fn poisson_log_with_shape<'b>(
        &'b self,
        shape: &[usize],
        log_mu: impl IntoDoubleVar<'b>,
    ) -> IntVar<'b> {
        let log_mu = log_mu.into_var(self).id();
        IntVar::new(
            self,
            self.distribution(DistributionKind::PoissonLog, Some(shape), &[log_mu]),
        )
    }

    /// A category index in 0..logits.len() with probabilities softmax(`logits`): one node of unnormalised log
    /// probabilities per category, all of the same shape (as [`Self::categorical`] takes probabilities). The log
    /// probability is ηₓ − logsumexp(η), so no exponential overflows.
    pub fn categorical_logit<'b>(
        &'b self,
        shape: &[usize],
        logits: &[DoubleVar<'b>],
    ) -> IntVar<'b> {
        let ids: Vec<_> = logits.iter().map(|p| p.id()).collect();
        IntVar::new(
            self,
            self.distribution(DistributionKind::CategoricalLogit, Some(shape), &ids),
        )
    }
}

impl GraphBuilder {
    /// A zero-inflated count: 0 with probability `psi` (a structural zero), else a draw from `inner` with
    /// `parameters` (its parameter nodes, in its order). So P(0) = ψ + (1 − ψ) p(0) and P(k) = (1 − ψ) p(k) for
    /// k ≠ 0, computed in log space. The inner kind is an element-wise count distribution (a Poisson, negative
    /// binomial, binomial, beta-binomial, their logit and log forms, or a truncated count). The value's shape is
    /// the broadcast of the parameters' and ψ's shapes.
    ///
    /// A ψ outside [0, 1] makes every value impossible (log probability −∞), and sampling with it is an error.
    /// The gradient and Hessian-vector products with respect to ψ and the inner parameters are exact, and Gibbs
    /// steps enumerate a zero-inflated latent whose inner kind has finitely many values.
    ///
    /// ```
    /// use keanu_graph::{GraphBuilder, IntVar, op::DistributionKind};
    /// use keanu_tensor::IntTensor;
    ///
    /// let b = GraphBuilder::new();
    /// let rate = b.gamma(1.0, 2.0);
    /// let psi = b.beta(1.0, 1.0);
    /// let counts: IntVar = b.zero_inflated(DistributionKind::Poisson, &[rate.id()], psi);
    /// # let _ = counts;
    /// b.build()?;
    /// # Ok::<(), keanu_graph::GraphError>(())
    /// ```
    pub fn zero_inflated<'b, V: Var<'b>>(
        &'b self,
        inner: DistributionKind,
        parameters: &[NodeId],
        psi: impl IntoDoubleVar<'b>,
    ) -> V {
        let kind = DistributionKind::ZeroInflated(Arc::new(inner));
        self.inflated_node(None, kind, parameters, psi.into_var(self))
    }

    /// Like [`Self::zero_inflated`], with a value of the given shape (broadcast with the parameters' shapes).
    pub fn zero_inflated_with_shape<'b, V: Var<'b>>(
        &'b self,
        shape: &[usize],
        inner: DistributionKind,
        parameters: &[NodeId],
        psi: impl IntoDoubleVar<'b>,
    ) -> V {
        let kind = DistributionKind::ZeroInflated(Arc::new(inner));
        self.inflated_node(Some(shape), kind, parameters, psi.into_var(self))
    }

    /// A hurdle model: 0 with probability `psi`, else a draw from `inner` with `parameters` restricted to positive
    /// values. For a count inner kind (as for [`Self::zero_inflated`]) P(0) = ψ and P(k) = (1 − ψ) p(k) / (1 − p(0))
    /// for k > 0, the inner kind truncated at zero by its own mass there. For a positive continuous inner kind (a
    /// half Gaussian, half Cauchy, chi-squared, exponential, gamma, inverse gamma, log-normal, Weibull or beta) a value of
    /// exactly 0 has probability ψ and a positive value the density (1 − ψ) p(x), for data such as rainfall with dry
    /// days; such a node is meant to be observed. Negative values are impossible. The handle type must match the
    /// inner kind's value type ([`IntVar`] or [`DoubleVar`]).
    ///
    /// Positive counts are drawn by rejection from the inner kind, which fails with an error if it puts too little
    /// probability above zero.
    pub fn hurdle<'b, V: Var<'b>>(
        &'b self,
        inner: DistributionKind,
        parameters: &[NodeId],
        psi: impl IntoDoubleVar<'b>,
    ) -> V {
        let kind = DistributionKind::Hurdle(Arc::new(inner));
        self.inflated_node(None, kind, parameters, psi.into_var(self))
    }

    /// Like [`Self::hurdle`], with a value of the given shape (broadcast with the parameters' shapes).
    pub fn hurdle_with_shape<'b, V: Var<'b>>(
        &'b self,
        shape: &[usize],
        inner: DistributionKind,
        parameters: &[NodeId],
        psi: impl IntoDoubleVar<'b>,
    ) -> V {
        let kind = DistributionKind::Hurdle(Arc::new(inner));
        self.inflated_node(Some(shape), kind, parameters, psi.into_var(self))
    }

    fn inflated_node<'b, V: Var<'b>>(
        &'b self,
        shape: Option<&[usize]>,
        kind: DistributionKind,
        parameters: &[NodeId],
        psi: DoubleVar<'b>,
    ) -> V {
        if kind.value_type() != V::DTYPE {
            self.record(GraphError::InvalidNode {
                op: kind.name(),
                message: format!(
                    "the inner distribution's values are {:?} but a {:?} node was requested",
                    kind.value_type(),
                    V::DTYPE
                ),
            });
        }
        let mut ids = parameters.to_vec();
        ids.push(psi.id());
        V::wrap(self, self.distribution(kind, shape, &ids))
    }
}

impl GraphBuilder {
    /// A truncated distribution: `base` (a univariate distribution kind with a CDF: Gaussian, LogNormal,
    /// Exponential, Gamma, Beta, StudentT, Cauchy, Laplace, Logistic, Uniform, Weibull, Poisson, Binomial,
    /// BetaBinomial or NegativeBinomial) with
    /// `parameters` (its parameter nodes, in its order), restricted to `[lower, upper]` and renormalised by the
    /// probability the base gives that range. Either bound may be infinite; for a discrete base both must be whole
    /// numbers (or infinite) and the range includes them.
    ///
    /// The node's support is the base's intersected with the bounds, so gradient based samplers keep a latent
    /// truncated node inside them. Values are drawn by inverting the base's CDF where it has a closed-form inverse
    /// (Gaussian, LogNormal, Exponential, Cauchy, Laplace, Logistic, Uniform, Weibull) and by rejection otherwise,
    /// which fails with an error if the bounds hold too little of the base's probability.
    ///
    /// The handle type must match the base's value type ([`DoubleVar`] or [`IntVar`]). Building fails if the base
    /// has no CDF, the parameters don't match it, or constant bounds have `lower >= upper`. Bounds that are not
    /// constant and cross (`lower >= upper`, or NaN) make the node's log probability −∞, as a uniform's bounds do,
    /// so samplers reject such a state; drawing a value between them, or taking the node's support, is an error.
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

impl GraphBuilder {
    /// A mixture of C distributions of the `component` kind: density Σ_c w_c p_c(x), with the weights `weights`
    /// (non-negative, summing to 1 along their last axis, the component axis, of length C) and `parameters` the
    /// component's parameter nodes, in its order, each with the same trailing component axis (of length C, or 1, or
    /// no axes at all, to share one value between the components).
    ///
    /// The component is an element-wise univariate kind: one of the continuous or count distributions with one
    /// value per element (not Dirichlet, MultivariateGaussian, LkjCholesky, ZeroSumNormal, Kde, Categorical,
    /// Multinomial, OrderedLogistic, Bernoulli, BernoulliLogit, CategoricalLogit, a custom distribution or another
    /// mixture), or a truncated, censored, zero-inflated or hurdle one. The
    /// value has the batch shape: the parameters' shapes without their component axes, broadcast together; use
    /// [`Self::mixture_with_shape`] for a larger one (a vector of observations from one mixture). The log density is
    /// computed as logsumexp_c(log w_c + log p_c(x)), so components far in the tails stay accurate, and its gradient
    /// with respect to the value, the component parameters and the weights is exact.
    ///
    /// The support is the component's, widened to hold every component's (the smallest lower bound and largest
    /// upper bound across components). A mixture can't itself be truncated or censored; truncate the components
    /// instead. The handle type must match the component's value type ([`DoubleVar`] or [`IntVar`]).
    ///
    /// ```
    /// use keanu_graph::{DoubleVar, GraphBuilder, op::DistributionKind};
    /// use keanu_tensor::DoubleTensor;
    ///
    /// let b = GraphBuilder::new();
    /// let means = b.constant(DoubleTensor::vector(vec![-2.0, 3.0]));
    /// let sigma = b.scalar(1.0);
    /// let weights = b.dirichlet(DoubleTensor::vector(vec![1.0, 1.0]));
    /// // 100 observations from a two-component Gaussian mixture
    /// let y: DoubleVar =
    ///     b.mixture_with_shape(&[100], DistributionKind::Gaussian, &[means.id(), sigma.id()], weights);
    /// # let _ = y;
    /// b.build()?;
    /// # Ok::<(), keanu_graph::GraphError>(())
    /// ```
    pub fn mixture<'b, V: Var<'b>>(
        &'b self,
        component: DistributionKind,
        parameters: &[NodeId],
        weights: impl IntoDoubleVar<'b>,
    ) -> V {
        self.mixture_node(None, component, parameters, weights.into_var(self))
    }

    /// Like [`Self::mixture`], with a value of the given shape (broadcast with the parameters' batch shapes).
    pub fn mixture_with_shape<'b, V: Var<'b>>(
        &'b self,
        shape: &[usize],
        component: DistributionKind,
        parameters: &[NodeId],
        weights: impl IntoDoubleVar<'b>,
    ) -> V {
        self.mixture_node(Some(shape), component, parameters, weights.into_var(self))
    }

    fn mixture_node<'b, V: Var<'b>>(
        &'b self,
        shape: Option<&[usize]>,
        component: DistributionKind,
        parameters: &[NodeId],
        weights: DoubleVar<'b>,
    ) -> V {
        let kind = DistributionKind::Mixture(Arc::new(component));
        if kind.value_type() != V::DTYPE {
            self.record(GraphError::InvalidNode {
                op: "mixture",
                message: format!(
                    "the component distribution's values are {:?} but a {:?} node was requested",
                    kind.value_type(),
                    V::DTYPE
                ),
            });
        }
        let mut ids = parameters.to_vec();
        ids.push(weights.id());
        V::wrap(self, self.distribution(kind, shape, &ids))
    }
}

impl GraphBuilder {
    /// A hidden Markov model over `steps` time steps: a hidden chain of K states, the first drawn from the
    /// `initial` probabilities (`[K]`) and each later one from the row of the `transition` probabilities ([K, K]) for
    /// the state before it, and at each step an observation from the `emission` kind with the current state's
    /// parameters. `parameters` are the emission's parameter nodes, in its order, each with a trailing state axis
    /// (of length K, or 1, or no axes at all, to share one value between the states); a parameter may also have a
    /// leading time axis (of length `steps`) for parameters that vary over time.
    ///
    /// The initial and transition parameters are probabilities, not log probabilities: non-negative and summing to
    /// 1 along their last axis (within 1e-5), so a Dirichlet prior (one per transition row) puts them on the
    /// simplex for the gradient based algorithms. Zeros are allowed and rule out the paths through them.
    ///
    /// The emission is an element-wise univariate kind, as for [`Self::mixture`]. The value is the observation
    /// series, of shape `[steps]`; use [`Self::hidden_markov_model_with_shape`] for several observations per step.
    /// The log density is the exact marginal likelihood, summed over every state path by the log-space forward
    /// algorithm in O(T K²), and its gradient with respect to the emission parameters, the probabilities and (for
    /// a continuous emission) the observations is exact (the forward-backward algorithm). Drawing a value
    /// simulates the chain and then each step's observation. The support is the emission's, widened to hold
    /// every state's. The handle type must match the emission's value type ([`DoubleVar`] or [`IntVar`]).
    ///
    /// ```
    /// use keanu_graph::{IntVar, GraphBuilder, op::DistributionKind};
    /// use keanu_tensor::{DoubleTensor, IntTensor};
    ///
    /// let b = GraphBuilder::new();
    /// // two regimes with Poisson counts of mean 2 and 9
    /// let rates = b.constant(DoubleTensor::vector(vec![2.0, 9.0]));
    /// let initial = b.constant(DoubleTensor::vector(vec![0.5, 0.5]));
    /// let transition = b.constant(DoubleTensor::new(&[2, 2], vec![0.95, 0.05, 0.1, 0.9])?);
    /// let counts: IntVar =
    ///     b.hidden_markov_model(6, DistributionKind::Poisson, &[rates.id()], initial, transition);
    /// b.observe(counts, IntTensor::vector(vec![1, 3, 2, 10, 8, 11]));
    /// b.build()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn hidden_markov_model<'b, V: Var<'b>>(
        &'b self,
        steps: usize,
        emission: DistributionKind,
        parameters: &[NodeId],
        initial: impl IntoDoubleVar<'b>,
        transition: impl IntoDoubleVar<'b>,
    ) -> V {
        self.hidden_markov_model_with_shape(&[steps], emission, parameters, initial, transition)
    }

    /// Like [`Self::hidden_markov_model`], with observations of the given shape [T, ...]: time first, then the
    /// shape of each step's observation (all of whose elements are emitted by that step's state). Each emission
    /// parameter's shape without its state axis must broadcast to `shape`.
    pub fn hidden_markov_model_with_shape<'b, V: Var<'b>>(
        &'b self,
        shape: &[usize],
        emission: DistributionKind,
        parameters: &[NodeId],
        initial: impl IntoDoubleVar<'b>,
        transition: impl IntoDoubleVar<'b>,
    ) -> V {
        let kind = DistributionKind::HiddenMarkovModel(Arc::new(emission));
        if kind.value_type() != V::DTYPE {
            self.record(GraphError::InvalidNode {
                op: "hidden_markov_model",
                message: format!(
                    "the emission distribution's values are {:?} but a {:?} node was requested",
                    kind.value_type(),
                    V::DTYPE
                ),
            });
        }
        let mut ids = parameters.to_vec();
        ids.extend([initial.into_var(self).id(), transition.into_var(self).id()]);
        V::wrap(self, self.distribution(kind, Some(shape), &ids))
    }

    /// A linear-Gaussian state-space model over `steps` time steps (the model of the Kalman filter): a hidden
    /// state x_t of dimension d with x_0 ~ N(`initial_mean`, `initial_covariance`) and
    /// x_{t+1} = `transition_matrix` x_t + N(0, `transition_covariance`), observed as
    /// y_t = `observation_matrix` x_t + N(0, `observation_covariance`) in dimension m. The value is the
    /// observations, of shape [steps, m].
    ///
    /// Shapes: `initial_mean` `[d]`, `initial_covariance`, `transition_matrix` and `transition_covariance` [d, d],
    /// `observation_matrix` [m, d], `observation_covariance` [m, m]; the covariances must be symmetric positive
    /// semidefinite (with H P Hᵀ + R positive definite at each step, else the observations are impossible). The
    /// log density is the Kalman filter's exact marginal likelihood, computed with Cholesky solves and the
    /// Joseph-form covariance update, and its gradient and Hessian-vector products with respect to the observations
    /// and every parameter are exact (so it works with NUTS, optimizers and the Laplace approximation); see [`keanu_distributions::LinearGaussianStateSpace`]. Shape mismatches are build errors.
    ///
    /// ```
    /// use keanu_graph::GraphBuilder;
    /// use keanu_tensor::DoubleTensor;
    ///
    /// let b = GraphBuilder::new();
    /// // a local level model: a random walk observed with noise
    /// let level_noise = b.half_gaussian(1.0);
    /// let q = (level_noise * level_noise).reshape(&[1, 1]);
    /// let one = || b.constant(DoubleTensor::new(&[1, 1], vec![1.0]).unwrap());
    /// let y = b.linear_gaussian_state_space(
    ///     4,
    ///     b.constant(DoubleTensor::vector(vec![0.0])),
    ///     b.constant(DoubleTensor::new(&[1, 1], vec![10.0])?),
    ///     one(),
    ///     q,
    ///     one(),
    ///     b.constant(DoubleTensor::new(&[1, 1], vec![0.25])?),
    /// );
    /// b.observe(y, DoubleTensor::new(&[4, 1], vec![0.3, 0.9, 1.4, 1.1])?);
    /// b.build()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn linear_gaussian_state_space<'b>(
        &'b self,
        steps: usize,
        initial_mean: impl IntoDoubleVar<'b>,
        initial_covariance: impl IntoDoubleVar<'b>,
        transition_matrix: impl IntoDoubleVar<'b>,
        transition_covariance: impl IntoDoubleVar<'b>,
        observation_matrix: impl IntoDoubleVar<'b>,
        observation_covariance: impl IntoDoubleVar<'b>,
    ) -> DoubleVar<'b> {
        let ids = [
            initial_mean.into_var(self).id(),
            initial_covariance.into_var(self).id(),
            transition_matrix.into_var(self).id(),
            transition_covariance.into_var(self).id(),
            observation_matrix.into_var(self).id(),
            observation_covariance.into_var(self).id(),
        ];
        DoubleVar::new(
            self,
            self.distribution(
                DistributionKind::LinearGaussianStateSpace,
                Some(&[steps]),
                &ids,
            ),
        )
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
