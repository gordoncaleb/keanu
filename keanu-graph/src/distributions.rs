//! Probabilistic nodes: one constructor per distribution on [`GraphBuilder`].

use crate::builder::GraphBuilder;
use crate::graph::NodeId;
use crate::op::{CustomDistribution, DistributionKind, ElementBijector, Simulator};
use crate::vars::{BoolVar, DoubleVar, IntVar, IntoDoubleVar, IntoIntVar, Var};
use crate::{GraphError, Value};
use keanu_distributions::{Kde, KeanuRng};
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
    /// Skew normal with location xi, scale omega and shape alpha (Stan's `skew_normal`); see
    /// [`keanu_distributions::SkewNormal`].
    skew_normal, skew_normal_with_shape => SkewNormal(xi, omega, alpha);
    /// Exponentially modified Gaussian: N(mu, sigma) plus an exponential with rate lambda (Stan's
    /// `exp_mod_normal`); see [`keanu_distributions::ExGaussian`].
    ex_gaussian, ex_gaussian_with_shape => ExGaussian(mu, sigma, lambda);
    /// Gumbel with location mu and scale beta; see [`keanu_distributions::Gumbel`].
    gumbel, gumbel_with_shape => Gumbel(mu, beta);
    /// Generalized extreme value with location mu, scale sigma and shape xi (ξ > 0 heavy-tailed); its support
    /// depends on the parameters. See [`keanu_distributions::GeneralizedExtremeValue`].
    generalized_extreme_value, generalized_extreme_value_with_shape => GeneralizedExtremeValue(mu, sigma, xi);
    /// Generalized Pareto with location (threshold) mu, scale sigma and shape xi; its support depends on the
    /// parameters. See [`keanu_distributions::GeneralizedPareto`].
    generalized_pareto, generalized_pareto_with_shape => GeneralizedPareto(mu, sigma, xi);
    /// Inverse Gaussian (Wald) with mean mu and shape lambda; see [`keanu_distributions::InverseGaussian`].
    inverse_gaussian, inverse_gaussian_with_shape => InverseGaussian(mu, lambda);
    /// Kumaraswamy with shapes a and b, on [0, 1]; see [`keanu_distributions::Kumaraswamy`].
    kumaraswamy, kumaraswamy_with_shape => Kumaraswamy(a, b);
    /// Logit-normal: the sigmoid of N(mu, sigma), on [0, 1]; see [`keanu_distributions::LogitNormal`].
    logit_normal, logit_normal_with_shape => LogitNormal(mu, sigma);
    /// Half Student-t with nu (real) degrees of freedom and scale sigma; see
    /// [`keanu_distributions::HalfStudentT`].
    half_student_t, half_student_t_with_shape => HalfStudentT(nu, sigma);
    /// Student-t with real positive degrees of freedom, location and positive scale.
    location_scale_student_t, location_scale_student_t_with_shape => LocationScaleStudentT(nu, mu, sigma);
    /// Asymmetric Laplace with location mu (the tau-quantile), scale sigma and skewness tau in (0, 1) (Stan's
    /// `skew_double_exponential`); see [`keanu_distributions::AsymmetricLaplace`].
    asymmetric_laplace, asymmetric_laplace_with_shape => AsymmetricLaplace(mu, sigma, tau);
    /// Log-logistic with scale alpha and shape beta (Stan's `loglogistic`); see
    /// [`keanu_distributions::LogLogistic`].
    log_logistic, log_logistic_with_shape => LogLogistic(alpha, beta);
    /// Rice with noncentrality nu and scale sigma; see [`keanu_distributions::Rice`].
    rice, rice_with_shape => Rice(nu, sigma);
    /// Generalized gamma with scale theta, shape k and power p (Gamma(theta, k) at p = 1); see
    /// [`keanu_distributions::GeneralizedGamma`].
    generalized_gamma, generalized_gamma_with_shape => GeneralizedGamma(theta, k, p);
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

    /// A simulator node (see [`Simulator`]): a probabilistic node whose value, of the given shape, is drawn by
    /// `function` from its parameters' values (the given nodes, in order) and a random number generator, like PyMC's
    /// `pm.Simulator`. The handle type sets the value type (`DoubleVar`, `IntVar` or `BoolVar`).
    ///
    /// The node has no likelihood. It can be sampled from its prior (forward sampling, prior and posterior
    /// predictive draws) and observed; inferring its parameters from an observation needs approximate Bayesian
    /// computation (`keanu_algorithms::abc`), and every other inference algorithm rejects a model with a simulator
    /// node. The function must be thread safe, as parallel chains call it at once, and return a value of the
    /// node's type and shape (anything else is [`GraphError::Simulator`] when it is sampled). A graph with a
    /// simulator node can't be saved.
    ///
    /// ```
    /// use keanu_distributions::KeanuRng;
    /// use keanu_graph::{DoubleVar, GraphBuilder, Value};
    /// use keanu_tensor::DoubleTensor;
    ///
    /// // 10 draws from N(mu, 1), simulated by hand
    /// let b = GraphBuilder::new();
    /// let mu = b.gaussian(0.0, 5.0);
    /// let data: DoubleVar = b.simulator(&[mu.id()], &[10], |parameters: &[&Value], rng: &mut KeanuRng| {
    ///     let mu = parameters[0].as_f64().ok_or("mu is f64")?.as_slice()[0];
    ///     let draws = (0..10).map(|_| mu + rng.next_gaussian()).collect();
    ///     Ok(Value::F64(DoubleTensor::vector(draws)))
    /// });
    /// let data = data.id();
    /// let (graph, mut state) = b.build()?;
    /// state.lazy_eval(&graph, &[data], &mut KeanuRng::seed_from_u64(1))?;
    /// assert_eq!(state.f64(data).unwrap().shape(), &[10]);
    /// # Ok::<(), keanu_graph::GraphError>(())
    /// ```
    pub fn simulator<'b, V: Var<'b>>(
        &'b self,
        parameters: &[NodeId],
        shape: &[usize],
        function: impl Fn(&[&Value], &mut KeanuRng) -> std::result::Result<Value, String>
        + Send
        + Sync
        + 'static,
    ) -> V {
        let kind = DistributionKind::Simulator(Arc::new(Simulator::new(V::DTYPE, function)));
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

    /// Negative binomial with mean exp(log_mu) and positive dispersion alpha.
    pub fn negative_binomial_log<'b>(
        &'b self,
        log_mu: impl IntoDoubleVar<'b>,
        alpha: impl IntoDoubleVar<'b>,
    ) -> IntVar<'b> {
        let ids = [log_mu.into_var(self).id(), alpha.into_var(self).id()];
        IntVar::new(
            self,
            self.distribution(DistributionKind::NegativeBinomialLog, None, &ids),
        )
    }

    /// Like [`Self::negative_binomial_log`], with the requested shape broadcast with the parameters.
    pub fn negative_binomial_log_with_shape<'b>(
        &'b self,
        shape: &[usize],
        log_mu: impl IntoDoubleVar<'b>,
        alpha: impl IntoDoubleVar<'b>,
    ) -> IntVar<'b> {
        let ids = [log_mu.into_var(self).id(), alpha.into_var(self).id()];
        IntVar::new(
            self,
            self.distribution(DistributionKind::NegativeBinomialLog, Some(shape), &ids),
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
    /// Positive counts are drawn by inverting the inner kind's mass above zero where it puts at least half its mass
    /// at zero (so a small rate samples quickly and exactly), and by rejection of its non-positive draws elsewhere.
    /// Sampling is an error only if the inner kind puts no probability above zero while ψ < 1.
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
    /// Exponential, Gamma, Beta, StudentT, Cauchy, Laplace, Logistic, Uniform, Weibull, SkewNormal, ExGaussian,
    /// Gumbel, GeneralizedExtremeValue, GeneralizedPareto, InverseGaussian, Kumaraswamy, LogitNormal, HalfStudentT,
    /// AsymmetricLaplace, LogLogistic, GeneralizedGamma, Poisson, Binomial, PoissonBinomial, BetaBinomial or
    /// NegativeBinomial) with
    /// `parameters` (its parameter nodes, in its order), restricted to `[lower, upper]` and renormalised by the
    /// probability the base gives that range. Either bound may be infinite; for a discrete base both must be whole
    /// numbers (or infinite) and the range includes them.
    ///
    /// The node's support is the base's intersected with the bounds, so gradient based samplers keep a latent
    /// truncated node inside them. Values are drawn by inverting the base's CDF where it has a closed-form inverse
    /// (Gaussian, LogNormal, Exponential, Cauchy, Laplace, Logistic, Uniform, Weibull, Gumbel,
    /// GeneralizedExtremeValue, GeneralizedPareto, Kumaraswamy, LogitNormal, AsymmetricLaplace, LogLogistic).
    /// For a Poisson-binomial base, sampling uses the conditional count masses if the first draw is outside the
    /// bounds. Other bases use rejection, which fails with an error if the bounds hold too little probability.
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
    /// A mixture of components of different kinds: density Σ_c w_c p_c(x), with `components` the C pairs of a
    /// component kind and its parameter nodes (in its order, with no component axis) and `weights` non-negative,
    /// summing to 1 along their last axis, of length C. The node's parents are the weights, then each component's
    /// parameters in turn ([`DistributionKind::MixtureOf`]).
    ///
    /// Each component is an element-wise kind, as for [`Self::mixture`] (truncated, censored, zero-inflated,
    /// hurdle and [transformed](Self::transformed) ones included), and all have the same value type, which the
    /// handle type must match ([`DoubleVar`] or [`IntVar`]). The value has the batch shape: the weights' shape
    /// without the component axis and every component parameter's shape, broadcast; use
    /// [`Self::mixture_of_with_shape`] for a larger one. The log density is logsumexp_c(log w_c + log p_c(x)), with
    /// an exact gradient and exact Hessian-vector products in the value, the weights and every component parameter.
    /// Weights off the simplex make the log probability −∞ (and sampling an error), as for [`Self::mixture`].
    ///
    /// The support is the hull of the components' supports (the smallest lower bound and the largest upper bound;
    /// each component's must be a range of real numbers), or discrete for integer components. Where the components'
    /// supports don't overlap the density is 0 between them although the support includes them. A heterogeneous
    /// mixture has an element-wise [`log_density`](Self::log_density) and pointwise log likelihood, but
    /// no CDF (it can't be truncated or censored; censor its components instead) and no finite support for Gibbs
    /// steps.
    ///
    /// ```
    /// use keanu_graph::{DoubleVar, GraphBuilder, op::DistributionKind};
    /// use keanu_tensor::DoubleTensor;
    ///
    /// let b = GraphBuilder::new();
    /// let (theta, k) = (b.scalar(2.0), b.scalar(3.0));
    /// let (mu, sigma) = (b.scalar(0.5), b.scalar(0.4));
    /// let weights = b.constant(DoubleTensor::vector(vec![0.3, 0.7]));
    /// // 50 observations from a mixture of a gamma and a log-normal
    /// let y: DoubleVar = b.mixture_of_with_shape(
    ///     &[50],
    ///     &[
    ///         (DistributionKind::Gamma, &[theta.id(), k.id()]),
    ///         (DistributionKind::LogNormal, &[mu.id(), sigma.id()]),
    ///     ],
    ///     weights,
    /// );
    /// # let _ = y;
    /// b.build()?;
    /// # Ok::<(), keanu_graph::GraphError>(())
    /// ```
    pub fn mixture_of<'b, V: Var<'b>>(
        &'b self,
        components: &[(DistributionKind, &[NodeId])],
        weights: impl IntoDoubleVar<'b>,
    ) -> V {
        self.mixture_of_node(None, components, weights.into_var(self))
    }

    /// Like [`Self::mixture_of`], with a value of the given shape (broadcast with the batch shape).
    pub fn mixture_of_with_shape<'b, V: Var<'b>>(
        &'b self,
        shape: &[usize],
        components: &[(DistributionKind, &[NodeId])],
        weights: impl IntoDoubleVar<'b>,
    ) -> V {
        self.mixture_of_node(Some(shape), components, weights.into_var(self))
    }

    fn mixture_of_node<'b, V: Var<'b>>(
        &'b self,
        shape: Option<&[usize]>,
        components: &[(DistributionKind, &[NodeId])],
        weights: DoubleVar<'b>,
    ) -> V {
        let kinds: Arc<[DistributionKind]> = components.iter().map(|(k, _)| k.clone()).collect();
        let kind = DistributionKind::MixtureOf(kinds);
        if !components.is_empty() && kind.value_type() != V::DTYPE {
            self.record(GraphError::InvalidNode {
                op: "mixture_of",
                message: format!(
                    "the components' values are {:?} but a {:?} node was requested",
                    kind.value_type(),
                    V::DTYPE
                ),
            });
        }
        let mut ids = vec![weights.id()];
        for (_, parameters) in components {
            ids.extend_from_slice(parameters);
        }
        V::wrap(self, self.distribution(kind, shape, &ids))
    }

    /// A transformed distribution: the value is f(y) for a draw y of `base` with `parameters` (its parameter nodes,
    /// in its order), f the forward map of `bijector`, so its density is p(f⁻¹(x)) |(f⁻¹)′(x)|
    /// ([`DistributionKind::Transformed`]). The base is a continuous element-wise kind with a density: one of the
    /// continuous distributions with one value per element, or a truncated or transformed one (not a censored or
    /// hurdle kind, whose point masses have no Jacobian). The bijector's bounds or location and scale are constants
    /// (see [`ElementBijector`](crate::op::ElementBijector)).
    ///
    /// The value's shape is the broadcast of the parameters' shapes; use [`Self::transformed_with_shape`] for a
    /// larger one. The node has an exact gradient and exact Hessian-vector products, from the base's element-wise
    /// rules and the chain rule through the inverse map, and is element-wise itself (a mixture component, a
    /// [`log_density`](Self::log_density), pointwise log likelihoods). Its support is the image of the
    /// base's support (which must be a range of real numbers), so gradient based samplers keep a latent
    /// transformed node inside it. It can be truncated or censored when its base has a CDF.
    ///
    /// ```
    /// use keanu_graph::{DoubleVar, GraphBuilder, op::{DistributionKind, ElementBijector}};
    ///
    /// let b = GraphBuilder::new();
    /// let (mu, sigma) = (b.scalar(0.0), b.scalar(1.0));
    /// // exp of a Gaussian, 2 + exp(y): a log-normal shifted to start at 2
    /// let x: DoubleVar = b.transformed(
    ///     DistributionKind::Gaussian,
    ///     &[mu.id(), sigma.id()],
    ///     ElementBijector::LowerBound(2.0),
    /// );
    /// # let _ = x;
    /// b.build()?;
    /// # Ok::<(), keanu_graph::GraphError>(())
    /// ```
    pub fn transformed<'b>(
        &'b self,
        base: DistributionKind,
        parameters: &[NodeId],
        bijector: ElementBijector,
    ) -> DoubleVar<'b> {
        let kind = DistributionKind::Transformed(Arc::new(base), bijector);
        DoubleVar::new(self, self.distribution(kind, None, parameters))
    }

    /// Like [`Self::transformed`], with a value of the given shape (broadcast with the parameters' shapes).
    pub fn transformed_with_shape<'b>(
        &'b self,
        shape: &[usize],
        base: DistributionKind,
        parameters: &[NodeId],
        bijector: ElementBijector,
    ) -> DoubleVar<'b> {
        let kind = DistributionKind::Transformed(Arc::new(base), bijector);
        DoubleVar::new(self, self.distribution(kind, Some(shape), parameters))
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

impl GraphBuilder {
    /// A Gaussian random walk of `steps` values (see [`keanu_distributions::GaussianRandomWalk`], as PyMC's
    /// `GaussianRandomWalk`): the first value is N(`init_mu`, `init_sigma`) and each later one the one before plus
    /// N(`mu`, `sigma`). PyMC's `steps` counts the steps after the initial value, so it is `steps − 1` here. Every
    /// parameter must be a scalar here; see [`Self::gaussian_random_walk_with_shape`] for a batch of series.
    ///
    /// ```
    /// use keanu_graph::GraphBuilder;
    ///
    /// let b = GraphBuilder::new();
    /// let drift = b.gaussian(0.0, 1.0);
    /// let scale = b.half_gaussian(1.0);
    /// let walk = b.gaussian_random_walk(50, drift, scale, 0.0, 10.0);
    /// assert_eq!(walk.shape(), vec![50]);
    /// b.build()?;
    /// # Ok::<(), keanu_graph::GraphError>(())
    /// ```
    pub fn gaussian_random_walk<'b>(
        &'b self,
        steps: usize,
        mu: impl IntoDoubleVar<'b>,
        sigma: impl IntoDoubleVar<'b>,
        init_mu: impl IntoDoubleVar<'b>,
        init_sigma: impl IntoDoubleVar<'b>,
    ) -> DoubleVar<'b> {
        self.gaussian_random_walk_with_shape(&[steps], mu, sigma, init_mu, init_sigma)
    }

    /// Like [`Self::gaussian_random_walk`], with a batch of independent series: `shape` is [..., T], time last, and
    /// each parameter broadcasts to the batch shape [...] (one value per series, or shared).
    pub fn gaussian_random_walk_with_shape<'b>(
        &'b self,
        shape: &[usize],
        mu: impl IntoDoubleVar<'b>,
        sigma: impl IntoDoubleVar<'b>,
        init_mu: impl IntoDoubleVar<'b>,
        init_sigma: impl IntoDoubleVar<'b>,
    ) -> DoubleVar<'b> {
        let ids = [
            mu.into_var(self).id(),
            sigma.into_var(self).id(),
            init_mu.into_var(self).id(),
            init_sigma.into_var(self).id(),
        ];
        DoubleVar::new(
            self,
            self.distribution(DistributionKind::GaussianRandomWalk, Some(shape), &ids),
        )
    }

    /// A multivariate local-level state-space model. Both transition and observation use the identity
    /// matrix. The level starts at the specified Gaussian and evolves with zero drift and process
    /// covariance `level_covariance`. Observations add noise with covariance `observation_covariance`.
    /// The observation node has shape `[steps, k]`. Its log density integrates out the levels with the
    /// Kalman filter, as in [`Self::linear_gaussian_state_space`].
    pub fn multivariate_local_level<'b>(
        &'b self,
        steps: usize,
        initial_mean: impl IntoDoubleVar<'b>,
        initial_covariance: impl IntoDoubleVar<'b>,
        level_covariance: impl IntoDoubleVar<'b>,
        observation_covariance: impl IntoDoubleVar<'b>,
    ) -> DoubleVar<'b> {
        let initial_mean = initial_mean.into_var(self);
        let shape = initial_mean.shape();
        let [k] = shape.as_slice() else {
            self.record(GraphError::InvalidValue(
                "a multivariate local level needs a vector initial mean".into(),
            ));
            return initial_mean;
        };
        let identity = match DoubleTensor::eye(*k) {
            Ok(matrix) => self.constant(matrix),
            Err(error) => {
                self.record(error.into());
                return initial_mean;
            }
        };
        self.linear_gaussian_state_space(
            steps,
            initial_mean,
            initial_covariance,
            identity,
            level_covariance,
            identity,
            observation_covariance,
        )
    }

    /// A multivariate Gaussian random walk of shape `[steps, k]`. The first row is the supplied
    /// `initial` vector. Each later row adds `drift` (scalar or vector `[k]`) and an independent
    /// innovation with covariance `L Lᵀ`, where L is the lower triangle of `innovation_cholesky`.
    /// The walk is deterministic given the initial vector and its standard-normal innovation node.
    /// Pass a multivariate Gaussian initial node to give the initial state a prior.
    pub fn multivariate_gaussian_random_walk<'b>(
        &'b self,
        steps: usize,
        drift: impl IntoDoubleVar<'b>,
        innovation_cholesky: impl IntoDoubleVar<'b>,
        initial: impl IntoDoubleVar<'b>,
    ) -> DoubleVar<'b> {
        let (drift, l, initial) = (
            drift.into_var(self),
            innovation_cholesky.into_var(self),
            initial.into_var(self),
        );
        let shape = initial.shape();
        let [k] = shape.as_slice() else {
            self.record(GraphError::InvalidValue(
                "a multivariate walk needs a vector initial state".into(),
            ));
            return initial;
        };
        if steps == 0
            || *k == 0
            || l.shape() != [*k, *k]
            || !(drift.shape().is_empty() || drift.shape() == [*k])
        {
            self.record(GraphError::InvalidValue(format!(
                "a multivariate walk needs positive steps and dimension, a [{k}, {k}] innovation factor, and scalar or [{k}] drift"
            )));
            return initial;
        }
        let first = initial.reshape(&[1, *k as isize]);
        if steps == 1 {
            return first;
        }
        let z = self.gaussian_with_shape(&[steps - 1, *k], 0.0, 1.0);
        let increments = z.matmul_transposed(l.tril(0), false, true) + drift;
        self.concat(0, &[first, first + increments.cumsum(0)])
    }

    /// An autoregressive process AR(p) of `steps` values (see [`keanu_distributions::Autoregressive`], as PyMC's
    /// `AR`): the first p values are N(`init_mu`, `init_sigma`) and each later one
    /// N(`constant` + Σᵢ ρᵢ x_{t−i}, `sigma`), with `rho` the vector `[p]` of the coefficients of lags 1 to p. The
    /// process is conditioned on its first p values rather than started from its stationary distribution; for a
    /// stationary AR(1) pass `init_mu` = c / (1 − ρ) and `init_sigma` = σ / √(1 − ρ²) as nodes of ρ and σ.
    ///
    /// ```
    /// use keanu_graph::GraphBuilder;
    /// use keanu_tensor::DoubleTensor;
    ///
    /// let b = GraphBuilder::new();
    /// let rho = b.uniform_with_shape(&[2], -1.0, 1.0);
    /// let sigma = b.half_gaussian(1.0);
    /// let x = b.autoregressive(100, 0.0, rho, sigma, 0.0, 1.0);
    /// b.observe(x, DoubleTensor::zeros(&[100]));
    /// b.build()?;
    /// # Ok::<(), keanu_graph::GraphError>(())
    /// ```
    pub fn autoregressive<'b>(
        &'b self,
        steps: usize,
        constant: impl IntoDoubleVar<'b>,
        rho: impl IntoDoubleVar<'b>,
        sigma: impl IntoDoubleVar<'b>,
        init_mu: impl IntoDoubleVar<'b>,
        init_sigma: impl IntoDoubleVar<'b>,
    ) -> DoubleVar<'b> {
        self.autoregressive_with_shape(&[steps], constant, rho, sigma, init_mu, init_sigma)
    }

    /// Like [`Self::autoregressive`], with a batch of independent series: `shape` is [..., T], time last; `rho` is
    /// [..., p] and every other parameter broadcasts to the batch shape [...].
    pub fn autoregressive_with_shape<'b>(
        &'b self,
        shape: &[usize],
        constant: impl IntoDoubleVar<'b>,
        rho: impl IntoDoubleVar<'b>,
        sigma: impl IntoDoubleVar<'b>,
        init_mu: impl IntoDoubleVar<'b>,
        init_sigma: impl IntoDoubleVar<'b>,
    ) -> DoubleVar<'b> {
        let ids = [
            constant.into_var(self).id(),
            rho.into_var(self).id(),
            sigma.into_var(self).id(),
            init_mu.into_var(self).id(),
            init_sigma.into_var(self).id(),
        ];
        DoubleVar::new(
            self,
            self.distribution(DistributionKind::Autoregressive, Some(shape), &ids),
        )
    }

    /// A GARCH(1,1) series of `steps` values (see [`keanu_distributions::Garch11`], as PyMC's `GARCH11`):
    /// r_t ~ N(0, σ_t) with σ_0 = `initial_vol` and σ_t² = `omega` + `alpha` r_{t−1}² + `beta` σ_{t−1}². Parameters
    /// outside ω > 0, α ≥ 0, β ≥ 0, α + β < 1 and `initial_vol` > 0 make the series impossible (log probability −∞)
    /// and are an error to sample, so give α and β a prior on that region, such as the first two coordinates of a
    /// Dirichlet.
    ///
    /// ```
    /// use keanu_graph::GraphBuilder;
    /// use keanu_tensor::DoubleTensor;
    ///
    /// let b = GraphBuilder::new();
    /// // (alpha, beta, 1 − alpha − beta) on the simplex keeps the process stationary
    /// let weights = b.dirichlet(DoubleTensor::vector(vec![2.0, 6.0, 2.0]));
    /// let omega = b.half_gaussian(1.0);
    /// let returns = b.garch11(250, omega, weights.slice("0"), weights.slice("1"), 1.0);
    /// b.observe(returns, DoubleTensor::zeros(&[250]));
    /// b.build()?;
    /// # Ok::<(), keanu_graph::GraphError>(())
    /// ```
    pub fn garch11<'b>(
        &'b self,
        steps: usize,
        omega: impl IntoDoubleVar<'b>,
        alpha: impl IntoDoubleVar<'b>,
        beta: impl IntoDoubleVar<'b>,
        initial_vol: impl IntoDoubleVar<'b>,
    ) -> DoubleVar<'b> {
        self.garch11_with_shape(&[steps], omega, alpha, beta, initial_vol)
    }

    /// Like [`Self::garch11`], with a batch of independent series: `shape` is [..., T], time last, and each
    /// parameter broadcasts to the batch shape [...].
    pub fn garch11_with_shape<'b>(
        &'b self,
        shape: &[usize],
        omega: impl IntoDoubleVar<'b>,
        alpha: impl IntoDoubleVar<'b>,
        beta: impl IntoDoubleVar<'b>,
        initial_vol: impl IntoDoubleVar<'b>,
    ) -> DoubleVar<'b> {
        let ids = [
            omega.into_var(self).id(),
            alpha.into_var(self).id(),
            beta.into_var(self).id(),
            initial_vol.into_var(self).id(),
        ];
        DoubleVar::new(
            self,
            self.distribution(DistributionKind::Garch11, Some(shape), &ids),
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

/// Builder methods `$name(params...)` and `$with_shape(shape, params...)` for a kind, each parameter converted with
/// its `Into...Var` trait, returning the given handle type.
macro_rules! builders {
    ($($(#[$doc:meta])* $name:ident, $with_shape:ident => $kind:ident ($($p:ident: $into:ident),+) -> $var:ident;)*) => {
        impl GraphBuilder {
            $(
                $(#[$doc])*
                pub fn $name<'b>(&'b self, $($p: impl $into<'b>),+) -> $var<'b> {
                    let ids = [$($p.into_var(self).id()),+];
                    $var::new(self, self.distribution(DistributionKind::$kind, None, &ids))
                }

                #[doc = concat!(
                    "Like [`Self::", stringify!($name), "`], with a value of the given shape: its last axes are ",
                    "the event (or, for a distribution of one value per element, it is broadcast with the ",
                    "parameters' shapes) and its leading axes a batch that the parameters broadcast to."
                )]
                #[allow(clippy::too_many_arguments)]
                pub fn $with_shape<'b>(&'b self, shape: &[usize], $($p: impl $into<'b>),+) -> $var<'b> {
                    let ids = [$($p.into_var(self).id()),+];
                    $var::new(self, self.distribution(DistributionKind::$kind, Some(shape), &ids))
                }
            )*
        }
    };
}

builders! {
    /// Jones–Faddy skew Student-t with positive shapes a and b, location, and positive scale.
    skew_student_t, skew_student_t_with_shape => SkewStudentT(a: IntoDoubleVar, b: IntoDoubleVar, mu: IntoDoubleVar, sigma: IntoDoubleVar) -> DoubleVar;
    /// Shape-one Pólya–Gamma PG(1, z), with finite tilt z and positive support.
    polya_gamma1, polya_gamma1_with_shape => PolyaGamma1(z: IntoDoubleVar) -> DoubleVar;
    /// A success count from independent Bernoulli trials, with the trial probabilities on the last axis of p.
    poisson_binomial, poisson_binomial_with_shape => PoissonBinomial(p: IntoDoubleVar) -> IntVar;
    /// A discrete Weibull count with survival P(X ≥ k) = q^(k^β) for k ≥ 0.
    discrete_weibull, discrete_weibull_with_shape => DiscreteWeibull(q: IntoDoubleVar, beta: IntoDoubleVar) -> IntVar;
    /// Counts over the last axis of the logits, with the specified number of trials per event.
    /// The log mass is evaluated on the logit scale. See [`keanu_distributions::MultinomialLogit`].
    multinomial_logit, multinomial_logit_with_shape => MultinomialLogit(n: IntoIntVar, logits: IntoDoubleVar) -> IntVar;
    /// A multivariate Gaussian given its precision matrix. Only the lower triangle of the matrix is read.
    /// See [`keanu_distributions::MultivariateGaussianPrecision`].
    multivariate_gaussian_precision, multivariate_gaussian_precision_with_shape => MultivariateGaussianPrecision(mu: IntoDoubleVar, precision: IntoDoubleVar) -> DoubleVar;
    /// An ordered probit node (Stan's `ordered_probit`): a category index in 0..K from a location eta and K − 1
    /// strictly increasing cutpoints along the cutpoints' last axis, P(k) = Φ(η − c_k) − Φ(η − c_(k+1)). Shapes and
    /// conventions are those of [`Self::ordered_logistic`]; see [`keanu_distributions::OrderedProbit`].
    ordered_probit, ordered_probit_with_shape => OrderedProbit(eta: IntoDoubleVar, cutpoints: IntoDoubleVar) -> IntVar;
    /// A hypergeometric node (PyMC's `HyperGeometric(N, k, n)`): the number of successes among `draws` draws without
    /// replacement from `population` items of which `successes` are successes. See
    /// [`keanu_distributions::HyperGeometric`].
    hyper_geometric, hyper_geometric_with_shape =>
        HyperGeometric(population: IntoIntVar, successes: IntoIntVar, draws: IntoIntVar) -> IntVar;
    /// A Wiener first-passage time node, the drift-diffusion model of two-choice response times (Stan's `wiener`,
    /// with `wiener_full`'s drift variability): the response time of a trial ending at `boundary` (0 the lower,
    /// 1 the upper; usually observed data), with boundary separation `a`, non-decision time `t0`, relative starting
    /// point `w`, drift `v` and across-trial drift standard deviation `sv` (0.0 for the four-parameter model). Its
    /// support is greater than `t0`. Sampled given the boundary; draw choices and times together with
    /// [`keanu_distributions::sample_wiener`]. See [`keanu_distributions::Wiener`].
    wiener, wiener_with_shape => Wiener(
        a: IntoDoubleVar, t0: IntoDoubleVar, w: IntoDoubleVar, v: IntoDoubleVar, sv: IntoDoubleVar,
        boundary: IntoIntVar
    ) -> DoubleVar;
    /// A Skellam node: the difference of independent Poisson(mu1) and Poisson(mu2) counts. See
    /// [`keanu_distributions::Skellam`].
    skellam, skellam_with_shape => Skellam(mu1: IntoDoubleVar, mu2: IntoDoubleVar) -> IntVar;
    /// A Dirichlet-multinomial node (PyMC's `DirichletMultinomial(n, a)`): counts over the K categories of `alpha`'s
    /// last axis summing to `n`, a multinomial whose probabilities are Dirichlet(alpha). See
    /// [`keanu_distributions::DirichletMultinomial`].
    dirichlet_multinomial, dirichlet_multinomial_with_shape =>
        DirichletMultinomial(n: IntoIntVar, alpha: IntoDoubleVar) -> IntVar;
    /// A Plackett–Luce node: a ranking of the K items of `worths`' last axis, the zero-based indices of the items from
    /// first to last, each chosen in proportion to its (positive) worth among those left. See
    /// [`keanu_distributions::PlackettLuce`].
    plackett_luce, plackett_luce_with_shape => PlackettLuce(worths: IntoDoubleVar) -> IntVar;
    /// A multivariate Student-t node (Stan's `multi_student_t`) with `nu` degrees of freedom, location `mu` [..., K]
    /// and scale matrix `scale` [..., K, K]. See [`keanu_distributions::MultivariateStudentT`].
    multivariate_student_t, multivariate_student_t_with_shape =>
        MultivariateStudentT(nu: IntoDoubleVar, mu: IntoDoubleVar, scale: IntoDoubleVar) -> DoubleVar;
    /// A multivariate Student-t node given the lower triangular Cholesky factor of its scale matrix (Stan's
    /// `multi_student_t_cholesky`). See [`keanu_distributions::MultivariateStudentTCholesky`].
    multivariate_student_t_cholesky, multivariate_student_t_cholesky_with_shape =>
        MultivariateStudentTCholesky(nu: IntoDoubleVar, mu: IntoDoubleVar, scale_cholesky: IntoDoubleVar) -> DoubleVar;
    /// A Wishart node (Stan's `wishart`): a symmetric positive definite K × K matrix with `nu` (> K − 1) degrees of
    /// freedom and scale matrix `scale`, mean ν·scale. Its support is
    /// [`Support::CovarianceMatrix`](crate::Support::CovarianceMatrix), which the gradient based algorithms map to
    /// K(K + 1)/2 unconstrained coordinates. See [`keanu_distributions::Wishart`].
    wishart, wishart_with_shape => Wishart(nu: IntoDoubleVar, scale: IntoDoubleVar) -> DoubleVar;
    /// A Wishart node over the Cholesky factor of the matrix, given the scale's Cholesky factor (Stan's
    /// `wishart_cholesky`). Its support is [`Support::CovarianceCholesky`](crate::Support::CovarianceCholesky). See
    /// [`keanu_distributions::WishartCholesky`].
    wishart_cholesky, wishart_cholesky_with_shape =>
        WishartCholesky(nu: IntoDoubleVar, scale_cholesky: IntoDoubleVar) -> DoubleVar;
    /// An inverse Wishart node (Stan's `inv_wishart`), a conjugate prior for a covariance matrix: mean
    /// scale/(ν − K − 1). Its support is [`Support::CovarianceMatrix`](crate::Support::CovarianceMatrix). See
    /// [`keanu_distributions::InverseWishart`].
    inverse_wishart, inverse_wishart_with_shape => InverseWishart(nu: IntoDoubleVar, scale: IntoDoubleVar) -> DoubleVar;
    /// An inverse Wishart node over the Cholesky factor of the matrix, given the scale's Cholesky factor (Stan's
    /// `inv_wishart_cholesky`). Its support is [`Support::CovarianceCholesky`](crate::Support::CovarianceCholesky).
    /// See [`keanu_distributions::InverseWishartCholesky`].
    inverse_wishart_cholesky, inverse_wishart_cholesky_with_shape =>
        InverseWishartCholesky(nu: IntoDoubleVar, scale_cholesky: IntoDoubleVar) -> DoubleVar;
    /// A matrix normal node (PyMC's `MatrixNormal`): an n × p matrix with mean `mu`, row covariance
    /// `row_covariance` (n × n) and column covariance `column_covariance` (p × p). See
    /// [`keanu_distributions::MatrixNormal`].
    matrix_normal, matrix_normal_with_shape =>
        MatrixNormal(mu: IntoDoubleVar, row_covariance: IntoDoubleVar, column_covariance: IntoDoubleVar) -> DoubleVar;
    /// A von Mises–Fisher node: a unit vector in p >= 2 dimensions with mean direction `mu` (a unit vector) and
    /// concentration `kappa`. Its support is [`Support::UnitVector`](crate::Support::UnitVector), which the gradient
    /// based algorithms map with Stan's `unit_vector` transform. See [`keanu_distributions::VonMisesFisher`].
    von_mises_fisher, von_mises_fisher_with_shape => VonMisesFisher(mu: IntoDoubleVar, kappa: IntoDoubleVar) -> DoubleVar;
}
