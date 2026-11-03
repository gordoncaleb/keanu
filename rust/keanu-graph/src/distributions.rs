//! Probabilistic nodes: one constructor per distribution on [`GraphBuilder`].

use crate::builder::GraphBuilder;
use crate::op::DistributionKind;
use crate::vars::{BoolVar, DoubleVar, IntVar, IntoDoubleVar, IntoIntVar, Var};
use keanu_distributions::Kde;
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
    pub fn chi_squared<'b>(&'b self, k: impl IntoIntVar<'b>) -> DoubleVar<'b> {
        let k = k.into_var(self).id();
        DoubleVar::new(
            self,
            self.distribution(DistributionKind::ChiSquared, None, &[k]),
        )
    }

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

    pub fn student_t<'b>(&'b self, v: impl IntoIntVar<'b>) -> DoubleVar<'b> {
        let v = v.into_var(self).id();
        DoubleVar::new(
            self,
            self.distribution(DistributionKind::StudentT, None, &[v]),
        )
    }

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

    /// A kernel density estimate from fixed samples, with values of the given shape.
    pub fn kde(&self, kde: Kde, shape: &[usize]) -> DoubleVar<'_> {
        DoubleVar::new(
            self,
            self.distribution(DistributionKind::Kde(Arc::new(kde)), Some(shape), &[]),
        )
    }

    /// True with the given probability.
    pub fn bernoulli<'b>(&'b self, probability: impl IntoDoubleVar<'b>) -> BoolVar<'b> {
        let p = probability.into_var(self).id();
        BoolVar::new(
            self,
            self.distribution(DistributionKind::Bernoulli, None, &[p]),
        )
    }

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

    pub fn binomial<'b>(&'b self, p: impl IntoDoubleVar<'b>, n: impl IntoIntVar<'b>) -> IntVar<'b> {
        let ids = [p.into_var(self).id(), n.into_var(self).id()];
        IntVar::new(
            self,
            self.distribution(DistributionKind::Binomial, None, &ids),
        )
    }

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

    pub fn geometric<'b>(&'b self, p: impl IntoDoubleVar<'b>) -> IntVar<'b> {
        let p = p.into_var(self).id();
        IntVar::new(
            self,
            self.distribution(DistributionKind::Geometric, None, &[p]),
        )
    }

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

    pub fn poisson<'b>(&'b self, mu: impl IntoDoubleVar<'b>) -> IntVar<'b> {
        let mu = mu.into_var(self).id();
        IntVar::new(
            self,
            self.distribution(DistributionKind::Poisson, None, &[mu]),
        )
    }

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
