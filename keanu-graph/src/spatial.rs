//! Areal spatial random effects: the ICAR, the proper CAR and the BYM2 model over an [`Adjacency`].
//!
//! ```
//! use keanu_distributions::Adjacency;
//! use keanu_graph::GraphBuilder;
//!
//! // a 3 × 3 lattice of areas
//! let adjacency = Adjacency::lattice(3, 3)?;
//! let b = GraphBuilder::new();
//! let sigma = b.half_gaussian(1.0);
//! let rho = b.beta(0.5, 0.5);
//! let bym2 = b.bym2(adjacency, sigma, rho)?;
//! // the log relative risk of each area: an intercept plus the spatial effect
//! let log_risk = bym2.effect + b.gaussian(0.0, 2.0);
//! assert_eq!(log_risk.shape(), vec![9]);
//! # Ok::<(), keanu_graph::GraphError>(())
//! ```

use crate::builder::GraphBuilder;
use crate::error::Result;
use crate::op::DistributionKind;
use crate::vars::{DoubleVar, IntoDoubleVar};
use keanu_distributions::Adjacency;
use keanu_tensor::DoubleTensor;
use std::sync::Arc;

/// The nodes of a BYM2 spatial random effect (see [`GraphBuilder::bym2`]).
#[derive(Debug, Clone, Copy)]
pub struct Bym2<'b> {
    /// The structured (spatial) component φ, a unit-scale ICAR summing to zero on each component.
    pub spatial: DoubleVar<'b>,
    /// The unstructured component θ, independent N(0, 1) per area.
    pub unstructured: DoubleVar<'b>,
    /// The combined effect b = σ(√(ρ / s) φ + √(1 − ρ) θ), one per area.
    pub effect: DoubleVar<'b>,
}

impl GraphBuilder {
    /// An intrinsic CAR (ICAR) random effect over the areas of `adjacency`, with scale `sigma` (a scalar): one
    /// value per area, shape `[n]`, summing to zero on each connected component of two or more areas, with log
    /// density −½ Σ_{i~j} (φᵢ − φⱼ)² / σ² − (n − c) ln σ (see [`keanu_distributions::Icar`], c the number of such
    /// components). An island gets an independent N(0, σ²). Its support is
    /// [`Support::SumToZeroComponents`](crate::Support::SumToZeroComponents), which the gradient based algorithms
    /// map to n − c unconstrained coordinates with the sum-to-zero transform on each component.
    pub fn icar<'b>(
        &'b self,
        adjacency: impl Into<Arc<Adjacency>>,
        sigma: impl IntoDoubleVar<'b>,
    ) -> DoubleVar<'b> {
        let adjacency = adjacency.into();
        let shape = [adjacency.len()];
        let sigma = sigma.into_var(self).id();
        DoubleVar::new(
            self,
            self.distribution(DistributionKind::Icar(adjacency), Some(&shape), &[sigma]),
        )
    }

    /// A proper CAR random effect over the areas of `adjacency`: a zero-mean multivariate normal with precision
    /// τ(D − αW), shape `[n]` (see [`keanu_distributions::Car`]). `tau` and `alpha` are scalars; the precision is
    /// positive definite for τ > 0 and α between 1/λ_min and 1 (0 ≤ α < 1 is the usual prior range). Every area
    /// must have a neighbour. Its log determinant comes from eigenvalues computed once here, so evaluating the
    /// density costs O(n + edges).
    ///
    /// # Errors
    ///
    /// [`GraphError::Distribution`](crate::GraphError::Distribution) if an area has no neighbours.
    pub fn car<'b>(
        &'b self,
        adjacency: impl Into<Arc<Adjacency>>,
        tau: impl IntoDoubleVar<'b>,
        alpha: impl IntoDoubleVar<'b>,
    ) -> Result<DoubleVar<'b>> {
        let adjacency = adjacency.into();
        adjacency.car_eigenvalues()?;
        let shape = [adjacency.len()];
        let (tau, alpha) = (tau.into_var(self).id(), alpha.into_var(self).id());
        Ok(DoubleVar::new(
            self,
            self.distribution(
                DistributionKind::Car(adjacency),
                Some(&shape),
                &[tau, alpha],
            ),
        ))
    }

    /// A BYM2 spatial random effect over the areas of `adjacency` (Riebler et al. 2016), with total scale `sigma`
    /// and spatial proportion `rho` in (0, 1), both scalars:
    ///
    /// `b = σ (√(ρ / s) φ + √(1 − ρ) θ)`,
    ///
    /// where φ is a unit-scale [ICAR](Self::icar) (the structured part), θ independent N(0, 1) per area (the
    /// unstructured part) and s each area's [scaling factor](Adjacency::scaling_factors): the geometric mean of the
    /// ICAR's marginal variances over its component (1 for an island, which then gets two independent normals).
    /// Scaling makes Var(√(1/s) φ) about 1 whatever the neighbourhood, so σ is the effect's typical standard
    /// deviation and ρ the share of its variance that is spatial, which makes priors on them portable. The
    /// scaling factors are computed once here, densely (O(m³) for a component of m areas).
    ///
    /// # Errors
    ///
    /// Only if the scaling factors' eigendecomposition fails, which it doesn't for a valid adjacency.
    pub fn bym2<'b>(
        &'b self,
        adjacency: impl Into<Arc<Adjacency>>,
        sigma: impl IntoDoubleVar<'b>,
        rho: impl IntoDoubleVar<'b>,
    ) -> Result<Bym2<'b>> {
        let adjacency = adjacency.into();
        let n = adjacency.len();
        let scale = DoubleTensor::vector(
            adjacency
                .scaling_factors()?
                .iter()
                .map(|s| 1.0 / s.sqrt())
                .collect(),
        );
        let (sigma, rho) = (sigma.into_var(self), rho.into_var(self));
        let spatial = self.icar(adjacency, 1.0);
        let unstructured = self.gaussian_with_shape(&[n], 0.0, 1.0);
        let structured = spatial * self.constant(scale) * rho.sqrt();
        let independent = unstructured * (1.0 - rho).sqrt();
        let effect = (structured + independent) * sigma;
        Ok(Bym2 {
            spatial,
            unstructured,
            effect,
        })
    }
}
