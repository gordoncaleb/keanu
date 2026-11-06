# Spatial models

Areal data has one value per area: cases of a disease per district, crop yield per field, votes per constituency.
Neighbouring areas tend to be alike, because they share causes nobody measured. Conditional autoregressive (CAR)
priors model this: each area's effect is centred on the average of its neighbours'. Keanu has the three standard
ones, the intrinsic CAR (ICAR), the proper CAR and the BYM2 model, over a neighbourhood given as an `Adjacency`.

## Neighbourhoods and sparse products

An `Adjacency` is an undirected graph of n areas. Build it from an edge list (either orientation, repeats merged),
from a symmetric 0/1 matrix (`Adjacency::from_matrix`), or as the rook neighbourhood of a lattice
(`Adjacency::lattice(rows, cols)`, numbered row by row). It knows each area's number of neighbours (the diagonal of
D), its connected components, the adjacency matrix W and the graph Laplacian Q = D − W, both as `SparseMatrix`.

A `SparseMatrix` is a constant matrix stored in compressed sparse row form. `x.sparse_matmul(a)` multiplies a node,
a vector `[n]` or a matrix `[n, k]`, by it at a cost proportional to the stored entries, with exact first and
second derivatives. Here it sums each area's neighbours' values:

```rust,ignore
{{#include ../../keanu/tests/book.rs:adjacency}}
```

## The ICAR

`icar(adjacency, sigma)` is the intrinsic CAR in the pairwise-difference form of Morris et al. (2019):

ln p(φ) = −½ Σ_{i~j} (φᵢ − φⱼ)² / σ² − (n − c) ln σ,

summed over pairs of neighbours, with c the number of connected components of two or more areas. In matrix form the
sum is φᵀQφ / σ². The density only depends on differences between neighbours, so adding a constant to every area
of a component leaves it unchanged: the ICAR is improper. It is identified by a hard constraint, the values of each
component summing to zero. Its support, `Support::SumToZeroComponents`, says so, and NUTS and the optimisers move in
n − c unconstrained coordinates with the [sum-to-zero transform](building-models.md#sum-to-zero-group-effects) on
each component, so every draw satisfies the constraint exactly. An island (an area with no neighbours) has nothing
to be smoothed towards: it gets an independent N(0, σ²), the usual convention, and is not constrained. The
normalising constant is left out, as in Stan's ICAR, which only matters when comparing models by their
marginal likelihood.

Restricted to its support, the ICAR is a Gaussian with covariance σ² Q⁺ (Q⁺ the generalised inverse of the
Laplacian). Sampling it from the prior draws that Gaussian exactly, from each component's eigendecomposition
(O(m³) for a component of m areas, computed once).

## The proper CAR

`car(adjacency, tau, alpha)` is a zero-mean multivariate normal with precision τ(D − αW). Each area's conditional
mean is α times the average of its neighbours, and its conditional precision τ times its number of neighbours. It is
proper for τ > 0 and α between 1/λ_min and 1, where λ are the eigenvalues of D^{-1/2} W D^{-1/2}; 0 ≤ α < 1 is the
usual prior range. Its log determinant is Σ ln(1 − αλᵢ) + Σ ln Dᵢᵢ + n ln τ, with the eigenvalues computed once
when the node is built (Jin, Carlin and Banerjee 2005). Evaluating the density then costs O(n + edges), as the
quadratic form is a sum over the edges. Every area needs a neighbour (D must be invertible), so `car` returns an
error for a neighbourhood with an island. Add a mean outside it if you need one.

## BYM2

The Besag–York–Mollié model adds independent noise to the ICAR, since not all of the variation between areas is
spatial. Its BYM2 parameterisation (Riebler et al. 2016) is

b = σ (√(ρ / s) φ + √(1 − ρ) θ),

where φ is a unit-scale ICAR, θ independent N(0, 1) per area, ρ in (0, 1) the share of the variance that is spatial
and σ the effect's overall scale. The scaling factor s is the geometric mean of the ICAR's marginal variances on each
component, the diagonal of Q⁺, as INLA's `scale.model` computes it (`Adjacency::scaling_factors`, 1 for an island).
Dividing by √s gives the structured part a typical variance of 1 whatever the map, so σ and ρ mean the same on any
neighbourhood and their priors carry over. `bym2(adjacency, sigma, rho)` builds b and returns its three nodes:
`spatial` (φ), `unstructured` (θ) and `effect` (b).

A Poisson disease-mapping model takes the log of each area's relative risk to be an intercept plus b, with the
expected counts as an offset:

```rust,ignore
{{#include ../../keanu/tests/book.rs:bym2}}
```

The [disease mapping example](examples.md#disease-mapping) fits the same model with a covariate on a 10 × 10
lattice, from counts simulated with known σ, ρ and coefficients, and recovers them. Two practical notes:

- **Start inside the support.** The ICAR's starting value must sum to zero on each component (zeros do), or its log
  probability is −∞.
- **Watch for spatial confounding.** A covariate that varies as smoothly as the spatial effect competes with it, and
  its coefficient can move a long way when the effect is added. The example's covariate has no spatial structure.

## Cost

The eigendecompositions behind the ICAR's sampler, the BYM2 scaling factors and the proper CAR's eigenvalues are
dense, O(m³) time and O(m²) memory for a component of m areas, computed once per `Adjacency` and cached. That is a
fraction of a second for hundreds of areas and seconds for a few thousand. The densities and their gradients and
Hessian-vector products are sparse, O(n + edges) per evaluation.
