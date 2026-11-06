//! Areal spatial models: a neighbourhood structure ([`Adjacency`]) and the intrinsic and proper conditional
//! autoregressive distributions over it ([`Icar`], [`Car`]).
//!
//! An areal model has one value per area (a region, a district, a cell of a lattice), and areas that share a border
//! are neighbours. The conditional autoregressive (CAR) family makes neighbours similar: each value's conditional
//! distribution given the others is centred on the average of its neighbours'.

use crate::error::{DistributionError, Result, invalid_parameter};
use crate::multivariate::sums_to_zero;
use crate::param::{IntoParam, Param};
use crate::rng::KeanuRng;
use keanu_tensor::{DoubleTensor, SparseMatrix};
use std::f64::consts::PI;
use std::sync::OnceLock;

/// The neighbourhood structure of `n` areas: an undirected graph without self loops, with its connected components,
/// degrees and the derived quantities the spatial distributions need.
///
/// The quantities that need an eigendecomposition (the proper CAR's eigenvalues, the ICAR's per-component
/// eigenvectors for sampling and the BYM2 scaling factors) are computed densely the first time they are needed and
/// kept: O(m³) time and O(m²) memory for a component of m areas, which is quick up to a few thousand areas.
///
/// ```
/// use keanu_distributions::Adjacency;
///
/// // a path 0 - 1 - 2 and an island 3
/// let a = Adjacency::from_edges(4, &[(0, 1), (2, 1)])?;
/// assert_eq!(a.edges(), &[(0, 1), (1, 2)]);
/// assert_eq!(a.degrees(), &[1.0, 2.0, 1.0, 0.0]);
/// assert_eq!(a.components(), &[vec![0, 1, 2], vec![3]]);
/// # Ok::<(), keanu_distributions::DistributionError>(())
/// ```
///
/// Two neighbourhoods are equal if they have the same areas and edges, and it formats as its number of areas, its
/// edges and its components (not the cached decompositions).
#[derive(Clone)]
pub struct Adjacency {
    n: usize,
    edges: Vec<(usize, usize)>,
    degrees: Vec<f64>,
    neighbours: SparseMatrix,
    components: Vec<Vec<usize>>,
    /// The eigenvalues of D^{-1/2} W D^{-1/2}, ascending.
    car_eigenvalues: OnceLock<Vec<f64>>,
    /// For each component of two or more areas, the eigenvalues (ascending) and eigenvectors (row-major, one per
    /// column) of its graph Laplacian.
    laplacian_eigen: OnceLock<Vec<(Vec<f64>, Vec<f64>)>>,
    scaling_factors: OnceLock<Vec<f64>>,
}

impl PartialEq for Adjacency {
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n && self.edges == other.edges
    }
}

impl std::fmt::Debug for Adjacency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Adjacency")
            .field("areas", &self.n)
            .field("edges", &self.edges)
            .field("components", &self.components)
            .finish()
    }
}

impl Adjacency {
    /// The neighbourhood of `n` areas with the given edges between pairs of areas. Each edge may be listed once,
    /// in either orientation, or in both (as an adjacency list does); repeats are merged.
    ///
    /// # Errors
    ///
    /// [`DistributionError::InvalidParameter`] for an area index of `n` or more, or an edge from an area to
    /// itself.
    pub fn from_edges(n: usize, edges: &[(usize, usize)]) -> Result<Self> {
        let mut normalised = Vec::with_capacity(edges.len());
        for &(i, j) in edges {
            if i >= n || j >= n {
                return invalid_parameter(format!(
                    "the edge ({i}, {j}) names an area outside 0..{n}"
                ));
            }
            if i == j {
                return invalid_parameter(format!("the edge ({i}, {i}) joins an area to itself"));
            }
            normalised.push((i.min(j), i.max(j)));
        }
        normalised.sort_unstable();
        normalised.dedup();
        let mut degrees = vec![0.0; n];
        let mut triplets = Vec::with_capacity(2 * normalised.len());
        for &(i, j) in &normalised {
            degrees[i] += 1.0;
            degrees[j] += 1.0;
            triplets.push((i, j, 1.0));
            triplets.push((j, i, 1.0));
        }
        let neighbours = SparseMatrix::from_triplets(n, n, &triplets)?;
        let components = connected_components(&neighbours);
        Ok(Adjacency {
            n,
            edges: normalised,
            degrees,
            neighbours,
            components,
            car_eigenvalues: OnceLock::new(),
            laplacian_eigen: OnceLock::new(),
            scaling_factors: OnceLock::new(),
        })
    }

    /// The neighbourhood given by a symmetric 0/1 adjacency matrix (shape `[n, n]`, zero diagonal).
    ///
    /// # Errors
    ///
    /// [`DistributionError::InvalidParameter`] unless the matrix is square, symmetric, has only 0 and 1 entries
    /// and a zero diagonal.
    pub fn from_matrix(matrix: &DoubleTensor) -> Result<Self> {
        let &[n, m] = matrix.shape() else {
            return invalid_parameter(format!(
                "an adjacency matrix must be a matrix, not a tensor of shape {:?}",
                matrix.shape()
            ));
        };
        if n != m {
            return invalid_parameter(format!("an adjacency matrix must be square, not {n} × {m}"));
        }
        let w = matrix.as_slice();
        let mut edges = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let v = w[i * n + j];
                if v != 0.0 && v != 1.0 {
                    return invalid_parameter(format!(
                        "an adjacency matrix has only 0 and 1 entries, not {v} at ({i}, {j})"
                    ));
                }
                if v != w[j * n + i] {
                    return invalid_parameter(format!(
                        "an adjacency matrix must be symmetric, but ({i}, {j}) and ({j}, {i}) differ"
                    ));
                }
                if v == 1.0 && i == j {
                    return invalid_parameter(format!(
                        "an adjacency matrix has a nonzero diagonal at {i}"
                    ));
                }
                if v == 1.0 && i < j {
                    edges.push((i, j));
                }
            }
        }
        Self::from_edges(n, &edges)
    }

    /// The neighbourhood of the cells of a `rows` × `cols` lattice, numbered row by row, each a neighbour of the
    /// cells above, below, left and right of it (rook adjacency).
    ///
    /// # Errors
    ///
    /// Never, in practice (the edges are valid by construction).
    pub fn lattice(rows: usize, cols: usize) -> Result<Self> {
        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let i = r * cols + c;
                if c + 1 < cols {
                    edges.push((i, i + 1));
                }
                if r + 1 < rows {
                    edges.push((i, i + cols));
                }
            }
        }
        Self::from_edges(rows * cols, &edges)
    }

    /// The number of areas.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether there are no areas.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The edges, each as `(i, j)` with i < j, sorted.
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    /// The number of neighbours of each area (the diagonal of D).
    pub fn degrees(&self) -> &[f64] {
        &self.degrees
    }

    /// The adjacency matrix W, sparse (1 between neighbours, 0 elsewhere).
    pub fn neighbours(&self) -> &SparseMatrix {
        &self.neighbours
    }

    /// The graph Laplacian Q = D − W, the precision matrix of the ICAR (up to its scale), sparse.
    pub fn laplacian(&self) -> SparseMatrix {
        let mut triplets: Vec<(usize, usize, f64)> = self
            .neighbours
            .triplets()
            .map(|(i, j, _)| (i, j, -1.0))
            .collect();
        triplets.extend(
            self.degrees
                .iter()
                .enumerate()
                .filter(|(_, d)| **d > 0.0)
                .map(|(i, &d)| (i, i, d)),
        );
        SparseMatrix::from_triplets(self.n, self.n, &triplets).expect("the Laplacian is valid")
    }

    /// The connected components, each a sorted list of areas, ordered by their first area. An area without
    /// neighbours (an island) is a component of its own.
    pub fn components(&self) -> &[Vec<usize>] {
        &self.components
    }

    /// The components of two or more areas: the ones an ICAR constrains to sum to zero.
    pub fn connected_groups(&self) -> impl Iterator<Item = &Vec<usize>> + '_ {
        self.components.iter().filter(|c| c.len() > 1)
    }

    /// The number of areas minus the number of components of two or more areas: the dimension of an ICAR's
    /// support, and the power of its scale in the density.
    pub fn icar_rank(&self) -> usize {
        self.n - self.connected_groups().count()
    }

    /// The eigenvalues of D^{-1/2} W D^{-1/2} (ascending), which give a proper CAR's log determinant: the
    /// precision τ(D − αW) is positive definite exactly when 1 − αλᵢ > 0 for every eigenvalue λᵢ, i.e. for α
    /// between 1/λ_min and 1 (the largest eigenvalue is 1 on every component of two or more areas).
    ///
    /// # Errors
    ///
    /// [`DistributionError::InvalidParameter`] if an area has no neighbours (D is then singular, so a proper CAR
    /// is undefined there).
    pub fn car_eigenvalues(&self) -> Result<&[f64]> {
        if let Some(i) = self.degrees.iter().position(|&d| d == 0.0) {
            return invalid_parameter(format!(
                "area {i} has no neighbours, so a proper CAR is undefined (use an ICAR, which gives islands an \
                 independent normal, or join it to a neighbour)"
            ));
        }
        if let Some(values) = self.car_eigenvalues.get() {
            return Ok(values);
        }
        let n = self.n;
        let mut scaled = vec![0.0; n * n];
        for (i, j, _) in self.neighbours.triplets() {
            scaled[i * n + j] = 1.0 / (self.degrees[i] * self.degrees[j]).sqrt();
        }
        let (values, _) = DoubleTensor::new(&[n, n], scaled)?.eigh()?;
        Ok(self.car_eigenvalues.get_or_init(|| values.into_vec()))
    }

    /// The eigendecomposition of each component's Laplacian, for components of two or more areas, in the order of
    /// [`Self::connected_groups`].
    fn laplacian_eigen(&self) -> Result<&[(Vec<f64>, Vec<f64>)]> {
        if let Some(eigen) = self.laplacian_eigen.get() {
            return Ok(eigen);
        }
        let mut position = vec![0; self.n];
        let mut eigen = Vec::new();
        for group in self.connected_groups() {
            let m = group.len();
            for (k, &i) in group.iter().enumerate() {
                position[i] = k;
            }
            let mut q = vec![0.0; m * m];
            for &i in group {
                let pi = position[i];
                q[pi * m + pi] = self.degrees[i];
                let row = self.neighbours.row_offsets()[i]..self.neighbours.row_offsets()[i + 1];
                for &j in &self.neighbours.col_indices()[row] {
                    q[pi * m + position[j]] = -1.0;
                }
            }
            let (values, vectors) = DoubleTensor::new(&[m, m], q)?.eigh()?;
            eigen.push((values.into_vec(), vectors.into_vec()));
        }
        Ok(self.laplacian_eigen.get_or_init(|| eigen))
    }

    /// The BYM2 scaling factor of each area (Riebler et al. 2016, as INLA's `scale.model`): for an area in a
    /// component of two or more areas, the geometric mean over that component of the diagonal of the generalised
    /// (Moore–Penrose) inverse of its Laplacian, i.e. of the marginal variances of a unit-scale ICAR; 1 for an
    /// island. Dividing an ICAR by the square root of its area's factor gives it a typical marginal variance of 1,
    /// so its scale means the same whatever the neighbourhood.
    ///
    /// # Errors
    ///
    /// Only if the eigendecomposition fails, which it doesn't for a valid adjacency.
    pub fn scaling_factors(&self) -> Result<&[f64]> {
        if let Some(factors) = self.scaling_factors.get() {
            return Ok(factors);
        }
        let mut factors = vec![1.0; self.n];
        for (group, (values, vectors)) in self.connected_groups().zip(self.laplacian_eigen()?) {
            let m = group.len();
            // the diagonal of Q⁺ = Σ_{k ≥ 1} v_k v_kᵀ / λ_k, skipping the zero eigenvalue of the constant vector
            let mut log_sum = 0.0;
            for i in 0..m {
                let variance: f64 = (1..m).map(|k| vectors[i * m + k].powi(2) / values[k]).sum();
                log_sum += variance.ln();
            }
            let factor = (log_sum / m as f64).exp();
            for &i in group {
                factors[i] = factor;
            }
        }
        Ok(self.scaling_factors.get_or_init(|| factors))
    }

    fn check_value(&self, x: &DoubleTensor, what: &str) -> Result<()> {
        if x.shape() == [self.n] {
            Ok(())
        } else {
            Err(DistributionError::InvalidValue(format!(
                "a {what} value over {} areas has shape [{}], not {:?}",
                self.n,
                self.n,
                x.shape()
            )))
        }
    }
}

/// The connected components of a symmetric sparse matrix's graph, each sorted, ordered by their first node.
fn connected_components(w: &SparseMatrix) -> Vec<Vec<usize>> {
    let n = w.rows();
    let mut seen = vec![false; n];
    let mut components = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut component = vec![start];
        let mut next = 0;
        while next < component.len() {
            let i = component[next];
            next += 1;
            let row = w.row_offsets()[i]..w.row_offsets()[i + 1];
            for &j in &w.col_indices()[row] {
                if !seen[j] {
                    seen[j] = true;
                    component.push(j);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components
}

/// The one value of a scalar parameter.
fn scalar(name: &str, p: &DoubleTensor) -> Result<f64> {
    match p.as_slice() {
        [v] => Ok(*v),
        _ => invalid_parameter(format!(
            "{name} must be a scalar, not a tensor of shape {:?}",
            p.shape()
        )),
    }
}

/// The intrinsic conditional autoregressive (ICAR) distribution over the areas of an [`Adjacency`], in the
/// pairwise-difference form of Morris et al. (2019, "Bayesian hierarchical spatial models: implementing the
/// Besag York Mollié model in Stan"), with scale σ:
///
/// `ln p(φ) = −½ Σ_{i~j} (φᵢ − φⱼ)² / σ² − ½ Σ_{islands} φᵢ² / σ² − r ln σ`,
///
/// summed over the edges, where r is [`Adjacency::icar_rank`] (the number of areas minus the number of components
/// of two or more areas). In matrix form the first term is −½ φᵀQφ / σ² with Q = D − W the graph Laplacian. Its
/// normalising constant is dropped, as in Stan's ICAR, so it is a density up to a constant.
///
/// The ICAR is improper: adding a constant to every area of a component doesn't change it. Its support is the
/// vectors summing to zero on each component of two or more areas (a hard constraint, which is the identifying
/// constraint of Morris et al. per component). An island (an area without neighbours) gets an independent
/// N(0, σ²), the usual convention (Freni-Sterrantino et al. 2018), and is not constrained. Restricted to its
/// support it is a proper Gaussian whose covariance is σ² Q⁺, Q⁺ the generalised inverse of the Laplacian.
#[derive(Debug, Clone, PartialEq)]
pub struct Icar<'a> {
    /// The neighbourhood.
    pub adjacency: &'a Adjacency,
    /// The scale σ > 0, a scalar.
    pub sigma: Param<'a>,
}

/// Derivatives of [`Icar::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct IcarGradient {
    /// With respect to φ: −(Qφ + φ on the islands) / σ².
    pub x: DoubleTensor,
    /// With respect to `sigma`, shaped like it.
    pub sigma: DoubleTensor,
}

impl<'a> Icar<'a> {
    /// Creates the distribution over a neighbourhood, with scale `sigma`.
    pub fn new(adjacency: &'a Adjacency, sigma: impl IntoParam<'a, f64>) -> Self {
        Self {
            adjacency,
            sigma: sigma.into_param(),
        }
    }

    /// Pφ, where P is the Laplacian plus 1 on the diagonal of each island, and φᵀPφ.
    fn precision_times(&self, x: &[f64]) -> (Vec<f64>, f64) {
        let a = self.adjacency;
        let mut px = vec![0.0; x.len()];
        for &(i, j) in &a.edges {
            let d = x[i] - x[j];
            px[i] += d;
            px[j] -= d;
        }
        for (i, &d) in a.degrees.iter().enumerate() {
            if d == 0.0 {
                px[i] += x[i];
            }
        }
        let quadratic = px.iter().zip(x).map(|(p, x)| p * x).sum();
        (px, quadratic)
    }

    /// The log density of φ (shape `[n]`). −infinity for a value that is not finite or doesn't sum to zero on a
    /// component of two or more areas (within
    /// [`ZERO_SUM_TOLERANCE`](crate::multivariate::ZERO_SUM_TOLERANCE)), NaN where `sigma` is not positive.
    ///
    /// # Errors
    ///
    /// If φ is not of shape `[n]` or `sigma` is not a scalar.
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<f64> {
        self.adjacency.check_value(x, "ICAR")?;
        let sigma = scalar("sigma", &self.sigma)?;
        if sigma.is_nan() || sigma <= 0.0 {
            return Ok(f64::NAN);
        }
        let xs = x.as_slice();
        if !xs.iter().all(|v| v.is_finite()) || !self.in_support(xs) {
            return Ok(f64::NEG_INFINITY);
        }
        let (_, quadratic) = self.precision_times(xs);
        Ok(-0.5 * quadratic / (sigma * sigma) - self.adjacency.icar_rank() as f64 * sigma.ln())
    }

    /// Whether every component of two or more areas sums to zero.
    fn in_support(&self, x: &[f64]) -> bool {
        self.adjacency.connected_groups().all(|group| {
            let values: Vec<f64> = group.iter().map(|&i| x[i]).collect();
            sums_to_zero(&values)
        })
    }

    /// The derivatives of [`Self::log_prob`] with respect to φ and `sigma` (the support is not checked).
    ///
    /// # Errors
    ///
    /// As [`Self::log_prob`].
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<IcarGradient> {
        self.adjacency.check_value(x, "ICAR")?;
        let sigma = scalar("sigma", &self.sigma)?;
        let (px, quadratic) = self.precision_times(x.as_slice());
        let s2 = sigma * sigma;
        let rank = self.adjacency.icar_rank() as f64;
        Ok(IcarGradient {
            x: DoubleTensor::new(x.shape(), px.iter().map(|p| -p / s2).collect())?,
            sigma: DoubleTensor::full(self.sigma.shape(), quadratic / (s2 * sigma) - rank / sigma),
        })
    }

    /// Draws φ (shape `[n]`, the only shape): on each component of two or more areas, an exact draw of the
    /// zero-sum Gaussian with covariance σ² Q⁺ from the eigendecomposition of the component's Laplacian, and an
    /// independent N(0, σ²) for each island.
    ///
    /// # Errors
    ///
    /// Unless `shape` is `[n]` and `sigma` a positive, finite scalar.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let a = self.adjacency;
        if shape != [a.n] {
            return Err(DistributionError::InvalidValue(format!(
                "an ICAR over {} areas draws values of shape [{}], not {shape:?}",
                a.n, a.n
            )));
        }
        let sigma = scalar("sigma", &self.sigma)?;
        if !sigma.is_finite() || sigma <= 0.0 {
            return invalid_parameter(format!("sigma must be positive, got {sigma}"));
        }
        let mut x = vec![0.0; a.n];
        for (i, &d) in a.degrees.iter().enumerate() {
            if d == 0.0 {
                x[i] = sigma * rng.next_gaussian();
            }
        }
        for (group, (values, vectors)) in a.connected_groups().zip(a.laplacian_eigen()?) {
            let m = group.len();
            // the first eigenvalue is the zero of the constant vector; the others are positive
            for k in 1..m {
                let z = sigma * rng.next_gaussian() / values[k].sqrt();
                for (row, &i) in group.iter().enumerate() {
                    x[i] += z * vectors[row * m + k];
                }
            }
        }
        Ok(DoubleTensor::new(shape, x)?)
    }
}

/// The proper conditional autoregressive (CAR) distribution over the areas of an [`Adjacency`]: a zero-mean
/// multivariate normal with precision τ(D − αW), where D is the diagonal matrix of the areas' numbers of
/// neighbours and W the adjacency matrix. Each area's conditional mean is α times the average of its neighbours'
/// values, and its conditional precision τ times its number of neighbours.
///
/// The log density is
///
/// `ln p(φ) = ½ [Σᵢ ln(1 − αλᵢ) + Σᵢ ln Dᵢᵢ + n ln τ] − ½ τ (φᵀDφ − α φᵀWφ) − ½ n ln 2π`,
///
/// the log determinant computed from the eigenvalues λᵢ of D^{-1/2} W D^{-1/2}
/// ([`Adjacency::car_eigenvalues`], computed once; Jin, Carlin and Banerjee 2005), and the quadratic form from the
/// edges, so each evaluation costs O(n + edges). The precision is positive definite for τ > 0 and α in
/// (1/λ_min, 1); 0 ≤ α < 1 is the usual range. Every area must have a neighbour.
#[derive(Debug, Clone, PartialEq)]
pub struct Car<'a> {
    /// The neighbourhood.
    pub adjacency: &'a Adjacency,
    /// The precision scale τ > 0, a scalar.
    pub tau: Param<'a>,
    /// The spatial dependence α, a scalar in (1/λ_min, 1).
    pub alpha: Param<'a>,
}

/// Derivatives of [`Car::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct CarGradient {
    /// With respect to φ: −τ(D − αW)φ.
    pub x: DoubleTensor,
    /// With respect to `tau`, shaped like it.
    pub tau: DoubleTensor,
    /// With respect to `alpha`, shaped like it.
    pub alpha: DoubleTensor,
}

/// The quadratic forms of a proper CAR at a value: Dφ, Wφ, φᵀDφ and φᵀWφ.
struct CarForms {
    dx: Vec<f64>,
    wx: Vec<f64>,
    xdx: f64,
    xwx: f64,
}

impl<'a> Car<'a> {
    /// Creates the distribution over a neighbourhood with precision scale `tau` and spatial dependence `alpha`.
    pub fn new(
        adjacency: &'a Adjacency,
        tau: impl IntoParam<'a, f64>,
        alpha: impl IntoParam<'a, f64>,
    ) -> Self {
        Self {
            adjacency,
            tau: tau.into_param(),
            alpha: alpha.into_param(),
        }
    }

    fn forms(&self, x: &[f64]) -> CarForms {
        let a = self.adjacency;
        let dx: Vec<f64> = a.degrees.iter().zip(x).map(|(d, x)| d * x).collect();
        let mut wx = vec![0.0; x.len()];
        for &(i, j) in &a.edges {
            wx[i] += x[j];
            wx[j] += x[i];
        }
        let xdx = dx.iter().zip(x).map(|(d, x)| d * x).sum();
        let xwx = wx.iter().zip(x).map(|(w, x)| w * x).sum();
        CarForms { dx, wx, xdx, xwx }
    }

    fn parameters(&self) -> Result<(f64, f64, &[f64])> {
        let tau = scalar("tau", &self.tau)?;
        let alpha = scalar("alpha", &self.alpha)?;
        Ok((tau, alpha, self.adjacency.car_eigenvalues()?))
    }

    /// The log density of φ (shape `[n]`). NaN where τ is not positive or the precision is not positive definite
    /// (1 − αλᵢ ≤ 0 for some eigenvalue), −infinity for a value that is not finite.
    ///
    /// # Errors
    ///
    /// If φ is not of shape `[n]`, a parameter is not a scalar, or an area has no neighbours.
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<f64> {
        self.adjacency.check_value(x, "CAR")?;
        let (tau, alpha, lambda) = self.parameters()?;
        if tau.is_nan() || tau <= 0.0 || alpha.is_nan() {
            return Ok(f64::NAN);
        }
        let mut log_det = 0.0;
        for &l in lambda {
            let s = 1.0 - alpha * l;
            if s <= 0.0 {
                return Ok(f64::NAN);
            }
            log_det += s.ln();
        }
        let xs = x.as_slice();
        if !xs.iter().all(|v| v.is_finite()) {
            return Ok(f64::NEG_INFINITY);
        }
        let n = self.adjacency.n as f64;
        log_det += self.adjacency.degrees.iter().map(|d| d.ln()).sum::<f64>() + n * tau.ln();
        let f = self.forms(xs);
        Ok(0.5 * log_det - 0.5 * tau * (f.xdx - alpha * f.xwx) - 0.5 * n * (2.0 * PI).ln())
    }

    /// The derivatives of [`Self::log_prob`] with respect to φ, `tau` and `alpha`.
    ///
    /// # Errors
    ///
    /// As [`Self::log_prob`].
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<CarGradient> {
        self.adjacency.check_value(x, "CAR")?;
        let (tau, alpha, lambda) = self.parameters()?;
        let f = self.forms(x.as_slice());
        let n = self.adjacency.n as f64;
        let dx =
            f.dx.iter()
                .zip(&f.wx)
                .map(|(d, w)| -tau * (d - alpha * w))
                .collect();
        let d_alpha =
            -0.5 * lambda.iter().map(|l| l / (1.0 - alpha * l)).sum::<f64>() + 0.5 * tau * f.xwx;
        Ok(CarGradient {
            x: DoubleTensor::new(x.shape(), dx)?,
            tau: DoubleTensor::full(
                self.tau.shape(),
                0.5 * n / tau - 0.5 * (f.xdx - alpha * f.xwx),
            ),
            alpha: DoubleTensor::full(self.alpha.shape(), d_alpha),
        })
    }

    /// Draws φ (shape `[n]`, the only shape) exactly, from the Cholesky factor L of the precision: φ = L⁻ᵀ z.
    ///
    /// # Errors
    ///
    /// Unless `shape` is `[n]`, τ is positive and finite and the precision is positive definite.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let a = self.adjacency;
        if shape != [a.n] {
            return Err(DistributionError::InvalidValue(format!(
                "a CAR over {} areas draws values of shape [{}], not {shape:?}",
                a.n, a.n
            )));
        }
        let (tau, alpha, lambda) = self.parameters()?;
        if !tau.is_finite() || tau <= 0.0 {
            return invalid_parameter(format!("tau must be positive, got {tau}"));
        }
        if !alpha.is_finite() || lambda.iter().any(|l| 1.0 - alpha * l <= 0.0) {
            return invalid_parameter(format!(
                "alpha must be between 1/λ_min = {} and 1 for the precision to be positive definite, got {alpha}",
                lambda.first().map_or(f64::NEG_INFINITY, |l| 1.0 / l)
            ));
        }
        let n = a.n;
        let mut q = vec![0.0; n * n];
        for (i, &d) in a.degrees.iter().enumerate() {
            q[i * n + i] = tau * d;
        }
        for (i, j, _) in a.neighbours.triplets() {
            q[i * n + j] = -tau * alpha;
        }
        let l = DoubleTensor::new(&[n, n], q)?.cholesky()?;
        let z = DoubleTensor::new(&[n], (0..n).map(|_| rng.next_gaussian()).collect())?;
        Ok(l.solve_triangular(&z, true, true)?.reshape(&[n as isize])?)
    }
}
