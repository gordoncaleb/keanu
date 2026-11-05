//! The bijector for probability vectors.

use super::*;

/// Probability vectors over the last axis, for [`Support::Simplex`], as in Stan 2.37 and later: the K − 1
/// unconstrained coordinates are mapped by the inverse isometric log-ratio transform to a zero-sum K-vector `z`,
/// then `x = softmax(z)`. Leading axes are a batch.
///
/// The isometric log-ratio uses Stan's orthonormal basis of the zero-sum subspace (column `i` of `V`, for
/// `i = 1..K−1`, has `1/√(i(i+1))` in its first `i` entries and `−i/√(i(i+1))` in entry `i + 1`), so `z = V y` and
/// `y = Vᵀ log x`. The log-Jacobian of the map from `y` to the first K − 1 elements of `x` is
/// `Σᵢ log xᵢ + ½ log K`.
#[derive(Debug, Clone, PartialEq)]
pub struct Simplex {
    shape: Vec<usize>,
    k: usize,
}

/// The tolerance on a simplex value's sum.
const SIMPLEX_SUM_TOLERANCE: f64 = 1e-8;

impl Simplex {
    /// Simplices over the last axis of `shape`.
    ///
    /// # Errors
    ///
    /// [`GraphError::Transform`] if `shape` is a scalar shape or its last axis is empty.
    pub fn new(shape: &[usize]) -> Result<Self> {
        let (_, k) = last_axis("simplex", shape)?;
        if k == 0 {
            return Err(transform_error("a simplex needs at least one element"));
        }
        Ok(Simplex {
            shape: shape.to_vec(),
            k,
        })
    }

    fn batch(&self) -> usize {
        self.shape[..self.shape.len() - 1].iter().product()
    }

    /// `z = V y`, for `y` of length K − 1 and `z` of length K.
    pub(super) fn basis_times(y: &[f64], z: &mut [f64]) {
        z.fill(0.0);
        let mut suffix = 0.0;
        for i in (1..=y.len()).rev() {
            let n = i as f64;
            let w = y[i - 1] / (n * (n + 1.0)).sqrt();
            suffix += w;
            z[i - 1] += suffix;
            z[i] -= n * w;
        }
    }

    /// `y = Vᵀ z`, for `z` of length K and `y` of length K − 1.
    pub(super) fn basis_transpose_times(z: &[f64], y: &mut [f64]) {
        let mut prefix = 0.0;
        for i in 1..z.len() {
            prefix += z[i - 1];
            let n = i as f64;
            y[i - 1] = (prefix - n * z[i]) / (n * (n + 1.0)).sqrt();
        }
    }

    /// Calls `f(row, z, x, log_sum)` for each row's unconstrained coordinates `y`, with `z = V y`,
    /// `log_sum = log Σ exp(z)` and `x = softmax(z)`.
    fn for_each_row(&self, y: &[f64], mut f: impl FnMut(usize, &[f64], &[f64], f64)) {
        let k = self.k;
        let mut z = vec![0.0; k];
        let mut x = vec![0.0; k];
        for (row, y) in self.rows(y).enumerate() {
            Self::basis_times(y, &mut z);
            let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let log_sum = max + z.iter().map(|z| (z - max).exp()).sum::<f64>().ln();
            for (x, z) in x.iter_mut().zip(&z) {
                *x = (z - log_sum).exp();
            }
            f(row, &z, &x, log_sum);
        }
    }

    /// The unconstrained coordinates of each row. For K = 1 a row has none (and `chunks_exact(0)` is invalid), so
    /// this yields one empty row per batch member.
    fn rows<'a>(&self, y: &'a [f64]) -> Box<dyn Iterator<Item = &'a [f64]> + 'a> {
        if self.k == 1 {
            Box::new(std::iter::repeat_n(&[][..], self.batch()))
        } else {
            Box::new(y.chunks_exact(self.k - 1))
        }
    }
}

impl Bijector for Simplex {
    fn constrained_shape(&self) -> &[usize] {
        &self.shape
    }

    fn unconstrained_len(&self) -> usize {
        self.batch() * (self.k - 1)
    }

    fn forward(&self, y: &[f64]) -> Result<DoubleTensor> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        let mut out = Vec::with_capacity(self.batch() * self.k);
        self.for_each_row(y, |_, _, x, _| out.extend_from_slice(x));
        tensor(&self.shape, out)
    }

    fn inverse(&self, x: &DoubleTensor) -> Result<Vec<f64>> {
        check_shape("the value", x, &self.shape)?;
        let k = self.k;
        let mut y = vec![0.0; self.unconstrained_len()];
        let mut log_x = vec![0.0; k];
        for (row, x) in x.as_slice().chunks_exact(k).enumerate() {
            let sum: f64 = x.iter().sum();
            if !x.iter().all(|&v| v > 0.0 && v.is_finite())
                || (sum - 1.0).abs() > SIMPLEX_SUM_TOLERANCE
            {
                return Err(transform_error(format!(
                    "{x:?} is not a vector of positive elements summing to 1"
                )));
            }
            for (l, x) in log_x.iter_mut().zip(x) {
                *l = x.ln();
            }
            // Vᵀ is orthogonal to the ones vector, so the log-ratio's normaliser drops out.
            Self::basis_transpose_times(&log_x, &mut y[row * (k - 1)..(row + 1) * (k - 1)]);
        }
        Ok(y)
    }

    fn log_det_jacobian(&self, y: &[f64]) -> Result<f64> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        let half_log_k = 0.5 * (self.k as f64).ln();
        let mut total = 0.0;
        self.for_each_row(y, |_, z, _, log_sum| {
            // Σ log xᵢ with log xᵢ = zᵢ − log Σ exp(z), finite even where x underflows
            total += z.iter().map(|z| z - log_sum).sum::<f64>() + half_log_k;
        });
        Ok(total)
    }

    fn log_det_jacobian_gradient(&self, y: &[f64]) -> Result<Vec<f64>> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        let k = self.k;
        let mut g = vec![0.0; y.len()];
        let mut gz = vec![0.0; k];
        self.for_each_row(y, |row, _, x, _| {
            // ∂/∂z Σᵢ (zᵢ − log Σ exp z) = 1 − K x
            for (g, x) in gz.iter_mut().zip(x) {
                *g = 1.0 - k as f64 * x;
            }
            Self::basis_transpose_times(&gz, &mut g[row * (k - 1)..(row + 1) * (k - 1)]);
        });
        Ok(g)
    }

    fn forward_jvp(&self, y: &[f64], tangent: &[f64]) -> Result<DoubleTensor> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        check_len("the tangent", tangent.len(), self.unconstrained_len())?;
        let k = self.k;
        let tangents: Vec<&[f64]> = self.rows(tangent).collect();
        let mut dz = vec![0.0; k];
        let mut out = Vec::with_capacity(self.batch() * k);
        self.for_each_row(y, |row, _, x, _| {
            Self::basis_times(tangents[row], &mut dz);
            let m: f64 = x.iter().zip(&dz).map(|(x, d)| x * d).sum();
            out.extend(x.iter().zip(&dz).map(|(x, d)| x * (d - m)));
        });
        tensor(&self.shape, out)
    }

    fn forward_vjp(&self, y: &[f64], cotangent: &DoubleTensor) -> Result<Vec<f64>> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        let k = self.k;
        let c = cotangent.as_slice();
        let mut g = vec![0.0; y.len()];
        let mut gz = vec![0.0; k];
        self.for_each_row(y, |row, _, x, _| {
            let c = &c[row * k..(row + 1) * k];
            let m: f64 = x.iter().zip(c).map(|(x, c)| x * c).sum();
            for ((g, x), c) in gz.iter_mut().zip(x).zip(c) {
                *g = x * (c - m);
            }
            Self::basis_transpose_times(&gz, &mut g[row * (k - 1)..(row + 1) * (k - 1)]);
        });
        Ok(g)
    }

    fn forward_vjp_jvp(
        &self,
        y: &[f64],
        tangent: &[f64],
        cotangent: &DoubleTensor,
    ) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.unconstrained_len())?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        let k = self.k;
        let c = cotangent.as_slice();
        let tangents: Vec<&[f64]> = self.rows(tangent).collect();
        let mut g = vec![0.0; y.len()];
        let (mut dz, mut dx, mut gz) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
        self.for_each_row(y, |row, _, x, _| {
            // the vector-Jacobian product is Vᵀ (x ∘ (c − x·c)); along t, ẋ = x ∘ (ż − x·ż) with ż = V t, and its
            // derivative is Vᵀ (ẋ ∘ (c − x·c) − x (ẋ·c))
            let c = &c[row * k..(row + 1) * k];
            Self::basis_times(tangents[row], &mut dz);
            let xdz: f64 = x.iter().zip(&dz).map(|(x, d)| x * d).sum();
            for ((dx, x), dz) in dx.iter_mut().zip(x).zip(&dz) {
                *dx = x * (dz - xdz);
            }
            let m: f64 = x.iter().zip(c).map(|(x, c)| x * c).sum();
            let dm: f64 = dx.iter().zip(c).map(|(d, c)| d * c).sum();
            for (((g, x), dx), c) in gz.iter_mut().zip(x).zip(&dx).zip(c) {
                *g = dx * (c - m) - x * dm;
            }
            Self::basis_transpose_times(&gz, &mut g[row * (k - 1)..(row + 1) * (k - 1)]);
        });
        Ok(g)
    }

    fn log_det_jacobian_hvp(&self, y: &[f64], tangent: &[f64]) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.unconstrained_len())?;
        let k = self.k;
        let tangents: Vec<&[f64]> = self.rows(tangent).collect();
        let mut g = vec![0.0; y.len()];
        let (mut dz, mut gz) = (vec![0.0; k], vec![0.0; k]);
        self.for_each_row(y, |row, _, x, _| {
            // the gradient is Vᵀ (1 − K x), so the product is Vᵀ (−K ẋ)
            Self::basis_times(tangents[row], &mut dz);
            let xdz: f64 = x.iter().zip(&dz).map(|(x, d)| x * d).sum();
            for ((g, x), dz) in gz.iter_mut().zip(x).zip(&dz) {
                *g = -(k as f64) * x * (dz - xdz);
            }
            Self::basis_transpose_times(&gz, &mut g[row * (k - 1)..(row + 1) * (k - 1)]);
        });
        Ok(g)
    }
}
