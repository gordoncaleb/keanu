//! The bijectors for covariance matrices and their Cholesky factors.

use super::*;

/// The tolerance on a covariance matrix's symmetry (relative) and a Cholesky factor's upper triangle.
const TOLERANCE: f64 = 1e-8;

/// The lower triangular factor L of each K × K matrix from its unconstrained coordinates (the lower triangle row by
/// row, the diagonal entries as logarithms), into `l` (K × K).
fn factor(y: &[f64], k: usize, l: &mut [f64]) {
    l.fill(0.0);
    let mut next = y.iter();
    for i in 0..k {
        for j in 0..=i {
            let v = *next.next().expect("K(K + 1)/2 coordinates");
            l[i * k + j] = if i == j { v.exp() } else { v };
        }
    }
}

/// The tangent of L along the unconstrained tangent `t` at the factor `l`: t below the diagonal and L_ii t_ii on it.
fn factor_tangent(t: &[f64], l: &[f64], k: usize, dl: &mut [f64]) {
    factor_coordinates(t, k, dl);
    for i in 0..k {
        dl[i * k + i] *= l[i * k + i];
    }
}

/// The lower triangle of a K × K matrix laid out from coordinates as they are (no exponential).
fn factor_coordinates(y: &[f64], k: usize, out: &mut [f64]) {
    out.fill(0.0);
    let mut next = y.iter();
    for i in 0..k {
        for j in 0..=i {
            out[i * k + j] = *next.next().expect("K(K + 1)/2 coordinates");
        }
    }
}

/// The coordinates of the lower triangle of a K × K matrix, row by row.
fn lower_coordinates(a: &[f64], k: usize, out: &mut Vec<f64>) {
    for i in 0..k {
        out.extend_from_slice(&a[i * k..=i * k + i]);
    }
}

/// The symmetric product A Bᵀ + B Aᵀ of two K × K matrices, into `out`.
fn symmetric_product(a: &[f64], b: &[f64], k: usize, out: &mut [f64]) {
    for i in 0..k {
        for j in 0..=i {
            let v: f64 = (0..k)
                .map(|m| a[i * k + m] * b[j * k + m] + b[i * k + m] * a[j * k + m])
                .sum();
            out[i * k + j] = v;
            out[j * k + i] = v;
        }
    }
}

/// The product of two K × K matrices into `out`.
fn product(a: &[f64], b: &[f64], k: usize, out: &mut [f64]) {
    for i in 0..k {
        for j in 0..k {
            out[i * k + j] = (0..k).map(|m| a[i * k + m] * b[m * k + j]).sum();
        }
    }
}

/// The square matrices over the last two axes of `shape`: (batch, K).
fn square(what: &str, shape: &[usize]) -> Result<(usize, usize)> {
    match shape {
        [batch @ .., rows, cols] if rows == cols => Ok((batch.iter().product(), *rows)),
        _ => Err(transform_error(format!(
            "a {what} needs a shape ending in two equal axes, not {shape:?}"
        ))),
    }
}

/// Symmetric positive definite K × K matrices over the last two axes, for [`Support::CovarianceMatrix`], with Stan's
/// transform ("Covariance matrices" in the Stan reference manual). Leading axes are a batch.
///
/// The K(K + 1)/2 unconstrained coordinates of each matrix fill the lower triangle of a factor L row by row, the
/// diagonal entries as their logarithms (`Lᵢᵢ = exp(yᵢᵢ)`), and the value is `Σ = L Lᵀ`. The log-Jacobian of the map
/// from `y` to the lower triangle of Σ (the free elements a density over covariance matrices, such as the Wishart, is
/// a density of) is `K ln 2 + Σᵢ (K − i + 1) yᵢᵢ` (zero-based i): the map from L to Σ contributes
/// `2^K Π Lᵢᵢ^(K − i)` and the exponentials one more power of each Lᵢᵢ.
#[derive(Debug, Clone, PartialEq)]
pub struct CovarianceMatrix {
    shape: Vec<usize>,
    batch: usize,
    k: usize,
}

impl CovarianceMatrix {
    /// Covariance matrices over the last two axes of `shape`.
    ///
    /// # Errors
    ///
    /// [`GraphError::Transform`] if `shape` has fewer than two axes or its last two axes differ.
    pub fn new(shape: &[usize]) -> Result<Self> {
        let (batch, k) = square("covariance matrix", shape)?;
        Ok(Self {
            shape: shape.to_vec(),
            batch,
            k,
        })
    }

    fn per_matrix(&self) -> usize {
        self.k * (self.k + 1) / 2
    }
}

impl Bijector for CovarianceMatrix {
    fn constrained_shape(&self) -> &[usize] {
        &self.shape
    }

    fn unconstrained_len(&self) -> usize {
        self.batch * self.per_matrix()
    }

    fn forward(&self, y: &[f64]) -> Result<DoubleTensor> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        let (k, n) = (self.k, self.per_matrix());
        let mut out = vec![0.0; self.batch * k * k];
        let mut l = vec![0.0; k * k];
        for (m, sigma) in rows_mut(&mut out, k * k).enumerate() {
            factor(&y[m * n..(m + 1) * n], k, &mut l);
            for i in 0..k {
                for j in 0..=i {
                    let v: f64 = (0..=j).map(|q| l[i * k + q] * l[j * k + q]).sum();
                    sigma[i * k + j] = v;
                    sigma[j * k + i] = v;
                }
            }
        }
        tensor(&self.shape, out)
    }

    fn inverse(&self, x: &DoubleTensor) -> Result<Vec<f64>> {
        check_shape("the value", x, &self.shape)?;
        let k = self.k;
        let mut y = Vec::with_capacity(self.unconstrained_len());
        for sigma in rows(x.as_slice(), k * k) {
            let symmetric = (0..k).all(|i| {
                (0..i).all(|j| {
                    let (a, b) = (sigma[i * k + j], sigma[j * k + i]);
                    (a - b).abs() <= TOLERANCE * a.abs().max(b.abs()).max(1.0)
                })
            });
            let invalid = || {
                transform_error(format!(
                    "a covariance matrix must be finite, symmetric and positive definite, not {sigma:?}"
                ))
            };
            if !symmetric || sigma.iter().any(|v| !v.is_finite()) {
                return Err(invalid());
            }
            // the Cholesky factor, from the lower triangle
            let mut l = vec![0.0; k * k];
            for i in 0..k {
                for j in 0..=i {
                    let s =
                        sigma[i * k + j] - (0..j).map(|m| l[i * k + m] * l[j * k + m]).sum::<f64>();
                    if i == j {
                        if s.is_nan() || s <= 0.0 {
                            return Err(invalid());
                        }
                        l[i * k + i] = s.sqrt();
                    } else {
                        l[i * k + j] = s / l[j * k + j];
                    }
                }
            }
            for i in 0..k {
                y.extend_from_slice(&l[i * k..i * k + i]);
                y.push(l[i * k + i].ln());
            }
        }
        Ok(y)
    }

    fn log_det_jacobian(&self, y: &[f64]) -> Result<f64> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        Ok(self
            .log_det_jacobian_gradient(y)?
            .iter()
            .zip(y)
            .map(|(g, y)| g * y)
            .sum::<f64>()
            + (self.batch * self.k) as f64 * std::f64::consts::LN_2)
    }

    fn log_det_jacobian_gradient(&self, y: &[f64]) -> Result<Vec<f64>> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        let k = self.k;
        let mut g = Vec::with_capacity(y.len());
        for _ in 0..self.batch {
            for i in 0..k {
                g.extend(std::iter::repeat_n(0.0, i));
                g.push((k - i + 1) as f64);
            }
        }
        Ok(g)
    }

    fn forward_jvp(&self, y: &[f64], tangent: &[f64]) -> Result<DoubleTensor> {
        check_second_order(y, tangent, self.unconstrained_len())?;
        let (k, n) = (self.k, self.per_matrix());
        let (mut l, mut dl) = (vec![0.0; k * k], vec![0.0; k * k]);
        let mut out = vec![0.0; self.batch * k * k];
        for (m, d) in rows_mut(&mut out, k * k).enumerate() {
            let range = m * n..(m + 1) * n;
            factor(&y[range.clone()], k, &mut l);
            factor_tangent(&tangent[range], &l, k, &mut dl);
            // dΣ = dL Lᵀ + L dLᵀ
            symmetric_product(&dl, &l, k, d);
        }
        tensor(&self.shape, out)
    }

    fn forward_vjp(&self, y: &[f64], cotangent: &DoubleTensor) -> Result<Vec<f64>> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        let (k, n) = (self.k, self.per_matrix());
        let (mut l, mut g_sym, mut gl) = (vec![0.0; k * k], vec![0.0; k * k], vec![0.0; k * k]);
        let mut g = Vec::with_capacity(y.len());
        for (m, c) in rows(cotangent.as_slice(), k * k).enumerate() {
            factor(&y[m * n..(m + 1) * n], k, &mut l);
            // the gradient in L is (C + Cᵀ) L, and in yᵢᵢ it is scaled by Lᵢᵢ
            for i in 0..k {
                for j in 0..k {
                    g_sym[i * k + j] = c[i * k + j] + c[j * k + i];
                }
            }
            product(&g_sym, &l, k, &mut gl);
            for i in 0..k {
                gl[i * k + i] *= l[i * k + i];
            }
            lower_coordinates(&gl, k, &mut g);
        }
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
        let (k, n) = (self.k, self.per_matrix());
        let mut l = vec![0.0; k * k];
        let mut dl = vec![0.0; k * k];
        let mut g_sym = vec![0.0; k * k];
        let (mut gl, mut dgl) = (vec![0.0; k * k], vec![0.0; k * k]);
        let mut out = Vec::with_capacity(y.len());
        for (m, c) in rows(cotangent.as_slice(), k * k).enumerate() {
            let range = m * n..(m + 1) * n;
            factor(&y[range.clone()], k, &mut l);
            factor_tangent(&tangent[range], &l, k, &mut dl);
            for i in 0..k {
                for j in 0..k {
                    g_sym[i * k + j] = c[i * k + j] + c[j * k + i];
                }
            }
            product(&g_sym, &l, k, &mut gl);
            product(&g_sym, &dl, k, &mut dgl);
            // d(gL_ii L_ii) = dgL_ii L_ii + gL_ii dL_ii
            for i in 0..k {
                dgl[i * k + i] = dgl[i * k + i] * l[i * k + i] + gl[i * k + i] * dl[i * k + i];
            }
            lower_coordinates(&dgl, k, &mut out);
        }
        Ok(out)
    }

    fn log_det_jacobian_hvp(&self, y: &[f64], tangent: &[f64]) -> Result<Vec<f64>> {
        // linear in y
        check_second_order(y, tangent, self.unconstrained_len())?;
        Ok(vec![0.0; y.len()])
    }
}

/// Lower triangular K × K matrices with a positive diagonal over the last two axes (Cholesky factors of covariance
/// matrices), for [`Support::CovarianceCholesky`], with Stan's `cholesky_factor_cov` transform for square factors.
/// Leading axes are a batch.
///
/// The K(K + 1)/2 unconstrained coordinates of each matrix fill its lower triangle row by row, the diagonal
/// entries as their logarithms (`Lᵢᵢ = exp(yᵢᵢ)`); the upper triangle is zero. The log-Jacobian of the map to the
/// lower triangle is `Σᵢ yᵢᵢ`.
#[derive(Debug, Clone, PartialEq)]
pub struct CovarianceCholesky {
    shape: Vec<usize>,
    batch: usize,
    k: usize,
}

impl CovarianceCholesky {
    /// Cholesky factors over the last two axes of `shape`.
    ///
    /// # Errors
    ///
    /// [`GraphError::Transform`] if `shape` has fewer than two axes or its last two axes differ.
    pub fn new(shape: &[usize]) -> Result<Self> {
        let (batch, k) = square("covariance Cholesky factor", shape)?;
        Ok(Self {
            shape: shape.to_vec(),
            batch,
            k,
        })
    }

    fn per_matrix(&self) -> usize {
        self.k * (self.k + 1) / 2
    }

    /// Whether each unconstrained coordinate is a diagonal entry's.
    fn diagonal_flags(&self) -> impl Iterator<Item = bool> + '_ {
        (0..self.batch).flat_map(move |_| (0..self.k).flat_map(|i| (0..=i).map(move |j| i == j)))
    }
}

impl Bijector for CovarianceCholesky {
    fn constrained_shape(&self) -> &[usize] {
        &self.shape
    }

    fn unconstrained_len(&self) -> usize {
        self.batch * self.per_matrix()
    }

    fn forward(&self, y: &[f64]) -> Result<DoubleTensor> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        let (k, n) = (self.k, self.per_matrix());
        let mut out = vec![0.0; self.batch * k * k];
        for (m, l) in rows_mut(&mut out, k * k).enumerate() {
            factor(&y[m * n..(m + 1) * n], k, l);
        }
        tensor(&self.shape, out)
    }

    fn inverse(&self, x: &DoubleTensor) -> Result<Vec<f64>> {
        check_shape("the value", x, &self.shape)?;
        let k = self.k;
        let mut y = Vec::with_capacity(self.unconstrained_len());
        for l in rows(x.as_slice(), k * k) {
            for i in 0..k {
                let row = &l[i * k..(i + 1) * k];
                // written so that NaN fails every check
                let valid = row[i] > 0.0
                    && row[i].is_finite()
                    && row[..i].iter().all(|v| v.is_finite())
                    && row[i + 1..].iter().all(|v| v.abs() <= TOLERANCE);
                if !valid {
                    return Err(transform_error(format!(
                        "row {i} of a covariance Cholesky factor, {row:?}, does not have a positive diagonal and \
                         zeros above it"
                    )));
                }
                y.extend_from_slice(&row[..i]);
                y.push(row[i].ln());
            }
        }
        Ok(y)
    }

    fn log_det_jacobian(&self, y: &[f64]) -> Result<f64> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        Ok(y.iter()
            .zip(self.diagonal_flags())
            .filter(|(_, d)| *d)
            .map(|(y, _)| y)
            .sum())
    }

    fn log_det_jacobian_gradient(&self, y: &[f64]) -> Result<Vec<f64>> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        Ok(self
            .diagonal_flags()
            .map(|d| if d { 1.0 } else { 0.0 })
            .collect())
    }

    fn forward_jvp(&self, y: &[f64], tangent: &[f64]) -> Result<DoubleTensor> {
        check_second_order(y, tangent, self.unconstrained_len())?;
        let (k, n) = (self.k, self.per_matrix());
        let mut l = vec![0.0; k * k];
        let mut out = vec![0.0; self.batch * k * k];
        for (m, d) in rows_mut(&mut out, k * k).enumerate() {
            let range = m * n..(m + 1) * n;
            factor(&y[range.clone()], k, &mut l);
            factor_tangent(&tangent[range], &l, k, d);
        }
        tensor(&self.shape, out)
    }

    fn forward_vjp(&self, y: &[f64], cotangent: &DoubleTensor) -> Result<Vec<f64>> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        let (k, n) = (self.k, self.per_matrix());
        let mut l = vec![0.0; k * k];
        let mut scaled = vec![0.0; k * k];
        let mut g = Vec::with_capacity(y.len());
        for (m, c) in rows(cotangent.as_slice(), k * k).enumerate() {
            factor(&y[m * n..(m + 1) * n], k, &mut l);
            scaled.copy_from_slice(c);
            for i in 0..k {
                scaled[i * k + i] *= l[i * k + i];
            }
            lower_coordinates(&scaled, k, &mut g);
        }
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
        // only the diagonal's gradient cᵢᵢ Lᵢᵢ depends on y: its derivative is cᵢᵢ Lᵢᵢ tᵢᵢ
        let c = cotangent.as_slice();
        let k = self.k;
        let mut out = Vec::with_capacity(y.len());
        let mut index = 0;
        for m in 0..self.batch {
            for i in 0..k {
                for j in 0..=i {
                    out.push(if i == j {
                        c[m * k * k + i * k + i] * y[index].exp() * tangent[index]
                    } else {
                        0.0
                    });
                    index += 1;
                }
            }
        }
        Ok(out)
    }

    fn log_det_jacobian_hvp(&self, y: &[f64], tangent: &[f64]) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.unconstrained_len())?;
        Ok(vec![0.0; y.len()])
    }
}
