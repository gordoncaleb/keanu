//! The bijector for Cholesky factors of correlation matrices.

use super::*;

/// Lower triangular Cholesky factors of correlation matrices over the last two axes, for
/// [`Support::CorrelationCholesky`], as in the Stan reference manual ("Cholesky factors of correlation matrices").
/// Leading axes are a batch.
///
/// The K(K − 1)/2 unconstrained coordinates of each matrix, in the row-major order of its strictly lower triangle,
/// are mapped to canonical partial correlations `zᵢⱼ = tanh(yᵢⱼ)`, and then row by row
/// `Lᵢ₀ = zᵢ₀`, `Lᵢⱼ = zᵢⱼ √(1 − Σₘ₍ⱼ Lᵢₘ²)` and `Lᵢᵢ = √(1 − Σₘ₍ᵢ Lᵢₘ²)`, with `L₀₀ = 1`. Each row has unit
/// length and the diagonal is positive. The log-Jacobian of the map from `y` to the strictly lower triangle is
/// `−Σᵢⱼ (i + 1 − j) log cosh yᵢⱼ` (zero-based `i > j`), Stan's `Σ log(1 − z²) + ½ Σ log(1 − Σ L²)` written in
/// `y`.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrelationCholesky {
    shape: Vec<usize>,
    k: usize,
}

/// The tolerance on a correlation Cholesky factor's row lengths and upper triangle.
const CHOLESKY_TOLERANCE: f64 = 1e-8;

impl CorrelationCholesky {
    /// Correlation Cholesky factors over the last two axes of `shape`.
    ///
    /// # Errors
    ///
    /// [`GraphError::Transform`] if `shape` has fewer than two axes or its last two axes differ.
    pub fn new(shape: &[usize]) -> Result<Self> {
        match shape {
            [.., rows, cols] if rows == cols => Ok(CorrelationCholesky {
                shape: shape.to_vec(),
                k: *rows,
            }),
            _ => Err(transform_error(format!(
                "a correlation Cholesky factor needs a shape ending in two equal axes, not {shape:?}"
            ))),
        }
    }

    fn batch(&self) -> usize {
        self.shape[..self.shape.len() - 2].iter().product()
    }

    fn per_matrix(&self) -> usize {
        self.k * self.k.saturating_sub(1) / 2
    }

    /// Calls `f(matrix, row, y)` with the unconstrained coordinates of each row of each matrix (row `i` has `i`).
    fn for_each_row<'a>(&self, y: &'a [f64], mut f: impl FnMut(usize, usize, &'a [f64])) {
        let n = self.per_matrix();
        for m in 0..self.batch() {
            let y = &y[m * n..(m + 1) * n];
            for i in 0..self.k {
                let start = i * i.saturating_sub(1) / 2;
                f(m, i, &y[start..start + i]);
            }
        }
    }
}

impl Bijector for CorrelationCholesky {
    fn constrained_shape(&self) -> &[usize] {
        &self.shape
    }

    fn unconstrained_len(&self) -> usize {
        self.batch() * self.per_matrix()
    }

    fn forward(&self, y: &[f64]) -> Result<DoubleTensor> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        let k = self.k;
        let mut l = vec![0.0; self.batch() * k * k];
        self.for_each_row(y, |m, i, y| {
            let row = &mut l[(m * k + i) * k..(m * k + i + 1) * k];
            // s is the length still available to the row: √(1 − Σ L²) = Π sech yₘ
            let mut s = 1.0;
            for (j, &y) in y.iter().enumerate() {
                row[j] = y.tanh() * s;
                s /= y.cosh();
            }
            row[i] = s;
        });
        tensor(&self.shape, l)
    }

    fn inverse(&self, x: &DoubleTensor) -> Result<Vec<f64>> {
        check_shape("the value", x, &self.shape)?;
        let k = self.k;
        let mut y = Vec::with_capacity(self.unconstrained_len());
        for (index, row) in rows(x.as_slice(), k).enumerate() {
            let i = index % k;
            let invalid = || {
                transform_error(format!(
                    "row {i} of a correlation Cholesky factor, {row:?}, does not have unit length, a positive \
                     diagonal and zeros above the diagonal"
                ))
            };
            let norm: f64 = row.iter().map(|v| v * v).sum();
            // written so that NaN fails every check
            let valid = row[i] > 0.0
                && (norm - 1.0).abs() <= CHOLESKY_TOLERANCE
                && row[i + 1..].iter().all(|v| v.abs() <= CHOLESKY_TOLERANCE);
            if !valid {
                return Err(invalid());
            }
            // zᵢⱼ = Lᵢⱼ / √(Σₘ₌ⱼ..ᵢ Lᵢₘ²), from tail sums so the diagonal's share stays accurate
            let mut tail: f64 = row[..=i].iter().map(|v| v * v).sum();
            for &v in &row[..i] {
                let z = v / tail.sqrt();
                let yij = z.atanh();
                if !yij.is_finite() {
                    return Err(invalid());
                }
                y.push(yij);
                tail -= v * v;
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
        let mut total = 0.0;
        self.for_each_row(y, |_, i, y| {
            total -= y
                .iter()
                .enumerate()
                .map(|(j, &y)| (i + 1 - j) as f64 * log_cosh(y))
                .sum::<f64>();
        });
        Ok(total)
    }

    fn log_det_jacobian_gradient(&self, y: &[f64]) -> Result<Vec<f64>> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        let mut g = Vec::with_capacity(y.len());
        self.for_each_row(y, |_, i, y| {
            g.extend(
                y.iter()
                    .enumerate()
                    .map(|(j, &y)| -((i + 1 - j) as f64) * y.tanh()),
            );
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
        let mut dl = vec![0.0; self.batch() * k * k];
        let mut offset = 0;
        self.for_each_row(y, |m, i, y| {
            let t = &tangent[offset..offset + i];
            offset += i;
            let row = &mut dl[(m * k + i) * k..(m * k + i + 1) * k];
            // With c = sech y: dz = c² dy, dc = −c z dy, and s ← s c.
            let (mut s, mut ds) = (1.0, 0.0);
            for (j, (&y, &t)) in y.iter().zip(t).enumerate() {
                let (z, c) = (y.tanh(), 1.0 / y.cosh());
                row[j] = c * c * t * s + z * ds;
                ds = ds * c - s * c * z * t;
                s *= c;
            }
            row[i] = ds;
        });
        tensor(&self.shape, dl)
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
        let mut offset = 0;
        let mut s_before = vec![0.0; k];
        self.for_each_row(y, |m, i, y| {
            let cot = &c[(m * k + i) * k..(m * k + i + 1) * k];
            let g = &mut g[offset..offset + i];
            offset += i;
            // forward pass: the remaining length before each element
            let mut s = 1.0;
            for (j, &y) in y.iter().enumerate() {
                s_before[j] = s;
                s /= y.cosh();
            }
            // reverse pass: Lᵢᵢ = s, Lᵢⱼ = zⱼ s₍ⱼ₋₁₎, sⱼ = s₍ⱼ₋₁₎ cⱼ
            let mut gs = cot[i];
            for j in (0..i).rev() {
                let (z, sech) = (y[j].tanh(), 1.0 / y[j].cosh());
                let before = s_before[j];
                g[j] = cot[j] * before * sech * sech - gs * before * sech * z;
                gs = gs * sech + cot[j] * z;
            }
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
        let mut g = vec![0.0; y.len()];
        let mut offset = 0;
        let mut s_before = vec![Dual::constant(0.0); k];
        let mut duals = Vec::with_capacity(k);
        self.for_each_row(y, |m, i, y| {
            // forward_vjp's recurrences evaluated on dual numbers yⱼ + ε tⱼ: the ε parts are the derivative
            let cot = &c[(m * k + i) * k..(m * k + i + 1) * k];
            let t = &tangent[offset..offset + i];
            let g = &mut g[offset..offset + i];
            offset += i;
            duals.clear();
            duals.extend(y.iter().zip(t).map(|(&y, &t)| {
                let (z, sech) = (y.tanh(), 1.0 / y.cosh());
                (Dual(z, (1.0 - z * z) * t), Dual(sech, -sech * z * t))
            }));
            let mut s = Dual::constant(1.0);
            for (j, &(_, sech)) in duals.iter().enumerate() {
                s_before[j] = s;
                s = s * sech;
            }
            let mut gs = Dual::constant(cot[i]);
            for j in (0..i).rev() {
                let (z, sech) = duals[j];
                let before = s_before[j];
                let gj = before * sech * sech * cot[j] - gs * before * sech * z;
                g[j] = gj.1;
                gs = gs * sech + z * cot[j];
            }
        });
        Ok(g)
    }

    fn log_det_jacobian_hvp(&self, y: &[f64], tangent: &[f64]) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.unconstrained_len())?;
        let mut h = Vec::with_capacity(y.len());
        let mut offset = 0;
        self.for_each_row(y, |_, i, y| {
            // the gradient is −(i + 1 − j) tanh yᵢⱼ, so the Hessian is diagonal, −(i + 1 − j) sech² yᵢⱼ
            let t = &tangent[offset..offset + i];
            offset += i;
            h.extend(y.iter().zip(t).enumerate().map(|(j, (&y, &t))| {
                let sech = 1.0 / y.cosh();
                -((i + 1 - j) as f64) * sech * sech * t
            }));
        });
        Ok(h)
    }
}

/// A dual number `a + b ε` with `ε² = 0`: the value and directional derivative of a quantity, for derivatives of
/// scalar recurrences.
#[derive(Debug, Clone, Copy)]
struct Dual(f64, f64);

impl Dual {
    fn constant(value: f64) -> Self {
        Dual(value, 0.0)
    }
}

impl std::ops::Mul for Dual {
    type Output = Dual;
    fn mul(self, other: Dual) -> Dual {
        Dual(self.0 * other.0, self.0 * other.1 + self.1 * other.0)
    }
}

impl std::ops::Mul<f64> for Dual {
    type Output = Dual;
    fn mul(self, other: f64) -> Dual {
        Dual(self.0 * other, self.1 * other)
    }
}

impl std::ops::Add for Dual {
    type Output = Dual;
    fn add(self, other: Dual) -> Dual {
        Dual(self.0 + other.0, self.1 + other.1)
    }
}

impl std::ops::Sub for Dual {
    type Output = Dual;
    fn sub(self, other: Dual) -> Dual {
        Dual(self.0 - other.0, self.1 - other.1)
    }
}
