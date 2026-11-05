//! Hilbert space approximate Gaussian processes (HSGP), following Riutort-Mayol, Bürkner, Andersen, Solin and
//! Vehtari (2023), "Practical Hilbert space approximate Bayesian Gaussian processes for probabilistic programming".
//!
//! A stationary kernel's GP on a box [centre − L, centre + L] is approximated by a weighted sum of the Laplacian's
//! eigenfunctions with Dirichlet boundary conditions,
//!
//! f(x) ≈ Σⱼ φⱼ(x) · √S(√λⱼ) · βⱼ, βⱼ ~ N(0, 1),
//!
//! where S is the kernel's spectral density. In one dimension φⱼ(x) = L^(−1/2) sin(π j (x − centre + L) / (2L)) and
//! √λⱼ = π j / (2L) for j = 1..m; in d dimensions the basis is the full tensor grid of m functions per dimension, so
//! m^d in all, each the product of one function per dimension with √λ a vector of one frequency per dimension.
//!
//! The basis functions depend only on the inputs, so they are a constant matrix Φ, and f = Φ (√S ⊙ β) is built from
//! ordinary graph nodes (a constant, element-wise ops and a matrix product): its derivatives in the kernel's
//! hyperparameters and in β, first and second order, come from the graph's own autodiff. Evaluating f costs
//! O(n·m^d) instead of an exact GP's O(n³).

use super::{Form, Kernel, Kind, fail};
use crate::builder::GraphBuilder;
use crate::error::{GraphError, Result};
use crate::vars::DoubleVar;
use keanu_tensor::DoubleTensor;
use keanu_tensor::special::ln_gamma;
use std::f64::consts::PI;

const HSGP: &str = "hsgp";

/// The most basis functions (m^d) an [`Hsgp`] may have.
pub const MAX_BASIS_FUNCTIONS: usize = 1 << 20;

/// A Hilbert space approximate Gaussian process: a zero-mean GP with an exponentiated quadratic or Matérn kernel,
/// approximated by `m` basis functions per input dimension on a box that extends the inputs' range by the factor
/// `c` (see the [module documentation](self)).
///
/// [`prior`](Self::prior) adds the weights β ~ N(0, 1) and returns the function's values at the inputs;
/// [`HsgpPrior::predict`] evaluates the same function, with the same weights, at new inputs.
///
/// Riutort-Mayol et al. recommend c ≥ 1.2 and, for the exponentiated quadratic kernel, c ≥ 3.2 ℓ / S and
/// m ≥ 1.75 c / (ℓ / S), with S the inputs' half range and ℓ the length scale (more for the Matérn kernels, whose spectra decay more slowly, e.g.
/// m ≥ 3.42 c / (ℓ / S) for Matérn 3/2). The approximation is worst near the box's edges, where every basis function
/// vanishes; a larger c moves the edges away from the data and needs a larger m for the same length scale.
///
/// ```
/// use keanu_distributions::KeanuRng;
/// use keanu_graph::GraphBuilder;
/// use keanu_graph::gp::{Hsgp, Kernel};
/// use keanu_tensor::DoubleTensor;
///
/// let b = GraphBuilder::new();
/// let hsgp = Hsgp::new(&Kernel::exp_quad(&b, 1.0, 0.5), 20, 1.5)?;
/// let x = DoubleTensor::vector((0..50).map(|i| i as f64 / 49.0).collect());
/// let prior = hsgp.prior(x);
/// let at_new = prior.predict(DoubleTensor::vector(vec![0.25, 0.75]));
/// let y = b.gaussian(prior.f, 0.1);
/// let (y, beta, at_new) = (y.id(), prior.beta.id(), at_new.id());
/// let (graph, mut state) = b.build()?;
/// state.lazy_eval(&graph, &[y, at_new], &mut KeanuRng::seed_from_u64(1))?;
/// assert_eq!(state.f64(beta).unwrap().shape(), &[20]);
/// assert_eq!(state.f64(at_new).unwrap().shape(), &[2]);
/// # Ok::<(), keanu_graph::GraphError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Hsgp<'b> {
    builder: &'b GraphBuilder,
    form: Form,
    amplitude: DoubleVar<'b>,
    length_scale: DoubleVar<'b>,
    basis_functions: usize,
    proportion: f64,
    domain: Option<HsgpDomain>,
}

/// The box [centre − L, centre + L] (one centre and half-width L per input dimension) on which an [`Hsgp`]'s basis
/// functions live. Every input must lie in it.
#[derive(Clone, Debug, PartialEq)]
pub struct HsgpDomain {
    centre: Vec<f64>,
    half_width: Vec<f64>,
}

impl HsgpDomain {
    /// The box with these centres and half-widths, one per input dimension.
    ///
    /// # Errors
    ///
    /// [`GraphError::InvalidNode`] if the two have different lengths, are empty, or a centre is not finite or a
    /// half-width not finite and positive.
    pub fn new(centre: Vec<f64>, half_width: Vec<f64>) -> Result<Self> {
        if centre.is_empty() || centre.len() != half_width.len() {
            return Err(invalid(format!(
                "the domain needs one centre and one half-width per input dimension, not {} and {}",
                centre.len(),
                half_width.len()
            )));
        }
        if let Some(c) = centre.iter().find(|c| !c.is_finite()) {
            return Err(invalid(format!("a domain centre must be finite, not {c}")));
        }
        if let Some(l) = half_width.iter().find(|l| !(l.is_finite() && **l > 0.0)) {
            return Err(invalid(format!(
                "a domain half-width must be finite and positive, not {l}"
            )));
        }
        Ok(Self { centre, half_width })
    }

    /// The box that extends the range of inputs `x` (`[n]`, or `[n, d]` with one row per point) by the factor `c` in
    /// every dimension: centred on the midpoint of the inputs' range, with half-width L = c · S where S is half the
    /// range (Riutort-Mayol et al.'s L = c · max |x − centre|).
    ///
    /// # Errors
    ///
    /// [`GraphError::InvalidNode`] if `x` is not a non-empty vector or matrix of finite values, if `c` is not finite
    /// and at least 1, or if the inputs have no spread in some dimension (give the domain explicitly then).
    pub fn from_inputs(x: &DoubleTensor, c: f64) -> Result<Self> {
        check_proportion(c)?;
        let (n, d) = input_shape(x)?;
        if n == 0 {
            return Err(invalid("the inputs have no points".into()));
        }
        let data = x.as_slice();
        if let Some(v) = data.iter().find(|v| !v.is_finite()) {
            return Err(invalid(format!("the inputs must be finite, not {v}")));
        }
        let mut centre = Vec::with_capacity(d);
        let mut half_width = Vec::with_capacity(d);
        for k in 0..d {
            let column = data.iter().skip(k).step_by(d);
            let (lo, hi) = column.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
            let half_range = (hi - lo) / 2.0;
            if half_range <= 0.0 {
                return Err(invalid(format!(
                    "the inputs have no spread in dimension {k}, so the domain must be given explicitly"
                )));
            }
            centre.push(lo + half_range);
            half_width.push(c * half_range);
        }
        Self::new(centre, half_width)
    }

    /// The centre of each dimension.
    pub fn centre(&self) -> &[f64] {
        &self.centre
    }

    /// The half-width L of each dimension.
    pub fn half_width(&self) -> &[f64] {
        &self.half_width
    }

    /// The number of input dimensions.
    pub fn dimensions(&self) -> usize {
        self.centre.len()
    }
}

/// What [`Hsgp::prior`] adds to the graph: the function at the inputs, its weights, and what is needed to evaluate
/// the same function elsewhere.
#[derive(Clone, Debug)]
pub struct HsgpPrior<'b> {
    /// The function's values at the inputs, `[n]`.
    pub f: DoubleVar<'b>,
    /// The basis functions' weights β ~ N(0, 1), `[M]` with M = m^d: the latent variable to sample.
    pub beta: DoubleVar<'b>,
    /// The kernel's spectral density at each basis function's frequency, S(√λⱼ), `[M]`.
    pub spectral_density: DoubleVar<'b>,
    builder: &'b GraphBuilder,
    weights: DoubleVar<'b>,
    domain: Option<HsgpDomain>,
    basis_functions: usize,
}

impl<'b> Hsgp<'b> {
    /// An HSGP for `kernel` with `m` basis functions per input dimension on the box that extends the inputs' range
    /// by the factor `c` (see [`HsgpDomain::from_inputs`]). The kernel's amplitude and length scale (a scalar, or
    /// one per input dimension) may be latent variables.
    ///
    /// # Errors
    ///
    /// [`GraphError::InvalidNode`] for a kernel without a spectral density here (only
    /// [`exp_quad`](Kernel::exp_quad), [`matern12`](Kernel::matern12), [`matern32`](Kernel::matern32) and
    /// [`matern52`](Kernel::matern52) have one; for a sum of kernels, add one HSGP per term), for `m` = 0, and for
    /// `c` not finite or below 1.
    pub fn new(kernel: &Kernel<'b>, m: usize, c: f64) -> Result<Self> {
        let Kind::Stationary {
            form,
            amplitude,
            length_scale,
        } = kernel.kind
        else {
            let which = match &kernel.kind {
                Kind::RationalQuadratic { .. } => "the rational quadratic kernel",
                Kind::Periodic { .. } => "the periodic kernel",
                Kind::Linear { .. } => "the linear kernel",
                Kind::Sum(..) => "a sum of kernels (add one HSGP per term instead)",
                Kind::Product(..) => "a product of kernels",
                Kind::Stationary { .. } => unreachable!("handled above"),
            };
            return Err(invalid(format!(
                "{which} has no spectral density for the HSGP approximation; use exp_quad, matern12, matern32 or \
                 matern52"
            )));
        };
        if m == 0 {
            return Err(invalid("needs at least one basis function".into()));
        }
        check_proportion(c)?;
        Ok(Self {
            builder: kernel.builder,
            form,
            amplitude,
            length_scale,
            basis_functions: m,
            proportion: c,
            domain: None,
        })
    }

    /// With this domain instead of the one [`prior`](Self::prior) derives from its inputs and `c`: for instance to
    /// predict beyond the inputs' range, or to share basis functions between models.
    pub fn with_domain(mut self, domain: HsgpDomain) -> Self {
        self.domain = Some(domain);
        self
    }

    /// The number of basis functions per input dimension, m.
    pub fn basis_functions(&self) -> usize {
        self.basis_functions
    }

    /// The factor c by which the domain extends the inputs' range.
    pub fn proportion(&self) -> f64 {
        self.proportion
    }

    /// The domain set by [`with_domain`](Self::with_domain), if any.
    pub fn domain(&self) -> Option<&HsgpDomain> {
        self.domain.as_ref()
    }

    /// Adds the weights β ~ N(0, 1) (`[m^d]`) and returns the approximate GP's values at inputs `x` (`[n]`, or
    /// `[n, d]` with one row per point) with them.
    ///
    /// Inputs of the wrong rank, a length scale with the wrong number of elements, a domain of the wrong dimension
    /// or one that doesn't contain the inputs, and more than [`MAX_BASIS_FUNCTIONS`] basis functions are recorded
    /// as build errors, so [`GraphBuilder::build`] returns them.
    pub fn prior(&self, x: impl Into<DoubleTensor>) -> HsgpPrior<'b> {
        let b = self.builder;
        let x = x.into();
        let failed = || HsgpPrior {
            f: b.scalar(0.0),
            beta: b.scalar(0.0),
            spectral_density: b.scalar(0.0),
            builder: b,
            weights: b.scalar(0.0),
            domain: None,
            basis_functions: self.basis_functions,
        };
        let domain = match &self.domain {
            Some(domain) => domain.clone(),
            None => match HsgpDomain::from_inputs(&x, self.proportion) {
                Ok(domain) => domain,
                Err(e) => {
                    b.record(e);
                    return failed();
                }
            },
        };
        let d = domain.dimensions();
        let Some(count) = basis_count(self.basis_functions, d) else {
            fail(
                b,
                HSGP,
                format!(
                    "{} basis functions in each of {d} dimensions is more than {MAX_BASIS_FUNCTIONS}",
                    self.basis_functions
                ),
            );
            return failed();
        };
        let length_shape = self.length_scale.shape();
        if !(length_shape.is_empty() || length_shape == [d]) {
            fail(
                b,
                HSGP,
                format!(
                    "the length scale must be a scalar or one element per input dimension ([{d}]), not \
                     {length_shape:?}"
                ),
            );
            return failed();
        }
        if !self.amplitude.shape().is_empty() {
            fail(
                b,
                HSGP,
                format!(
                    "the amplitude must be a scalar, not {:?}",
                    self.amplitude.shape()
                ),
            );
            return failed();
        }
        let phi = match basis(&domain, self.basis_functions, &x) {
            Ok(phi) => phi,
            Err(e) => {
                b.record(e);
                return failed();
            }
        };
        let (spectral_density, sqrt_spectral_density) = self.spectral(&domain, count);
        let beta = b.gaussian_with_shape(&[count], 0.0, 1.0);
        let weights = sqrt_spectral_density * beta;
        let f = project(b, phi, weights, count);
        HsgpPrior {
            f,
            beta,
            spectral_density,
            builder: b,
            weights,
            domain: Some(domain),
            basis_functions: self.basis_functions,
        }
    }

    /// The spectral density S and its square root at the basis functions' frequencies, `[M]` each, computed in log
    /// space: for d dimensions, with q = Σ_k (ℓ_k ω_k)²,
    ///
    /// - exponentiated quadratic: S(ω) = σ² (2π)^(d/2) Π ℓ_k exp(−q/2);
    /// - Matérn ν: S(ω) = σ² 2^d π^(d/2) Γ(ν + d/2) (2ν)^ν / Γ(ν) · Π ℓ_k · (2ν + q)^(−(ν + d/2)).
    fn spectral(&self, domain: &HsgpDomain, count: usize) -> (DoubleVar<'b>, DoubleVar<'b>) {
        let b = self.builder;
        let d = domain.dimensions();
        let df = d as f64;
        let frequencies = frequencies(domain, self.basis_functions, count);
        let length_scale = self.length_scale.broadcast_to(&[d]);
        let scaled = b.constant(frequencies) * length_scale;
        let q = (scaled * scaled).sum_axes(&[-1]);
        let log_lengths = length_scale.ln().sum();
        let log_density = match self.form {
            Form::ExpQuad => q * -0.5 + log_lengths + 0.5 * df * (2.0 * PI).ln(),
            Form::Matern12 | Form::Matern32 | Form::Matern52 => {
                let nu = match self.form {
                    Form::Matern12 => 0.5,
                    Form::Matern32 => 1.5,
                    _ => 2.5,
                };
                let constant =
                    df * std::f64::consts::LN_2 + 0.5 * df * PI.ln() + ln_gamma(nu + df / 2.0)
                        - ln_gamma(nu)
                        + nu * (2.0 * nu).ln();
                (q + 2.0 * nu).ln() * -(nu + df / 2.0) + log_lengths + constant
            }
        };
        let amplitude = self.amplitude;
        (
            log_density.exp() * (amplitude * amplitude),
            (log_density * 0.5).exp() * amplitude,
        )
    }
}

impl<'b> HsgpPrior<'b> {
    /// The approximate GP's values at new inputs `x_new` (`[m]` or `[m, d]`), with the same weights as
    /// [`f`](Self::f): draws of the weights give draws of the function there. The inputs must lie in the
    /// [domain](Self::domain), or the build fails.
    pub fn predict(&self, x_new: impl Into<DoubleTensor>) -> DoubleVar<'b> {
        let b = self.builder;
        let Some(domain) = &self.domain else {
            return fail(b, HSGP, "the HSGP prior failed to build".into());
        };
        match basis(domain, self.basis_functions, &x_new.into()) {
            Ok(phi) => project(b, phi, self.weights, self.weights.shape()[0]),
            Err(e) => {
                b.record(e);
                b.scalar(0.0)
            }
        }
    }

    /// The covariance matrix the approximation implies between inputs `x1` and `x2`, Φ(x1) diag(S) Φ(x2)ᵀ,
    /// `[n1, n2]`: the exact kernel's covariance up to the approximation error.
    pub fn covariance(
        &self,
        x1: impl Into<DoubleTensor>,
        x2: impl Into<DoubleTensor>,
    ) -> DoubleVar<'b> {
        let b = self.builder;
        let Some(domain) = &self.domain else {
            return fail(b, HSGP, "the HSGP prior failed to build".into());
        };
        let m = self.basis_functions;
        match (basis(domain, m, &x1.into()), basis(domain, m, &x2.into())) {
            (Ok(phi1), Ok(phi2)) => (b.constant(phi1) * self.spectral_density).matmul_transposed(
                b.constant(phi2),
                false,
                true,
            ),
            (Err(e), _) | (_, Err(e)) => {
                b.record(e);
                b.scalar(0.0)
            }
        }
    }

    /// The basis functions φⱼ at inputs `x` (`[n]` or `[n, d]`), `[n, M]`: the constant matrix Φ in f = Φ (√S ⊙ β).
    ///
    /// # Errors
    ///
    /// [`GraphError::InvalidNode`] if the inputs have the wrong rank or dimension, or lie outside the domain.
    pub fn basis(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        match &self.domain {
            Some(domain) => basis(domain, self.basis_functions, x),
            None => Err(invalid("the HSGP prior failed to build".into())),
        }
    }

    /// The domain of the basis functions (None if the prior failed to build).
    pub fn domain(&self) -> Option<&HsgpDomain> {
        self.domain.as_ref()
    }
}

fn invalid(message: String) -> GraphError {
    GraphError::InvalidNode { op: HSGP, message }
}

fn check_proportion(c: f64) -> Result<()> {
    if c.is_finite() && c >= 1.0 {
        Ok(())
    } else {
        Err(invalid(format!(
            "the domain's extension factor c must be finite and at least 1, not {c}"
        )))
    }
}

/// m^d, if it is at most [`MAX_BASIS_FUNCTIONS`].
fn basis_count(m: usize, d: usize) -> Option<usize> {
    let d = u32::try_from(d).ok()?;
    m.checked_pow(d)
        .filter(|&count| count <= MAX_BASIS_FUNCTIONS)
}

/// (points, dimensions) of inputs `[n]` or `[n, d]`.
fn input_shape(x: &DoubleTensor) -> Result<(usize, usize)> {
    match *x.shape() {
        [n] => Ok((n, 1)),
        [n, d] if d > 0 => Ok((n, d)),
        ref shape => Err(invalid(format!(
            "the inputs must be a vector [n] or a matrix [n, dimensions], not {shape:?}"
        ))),
    }
}

/// The index of each dimension's basis function for basis function `j` of the tensor grid (row-major, the last
/// dimension fastest), zero-based.
fn grid_index(mut j: usize, m: usize, d: usize, index: &mut [usize]) {
    for k in (0..d).rev() {
        index[k] = j % m;
        j /= m;
    }
}

/// √λ for each basis function and dimension, `[M, d]`: π (j_k + 1) / (2 L_k).
fn frequencies(domain: &HsgpDomain, m: usize, count: usize) -> DoubleTensor {
    let d = domain.dimensions();
    let mut index = vec![0; d];
    let mut data = Vec::with_capacity(count * d);
    for j in 0..count {
        grid_index(j, m, d, &mut index);
        for (k, &i) in index.iter().enumerate() {
            data.push(PI * (i + 1) as f64 / (2.0 * domain.half_width[k]));
        }
    }
    DoubleTensor::new(&[count, d], data).expect("count × d elements")
}

/// Φ at inputs `x`, `[n, M]`: Φ_ij = Π_k L_k^(−1/2) sin(π (j_k + 1) (x_ik − centre_k + L_k) / (2 L_k)).
fn basis(domain: &HsgpDomain, m: usize, x: &DoubleTensor) -> Result<DoubleTensor> {
    let (n, d) = input_shape(x)?;
    if d != domain.dimensions() {
        return Err(invalid(format!(
            "the inputs have {d} dimensions but the domain has {}",
            domain.dimensions()
        )));
    }
    let count = basis_count(m, d).ok_or_else(|| invalid("too many basis functions".to_string()))?;
    let data = x.as_slice();
    // each dimension's m functions at each point, [n, d, m]
    let mut factors = vec![0.0; n * d * m];
    for i in 0..n {
        for k in 0..d {
            let (centre, l) = (domain.centre[k], domain.half_width[k]);
            let v = data[i * d + k];
            // a tolerance for inputs exactly on the edge, where every function is 0 anyway
            if !(v.is_finite() && (v - centre).abs() <= l * (1.0 + 1e-12)) {
                return Err(invalid(format!(
                    "the input {v} in dimension {k} is outside the domain [{}, {}]",
                    centre - l,
                    centre + l
                )));
            }
            let scale = l.sqrt().recip();
            let angle = PI * (v - centre + l) / (2.0 * l);
            for j in 0..m {
                factors[(i * d + k) * m + j] = scale * (angle * (j + 1) as f64).sin();
            }
        }
    }
    let mut index = vec![0; d];
    let mut phi = vec![0.0; n * count];
    for j in 0..count {
        grid_index(j, m, d, &mut index);
        for i in 0..n {
            phi[i * count + j] = index
                .iter()
                .enumerate()
                .map(|(k, &jk)| factors[(i * d + k) * m + jk])
                .product();
        }
    }
    Ok(DoubleTensor::new(&[n, count], phi)?)
}

/// Φ w as a vector `[n]`.
fn project<'b>(
    b: &'b GraphBuilder,
    phi: DoubleTensor,
    weights: DoubleVar<'b>,
    count: usize,
) -> DoubleVar<'b> {
    let n = phi.shape()[0];
    b.constant(phi)
        .matmul(weights.reshape(&[count as isize, 1]))
        .reshape(&[n as isize])
}
