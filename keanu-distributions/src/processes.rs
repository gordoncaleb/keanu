//! Distributions over whole time series defined by a recursion: the Gaussian random walk, the autoregressive
//! process AR(p) and the GARCH(1,1) volatility process, following PyMC's `GaussianRandomWalk`, `AR` and `GARCH11`.
//!
//! Each is a distribution over the last axis of its value, `[..., T]` (time last, as in PyMC), whose leading axes
//! are a batch of independent series; each parameter is a scalar or broadcasts to the batch shape (the AR
//! coefficients have a trailing axis of lags). The log density is the sum of each step's Gaussian conditional
//! density given the steps before it, the first steps (one for the random walk and GARCH, p for AR(p)) having an
//! initial distribution of their own.
//!
//! Their gradients are written out by hand (reverse mode over the series), and their Hessian-vector products are
//! that same code run on dual numbers (forward mode over the reverse pass), so both are exact and cost O(T) per
//! direction.

use crate::error::{DistributionError, Result, invalid_parameter};
use crate::multivariate::sum_over_batch;
use crate::param::{IntoParam, Param};
use crate::rng::KeanuRng;
use keanu_tensor::{DoubleTensor, broadcast_shape};
use std::f64::consts::PI;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// The arithmetic the log densities' gradients are written in: plain numbers for the gradient, and dual numbers
/// (a value and one tangent) for its directional derivatives.
trait Real:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn constant(v: f64) -> Self;
    fn ln(self) -> Self;
}

impl Real for f64 {
    fn constant(v: f64) -> Self {
        v
    }
    fn ln(self) -> Self {
        f64::ln(self)
    }
}

/// A number with one tangent.
#[derive(Debug, Clone, Copy)]
struct Dual {
    v: f64,
    d: f64,
}

impl Add for Dual {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self {
            v: self.v + o.v,
            d: self.d + o.d,
        }
    }
}

impl Sub for Dual {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self {
            v: self.v - o.v,
            d: self.d - o.d,
        }
    }
}

impl Mul for Dual {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self {
            v: self.v * o.v,
            d: self.d * o.v + self.v * o.d,
        }
    }
}

impl Div for Dual {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        let v = self.v / o.v;
        Self {
            v,
            d: (self.d - v * o.d) / o.v,
        }
    }
}

impl Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            d: -self.d,
        }
    }
}

impl Real for Dual {
    fn constant(v: f64) -> Self {
        Self { v, d: 0.0 }
    }
    fn ln(self) -> Self {
        Self {
            v: self.v.ln(),
            d: self.d / self.v,
        }
    }
}

/// One series' log density and its gradient with respect to the series and each parameter (each parameter a slice:
/// one number, or the AR coefficients).
struct SeriesGradient<S> {
    log_prob: S,
    x: Vec<S>,
    parameters: Vec<Vec<S>>,
}

/// A process: its log density's gradient for one series, and whether its parameters are valid.
trait Process {
    /// The log density of the series `x` and its gradient, given each parameter's values for this series.
    fn gradient<S: Real>(&self, x: &[S], parameters: &[&[S]]) -> SeriesGradient<S>;
    /// None if the parameters are valid, else the log density to report (NaN or -infinity).
    fn invalid(&self, parameters: &[&[f64]]) -> Option<f64>;
}

/// ½ ln 2π.
fn half_ln_2pi() -> f64 {
    0.5 * (2.0 * PI).ln()
}

/// The autoregressive recursion x_t ~ N(c + Σᵢ ρᵢ x_{t−i}, σ) after the first p values, which are independent
/// N(μ₀, σ₀). Parameters: c, ρ (p of them), σ, μ₀, σ₀.
struct ArKernel;

impl Process for ArKernel {
    fn gradient<S: Real>(&self, x: &[S], parameters: &[&[S]]) -> SeriesGradient<S> {
        let (c, rho, sigma, m0, s0) = (
            parameters[0][0],
            parameters[1],
            parameters[2][0],
            parameters[3][0],
            parameters[4][0],
        );
        let p = rho.len();
        let zero = S::constant(0.0);
        let half = S::constant(0.5);
        let normaliser = S::constant(half_ln_2pi());
        let mut log_prob = zero;
        let mut dx = vec![zero; x.len()];
        let (mut dc, mut dsigma, mut dm0, mut ds0) = (zero, zero, zero, zero);
        let mut drho = vec![zero; p];
        let (ln_s0, ln_sigma) = (s0.ln(), sigma.ln());
        for (i, &xi) in x.iter().enumerate().take(p) {
            let z = (xi - m0) / s0;
            log_prob = log_prob - half * z * z - ln_s0 - normaliser;
            let g = z / s0;
            dx[i] = dx[i] - g;
            dm0 = dm0 + g;
            ds0 = ds0 + (z * z - S::constant(1.0)) / s0;
        }
        for t in p..x.len() {
            let mut prediction = c;
            for (j, &r) in rho.iter().enumerate() {
                prediction = prediction + r * x[t - 1 - j];
            }
            let e = (x[t] - prediction) / sigma;
            log_prob = log_prob - half * e * e - ln_sigma - normaliser;
            let g = e / sigma;
            dx[t] = dx[t] - g;
            for (j, &r) in rho.iter().enumerate() {
                dx[t - 1 - j] = dx[t - 1 - j] + g * r;
                drho[j] = drho[j] + g * x[t - 1 - j];
            }
            dc = dc + g;
            dsigma = dsigma + (e * e - S::constant(1.0)) / sigma;
        }
        SeriesGradient {
            log_prob,
            x: dx,
            parameters: vec![vec![dc], drho, vec![dsigma], vec![dm0], vec![ds0]],
        }
    }

    fn invalid(&self, parameters: &[&[f64]]) -> Option<f64> {
        let (sigma, s0) = (parameters[2][0], parameters[4][0]);
        (sigma.is_nan() || sigma <= 0.0 || s0.is_nan() || s0 <= 0.0).then_some(f64::NAN)
    }
}

/// The Gaussian random walk: AR(1) with ρ = 1 and the drift as its constant. Parameters: μ, σ, μ₀, σ₀.
struct RandomWalkKernel;

impl Process for RandomWalkKernel {
    fn gradient<S: Real>(&self, x: &[S], parameters: &[&[S]]) -> SeriesGradient<S> {
        let one = [S::constant(1.0)];
        let ar = ArKernel.gradient(
            x,
            &[
                parameters[0],
                &one,
                parameters[1],
                parameters[2],
                parameters[3],
            ],
        );
        let mut p = ar.parameters.into_iter();
        let mu = p.next().unwrap_or_default();
        let _rho = p.next();
        SeriesGradient {
            log_prob: ar.log_prob,
            x: ar.x,
            parameters: std::iter::once(mu).chain(p).collect(),
        }
    }

    fn invalid(&self, parameters: &[&[f64]]) -> Option<f64> {
        let (sigma, s0) = (parameters[1][0], parameters[3][0]);
        (sigma.is_nan() || sigma <= 0.0 || s0.is_nan() || s0 <= 0.0).then_some(f64::NAN)
    }
}

/// GARCH(1,1): r_t ~ N(0, σ_t) with σ_0 the initial volatility and σ_t² = ω + α r_{t−1}² + β σ_{t−1}².
/// Parameters: ω, α, β, σ_0.
struct GarchKernel;

impl Process for GarchKernel {
    fn gradient<S: Real>(&self, r: &[S], parameters: &[&[S]]) -> SeriesGradient<S> {
        let (omega, alpha, beta, v0) = (
            parameters[0][0],
            parameters[1][0],
            parameters[2][0],
            parameters[3][0],
        );
        let zero = S::constant(0.0);
        let (half, two) = (S::constant(0.5), S::constant(2.0));
        let normaliser = S::constant(half_ln_2pi());
        let t_len = r.len();
        // the variances σ_t²
        let mut s = Vec::with_capacity(t_len);
        for t in 0..t_len {
            s.push(if t == 0 {
                v0 * v0
            } else {
                omega + alpha * r[t - 1] * r[t - 1] + beta * s[t - 1]
            });
        }
        let mut log_prob = zero;
        for t in 0..t_len {
            log_prob = log_prob - normaliser - half * (s[t].ln() + r[t] * r[t] / s[t]);
        }
        // adjoints of the variances, backwards: ŝ_t = ∂/∂σ_t² + β ŝ_{t+1}
        let mut dr = vec![zero; t_len];
        let (mut domega, mut dalpha, mut dbeta) = (zero, zero, zero);
        let mut next = zero;
        for t in (0..t_len).rev() {
            let direct = -half * (S::constant(1.0) / s[t] - r[t] * r[t] / (s[t] * s[t]));
            dr[t] = -r[t] / s[t] + two * alpha * r[t] * next;
            let adjoint = direct + beta * next;
            if t > 0 {
                domega = domega + adjoint;
                dalpha = dalpha + adjoint * r[t - 1] * r[t - 1];
                dbeta = dbeta + adjoint * s[t - 1];
            }
            next = adjoint;
        }
        let dv0 = if t_len > 0 { two * v0 * next } else { zero };
        SeriesGradient {
            log_prob,
            x: dr,
            parameters: vec![vec![domega], vec![dalpha], vec![dbeta], vec![dv0]],
        }
    }

    fn invalid(&self, parameters: &[&[f64]]) -> Option<f64> {
        garch_invalid(
            parameters[0][0],
            parameters[1][0],
            parameters[2][0],
            parameters[3][0],
        )
        .map(|_| f64::NEG_INFINITY)
    }
}

/// Why GARCH(1,1) parameters are invalid, if they are: ω > 0, α ≥ 0, β ≥ 0, α + β < 1 (covariance stationarity)
/// and an initial volatility σ_0 > 0, all finite.
fn garch_invalid(omega: f64, alpha: f64, beta: f64, v0: f64) -> Option<String> {
    if !(omega.is_finite() && omega > 0.0) {
        Some(format!("omega must be positive, got {omega}"))
    } else if !(alpha.is_finite() && alpha >= 0.0 && beta.is_finite() && beta >= 0.0) {
        Some(format!(
            "alpha and beta must be non-negative, got {alpha} and {beta}"
        ))
    } else if alpha + beta >= 1.0 {
        Some(format!(
            "alpha + beta must be less than 1 for a stationary process, got {}",
            alpha + beta
        ))
    } else if !(v0.is_finite() && v0 > 0.0) {
        Some(format!("the initial volatility must be positive, got {v0}"))
    } else {
        None
    }
}

/// A batch of series of `steps` values each: the batch shape and every parameter spread to it (with its trailing
/// lag axis, for the AR coefficients).
struct Layout {
    batch: Vec<usize>,
    series: usize,
    steps: usize,
    /// Each parameter's values at every series, flattened [series, event], and its event length.
    parameters: Vec<(Vec<f64>, usize)>,
    /// Whether each parameter has an event axis.
    events: Vec<bool>,
}

impl Layout {
    /// The layout of a value `shape` ([..., T], T > 0) with `parameters` (name, tensor, and whether its last axis is
    /// an event axis rather than a batch axis).
    fn new(
        what: &str,
        shape: &[usize],
        parameters: &[(&str, &DoubleTensor, bool)],
    ) -> Result<Self> {
        let Some((&steps, batch)) = shape.split_last().filter(|(t, _)| **t > 0) else {
            return Err(DistributionError::InvalidValue(format!(
                "a {what} needs a shape [..., T] with T > 0 time steps, not {shape:?}"
            )));
        };
        let series = batch.iter().product();
        let mut spread = Vec::with_capacity(parameters.len());
        for &(name, p, event) in parameters {
            let (p_batch, event_len) = if event {
                match p.shape().split_last() {
                    Some((&e, b)) if e > 0 => (b, e),
                    _ => {
                        return invalid_parameter(format!(
                            "{name} must have a trailing axis of at least one lag, not shape {:?}",
                            p.shape()
                        ));
                    }
                }
            } else {
                (p.shape(), 1)
            };
            if broadcast_shape(batch, p_batch)?.as_slice() != batch {
                return invalid_parameter(format!(
                    "{name} of shape {:?} does not broadcast to the batch shape {batch:?}",
                    p.shape()
                ));
            }
            let mut full = batch.to_vec();
            if event {
                full.push(event_len);
            }
            spread.push((p.broadcast_to(&full)?.into_vec(), event_len));
        }
        Ok(Self {
            batch: batch.to_vec(),
            series,
            steps,
            parameters: spread,
            events: parameters.iter().map(|p| p.2).collect(),
        })
    }

    /// The parameters of series `i`.
    fn at(&self, i: usize) -> Vec<&[f64]> {
        self.parameters
            .iter()
            .map(|(values, e)| &values[i * e..(i + 1) * e])
            .collect()
    }

    /// The full shape of parameter `k`'s spread values.
    fn full_shape(&self, k: usize) -> Vec<usize> {
        let mut shape = self.batch.clone();
        if self.events[k] {
            shape.push(self.parameters[k].1);
        }
        shape
    }
}

/// The log density, gradient and Hessian-vector products of a process over a batch of series.
struct Batch<'p, P: Process> {
    process: P,
    what: &'static str,
    parameters: Vec<(&'static str, &'p DoubleTensor, bool)>,
}

impl<P: Process> Batch<'_, P> {
    fn layout(&self, shape: &[usize]) -> Result<Layout> {
        Layout::new(self.what, shape, &self.parameters)
    }

    fn log_prob(&self, x: &DoubleTensor) -> Result<f64> {
        let layout = self.layout(x.shape())?;
        let mut total = 0.0;
        for (i, series) in x.as_slice().chunks(layout.steps).enumerate() {
            let parameters = layout.at(i);
            if let Some(invalid) = self.process.invalid(&parameters) {
                return Ok(invalid);
            }
            total += self.process.gradient(series, &parameters).log_prob;
        }
        Ok(total)
    }

    /// The gradient with respect to x and each parameter, each shaped like it.
    fn d_log_prob(&self, x: &DoubleTensor) -> Result<(DoubleTensor, Vec<DoubleTensor>)> {
        let layout = self.layout(x.shape())?;
        let mut dx = Vec::with_capacity(x.len());
        let mut dp: Vec<Vec<f64>> = layout
            .parameters
            .iter()
            .map(|(v, _)| Vec::with_capacity(v.len()))
            .collect();
        for (i, series) in x.as_slice().chunks(layout.steps).enumerate() {
            let g = self.process.gradient(series, &layout.at(i));
            dx.extend(g.x);
            for (d, g) in dp.iter_mut().zip(g.parameters) {
                d.extend(g);
            }
        }
        let gradients = dp
            .into_iter()
            .enumerate()
            .map(|(k, d)| {
                sum_over_batch(
                    &DoubleTensor::new(&layout.full_shape(k), d)?,
                    self.parameters[k].1.shape(),
                )
            })
            .collect::<Result<_>>()?;
        Ok((DoubleTensor::new(x.shape(), dx)?, gradients))
    }

    /// Hessian-vector products along `b` directions given as tangents of x ([b, x shape]) and of each parameter
    /// ([b, parameter shape]), None for zero: each output shaped [b, input shape].
    fn hvp(
        &self,
        x: &DoubleTensor,
        x_tangent: Option<&DoubleTensor>,
        parameter_tangents: &[Option<&DoubleTensor>],
        b: usize,
    ) -> Result<(DoubleTensor, Vec<DoubleTensor>)> {
        let layout = self.layout(x.shape())?;
        if parameter_tangents.len() != self.parameters.len() {
            return invalid_parameter(format!(
                "a {} has {} parameters, not {} parameter tangents",
                self.what,
                self.parameters.len(),
                parameter_tangents.len()
            ));
        }
        let tx = tangent(x_tangent, b, x.shape(), x.shape(), "the series")?;
        let tp: Vec<Option<Vec<f64>>> = parameter_tangents
            .iter()
            .enumerate()
            .map(|(k, t)| {
                let (name, p, _) = self.parameters[k];
                tangent(*t, b, p.shape(), &layout.full_shape(k), name)
            })
            .collect::<Result<_>>()?;
        let (steps, len) = (layout.steps, x.len());
        let mut hx = vec![0.0; b * len];
        let mut hp: Vec<Vec<f64>> = layout
            .parameters
            .iter()
            .map(|(v, _)| vec![0.0; b * v.len()])
            .collect();
        for (i, series) in x.as_slice().chunks(steps).enumerate() {
            let parameters = layout.at(i);
            for k in 0..b {
                let xs: Vec<Dual> = series
                    .iter()
                    .enumerate()
                    .map(|(t, &v)| Dual {
                        v,
                        d: tx.as_ref().map_or(0.0, |tx| tx[k * len + i * steps + t]),
                    })
                    .collect();
                let ps: Vec<Vec<Dual>> = parameters
                    .iter()
                    .enumerate()
                    .map(|(q, values)| {
                        let (all, e) = (&layout.parameters[q].0, layout.parameters[q].1);
                        values
                            .iter()
                            .enumerate()
                            .map(|(j, &v)| Dual {
                                v,
                                d: tp[q].as_ref().map_or(0.0, |t| t[k * all.len() + i * e + j]),
                            })
                            .collect()
                    })
                    .collect();
                let refs: Vec<&[Dual]> = ps.iter().map(Vec::as_slice).collect();
                let g = self.process.gradient(&xs, &refs);
                for (t, d) in g.x.iter().enumerate() {
                    hx[k * len + i * steps + t] = d.d;
                }
                for (q, d) in g.parameters.iter().enumerate() {
                    let (all, e) = (layout.parameters[q].0.len(), layout.parameters[q].1);
                    for (j, d) in d.iter().enumerate() {
                        hp[q][k * all + i * e + j] = d.d;
                    }
                }
            }
        }
        let mut x_shape = vec![b];
        x_shape.extend_from_slice(x.shape());
        let products = hp
            .into_iter()
            .enumerate()
            .map(|(q, h)| {
                let full = layout.full_shape(q);
                let target = self.parameters[q].1.shape();
                let per = h.len() / b.max(1);
                let mut out = Vec::new();
                for k in 0..b {
                    let one = DoubleTensor::new(&full, h[k * per..(k + 1) * per].to_vec())?;
                    out.extend_from_slice(sum_over_batch(&one, target)?.as_slice());
                }
                let mut shape = vec![b];
                shape.extend_from_slice(target);
                Ok(DoubleTensor::new(&shape, out)?)
            })
            .collect::<Result<_>>()?;
        Ok((DoubleTensor::new(&x_shape, hx)?, products))
    }
}

/// A tangent of an input of shape `shape`, checked to be [b, shape...], broadcast to [b, full...] and flattened.
fn tangent(
    t: Option<&DoubleTensor>,
    b: usize,
    shape: &[usize],
    full: &[usize],
    what: &str,
) -> Result<Option<Vec<f64>>> {
    let Some(t) = t else {
        return Ok(None);
    };
    let mut expected = vec![b];
    expected.extend_from_slice(shape);
    if t.shape() != expected {
        return invalid_parameter(format!(
            "the tangent of {what} must have shape {expected:?}, not {:?}",
            t.shape()
        ));
    }
    // broadcast each direction's [shape] to [full]
    let per: usize = shape.iter().product();
    let mut out = Vec::with_capacity(b * full.iter().product::<usize>());
    for k in 0..b {
        let one = DoubleTensor::new(shape, t.as_slice()[k * per..(k + 1) * per].to_vec())?;
        out.extend(one.broadcast_to(full)?.into_vec());
    }
    Ok(Some(out))
}

/// Sampling a batch: the layout of `shape` and a draw of each series from its parameters.
fn sample_batch(
    layout: &Layout,
    mut draw: impl FnMut(&[&[f64]], &mut [f64], &mut KeanuRng) -> Result<()>,
    shape: &[usize],
    rng: &mut KeanuRng,
) -> Result<DoubleTensor> {
    let mut values = vec![0.0; layout.series * layout.steps];
    for (i, series) in values.chunks_mut(layout.steps).enumerate() {
        draw(&layout.at(i), series, rng)?;
    }
    Ok(DoubleTensor::new(shape, values)?)
}

/// A positive, finite scale, or an error naming it.
fn require_scale(name: &str, s: f64) -> Result<()> {
    if s.is_finite() && s > 0.0 {
        Ok(())
    } else {
        invalid_parameter(format!("{name} must be positive, got {s}"))
    }
}

macro_rules! gradient_struct {
    ($(#[$doc:meta])* $name:ident { $($field:ident: $field_doc:literal),+ }) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $name {
            /// With respect to the series.
            pub x: DoubleTensor,
            $(#[doc = $field_doc] pub $field: DoubleTensor,)+
        }

        impl $name {
            fn from_parts((x, parameters): (DoubleTensor, Vec<DoubleTensor>)) -> Self {
                let mut parameters = parameters.into_iter();
                Self {
                    x,
                    $($field: parameters.next().unwrap_or_else(|| DoubleTensor::scalar(0.0)),)+
                }
            }
        }
    };
}

/// The Gaussian random walk, as PyMC's `GaussianRandomWalk`: over the last axis of a value [..., T], the first
/// value is drawn from N(`init_mu`, `init_sigma`) and each later one is the one before it plus N(`mu`, `sigma`)
/// noise, so
///
/// `ln p(x) = ln N(x_0; μ₀, σ₀) + Σ_{t ≥ 1} ln N(x_t − x_{t−1}; μ, σ)`.
///
/// `mu` is the drift. The parameters are scalars or broadcast to the batch shape [...] (one per series). PyMC's
/// `steps` is T − 1 here: T counts every value, the initial one included. Scales that are not positive give a NaN
/// log density.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianRandomWalk<'a> {
    /// The drift μ of each step.
    pub mu: Param<'a>,
    /// The scale σ > 0 of each step's innovation.
    pub sigma: Param<'a>,
    /// The mean μ₀ of the first value.
    pub init_mu: Param<'a>,
    /// The scale σ₀ > 0 of the first value.
    pub init_sigma: Param<'a>,
}

gradient_struct!(
    /// Derivatives of [`GaussianRandomWalk::log_prob`] (each shaped like its input), or its Hessian-vector products
    /// along `b` directions (each shaped [b, input shape...]).
    GaussianRandomWalkGradient {
        mu: "With respect to `mu`.",
        sigma: "With respect to `sigma`.",
        init_mu: "With respect to `init_mu`.",
        init_sigma: "With respect to `init_sigma`."
    }
);

impl<'a> GaussianRandomWalk<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(
        mu: impl IntoParam<'a, f64>,
        sigma: impl IntoParam<'a, f64>,
        init_mu: impl IntoParam<'a, f64>,
        init_sigma: impl IntoParam<'a, f64>,
    ) -> Self {
        Self {
            mu: mu.into_param(),
            sigma: sigma.into_param(),
            init_mu: init_mu.into_param(),
            init_sigma: init_sigma.into_param(),
        }
    }

    fn batch(&self) -> Batch<'_, RandomWalkKernel> {
        Batch {
            process: RandomWalkKernel,
            what: "Gaussian random walk",
            parameters: vec![
                ("mu", &self.mu, false),
                ("sigma", &self.sigma, false),
                ("init_mu", &self.init_mu, false),
                ("init_sigma", &self.init_sigma, false),
            ],
        }
    }

    /// The total log density of every series in x ([..., T], T > 0), summed over the batch. An error if the shapes
    /// don't fit.
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<f64> {
        self.batch().log_prob(x)
    }

    /// The derivatives of [`Self::log_prob`] with respect to x and every parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<GaussianRandomWalkGradient> {
        Ok(GaussianRandomWalkGradient::from_parts(
            self.batch().d_log_prob(x)?,
        ))
    }

    /// Hessian-vector products of [`Self::log_prob`] along `b` directions, given as tangents of x ([b, x shape])
    /// and of each parameter (in the order of [`Self::new`], each [b, parameter shape]), None for zero: for each
    /// input u, Σᵥ ∂² log p / ∂u ∂v · v̇, shaped [b, input shape].
    ///
    /// # Errors
    ///
    /// An error if there aren't four parameter tangents or a tangent has the wrong shape.
    pub fn hvp(
        &self,
        x: &DoubleTensor,
        x_tangent: Option<&DoubleTensor>,
        parameter_tangents: &[Option<&DoubleTensor>],
        b: usize,
    ) -> Result<GaussianRandomWalkGradient> {
        Ok(GaussianRandomWalkGradient::from_parts(self.batch().hvp(
            x,
            x_tangent,
            parameter_tangents,
            b,
        )?))
    }

    /// Draws series of the given shape ([..., T]) by simulating the walk. An error unless the scales are positive
    /// and finite.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let layout = self.batch().layout(shape)?;
        sample_batch(
            &layout,
            |p, series, rng| {
                let (mu, sigma, m0, s0) = (p[0][0], p[1][0], p[2][0], p[3][0]);
                require_scale("sigma", sigma)?;
                require_scale("init_sigma", s0)?;
                let mut value = m0 + s0 * rng.next_gaussian();
                for (t, x) in series.iter_mut().enumerate() {
                    if t > 0 {
                        value += mu + sigma * rng.next_gaussian();
                    }
                    *x = value;
                }
                Ok(())
            },
            shape,
            rng,
        )
    }
}

/// The autoregressive process AR(p), as PyMC's `AR`: over the last axis of a value [..., T], the first p values
/// are independent draws from N(`init_mu`, `init_sigma`) and each later one is
///
/// `x_t ~ N(c + Σᵢ₌₁ᵖ ρᵢ x_{t−i}, σ)`,
///
/// with `constant` c (PyMC's `constant=True` puts it first in ρ; here it is a parameter of its own, 0 for none) and
/// coefficients `rho` [..., p], ρᵢ the coefficient of lag i. The order p is the length of the last axis of `rho`,
/// whose leading axes (like every other parameter's) broadcast to the batch shape. A series of T ≤ p values is all
/// initial values.
///
/// The process is conditioned on its first p values, which have the given initial distribution, rather than on
/// the stationary one; nothing requires the coefficients to be stationary. For a stationary AR(1), give it the
/// stationary initial distribution through its parameters: `init_mu` = c / (1 − ρ) and `init_sigma` =
/// σ / √(1 − ρ²) (deterministic nodes of ρ and σ in a graph). Scales that are not positive give a NaN log
/// density.
#[derive(Debug, Clone, PartialEq)]
pub struct Autoregressive<'a> {
    /// The constant c.
    pub constant: Param<'a>,
    /// The coefficients ρ, [..., p]: the last axis holds lags 1 to p.
    pub rho: Param<'a>,
    /// The scale σ > 0 of each step's innovation.
    pub sigma: Param<'a>,
    /// The mean μ₀ of each of the first p values.
    pub init_mu: Param<'a>,
    /// The scale σ₀ > 0 of each of the first p values.
    pub init_sigma: Param<'a>,
}

gradient_struct!(
    /// Derivatives of [`Autoregressive::log_prob`] (each shaped like its input), or its Hessian-vector products
    /// along `b` directions (each shaped [b, input shape...]).
    AutoregressiveGradient {
        constant: "With respect to `constant`.",
        rho: "With respect to `rho`.",
        sigma: "With respect to `sigma`.",
        init_mu: "With respect to `init_mu`.",
        init_sigma: "With respect to `init_sigma`."
    }
);

impl<'a> Autoregressive<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(
        constant: impl IntoParam<'a, f64>,
        rho: impl IntoParam<'a, f64>,
        sigma: impl IntoParam<'a, f64>,
        init_mu: impl IntoParam<'a, f64>,
        init_sigma: impl IntoParam<'a, f64>,
    ) -> Self {
        Self {
            constant: constant.into_param(),
            rho: rho.into_param(),
            sigma: sigma.into_param(),
            init_mu: init_mu.into_param(),
            init_sigma: init_sigma.into_param(),
        }
    }

    /// The order p: the length of the last axis of `rho` (an error if it has none, or it is empty).
    pub fn order(&self) -> Result<usize> {
        match self.rho.shape().last() {
            Some(&p) if p > 0 => Ok(p),
            _ => invalid_parameter(format!(
                "rho must have a trailing axis of at least one lag, not shape {:?}",
                self.rho.shape()
            )),
        }
    }

    fn batch(&self) -> Batch<'_, ArKernel> {
        Batch {
            process: ArKernel,
            what: "autoregressive process",
            parameters: vec![
                ("constant", &self.constant, false),
                ("rho", &self.rho, true),
                ("sigma", &self.sigma, false),
                ("init_mu", &self.init_mu, false),
                ("init_sigma", &self.init_sigma, false),
            ],
        }
    }

    /// The total log density of every series in x ([..., T], T > 0), summed over the batch. An error if the shapes
    /// don't fit.
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<f64> {
        self.batch().log_prob(x)
    }

    /// The derivatives of [`Self::log_prob`] with respect to x and every parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<AutoregressiveGradient> {
        Ok(AutoregressiveGradient::from_parts(
            self.batch().d_log_prob(x)?,
        ))
    }

    /// Hessian-vector products of [`Self::log_prob`], as for [`GaussianRandomWalk::hvp`] with five parameter
    /// tangents in the order of [`Self::new`].
    ///
    /// # Errors
    ///
    /// An error if there aren't five parameter tangents or a tangent has the wrong shape.
    pub fn hvp(
        &self,
        x: &DoubleTensor,
        x_tangent: Option<&DoubleTensor>,
        parameter_tangents: &[Option<&DoubleTensor>],
        b: usize,
    ) -> Result<AutoregressiveGradient> {
        Ok(AutoregressiveGradient::from_parts(self.batch().hvp(
            x,
            x_tangent,
            parameter_tangents,
            b,
        )?))
    }

    /// Draws series of the given shape ([..., T]) by simulating the recursion from the initial values. An error
    /// unless the scales are positive and finite.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let layout = self.batch().layout(shape)?;
        sample_batch(
            &layout,
            |p, series, rng| {
                let (c, rho, sigma, m0, s0) = (p[0][0], p[1], p[2][0], p[3][0], p[4][0]);
                require_scale("sigma", sigma)?;
                require_scale("init_sigma", s0)?;
                for t in 0..series.len() {
                    series[t] = if t < rho.len() {
                        m0 + s0 * rng.next_gaussian()
                    } else {
                        let mean: f64 = c + rho
                            .iter()
                            .enumerate()
                            .map(|(j, r)| r * series[t - 1 - j])
                            .sum::<f64>();
                        mean + sigma * rng.next_gaussian()
                    };
                }
                Ok(())
            },
            shape,
            rng,
        )
    }
}

/// The GARCH(1,1) volatility process, as PyMC's `GARCH11`: over the last axis of a value [..., T] (returns, say),
/// each value is r_t ~ N(0, σ_t) with σ_0 = `initial_vol` and
///
/// `σ_t² = ω + α r_{t−1}² + β σ_{t−1}²`.
///
/// The parameters are scalars or broadcast to the batch shape. They must satisfy ω > 0, α ≥ 0, β ≥ 0,
/// α + β < 1 (covariance stationarity, so the unconditional variance ω / (1 − α − β) exists) and
/// `initial_vol` > 0: parameters outside that region make the series impossible (log density −∞, so a sampler
/// rejects them, as for a hidden Markov model's probabilities off the simplex), and sampling them is an error. To
/// keep a gradient based sampler inside it, give α and β a joint prior on it: for example
/// (α, β, 1 − α − β) ~ Dirichlet, or a persistence α + β in (0, 1) and its split.
#[derive(Debug, Clone, PartialEq)]
pub struct Garch11<'a> {
    /// The constant ω > 0 of the variance recursion.
    pub omega: Param<'a>,
    /// The weight α ≥ 0 of the last squared value.
    pub alpha: Param<'a>,
    /// The weight β ≥ 0 of the last variance.
    pub beta: Param<'a>,
    /// The first value's volatility σ_0 > 0 (a standard deviation).
    pub initial_vol: Param<'a>,
}

gradient_struct!(
    /// Derivatives of [`Garch11::log_prob`] (each shaped like its input), or its Hessian-vector products along `b`
    /// directions (each shaped [b, input shape...]).
    Garch11Gradient {
        omega: "With respect to `omega`.",
        alpha: "With respect to `alpha`.",
        beta: "With respect to `beta`.",
        initial_vol: "With respect to `initial_vol`."
    }
);

impl<'a> Garch11<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(
        omega: impl IntoParam<'a, f64>,
        alpha: impl IntoParam<'a, f64>,
        beta: impl IntoParam<'a, f64>,
        initial_vol: impl IntoParam<'a, f64>,
    ) -> Self {
        Self {
            omega: omega.into_param(),
            alpha: alpha.into_param(),
            beta: beta.into_param(),
            initial_vol: initial_vol.into_param(),
        }
    }

    fn batch(&self) -> Batch<'_, GarchKernel> {
        Batch {
            process: GarchKernel,
            what: "GARCH(1,1) process",
            parameters: vec![
                ("omega", &self.omega, false),
                ("alpha", &self.alpha, false),
                ("beta", &self.beta, false),
                ("initial_vol", &self.initial_vol, false),
            ],
        }
    }

    /// The total log density of every series in x ([..., T], T > 0), summed over the batch: −∞ if the parameters
    /// of any series are outside the valid region (see the type's documentation). An error if the shapes don't fit.
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<f64> {
        self.batch().log_prob(x)
    }

    /// The derivatives of [`Self::log_prob`] with respect to x and every parameter (the formulas' values, whether
    /// or not the parameters are valid).
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<Garch11Gradient> {
        Ok(Garch11Gradient::from_parts(self.batch().d_log_prob(x)?))
    }

    /// Hessian-vector products of [`Self::log_prob`], as for [`GaussianRandomWalk::hvp`] with four parameter
    /// tangents in the order of [`Self::new`].
    ///
    /// # Errors
    ///
    /// An error if there aren't four parameter tangents or a tangent has the wrong shape.
    pub fn hvp(
        &self,
        x: &DoubleTensor,
        x_tangent: Option<&DoubleTensor>,
        parameter_tangents: &[Option<&DoubleTensor>],
        b: usize,
    ) -> Result<Garch11Gradient> {
        Ok(Garch11Gradient::from_parts(self.batch().hvp(
            x,
            x_tangent,
            parameter_tangents,
            b,
        )?))
    }

    /// Draws series of the given shape ([..., T]) by simulating the recursion. An error if the parameters are
    /// outside the valid region.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let layout = self.batch().layout(shape)?;
        sample_batch(
            &layout,
            |p, series, rng| {
                let (omega, alpha, beta, v0) = (p[0][0], p[1][0], p[2][0], p[3][0]);
                if let Some(message) = garch_invalid(omega, alpha, beta, v0) {
                    return invalid_parameter(message);
                }
                let mut variance = v0 * v0;
                for t in 0..series.len() {
                    if t > 0 {
                        variance = omega + alpha * series[t - 1] * series[t - 1] + beta * variance;
                    }
                    series[t] = variance.sqrt() * rng.next_gaussian();
                }
                Ok(())
            },
            shape,
            rng,
        )
    }
}
