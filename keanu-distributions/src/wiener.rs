//! The Wiener first-passage time distribution of the drift-diffusion model of two-choice response times.
//!
//! See [`Wiener`] for the density, its parameterisation and its accuracy, and [`sample_wiener`] for joint draws of
//! a response time and the boundary it ends at.

// the negated comparisons (`!(a > 0.0)`) are deliberate: they also catch NaN
#![allow(clippy::neg_cmp_op_on_partial_ord)]

mod full;
mod tails;

pub use full::{WienerFull, WienerFullGradient};

use crate::error::{Result, invalid_parameter};
use crate::jet::{ElementDerivatives, Jet};
use crate::param::{IntoParam, Param};
use crate::rng::KeanuRng;
use crate::util::{map, map_n, operand, sample_shape};
use keanu_tensor::special::{log_ndtr, ndtri};
use keanu_tensor::{DoubleTensor, IntTensor};
use std::f64::consts::PI;

/// ln √(2π).
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

/// The relative size, against the partial sum, below which the error bound of a truncated series must fall.
const SERIES_TOLERANCE: f64 = 1e-17;

/// ln(1/ε) for the absolute tolerance ε = 1e-15 at which the two series' term counts are compared to choose one.
const SELECTION_LOG_TOLERANCE: f64 = 34.538_776_394_910_684;

/// The most proposals a rejection sampler makes for one draw before giving up.
const MAX_PROPOSALS: usize = 10_000_000;

/// Wiener first-passage time distribution: the time at which a Wiener diffusion process with drift `v`, started at
/// a fraction `w` of the way between two absorbing boundaries `a` apart, first reaches the boundary given by
/// `boundary` (0 the lower, 1 the upper), plus a non-decision time `t0`; with across-trial drift variability `sv`,
/// the drift is itself N(v, sv²) on each trial. This is the drift-diffusion model of two-choice response times
/// (Ratcliff 1978), with Stan's `wiener` parameterisation (boundary separation α = a, non-decision time τ = t0,
/// relative starting point β = w, drift δ = v, and `wiener_full`'s sv), the process having unit diffusion
/// coefficient.
///
/// The value x is the response time; `boundary` is the response, an observed integer parameter rather than part of
/// the value, as in Stan, where each boundary has a density of its own. The density is defective: over x and both
/// boundaries it integrates to 1, and over x alone to the probability of reaching that boundary. Its density at the
/// lower boundary, with decision time t = x − t0 > 0, is (Navarro and Fuss 2009; Blurton et al. 2017)
///
/// ```text
/// p(t | a, v, w, sv) = 1/(a² √(1 + sv² t)) · exp((sv² a² w² − 2avw − v² t) / (2(1 + sv² t))) · f(t/a² | w)
/// ```
///
/// where f(u | w) is the density of the standard problem (a = 1, no drift) at the lower boundary, and the upper
/// boundary's density is the same with v ↦ −v and w ↦ 1 − w. With sv = 0 this is Navarro and Fuss's
/// (1/a²) e^{−avw − v²t/2} f(t/a² | w); the sv factor is the closed-form integral over the drift (Blurton,
/// Kesselmeier and Gondan 2017). f is computed from whichever of its two series needs fewer terms at u = t/a²
/// (Navarro and Fuss 2009; Gondan, Blurton and Kesselmeier 2014):
///
/// ```text
/// small time: f(u | w) = (2π u³)^{−1/2} Σ_{k ∈ ℤ} (w + 2k) e^{−(w + 2k)²/(2u)}
///                      = (2π u³)^{−1/2} w e^{−w²/(2u)} (1 + Σ_{k ≥ 1} e^{−2k(k − w)/u} (1 + e^{−4kw/u}
///                                                       + (2k/w)(e^{−4kw/u} − 1)))
/// large time: f(u | w) = π Σ_{k ≥ 1} k e^{−k²π²u/2} sin(kπw)
/// ```
///
/// both summed in logs relative to their leading term, so that neither underflows. The small-time series is an
/// alternating series in |w + 2k| whose terms decrease once |w + 2k| ≥ √u, so its truncation error is at most the
/// first omitted term (Gondan et al.'s bound); the large-time series' tail after K terms is at most
/// e^{−K²π²u/2}/(π u) for K ≥ 1/(π√u) (Navarro and Fuss's bound). Each is summed until its bound falls below 1e-17
/// of the partial sum, plus one more term (or pair of images) for the derivatives' sake. The log density agrees
/// with 50-digit references to about 1e-14 relative.
///
/// The derivatives in every input are exact: the gradient is the chain rule through the closed-form factor and the
/// series' derivatives in u and w (the derivative series of Henrich et al. 2023, here obtained by evaluating the
/// series over dual numbers), and the Hessian ([`Self::log_prob_derivatives`]) comes from the whole formula over
/// second-order dual numbers. Parameters must satisfy a > 0, t0 >= 0, 0 < w < 1, sv >= 0, with v finite and
/// `boundary` 0 or 1; the log density is NaN otherwise, and −∞ where x <= t0.
///
/// References: Navarro, D. J. and Fuss, I. G. (2009), Fast and accurate calculations for first-passage times in
/// Wiener diffusion models, J. Math. Psych. 53; Gondan, M., Blurton, S. J. and Kesselmeier, M. (2014), Even faster
/// and even more accurate first-passage time densities and distributions for the Wiener diffusion model, J. Math.
/// Psych. 60; Blurton, S. J., Kesselmeier, M. and Gondan, M. (2017), The first-passage time distribution for the
/// diffusion model with variable drift, J. Math. Psych. 76; Henrich, F., Hartmann, R., Pratz, V., Voss, A. and
/// Klauer, K. C. (2023), The seven-parameter diffusion model: an implementation in Stan for Bayesian analyses,
/// Behav. Res. Methods.
#[derive(Debug, Clone, PartialEq)]
pub struct Wiener<'a> {
    /// The boundary separation a.
    pub a: Param<'a>,
    /// The non-decision time t0.
    pub t0: Param<'a>,
    /// The relative starting point w, the start's distance from the lower boundary as a fraction of a.
    pub w: Param<'a>,
    /// The drift v (towards the upper boundary when positive).
    pub v: Param<'a>,
    /// The standard deviation sv of the drift across trials (0 for the four-parameter model).
    pub sv: Param<'a>,
    /// The boundary reached: 0 the lower, 1 the upper.
    pub boundary: Param<'a, i64>,
}

/// Derivatives of [`Wiener::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct WienerGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `a`.
    pub a: DoubleTensor,
    /// With respect to `t0`.
    pub t0: DoubleTensor,
    /// With respect to `w`.
    pub w: DoubleTensor,
    /// With respect to `v`.
    pub v: DoubleTensor,
    /// With respect to `sv`.
    pub sv: DoubleTensor,
}

impl<'a> Wiener<'a> {
    /// Creates the distribution from its parameters (see [`Wiener`]).
    pub fn new(
        a: impl IntoParam<'a, f64>,
        t0: impl IntoParam<'a, f64>,
        w: impl IntoParam<'a, f64>,
        v: impl IntoParam<'a, f64>,
        sv: impl IntoParam<'a, f64>,
        boundary: impl IntoParam<'a, i64>,
    ) -> Self {
        Self {
            a: a.into_param(),
            t0: t0.into_param(),
            w: w.into_param(),
            v: v.into_param(),
            sv: sv.into_param(),
            boundary: boundary.into_param(),
        }
    }

    /// The element-wise log density of the response times x at their boundaries, broadcasting x with the
    /// parameters (−∞ where x <= t0, NaN for invalid parameters).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map(
            (
                x,
                &*self.a,
                &*self.t0,
                &*self.w,
                &*self.v,
                &*self.sv,
                &*self.boundary,
            ),
            |(x, a, t0, w, v, sv, boundary)| evaluate::<0>(x, [a, t0, w, v, sv], boundary).0,
        )
    }

    /// [`Self::log_prob`] and [`Self::d_log_prob`] in one pass, each exactly as the separate method computes it.
    pub fn log_prob_and_d_log_prob(
        &self,
        x: &DoubleTensor,
    ) -> Result<(DoubleTensor, WienerGradient)> {
        let [value, x, a, t0, w, v, sv] = map_n(
            (
                x,
                &*self.a,
                &*self.t0,
                &*self.w,
                &*self.v,
                &*self.sv,
                &*self.boundary,
            ),
            |(x, a, t0, w, v, sv, boundary)| {
                let (value, g) = evaluate::<2>(x, [a, t0, w, v, sv], boundary);
                [value, g[0], g[1], g[2], g[3], g[4], g[5]]
            },
        )?;
        Ok((value, WienerGradient { x, a, t0, w, v, sv }))
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each f64 parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<WienerGradient> {
        let [x, a, t0, w, v, sv] = map_n(
            (
                x,
                &*self.a,
                &*self.t0,
                &*self.w,
                &*self.v,
                &*self.sv,
                &*self.boundary,
            ),
            |(x, a, t0, w, v, sv, boundary)| evaluate::<2>(x, [a, t0, w, v, sv], boundary).1,
        )?;
        Ok(WienerGradient { x, a, t0, w, v, sv })
    }

    /// The log density of one response time `x` with its gradient, as [`Self::d_log_prob`] computes it, with
    /// respect to x (index 0) and the parameters `[a, t0, w, v, sv]` (indices 1 to 5).
    pub fn log_prob_and_gradient(x: f64, parameters: [f64; 5], boundary: i64) -> (f64, [f64; 6]) {
        evaluate::<2>(x, parameters, boundary)
    }

    /// The log density of one response time `x` with its gradient and Hessian with respect to x (index 0) and the
    /// parameters `[a, t0, w, v, sv]` (indices 1 to 5), computed exactly by forward-mode differentiation to second
    /// order.
    pub fn log_prob_derivatives(
        x: f64,
        parameters: [f64; 5],
        boundary: i64,
    ) -> ElementDerivatives<6> {
        crate::jet::derivatives(x, parameters, |x, p| log_density(x, p, boundary))
    }

    /// Draws response times of the given shape (broadcast with the parameters' shapes), each from the
    /// distribution of the first-passage time conditional on reaching its `boundary`, exactly (see
    /// [`sample_wiener`] for the method; the time is drawn given the boundary rather than with it, the drift
    /// first, if sv > 0, from its distribution given the boundary). An error for invalid parameters.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let params: [&DoubleTensor; 5] = [&*self.a, &*self.t0, &*self.w, &*self.v, &*self.sv];
        let mut shapes: Vec<&[usize]> = params.iter().map(|p| p.shape()).collect();
        shapes.push(self.boundary.shape());
        let out = sample_shape(shape, &shapes)?;
        let expanded = params
            .iter()
            .map(|p| operand(*p, &out))
            .collect::<Result<Vec<_>>>()?;
        let boundary = operand(&self.boundary, &out)?;
        let len: usize = out.iter().product();
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            let [a, t0, w, v, sv] = std::array::from_fn(|j| expanded[j].at(i));
            let p = check(a, t0, w, v, sv)?;
            let upper = match boundary.at(i) {
                0 => false,
                1 => true,
                b => return invalid_parameter(format!("the boundary must be 0 or 1, got {b}")),
            };
            data.push(p.conditional_time(upper, rng)?);
        }
        Ok(DoubleTensor::new(&out, data)?)
    }
}

/// Draws `shape` (broadcast with the parameters' shapes) response times and the boundaries they end at (0 the
/// lower, 1 the upper) from the Wiener diffusion model with boundary separation `a`, non-decision time `t0`,
/// relative starting point `w`, drift `v` and drift standard deviation `sv` (see [`Wiener`]), jointly and exactly:
/// the simulated choices and response times of a two-choice task. An error for invalid parameters.
///
/// Each draw takes the trial's drift ν ~ N(v, sv²), the boundary with its probability given ν (in units of a, the
/// upper with probability (1 − e^{−2νaw})/(1 − e^{−2νa})), and then the time given the boundary and ν, by
/// rejection: in units of a² and for the lower boundary (the upper reflects w and ν), the first-passage density is
/// at most the density of the time to first reach that boundary alone, an inverse Gaussian with mean w/|aν| and
/// shape w² times e^{−aνw − a|ν|w}, and the ratio of the two is the method-of-images sum
/// f(u | w)/((2πu³)^{−1/2} w e^{−w²/(2u)}) <= 1, in which the drift cancels. A proposal is accepted with at least
/// the probability of reaching that boundary without drift (w or 1 − w), so the method never discretises the path
/// and needs few proposals unless the start is very close to the other boundary. It is a rejection method in the
/// spirit of Tuerlinckx et al. (2001) and Drugowitsch (2016), with this envelope rather than theirs.
pub fn sample_wiener(
    a: &DoubleTensor,
    t0: &DoubleTensor,
    w: &DoubleTensor,
    v: &DoubleTensor,
    sv: &DoubleTensor,
    shape: &[usize],
    rng: &mut KeanuRng,
) -> Result<(DoubleTensor, IntTensor)> {
    let params = [a, t0, w, v, sv];
    let shapes: Vec<&[usize]> = params.iter().map(|p| p.shape()).collect();
    let out = sample_shape(shape, &shapes)?;
    let expanded = params
        .iter()
        .map(|p| operand(*p, &out))
        .collect::<Result<Vec<_>>>()?;
    let len: usize = out.iter().product();
    let (mut times, mut boundaries) = (Vec::with_capacity(len), Vec::with_capacity(len));
    for i in 0..len {
        let [a, t0, w, v, sv] = std::array::from_fn(|j| expanded[j].at(i));
        let p = check(a, t0, w, v, sv)?;
        let nu = p.a * (p.v + p.sv * rng.next_gaussian());
        let upper = rng.next_f64() < upper_probability(nu, p.w);
        let u = if upper {
            lower_time(1.0 - p.w, p.w, -nu, rng)?
        } else {
            lower_time(p.w, 1.0 - p.w, nu, rng)?
        };
        times.push(p.t0 + p.a * p.a * u);
        boundaries.push(i64::from(upper));
    }
    Ok((
        DoubleTensor::new(&out, times)?,
        IntTensor::new(&out, boundaries)?,
    ))
}

/// Validated parameters of one element.
#[derive(Debug, Clone, Copy)]
struct Parameters {
    a: f64,
    t0: f64,
    w: f64,
    v: f64,
    sv: f64,
}

/// The parameters if they are valid, else an error.
fn check(a: f64, t0: f64, w: f64, v: f64, sv: f64) -> Result<Parameters> {
    if !valid(a, t0, w, v, sv) {
        return invalid_parameter(format!(
            "the Wiener distribution needs a > 0, t0 >= 0, 0 < w < 1, sv >= 0 and a finite v, got a = {a}, \
             t0 = {t0}, w = {w}, v = {v}, sv = {sv}"
        ));
    }
    Ok(Parameters { a, t0, w, v, sv })
}

fn valid(a: f64, t0: f64, w: f64, v: f64, sv: f64) -> bool {
    a > 0.0
        && a.is_finite()
        && t0 >= 0.0
        && t0.is_finite()
        && w > 0.0
        && w < 1.0
        && v.is_finite()
        && sv >= 0.0
        && sv.is_finite()
}

impl Parameters {
    /// A decision-plus-non-decision time drawn given that the process ends at the upper boundary (`upper`) or the
    /// lower one. Reflected so that the boundary is the lower one, the trial's drift ν (in units of a) is drawn
    /// from its distribution given the boundary (if sv > 0), then the time given ν.
    fn conditional_time(&self, upper: bool, rng: &mut KeanuRng) -> Result<f64> {
        // the start's distance from the boundary and from the other one, each exact
        let (w, complement, m) = if upper {
            (1.0 - self.w, self.w, -self.a * self.v)
        } else {
            (self.w, 1.0 - self.w, self.a * self.v)
        };
        let nu = if self.sv > 0.0 {
            drift_given_lower(w, m, self.a * self.sv, rng)?
        } else {
            m
        };
        Ok(self.t0 + self.a * self.a * lower_time(w, complement, nu, rng)?)
    }
}

/// The probability that the standard process (boundaries 0 and 1, start w, drift ν) ends at the upper boundary:
/// (1 − e^{−2νw})/(1 − e^{−2ν}), and for ν < 0, to avoid overflow, e^{2ν(1−w)}(1 − e^{2νw})/(1 − e^{2ν}).
fn upper_probability(nu: f64, w: f64) -> f64 {
    if nu == 0.0 {
        w
    } else if nu > 0.0 {
        (-2.0 * nu * w).exp_m1() / (-2.0 * nu).exp_m1()
    } else {
        (2.0 * nu * (1.0 - w)).exp() * (2.0 * nu * w).exp_m1() / (2.0 * nu).exp_m1()
    }
}

/// A drift ν drawn from N(m, s²) given that the standard process from w ends at the lower boundary, by rejection:
/// the probability of the lower boundary, P(ν) = 1 − [`upper_probability`], is at most min(1, e^{−2νw}), so the
/// envelope N(ν; m, s²) min(1, e^{−2νw}) is a normal truncated to ν <= 0 plus, for ν > 0, the normal
/// N(m − 2ws², s²) scaled by e^{−2mw + 2w²s²}; a draw from it is accepted with probability P(ν) for ν <= 0 and
/// P(ν)e^{2νw} = P(−ν) for ν > 0, at least 1 − w either way.
fn drift_given_lower(w: f64, m: f64, s: f64, rng: &mut KeanuRng) -> Result<f64> {
    let lower_probability = |nu: f64| 1.0 - upper_probability(nu, w);
    let shifted = m - 2.0 * w * s * s;
    let log_below = log_ndtr(-m / s);
    let log_above = -2.0 * m * w + 2.0 * w * w * s * s + log_ndtr(shifted / s);
    // the probability of the envelope's piece below 0
    let below = 1.0 / (1.0 + (log_above - log_below).exp());
    for _ in 0..MAX_PROPOSALS {
        let (nu, accept) = if rng.next_f64() < below {
            let p = rng.next_f64_non_zero() * log_below.exp();
            let nu = (m + s * ndtri(p)).min(0.0);
            (nu, lower_probability(nu))
        } else {
            let p = rng.next_f64_non_zero() * log_ndtr(shifted / s).exp();
            let nu = (shifted - s * ndtri(p)).max(0.0);
            (nu, lower_probability(-nu))
        };
        if nu.is_finite() && rng.next_f64() < accept {
            return Ok(nu);
        }
    }
    gave_up()
}

/// A first-passage time, in units of a², of the standard process from w with drift ν given that it ends at the
/// lower boundary, by rejection from the inverse Gaussian time to first reach that boundary alone (see
/// [`sample_wiener`]). `complement` is 1 − w, given separately so that it keeps its precision when w is close to 1.
fn lower_time(w: f64, complement: f64, nu: f64, rng: &mut KeanuRng) -> Result<f64> {
    let mean = w / nu.abs();
    let shape = w * w;
    for _ in 0..MAX_PROPOSALS {
        let u = inverse_gaussian(mean, shape, rng);
        if !(u > 0.0 && u.is_finite()) {
            continue;
        }
        let standard = log_standard_density(Jet::<0>::constant(u), Jet::constant(w), complement).v;
        let log_ratio = standard + LN_SQRT_2PI + 1.5 * u.ln() - w.ln() + w * w / (2.0 * u);
        if rng.next_f64_non_zero().ln() <= log_ratio {
            return Ok(u);
        }
    }
    gave_up()
}

fn gave_up<T>() -> Result<T> {
    invalid_parameter(format!(
        "Wiener sampling accepted none of {MAX_PROPOSALS} proposals: the start is too close to the other boundary"
    ))
}

/// An inverse Gaussian draw with the given mean (infinite for the Lévy distribution) and shape, by the
/// transformation of Michael, Schucany and Haas (1976), its smaller root written as μ/(1 + c + √c √(c + 2)) with
/// c = μy/(2λ), which neither cancels nor overflows for a large mean (as from a tiny drift). Where c itself
/// overflows the mean is so large that the draw is the Lévy distribution's, the limit of an infinite mean.
fn inverse_gaussian(mean: f64, shape: f64, rng: &mut KeanuRng) -> f64 {
    let n = rng.next_gaussian();
    let y = n * n;
    let c = mean * y / (2.0 * shape);
    if !c.is_finite() {
        return shape / y;
    }
    // μ / x, the larger root's ratio to the mean
    let ratio = 1.0 + c + c.sqrt() * (c + 2.0).sqrt();
    let x = mean / ratio;
    if rng.next_f64() * (mean + x) <= mean {
        x
    } else {
        mean * ratio
    }
}

/// The log density at the lower boundary after the reflection for the upper one, with t the decision time:
/// −2 ln a − ½ ln(1 + sv²t) + (sv²a²w² − 2avw − v²t)/(2(1 + sv²t)) + ln f(t/a² | w), over jets of any order.
fn log_density<const K: usize, const H: usize>(
    x: Jet<K, H>,
    [a, t0, w, v, sv]: [Jet<K, H>; 5],
    boundary: i64,
) -> Jet<K, H> {
    if !valid(a.v, t0.v, w.v, v.v, sv.v) || !(boundary == 0 || boundary == 1) || x.v.is_nan() {
        return Jet::constant(f64::NAN);
    }
    let t = x - t0;
    if t.v <= 0.0 || !t.v.is_finite() {
        return Jet::constant(f64::NEG_INFINITY);
    }
    // the start's distance from the boundary (reflected for the upper one), and exactly its distance from the other
    let (w, v, complement) = if boundary == 1 {
        (1.0 - w, -v, w.v)
    } else {
        (w, v, 1.0 - w.v)
    };
    let u = t / (a * a);
    let series = if K == 0 {
        Jet::constant(
            log_standard_density(Jet::<0>::constant(u.v), Jet::constant(w.v), complement).v,
        )
    } else {
        let s = log_standard_density(Jet::<2>::var(u.v, 0), Jet::var(w.v, 1), complement);
        Jet::compose(s.v, &[u, w], s.g, s.h)
    };
    let s2 = sv * sv;
    let q = 1.0 + s2 * t;
    -2.0 * a.ln() - 0.5 * q.ln()
        + (s2 * (a * w).square() - 2.0 * a * v * w - v * v * t) / (2.0 * q)
        + series
}

/// The log density and, for M = 2, its gradient in [x, a, t0, w, v, sv] (zero for M = 0), by hand through the
/// closed-form factor, with the series and its derivatives in (u, w) from jets. The value is the same arithmetic
/// either way, so [`Wiener::log_prob`] and [`Wiener::d_log_prob`] agree exactly.
fn evaluate<const M: usize>(x: f64, [a, t0, w, v, sv]: [f64; 5], boundary: i64) -> (f64, [f64; 6]) {
    if !valid(a, t0, w, v, sv) || !(boundary == 0 || boundary == 1) || x.is_nan() {
        return (f64::NAN, [f64::NAN; 6]);
    }
    let t = x - t0;
    if t <= 0.0 || !t.is_finite() {
        return (f64::NEG_INFINITY, [0.0; 6]);
    }
    // reflected for the upper boundary: d/dw and d/dv change sign; the start's distance from the other boundary is
    // passed exactly, as it is w itself there and 1 − w would round it away when w is tiny
    let (wl, vl, sign, complement) = if boundary == 1 {
        (1.0 - w, -v, -1.0, w)
    } else {
        (w, v, 1.0, 1.0 - w)
    };
    let a2 = a * a;
    let u = t / a2;
    let f = log_standard_density(Jet::<M>::var(u, 0), Jet::var(wl, 1), complement);
    let s2 = sv * sv;
    let q = 1.0 + s2 * t;
    let n = s2 * a2 * wl * wl - 2.0 * a * vl * wl - vl * vl * t;
    let value = -2.0 * a.ln() - 0.5 * q.ln() + n / (2.0 * q) + f.v;
    let &[f_u, f_w] = f.g.as_slice() else {
        return (value, [0.0; 6]);
    };
    let d_t = -0.5 * s2 / q - vl * vl / (2.0 * q) - n * s2 / (2.0 * q * q) + f_u / a2;
    let d_a = -2.0 / a + (s2 * a * wl * wl - vl * wl) / q - 2.0 * f_u * u / a;
    let d_w = (s2 * a2 * wl - a * vl) / q + f_w;
    let d_v = -(a * wl + vl * t) / q;
    let d_sv = -sv * t / q + sv * a2 * wl * wl / q - n * sv * t / (q * q);
    (value, [d_t, d_a, -d_t, sign * d_w, sign * d_v, d_sv])
}

/// ln f(u | w), the log density of the standard first-passage time (boundaries 0 and 1, no drift, start w) at the
/// lower boundary, from whichever series needs fewer terms (see [`Wiener`]). `complement` is 1 − w, given
/// separately so that a start very close to the other boundary (w = 1 − ε, as for the upper boundary's reflection
/// of a tiny w) keeps its precision: ε is not rounded away, and the series are written in ε where w is near 1.
pub(crate) fn log_standard_density<const M: usize>(
    u: Jet<M>,
    w: Jet<M>,
    complement: f64,
) -> Jet<M> {
    // 1 − w with its derivatives, and its value exactly
    let mut epsilon = 1.0 - w;
    epsilon.v = complement;
    let near_other = w.v > 0.5;
    if small_time_terms(u.v, w.v) <= large_time_terms(u.v) {
        // the images of a start near the other boundary nearly cancel in pairs (by e^{−2ε/u} relative to the
        // first) unless ε is large against u
        if near_other && complement < u.v {
            small_time_near_other_boundary(u, epsilon)
        } else {
            small_time(u, w)
        }
    } else {
        large_time(u, w, near_other.then_some(epsilon))
    }
}

/// The number of images the small-time series needs for an absolute error of 1e-15 in f(u | w): those with
/// |w + 2k| below d, where d >= √u and (d²/(2u)) >= ln(1/ε), about one per unit of d.
fn small_time_terms(u: f64, w: f64) -> f64 {
    let d = (w * w + 2.0 * u * SELECTION_LOG_TOLERANCE)
        .sqrt()
        .max(u.sqrt());
    ((d - w) / 2.0).floor() + 1.0 + ((d + w) / 2.0).floor()
}

/// The number of terms the large-time series needs for an absolute error of 1e-15: the K with
/// e^{−(K² − 1)c}/(2c) <= ε relative to its first term, c = π²u/2, and K >= 1/√(2c).
fn large_time_terms(u: f64) -> f64 {
    let c = PI * PI * u / 2.0;
    let squared = (1.0 + (SELECTION_LOG_TOLERANCE - (2.0 * c).ln()) / c).max(1.0 / (2.0 * c));
    squared.max(1.0).sqrt().ceil()
}

/// ln f(u | w) by the small-time series, as −ln √(2π) − (3/2) ln u + ln w − w²/(2u) + ln(1 + S), S the sum over the
/// pairs of images k >= 1 (see [`Wiener`]).
fn small_time<const M: usize>(u: Jet<M>, w: Jet<M>) -> Jet<M> {
    let mut sum = Jet::constant(0.0);
    let mut extra = false;
    for k in 1..10_000 {
        let k = k as f64;
        // with m = e^{−4kw/u} − 1: 1 + e^{−4kw/u} = 2 + m
        let m = (-4.0 * k * w / u).exp_m1();
        let term = (-2.0 * k * (k - w) / u).exp() * (2.0 + m + 2.0 * k * m / w);
        sum = sum + term;
        if extra {
            break;
        }
        // the alternating series' error bound: the next image, 2(k + 1) − w, relative to the first, w
        let d = 2.0 * (k + 1.0) - w.v;
        if d * d >= u.v {
            let bound = d / w.v * (-2.0 * (k + 1.0) * (k + 1.0 - w.v) / u.v).exp();
            extra = bound <= SERIES_TOLERANCE * (1.0 + sum.v).abs();
        }
    }
    -LN_SQRT_2PI - 1.5 * u.ln() + w.ln() - w * w / (2.0 * u) + sum.ln_1p()
}

/// ln f(u | w) by the small-time series for a start w = 1 − ε near the other boundary, its images paired so that
/// they don't cancel: f(u | w) = (2πu³)^{−½} Σ_{k >= 0} [(2k + w) e^{−(2k + w)²/(2u)} − (2k + 2 − w) e^{−(2k + 2 − w)²/(2u)}],
/// and with m = 2k + 1 each pair is 2 e^{−(m² + ε²)/(2u)} (m sinh(mε/u) − ε cosh(mε/u)), so
/// ln f = −ln √(2π) − (3/2) ln u − (1 + ε²)/(2u) + ln Σ_{k >= 0} 2 e^{−2k(k + 1)/u} (m sinh(mε/u) − ε cosh(mε/u)).
/// For ε < u < 1 (where the small-time series is used) every pair is positive and mε/u stays moderate.
fn small_time_near_other_boundary<const M: usize>(u: Jet<M>, epsilon: Jet<M>) -> Jet<M> {
    let mut sum = Jet::constant(0.0);
    for k in 0..10_000 {
        let k = k as f64;
        let m = 2.0 * k + 1.0;
        let s = m * epsilon / u;
        let (up, down) = (s.exp_m1(), (-s).exp_m1());
        let sinh = 0.5 * (up - down);
        let cosh = 1.0 + 0.5 * (up + down);
        let term = 2.0 * (-2.0 * k * (k + 1.0) / u).exp() * (m * sinh - epsilon * cosh);
        sum = sum + term;
        if k >= 1.0 && term.v.abs() <= SERIES_TOLERANCE * sum.v.abs() {
            break;
        }
    }
    -LN_SQRT_2PI - 1.5 * u.ln() - (1.0 + epsilon.square()) / (2.0 * u) + sum.ln()
}

/// ln f(u | w) by the large-time series, as ln π − c + ln Σ_{k >= 1} k e^{−(k² − 1)c} sin(kπw), c = π²u/2. With
/// `complement` ε = 1 − w (for w near 1), sin(kπw) is computed as (−1)^{k+1} sin(kπε), which doesn't lose ε.
fn large_time<const M: usize>(u: Jet<M>, w: Jet<M>, complement: Option<Jet<M>>) -> Jet<M> {
    let c = PI * PI / 2.0 * u;
    let mut sum = Jet::constant(0.0);
    let mut extra = false;
    for k in 1..10_000 {
        let k = k as f64;
        let sine = match complement {
            Some(epsilon) if k % 2.0 == 0.0 => -(k * PI * epsilon).sin(),
            Some(epsilon) => (k * PI * epsilon).sin(),
            None => (k * PI * w).sin(),
        };
        sum = sum + k * (-(k * k - 1.0) * c).exp() * sine;
        if extra {
            break;
        }
        // the tail after k terms is at most e^{−(k² − 1)c}/(2c) once k >= 1/√(2c)
        if 2.0 * c.v * k * k >= 1.0 {
            let bound = (-(k * k - 1.0) * c.v).exp() / (2.0 * c.v);
            extra = bound <= SERIES_TOLERANCE * sum.v.abs();
        }
    }
    PI.ln() - c + sum.ln()
}
