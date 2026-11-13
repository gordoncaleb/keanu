use super::{
    MAX_PROPOSALS, Parameters, check, drift_given_lower, gave_up, log_density, lower_time, tails,
    valid,
};
use crate::error::{Result, invalid_parameter};
use crate::jet::{self, ElementDerivatives, Jet};
use crate::param::{IntoParam, Param};
use crate::quadrature::integrate_log;
use crate::rng::KeanuRng;
use crate::util::{Operands, map, map_n, operand, sample_shape};
use keanu_tensor::DoubleTensor;
use keanu_tensor::special::log1mexp;

/// Wiener first-passage time with uniform starting-point and non-decision-time variability.
///
/// The starting point is uniform on `[w - sw/2, w + sw/2]` and non-decision time is uniform on
/// `[t0, t0 + st0]`. Zero width fixes the corresponding quantity. The density and tails are joint
/// in time and the selected boundary. Sampling is conditional on that boundary, as for [`super::Wiener`].
#[derive(Debug, Clone, PartialEq)]
pub struct WienerFull<'a> {
    parameters: [Param<'a>; 7],
    boundary: Param<'a, i64>,
}

/// Elementwise score of a full Wiener density or joint tail.
#[derive(Debug, Clone, PartialEq)]
pub struct WienerFullGradient {
    /// With respect to the response time.
    pub x: DoubleTensor,
    /// With respect to boundary separation.
    pub a: DoubleTensor,
    /// With respect to the minimum non-decision time.
    pub t0: DoubleTensor,
    /// With respect to the mean relative starting point.
    pub w: DoubleTensor,
    /// With respect to the mean drift.
    pub v: DoubleTensor,
    /// With respect to drift standard deviation.
    pub sv: DoubleTensor,
    /// With respect to the width of the starting-point distribution.
    pub sw: DoubleTensor,
    /// With respect to the width of the non-decision-time distribution.
    pub st0: DoubleTensor,
}

#[derive(Clone, Copy)]
enum Event {
    Density,
    Cdf,
    Survival,
}

impl<'a> WienerFull<'a> {
    /// Creates a full Wiener distribution. The boundary is 0 for lower and 1 for upper.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        a: impl IntoParam<'a, f64>,
        t0: impl IntoParam<'a, f64>,
        w: impl IntoParam<'a, f64>,
        v: impl IntoParam<'a, f64>,
        sv: impl IntoParam<'a, f64>,
        sw: impl IntoParam<'a, f64>,
        st0: impl IntoParam<'a, f64>,
        boundary: impl IntoParam<'a, i64>,
    ) -> Self {
        Self {
            parameters: [
                a.into_param(),
                t0.into_param(),
                w.into_param(),
                v.into_param(),
                sv.into_param(),
                sw.into_param(),
                st0.into_param(),
            ],
            boundary: boundary.into_param(),
        }
    }

    fn inputs<'b>(
        &'b self,
        x: &'b DoubleTensor,
    ) -> impl Operands<'b, Item = (f64, f64, f64, f64, f64, f64, f64, f64, i64)> {
        let [a, t0, w, v, sv, sw, st0] = &self.parameters;
        (
            x,
            &**a,
            &**t0,
            &**w,
            &**v,
            &**sv,
            &**sw,
            &**st0,
            &*self.boundary,
        )
    }

    fn log_event(&self, x: &DoubleTensor, event: Event) -> Result<DoubleTensor> {
        map(self.inputs(x), |(x, a, t0, w, v, sv, sw, st0, b)| {
            jet::value(x, [a, t0, w, v, sv, sw, st0], |x, p| {
                evaluate(x, p, b, event)
            })
        })
    }

    fn score(&self, x: &DoubleTensor, event: Event) -> Result<WienerFullGradient> {
        let [x, a, t0, w, v, sv, sw, st0] =
            map_n(self.inputs(x), |(x, a, t0, w, v, sv, sw, st0, b)| {
                jet::derivatives(x, [a, t0, w, v, sv, sw, st0], |x, p| {
                    evaluate(x, p, b, event)
                })
                .gradient
            })?;
        Ok(WienerFullGradient {
            x,
            a,
            t0,
            w,
            v,
            sv,
            sw,
            st0,
        })
    }

    /// Elementwise joint log density.
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        self.log_event(x, Event::Density)
    }
    /// Elementwise log P(T ≤ x, B = boundary).
    pub fn log_cdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        self.log_event(x, Event::Cdf)
    }
    /// Elementwise log P(T > x, B = boundary).
    pub fn log_ccdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        self.log_event(x, Event::Survival)
    }
    /// Gradient of the joint log density.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<WienerFullGradient> {
        self.score(x, Event::Density)
    }
    /// Gradient of the joint log CDF.
    pub fn d_log_cdf(&self, x: &DoubleTensor) -> Result<WienerFullGradient> {
        self.score(x, Event::Cdf)
    }
    /// Gradient of the joint log survival probability.
    pub fn d_log_ccdf(&self, x: &DoubleTensor) -> Result<WienerFullGradient> {
        self.score(x, Event::Survival)
    }

    /// Joint log density, gradient and Hessian in the order `[x, a, t0, w, v, sv, sw, st0]`.
    pub fn log_prob_derivatives(x: f64, p: [f64; 7], boundary: i64) -> ElementDerivatives<8> {
        jet::derivatives(x, p, |x, p| evaluate(x, p, boundary, Event::Density))
    }

    /// Joint log CDF or survival probability, gradient and Hessian, in density derivative order.
    pub fn log_tail_derivatives(
        x: f64,
        p: [f64; 7],
        boundary: i64,
        upper: bool,
    ) -> ElementDerivatives<8> {
        let event = if upper { Event::Survival } else { Event::Cdf };
        jet::derivatives(x, p, |x, p| evaluate(x, p, boundary, event))
    }

    /// Draws response times conditional on the selected boundary, broadcasting with `shape`.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let mut shapes: Vec<_> = self.parameters.iter().map(|p| p.shape()).collect();
        shapes.push(self.boundary.shape());
        let shape = sample_shape(shape, &shapes)?;
        let expanded = self
            .parameters
            .iter()
            .map(|p| operand(p, &shape))
            .collect::<Result<Vec<_>>>()?;
        let boundary = operand(&self.boundary, &shape)?;
        let values = (0..shape.iter().product()).map(|i| {
            let [a, t0, w, v, sv, sw, st0] = std::array::from_fn(|j| expanded[j].at(i));
            let b = boundary.at(i);
            let p = check(a, t0, w, v, sv)?;
            if !valid_widths(w, sw, t0, st0) || !matches!(b, 0 | 1) {
                return invalid_parameter("Wiener requires an interior starting-point interval, finite nonnegative time width, and boundary 0 or 1");
            }
            sample_time(p, sw, st0, b == 1, rng)
        }).collect::<Result<_>>()?;
        Ok(DoubleTensor::new(&shape, values)?)
    }
}

fn valid_widths(w: f64, sw: f64, t0: f64, st0: f64) -> bool {
    sw >= 0.0 && 0.5 * sw < w.min(1.0 - w) && st0 >= 0.0 && (t0 + st0).is_finite()
}

fn evaluate<const K: usize, const H: usize>(
    x: Jet<K, H>,
    [a, t0, w, v, sv, sw, st0]: [Jet<K, H>; 7],
    b: i64,
    event: Event,
) -> Jet<K, H> {
    if !valid(a.v, t0.v, w.v, v.v, sv.v)
        || !valid_widths(w.v, sw.v, t0.v, st0.v)
        || !matches!(b, 0 | 1)
        || x.v.is_nan()
    {
        return Jet::constant(f64::NAN);
    }
    let impossible = Jet::constant(f64::NEG_INFINITY);
    let survival = matches!(event, Event::Survival);
    if x.v <= t0.v && !survival || x.v == f64::INFINITY && !matches!(event, Event::Cdf) {
        return impossible;
    }
    let with_start = |start| {
        let term = |offset| {
            let p = [a, t0 + offset, start, v, sv];
            match event {
                Event::Density => log_density(x, p, b),
                _ => tails::log_tail(x, p, b, survival),
            }
        };
        if x.v <= t0.v || x.v.is_infinite() || K == 0 && st0.v == 0.0 {
            return term(Jet::constant(0.0));
        }
        // The portion of the time interval after the deadline only contributes to survival.
        let fraction = if x.v - t0.v < st0.v {
            (x - t0) / st0
        } else {
            Jet::constant(1.0)
        };
        let inside = integrate_log(&|u| term(st0 * fraction * u)) + fraction.ln();
        if survival && fraction.v < 1.0 {
            let total = tails::log_tail(
                Jet::constant(f64::NEG_INFINITY),
                [a, t0, start, v, sv],
                b,
                true,
            );
            inside.log_add_exp(total + (-fraction).ln_1p())
        } else {
            inside
        }
    };
    if K == 0 && sw.v == 0.0 {
        with_start(w)
    } else {
        integrate_log(&|u| with_start(w + (u - 0.5) * sw))
    }
}

fn sample_time(p: Parameters, sw: f64, st0: f64, upper: bool, rng: &mut KeanuRng) -> Result<f64> {
    if sw == 0.0 {
        let time = p.conditional_time(upper, rng)?;
        return Ok(time + st0 * rng.next_f64());
    }
    let (w, complement, m) = if upper {
        (1.0 - p.w, p.w, -p.a * p.v)
    } else {
        (p.w, 1.0 - p.w, p.a * p.v)
    };
    let (min_w, max_complement) = (w - 0.5 * sw, complement + 0.5 * sw);
    let spread = p.a * p.sv;
    for _ in 0..MAX_PROPOSALS {
        // The closest starting point gives an envelope for the boundary probability.
        let drift = if spread > 0.0 {
            drift_given_lower(min_w, m, spread, rng)?
        } else {
            m
        };
        let rate = 2.0 * drift.abs();
        let u = rng.next_f64();
        let tilted = spread == 0.0 && drift > 0.0 && rate * sw > 1e-8;
        let offset = if tilted {
            -(u * (-rate * sw).exp_m1()).ln_1p() / rate
        } else {
            sw * u
        };
        let c = max_complement - offset;
        let log_ratio = if rate == 0.0 {
            (c / max_complement).ln()
        } else {
            log1mexp(-rate * c) - log1mexp(-rate * max_complement)
        };
        let log_ratio = if drift > 0.0 && !tilted {
            log_ratio - rate * offset
        } else {
            log_ratio
        };
        if rng.next_f64_non_zero().ln() <= log_ratio {
            return Ok(p.t0
                + st0 * rng.next_f64()
                + p.a * p.a * lower_time(min_w + offset, c, drift, rng)?);
        }
    }
    gave_up()
}
