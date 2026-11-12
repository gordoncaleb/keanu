use crate::error::{DistributionError, Result, invalid_parameter};
use crate::multivariate::sum_over_batch;
use crate::param::{IntoParam, Param};
use crate::rng::KeanuRng;
use crate::util::{require_probability, sample_shape};
use keanu_tensor::special::log1mexp;
use keanu_tensor::{DoubleTensor, IntTensor, Shape, broadcast_shape};

const ZERO: f64 = f64::NEG_INFINITY;

fn log_add(a: f64, b: f64) -> f64 {
    let high = a.max(b);
    if high == ZERO {
        ZERO
    } else {
        high + (a.min(b) - high).exp().ln_1p()
    }
}

/// A signed sum stored as log positive and log negative contributions.
#[derive(Clone, Copy)]
struct Signed {
    positive: f64,
    negative: f64,
}

impl Signed {
    const ZERO: Self = Self {
        positive: ZERO,
        negative: ZERO,
    };

    fn difference(a: f64, b: f64) -> Self {
        Self {
            positive: a,
            negative: b,
        }
    }

    fn add(self, rhs: Self) -> Self {
        Self::difference(
            log_add(self.positive, rhs.positive),
            log_add(self.negative, rhs.negative),
        )
    }

    fn scale(self, log_magnitude: f64) -> Self {
        Self::difference(self.positive + log_magnitude, self.negative + log_magnitude)
    }

    fn neg(self) -> Self {
        Self::difference(self.negative, self.positive)
    }

    fn ratio(self, log_denominator: f64) -> f64 {
        let (a, b) = (self.positive, self.negative);
        if a == b {
            0.0
        } else if a > b {
            (a - log_denominator + log1mexp(b - a)).exp()
        } else {
            -(b - log_denominator + log1mexp(a - b)).exp()
        }
    }
}

#[derive(Clone, Copy)]
struct Coefficient {
    log_mass: f64,
    dot: Signed,
}

impl Coefficient {
    const ZERO: Self = Self {
        log_mass: ZERO,
        dot: Signed::ZERO,
    };

    fn add(self, other: Self, directional: bool) -> Self {
        let dot = if directional {
            self.dot.add(other.dot)
        } else {
            Signed::ZERO
        };
        Self {
            log_mass: log_add(self.log_mass, other.log_mass),
            dot,
        }
    }
}

fn initial_row(width: usize) -> Vec<Coefficient> {
    let mut row = vec![Coefficient::ZERO; width];
    row[0].log_mass = 0.0;
    row
}

/// Multiply the count polynomial by (1 − p) + p z, or by p + (1 − p) z when counting failures.
fn advance(row: &mut [Coefficient], p: f64, tangent: Option<f64>, failures: bool) {
    let (mut miss, mut hit) = ((-p).ln_1p(), p.ln());
    if failures {
        std::mem::swap(&mut miss, &mut hit);
    }
    let delta = tangent.map(|t| if failures { -t } else { t });
    for j in (0..row.len()).rev() {
        let a = row[j];
        let b = if j == 0 {
            Coefficient::ZERO
        } else {
            row[j - 1]
        };
        let dot = if let Some(t) = delta {
            let mut dot = a.dot.scale(miss).add(b.dot.scale(hit));
            if t != 0.0 {
                let change = Signed::difference(b.log_mass, a.log_mass).scale(t.abs().ln());
                dot = dot.add(if t > 0.0 { change } else { change.neg() });
            }
            dot
        } else {
            Signed::ZERO
        };
        row[j] = Coefficient {
            log_mass: log_add(a.log_mass + miss, b.log_mass + hit),
            dot,
        };
    }
}

fn convolution(
    prefix: &[Coefficient],
    suffix: &[Coefficient],
    count: usize,
    directional: bool,
) -> Coefficient {
    let mut sum = Coefficient::ZERO;
    for (&a, &b) in prefix[..=count].iter().zip(suffix[..=count].iter().rev()) {
        let dot = if directional {
            a.dot.scale(b.log_mass).add(b.dot.scale(a.log_mass))
        } else {
            Signed::ZERO
        };
        sum = sum.add(
            Coefficient {
                log_mass: a.log_mass + b.log_mass,
                dot,
            },
            directional,
        );
    }
    sum
}

#[derive(Clone, Copy)]
enum CountEvent {
    Mass(i64),
    AtMost(i64),
    Above(i64),
    Between(i64, i64),
}

impl CountEvent {
    fn window(self, n: usize) -> Option<(usize, usize, bool)> {
        let (lower, upper) = match self {
            Self::Mass(k) => (k, k),
            Self::AtMost(k) => (0, k),
            Self::Above(k) => (k.checked_add(1)?, n as i64),
            Self::Between(lo, hi) => (lo, hi),
        };
        let (lower, upper) = (lower.max(0), upper.min(n as i64));
        if lower > upper {
            return None;
        }
        let (lower, upper) = (lower as usize, upper as usize);
        Some(if upper > n - lower {
            (n - upper, n - lower, true)
        } else {
            (lower, upper, false)
        })
    }
}

/// Reusable dynamic-programming buffers for the gradient sweep.
///
/// The reverse sweep needs the prefix polynomial before every trial. Storing all of them takes
/// O(n·w) memory for DP width w, so the forward pass keeps only a checkpoint every `block` trials and
/// the reverse sweep replays one block at a time from its checkpoint. With `block ≈ √n` this holds
/// O(√n·w) coefficients at the cost of one extra forward pass. Replaying `advance` from a stored row
/// is deterministic, so the result is identical to keeping every prefix. The buffers are reused
/// across batch rows, along with the normalizer gradient of a truncated count.
#[derive(Default)]
struct Workspace {
    checkpoints: Vec<Coefficient>,
    block: Vec<Coefficient>,
    normalizer: Vec<f64>,
}

/// The checkpoint spacing for `n` trials: ⌈√n⌉, so checkpoints and one replayed block are balanced.
fn checkpoint_interval(n: usize) -> usize {
    n.isqrt() + usize::from(n.isqrt().pow(2) < n)
}

fn row_log_prob(
    p: &[f64],
    event: CountEvent,
    gradient: Option<&mut [f64]>,
    tangent: Option<&[f64]>,
    workspace: &mut Workspace,
) -> f64 {
    let interval = checkpoint_interval(p.len());
    row_log_prob_with_interval(p, event, gradient, tangent, workspace, interval)
}

fn row_log_prob_with_interval(
    p: &[f64],
    event: CountEvent,
    gradient: Option<&mut [f64]>,
    tangent: Option<&[f64]>,
    workspace: &mut Workspace,
    interval: usize,
) -> f64 {
    if p.iter().any(|p| !(0.0..=1.0).contains(p)) {
        return f64::NAN;
    }
    let Some((lower, upper, failures)) = event.window(p.len()) else {
        return ZERO;
    };
    if lower == 0 && upper == p.len() {
        return 0.0;
    }
    let width = upper + 1;
    let interval = interval.max(1);
    let step = |row: &mut [Coefficient], i: usize| {
        advance(row, p[i], tangent.map(|t| t[i]), failures);
    };
    let mut row = initial_row(width);
    let Workspace {
        checkpoints, block, ..
    } = workspace;
    checkpoints.clear();
    for i in 0..p.len() {
        if gradient.is_some() && i % interval == 0 {
            checkpoints.extend_from_slice(&row);
        }
        step(&mut row, i);
    }
    let directional = tangent.is_some();
    let total = row[lower..=upper]
        .iter()
        .copied()
        .reduce(|a, b| a.add(b, directional))
        .expect("nonempty count window");
    if let Some(gradient) = gradient.filter(|_| total.log_mass.is_finite()) {
        let total_score = total.dot.ratio(total.log_mass);
        let mut suffix = initial_row(width);
        // Walk the blocks backwards, replaying each block's prefixes from its checkpoint.
        for (index, checkpoint) in checkpoints.chunks_exact(width).enumerate().rev() {
            let start = index * interval;
            let end = (start + interval).min(p.len());
            block.clear();
            block.extend_from_slice(checkpoint);
            for i in start..end - 1 {
                let previous = block.len() - width;
                block.extend_from_within(previous..);
                step(&mut block[previous + width..], i);
            }
            for i in (start..end).rev() {
                let prefix = &block[(i - start) * width..(i - start + 1) * width];
                let a = if lower == 0 {
                    Coefficient::ZERO
                } else {
                    convolution(prefix, &suffix, lower - 1, directional)
                };
                let b = convolution(prefix, &suffix, upper, directional);
                // The sum of derivatives telescopes to the two omitted-trial boundary coefficients,
                // even at p = 0 or 1.
                let score = Signed::difference(a.log_mass, b.log_mass).ratio(total.log_mass);
                let value = if directional {
                    let change = a.dot.add(b.dot.neg()).ratio(total.log_mass);
                    change
                        - if total_score == 0.0 {
                            0.0
                        } else {
                            score * total_score
                        }
                } else {
                    score
                };
                gradient[i] = if failures { -value } else { value };
                step(&mut suffix, i);
            }
        }
    }
    total.log_mass.min(0.0)
}

/// The count of successes among independent Bernoulli trials, with one probability per trial on the
/// last axis. Leading axes batch independent counts and broadcast with the value.
#[derive(Debug, Clone, PartialEq)]
pub struct PoissonBinomial<'a> {
    p: Param<'a>,
}

/// First derivatives of the joint log mass.
#[derive(Debug, Clone, PartialEq)]
pub struct PoissonBinomialGradient {
    /// With respect to the trial probabilities, summed to their original shape.
    pub p: DoubleTensor,
}

/// A Poisson-binomial count observed within bounds, either clipped or conditioned on being inside.
#[derive(Debug, Clone)]
pub struct BoundedPoissonBinomial<'a, 'b> {
    base: &'b PoissonBinomial<'a>,
    bounds: Bounds<'b>,
}

#[derive(Debug, Clone, Copy)]
struct Bounds<'a> {
    lower: &'a DoubleTensor,
    upper: &'a DoubleTensor,
    censored: bool,
}

impl Bounds<'_> {
    fn broadcast(&self, shape: &[usize]) -> Result<(DoubleTensor, DoubleTensor)> {
        if [self.lower, self.upper].iter().any(|b| {
            b.as_slice()
                .iter()
                .any(|x| x.is_finite() && x.fract() != 0.0)
        }) {
            return invalid_parameter("Poisson-binomial bounds must be whole numbers or infinite");
        }
        Ok((
            self.lower.broadcast_to(shape)?,
            self.upper.broadcast_to(shape)?,
        ))
    }
}

fn bounded_row(
    p: &[f64],
    count: i64,
    (lower, upper): (f64, f64),
    censored: bool,
    mut gradient: Option<&mut [f64]>,
    tangent: Option<&[f64]>,
    workspace: &mut Workspace,
) -> f64 {
    let (lo, hi) = (lower as i64, upper as i64);
    if lower.partial_cmp(&upper) != Some(std::cmp::Ordering::Less) || count < lo || count > hi {
        return ZERO;
    }
    if censored {
        let event = if lo == hi {
            // The integer bounds saturate to the same representable count.
            CountEvent::Between(0, p.len() as i64)
        } else if count == lo && lower.is_finite() {
            CountEvent::AtMost(lo)
        } else if count == hi && upper.is_finite() {
            CountEvent::Between(hi, p.len() as i64)
        } else {
            CountEvent::Mass(count)
        };
        return row_log_prob(p, event, gradient, tangent, workspace);
    }
    let event = CountEvent::Mass(count);
    if lo.max(0) == hi.min(p.len() as i64) {
        let mass = row_log_prob(p, event, None, None, workspace);
        return if mass.is_finite() { 0.0 } else { mass };
    }
    let mass = row_log_prob(p, event, gradient.as_deref_mut(), tangent, workspace);
    if !mass.is_finite() {
        return mass;
    }
    let mut scratch = std::mem::take(&mut workspace.normalizer);
    scratch.clear();
    scratch.resize(p.len(), 0.0);
    let normalizer = row_log_prob(
        p,
        CountEvent::Between(lo, hi),
        gradient.is_some().then_some(&mut scratch[..]),
        tangent,
        workspace,
    );
    if let Some(gradient) = gradient {
        for (g, &s) in gradient.iter_mut().zip(scratch.iter()) {
            *g -= s;
        }
    }
    workspace.normalizer = scratch;
    (mass - normalizer).min(0.0)
}

fn draw_count(p: &[f64], rng: &mut KeanuRng) -> i64 {
    p.iter().filter(|&&p| rng.next_f64() < p).count() as i64
}

fn draw_truncated(p: &[f64], lo: i64, hi: i64, rng: &mut KeanuRng) -> Result<i64> {
    let Some((lower, upper, failures)) = CountEvent::Between(lo, hi).window(p.len()) else {
        return invalid_parameter("the truncation bounds contain no Poisson-binomial counts");
    };
    // An accepted unconditional draw already has the conditional law.
    let first = draw_count(p, rng);
    if (lo..=hi).contains(&first) {
        return Ok(first);
    }
    let mut row = initial_row(upper + 1);
    for &p in p {
        advance(&mut row, p, None, failures);
    }
    let log_z = row[lower..=upper]
        .iter()
        .fold(ZERO, |sum, c| log_add(sum, c.log_mass));
    if log_z == ZERO {
        return invalid_parameter("the truncation bounds contain no Poisson-binomial probability");
    }
    let threshold = log_z + rng.next_f64().ln();
    let mut cumulative = ZERO;
    let mut selected = lower;
    for (i, coefficient) in row.iter().enumerate().take(upper + 1).skip(lower) {
        if coefficient.log_mass != ZERO {
            selected = i;
            cumulative = log_add(cumulative, coefficient.log_mass);
            if cumulative >= threshold {
                break;
            }
        }
    }
    Ok(if failures {
        p.len() - selected
    } else {
        selected
    } as i64)
}

impl<'a> PoissonBinomial<'a> {
    /// Creates the distribution. The last axis must contain at least one trial.
    pub fn new(p: impl IntoParam<'a, f64>) -> Result<Self> {
        let p = p.into_param();
        if p.shape().last().is_none_or(|&n| n == 0) {
            return invalid_parameter("p needs a nonempty last axis of trial probabilities");
        }
        Ok(Self { p })
    }

    fn probabilities(&self, shape: &[usize]) -> Result<DoubleTensor> {
        let mut shape = shape.to_vec();
        shape.push(self.p.shape()[self.p.rank() - 1]);
        Ok(self.p.broadcast_to(&shape)?)
    }

    fn batch_shape(&self, shape: &[usize], bounds: Option<Bounds<'_>>) -> Result<Shape> {
        let mut batch = broadcast_shape(shape, &self.p.shape()[..self.p.rank() - 1])?;
        if let Some(bounds) = bounds {
            batch = broadcast_shape(&batch, bounds.lower.shape())?;
            batch = broadcast_shape(&batch, bounds.upper.shape())?;
        }
        Ok(batch)
    }

    fn evaluate(
        &self,
        x: &IntTensor,
        event: fn(i64) -> CountEvent,
        bounds: Option<Bounds<'_>>,
        want_gradient: bool,
        tangent: Option<&DoubleTensor>,
    ) -> Result<(DoubleTensor, Option<DoubleTensor>)> {
        let batch = self.batch_shape(x.shape(), bounds)?;
        let x = x.broadcast_to(&batch)?;
        let limits = bounds.map(|b| b.broadcast(&batch)).transpose()?;
        let p = self.probabilities(&batch)?;
        let n = p.shape()[p.rank() - 1];
        let tangent = tangent.map(|t| t.broadcast_to(p.shape())).transpose()?;
        let mut gradient = want_gradient.then(|| vec![0.0; p.len()]);
        let mut workspace = Workspace::default();
        let values = x
            .as_slice()
            .iter()
            .zip(p.as_slice().chunks_exact(n))
            .enumerate()
            .map(|(i, (&count, p))| {
                let row = i * n..(i + 1) * n;
                let t = tangent.as_ref().map(|t| &t.as_slice()[row.clone()]);
                let g = gradient.as_mut().map(|g| &mut g[row]);
                if let (Some(bounds), Some((lower, upper))) = (bounds, limits.as_ref()) {
                    bounded_row(
                        p,
                        count,
                        (lower.as_slice()[i], upper.as_slice()[i]),
                        bounds.censored,
                        g,
                        t,
                        &mut workspace,
                    )
                } else {
                    row_log_prob(p, event(count), g, t, &mut workspace)
                }
            })
            .collect();
        let gradient = gradient
            .map(|g| -> Result<_> {
                sum_over_batch(&DoubleTensor::new(p.shape(), g)?, self.p.shape())
            })
            .transpose()?;
        Ok((DoubleTensor::new(&batch, values)?, gradient))
    }

    /// Elementwise log mass, computed in log space in O(n (1 + min(k, n − k))) time per count k.
    /// Outside the support the mass is negative infinity. Invalid probabilities produce NaN.
    pub fn log_prob(&self, x: &IntTensor) -> Result<DoubleTensor> {
        Ok(self.evaluate(x, CountEvent::Mass, None, false, None)?.0)
    }

    /// Computes the pointwise log mass and derivatives of the joint log mass in a shared pass.
    pub fn log_prob_and_gradient(
        &self,
        x: &IntTensor,
    ) -> Result<(DoubleTensor, PoissonBinomialGradient)> {
        self.mass_and_gradient(x, None)
    }

    fn mass_and_gradient(
        &self,
        x: &IntTensor,
        bounds: Option<Bounds<'_>>,
    ) -> Result<(DoubleTensor, PoissonBinomialGradient)> {
        let (lp, p) = self.evaluate(x, CountEvent::Mass, bounds, true, None)?;
        Ok((
            lp,
            PoissonBinomialGradient {
                p: p.expect("requested gradient"),
            },
        ))
    }

    /// Joint log-mass derivatives. At an impossible count the corresponding derivative is zero.
    pub fn d_log_prob(&self, x: &IntTensor) -> Result<PoissonBinomialGradient> {
        Ok(self.log_prob_and_gradient(x)?.1)
    }

    /// Log P(X ≤ k), evaluated directly as a sum of log-space coefficients.
    pub fn log_cdf(&self, k: &IntTensor) -> Result<DoubleTensor> {
        Ok(self.evaluate(k, CountEvent::AtMost, None, false, None)?.0)
    }

    /// Log P(X > k), computed from the failure count without subtracting the CDF from one.
    pub fn log_ccdf(&self, k: &IntTensor) -> Result<DoubleTensor> {
        Ok(self.evaluate(k, CountEvent::Above, None, false, None)?.0)
    }

    /// Derivative of the sum of the log CDFs with respect to the trial probabilities.
    pub fn d_log_cdf(&self, k: &IntTensor) -> Result<PoissonBinomialGradient> {
        self.tail_gradient(k, CountEvent::AtMost)
    }

    /// Derivative of the sum of the log survival probabilities with respect to the trial probabilities.
    pub fn d_log_ccdf(&self, k: &IntTensor) -> Result<PoissonBinomialGradient> {
        self.tail_gradient(k, CountEvent::Above)
    }

    fn tail_gradient(
        &self,
        k: &IntTensor,
        event: fn(i64) -> CountEvent,
    ) -> Result<PoissonBinomialGradient> {
        let (_, gradient) = self.evaluate(k, event, None, true, None)?;
        Ok(PoissonBinomialGradient {
            p: gradient.expect("requested gradient"),
        })
    }

    /// Hessian-vector products of the summed log CDFs, with directions shaped as in [`Self::hvp`].
    pub fn log_cdf_hvp(&self, k: &IntTensor, tangents: &DoubleTensor) -> Result<DoubleTensor> {
        self.event_hvp(k, tangents, CountEvent::AtMost, None)
    }

    /// Hessian-vector products of the summed log survival probabilities, with directions shaped as in [`Self::hvp`].
    pub fn log_ccdf_hvp(&self, k: &IntTensor, tangents: &DoubleTensor) -> Result<DoubleTensor> {
        self.event_hvp(k, tangents, CountEvent::Above, None)
    }

    /// Joint log-mass Hessian-vector products with respect to p. Directions and output have shape
    /// `[directions, p shape...]`.
    pub fn hvp(&self, x: &IntTensor, tangents: &DoubleTensor) -> Result<DoubleTensor> {
        self.event_hvp(x, tangents, CountEvent::Mass, None)
    }

    fn event_hvp(
        &self,
        x: &IntTensor,
        tangents: &DoubleTensor,
        event: fn(i64) -> CountEvent,
        bounds: Option<Bounds<'_>>,
    ) -> Result<DoubleTensor> {
        let [directions, shape @ ..] = tangents.shape() else {
            return Err(DistributionError::InvalidValue(
                "expected batched directions for p".into(),
            ));
        };
        if shape != self.p.shape() {
            return Err(DistributionError::InvalidValue(
                "direction shape does not match p".into(),
            ));
        }
        let mut data = Vec::with_capacity(tangents.len());
        for direction in tangents.as_slice().chunks_exact(self.p.len().max(1)) {
            let tangent = DoubleTensor::new(shape, direction.to_vec())?;
            let (_, product) = self.evaluate(x, event, bounds, true, Some(&tangent))?;
            data.extend(
                product
                    .expect("requested Hessian-vector product")
                    .as_slice(),
            );
        }
        debug_assert_eq!(data.len(), directions * self.p.len());
        Ok(DoubleTensor::new(tangents.shape(), data)?)
    }

    /// Samples a count by drawing the independent trials.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        require_probability("p", &self.p)?;
        let shape = sample_shape(shape, &[&self.p.shape()[..self.p.rank() - 1]])?;
        let p = self.probabilities(&shape)?;
        let n = p.shape()[p.rank() - 1];
        let counts = p
            .as_slice()
            .chunks_exact(n)
            .map(|row| draw_count(row, rng))
            .collect();
        Ok(IntTensor::new(&shape, counts)?)
    }

    /// Observes counts clipped to whole-number bounds. Infinite bounds leave that side unclipped.
    pub fn censored<'b>(
        &'b self,
        lower: &'b DoubleTensor,
        upper: &'b DoubleTensor,
    ) -> BoundedPoissonBinomial<'a, 'b> {
        BoundedPoissonBinomial {
            base: self,
            bounds: Bounds {
                lower,
                upper,
                censored: true,
            },
        }
    }

    /// Conditions counts on being within inclusive whole-number or infinite bounds.
    pub fn truncated<'b>(
        &'b self,
        lower: &'b DoubleTensor,
        upper: &'b DoubleTensor,
    ) -> BoundedPoissonBinomial<'a, 'b> {
        BoundedPoissonBinomial {
            base: self,
            bounds: Bounds {
                lower,
                upper,
                censored: false,
            },
        }
    }
}

impl BoundedPoissonBinomial<'_, '_> {
    /// Pointwise log mass. Unordered bounds and counts outside the allowed range are impossible.
    pub fn log_prob(&self, x: &IntTensor) -> Result<DoubleTensor> {
        Ok(self
            .base
            .evaluate(x, CountEvent::Mass, Some(self.bounds), false, None)?
            .0)
    }

    /// Pointwise log mass and the gradient of its sum with respect to the trial probabilities.
    pub fn log_prob_and_gradient(
        &self,
        x: &IntTensor,
    ) -> Result<(DoubleTensor, PoissonBinomialGradient)> {
        self.base.mass_and_gradient(x, Some(self.bounds))
    }

    /// Hessian-vector products with respect to p, shaped as in [`PoissonBinomial::hvp`].
    pub fn hvp(&self, x: &IntTensor, tangents: &DoubleTensor) -> Result<DoubleTensor> {
        self.base
            .event_hvp(x, tangents, CountEvent::Mass, Some(self.bounds))
    }

    /// Draws clipped counts or exact conditional counts. Unordered bounds are an error.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        require_probability("p", &self.base.p)?;
        let shape = self.base.batch_shape(shape, Some(self.bounds))?;
        let (lower, upper) = self.bounds.broadcast(&shape)?;
        let limits = || lower.as_slice().iter().zip(upper.as_slice());
        if limits().any(|(lo, hi)| lo.partial_cmp(hi) != Some(std::cmp::Ordering::Less)) {
            return invalid_parameter("the lower bound must be less than the upper bound");
        }
        let p = self.base.probabilities(&shape)?;
        let n = p.shape()[p.rank() - 1];
        let counts = p
            .as_slice()
            .chunks_exact(n)
            .zip(limits())
            .map(|(p, (&lo, &hi))| {
                let (lo, hi) = (lo as i64, hi as i64);
                if self.bounds.censored {
                    Ok(draw_count(p, rng).clamp(lo, hi))
                } else {
                    draw_truncated(p, lo, hi, rng)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(IntTensor::new(&shape, counts)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(
        p: &[f64],
        event: CountEvent,
        tangent: Option<&[f64]>,
        interval: usize,
    ) -> Vec<f64> {
        let mut out = vec![0.0; p.len()];
        let mut workspace = Workspace::default();
        row_log_prob_with_interval(p, event, Some(&mut out), tangent, &mut workspace, interval);
        out
    }

    #[test]
    fn checkpoint_interval_is_ceiling_square_root() {
        let expected = [
            (1, 1),
            (2, 2),
            (4, 2),
            (5, 3),
            (9, 3),
            (10, 4),
            (20_000, 142),
        ];
        for (n, interval) in expected {
            assert_eq!(checkpoint_interval(n), interval, "n = {n}");
        }
    }

    #[test]
    fn checkpointed_sweep_matches_storing_every_prefix() {
        let n = 23;
        let p: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 19) as f64 / 19.0).collect();
        let tangent: Vec<f64> = (0..n).map(|i| ((i * 5) % 11) as f64 / 5.0 - 1.0).collect();
        let events = [
            CountEvent::Mass(9),
            CountEvent::Mass(17),
            CountEvent::AtMost(6),
            CountEvent::Above(14),
            CountEvent::Between(4, 12),
        ];
        // Interval one stores every prefix, as the sweep did before checkpointing; the others cover a
        // short final block, an exact multiple of n, a single block and an interval beyond n.
        for event in events {
            for t in [None, Some(tangent.as_slice())] {
                let reference = gradient(&p, event, t, 1);
                for interval in [2, 3, 5, 7, 22, 23, 40, checkpoint_interval(n)] {
                    assert_eq!(
                        gradient(&p, event, t, interval),
                        reference,
                        "interval {interval}"
                    );
                }
            }
        }
    }

    #[test]
    fn workspace_reuse_across_rows_of_different_width() {
        let mut workspace = Workspace::default();
        let long: Vec<f64> = (0..50).map(|i| (i as f64 + 0.5) / 50.0).collect();
        let short = [0.2, 0.7, 0.4];
        let mut first = vec![0.0; long.len()];
        row_log_prob(
            &long,
            CountEvent::Mass(25),
            Some(&mut first),
            None,
            &mut workspace,
        );
        let mut reused = vec![0.0; short.len()];
        row_log_prob(
            &short,
            CountEvent::Mass(1),
            Some(&mut reused),
            None,
            &mut workspace,
        );
        assert_eq!(reused, gradient(&short, CountEvent::Mass(1), None, 1));
    }
}
