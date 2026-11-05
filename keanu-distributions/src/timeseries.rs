//! Distributions over time series with exact marginal likelihoods: the state path of a hidden Markov model and
//! the linear-Gaussian state-space model (the Kalman filter).
//!
//! Both are dedicated computations rather than compositions of tensor operations, so their log likelihoods and
//! gradients are exact and cost one forward and one reverse pass over the series:
//!
//! - [`HiddenMarkovChain`] holds a chain's initial and transition probabilities and, given the log probability of
//!   each observation under each state, gives the log likelihood by the log-space forward algorithm and its
//!   gradient by the backward algorithm. The emission distribution is left to the caller (the graph's
//!   `HiddenMarkovModel` evaluates any element-wise distribution kind with per-state parameters).
//! - [`LinearGaussianStateSpace`] gives the log likelihood of observations by the Kalman filter, with Cholesky
//!   solves of the innovation covariance and the Joseph-form covariance update, and its gradient by reverse-mode
//!   differentiation of the filter, written out by hand.

use crate::error::{DistributionError, Result, invalid_parameter};
use crate::param::{IntoParam, Param};
use crate::rng::KeanuRng;
use keanu_tensor::special::log_sum_exp;
use keanu_tensor::{DoubleTensor, IntTensor};
use std::f64::consts::PI;

/// How far each set of probabilities may sum from 1.
const PROBABILITY_TOLERANCE: f64 = 1e-5;

/// The Markov chain of a hidden Markov model with K states: the state at the first step is drawn from the
/// `initial` probabilities (`[K]`) and each later state from the row of the `transition` matrix ([K, K], row i the
/// probabilities of moving from state i to each state) for the state before it.
///
/// The probabilities must be non-negative and each set must sum to 1 (within 1e-5); zeros are allowed, and make
/// paths through them impossible. The observations' distribution is given to [`Self::log_likelihood`] as the log
/// probability of each step's observation under each state, a [T, K] tensor, so any emission distribution works.
///
/// ```
/// use keanu_distributions::HiddenMarkovChain;
/// use keanu_tensor::DoubleTensor;
///
/// let chain = HiddenMarkovChain::new(
///     DoubleTensor::vector(vec![0.5, 0.5]),
///     DoubleTensor::new(&[2, 2], vec![0.9, 0.1, 0.2, 0.8])?,
/// );
/// // log p(observation at step t | state j), for three steps
/// let emission = DoubleTensor::new(&[3, 2], vec![-0.1, -2.0, -0.3, -1.5, -2.5, -0.2])?;
/// let log_likelihood = chain.log_likelihood(&emission)?;
/// assert!(log_likelihood < 0.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HiddenMarkovChain<'a> {
    /// The probabilities of the first state, `[K]`.
    pub initial: Param<'a>,
    /// The transition probabilities, [K, K]: row i holds the probabilities of moving from state i to each state.
    pub transition: Param<'a>,
}

/// Derivatives of [`HiddenMarkovChain::log_likelihood`].
#[derive(Debug, Clone, PartialEq)]
pub struct HiddenMarkovChainGradient {
    /// With respect to the initial probabilities, `[K]` (each probability treated as a free variable, off the
    /// simplex).
    pub initial: DoubleTensor,
    /// With respect to the transition probabilities, [K, K] (likewise).
    pub transition: DoubleTensor,
    /// With respect to the emission log probabilities, [T, K]: the posterior probability of each state at each
    /// step given every observation (the smoothed state marginals).
    pub emission: DoubleTensor,
}

impl<'a> HiddenMarkovChain<'a> {
    /// Creates the chain from its initial (`[K]`) and transition ([K, K]) probabilities.
    pub fn new(initial: impl IntoParam<'a, f64>, transition: impl IntoParam<'a, f64>) -> Self {
        Self {
            initial: initial.into_param(),
            transition: transition.into_param(),
        }
    }

    /// The number of states K, checking the shapes and the probabilities.
    pub fn states(&self) -> Result<usize> {
        let &[k] = self.initial.shape() else {
            return invalid_parameter(format!(
                "the initial probabilities must be a vector [K], not of shape {:?}",
                self.initial.shape()
            ));
        };
        if k == 0 {
            return invalid_parameter("a hidden Markov model needs at least one state");
        }
        if self.transition.shape() != [k, k] {
            return invalid_parameter(format!(
                "the transition matrix must be [{k}, {k}] for {k} initial probabilities, not of shape {:?}",
                self.transition.shape()
            ));
        }
        let check = |what: &str, row: &[f64]| -> Result<()> {
            // written so that NaN fails too
            if !row.iter().all(|&p| p >= 0.0) {
                return invalid_parameter(format!("{what} must be non-negative, got {row:?}"));
            }
            let total: f64 = row.iter().sum();
            if (total - 1.0).abs() > PROBABILITY_TOLERANCE {
                return invalid_parameter(format!("{what} must sum to 1, got {row:?}"));
            }
            Ok(())
        };
        check("the initial probabilities", self.initial.as_slice())?;
        for row in self.transition.as_slice().chunks(k) {
            check("each row of the transition matrix", row)?;
        }
        Ok(k)
    }

    /// Checks the emission log probabilities against K states, returning the number of steps T.
    fn steps(&self, emission: &DoubleTensor) -> Result<(usize, usize)> {
        let k = self.states()?;
        match *emission.shape() {
            [t, kk] if kk == k && t > 0 => Ok((t, k)),
            _ => Err(DistributionError::InvalidValue(format!(
                "the emission log probabilities must be [T, {k}] with T > 0, not of shape {:?}",
                emission.shape()
            ))),
        }
    }

    /// The log likelihood of a series whose step t observation has log probability `emission[t, j]` under state
    /// j ([T, K], T > 0): log Σ over state paths of the path's probability times its observations', by the
    /// log-space forward algorithm α_0(j) = log π_j + e_0j, α_t(j) = logsumexp_i(α_{t−1}(i) + log A_ij) + e_tj,
    /// and the result logsumexp_j α_{T−1}(j).
    pub fn log_likelihood(&self, emission: &DoubleTensor) -> Result<f64> {
        let (t, k) = self.steps(emission)?;
        let alpha = self.forward(emission.as_slice(), t, k);
        Ok(log_sum_exp(alpha[(t - 1) * k..].iter().copied()))
    }

    /// [`Self::log_likelihood`] and, unless it is impossible (-∞ or NaN), its gradient with respect to the
    /// initial and transition probabilities and the emission log probabilities, by the backward algorithm.
    pub fn log_likelihood_and_gradient(
        &self,
        emission: &DoubleTensor,
    ) -> Result<(f64, Option<HiddenMarkovChainGradient>)> {
        let (t_len, k) = self.steps(emission)?;
        let e = emission.as_slice();
        let alpha = self.forward(e, t_len, k);
        let total = log_sum_exp(alpha[(t_len - 1) * k..].iter().copied());
        if total == f64::NEG_INFINITY || total.is_nan() {
            return Ok((total, None));
        }
        let log_a: Vec<f64> = self.transition.as_slice().iter().map(|p| p.ln()).collect();
        // β_t(i) = log p(observations after t | state i at t); β_{T−1} = 0
        let mut beta = vec![0.0; t_len * k];
        for t in (0..t_len - 1).rev() {
            for i in 0..k {
                beta[t * k + i] = log_sum_exp(
                    (0..k).map(|j| log_a[i * k + j] + e[(t + 1) * k + j] + beta[(t + 1) * k + j]),
                );
            }
        }
        let posterior: Vec<f64> = alpha
            .iter()
            .zip(&beta)
            .map(|(a, b)| (a + b - total).exp())
            .collect();
        // ∂/∂π_j = p(x_0 | j) p(later observations | j) / p(x)
        let initial = (0..k).map(|j| (e[j] + beta[j] - total).exp()).collect();
        // ∂/∂A_ij = Σ_t p(x_0..t, z_t = i) p(x_t+1 | j) p(later | j) / p(x)
        let mut transition = vec![0.0; k * k];
        for t in 0..t_len - 1 {
            for i in 0..k {
                for j in 0..k {
                    transition[i * k + j] +=
                        (alpha[t * k + i] + e[(t + 1) * k + j] + beta[(t + 1) * k + j] - total)
                            .exp();
                }
            }
        }
        Ok((
            total,
            Some(HiddenMarkovChainGradient {
                initial: DoubleTensor::new(&[k], initial)?,
                transition: DoubleTensor::new(&[k, k], transition)?,
                emission: DoubleTensor::new(&[t_len, k], posterior)?,
            }),
        ))
    }

    /// The forward variables α_t(j) = log p(x_0..t, z_t = j), [T, K] flattened.
    fn forward(&self, e: &[f64], t_len: usize, k: usize) -> Vec<f64> {
        let log_a: Vec<f64> = self.transition.as_slice().iter().map(|p| p.ln()).collect();
        let mut alpha = vec![0.0; t_len * k];
        for (j, &p) in self.initial.as_slice().iter().enumerate() {
            alpha[j] = p.ln() + e[j];
        }
        for t in 1..t_len {
            for j in 0..k {
                let previous = &alpha[(t - 1) * k..t * k];
                alpha[t * k + j] =
                    log_sum_exp((0..k).map(|i| previous[i] + log_a[i * k + j])) + e[t * k + j];
            }
        }
        alpha
    }

    /// Draws a path of `steps` states (> 0) by ancestral simulation: the first from the initial probabilities,
    /// each later one from the transition row of the state before it.
    pub fn sample_states(&self, steps: usize, rng: &mut KeanuRng) -> Result<Vec<usize>> {
        let k = self.states()?;
        if steps == 0 {
            return invalid_parameter("a hidden Markov model needs at least one step");
        }
        let transition = self.transition.as_slice();
        let mut states = Vec::with_capacity(steps);
        let mut state = pick(self.initial.as_slice(), rng);
        states.push(state);
        for _ in 1..steps {
            state = pick(&transition[state * k..(state + 1) * k], rng);
            states.push(state);
        }
        Ok(states)
    }

    /// [`Self::sample_states`] as an [`IntTensor`] of shape `[steps]`.
    pub fn sample(&self, steps: usize, rng: &mut KeanuRng) -> Result<IntTensor> {
        let states = self.sample_states(steps, rng)?;
        Ok(IntTensor::new(
            &[steps],
            states.into_iter().map(|s| s as i64).collect(),
        )?)
    }
}

/// An index drawn with the given probabilities (which sum to 1 within tolerance).
fn pick(probabilities: &[f64], rng: &mut KeanuRng) -> usize {
    let u = rng.next_f64();
    let mut cumulative = 0.0;
    for (i, &p) in probabilities.iter().enumerate() {
        cumulative += p;
        if u < cumulative && p > 0.0 {
            return i;
        }
    }
    // rounding left u above the cumulative total: the last state with any probability
    probabilities.iter().rposition(|&p| p > 0.0).unwrap_or(0)
}

/// A dense row-major matrix, for the Kalman filter's small d × d and m × m algebra.
#[derive(Debug, Clone, PartialEq)]
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Mat {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    fn of(t: &DoubleTensor) -> Self {
        let (rows, cols) = match *t.shape() {
            [r, c] => (r, c),
            [r] => (r, 1),
            _ => unreachable!("shapes are checked before conversion"),
        };
        Self {
            rows,
            cols,
            data: t.as_slice().to_vec(),
        }
    }

    fn column(data: &[f64]) -> Self {
        Self {
            rows: data.len(),
            cols: 1,
            data: data.to_vec(),
        }
    }

    fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    fn t(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.data[j * self.rows + i] = self.at(i, j);
            }
        }
        out
    }

    fn mul(&self, other: &Self) -> Self {
        debug_assert_eq!(self.cols, other.rows);
        let mut out = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for l in 0..self.cols {
                let a = self.at(i, l);
                if a == 0.0 {
                    continue;
                }
                for j in 0..other.cols {
                    out.data[i * other.cols + j] += a * other.at(l, j);
                }
            }
        }
        out
    }

    fn zip(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        debug_assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(&a, &b)| f(a, b))
                .collect(),
        }
    }

    fn add(&self, other: &Self) -> Self {
        self.zip(other, |a, b| a + b)
    }

    fn sub(&self, other: &Self) -> Self {
        self.zip(other, |a, b| a - b)
    }

    fn scale(&self, s: f64) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|x| x * s).collect(),
        }
    }

    fn add_assign(&mut self, other: &Self) {
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b;
        }
    }

    /// (A + Aᵀ) / 2.
    fn symmetric(&self) -> Self {
        self.add(&self.t()).scale(0.5)
    }

    /// The lower Cholesky factor of a symmetric positive definite matrix, reading its lower triangle; None if
    /// it is not positive definite (or has NaNs).
    fn cholesky(&self) -> Option<Self> {
        let n = self.rows;
        let mut l = Self::zeros(n, n);
        for j in 0..n {
            let mut pivot = self.at(j, j);
            for k in 0..j {
                pivot -= l.at(j, k) * l.at(j, k);
            }
            if pivot.is_nan() || pivot <= 0.0 {
                return None;
            }
            let pivot = pivot.sqrt();
            l.data[j * n + j] = pivot;
            for i in j + 1..n {
                let mut s = self.at(i, j);
                for k in 0..j {
                    s -= l.at(i, k) * l.at(j, k);
                }
                l.data[i * n + j] = s / pivot;
            }
        }
        Some(l)
    }

    /// A factor L with L Lᵀ = A for a symmetric positive semidefinite matrix (columns of zero where A is
    /// singular), for sampling; an error if A is clearly not positive semidefinite.
    fn psd_factor(&self, what: &str) -> Result<Self> {
        let n = self.rows;
        let scale = (0..n).map(|i| self.at(i, i).abs()).fold(0.0, f64::max);
        let tolerance = 1e-12 * scale.max(f64::MIN_POSITIVE);
        let mut l = Self::zeros(n, n);
        for j in 0..n {
            let mut pivot = self.at(j, j);
            for k in 0..j {
                pivot -= l.at(j, k) * l.at(j, k);
            }
            if pivot.is_nan() || pivot < -tolerance * 1e3 {
                return invalid_parameter(format!("{what} must be positive semidefinite"));
            }
            if pivot <= tolerance {
                continue;
            }
            let pivot = pivot.sqrt();
            l.data[j * n + j] = pivot;
            for i in j + 1..n {
                let mut s = self.at(i, j);
                for k in 0..j {
                    s -= l.at(i, k) * l.at(j, k);
                }
                l.data[i * n + j] = s / pivot;
            }
        }
        Ok(l)
    }

    /// The inverse (L Lᵀ)⁻¹ from a lower Cholesky factor L, by solving against the identity.
    fn cholesky_inverse(l: &Self) -> Self {
        let n = l.rows;
        let mut out = Self::zeros(n, n);
        for c in 0..n {
            // L y = e_c, then Lᵀ x = y
            let mut y = vec![0.0; n];
            for i in 0..n {
                let mut s = if i == c { 1.0 } else { 0.0 };
                for (k, y) in y.iter().enumerate().take(i) {
                    s -= l.at(i, k) * y;
                }
                y[i] = s / l.at(i, i);
            }
            for i in (0..n).rev() {
                let mut s = y[i];
                for k in i + 1..n {
                    s -= l.at(k, i) * out.data[k * n + c];
                }
                out.data[i * n + c] = s / l.at(i, i);
            }
        }
        out.symmetric()
    }

    fn into_tensor(self, shape: &[usize]) -> Result<DoubleTensor> {
        Ok(DoubleTensor::new(shape, self.data)?)
    }
}

/// A linear-Gaussian state-space model, the model of the Kalman filter: a hidden state x_t ∈ ℝ^d and
/// observations y_t ∈ ℝ^m for t = 0..T with
///
/// - x_0 ~ N(`initial_mean`, `initial_covariance`),
/// - x_{t+1} = `transition_matrix` x_t + w_t, w_t ~ N(0, `transition_covariance`),
/// - y_t = `observation_matrix` x_t + v_t, v_t ~ N(0, `observation_covariance`),
///
/// with the noise terms independent. The value is the observations, [T, m]. The covariances must be symmetric
/// positive semidefinite (only the innovation covariance H P Hᵀ + R at each step must be positive definite; one
/// that isn't makes the observations impossible, log probability -∞). There are no batch dimensions.
///
/// The log likelihood is the Kalman filter's sum of innovation log densities, each computed with a Cholesky
/// solve of the innovation covariance (no inverse of it is used for the density), and the filtered covariance
/// is updated in the Joseph form P⁺ = (I − KH) P (I − KH)ᵀ + K R Kᵀ, which keeps it symmetric positive
/// semidefinite over long series and near-deterministic models. The gradient is exact: the filter's recursion
/// differentiated in reverse mode.
///
/// Gradients with respect to a covariance follow the convention of [`crate::MultivariateGaussian`]: a symmetric
/// matrix G whose (i, j) and (j, i) entries are each half the derivative of moving both entries together.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearGaussianStateSpace<'a> {
    /// The mean of the first state, `[d]`.
    pub initial_mean: Param<'a>,
    /// The covariance of the first state, [d, d].
    pub initial_covariance: Param<'a>,
    /// The transition matrix F, [d, d].
    pub transition_matrix: Param<'a>,
    /// The covariance Q of the transition noise, [d, d].
    pub transition_covariance: Param<'a>,
    /// The observation matrix H, [m, d].
    pub observation_matrix: Param<'a>,
    /// The covariance R of the observation noise, [m, m].
    pub observation_covariance: Param<'a>,
}

/// Derivatives of [`LinearGaussianStateSpace::log_prob`], each shaped like the corresponding input.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearGaussianStateSpaceGradient {
    /// With respect to the observations.
    pub x: DoubleTensor,
    /// With respect to `initial_mean`.
    pub initial_mean: DoubleTensor,
    /// With respect to `initial_covariance` (symmetric).
    pub initial_covariance: DoubleTensor,
    /// With respect to `transition_matrix`.
    pub transition_matrix: DoubleTensor,
    /// With respect to `transition_covariance` (symmetric).
    pub transition_covariance: DoubleTensor,
    /// With respect to `observation_matrix`.
    pub observation_matrix: DoubleTensor,
    /// With respect to `observation_covariance` (symmetric).
    pub observation_covariance: DoubleTensor,
}

/// The Kalman filter's output for a series: see [`LinearGaussianStateSpace::filter`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct KalmanFilter {
    /// The log likelihood of the observations (-∞ if an innovation covariance is not positive definite, in which
    /// case the filtered moments stop at that step and are NaN after it).
    pub log_likelihood: f64,
    /// The filtered means E[x_t | y_0..t], [T, d].
    pub means: DoubleTensor,
    /// The filtered covariances Var[x_t | y_0..t], [T, d, d].
    pub covariances: DoubleTensor,
}

/// One step of the filter, kept for the reverse pass.
struct Step {
    /// The predicted mean and covariance of x_t given y_0..t−1.
    mean: Mat,
    covariance: Mat,
    /// The innovation e = y − H m, its covariance's inverse S⁻¹ and S⁻¹ e.
    innovation: Mat,
    innovation_precision: Mat,
    weighted_innovation: Mat,
    /// The gain K = P Hᵀ S⁻¹ and I − K H.
    gain: Mat,
    complement: Mat,
    /// The filtered mean and covariance.
    filtered_mean: Mat,
    filtered_covariance: Mat,
}

/// The forward pass over a series: the log likelihood and each step, stopping at the first impossible one.
struct Forward {
    log_likelihood: f64,
    steps: Vec<Step>,
}

impl<'a> LinearGaussianStateSpace<'a> {
    /// Creates the model from its parameters: the initial mean `[d]` and covariance [d, d], the transition matrix
    /// and noise covariance [d, d], the observation matrix [m, d] and the observation noise covariance [m, m].
    pub fn new(
        initial_mean: impl IntoParam<'a, f64>,
        initial_covariance: impl IntoParam<'a, f64>,
        transition_matrix: impl IntoParam<'a, f64>,
        transition_covariance: impl IntoParam<'a, f64>,
        observation_matrix: impl IntoParam<'a, f64>,
        observation_covariance: impl IntoParam<'a, f64>,
    ) -> Self {
        Self {
            initial_mean: initial_mean.into_param(),
            initial_covariance: initial_covariance.into_param(),
            transition_matrix: transition_matrix.into_param(),
            transition_covariance: transition_covariance.into_param(),
            observation_matrix: observation_matrix.into_param(),
            observation_covariance: observation_covariance.into_param(),
        }
    }

    /// The state and observation dimensions (d, m), checking the parameters' shapes.
    pub fn dimensions(&self) -> Result<(usize, usize)> {
        let &[d] = self.initial_mean.shape() else {
            return invalid_parameter(format!(
                "the initial mean must be a vector [d], not of shape {:?}",
                self.initial_mean.shape()
            ));
        };
        let Some(&m) = self.observation_matrix.shape().first() else {
            return invalid_parameter("the observation matrix must be a matrix [m, d]");
        };
        if d == 0 || m == 0 {
            return invalid_parameter("the state and observation dimensions must be at least 1");
        }
        for (what, shape, expected) in [
            (
                "initial covariance",
                self.initial_covariance.shape(),
                [d, d],
            ),
            ("transition matrix", self.transition_matrix.shape(), [d, d]),
            (
                "transition covariance",
                self.transition_covariance.shape(),
                [d, d],
            ),
            (
                "observation matrix",
                self.observation_matrix.shape(),
                [m, d],
            ),
            (
                "observation covariance",
                self.observation_covariance.shape(),
                [m, m],
            ),
        ] {
            if shape != expected {
                return invalid_parameter(format!(
                    "the {what} must have shape {expected:?} for a state of dimension {d} observed in dimension \
                     {m}, not {shape:?}"
                ));
            }
        }
        Ok((d, m))
    }

    /// Checks the observations' shape [T, m], returning T.
    fn steps(&self, x_shape: &[usize]) -> Result<(usize, usize, usize)> {
        let (d, m) = self.dimensions()?;
        match *x_shape {
            [t, mm] if mm == m => Ok((t, d, m)),
            _ => Err(DistributionError::InvalidValue(format!(
                "the observations must have shape [T, {m}], not {x_shape:?}"
            ))),
        }
    }

    fn forward(&self, x: &DoubleTensor, keep_going: bool) -> Result<Forward> {
        let (t_len, d, obs) = self.steps(x.shape())?;
        let f = Mat::of(&self.transition_matrix);
        let q = Mat::of(&self.transition_covariance);
        let h = Mat::of(&self.observation_matrix);
        let r = Mat::of(&self.observation_covariance);
        let identity = Mat::identity(d);
        let log_2pi = (2.0 * PI).ln();
        let mut mean = Mat::column(self.initial_mean.as_slice());
        let mut covariance = Mat::of(&self.initial_covariance);
        let mut log_likelihood = 0.0;
        let mut steps = Vec::with_capacity(t_len);
        for y in x.as_slice().chunks(obs.max(1)).take(t_len) {
            let ph = covariance.mul(&h.t());
            let s = h.mul(&ph).add(&r);
            let Some(l) = s.cholesky() else {
                log_likelihood = f64::NEG_INFINITY;
                break;
            };
            let innovation = Mat::column(y).sub(&h.mul(&mean));
            let precision = Mat::cholesky_inverse(&l);
            let weighted = precision.mul(&innovation);
            let log_det: f64 = (0..obs).map(|i| l.at(i, i).ln()).sum::<f64>() * 2.0;
            let quadratic: f64 = innovation
                .data
                .iter()
                .zip(&weighted.data)
                .map(|(a, b)| a * b)
                .sum();
            log_likelihood -= 0.5 * (obs as f64 * log_2pi + log_det + quadratic);
            let gain = ph.mul(&precision);
            let complement = identity.sub(&gain.mul(&h));
            let filtered_mean = mean.add(&gain.mul(&innovation));
            let filtered_covariance = complement
                .mul(&covariance)
                .mul(&complement.t())
                .add(&gain.mul(&r).mul(&gain.t()))
                .symmetric();
            let next_mean = f.mul(&filtered_mean);
            let next_covariance = f.mul(&filtered_covariance).mul(&f.t()).add(&q).symmetric();
            steps.push(Step {
                mean,
                covariance,
                innovation,
                innovation_precision: precision,
                weighted_innovation: weighted,
                gain,
                complement,
                filtered_mean,
                filtered_covariance,
            });
            mean = next_mean;
            covariance = next_covariance;
            if !keep_going && log_likelihood.is_nan() {
                break;
            }
        }
        Ok(Forward {
            log_likelihood,
            steps,
        })
    }

    /// The log density of the observations `x` ([T, m]; T may be 0, giving 0): the Kalman filter's sum of
    /// innovation log densities. -∞ if an innovation covariance is not positive definite.
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<f64> {
        Ok(self.forward(x, false)?.log_likelihood)
    }

    /// Runs the Kalman filter over the observations `x` ([T, m]), giving the log likelihood and the filtered
    /// means and covariances of the state at each step.
    pub fn filter(&self, x: &DoubleTensor) -> Result<KalmanFilter> {
        let (t_len, d, _) = self.steps(x.shape())?;
        let forward = self.forward(x, true)?;
        let mut means = vec![f64::NAN; t_len * d];
        let mut covariances = vec![f64::NAN; t_len * d * d];
        for (t, step) in forward.steps.iter().enumerate() {
            means[t * d..(t + 1) * d].copy_from_slice(&step.filtered_mean.data);
            covariances[t * d * d..(t + 1) * d * d].copy_from_slice(&step.filtered_covariance.data);
        }
        Ok(KalmanFilter {
            log_likelihood: forward.log_likelihood,
            means: DoubleTensor::new(&[t_len, d], means)?,
            covariances: DoubleTensor::new(&[t_len, d, d], covariances)?,
        })
    }

    /// The derivatives of [`Self::log_prob`] with respect to the observations and every parameter, by reverse-mode
    /// differentiation of the filter. An error if an innovation covariance is not positive definite (the log
    /// density is -∞ there).
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<LinearGaussianStateSpaceGradient> {
        let (t_len, d, obs) = self.steps(x.shape())?;
        let forward = self.forward(x, false)?;
        if forward.steps.len() < t_len {
            return invalid_parameter(
                "an innovation covariance H P Hᵀ + R is not positive definite, so the log density has no gradient",
            );
        }
        let f = Mat::of(&self.transition_matrix);
        let h = Mat::of(&self.observation_matrix);
        let r = Mat::of(&self.observation_covariance);
        let mut df = Mat::zeros(d, d);
        let mut dq = Mat::zeros(d, d);
        let mut dh = Mat::zeros(obs, d);
        let mut dr = Mat::zeros(obs, obs);
        let mut dx = vec![0.0; t_len * obs];
        // adjoints of the predicted mean and covariance of the step after the current one
        let mut d_mean = Mat::zeros(d, 1);
        let mut d_covariance = Mat::zeros(d, d);
        for (t, step) in forward.steps.iter().enumerate().rev() {
            // prediction: m' = F m⁺, P' = sym(F P⁺ Fᵀ + Q)
            let d_next = d_covariance.symmetric();
            df.add_assign(&d_mean.mul(&step.filtered_mean.t()));
            df.add_assign(&d_next.mul(&f).mul(&step.filtered_covariance).scale(2.0));
            dq.add_assign(&d_next);
            let d_filtered_mean = f.t().mul(&d_mean);
            let d_filtered_covariance = f.t().mul(&d_next).mul(&f).symmetric();
            // Joseph update: P⁺ = sym(A P Aᵀ + K R Kᵀ), A = I − K H
            let (p, k, a) = (&step.covariance, &step.gain, &step.complement);
            let twice = d_filtered_covariance.scale(2.0);
            let d_a = twice.mul(a).mul(p);
            let mut d_p = a.t().mul(&d_filtered_covariance).mul(a);
            let mut d_k = twice.mul(k).mul(&r);
            dr.add_assign(&k.t().mul(&d_filtered_covariance).mul(k));
            d_k.add_assign(&d_a.mul(&h.t()).scale(-1.0));
            dh.add_assign(&k.t().mul(&d_a).scale(-1.0));
            // mean update: m⁺ = m + K e
            let mut d_m = d_filtered_mean.clone();
            d_k.add_assign(&d_filtered_mean.mul(&step.innovation.t()));
            let mut d_e = k.t().mul(&d_filtered_mean);
            // log density: −½ (log det S + eᵀ S⁻¹ e)
            let (precision, weighted) = (&step.innovation_precision, &step.weighted_innovation);
            d_e.add_assign(&weighted.scale(-1.0));
            let mut d_s = precision.sub(&weighted.mul(&weighted.t())).scale(-0.5);
            // gain: K = P Hᵀ S⁻¹
            d_p.add_assign(&d_k.mul(precision).mul(&h));
            dh.add_assign(&precision.mul(&d_k.t()).mul(p));
            d_s.add_assign(&k.t().mul(&d_k).mul(precision).scale(-1.0));
            // innovation: e = y − H m
            dx[t * obs..(t + 1) * obs].copy_from_slice(&d_e.data);
            dh.add_assign(&d_e.mul(&step.mean.t()).scale(-1.0));
            d_m.add_assign(&h.t().mul(&d_e).scale(-1.0));
            // innovation covariance: S = H P Hᵀ + R
            dr.add_assign(&d_s);
            d_p.add_assign(&h.t().mul(&d_s).mul(&h));
            dh.add_assign(&d_s.add(&d_s.t()).mul(&h).mul(p));
            d_mean = d_m;
            d_covariance = d_p;
        }
        Ok(LinearGaussianStateSpaceGradient {
            x: DoubleTensor::new(x.shape(), dx)?,
            initial_mean: d_mean.into_tensor(&[d])?,
            initial_covariance: d_covariance.symmetric().into_tensor(&[d, d])?,
            transition_matrix: df.into_tensor(&[d, d])?,
            transition_covariance: dq.symmetric().into_tensor(&[d, d])?,
            observation_matrix: dh.into_tensor(&[obs, d])?,
            observation_covariance: dr.symmetric().into_tensor(&[obs, obs])?,
        })
    }

    /// Draws observations of shape [T, m] by simulating the state forward. The covariances may be singular
    /// (positive semidefinite) here.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let (t_len, d, obs) = self.steps(shape)?;
        let f = Mat::of(&self.transition_matrix);
        let h = Mat::of(&self.observation_matrix);
        let initial = Mat::of(&self.initial_covariance).psd_factor("the initial covariance")?;
        let transition =
            Mat::of(&self.transition_covariance).psd_factor("the transition covariance")?;
        let observation =
            Mat::of(&self.observation_covariance).psd_factor("the observation covariance")?;
        let noise = |factor: &Mat, n: usize, rng: &mut KeanuRng| {
            let z: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
            factor.mul(&Mat::column(&z))
        };
        let mut state = Mat::column(self.initial_mean.as_slice()).add(&noise(&initial, d, rng));
        let mut out = Vec::with_capacity(t_len * obs);
        for t in 0..t_len {
            let y = h.mul(&state).add(&noise(&observation, obs, rng));
            out.extend_from_slice(&y.data);
            if t + 1 < t_len {
                state = f.mul(&state).add(&noise(&transition, d, rng));
            }
        }
        Ok(DoubleTensor::new(&[t_len, obs], out)?)
    }
}
