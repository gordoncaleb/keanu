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
//!   solves of the innovation covariance and the Joseph-form covariance update, its gradient by reverse-mode
//!   differentiation of the filter, written out by hand, and its Hessian-vector products by differentiating that
//!   reverse pass in forward mode. The matrix algebra is keanu-tensor's.

use crate::error::{DistributionError, Result, invalid_parameter};
use crate::param::{IntoParam, Param};
use crate::rng::KeanuRng;
use keanu_tensor::special::log_sum_exp;
use keanu_tensor::{DoubleTensor, IntTensor, TensorError};
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
        let beta = self.backward(e, t_len, k);
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

    /// The most probable state path given the observations (emission log probabilities [T, K], as for
    /// [`Self::log_likelihood`]), by the Viterbi algorithm in log space: δ_0(j) = log π_j + e_0j,
    /// δ_t(j) = max_i(δ_{t−1}(i) + log A_ij) + e_tj, then back-tracking from the best last state. Returns the
    /// path and its log joint probability log p(z, x). Ties go to the lowest state index.
    ///
    /// # Errors
    ///
    /// An error if the chain or the emission's shape is invalid, or if every path is impossible.
    pub fn viterbi(&self, emission: &DoubleTensor) -> Result<(Vec<usize>, f64)> {
        let (t_len, k) = self.steps(emission)?;
        let e = emission.as_slice();
        let log_a: Vec<f64> = self.transition.as_slice().iter().map(|p| p.ln()).collect();
        let mut delta: Vec<f64> = self
            .initial
            .as_slice()
            .iter()
            .zip(e)
            .map(|(p, e)| p.ln() + e)
            .collect();
        let mut back = vec![0usize; t_len * k];
        for t in 1..t_len {
            let mut next = vec![f64::NEG_INFINITY; k];
            for (j, next) in next.iter_mut().enumerate() {
                let (mut best, mut arg) = (f64::NEG_INFINITY, 0);
                for (i, d) in delta.iter().enumerate() {
                    let v = d + log_a[i * k + j];
                    if v > best {
                        (best, arg) = (v, i);
                    }
                }
                *next = best + e[t * k + j];
                back[t * k + j] = arg;
            }
            delta = next;
        }
        let (mut best, mut state) = (f64::NEG_INFINITY, 0);
        for (j, &d) in delta.iter().enumerate() {
            if d > best {
                (best, state) = (d, j);
            }
        }
        if !best.is_finite() {
            return impossible_observations();
        }
        let mut path = vec![0; t_len];
        path[t_len - 1] = state;
        for t in (1..t_len).rev() {
            state = back[t * k + state];
            path[t - 1] = state;
        }
        Ok((path, best))
    }

    /// The posterior probability of each state at each step given every observation (emission log probabilities
    /// [T, K], as for [`Self::log_likelihood`]): γ_t(j) = p(z_t = j | x), [T, K], by the forward-backward
    /// algorithm. Each row sums to 1.
    ///
    /// # Errors
    ///
    /// An error if the chain or the emission's shape is invalid, or if the observations are impossible.
    pub fn posterior_marginals(&self, emission: &DoubleTensor) -> Result<DoubleTensor> {
        let (t_len, k) = self.steps(emission)?;
        let e = emission.as_slice();
        let alpha = self.forward(e, t_len, k);
        let total = log_sum_exp(alpha[(t_len - 1) * k..].iter().copied());
        if !total.is_finite() {
            return impossible_observations();
        }
        let beta = self.backward(e, t_len, k);
        let posterior = alpha
            .iter()
            .zip(&beta)
            .map(|(a, b)| {
                let v = a + b;
                if v == f64::NEG_INFINITY {
                    0.0
                } else {
                    (v - total).exp()
                }
            })
            .collect();
        Ok(DoubleTensor::new(&[t_len, k], posterior)?)
    }

    /// Draws a state path from its posterior given the observations (emission log probabilities [T, K], as for
    /// [`Self::log_likelihood`]), by forward filtering and backward sampling: the last state from
    /// p(z_{T−1} | x) ∝ exp α_{T−1}, then each earlier one from p(z_t = i | z_{t+1}, x) ∝ exp(α_t(i)) A_{i, z_{t+1}}.
    /// (Contrast [`Self::sample_states`], which draws from the prior.)
    ///
    /// # Errors
    ///
    /// An error if the chain or the emission's shape is invalid, or if the observations are impossible.
    pub fn sample_posterior_states(
        &self,
        emission: &DoubleTensor,
        rng: &mut KeanuRng,
    ) -> Result<Vec<usize>> {
        let (t_len, k) = self.steps(emission)?;
        let alpha = self.forward(emission.as_slice(), t_len, k);
        let a = self.transition.as_slice();
        let draw = |logs: &[f64], rng: &mut KeanuRng| -> Result<usize> {
            let total = log_sum_exp(logs.iter().copied());
            if !total.is_finite() {
                return impossible_observations();
            }
            let p: Vec<f64> = logs.iter().map(|l| (l - total).exp()).collect();
            Ok(pick(&p, rng))
        };
        let mut path = vec![0; t_len];
        path[t_len - 1] = draw(&alpha[(t_len - 1) * k..], rng)?;
        for t in (0..t_len - 1).rev() {
            let next = path[t + 1];
            let logs: Vec<f64> = (0..k)
                .map(|i| alpha[t * k + i] + a[i * k + next].ln())
                .collect();
            path[t] = draw(&logs, rng)?;
        }
        Ok(path)
    }

    /// The backward variables β_t(i) = log p(x_{t+1}..x_{T−1} | z_t = i), [T, K] flattened.
    fn backward(&self, e: &[f64], t_len: usize, k: usize) -> Vec<f64> {
        let log_a: Vec<f64> = self.transition.as_slice().iter().map(|p| p.ln()).collect();
        let mut beta = vec![0.0; t_len * k];
        for t in (0..t_len - 1).rev() {
            for i in 0..k {
                beta[t * k + i] = log_sum_exp(
                    (0..k).map(|j| log_a[i * k + j] + e[(t + 1) * k + j] + beta[(t + 1) * k + j]),
                );
            }
        }
        beta
    }
}

/// The error of a posterior computation for observations of probability zero.
fn impossible_observations<T>() -> Result<T> {
    Err(DistributionError::InvalidValue(
        "the observations are impossible under the model (probability zero), so they have no posterior".into(),
    ))
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

/// The matrix algebra the Kalman filter and its reverse pass are written in, so that one implementation of each
/// serves both plain matrices ([`DoubleTensor`]s: the log density and its gradient) and matrices carrying `b`
/// tangents ([`Dual`]: the gradient's directional derivatives, i.e. Hessian-vector products, by forward-mode
/// differentiation of the reverse pass). The numerical kernels are keanu-tensor's.
trait Algebra: Clone + Sized {
    /// op(self) op(other), with op the transpose where asked.
    fn mm(&self, other: &Self, transpose_left: bool, transpose_right: bool) -> Result<Self>;
    fn plus(&self, other: &Self) -> Result<Self>;
    fn minus(&self, other: &Self) -> Result<Self>;
    fn times(&self, s: f64) -> Self;
    fn tr(&self) -> Result<Self>;
    /// The matrix itself, without any tangents.
    fn value(&self) -> &DoubleTensor;
    /// A zero matrix (with as many zero tangents as self has).
    fn zeros(&self, rows: usize, cols: usize) -> Self;
    /// The inverse of a symmetric positive definite matrix, by its Cholesky factor, and the log of its
    /// determinant; None if the matrix is not positive definite (or has NaNs).
    fn spd_inverse(&self) -> Result<Option<(Self, f64)>>;
}

/// (A + Aᵀ) / 2.
fn sym<A: Algebra>(a: &A) -> Result<A> {
    Ok(a.plus(&a.tr()?)?.times(0.5))
}

impl Algebra for DoubleTensor {
    fn mm(&self, other: &Self, transpose_left: bool, transpose_right: bool) -> Result<Self> {
        Ok(self.matmul_transposed(other, transpose_left, transpose_right)?)
    }

    fn plus(&self, other: &Self) -> Result<Self> {
        Ok(self.add(other)?)
    }

    fn minus(&self, other: &Self) -> Result<Self> {
        Ok(self.sub(other)?)
    }

    fn times(&self, s: f64) -> Self {
        self * s
    }

    fn tr(&self) -> Result<Self> {
        Ok(self.transpose()?)
    }

    fn value(&self) -> &DoubleTensor {
        self
    }

    fn zeros(&self, rows: usize, cols: usize) -> Self {
        DoubleTensor::zeros(&[rows, cols])
    }

    fn spd_inverse(&self) -> Result<Option<(Self, f64)>> {
        let l = match self.cholesky() {
            Ok(l) => l,
            Err(TensorError::NotPositiveDefinite) => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let diagonal = l.diag_part()?;
        if diagonal.as_slice().iter().any(|v| v.is_nan() || *v <= 0.0) {
            return Ok(None);
        }
        let log_det = 2.0 * diagonal.as_slice().iter().map(|v| v.ln()).sum::<f64>();
        Ok(Some((l.cholesky_inverse()?, log_det)))
    }
}

/// A matrix [r, c] with `b` tangents [b, r, c]: each operation carries the tangents along by the product rule.
#[derive(Debug, Clone)]
struct Dual {
    value: DoubleTensor,
    tangent: DoubleTensor,
}

impl Dual {
    /// A matrix with the given tangents ([b, r, c]), or zero tangents.
    fn new(value: DoubleTensor, tangent: Option<DoubleTensor>, b: usize) -> Self {
        let tangent = tangent.unwrap_or_else(|| {
            let shape = value.shape();
            DoubleTensor::zeros(&[b, shape[0], shape[1]])
        });
        Self { value, tangent }
    }
}

impl Algebra for Dual {
    fn mm(&self, other: &Self, transpose_left: bool, transpose_right: bool) -> Result<Self> {
        let product = |a: &DoubleTensor, b: &DoubleTensor| -> Result<DoubleTensor> {
            Ok(a.matmul_transposed(b, transpose_left, transpose_right)?)
        };
        Ok(Self {
            value: product(&self.value, &other.value)?,
            tangent: product(&self.tangent, &other.value)?
                .add(&product(&self.value, &other.tangent)?)?,
        })
    }

    fn plus(&self, other: &Self) -> Result<Self> {
        Ok(Self {
            value: self.value.add(&other.value)?,
            tangent: self.tangent.add(&other.tangent)?,
        })
    }

    fn minus(&self, other: &Self) -> Result<Self> {
        Ok(Self {
            value: self.value.sub(&other.value)?,
            tangent: self.tangent.sub(&other.tangent)?,
        })
    }

    fn times(&self, s: f64) -> Self {
        Self {
            value: &self.value * s,
            tangent: &self.tangent * s,
        }
    }

    fn tr(&self) -> Result<Self> {
        Ok(Self {
            value: self.value.transpose()?,
            tangent: self.tangent.transpose()?,
        })
    }

    fn value(&self) -> &DoubleTensor {
        &self.value
    }

    fn zeros(&self, rows: usize, cols: usize) -> Self {
        Self::new(
            DoubleTensor::zeros(&[rows, cols]),
            None,
            self.tangent.shape()[0],
        )
    }

    /// The tangent of S⁻¹ is −S⁻¹ Ṡ S⁻¹ (Ṡ is symmetric here, as every covariance tangent is).
    fn spd_inverse(&self) -> Result<Option<(Self, f64)>> {
        let Some((inverse, log_det)) = self.value.spd_inverse()? else {
            return Ok(None);
        };
        let tangent = inverse.matmul(&self.tangent)?.matmul(&inverse)?.neg();
        Ok(Some((
            Self {
                value: inverse,
                tangent,
            },
            log_det,
        )))
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
/// differentiated in reverse mode. So are the Hessian-vector products ([`Self::hvp`]): that reverse pass
/// differentiated in forward mode.
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

/// Derivatives of [`LinearGaussianStateSpace::log_prob`], each shaped like the corresponding input (from
/// [`LinearGaussianStateSpace::d_log_prob`]), or Hessian-vector products along `b` directions, each shaped
/// [b, input shape...] (from [`LinearGaussianStateSpace::hvp`]).
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

/// The Rauch–Tung–Striebel smoother's output for a series: see [`LinearGaussianStateSpace::smooth`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct KalmanSmoother {
    /// The log likelihood of the observations.
    pub log_likelihood: f64,
    /// The smoothed means E[x_t | y_0..T−1], [T, d].
    pub means: DoubleTensor,
    /// The smoothed covariances Var[x_t | y_0..T−1], [T, d, d].
    pub covariances: DoubleTensor,
}

/// The model's matrices for one step after the observed series, for [`LinearGaussianStateSpace::forecast_with`]:
/// the transition matrix F and noise covariance Q that move the state into the step, and the observation matrix H
/// and noise covariance R that observe it there.
#[derive(Debug, Clone, PartialEq)]
pub struct FutureStep {
    /// The transition matrix F, [d, d], from the previous step's state to this one's.
    pub transition_matrix: DoubleTensor,
    /// The transition noise covariance Q, [d, d].
    pub transition_covariance: DoubleTensor,
    /// The observation matrix H, [m, d].
    pub observation_matrix: DoubleTensor,
    /// The observation noise covariance R, [m, m].
    pub observation_covariance: DoubleTensor,
}

impl FutureStep {
    /// A step from its transition matrix and covariance and its observation matrix and covariance.
    pub fn new(
        transition_matrix: DoubleTensor,
        transition_covariance: DoubleTensor,
        observation_matrix: DoubleTensor,
        observation_covariance: DoubleTensor,
    ) -> Self {
        Self {
            transition_matrix,
            transition_covariance,
            observation_matrix,
            observation_covariance,
        }
    }
}

/// The predictive distribution of the steps after an observed series, from [`LinearGaussianStateSpace::forecast`]:
/// the Gaussian marginals of each future step's state and observation given every observation, h steps.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct StateSpaceForecast {
    /// The predicted state means E[x_{T+k} | y_0..T−1], [h, d].
    pub state_means: DoubleTensor,
    /// The predicted state covariances, [h, d, d].
    pub state_covariances: DoubleTensor,
    /// The predicted observation means E[y_{T+k} | y_0..T−1] = H m, [h, m].
    pub observation_means: DoubleTensor,
    /// The predicted observation covariances H P Hᵀ + R, [h, m, m].
    pub observation_covariances: DoubleTensor,
}

/// A simulated path of states and observations, from [`LinearGaussianStateSpace::sample_forecast`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct StateSpacePath {
    /// The states, [h, d].
    pub states: DoubleTensor,
    /// The observations, [h, m].
    pub observations: DoubleTensor,
}

/// One step of the filter, kept for the reverse pass.
struct Step<A> {
    /// The predicted mean and covariance of x_t given y_0..t−1.
    mean: A,
    covariance: A,
    /// The innovation e = y − H m, its covariance's inverse S⁻¹ and S⁻¹ e.
    innovation: A,
    innovation_precision: A,
    weighted_innovation: A,
    /// The gain K = P Hᵀ S⁻¹ and I − K H.
    gain: A,
    complement: A,
    /// The filtered mean and covariance.
    filtered_mean: A,
    filtered_covariance: A,
}

/// The forward pass over a series: the log likelihood and each step, stopping at the first impossible one.
struct Forward<A> {
    log_likelihood: f64,
    steps: Vec<Step<A>>,
}

/// The model's parameters as matrices (the initial mean a column [d, 1]), and the identity [d, d].
struct Matrices<A> {
    initial_mean: A,
    initial_covariance: A,
    f: A,
    q: A,
    h: A,
    r: A,
    identity: A,
}

/// The derivatives of the log likelihood with respect to each observation (a column [m, 1] per step) and each
/// parameter (the initial mean a column [d, 1]).
struct Adjoints<A> {
    x: Vec<A>,
    initial_mean: A,
    initial_covariance: A,
    f: A,
    q: A,
    h: A,
    r: A,
}

/// The Kalman filter over the observations (a column [m, 1] each), with Cholesky solves of the innovation
/// covariance and the Joseph-form covariance update. It stops at the first innovation covariance that is not
/// positive definite (the log likelihood then -∞), and, unless `keep_going`, at a NaN log likelihood.
fn kalman_forward<A: Algebra>(
    p: &Matrices<A>,
    observations: impl Iterator<Item = Result<A>>,
    keep_going: bool,
) -> Result<Forward<A>> {
    let obs = p.h.value().shape()[0];
    let log_2pi = (2.0 * PI).ln();
    let mut mean = p.initial_mean.clone();
    let mut covariance = p.initial_covariance.clone();
    let mut log_likelihood = 0.0;
    let mut steps = Vec::with_capacity(observations.size_hint().0);
    for y in observations {
        let ph = covariance.mm(&p.h, false, true)?;
        let s = p.h.mm(&ph, false, false)?.plus(&p.r)?;
        let Some((precision, log_det)) = s.spd_inverse()? else {
            log_likelihood = f64::NEG_INFINITY;
            break;
        };
        let innovation = y?.minus(&p.h.mm(&mean, false, false)?)?;
        let weighted = precision.mm(&innovation, false, false)?;
        let quadratic: f64 = innovation
            .value()
            .as_slice()
            .iter()
            .zip(weighted.value().as_slice())
            .map(|(a, b)| a * b)
            .sum();
        log_likelihood -= 0.5 * (obs as f64 * log_2pi + log_det + quadratic);
        let gain = ph.mm(&precision, false, false)?;
        let complement = p.identity.minus(&gain.mm(&p.h, false, false)?)?;
        let filtered_mean = mean.plus(&gain.mm(&innovation, false, false)?)?;
        let filtered_covariance = sym(&complement
            .mm(&covariance, false, false)?
            .mm(&complement, false, true)?
            .plus(&gain.mm(&p.r, false, false)?.mm(&gain, false, true)?)?)?;
        let next_mean = p.f.mm(&filtered_mean, false, false)?;
        let next_covariance = sym(&p
            .f
            .mm(&filtered_covariance, false, false)?
            .mm(&p.f, false, true)?
            .plus(&p.q)?)?;
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

/// Reverse-mode differentiation of a complete [`kalman_forward`] pass. Covariance derivatives are symmetrised.
fn kalman_reverse<A: Algebra>(p: &Matrices<A>, steps: &[Step<A>]) -> Result<Adjoints<A>> {
    let (obs, d) = (p.h.value().shape()[0], p.h.value().shape()[1]);
    let (f, h, r) = (&p.f, &p.h, &p.r);
    let mut df = f.zeros(d, d);
    let mut dq = f.zeros(d, d);
    let mut dh = f.zeros(obs, d);
    let mut dr = f.zeros(obs, obs);
    let mut dx = Vec::with_capacity(steps.len());
    // adjoints of the predicted mean and covariance of the step after the current one
    let mut d_mean = f.zeros(d, 1);
    let mut d_covariance = f.zeros(d, d);
    for step in steps.iter().rev() {
        // prediction: m' = F m⁺, P' = sym(F P⁺ Fᵀ + Q)
        let d_next = sym(&d_covariance)?;
        df = df.plus(&d_mean.mm(&step.filtered_mean, false, true)?)?;
        df = df.plus(
            &d_next
                .mm(f, false, false)?
                .mm(&step.filtered_covariance, false, false)?
                .times(2.0),
        )?;
        dq = dq.plus(&d_next)?;
        let d_filtered_mean = f.mm(&d_mean, true, false)?;
        let d_filtered_covariance = sym(&f.mm(&d_next, true, false)?.mm(f, false, false)?)?;
        // Joseph update: P⁺ = sym(A P Aᵀ + K R Kᵀ), A = I − K H
        let (cov, k, a) = (&step.covariance, &step.gain, &step.complement);
        let twice = d_filtered_covariance.times(2.0);
        let d_a = twice.mm(a, false, false)?.mm(cov, false, false)?;
        let mut d_p = a
            .mm(&d_filtered_covariance, true, false)?
            .mm(a, false, false)?;
        let mut d_k = twice.mm(k, false, false)?.mm(r, false, false)?;
        dr = dr.plus(
            &k.mm(&d_filtered_covariance, true, false)?
                .mm(k, false, false)?,
        )?;
        d_k = d_k.minus(&d_a.mm(h, false, true)?)?;
        dh = dh.minus(&k.mm(&d_a, true, false)?)?;
        // mean update: m⁺ = m + K e
        let mut d_m = d_filtered_mean.clone();
        d_k = d_k.plus(&d_filtered_mean.mm(&step.innovation, false, true)?)?;
        let mut d_e = k.mm(&d_filtered_mean, true, false)?;
        // log density: −½ (log det S + eᵀ S⁻¹ e)
        let (precision, weighted) = (&step.innovation_precision, &step.weighted_innovation);
        d_e = d_e.minus(weighted)?;
        let mut d_s = precision
            .minus(&weighted.mm(weighted, false, true)?)?
            .times(-0.5);
        // gain: K = P Hᵀ S⁻¹
        d_p = d_p.plus(&d_k.mm(precision, false, false)?.mm(h, false, false)?)?;
        dh = dh.plus(&precision.mm(&d_k, false, true)?.mm(cov, false, false)?)?;
        d_s = d_s.minus(&k.mm(&d_k, true, false)?.mm(precision, false, false)?)?;
        // innovation: e = y − H m
        dh = dh.minus(&d_e.mm(&step.mean, false, true)?)?;
        d_m = d_m.minus(&h.mm(&d_e, true, false)?)?;
        dx.push(d_e);
        // innovation covariance: S = H P Hᵀ + R
        dr = dr.plus(&d_s)?;
        d_p = d_p.plus(&h.mm(&d_s, true, false)?.mm(h, false, false)?)?;
        dh = dh.plus(
            &d_s.plus(&d_s.tr()?)?
                .mm(h, false, false)?
                .mm(cov, false, false)?,
        )?;
        d_mean = d_m;
        d_covariance = d_p;
    }
    dx.reverse();
    Ok(Adjoints {
        x: dx,
        initial_mean: d_mean,
        initial_covariance: sym(&d_covariance)?,
        f: df,
        q: sym(&dq)?,
        h: dh,
        r: sym(&dr)?,
    })
}

/// A factor L with L Lᵀ = A for a symmetric positive semidefinite n × n matrix (row-major; columns of zero where
/// A is singular), for sampling; an error if A is clearly not positive semidefinite. keanu-tensor's Cholesky
/// factorisation needs a positive definite matrix, and sampling allows singular covariances.
fn psd_factor(a: &DoubleTensor, what: &str) -> Result<DoubleTensor> {
    let n = a.shape()[0];
    let a = a.as_slice();
    let scale = (0..n).map(|i| a[i * n + i].abs()).fold(0.0, f64::max);
    let tolerance = 1e-12 * scale.max(f64::MIN_POSITIVE);
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let pivot = a[j * n + j] - (0..j).map(|k| l[j * n + k] * l[j * n + k]).sum::<f64>();
        if pivot.is_nan() || pivot < -tolerance * 1e3 {
            return invalid_parameter(format!("{what} must be positive semidefinite"));
        }
        if pivot <= tolerance {
            continue;
        }
        let pivot = pivot.sqrt();
        l[j * n + j] = pivot;
        for i in j + 1..n {
            let s = a[i * n + j] - (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum::<f64>();
            l[i * n + j] = s / pivot;
        }
    }
    Ok(DoubleTensor::new(&[n, n], l)?)
}

/// A draw from N(mean, covariance) for a column `mean` [n, 1] and a positive semidefinite covariance [n, n].
fn gaussian_draw(
    mean: &DoubleTensor,
    covariance: &DoubleTensor,
    what: &str,
    rng: &mut KeanuRng,
) -> Result<DoubleTensor> {
    let n = mean.shape()[0];
    let factor = psd_factor(covariance, what)?;
    let z: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
    Ok(mean.add(&factor.matmul(&DoubleTensor::new(&[n, 1], z)?)?)?)
}

/// The inverse of a symmetric positive semidefinite matrix by its Cholesky factor, or, where it is singular, its
/// Moore–Penrose pseudo-inverse from its eigendecomposition (eigenvalues below 10⁻¹² of the largest taken as 0).
fn psd_pseudo_inverse(a: &DoubleTensor) -> Result<DoubleTensor> {
    if let Some((inverse, _)) = a.spd_inverse()? {
        return Ok(inverse);
    }
    let n = a.shape()[0];
    let (values, vectors) = a.eigh()?;
    let largest = values
        .as_slice()
        .iter()
        .fold(0.0, |m: f64, v| m.max(v.abs()));
    let tolerance = 1e-12 * largest;
    let q = vectors.as_slice();
    let mut inverse = vec![0.0; n * n];
    for (k, &lambda) in values.as_slice().iter().enumerate() {
        if lambda.is_nan() {
            return invalid_parameter("a predicted state covariance is not finite");
        }
        if lambda <= tolerance {
            continue;
        }
        for i in 0..n {
            for j in 0..n {
                inverse[i * n + j] += q[i * n + k] * q[j * n + k] / lambda;
            }
        }
    }
    Ok(DoubleTensor::new(&[n, n], inverse)?)
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
        for (what, shape, expected) in self.parameter_shapes(d, m) {
            if shape != expected {
                return invalid_parameter(format!(
                    "the {what} must have shape {expected:?} for a state of dimension {d} observed in dimension \
                     {m}, not {shape:?}"
                ));
            }
        }
        Ok((d, m))
    }

    /// Each parameter's name, shape and expected shape, in the order of [`Self::new`].
    fn parameter_shapes(&self, d: usize, m: usize) -> [(&'static str, &[usize], Vec<usize>); 6] {
        [
            ("initial mean", self.initial_mean.shape(), vec![d]),
            (
                "initial covariance",
                self.initial_covariance.shape(),
                vec![d, d],
            ),
            (
                "transition matrix",
                self.transition_matrix.shape(),
                vec![d, d],
            ),
            (
                "transition covariance",
                self.transition_covariance.shape(),
                vec![d, d],
            ),
            (
                "observation matrix",
                self.observation_matrix.shape(),
                vec![m, d],
            ),
            (
                "observation covariance",
                self.observation_covariance.shape(),
                vec![m, m],
            ),
        ]
    }

    /// Checks the observations' shape [T, m], returning (T, d, m).
    fn steps(&self, x_shape: &[usize]) -> Result<(usize, usize, usize)> {
        let (d, m) = self.dimensions()?;
        match *x_shape {
            [t, mm] if mm == m => Ok((t, d, m)),
            _ => Err(DistributionError::InvalidValue(format!(
                "the observations must have shape [T, {m}], not {x_shape:?}"
            ))),
        }
    }

    /// The parameters as matrices, the initial mean as a column.
    fn matrices(&self, d: usize) -> Result<Matrices<DoubleTensor>> {
        Ok(Matrices {
            initial_mean: self.initial_mean.reshape(&[d as isize, 1])?,
            initial_covariance: self.initial_covariance.clone().into_owned(),
            f: self.transition_matrix.clone().into_owned(),
            q: self.transition_covariance.clone().into_owned(),
            h: self.observation_matrix.clone().into_owned(),
            r: self.observation_covariance.clone().into_owned(),
            identity: DoubleTensor::eye(d)?,
        })
    }

    fn forward(&self, x: &DoubleTensor, keep_going: bool) -> Result<Forward<DoubleTensor>> {
        let (t_len, d, obs) = self.steps(x.shape())?;
        let observations = x
            .as_slice()
            .chunks(obs)
            .take(t_len)
            .map(|y| Ok(DoubleTensor::new(&[obs, 1], y.to_vec())?));
        kalman_forward(&self.matrices(d)?, observations, keep_going)
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
            means[t * d..(t + 1) * d].copy_from_slice(step.filtered_mean.as_slice());
            covariances[t * d * d..(t + 1) * d * d]
                .copy_from_slice(step.filtered_covariance.as_slice());
        }
        Ok(KalmanFilter {
            log_likelihood: forward.log_likelihood,
            means: DoubleTensor::new(&[t_len, d], means)?,
            covariances: DoubleTensor::new(&[t_len, d, d], covariances)?,
        })
    }

    /// The filter over a series whose every innovation covariance is positive definite (an error otherwise), with
    /// the model's matrices, the state dimension and the number of steps.
    fn complete_filter(
        &self,
        x: &DoubleTensor,
    ) -> Result<(Matrices<DoubleTensor>, Forward<DoubleTensor>, usize, usize)> {
        let (t_len, d, _) = self.steps(x.shape())?;
        let forward = self.forward(x, false)?;
        if forward.steps.len() < t_len || !forward.log_likelihood.is_finite() {
            return impossible_observations();
        }
        Ok((self.matrices(d)?, forward, t_len, d))
    }

    /// The smoother's gain J_t = P_t|t Fᵀ P_{t+1|t}⁻¹ for each step but the last (a pseudo-inverse where the
    /// predicted covariance is singular), and the covariance of x_t given x_{t+1} and y_0..t in the Joseph-like
    /// form (I − J F) P_t|t (I − J F)ᵀ + J Q Jᵀ, which is positive semidefinite by construction (and equals
    /// P_t|t − J P_{t+1|t} Jᵀ).
    fn smoother_gains(
        p: &Matrices<DoubleTensor>,
        steps: &[Step<DoubleTensor>],
    ) -> Result<Vec<(DoubleTensor, DoubleTensor)>> {
        steps
            .windows(2)
            .map(|w| {
                let inverse = psd_pseudo_inverse(&w[1].covariance)?;
                let j = w[0]
                    .filtered_covariance
                    .matmul_transposed(&p.f, false, true)?
                    .matmul(&inverse)?;
                let a = p.identity.sub(&j.matmul(&p.f)?)?;
                let conditional = sym(&a
                    .matmul(&w[0].filtered_covariance)?
                    .matmul_transposed(&a, false, true)?
                    .add(&j.matmul(&p.q)?.matmul_transposed(&j, false, true)?)?)?;
                Ok((j, conditional))
            })
            .collect()
    }

    /// The Rauch–Tung–Striebel smoother: the mean and covariance of the state at each step given every observation
    /// `x` ([T, m]). It runs the same Joseph-form filter as [`Self::log_prob`], then recurses backwards with
    /// m_t|T = m_t|t + J_t (m_{t+1}|T − m_{t+1}|t) and P_t|T = C_t + J_t P_{t+1}|T J_tᵀ, where J_t = P_t|t Fᵀ
    /// P_{t+1}|t⁻¹ and C_t = (I − J_t F) P_t|t (I − J_t F)ᵀ + J_t Q J_tᵀ (so the smoothed covariances stay positive
    /// semidefinite). A singular predicted covariance P_{t+1}|t (a deterministic part of the state) is inverted
    /// with its pseudo-inverse.
    ///
    /// # Errors
    ///
    /// An error if the shapes are invalid or the observations are impossible (an innovation covariance is not
    /// positive definite).
    pub fn smooth(&self, x: &DoubleTensor) -> Result<KalmanSmoother> {
        let (p, forward, t_len, d) = self.complete_filter(x)?;
        let steps = &forward.steps;
        let mut means = vec![0.0; t_len * d];
        let mut covariances = vec![0.0; t_len * d * d];
        if let Some(last) = steps.last() {
            let gains = Self::smoother_gains(&p, steps)?;
            let mut mean = last.filtered_mean.clone();
            let mut covariance = last.filtered_covariance.clone();
            for t in (0..t_len).rev() {
                if t + 1 < t_len {
                    let (j, conditional) = &gains[t];
                    mean = steps[t]
                        .filtered_mean
                        .add(&j.matmul(&mean.sub(&steps[t + 1].mean)?)?)?;
                    covariance = sym(&conditional
                        .add(&j.matmul(&covariance)?.matmul_transposed(j, false, true)?)?)?;
                }
                means[t * d..(t + 1) * d].copy_from_slice(mean.as_slice());
                covariances[t * d * d..(t + 1) * d * d].copy_from_slice(covariance.as_slice());
            }
        }
        Ok(KalmanSmoother {
            log_likelihood: forward.log_likelihood,
            means: DoubleTensor::new(&[t_len, d], means)?,
            covariances: DoubleTensor::new(&[t_len, d, d], covariances)?,
        })
    }

    /// Draws the hidden states from their joint posterior given the observations `x` ([T, m]), by forward
    /// filtering and backward sampling (Carter and Kohn, 1994; Frühwirth-Schnatter, 1994): the last state from
    /// its filtered distribution N(m_{T−1}|T−1, P_{T−1}|T−1), then each earlier one from
    /// N(m_t|t + J_t (x_{t+1} − m_{t+1}|t), C_t) with the smoother's gain and conditional covariance (see
    /// [`Self::smooth`]). Returns the states, [T, d].
    ///
    /// # Errors
    ///
    /// As for [`Self::smooth`].
    pub fn sample_posterior_states(
        &self,
        x: &DoubleTensor,
        rng: &mut KeanuRng,
    ) -> Result<DoubleTensor> {
        let (p, forward, t_len, d) = self.complete_filter(x)?;
        let steps = &forward.steps;
        let mut states = vec![0.0; t_len * d];
        if let Some(last) = steps.last() {
            let gains = Self::smoother_gains(&p, steps)?;
            let mut state = gaussian_draw(
                &last.filtered_mean,
                &last.filtered_covariance,
                "a filtered covariance",
                rng,
            )?;
            states[(t_len - 1) * d..].copy_from_slice(state.as_slice());
            for t in (0..t_len - 1).rev() {
                let (j, conditional) = &gains[t];
                let mean = steps[t]
                    .filtered_mean
                    .add(&j.matmul(&state.sub(&steps[t + 1].mean)?)?)?;
                state = gaussian_draw(&mean, conditional, "a smoothing covariance", rng)?;
                states[t * d..(t + 1) * d].copy_from_slice(state.as_slice());
            }
        }
        Ok(DoubleTensor::new(&[t_len, d], states)?)
    }

    /// The model's own matrices as `horizon` future steps: forecasting a time-invariant model.
    fn invariant_future(&self, horizon: usize) -> Vec<FutureStep> {
        let step = FutureStep::new(
            self.transition_matrix.clone().into_owned(),
            self.transition_covariance.clone().into_owned(),
            self.observation_matrix.clone().into_owned(),
            self.observation_covariance.clone().into_owned(),
        );
        vec![step; horizon]
    }

    /// Checks each future step's shapes against the model's dimensions.
    fn check_future(&self, future: &[FutureStep], d: usize, m: usize) -> Result<()> {
        for (k, step) in future.iter().enumerate() {
            let shapes = [
                ("transition matrix", step.transition_matrix.shape(), [d, d]),
                (
                    "transition covariance",
                    step.transition_covariance.shape(),
                    [d, d],
                ),
                (
                    "observation matrix",
                    step.observation_matrix.shape(),
                    [m, d],
                ),
                (
                    "observation covariance",
                    step.observation_covariance.shape(),
                    [m, m],
                ),
            ];
            for (what, shape, expected) in shapes {
                if shape != expected {
                    return invalid_parameter(format!(
                        "the {what} of future step {k} must have shape {expected:?}, not {shape:?}"
                    ));
                }
            }
        }
        Ok(())
    }

    /// The state's distribution at the last observed step given every observation, or None for an empty series
    /// (whose first future step is the initial state).
    fn last_filtered(&self, x: &DoubleTensor) -> Result<Option<(DoubleTensor, DoubleTensor)>> {
        let (_, forward, _, _) = self.complete_filter(x)?;
        Ok(forward
            .steps
            .into_iter()
            .last()
            .map(|s| (s.filtered_mean, s.filtered_covariance)))
    }

    /// The predictive distribution of the next `horizon` steps after the observations `x` ([T, m]) of a
    /// time-invariant model: [`Self::forecast_with`] with the model's own matrices at every future step. Each step's
    /// state mean moves by m ← F m and its covariance by P ← F P Fᵀ + Q from the last filtered state, so for a
    /// local level model (F = H = 1) the k-th step's observation variance is P_{T−1}|T−1 + k Q + R.
    ///
    /// # Errors
    ///
    /// As for [`Self::forecast_with`].
    pub fn forecast(&self, x: &DoubleTensor, horizon: usize) -> Result<StateSpaceForecast> {
        self.forecast_with(x, &self.invariant_future(horizon))
    }

    /// The predictive distribution of the steps after the observations `x` ([T, m]), one per entry of `future`,
    /// which gives each step's matrices (for a time-varying model, or regressors known in advance): from the
    /// filtered state at the last observed step, the state moves by step k's transition matrix and noise and is
    /// observed with its observation matrix and noise. For an empty series (T = 0) the first future step is the
    /// initial state, and its transition matrix and covariance are not used.
    ///
    /// # Errors
    ///
    /// An error if the shapes are invalid (the future steps' included) or the observations are impossible.
    pub fn forecast_with(
        &self,
        x: &DoubleTensor,
        future: &[FutureStep],
    ) -> Result<StateSpaceForecast> {
        let (_, d, m) = self.steps(x.shape())?;
        self.check_future(future, d, m)?;
        let h = future.len();
        let mut state = self.last_filtered(x)?;
        let (mut sm, mut sc, mut om, mut oc) = (
            Vec::with_capacity(h * d),
            Vec::with_capacity(h * d * d),
            Vec::with_capacity(h * m),
            Vec::with_capacity(h * m * m),
        );
        for step in future {
            let (mean, covariance) = match state {
                Some((mean, covariance)) => {
                    let f = &step.transition_matrix;
                    (
                        f.matmul(&mean)?,
                        sym(&f
                            .matmul(&covariance)?
                            .matmul_transposed(f, false, true)?
                            .add(&step.transition_covariance)?)?,
                    )
                }
                None => (
                    self.initial_mean.reshape(&[d as isize, 1])?,
                    self.initial_covariance.clone().into_owned(),
                ),
            };
            let hm = &step.observation_matrix;
            sm.extend_from_slice(mean.as_slice());
            sc.extend_from_slice(covariance.as_slice());
            om.extend_from_slice(hm.matmul(&mean)?.as_slice());
            oc.extend_from_slice(
                sym(&hm
                    .matmul(&covariance)?
                    .matmul_transposed(hm, false, true)?
                    .add(&step.observation_covariance)?)?
                .as_slice(),
            );
            state = Some((mean, covariance));
        }
        Ok(StateSpaceForecast {
            state_means: DoubleTensor::new(&[h, d], sm)?,
            state_covariances: DoubleTensor::new(&[h, d, d], sc)?,
            observation_means: DoubleTensor::new(&[h, m], om)?,
            observation_covariances: DoubleTensor::new(&[h, m, m], oc)?,
        })
    }

    /// Draws one path of the next `horizon` steps after the observations `x` of a time-invariant model, jointly
    /// over the steps: [`Self::sample_forecast_with`] with the model's own matrices.
    ///
    /// # Errors
    ///
    /// As for [`Self::sample_forecast_with`].
    pub fn sample_forecast(
        &self,
        x: &DoubleTensor,
        horizon: usize,
        rng: &mut KeanuRng,
    ) -> Result<StateSpacePath> {
        self.sample_forecast_with(x, &self.invariant_future(horizon), rng)
    }

    /// Draws one path of the steps after the observations `x` ([T, m]), one per entry of `future` (see
    /// [`Self::forecast_with`]): the state at the last observed step from its filtered distribution, then each
    /// future state and observation by simulating the model forward. Unlike draws from each step's marginal in
    /// [`StateSpaceForecast`], the steps of one path are jointly distributed (for intervals of sums or maxima).
    ///
    /// # Errors
    ///
    /// As for [`Self::forecast_with`], and if a covariance is not positive semidefinite.
    pub fn sample_forecast_with(
        &self,
        x: &DoubleTensor,
        future: &[FutureStep],
        rng: &mut KeanuRng,
    ) -> Result<StateSpacePath> {
        let (_, d, m) = self.steps(x.shape())?;
        self.check_future(future, d, m)?;
        let h = future.len();
        let mut state = self
            .last_filtered(x)?
            .map(|(mean, covariance)| {
                gaussian_draw(&mean, &covariance, "a filtered covariance", rng)
            })
            .transpose()?;
        let (mut states, mut observations) = (Vec::with_capacity(h * d), Vec::with_capacity(h * m));
        for step in future {
            let next = match state {
                Some(previous) => step
                    .transition_matrix
                    .matmul(&previous)?
                    .add(&gaussian_draw(
                        &DoubleTensor::zeros(&[d, 1]),
                        &step.transition_covariance,
                        "the transition covariance",
                        rng,
                    )?)?,
                None => gaussian_draw(
                    &self.initial_mean.reshape(&[d as isize, 1])?,
                    &self.initial_covariance,
                    "the initial covariance",
                    rng,
                )?,
            };
            let y = step.observation_matrix.matmul(&next)?.add(&gaussian_draw(
                &DoubleTensor::zeros(&[m, 1]),
                &step.observation_covariance,
                "the observation covariance",
                rng,
            )?)?;
            states.extend_from_slice(next.as_slice());
            observations.extend_from_slice(y.as_slice());
            state = Some(next);
        }
        Ok(StateSpacePath {
            states: DoubleTensor::new(&[h, d], states)?,
            observations: DoubleTensor::new(&[h, m], observations)?,
        })
    }

    /// The derivatives of [`Self::log_prob`] with respect to the observations and every parameter, by reverse-mode
    /// differentiation of the filter. An error if an innovation covariance is not positive definite (the log
    /// density is -∞ there).
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<LinearGaussianStateSpaceGradient> {
        let (t_len, d, obs) = self.steps(x.shape())?;
        let matrices = self.matrices(d)?;
        let forward = self.forward(x, false)?;
        if forward.steps.len() < t_len {
            return not_positive_definite();
        }
        let adjoints = kalman_reverse(&matrices, &forward.steps)?;
        let dx: Vec<f64> = adjoints
            .x
            .iter()
            .flat_map(|e| e.as_slice().iter().copied())
            .collect();
        Ok(LinearGaussianStateSpaceGradient {
            x: DoubleTensor::new(&[t_len, obs], dx)?,
            initial_mean: adjoints.initial_mean.reshape(&[d as isize])?,
            initial_covariance: adjoints.initial_covariance,
            transition_matrix: adjoints.f,
            transition_covariance: adjoints.q,
            observation_matrix: adjoints.h,
            observation_covariance: adjoints.r,
        })
    }

    /// Hessian-vector products of [`Self::log_prob`]: the derivatives of [`Self::d_log_prob`] along `b`
    /// directions, given as tangents of the observations (`x_tangent`, [b, T, m]) and of each parameter
    /// (`parameter_tangents`, in the order of [`Self::new`], each [b, parameter shape...]), None for a zero
    /// tangent. Each field of the result is shaped [b, input shape...]: for each input u, Σᵥ ∂² log p / ∂u ∂v · v̇.
    ///
    /// They are exact: the reverse pass of [`Self::d_log_prob`] differentiated in forward mode, through the
    /// Cholesky inverses of the innovation covariances and the Joseph-form update. As for the gradient, a
    /// covariance's tangent is taken to be symmetric (its symmetric part is used) and the covariances' products
    /// are symmetric.
    ///
    /// # Errors
    ///
    /// An error if `parameter_tangents` does not have six entries or a tangent does not have the shape
    /// [b, input shape...], or if an innovation covariance is not positive definite.
    pub fn hvp(
        &self,
        x: &DoubleTensor,
        x_tangent: Option<&DoubleTensor>,
        parameter_tangents: &[Option<&DoubleTensor>],
        b: usize,
    ) -> Result<LinearGaussianStateSpaceGradient> {
        let (t_len, d, obs) = self.steps(x.shape())?;
        let &[t0, t1, t2, t3, t4, t5] = parameter_tangents else {
            return invalid_parameter(format!(
                "a linear Gaussian state space model has 6 parameters, not {} parameter tangents",
                parameter_tangents.len()
            ));
        };
        let shapes = self.parameter_shapes(d, obs);
        let tangents = [t0, t1, t2, t3, t4, t5];
        let mut parameters = Vec::with_capacity(6);
        for ((what, shape, _), tangent) in shapes.iter().zip(tangents) {
            parameters.push(checked_tangent(what, tangent, b, shape)?);
        }
        let x_tangent = checked_tangent("observations", x_tangent, b, x.shape())?;
        let base = self.matrices(d)?;
        // covariance tangents are taken to be symmetric
        let symmetric = |t: Option<&DoubleTensor>| -> Result<Option<DoubleTensor>> {
            t.map(|t| Ok(t.add(&t.transpose()?)? * 0.5)).transpose()
        };
        let dual =
            |value: DoubleTensor, tangent: Option<DoubleTensor>| Dual::new(value, tangent, b);
        let matrices = Matrices {
            initial_mean: dual(
                base.initial_mean,
                parameters[0]
                    .map(|t| t.reshape(&[b as isize, d as isize, 1]))
                    .transpose()?,
            ),
            initial_covariance: dual(base.initial_covariance, symmetric(parameters[1])?),
            f: dual(base.f, parameters[2].cloned()),
            q: dual(base.q, symmetric(parameters[3])?),
            h: dual(base.h, parameters[4].cloned()),
            r: dual(base.r, symmetric(parameters[5])?),
            identity: dual(base.identity, None),
        };
        let observations = (0..t_len).map(|t| {
            let value =
                DoubleTensor::new(&[obs, 1], x.as_slice()[t * obs..(t + 1) * obs].to_vec())?;
            let tangent = x_tangent
                .map(|xt| {
                    let data: Vec<f64> = (0..b)
                        .flat_map(|k| {
                            let start = (k * t_len + t) * obs;
                            xt.as_slice()[start..start + obs].iter().copied()
                        })
                        .collect();
                    DoubleTensor::new(&[b, obs, 1], data)
                })
                .transpose()?;
            Ok(dual(value, tangent))
        });
        let forward = kalman_forward(&matrices, observations, false)?;
        if forward.steps.len() < t_len {
            return not_positive_definite();
        }
        let adjoints = kalman_reverse(&matrices, &forward.steps)?;
        let mut dx = vec![0.0; b * t_len * obs];
        for (t, e) in adjoints.x.iter().enumerate() {
            let tangent = e.tangent.as_slice();
            for k in 0..b {
                let start = (k * t_len + t) * obs;
                dx[start..start + obs].copy_from_slice(&tangent[k * obs..(k + 1) * obs]);
            }
        }
        Ok(LinearGaussianStateSpaceGradient {
            x: DoubleTensor::new(&[b, t_len, obs], dx)?,
            initial_mean: adjoints
                .initial_mean
                .tangent
                .reshape(&[b as isize, d as isize])?,
            initial_covariance: adjoints.initial_covariance.tangent,
            transition_matrix: adjoints.f.tangent,
            transition_covariance: adjoints.q.tangent,
            observation_matrix: adjoints.h.tangent,
            observation_covariance: adjoints.r.tangent,
        })
    }

    /// Draws observations of shape [T, m] by simulating the state forward. The covariances may be singular
    /// (positive semidefinite) here.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let (t_len, d, obs) = self.steps(shape)?;
        let (f, h) = (&*self.transition_matrix, &*self.observation_matrix);
        let initial = psd_factor(&self.initial_covariance, "the initial covariance")?;
        let transition = psd_factor(&self.transition_covariance, "the transition covariance")?;
        let observation = psd_factor(&self.observation_covariance, "the observation covariance")?;
        let noise = |factor: &DoubleTensor, n: usize, rng: &mut KeanuRng| -> Result<DoubleTensor> {
            let z: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
            Ok(factor.matmul(&DoubleTensor::new(&[n, 1], z)?)?)
        };
        let mut state = self
            .initial_mean
            .reshape(&[d as isize, 1])?
            .add(&noise(&initial, d, rng)?)?;
        let mut out = Vec::with_capacity(t_len * obs);
        for t in 0..t_len {
            let y = h.matmul(&state)?.add(&noise(&observation, obs, rng)?)?;
            out.extend_from_slice(y.as_slice());
            if t + 1 < t_len {
                state = f.matmul(&state)?.add(&noise(&transition, d, rng)?)?;
            }
        }
        Ok(DoubleTensor::new(&[t_len, obs], out)?)
    }
}

/// A tangent, checked to have the shape [b, shape...].
fn checked_tangent<'t>(
    what: &str,
    tangent: Option<&'t DoubleTensor>,
    b: usize,
    shape: &[usize],
) -> Result<Option<&'t DoubleTensor>> {
    let Some(tangent) = tangent else {
        return Ok(None);
    };
    let mut expected = vec![b];
    expected.extend_from_slice(shape);
    if tangent.shape() != expected {
        return invalid_parameter(format!(
            "the tangent of the {what} must have shape {expected:?}, not {:?}",
            tangent.shape()
        ));
    }
    Ok(Some(tangent))
}

/// The error of a derivative where an innovation covariance is not positive definite.
fn not_positive_definite<T>() -> Result<T> {
    invalid_parameter(
        "an innovation covariance H P Hᵀ + R is not positive definite, so the log density has no gradient",
    )
}
