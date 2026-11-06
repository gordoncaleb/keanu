//! Decomposition and forecasting of a [structural time series](keanu_graph::sts) over posterior draws, as
//! `tfp.sts.decompose_by_component` and `tfp.sts.forecast` do.
//!
//! [`StructuralPosterior`] evaluates the model's state-space parameters and regression weights at each draw. At
//! each draw the components' states have an exact Gaussian posterior (the Kalman smoother of the series minus its
//! regression) and the future an exact Gaussian predictive distribution; over the draws these are mixtures. Means
//! are the mixtures' means (the per-draw means averaged, a Rao–Blackwellised estimate) and intervals the mixtures'
//! central quantiles, found by bisection on the mixture's distribution function, so they include both the
//! parameters' uncertainty and the states' (and, for forecasts of the series, the observation noise).
//!
//! ```
//! use keanu_algorithms::NetworkSamples;
//! use keanu_algorithms::sts::StructuralPosterior;
//! use keanu_graph::sts::{Component, StructuralModel};
//! use keanu_graph::{GraphBuilder, Model};
//! use keanu_tensor::DoubleTensor;
//!
//! let y = [1.0, 1.4, 0.9, 1.6, 1.2, 1.9];
//! let b = GraphBuilder::new();
//! let sts = StructuralModel::new(&b, &y)
//!     .with(Component::local_level())
//!     .build()?;
//! let (graph, mut state) = b.build()?;
//! let model = Model::new(graph, &state)?;
//! let nodes: Vec<_> = sts.parameters().into_iter().map(|(_, id)| id).collect();
//! // two "draws" of the level and noise scales
//! let mut samples = NetworkSamples::new(model.graph(), &nodes)?;
//! for (level, noise) in [(0.2, 0.3), (0.4, 0.2)] {
//!     state.set_and_cascade(model.graph(), nodes[0], DoubleTensor::scalar(level))?;
//!     state.set_and_cascade(model.graph(), nodes[1], DoubleTensor::scalar(noise))?;
//!     samples.record(&state, model.log_prob(&state)?)?;
//! }
//! let posterior = StructuralPosterior::new(&model, &state, &sts, &samples)?;
//! let parts = posterior.decompose(0.9)?;
//! let level = parts.component("level").unwrap();
//! assert!(level.lower[0] < level.mean[0] && level.mean[0] < level.upper[0]);
//! let forecast = posterior.forecast(3, &[], 0.9)?;
//! assert_eq!(forecast.observations.mean.len(), 3);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{AlgorithmError, Result};
use crate::samples::{ChainSamples, NetworkSamples};
use crate::timeseries::{chains, f64_value, values_at_draws};
use keanu_distributions::{KeanuRng, LinearGaussianStateSpace};
use keanu_graph::sts::{ComponentKind, StructuralTimeSeries};
use keanu_graph::{Model, State};
use keanu_tensor::DoubleTensor;
use keanu_tensor::special::{ndtr, ndtri};

/// A series' posterior mean and central interval at each step.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Band {
    /// The posterior mean at each step.
    pub mean: Vec<f64>,
    /// The lower end of the central interval at each step.
    pub lower: Vec<f64>,
    /// The upper end of the central interval at each step.
    pub upper: Vec<f64>,
}

/// The posterior of each component's contribution to the observed steps, from [`StructuralPosterior::decompose`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Decomposition {
    /// The intervals' probability.
    pub level: f64,
    /// Each component's contribution, by name, in the model's order.
    pub components: Vec<(String, Band)>,
    /// The smoothed mean of the series without observation noise: the sum of the components (its mean is the sum
    /// of their means).
    pub total: Band,
}

impl Decomposition {
    /// The component with this name.
    pub fn component(&self, name: &str) -> Option<&Band> {
        find(&self.components, name)
    }
}

/// The predictive distribution of the steps after the series, from [`StructuralPosterior::forecast`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Forecast {
    /// The intervals' probability.
    pub level: f64,
    /// The series itself, observation noise included.
    pub observations: Band,
    /// Each component's contribution, by name, in the model's order.
    pub components: Vec<(String, Band)>,
}

impl Forecast {
    /// The component with this name.
    pub fn component(&self, name: &str) -> Option<&Band> {
        find(&self.components, name)
    }
}

fn find<'a>(components: &'a [(String, Band)], name: &str) -> Option<&'a Band> {
    components.iter().find(|(n, _)| n == name).map(|(_, b)| b)
}

/// One draw: the state-space model's parameters, the series minus the regression ([T, 1]) and each regression
/// component's weights.
#[derive(Debug, Clone, PartialEq)]
struct Draw {
    parameters: [DoubleTensor; 6],
    residual: DoubleTensor,
    weights: Vec<DoubleTensor>,
}

impl Draw {
    fn model(&self) -> LinearGaussianStateSpace<'_> {
        let p = &self.parameters;
        LinearGaussianStateSpace::new(&p[0], &p[1], &p[2], &p[3], &p[4], &p[5])
    }
}

/// A structural time series at each posterior draw. See the [module documentation](self).
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralPosterior {
    sts: StructuralTimeSeries,
    draws: Vec<Draw>,
}

/// Each step's mean and variance of one quantity at one draw.
type Moments = Vec<(f64, f64)>;

impl StructuralPosterior {
    /// The model `sts` (built on the graph of `model`) at each draw in `samples`: its state-space parameters and
    /// regression weights evaluated from the draw, set in a copy of `state`.
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidArgument`] if `samples` doesn't record every latent the parameters depend on (naming
    /// the missing ones), an unknown-node error if a node isn't in the graph, and any error evaluating the
    /// parameters.
    pub fn new(
        model: &Model,
        state: &State,
        sts: &StructuralTimeSeries,
        samples: &NetworkSamples,
    ) -> Result<Self> {
        Self::from_samples(model, state, sts, &[samples])
    }

    /// [`Self::new`] over every chain's draws, chain by chain.
    ///
    /// # Errors
    ///
    /// As for [`Self::new`].
    pub fn from_chains(
        model: &Model,
        state: &State,
        sts: &StructuralTimeSeries,
        samples: &ChainSamples,
    ) -> Result<Self> {
        Self::from_samples(model, state, sts, &chains(samples))
    }

    fn from_samples(
        model: &Model,
        state: &State,
        sts: &StructuralTimeSeries,
        samples: &[&NetworkSamples],
    ) -> Result<Self> {
        let mut targets = sts.state_space_parameters().to_vec();
        let regressions: Vec<(&DoubleTensor, _)> = sts
            .components()
            .iter()
            .filter_map(|c| Some((c.design()?, c.weights()?)))
            .collect();
        targets.extend(regressions.iter().map(|(_, w)| *w));
        let observed = DoubleTensor::new(&[sts.steps(), 1], sts.observed().to_vec())?;
        let draws = values_at_draws(
            model,
            state,
            &targets,
            "the structural time series",
            samples,
        )?
        .into_iter()
        .map(|values| {
            let tensors: Vec<DoubleTensor> = values
                .iter()
                .map(|v| f64_value(v).cloned())
                .collect::<Result<_>>()?;
            let (parameters, weights) = tensors.split_at(6);
            let parameters: [DoubleTensor; 6] =
                parameters.to_vec().try_into().expect("six parameters");
            let mut residual = observed.clone();
            for ((design, _), w) in regressions.iter().zip(weights) {
                residual = residual.sub(&regression(design, w)?)?;
            }
            Ok(Draw {
                parameters,
                residual,
                weights: weights.to_vec(),
            })
        })
        .collect::<Result<_>>()?;
        Ok(Self {
            sts: sts.clone(),
            draws,
        })
    }

    /// The number of draws.
    pub fn len(&self) -> usize {
        self.draws.len()
    }

    /// Whether there are no draws.
    pub fn is_empty(&self) -> bool {
        self.draws.is_empty()
    }

    /// The structural time series.
    pub fn structure(&self) -> &StructuralTimeSeries {
        &self.sts
    }

    /// The state-space model at draw `index` (None past the last draw), whose observations are the series minus
    /// its regression ([`Self::residual`]).
    pub fn model(&self, index: usize) -> Option<LinearGaussianStateSpace<'_>> {
        self.draws.get(index).map(Draw::model)
    }

    /// The series minus its regression at draw `index`, [T, 1] (the series itself without a regression).
    pub fn residual(&self, index: usize) -> Option<&DoubleTensor> {
        self.draws.get(index).map(|d| &d.residual)
    }

    /// The posterior of each component's contribution to the observed steps, with central intervals of
    /// probability `level`: at each draw, the Kalman smoother's mean and variance of the component's observed part
    /// (its observation row times its states; a regression's X β exactly), mixed over the draws. The components'
    /// means sum to the smoothed mean of the series, `total`.
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidArgument`] if there are no draws or `level` is not in (0, 1), and an error if a
    /// draw's series is impossible under its parameters.
    pub fn decompose(&self, level: f64) -> Result<Decomposition> {
        self.check(level)?;
        let t_len = self.sts.steps();
        let d = self.sts.state_dimension();
        let total_row = joint_observation(&self.sts);
        let mut parts: Vec<Vec<Moments>> = vec![Vec::new(); self.sts.components().len()];
        let mut totals: Vec<Moments> = Vec::new();
        for draw in &self.draws {
            let smoothed = draw.model().smooth(&draw.residual)?;
            let (means, covariances) = (smoothed.means.as_slice(), smoothed.covariances.as_slice());
            let at = |t: usize, row: &[f64], states: std::ops::Range<usize>| {
                state_moments(&means[t * d..], &covariances[t * d * d..], d, row, states)
            };
            let mut weights = draw.weights.iter();
            let mut total = vec![(0.0, 0.0); t_len];
            for (c, part) in self.sts.components().iter().zip(&mut parts) {
                let moments: Moments = match c.design() {
                    Some(design) => {
                        let w = weights.next().expect("a weight per regression");
                        let offset = regression(design, w)?;
                        offset.as_slice().iter().map(|&m| (m, 0.0)).collect()
                    }
                    None => (0..t_len)
                        .map(|t| at(t, c.observation(), c.states()))
                        .collect(),
                };
                for (sum, (m, _)) in total.iter_mut().zip(&moments) {
                    sum.0 += m;
                }
                part.push(moments);
            }
            for (t, sum) in total.iter_mut().enumerate() {
                sum.1 = at(t, &total_row, 0..d).1;
            }
            totals.push(total);
        }
        Ok(Decomposition {
            level,
            components: self.named(parts, level),
            total: band(&totals, level),
        })
    }

    /// The predictive distribution of the next `horizon` steps, with central intervals of probability `level`:
    /// at each draw, the state-space model's forecast from the last filtered state (its matrices are
    /// time-invariant) plus each regression's future X β, mixed over the draws. `covariates` gives each regression
    /// component's future design matrix, [horizon, k], in the model's order (empty without a regression).
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidArgument`] if there are no draws, `level` is not in (0, 1), or `covariates` doesn't
    /// have one matrix of the right shape per regression, and an error if a draw's series is impossible.
    pub fn forecast(
        &self,
        horizon: usize,
        covariates: &[DoubleTensor],
        level: f64,
    ) -> Result<Forecast> {
        self.check(level)?;
        self.check_covariates(horizon, covariates)?;
        let d = self.sts.state_dimension();
        let mut parts: Vec<Vec<Moments>> = vec![Vec::new(); self.sts.components().len()];
        let mut observations: Vec<Moments> = Vec::new();
        for draw in &self.draws {
            let forecast = draw.model().forecast(&draw.residual, horizon)?;
            let (means, covariances) = (
                forecast.state_means.as_slice(),
                forecast.state_covariances.as_slice(),
            );
            let mut series: Moments = forecast
                .observation_means
                .as_slice()
                .iter()
                .zip(forecast.observation_covariances.as_slice())
                .map(|(&m, &v)| (m, v))
                .collect();
            let mut future = covariates.iter().zip(&draw.weights);
            for (c, part) in self.sts.components().iter().zip(&mut parts) {
                let moments: Moments = match c.design() {
                    Some(_) => {
                        let (x, w) = future.next().expect("checked");
                        let offset = regression(x, w)?;
                        for (s, m) in series.iter_mut().zip(offset.as_slice()) {
                            s.0 += m;
                        }
                        offset.as_slice().iter().map(|&m| (m, 0.0)).collect()
                    }
                    None => (0..horizon)
                        .map(|k| {
                            state_moments(
                                &means[k * d..],
                                &covariances[k * d * d..],
                                d,
                                c.observation(),
                                c.states(),
                            )
                        })
                        .collect(),
                };
                part.push(moments);
            }
            observations.push(series);
        }
        Ok(Forecast {
            level,
            observations: band(&observations, level),
            components: self.named(parts, level),
        })
    }

    /// One simulated path of the next `horizon` steps of the series for each draw (observation noise included),
    /// jointly over the steps ([`LinearGaussianStateSpace::sample_forecast`] plus the regressions' future X β), for
    /// intervals of totals or maxima. `covariates` is as for [`Self::forecast`]. Each path has `horizon` values.
    ///
    /// # Errors
    ///
    /// As for [`Self::forecast`], except that a `level` isn't needed.
    pub fn sample_forecast(
        &self,
        horizon: usize,
        covariates: &[DoubleTensor],
        rng: &mut KeanuRng,
    ) -> Result<Vec<Vec<f64>>> {
        self.check_covariates(horizon, covariates)?;
        self.draws
            .iter()
            .map(|draw| {
                let path = draw.model().sample_forecast(&draw.residual, horizon, rng)?;
                let mut values = path.observations.as_slice().to_vec();
                for (x, w) in covariates.iter().zip(&draw.weights) {
                    for (v, m) in values.iter_mut().zip(regression(x, w)?.as_slice()) {
                        *v += m;
                    }
                }
                Ok(values)
            })
            .collect()
    }

    fn check(&self, level: f64) -> Result<()> {
        if self.draws.is_empty() {
            return Err(AlgorithmError::InvalidArgument("there are no draws".into()));
        }
        if !(level > 0.0 && level < 1.0) {
            return Err(AlgorithmError::InvalidArgument(format!(
                "the interval probability must be in (0, 1), not {level}"
            )));
        }
        Ok(())
    }

    fn check_covariates(&self, horizon: usize, covariates: &[DoubleTensor]) -> Result<()> {
        let designs: Vec<(&str, usize)> = self
            .sts
            .components()
            .iter()
            .filter_map(|c| match c.kind() {
                ComponentKind::Regression { covariates } => Some((c.name(), covariates)),
                _ => None,
            })
            .collect();
        if designs.len() != covariates.len() {
            return Err(AlgorithmError::InvalidArgument(format!(
                "the model has {} regression components but {} future design matrices were given",
                designs.len(),
                covariates.len()
            )));
        }
        for ((name, k), x) in designs.iter().zip(covariates) {
            if x.shape() != [horizon, *k] {
                return Err(AlgorithmError::InvalidArgument(format!(
                    "the future design matrix of {name:?} must have shape [{horizon}, {k}], not {:?}",
                    x.shape()
                )));
            }
        }
        Ok(())
    }

    fn named(&self, parts: Vec<Vec<Moments>>, level: f64) -> Vec<(String, Band)> {
        self.sts
            .components()
            .iter()
            .zip(parts)
            .map(|(c, draws)| (c.name().to_string(), band(&draws, level)))
            .collect()
    }
}

/// X w as a column, [rows, 1].
fn regression(design: &DoubleTensor, weights: &DoubleTensor) -> Result<DoubleTensor> {
    let k = weights.len() as isize;
    Ok(design.matmul(&weights.reshape(&[k, 1])?)?)
}

/// The observation row of the whole state: every component's row in its states.
fn joint_observation(sts: &StructuralTimeSeries) -> Vec<f64> {
    let mut row = vec![0.0; sts.state_dimension()];
    for c in sts.components() {
        row[c.states()].copy_from_slice(c.observation());
    }
    row
}

/// The mean and variance of `row` times the states `states` of a state with this mean (`[d]`, at the start of
/// `mean`) and covariance ([d, d], at the start of `covariance`).
fn state_moments(
    mean: &[f64],
    covariance: &[f64],
    d: usize,
    row: &[f64],
    states: std::ops::Range<usize>,
) -> (f64, f64) {
    let index: Vec<usize> = states.collect();
    let m = index.iter().zip(row).map(|(&i, h)| h * mean[i]).sum();
    let mut v = 0.0;
    for (&i, hi) in index.iter().zip(row) {
        for (&j, hj) in index.iter().zip(row) {
            v += hi * covariance[i * d + j] * hj;
        }
    }
    (m, v.max(0.0))
}

/// The mean and central interval of probability `level` at each step of a mixture over draws of Gaussians, given
/// as each draw's (mean, variance) at each step.
fn band(draws: &[Moments], level: f64) -> Band {
    let steps = draws.first().map_or(0, Vec::len);
    let tail = (1.0 - level) / 2.0;
    let mut out = Band {
        mean: Vec::with_capacity(steps),
        lower: Vec::with_capacity(steps),
        upper: Vec::with_capacity(steps),
    };
    let mut points = Vec::with_capacity(draws.len());
    for t in 0..steps {
        points.clear();
        points.extend(draws.iter().map(|d| (d[t].0, d[t].1.sqrt())));
        out.mean
            .push(points.iter().map(|p| p.0).sum::<f64>() / points.len() as f64);
        out.lower.push(mixture_quantile(&points, tail));
        out.upper.push(mixture_quantile(&points, 1.0 - tail));
    }
    out
}

/// The `p` quantile of an equally weighted mixture of Gaussians N(mean, sd²) (a point mass where sd is 0), by
/// bisection between the smallest and largest of the components' own `p` quantiles, which bracket it, to a
/// millionth of that bracket.
fn mixture_quantile(points: &[(f64, f64)], p: f64) -> f64 {
    let z = ndtri(p);
    let (mut lo, mut hi) = points
        .iter()
        .map(|(m, s)| m + z * s)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), q| {
            (lo.min(q), hi.max(q))
        });
    let n = points.len() as f64;
    let cdf = |x: f64| {
        points
            .iter()
            .map(|&(m, s)| {
                if s > 0.0 {
                    ndtr((x - m) / s)
                } else if x >= m {
                    1.0
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / n
    };
    // to a millionth of the bracket's width, well within any Monte Carlo error of the mixture
    let tolerance = 1e-6 * (hi - lo);
    for _ in 0..64 {
        if hi - lo <= tolerance || (hi - lo).is_nan() {
            break;
        }
        let mid = 0.5 * (lo + hi);
        if cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}
