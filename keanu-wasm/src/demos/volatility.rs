//! How much could this position lose tomorrow? Volatility and value at risk from daily returns.
//!
//! Two models of a daily return series with mean 0 and a changing volatility, both fitted by NUTS:
//!
//! - [`VolatilityKind::Garch`]: GARCH(1,1), the industry's standard value-at-risk model. Today's variance is a
//!   constant plus yesterday's squared return and yesterday's variance, σ_t² = ω + α r_{t−1}² + β σ_{t−1}², with
//!   (α, β, 1 − α − β) ~ Dirichlet(2, 6, 2) keeping the process stationary and ω ~ HalfGaussian(1). Three
//!   parameters, so it fits in about a second.
//! - [`VolatilityKind::Stochastic`]: the stochastic volatility model of the Stan User's Guide (as in Keanu's
//!   `stochastic_volatility` example), where the log variance h_t is itself random, a stationary AR(1) process:
//!   mu ~ N(0, 10), (phi + 1) / 2 ~ Beta(20, 1.5), sigma ~ HalfCauchy(5), h_t = mu + phi (h_{t−1} − mu) + sigma eps_t
//!   in non-centred form, r_t ~ N(0, exp(h_t / 2)). One latent per day, so it takes longer.
//!
//! The report gives each day's annualised volatility, a forecast of the next days' volatility, and the next day's
//! value at risk and expected shortfall from the posterior predictive distribution of the return, which carries
//! the uncertainty in both tomorrow's volatility and the model's parameters. Returns are fractions (0.01 is 1%);
//! the models work in percent, where their priors are calibrated.

use crate::run::{ChainRun, Histogram, Interval, draws, mean, quantile, scalar_draws};
use keanu::prelude::*;
use serde::Serialize;

/// Trading days per year, to annualise daily volatility.
pub const TRADING_DAYS: f64 = 252.0;

/// The 200 daily returns bundled with Keanu's stochastic volatility example, simulated from that model with
/// mu = −1, phi = 0.9 and sigma = 0.4 for returns in percent (a daily volatility near 0.6%), as fractions.
pub fn sample_returns() -> Vec<f64> {
    include_str!("../../../keanu/examples/data/returns.csv")
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.rsplit(',').next().unwrap().trim().parse::<f64>().unwrap() / 100.0)
        .collect()
}

/// Which volatility model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolatilityKind {
    /// GARCH(1,1).
    Garch,
    /// Stochastic volatility.
    Stochastic,
}

/// The nodes a model's report reads.
#[derive(Debug, Clone, Copy)]
enum Nodes {
    Garch {
        omega: NodeId,
        alpha: NodeId,
        beta: NodeId,
    },
    Stochastic {
        mu: NodeId,
        phi: NodeId,
        sigma: NodeId,
        log_variance: NodeId,
    },
}

impl Nodes {
    fn recorded(self) -> Vec<NodeId> {
        match self {
            Nodes::Garch { omega, alpha, beta } => vec![omega, alpha, beta],
            Nodes::Stochastic {
                mu,
                phi,
                sigma,
                log_variance,
            } => vec![mu, phi, sigma, log_variance],
        }
    }
}

/// The standard deviation GARCH starts from: that of the first 20 returns (in percent).
fn initial_vol(returns: &[f64]) -> f64 {
    let head = &returns[..returns.len().min(20)];
    let m = mean(head);
    (head.iter().map(|r| (r - m).powi(2)).sum::<f64>() / head.len() as f64)
        .sqrt()
        .max(1e-3)
}

fn build(kind: VolatilityKind, returns: &[f64]) -> keanu::Result<(Model, State, Nodes)> {
    let days = returns.len();
    let b = GraphBuilder::new();
    let nodes = match kind {
        VolatilityKind::Garch => {
            let weights = b.dirichlet(DoubleTensor::vector(vec![2.0, 6.0, 2.0]));
            let omega = b.half_gaussian(1.0);
            b.set_value(weights, DoubleTensor::vector(vec![0.1, 0.8, 0.1]));
            b.set_value(omega, DoubleTensor::scalar(0.1));
            let (alpha, beta) = (weights.slice("0"), weights.slice("1"));
            let r = b.garch11(days, omega, alpha, beta, initial_vol(returns));
            b.observe(r, DoubleTensor::vector(returns.to_vec()));
            Nodes::Garch {
                omega: omega.id(),
                alpha: alpha.id(),
                beta: beta.id(),
            }
        }
        VolatilityKind::Stochastic => {
            let mu = b.gaussian(0.0, 10.0);
            let phi_raw = b.beta(20.0, 1.5);
            let phi = phi_raw * 2.0 - 1.0;
            let sigma = b.half_cauchy(5.0);
            let eps_0 = b.gaussian(0.0, 1.0);
            let eps = b.gaussian_with_shape(&[days], 0.0, 1.0);
            b.set_value(mu, DoubleTensor::scalar(0.0));
            b.set_value(phi_raw, DoubleTensor::scalar(0.9));
            b.set_value(sigma, DoubleTensor::scalar(0.5));
            b.set_value(eps_0, DoubleTensor::scalar(0.0));
            b.set_value(eps, DoubleTensor::vector(vec![0.0; days]));
            let h_0 = mu + sigma / (1.0 - phi * phi).sqrt() * eps_0;
            let level = mu * (1.0 - phi);
            let shocks = sigma * eps;
            let h = b.scan(h_0, &[shocks], &[level, phi], |_, h, shock, c| {
                c[0] + c[1] * h + shock[0]
            });
            let r = b.gaussian_with_shape(&[days], 0.0, (h * 0.5).exp());
            b.observe(r, DoubleTensor::vector(returns.to_vec()));
            Nodes::Stochastic {
                mu: mu.id(),
                phi: phi.id(),
                sigma: sigma.id(),
                log_variance: h.id(),
            }
        }
    };
    let (graph, state) = b.build()?;
    Ok((Model::new(graph, &state)?, state, nodes))
}

/// A volatility model of a return series, fitting or fitted.
#[derive(Debug)]
pub struct VolatilityModel {
    chain: ChainRun,
    nodes: Nodes,
    /// The returns in percent.
    returns: Vec<f64>,
}

/// The forecast for one future day.
#[derive(Debug, Clone, Serialize)]
pub struct ForecastDay {
    /// Days after the last observed return.
    pub ahead: usize,
    /// Annualised volatility.
    pub volatility: Interval,
}

/// The next day's loss at one confidence level.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct TailRisk {
    /// The confidence level, such as 0.99.
    pub level: f64,
    /// The loss exceeded with probability 1 − level.
    pub value_at_risk: f64,
    /// The mean loss beyond the value at risk.
    pub expected_shortfall: f64,
    /// The value at risk of a normal model with the whole sample's standard deviation.
    pub naive_value_at_risk: f64,
}

/// The confidence levels of [`VolatilityReport::tail`].
pub const TAIL_LEVELS: [f64; 4] = [0.95, 0.975, 0.99, 0.995];

/// The standard normal quantiles at [`TAIL_LEVELS`].
const NORMAL_QUANTILES: [f64; 4] = [1.644_853_627, 1.959_963_985, 2.326_347_874, 2.575_829_304];

/// What the risk manager sees.
#[derive(Debug, Clone, Serialize)]
pub struct VolatilityReport {
    /// Each observed day's annualised volatility.
    pub volatility: Vec<Interval>,
    /// The forecast annualised volatility of the next days.
    pub forecast: Vec<ForecastDay>,
    /// The long-run annualised volatility the process reverts to.
    pub long_run_volatility: Interval,
    /// How many days a volatility shock takes to halve.
    pub half_life_days: Interval,
    /// The position's value.
    pub position: f64,
    /// The next day's 95% value at risk, as a loss in the position's currency.
    pub value_at_risk_95: f64,
    /// The next day's 99% value at risk.
    pub value_at_risk_99: f64,
    /// The next day's 97.5% expected shortfall: the mean loss on the worst 2.5% of days.
    pub expected_shortfall_975: f64,
    /// The next day's value at risk and expected shortfall at each of [`TAIL_LEVELS`].
    pub tail: Vec<TailRisk>,
    /// The distribution of tomorrow's loss (negative for a gain), in the position's currency.
    pub loss_histogram: Histogram,
    /// The 99% value at risk of a normal model with the whole sample's standard deviation, for comparison.
    pub naive_value_at_risk_99: f64,
    /// Divergent NUTS transitions after adaptation.
    pub divergences: usize,
    /// The number of posterior draws.
    pub draws: usize,
}

/// Simulates one posterior draw's standard deviations (percent) over the next `horizon` days, tomorrow first.
type Forecaster = Box<dyn Fn(&mut KeanuRng, usize) -> Vec<f64>>;

/// One posterior draw's volatility story, in percent: each observed day's standard deviation, the long-run one and
/// the persistence of shocks.
struct DrawPath {
    sd: Vec<f64>,
    long_run_sd: f64,
    persistence: f64,
}

impl VolatilityModel {
    /// Starts a NUTS fit of `returns` (fractions) with `draws` draws after as many adaptation steps.
    pub fn new(
        kind: VolatilityKind,
        returns: &[f64],
        draws: usize,
        seed: u64,
    ) -> keanu::Result<Self> {
        if returns.len() < 30 {
            return Err(keanu::Error::from(keanu::graph::GraphError::InvalidValue(
                format!("need at least 30 daily returns, got {}", returns.len()),
            )));
        }
        let returns: Vec<f64> = returns.iter().map(|r| r * 100.0).collect();
        let (model, state, nodes) = build(kind, &returns)?;
        let nuts = match kind {
            VolatilityKind::Garch => Nuts::default(),
            VolatilityKind::Stochastic => Nuts::default().with_target_acceptance_prob(0.95),
        };
        let chain = ChainRun::new(model, state, &nodes.recorded(), nuts, draws, draws, seed)?;
        Ok(Self {
            chain,
            nodes,
            returns,
        })
    }

    /// The NUTS chain.
    pub fn chain(&mut self) -> &mut ChainRun {
        &mut self.chain
    }

    /// Each posterior draw's in-sample path, and a sampler of the next `horizon` days' standard deviations (the
    /// first of which is tomorrow's).
    fn paths(&self) -> (Vec<DrawPath>, Vec<Forecaster>) {
        let samples = self.chain.samples();
        match self.nodes {
            Nodes::Garch { omega, alpha, beta } => {
                let (omega, alpha, beta) = (
                    scalar_draws(samples, omega),
                    scalar_draws(samples, alpha),
                    scalar_draws(samples, beta),
                );
                let start = initial_vol(&self.returns);
                let last_return = *self.returns.last().unwrap();
                let mut paths = Vec::with_capacity(omega.len());
                let mut forecasts: Vec<Forecaster> = Vec::new();
                for i in 0..omega.len() {
                    let (w, a, bt) = (omega[i], alpha[i], beta[i]);
                    let mut variance = start * start;
                    let mut sd = Vec::with_capacity(self.returns.len());
                    for t in 0..self.returns.len() {
                        if t > 0 {
                            let previous = self.returns[t - 1];
                            variance = w + a * previous * previous + bt * variance;
                        }
                        sd.push(variance.sqrt());
                    }
                    let last_variance = variance;
                    paths.push(DrawPath {
                        sd,
                        long_run_sd: (w / (1.0 - a - bt)).sqrt(),
                        persistence: a + bt,
                    });
                    forecasts.push(Box::new(move |rng: &mut KeanuRng, horizon: usize| {
                        let (mut variance, mut previous) = (last_variance, last_return);
                        (0..horizon)
                            .map(|_| {
                                variance = w + a * previous * previous + bt * variance;
                                let sd = variance.sqrt();
                                previous = sd * rng.next_gaussian();
                                sd
                            })
                            .collect()
                    }));
                }
                (paths, forecasts)
            }
            Nodes::Stochastic {
                mu,
                phi,
                sigma,
                log_variance,
            } => {
                let (mu, phi, sigma) = (
                    scalar_draws(samples, mu),
                    scalar_draws(samples, phi),
                    scalar_draws(samples, sigma),
                );
                let h = draws(samples, log_variance);
                let mut paths = Vec::with_capacity(mu.len());
                let mut forecasts: Vec<Forecaster> = Vec::new();
                for i in 0..mu.len() {
                    let (m, p, s) = (mu[i], phi[i], sigma[i]);
                    let variance = s * s / (1.0 - p * p);
                    paths.push(DrawPath {
                        sd: h[i].iter().map(|h| (h / 2.0).exp()).collect(),
                        // E[exp(h / 2)] for a Gaussian h
                        long_run_sd: (m / 2.0 + variance / 8.0).exp(),
                        persistence: p,
                    });
                    let last = *h[i].last().unwrap();
                    forecasts.push(Box::new(move |rng: &mut KeanuRng, horizon: usize| {
                        let mut h = last;
                        (0..horizon)
                            .map(|_| {
                                h = m + p * (h - m) + s * rng.next_gaussian();
                                (h / 2.0).exp()
                            })
                            .collect()
                    }));
                }
                (paths, forecasts)
            }
        }
    }

    /// The report for a position worth `position`, forecasting `horizon` days with `scenarios` simulated days per
    /// posterior draw.
    pub fn report(
        &self,
        position: f64,
        horizon: usize,
        scenarios: usize,
    ) -> keanu::Result<VolatilityReport> {
        let (paths, forecasts) = self.paths();
        let n = paths.len();
        if n == 0 {
            return Err(keanu::Error::from(keanu::graph::GraphError::InvalidValue(
                "the model has no posterior draws yet".into(),
            )));
        }
        // a copy of the chain's generator, so a report changes neither the chain nor the next report
        let mut rng = self.chain.rng_copy();
        let annualised = |sd_percent: f64| sd_percent / 100.0 * TRADING_DAYS.sqrt();
        let days = self.returns.len();
        let volatility = (0..days)
            .map(|t| {
                Interval::of(
                    &paths
                        .iter()
                        .map(|p| annualised(p.sd[t]))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let horizon = horizon.max(1);
        let mut ahead = vec![Vec::with_capacity(n * scenarios); horizon];
        let mut tomorrow = Vec::with_capacity(n * scenarios);
        for forecast in &forecasts {
            for _ in 0..scenarios {
                let sd = forecast(&mut rng, horizon);
                for (k, sd) in sd.iter().enumerate() {
                    ahead[k].push(annualised(*sd));
                }
                tomorrow.push(-sd[0] / 100.0 * rng.next_gaussian() * position);
            }
        }
        let forecast = ahead
            .iter()
            .enumerate()
            .map(|(k, v)| ForecastDay {
                ahead: k + 1,
                volatility: Interval::of(v),
            })
            .collect();
        let value_at_risk_975 = quantile(&tomorrow, 0.975);
        let worst: Vec<f64> = tomorrow
            .iter()
            .copied()
            .filter(|&l| l >= value_at_risk_975)
            .collect();
        let sd = {
            let m = mean(&self.returns);
            (self.returns.iter().map(|r| (r - m).powi(2)).sum::<f64>() / (days - 1) as f64).sqrt()
        };
        Ok(VolatilityReport {
            volatility,
            forecast,
            long_run_volatility: Interval::of(
                &paths
                    .iter()
                    .map(|p| annualised(p.long_run_sd))
                    .collect::<Vec<_>>(),
            ),
            half_life_days: Interval::of(
                &paths
                    .iter()
                    .map(|p| 0.5f64.ln() / p.persistence.ln())
                    .collect::<Vec<_>>(),
            ),
            position,
            value_at_risk_95: quantile(&tomorrow, 0.95),
            value_at_risk_99: quantile(&tomorrow, 0.99),
            expected_shortfall_975: mean(&worst),
            tail: TAIL_LEVELS
                .iter()
                .zip(NORMAL_QUANTILES)
                .map(|(&level, z)| {
                    let value_at_risk = quantile(&tomorrow, level);
                    let beyond: Vec<f64> = tomorrow
                        .iter()
                        .copied()
                        .filter(|&l| l >= value_at_risk)
                        .collect();
                    TailRisk {
                        level,
                        value_at_risk,
                        expected_shortfall: mean(&beyond),
                        naive_value_at_risk: z * sd / 100.0 * position,
                    }
                })
                .collect(),
            loss_histogram: Histogram::of(&tomorrow, 60),
            naive_value_at_risk_99: 2.326_347_874 * sd / 100.0 * position,
            divergences: self.chain.divergences(),
            draws: n,
        })
    }
}
