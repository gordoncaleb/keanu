//! How much stock do we need next month? Demand forecasting for one product.
//!
//! A shop has a year of daily unit sales. Demand drifts slowly (a level that wanders from one four-week period to the
//! next around a steady trend), follows a weekly pattern (busier at weekends), and is noisier than a Poisson count
//! would be. A negative binomial regression on the log scale captures all three:
//!
//! ```text
//! level_p = c + 0.05 a + 0.05 g (p − mid) / 13 + sigma (z_1 + ... + z_p)      (p = 0 .. periods − 1)
//! day_d = 0.05 r_d (d = Monday .. Saturday)    day_Sunday = −(day_Monday + ... + day_Saturday)
//! sigma = 0.05 exp(0.06 s)    alpha = exp(2.5 + 0.1 l)
//! a, g ~ N(0, 20)    r_d, s, l ~ N(0, 10)    z_p ~ N(0, 1)
//! sales_t ~ NegativeBinomial(mean exp(level_{period t} + day_{weekday t}), dispersion alpha)
//! ```
//!
//! where c is the log of the history's mean daily sales and mid the middle period. The constants only rescale the
//! parameters so that NUTS sees posteriors of similar widths: the priors are a level within a factor of e of the
//! history's mean, yearly growth with a standard deviation of 100% (in log terms), weekday effects with a standard
//! deviation of 0.5 on the log scale, steps of the level of about 5% a period (log-normal, between about 1.5% and
//! 16%), and a dispersion between about 4 and 30. The random walk is non-centred (the sampler moves the standard
//! normal steps z, not the levels), and keeping its scale away from zero avoids the funnel a half-normal scale
//! would give it.
//!
//! Within a period the expected sales on a weekday are the same every week, and the sum of independent negative
//! binomials with a common mean and dispersion is negative binomial (mean and dispersion both multiplied by the
//! count), so the likelihood needs only one term per period and weekday: 91 terms for a year rather than 364, which
//! keeps the fit to a second or two in a browser.
//!
//! The posterior then plays the next weeks forward: for each draw the level keeps walking with its trend and scale,
//! and each day's sales are drawn from the negative binomial. These simulated futures are computed once and kept, so
//! the stock decision (the quantile of total demand over a period at a chosen service level, with the expected units
//! short and left over) updates without refitting or resimulating.
//!
//! The history is synthetic, generated from a seed by this model with known values, so a demo can show how well they
//! are recovered; the truth also continues past the history, for a held-out check.

use crate::run::{ChainRun, Histogram, Interval, draws, mean, quantile, scalar_draws};
use keanu::distributions::NegativeBinomialLog;
use keanu::prelude::*;
use serde::Serialize;
use std::cell::OnceCell;

/// Days of the week, starting with the first day of the history.
pub const WEEKDAYS: [&str; 7] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
];
/// The days in a period, over which the level is constant.
pub const PERIOD_DAYS: usize = 28;
/// Periods in a year.
const PERIODS_PER_YEAR: f64 = 13.0;
/// The average daily sales at the start of the history.
pub const TRUE_START: f64 = 40.0;
/// The growth of the log level per period (about 23% a year).
pub const TRUE_TREND: f64 = 0.016;
/// The standard deviation of the log level's random steps from one period to the next.
pub const TRUE_SIGMA: f64 = 0.05;
/// Each weekday's effect on log sales (summing to zero).
pub const TRUE_WEEKDAY: [f64; 7] = [-0.15, -0.1, -0.05, 0.0, 0.15, 0.25, -0.1];
/// The negative binomial dispersion: the variance of a day's sales is mean + mean² / alpha.
pub const TRUE_DISPERSION: f64 = 10.0;
/// The longest period a stock decision can cover, in days, and how far the futures are simulated.
pub const MAX_PERIOD: usize = 56;
/// The days the forecast chart shows.
pub const FORECAST_DAYS: usize = 28;

/// Synthetic sales: the observed history and the true process behind it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Sales {
    /// Units sold each day of the history, starting on a Monday.
    pub history: Vec<i64>,
    /// The true expected sales each day of the history.
    pub true_mean: Vec<f64>,
    /// The units actually sold on each of the [`MAX_PERIOD`] days after the history, held out from the fit.
    pub future: Vec<i64>,
    /// The true expected sales on each of the held-out days.
    pub future_mean: Vec<f64>,
}

/// `periods` four-week periods of daily sales generated from `seed` with the true parameters, and [`MAX_PERIOD`]
/// days beyond.
pub fn sales(periods: usize, seed: u64) -> keanu::Result<Sales> {
    let mut rng = KeanuRng::seed_from_u64(seed);
    let days = periods * PERIOD_DAYS;
    let mut level = TRUE_START.ln();
    let means: Vec<f64> = (0..days + MAX_PERIOD)
        .map(|t| {
            if t > 0 && t % PERIOD_DAYS == 0 {
                level += TRUE_TREND + TRUE_SIGMA * rng.next_gaussian();
            }
            level + TRUE_WEEKDAY[t % 7]
        })
        .collect();
    let counts = NegativeBinomialLog::new(
        DoubleTensor::vector(means.clone()),
        DoubleTensor::scalar(TRUE_DISPERSION),
    )
    .sample(&[means.len()], &mut rng)?;
    let counts = counts.as_slice();
    let exp = |m: &[f64]| m.iter().map(|m| m.exp()).collect();
    Ok(Sales {
        history: counts[..days].to_vec(),
        true_mean: exp(&means[..days]),
        future: counts[days..].to_vec(),
        future_mean: exp(&means[days..]),
    })
}

/// A fit in progress, and once finished, its simulated futures.
#[derive(Debug)]
pub struct DemandModel {
    chain: ChainRun,
    periods: usize,
    level: NodeId,
    trend: NodeId,
    sigma: NodeId,
    weekday: NodeId,
    dispersion: NodeId,
    /// For each simulated future, the units sold each day for [`MAX_PERIOD`] days.
    futures: OnceCell<Vec<Vec<u32>>>,
}

/// One day of the history fit or the forecast.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Day {
    /// The posterior mean and 90% interval: of expected sales for a history day, of sales for a forecast day.
    #[serde(flatten)]
    pub interval: Interval,
    /// The units sold (observed in the history, held out in the forecast).
    pub actual: i64,
    /// The true expected sales.
    pub truth: f64,
}

/// A weekday's effect on sales, as a multiplier on an average day.
#[derive(Debug, Clone, Serialize)]
pub struct WeekdayEffect {
    /// The weekday.
    pub day: &'static str,
    /// The posterior multiplier.
    pub multiplier: Interval,
    /// The multiplier the history was generated with.
    pub truth: f64,
    /// The average observed sales on this weekday, as a multiple of the average day.
    pub observed: f64,
}

/// The stock decision for a period and service level.
#[derive(Debug, Clone, Serialize)]
pub struct DemandReport {
    /// The days the stock must cover.
    pub period: usize,
    /// The probability that the stock covers the period's demand.
    pub service_level: f64,
    /// The stock that covers total demand over the period with that probability.
    pub stock: f64,
    /// The mean total demand over the period.
    pub expected_demand: f64,
    /// The 90% interval of total demand.
    pub demand: Interval,
    /// Stock beyond the expected demand.
    pub safety_stock: f64,
    /// The expected units of demand that the stock cannot meet.
    pub expected_short: f64,
    /// The expected units left at the end of the period.
    pub expected_left_over: f64,
    /// The expected fraction of demand met from stock.
    pub fill_rate: f64,
    /// Units actually sold in the period (held out from the fit).
    pub actual_demand: i64,
    /// The distribution of total demand over the period.
    pub demand_histogram: Histogram,
    /// The history: expected daily sales against what sold.
    pub history: Vec<Day>,
    /// The next [`FORECAST_DAYS`] days' sales, with what actually sold.
    pub forecast: Vec<Day>,
    /// Each weekday's effect.
    pub weekdays: Vec<WeekdayEffect>,
    /// The change in the level of demand from the first period of the history to the last, as a fraction.
    pub growth: Interval,
    /// The true change in level over the history.
    pub true_growth: f64,
    /// The trend's growth over a year, as a fraction, which carries into the forecast.
    pub yearly_trend: Interval,
    /// The true trend's growth over a year.
    pub true_yearly_trend: f64,
    /// The negative binomial dispersion.
    pub dispersion: Interval,
    /// The true dispersion.
    pub true_dispersion: f64,
    /// The standard deviation of the log level's random steps from one period to the next.
    pub drift: Interval,
    /// The true standard deviation of the steps.
    pub true_drift: f64,
    /// The number of posterior draws.
    pub draws: usize,
    /// The number of simulated futures.
    pub futures: usize,
    /// Divergent transitions after adaptation.
    pub divergences: usize,
}

fn invalid(message: String) -> keanu::Error {
    keanu::Error::from(keanu::graph::GraphError::InvalidValue(message))
}

impl DemandModel {
    /// Starts fitting a history of whole four-week periods with `draws` NUTS draws after as many adaptation steps.
    pub fn new(history: &[i64], draws: usize, seed: u64) -> keanu::Result<Self> {
        let n = history.len();
        let periods = n / PERIOD_DAYS;
        if periods < 3 || n % PERIOD_DAYS != 0 {
            return Err(invalid(format!(
                "the history must be at least 3 whole four-week periods of days, got {n} days"
            )));
        }
        let weeks = PERIOD_DAYS / 7;
        let centre = (history.iter().sum::<i64>() as f64 / n as f64)
            .max(0.1)
            .ln();
        let b = GraphBuilder::new();
        let a = b.gaussian(0.0, 20.0);
        let g = b.gaussian(0.0, 20.0);
        let s = b.gaussian(0.0, 10.0);
        let z = b.gaussian_with_shape(&[periods - 1], 0.0, 1.0);
        let r = b.gaussian_with_shape(&[6], 0.0, 10.0);
        let l = b.gaussian(0.0, 10.0);
        let sigma = (s * 0.06 + 0.05f64.ln()).exp();

        let walk = b.concat(
            0,
            &[
                b.constant(DoubleTensor::vector(vec![0.0])),
                (z * sigma).cumsum(0),
            ],
        );
        let mid = (periods - 1) as f64 / 2.0;
        let years = b.constant(DoubleTensor::vector(
            (0..periods)
                .map(|p| (p as f64 - mid) / PERIODS_PER_YEAR)
                .collect(),
        ));
        let trend = g * 0.05;
        let level = a * 0.05 + centre + trend * years + walk;
        let weekday = b.concat(0, &[r, r.sum().reshape(&[1]) * -1.0]) * 0.05;
        let dispersion = (l * 0.1 + 2.5).exp();

        // one term per period and weekday: the sum of its `weeks` days
        let cells = periods * 7;
        let period_of = IntTensor::vector((0..cells).map(|c| (c / 7) as i64).collect());
        let day_of = IntTensor::vector((0..cells).map(|c| (c % 7) as i64).collect());
        let totals: Vec<i64> = (0..cells)
            .map(|c| {
                (0..weeks)
                    .map(|w| history[(c / 7) * PERIOD_DAYS + w * 7 + c % 7])
                    .sum()
            })
            .collect();
        let log_mean = level.gather(0, period_of) + weekday.gather(0, day_of) + (weeks as f64).ln();
        let sales =
            b.negative_binomial_log_with_shape(&[cells], log_mean, dispersion * weeks as f64);
        b.observe(sales, IntTensor::vector(totals));

        let (level, trend, sigma, weekday, dispersion) = (
            level.id(),
            trend.id(),
            sigma.id(),
            weekday.id(),
            dispersion.id(),
        );
        let (graph, state) = b.build()?;
        let model = Model::new(graph, &state)?;
        let chain = ChainRun::new(
            model,
            state,
            &[level, trend, sigma, weekday, dispersion],
            Nuts::default(),
            draws,
            draws,
            seed,
        )?;
        Ok(Self {
            chain,
            periods,
            level,
            trend,
            sigma,
            weekday,
            dispersion,
            futures: OnceCell::new(),
        })
    }

    /// The chain.
    pub fn chain(&mut self) -> &mut ChainRun {
        &mut self.chain
    }

    /// `scenarios` simulated futures of [`MAX_PERIOD`] days per posterior draw, simulated on first use and kept.
    fn futures(&self, scenarios: usize) -> keanu::Result<&Vec<Vec<u32>>> {
        if let Some(futures) = self.futures.get() {
            return Ok(futures);
        }
        let samples = self.chain.samples();
        let levels = draws(samples, self.level);
        if levels.is_empty() {
            return Err(invalid("the model has no posterior draws yet".into()));
        }
        // the trend per period
        let trends: Vec<f64> = scalar_draws(samples, self.trend)
            .into_iter()
            .map(|t| t / PERIODS_PER_YEAR)
            .collect();
        let sigmas = scalar_draws(samples, self.sigma);
        let weekdays = draws(samples, self.weekday);
        let dispersions = scalar_draws(samples, self.dispersion);
        let mut rng = self.chain.rng_copy();
        let mut futures = Vec::with_capacity(levels.len() * scenarios);
        let mut log_mean = vec![0.0; MAX_PERIOD];
        for i in 0..levels.len() {
            for _ in 0..scenarios {
                // the history ends with a whole period, so the future starts a new one, on a Monday
                let mut level = levels[i][self.periods - 1];
                for (t, slot) in log_mean.iter_mut().enumerate() {
                    if t % PERIOD_DAYS == 0 {
                        level += trends[i] + sigmas[i] * rng.next_gaussian();
                    }
                    *slot = level + weekdays[i][t % 7];
                }
                let days = NegativeBinomialLog::new(
                    DoubleTensor::vector(log_mean.clone()),
                    DoubleTensor::scalar(dispersions[i]),
                )
                .sample(&[MAX_PERIOD], &mut rng)?;
                futures.push(days.as_slice().iter().map(|&k| k as u32).collect());
            }
        }
        Ok(self.futures.get_or_init(|| futures))
    }

    /// The stock that covers total demand over the next `period` days (1 to [`MAX_PERIOD`]) with probability
    /// `service_level`, with the history fit and forecast, from `scenarios` simulated futures per posterior draw.
    /// `sales` gives the observed history and the held-out truth. The futures are simulated once, on the first
    /// call, so later calls with other periods or service levels only read them.
    pub fn report(
        &self,
        sales: &Sales,
        period: usize,
        service_level: f64,
        scenarios: usize,
    ) -> keanu::Result<DemandReport> {
        if !(1..=MAX_PERIOD).contains(&period) || !(service_level > 0.0 && service_level < 1.0) {
            return Err(invalid(format!(
                "the period must be 1 to {MAX_PERIOD} days and the service level between 0 and 1, got {period} and \
                 {service_level}"
            )));
        }
        let futures = self.futures(scenarios)?;
        let totals: Vec<f64> = futures
            .iter()
            .map(|f| f[..period].iter().map(|&k| f64::from(k)).sum())
            .collect();
        let stock = quantile(&totals, service_level).ceil();
        let expected_demand = mean(&totals);
        let shortfall = |t: &f64| (t - stock).max(0.0);
        let left_over = |t: &f64| (stock - t).max(0.0);
        let expected_short = mean(&totals.iter().map(shortfall).collect::<Vec<_>>());
        let expected_left_over = mean(&totals.iter().map(left_over).collect::<Vec<_>>());

        let samples = self.chain.samples();
        let levels = draws(samples, self.level);
        let weekday_draws = draws(samples, self.weekday);
        let history = sales
            .history
            .iter()
            .enumerate()
            .map(|(t, &actual)| {
                let expected: Vec<f64> = levels
                    .iter()
                    .zip(&weekday_draws)
                    .map(|(l, d)| (l[t / PERIOD_DAYS] + d[t % 7]).exp())
                    .collect();
                Day {
                    interval: Interval::of(&expected),
                    actual,
                    truth: sales.true_mean[t],
                }
            })
            .collect();
        let forecast = (0..FORECAST_DAYS)
            .map(|t| {
                let day: Vec<f64> = futures.iter().map(|f| f64::from(f[t])).collect();
                Day {
                    interval: Interval::of(&day),
                    actual: sales.future[t],
                    truth: sales.future_mean[t],
                }
            })
            .collect();

        let overall = sales.history.iter().sum::<i64>() as f64 / sales.history.len() as f64;
        let weekdays = WEEKDAYS
            .iter()
            .enumerate()
            .map(|(d, &day)| {
                let on_day: Vec<f64> = sales
                    .history
                    .iter()
                    .skip(d)
                    .step_by(7)
                    .map(|&k| k as f64)
                    .collect();
                let multipliers: Vec<f64> = weekday_draws.iter().map(|w| w[d].exp()).collect();
                WeekdayEffect {
                    day,
                    multiplier: Interval::of(&multipliers),
                    truth: TRUE_WEEKDAY[d].exp(),
                    observed: mean(&on_day) / overall,
                }
            })
            .collect();
        let last = self.periods - 1;
        let growth: Vec<f64> = levels
            .iter()
            .map(|l| (l[last] - l[0]).exp() - 1.0)
            .collect();
        let true_level = |p: usize| (sales.true_mean[p * PERIOD_DAYS] / TRUE_WEEKDAY[0].exp()).ln();
        let trends: Vec<f64> = scalar_draws(samples, self.trend)
            .into_iter()
            .map(|t| t.exp() - 1.0)
            .collect();
        Ok(DemandReport {
            period,
            service_level,
            stock,
            expected_demand,
            demand: Interval::of(&totals),
            safety_stock: stock - expected_demand,
            expected_short,
            expected_left_over,
            fill_rate: 1.0 - expected_short / expected_demand.max(f64::MIN_POSITIVE),
            actual_demand: sales.future[..period].iter().sum(),
            demand_histogram: Histogram::of(&totals, 40),
            history,
            forecast,
            weekdays,
            growth: Interval::of(&growth),
            true_growth: (true_level(last) - true_level(0)).exp() - 1.0,
            yearly_trend: Interval::of(&trends),
            true_yearly_trend: (TRUE_TREND * PERIODS_PER_YEAR).exp() - 1.0,
            dispersion: Interval::of(&scalar_draws(samples, self.dispersion)),
            true_dispersion: TRUE_DISPERSION,
            drift: Interval::of(&scalar_draws(samples, self.sigma)),
            true_drift: TRUE_SIGMA,
            draws: levels.len(),
            futures: futures.len(),
            divergences: self.chain.divergences(),
        })
    }
}
