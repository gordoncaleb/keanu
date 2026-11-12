//! How many loans in this book will default, and what could they cost? Credit portfolio risk.
//!
//! A lender has the repayment history of past loans, each with a risk grade (A to E) and a standardised
//! debt-to-income ratio. A hierarchical logistic regression learns each grade's default rate, partially pooled
//! across grades, and the effect of debt-to-income:
//!
//! ```text
//! mu ~ N(−2.5, 1.5)    tau ~ HalfGaussian(1)    z_g ~ N(0, 1)    alpha_g = mu + tau z_g    (g = A..E)
//! beta ~ N(0, 1)       defaulted_i ~ Bernoulli(logistic(alpha_{grade_i} + beta dti_i))
//! ```
//!
//! The posterior then prices the current book: for each posterior draw every open loan has its own default
//! probability, so the number of defaults is Poisson-binomial. Its exact tail gives the probability that more than
//! K loans default, averaged over the posterior; simulated defaults weighted by exposure and loss given default give
//! the loss distribution, its expected loss, and the 99% value at risk and expected shortfall. A stress shift on the
//! log-odds scale (a downturn) is applied to the stored draws, so a scenario updates without refitting.
//!
//! The history and the book are synthetic, generated from a seed with known default rates, so a demo can show
//! how well they are recovered.

use crate::run::{ChainRun, Histogram, Interval, draws, mean, quantile, scalar_draws};
use keanu::distributions::PoissonBinomial;
use keanu::prelude::*;
use serde::Serialize;

/// Grade names, best to worst.
pub const GRADES: [&str; 5] = ["A", "B", "C", "D", "E"];
/// The default log-odds of each grade the synthetic history was generated with, at average debt-to-income.
pub const TRUE_GRADE_LOGIT: [f64; 5] = [-4.2, -3.5, -2.8, -2.2, -1.5];
/// The effect of one standard deviation of debt-to-income on the default log-odds.
pub const TRUE_DTI_EFFECT: f64 = 0.45;
/// The share of each grade among loans.
const GRADE_MIX: [f64; 5] = [0.2, 0.3, 0.25, 0.15, 0.1];
/// The fraction of exposure lost when a loan defaults.
pub const LOSS_GIVEN_DEFAULT: f64 = 0.45;

/// A loan: its grade (0 = A to 4 = E), standardised debt-to-income and amount outstanding.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Loan {
    /// The risk grade, 0 (A) to 4 (E).
    pub grade: usize,
    /// Debt-to-income, standardised.
    pub dti: f64,
    /// The amount outstanding.
    pub exposure: f64,
}

/// A past loan and whether it defaulted.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Outcome {
    /// The loan.
    pub loan: Loan,
    /// Whether it defaulted.
    pub defaulted: bool,
}

/// Draws `count` loans with the grade mix and debt-to-income of the synthetic lender.
pub fn loans(count: usize, rng: &mut KeanuRng) -> Vec<Loan> {
    (0..count)
        .map(|_| {
            let u = rng.next_f64();
            let mut cumulative = 0.0;
            let grade = GRADE_MIX
                .iter()
                .position(|&share| {
                    cumulative += share;
                    u < cumulative
                })
                .unwrap_or(GRADES.len() - 1);
            Loan {
                grade,
                dti: rng.next_gaussian(),
                // a log-normal amount around 20,000
                exposure: (20_000.0 * (0.6 * rng.next_gaussian()).exp()).round(),
            }
        })
        .collect()
}

/// A synthetic repayment history of `count` loans, defaulting with the true rates.
pub fn history(count: usize, seed: u64) -> Vec<Outcome> {
    let mut rng = KeanuRng::seed_from_u64(seed);
    loans(count, &mut rng)
        .into_iter()
        .map(|loan| {
            let logit = TRUE_GRADE_LOGIT[loan.grade] + TRUE_DTI_EFFECT * loan.dti;
            Outcome {
                loan,
                defaulted: rng.next_f64() < logistic(logit),
            }
        })
        .collect()
}

/// A synthetic book of `count` open loans.
pub fn book(count: usize, seed: u64) -> Vec<Loan> {
    loans(count, &mut KeanuRng::seed_from_u64(seed ^ 0x5eed))
}

fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// A fit in progress.
#[derive(Debug)]
pub struct CreditModel {
    chain: ChainRun,
    grade_logit: NodeId,
    dti_effect: NodeId,
}

/// The fitted default model: what each grade's loans default at.
#[derive(Debug, Clone, Serialize)]
pub struct GradeReport {
    /// The grade's name.
    pub grade: &'static str,
    /// Loans of the grade in the history.
    pub loans: usize,
    /// The observed default rate in the history.
    pub observed_rate: f64,
    /// The posterior default probability at average debt-to-income, under the stress shift.
    pub default_probability: Interval,
    /// The default probability the history was generated with (without stress).
    pub true_probability: f64,
}

/// The risk of the book under a stress shift.
#[derive(Debug, Clone, Serialize)]
pub struct PortfolioReport {
    /// The stress shift added to every loan's default log-odds.
    pub stress: f64,
    /// Each grade's default probability.
    pub grades: Vec<GradeReport>,
    /// The effect of one standard deviation of debt-to-income on the default odds, as an odds ratio.
    pub dti_odds_ratio: Interval,
    /// The number of loans in the book and their total exposure.
    pub book_loans: usize,
    /// The book's total exposure.
    pub book_exposure: f64,
    /// The expected number of defaults.
    pub expected_defaults: Interval,
    /// The threshold for `probability_defaults_exceed`.
    pub default_threshold: i64,
    /// The exact probability that more than `default_threshold` loans default (Poisson-binomial, averaged over the
    /// posterior).
    pub probability_defaults_exceed: f64,
    /// The distribution of the number of defaults, one bin per count, by simulation.
    pub defaults_histogram: Histogram,
    /// The mean credit loss.
    pub expected_loss: f64,
    /// The 99% value at risk: the loss exceeded with 1% probability.
    pub value_at_risk_99: f64,
    /// The 99% expected shortfall: the mean loss beyond the value at risk.
    pub expected_shortfall_99: f64,
    /// The distribution of losses.
    pub loss_histogram: Histogram,
    /// The number of posterior draws used.
    pub draws: usize,
}

impl CreditModel {
    /// Starts fitting the history with `draws` NUTS draws after as many adaptation steps.
    pub fn new(history: &[Outcome], draws: usize, seed: u64) -> keanu::Result<Self> {
        let n = history.len();
        let b = GraphBuilder::new();
        let mu = b.gaussian(-2.5, 1.5);
        let tau = b.half_gaussian(1.0);
        let z = b.gaussian_with_shape(&[GRADES.len()], 0.0, 1.0);
        let beta = b.gaussian(0.0, 1.0);
        b.set_value(mu, DoubleTensor::scalar(-2.5));
        b.set_value(tau, DoubleTensor::scalar(0.5));
        b.set_value(z, DoubleTensor::vector(vec![0.0; GRADES.len()]));
        b.set_value(beta, DoubleTensor::scalar(0.0));
        let grade_logit = mu + tau * z;
        let grades = IntTensor::vector(history.iter().map(|o| o.loan.grade as i64).collect());
        let dti = DoubleTensor::vector(history.iter().map(|o| o.loan.dti).collect());
        let logit = grade_logit.gather(0, grades) + beta * b.constant(dti);
        let defaulted = b.bernoulli_logit_with_shape(&[n], logit);
        b.observe(
            defaulted,
            BoolTensor::vector(history.iter().map(|o| o.defaulted).collect()),
        );
        let (grade_logit, dti_effect) = (grade_logit.id(), beta.id());
        let (graph, state) = b.build()?;
        let model = Model::new(graph, &state)?;
        let chain = ChainRun::new(
            model,
            state,
            &[grade_logit, dti_effect],
            Nuts::default(),
            draws,
            draws,
            seed,
        )?;
        Ok(Self {
            chain,
            grade_logit,
            dti_effect,
        })
    }

    /// The chain.
    pub fn chain(&mut self) -> &mut ChainRun {
        &mut self.chain
    }

    /// The risk of `book` with every loan's default log-odds shifted by `stress`, from at most `max_draws`
    /// posterior draws with `scenarios` simulated default outcomes each. `history` gives the observed rates.
    pub fn portfolio(
        &self,
        history: &[Outcome],
        book: &[Loan],
        stress: f64,
        default_threshold: i64,
        max_draws: usize,
        scenarios: usize,
    ) -> keanu::Result<PortfolioReport> {
        let samples = self.chain.samples();
        let all_logits = draws(samples, self.grade_logit);
        let all_effects = scalar_draws(samples, self.dti_effect);
        let stride = all_logits.len().div_ceil(max_draws.max(1)).max(1);
        let logits: Vec<&Vec<f64>> = all_logits.iter().step_by(stride).collect();
        let effects: Vec<f64> = all_effects.iter().copied().step_by(stride).collect();
        let draw_count = logits.len();
        if draw_count == 0 {
            return Err(keanu::Error::from(keanu::graph::GraphError::InvalidValue(
                "the model has no posterior draws yet".into(),
            )));
        }
        let m = book.len();

        // every open loan's default probability under each draw, as a [draws, loans] tensor
        let mut p = Vec::with_capacity(draw_count * m);
        for (alpha, &beta) in logits.iter().zip(&effects) {
            p.extend(
                book.iter()
                    .map(|l| logistic(alpha[l.grade] + beta * l.dti + stress)),
            );
        }
        let defaults_mean: Vec<f64> = p.chunks(m).map(|row| row.iter().sum()).collect();
        let tail = PoissonBinomial::new(DoubleTensor::new(&[draw_count, m], p.clone())?)?
            .log_ccdf(&IntTensor::scalar(default_threshold))?;
        let probability_defaults_exceed = mean(
            &tail
                .as_slice()
                .iter()
                .map(|lp| lp.exp())
                .collect::<Vec<_>>(),
        );

        let mut rng = self.chain.rng_copy();
        let mut losses = Vec::with_capacity(draw_count * scenarios);
        let mut counts = Vec::with_capacity(draw_count * scenarios);
        for row in p.chunks(m) {
            for _ in 0..scenarios {
                let (mut loss, mut count) = (0.0, 0usize);
                for (loan, &pd) in book.iter().zip(row) {
                    if rng.next_f64() < pd {
                        loss += loan.exposure * LOSS_GIVEN_DEFAULT;
                        count += 1;
                    }
                }
                losses.push(loss);
                counts.push(count);
            }
        }
        let value_at_risk_99 = quantile(&losses, 0.99);
        let tail_losses: Vec<f64> = losses
            .iter()
            .copied()
            .filter(|&l| l >= value_at_risk_99)
            .collect();

        let grades = GRADES
            .iter()
            .enumerate()
            .map(|(g, &grade)| {
                let in_grade: Vec<&Outcome> =
                    history.iter().filter(|o| o.loan.grade == g).collect();
                let rates: Vec<f64> = logits.iter().map(|a| logistic(a[g] + stress)).collect();
                GradeReport {
                    grade,
                    loans: in_grade.len(),
                    observed_rate: in_grade.iter().filter(|o| o.defaulted).count() as f64
                        / in_grade.len().max(1) as f64,
                    default_probability: Interval::of(&rates),
                    true_probability: logistic(TRUE_GRADE_LOGIT[g]),
                }
            })
            .collect();
        let odds_ratios: Vec<f64> = effects.iter().map(|b| b.exp()).collect();
        Ok(PortfolioReport {
            stress,
            grades,
            dti_odds_ratio: Interval::of(&odds_ratios),
            book_loans: m,
            book_exposure: book.iter().map(|l| l.exposure).sum(),
            expected_defaults: Interval::of(&defaults_mean),
            default_threshold,
            probability_defaults_exceed,
            defaults_histogram: Histogram::of_counts(&counts),
            expected_loss: mean(&losses),
            value_at_risk_99,
            expected_shortfall_99: mean(&tail_losses),
            loss_histogram: Histogram::of(&losses, 40),
            draws: draw_count,
        })
    }
}
