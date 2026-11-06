//! How do people trade speed for accuracy, and does an easier task change how fast they take in evidence?
//!
//! A hierarchical drift-diffusion model (Ratcliff 1978) of simulated two-choice response times. On each trial
//! evidence accumulates as a Wiener process with drift v until it reaches one of two boundaries a apart, starting a
//! fraction w of the way from the lower one; the boundary reached is the response (1 the upper, "correct"; 0 the
//! lower, "error") and the time taken plus a non-decision time t0 (encoding and motor response) is the response
//! time. The joint density of a response and its time is the Wiener first-passage time density
//! ([`keanu::distributions::Wiener`]), with the response an observed integer parameter.
//!
//! Four subjects each do 100 trials in two conditions, hard and easy, which differ in drift only. Each subject has
//! their own boundary separation (caution), non-decision time and drift in each condition, drawn around population
//! values:
//!
//! - `ln a_s ~ N(μ_ln a, σ_ln a)`,
//! - `v_sc ~ N(μ_v,c, σ_v)`,
//! - `t0_s ~ Uniform(0, the subject's fastest response)`,
//! - a common starting point `w ~ Beta(2, 2)`.
//!
//! With a hundred trials in each cell the data pin each subject down well, so the model is written centred (the
//! non-centred form made NUTS diverge in the funnel of the poorly identified population scales, which have
//! Gamma(2, ·) priors that keep them off zero with only four subjects). The data are simulated exactly ([`keanu::distributions::sample_wiener`]) from known values
//! ([`TRUE_DRIFT`], [`TRUE_W`], [`TRUE_A`], [`TRUE_T0`]) so the fit can be checked against them. Run with
//! `cargo run --release --example drift_diffusion`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::mcmc::chains::sample_chains;
use keanu::distributions::sample_wiener;
use keanu::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The number of subjects.
pub const SUBJECTS: usize = 4;
/// The number of conditions (hard, easy).
pub const CONDITIONS: usize = 2;
/// Trials per subject in each condition.
pub const TRIALS: usize = 100;
/// The population mean drift in each condition the data are simulated with.
pub const TRUE_DRIFT: [f64; CONDITIONS] = [0.7, 1.8];
/// The spread of the subjects' drifts around the condition means.
pub const TRUE_DRIFT_SD: f64 = 0.3;
/// The starting point the data are simulated with.
pub const TRUE_W: f64 = 0.45;
/// Each subject's boundary separation.
pub const TRUE_A: [f64; SUBJECTS] = [1.0, 1.3, 1.6, 1.15];
/// Each subject's non-decision time, in seconds.
pub const TRUE_T0: [f64; SUBJECTS] = [0.25, 0.32, 0.2, 0.28];
/// The seed the example's data are simulated with.
pub const DATA_SEED: u64 = 15;

/// Simulated trials: each one's subject, condition, response (1 upper, 0 lower) and response time, and the
/// subjects' true drifts by condition.
#[derive(Debug, Clone)]
pub struct Trials {
    pub subject: Vec<i64>,
    pub condition: Vec<i64>,
    pub response: Vec<i64>,
    pub time: Vec<f64>,
    pub drift: [[f64; CONDITIONS]; SUBJECTS],
}

/// Simulates every subject's trials from the true parameters.
pub fn data(seed: u64) -> keanu::Result<Trials> {
    let mut rng = KeanuRng::seed_from_u64(seed);
    let mut drift = [[0.0; CONDITIONS]; SUBJECTS];
    for row in &mut drift {
        for (c, d) in row.iter_mut().enumerate() {
            *d = TRUE_DRIFT[c] + TRUE_DRIFT_SD * rng.next_gaussian();
        }
    }
    let mut trials = Trials {
        subject: Vec::new(),
        condition: Vec::new(),
        response: Vec::new(),
        time: Vec::new(),
        drift,
    };
    let scalar = DoubleTensor::scalar;
    for s in 0..SUBJECTS {
        for (c, &v) in drift[s].iter().enumerate() {
            let (times, responses) = sample_wiener(
                &scalar(TRUE_A[s]),
                &scalar(TRUE_T0[s]),
                &scalar(TRUE_W),
                &scalar(v),
                &scalar(0.0),
                &[TRIALS],
                &mut rng,
            )?;
            trials.time.extend_from_slice(times.as_slice());
            trials.response.extend_from_slice(responses.as_slice());
            trials.subject.extend(std::iter::repeat_n(s as i64, TRIALS));
            trials
                .condition
                .extend(std::iter::repeat_n(c as i64, TRIALS));
        }
    }
    Ok(trials)
}

/// A fit: every chain's draws, and the nodes of the population drifts (one per condition), the starting point and
/// the subjects' boundary separations, non-decision times and drifts ([subject, condition]).
#[derive(Debug)]
pub struct Fit {
    pub chains: ChainSamples,
    pub drift: NodeId,
    pub w: NodeId,
    pub a: NodeId,
    pub t0: NodeId,
    pub subject_drift: NodeId,
    /// The number of divergent transitions after warm-up, over every chain.
    pub divergences: usize,
}

/// Fits the hierarchical model by NUTS, with `chains` chains of `draws` draws after as many warm-up steps.
pub fn run(trials: &Trials, chains: usize, draws: usize, seed: u64) -> keanu::Result<Fit> {
    let b = GraphBuilder::new();
    let mu_log_a = b.gaussian(0.2, 0.5).label("mu_log_a");
    let sigma_log_a = b.gamma(0.1, 2.0).label("sigma_log_a");
    let log_a = b.gaussian_with_shape(&[SUBJECTS], mu_log_a, sigma_log_a);
    let a = log_a.exp().label("a");
    let drift = b
        .gaussian_with_shape(&[CONDITIONS], 0.0, 2.0)
        .label("drift");
    let sigma_drift = b.gamma(0.15, 2.0).label("sigma_drift");
    let subject_drift = b
        .gaussian_with_shape(&[SUBJECTS, CONDITIONS], drift, sigma_drift)
        .label("subject_drift");
    let w = b.beta(2.0, 2.0).label("w");
    // each subject's non-decision time is below their fastest response
    let fastest: Vec<f64> = (0..SUBJECTS)
        .map(|s| {
            trials
                .time
                .iter()
                .zip(&trials.subject)
                .filter(|&(_, &subject)| subject == s as i64)
                .map(|(&t, _)| t)
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let t0 = b
        .uniform_with_shape(&[SUBJECTS], 0.0, DoubleTensor::vector(fastest.clone()))
        .label("t0");
    // each trial's parameters, picked out by its subject and condition
    let subject = b.int_constant(IntTensor::vector(trials.subject.clone()));
    let cell: Vec<i64> = trials
        .subject
        .iter()
        .zip(&trials.condition)
        .map(|(&s, &c)| s * CONDITIONS as i64 + c)
        .collect();
    let trial_drift = subject_drift
        .reshape(&[(SUBJECTS * CONDITIONS) as isize])
        .gather(0, b.int_constant(IntTensor::vector(cell)));
    let rt = b.wiener(
        a.gather(0, subject),
        t0.gather(0, subject),
        w,
        trial_drift,
        0.0,
        IntTensor::vector(trials.response.clone()),
    );
    b.observe(rt, DoubleTensor::vector(trials.time.clone()));
    // start in the middle of the prior, every non-decision time at half the fastest response
    b.set_value(mu_log_a, DoubleTensor::scalar(0.2));
    b.set_value(sigma_log_a, DoubleTensor::scalar(0.2));
    b.set_value(log_a, DoubleTensor::vector(vec![0.2; SUBJECTS]));
    b.set_value(drift, DoubleTensor::vector(vec![1.0; CONDITIONS]));
    b.set_value(sigma_drift, DoubleTensor::scalar(0.2));
    b.set_value(
        subject_drift,
        DoubleTensor::new(&[SUBJECTS, CONDITIONS], vec![1.0; SUBJECTS * CONDITIONS])?,
    );
    b.set_value(w, DoubleTensor::scalar(0.5));
    b.set_value(
        t0,
        DoubleTensor::vector(fastest.iter().map(|f| 0.5 * f).collect()),
    );
    let (drift, w, a, t0, subject_drift) =
        (drift.id(), w.id(), a.id(), t0.id(), subject_drift.id());
    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let nuts = Nuts::default().with_adapt_count(draws);
    let divergences = AtomicUsize::new(0);
    let chains = sample_chains(chains, seed, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &[drift, w, a, t0, subject_drift],
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            rng,
        )
        .map(|(samples, statistics)| {
            divergences.fetch_add(statistics.divergent_steps.len(), Ordering::Relaxed);
            samples
        })
    })?;
    Ok(Fit {
        chains,
        drift,
        w,
        a,
        t0,
        subject_drift,
        divergences: divergences.into_inner(),
    })
}

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let trials = data(DATA_SEED)?;
    let upper = trials.response.iter().filter(|&&r| r == 1).count();
    println!(
        "{} trials, {upper} at the upper boundary, mean response time {:.3} s",
        trials.time.len(),
        trials.time.iter().sum::<f64>() / trials.time.len() as f64
    );
    let fit = run(&trials, 4, 1000, 2)?;
    println!("{} divergent transitions", fit.divergences);
    let report = |name: String, node: NodeId, index: &[usize], truth: f64| -> keanu::Result<()> {
        let s = fit.chains.summarize(node, index)?;
        println!(
            "{name}: mean {:.3}, 90% interval [{:.3}, {:.3}], R-hat {:.3} (truth {truth:.3})",
            s.mean, s.q5, s.q95, s.r_hat
        );
        Ok(())
    };
    for (c, name) in ["hard", "easy"].iter().enumerate() {
        report(format!("drift ({name})"), fit.drift, &[c], TRUE_DRIFT[c])?;
    }
    report("w".into(), fit.w, &[], TRUE_W)?;
    for s in 0..SUBJECTS {
        report(format!("a[{s}]"), fit.a, &[s], TRUE_A[s])?;
        report(format!("t0[{s}]"), fit.t0, &[s], TRUE_T0[s])?;
        for c in 0..CONDITIONS {
            report(
                format!("drift[{s}, {c}]"),
                fit.subject_drift,
                &[s, c],
                trials.drift[s][c],
            )?;
        }
    }
    Ok(())
}
