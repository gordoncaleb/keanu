//! When does a series switch between a calm regime and an excited one, and how often does it switch?
//!
//! A two-state hidden Markov model fitted to synthetic regime-switching data with NUTS. A hidden state s_t in
//! {0, 1} starts with equal probabilities and moves from one step to the next with the transition matrix P (row i
//! holds the probabilities of moving from state i); each observation is Gaussian with its state's mean:
//!
//! ```text
//! P_0 ~ Dirichlet(1, 1)    P_1 ~ Dirichlet(1, 1)
//! m_0 ~ N(0, 5)    m_1 − m_0 ~ HalfGaussian(5)    sigma ~ HalfGaussian(2)
//! s_0 ~ Categorical(1/2, 1/2)    s_t ~ Categorical(P_{s_{t−1}})    y_t ~ N(m_{s_t}, sigma)
//! ```
//!
//! The hidden states are summed out exactly by the hidden Markov model distribution's forward algorithm, so NUTS
//! samples only the continuous parameters: the two transition rows (each a Dirichlet, stacked into P, which keeps
//! them on the simplex under NUTS's transforms), the means and the noise scale.
//!
//! Label switching: the likelihood is unchanged when the two states swap labels (swapping the means and both the
//! rows and the columns of P), so an unconstrained posterior has two copies of every mode and different chains
//! settle in different copies, which makes R-hat meaningless. Ordering the emission means removes the copy: the
//! first mean is a latent and the second is the first plus a positive gap, stacked and cumulatively summed as in
//! the Gaussian mixture example (Stan's ordered transform), so state 0 is always the regime with the lower mean.
//!
//! Run with `cargo run --release --example hmm`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::mcmc::chains::sample_chains;
use keanu::graph::op::DistributionKind;
use keanu::prelude::*;

/// The transition matrix the data are simulated with, row by row: state 0 persists for 20 steps on average and
/// state 1 for 10.
pub const TRUE_TRANSITION: [[f64; 2]; 2] = [[0.95, 0.05], [0.10, 0.90]];
/// The emission means of the two states.
pub const TRUE_MEANS: [f64; 2] = [-1.0, 2.0];
/// The emission standard deviation, shared by both states.
pub const TRUE_SIGMA: f64 = 1.0;
/// The probabilities of the first state.
pub const INITIAL: [f64; 2] = [0.5, 0.5];

/// The number of chains.
pub const CHAINS: usize = 4;
/// The seed the example simulates its data with.
pub const DATA_SEED: u64 = 3;
/// The number of steps the example simulates.
pub const STEPS: usize = 400;

/// A simulated series: the hidden states and the observations.
#[derive(Debug, Clone, Default)]
pub struct Series {
    /// The hidden state of each step.
    pub states: Vec<usize>,
    /// The observation of each step.
    pub observations: Vec<f64>,
}

/// Simulates `steps` steps of the true model from a generator seeded with `seed`.
pub fn simulate(steps: usize, seed: u64) -> Series {
    let mut rng = KeanuRng::seed_from_u64(seed);
    let mut series = Series::default();
    let mut state = usize::from(rng.next_f64() >= INITIAL[0]);
    for t in 0..steps {
        if t > 0 {
            state = usize::from(rng.next_f64() >= TRUE_TRANSITION[state][0]);
        }
        series.states.push(state);
        series
            .observations
            .push(TRUE_MEANS[state] + TRUE_SIGMA * rng.next_gaussian());
    }
    series
}

/// The posterior draws of every chain, and the nodes holding the parameters.
#[derive(Debug)]
pub struct Hmm {
    /// The draws of every chain.
    pub samples: ChainSamples,
    /// The transition matrix, shape [2, 2], row i the probabilities of moving from state i.
    pub transition: NodeId,
    /// The emission means, shape [2], increasing.
    pub means: NodeId,
    /// The emission standard deviation.
    pub sigma: NodeId,
}

/// Fits the model to `observations` with `CHAINS` NUTS chains of `draws` draws each (after as many adaptation
/// steps).
pub fn fit(observations: &[f64], seed: u64, draws: usize) -> keanu::Result<Hmm> {
    let b = GraphBuilder::new();
    // one Dirichlet per row of the transition matrix, stacked
    let rows = [
        b.dirichlet(DoubleTensor::vector(vec![1.0, 1.0])),
        b.dirichlet(DoubleTensor::vector(vec![1.0, 1.0])),
    ];
    let transition = b.stack(0, &rows).label("transition");
    // ordered means: the first, and a positive gap to the second
    let first = b.gaussian_with_shape(&[1], 0.0, 5.0);
    let gap = b.half_gaussian_with_shape(&[1], 5.0);
    let means = b.concat(0, &[first, gap]).cumsum(0).label("means");
    let sigma = b.half_gaussian(2.0).label("sigma");
    let y: DoubleVar = b.hidden_markov_model(
        observations.len(),
        DistributionKind::Gaussian,
        &[means.id(), sigma.id()],
        DoubleTensor::vector(INITIAL.to_vec()),
        transition,
    );
    b.observe(y, DoubleTensor::vector(observations.to_vec()));
    // start every chain from the same plausible point; NUTS adapts from there
    for row in rows {
        b.set_value(row, DoubleTensor::vector(vec![0.5, 0.5]));
    }
    b.set_value(first, DoubleTensor::vector(vec![0.0]));
    b.set_value(gap, DoubleTensor::vector(vec![1.0]));
    b.set_value(sigma, DoubleTensor::scalar(1.0));
    let (transition, means, sigma) = (transition.id(), means.id(), sigma.id());

    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let nuts = Nuts::default().with_adapt_count(draws);
    let samples = sample_chains(CHAINS, seed, &state, |state, rng| {
        let (samples, _) = nuts.sample_with_statistics(
            &model,
            state,
            &[transition, means, sigma],
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            rng,
        )?;
        Ok(samples)
    })?;
    Ok(Hmm {
        samples,
        transition,
        means,
        sigma,
    })
}

/// Simulates `STEPS` steps from a fixed seed and fits the model to them with `draws` draws per chain.
pub fn run(seed: u64, draws: usize) -> keanu::Result<Hmm> {
    fit(&simulate(STEPS, DATA_SEED).observations, seed, draws)
}

/// The number of draws per chain (after as many adaptation steps).
pub const DEFAULT_SAMPLES: usize = 1000;

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let r = run(1, DEFAULT_SAMPLES)?;
    for (i, row) in TRUE_TRANSITION.iter().enumerate() {
        for (j, truth) in row.iter().enumerate() {
            let s = r.samples.summarize(r.transition, &[i, j])?;
            println!(
                "P[{i}][{j}]   mean {:.3}  sd {:.3}  (true {truth:.2})  R-hat {:.3}",
                s.mean, s.sd, s.r_hat
            );
        }
    }
    for (k, truth) in TRUE_MEANS.iter().enumerate() {
        let s = r.samples.summarize(r.means, &[k])?;
        println!(
            "mean {k}   mean {:.3}  sd {:.3}  (true {truth:.2})  R-hat {:.3}",
            s.mean, s.sd, s.r_hat
        );
    }
    let s = r.samples.summarize(r.sigma, &[])?;
    println!(
        "sigma    mean {:.3}  sd {:.3}  (true {TRUE_SIGMA:.2})  R-hat {:.3}",
        s.mean, s.sd, s.r_hat
    );
    Ok(())
}
