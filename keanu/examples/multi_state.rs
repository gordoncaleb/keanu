//! How fast do healthy people fall ill, and how fast do the healthy and the ill die, when they are only seen at
//! clinic visits?
//!
//! The illness–death model: a continuous-time Markov chain on three states, healthy (0), ill (1) and dead (2), with
//! transition intensities q₀₁ (falling ill), q₀₂ (dying while healthy) and q₁₂ (dying while ill). Its generator is
//!
//! ```text
//!     ⎡ −(q₀₁ + q₀₂)   q₀₁    q₀₂ ⎤
//! Q = ⎢      0        −q₁₂    q₁₂ ⎥
//!     ⎣      0          0      0  ⎦
//! ```
//!
//! Each subject is seen at irregular visits and only their state at each visit is recorded (panel data): when the
//! transitions happened between visits is unknown, so the observations are interval-censored. The probability of
//! moving from state i at one visit to state j at the next, Δt later, is entry (i, j) of the transition matrix
//! P(Δt) = exp(Q Δt), a [`matrix_exp`](keanu::graph::DoubleVar::matrix_exp) node, which sums over every path between
//! the visits. The likelihood is a categorical draw of each visit's state from the row of P(Δt) of the state before
//! it (as R's `msm` package fits such models), and the intensities have weakly informative log-normal priors:
//!
//! ```text
//! q₀₁, q₀₂, q₁₂ ~ LogNormal(ln 0.2, 1)
//! state at visit k ~ Categorical(row (state at visit k − 1) of exp(Q · (t_k − t_{k−1})))
//! ```
//!
//! The visits are 0.5, 1 or 2 years apart, so one batched exponential of Q times each of the three gaps gives every
//! transition matrix the likelihood needs. The example also reports the posterior probability that someone healthy
//! now is dead within 5 years, entry (0, 2) of exp(5 Q).
//!
//! Data: 300 subjects, healthy at their first visit (time 0) and seen at up to 8 visits until they die, simulated
//! exactly from the chain (exponential holding times) by [`simulate`] with q₀₁ = 0.25, q₀₂ = 0.05 and q₁₂ = 0.4 per
//! year.
//!
//! Run with `cargo run --release --example multi_state`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::mcmc::chains::sample_chains;
use keanu::prelude::*;

/// The true rate of falling ill (per year).
pub const TRUE_ILLNESS: f64 = 0.25;
/// The true rate of dying while healthy (per year).
pub const TRUE_HEALTHY_DEATH: f64 = 0.05;
/// The true rate of dying while ill (per year).
pub const TRUE_ILL_DEATH: f64 = 0.4;
/// The possible gaps between visits (years).
pub const GAPS: [f64; 3] = [0.5, 1.0, 2.0];
/// The number of subjects simulated.
pub const SUBJECTS: usize = 300;
/// The most visits a subject has (including the first).
pub const VISITS: usize = 8;
/// The seed of the simulated data.
pub const DATA_SEED: u64 = 11;
/// The number of chains.
pub const CHAINS: usize = 4;

/// One observed transition: the state at a visit, the state at the next visit and which of [`GAPS`] separates them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Transition {
    /// The state at the earlier visit (0 healthy, 1 ill).
    pub from: usize,
    /// The state at the later visit (0 healthy, 1 ill, 2 dead).
    pub to: usize,
    /// The index into [`GAPS`] of the time between the visits.
    pub gap: usize,
}

/// Simulates [`SUBJECTS`] subjects from the chain with the true intensities, from a generator seeded with `seed`,
/// returning every pair of consecutive visits.
pub fn simulate(seed: u64) -> Vec<Transition> {
    let mut rng = KeanuRng::seed_from_u64(seed);
    let mut exponential = |rate: f64| -rng.next_f64_non_zero().ln() / rate;
    let mut transitions = Vec::new();
    let mut gaps = KeanuRng::seed_from_u64(seed + 1);
    for _ in 0..SUBJECTS {
        // the subject's path: the time it falls ill (if it does) and the time it dies
        let (onset, death) = {
            let to_illness = exponential(TRUE_ILLNESS);
            let to_death = exponential(TRUE_HEALTHY_DEATH);
            if to_illness < to_death {
                (to_illness, to_illness + exponential(TRUE_ILL_DEATH))
            } else {
                (f64::INFINITY, to_death)
            }
        };
        let state_at = |t: f64| {
            if t >= death {
                2
            } else if t >= onset {
                1
            } else {
                0
            }
        };
        let (mut t, mut state) = (0.0, 0);
        for _ in 1..VISITS {
            let gap = gaps.next_below(GAPS.len() as u64) as usize;
            t += GAPS[gap];
            let next = state_at(t);
            transitions.push(Transition {
                from: state,
                to: next,
                gap,
            });
            if next == 2 {
                break;
            }
            state = next;
        }
    }
    transitions
}

/// The model's quantities, by node.
#[derive(Debug, Clone, Copy)]
pub struct Nodes {
    /// The rate of falling ill, q₀₁.
    pub illness: NodeId,
    /// The rate of dying while healthy, q₀₂.
    pub healthy_death: NodeId,
    /// The rate of dying while ill, q₁₂.
    pub ill_death: NodeId,
    /// The probability that someone healthy is dead 5 years later.
    pub dead_in_five_years: NodeId,
}

/// The generator Q of the chain from the three intensities.
fn generator<'b>(
    b: &'b GraphBuilder,
    q01: DoubleVar<'b>,
    q02: DoubleVar<'b>,
    q12: DoubleVar<'b>,
) -> DoubleVar<'b> {
    let zero = b.scalar(0.0);
    let row = |entries: [DoubleVar<'b>; 3]| b.stack(0, &entries);
    b.stack(
        0,
        &[
            row([-(q01 + q02), q01, q02]),
            row([zero, -q12, q12]),
            row([zero, zero, zero]),
        ],
    )
}

/// The model of the observed `transitions`, its intensities at the priors' medians.
pub fn model(transitions: &[Transition]) -> keanu::Result<(Model, State, Nodes)> {
    let b = GraphBuilder::new();
    let rate = |label: &str| {
        let q = b.log_normal(0.2f64.ln(), 1.0).label(label);
        b.set_value(q, DoubleTensor::scalar(0.2));
        q
    };
    let (q01, q02, q12) = (rate("illness"), rate("healthy_death"), rate("ill_death"));
    let q = generator(&b, q01, q02, q12);
    // ANCHOR: transition_matrices
    // exp(Q Δt) for each gap, [gaps, 3, 3], then one row of 3 probabilities per (gap, from) pair
    let gaps = b.constant(DoubleTensor::new(&[GAPS.len(), 1, 1], GAPS.to_vec())?);
    let probabilities = (q * gaps)
        .matrix_exp()
        .reshape(&[3 * GAPS.len() as isize, 3]);
    // ANCHOR_END: transition_matrices
    let rows: Vec<i64> = transitions
        .iter()
        .map(|t| (t.gap * 3 + t.from) as i64)
        .collect();
    let next: Vec<i64> = transitions.iter().map(|t| t.to as i64).collect();
    // ANCHOR: panel_likelihood
    let p = probabilities.gather(0, b.int_constant(IntTensor::vector(rows)));
    let columns: Vec<DoubleVar> = (0..3).map(|j| p.slice(&format!(":, {j}"))).collect();
    let observed = b.categorical(&[transitions.len()], &columns);
    b.observe(observed, IntTensor::vector(next));
    // ANCHOR_END: panel_likelihood
    let dead_in_five_years = (q * 5.0)
        .matrix_exp()
        .take(&[0, 2])
        .label("dead_in_five_years");
    let nodes = Nodes {
        illness: q01.id(),
        healthy_death: q02.id(),
        ill_death: q12.id(),
        dead_in_five_years: dead_in_five_years.id(),
    };
    let (graph, mut state) = b.build()?;
    let all: Vec<NodeId> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(0))?;
    Ok((Model::new(graph, &state)?, state, nodes))
}

/// The posterior draws and where to find each quantity in them.
#[derive(Debug)]
pub struct MultiState {
    /// The simulated transitions.
    pub transitions: Vec<Transition>,
    /// The draws of every chain.
    pub samples: ChainSamples,
    /// The model's quantities.
    pub nodes: Nodes,
}

/// Simulates the data and fits the model with `CHAINS` NUTS chains of `draws` draws each, after as many adaptation
/// steps.
pub fn run(seed: u64, draws: usize) -> keanu::Result<MultiState> {
    let transitions = simulate(DATA_SEED);
    let (model, state, nodes) = model(&transitions)?;
    let nuts = Nuts::default().with_adapt_count(draws);
    let recorded = [
        nodes.illness,
        nodes.healthy_death,
        nodes.ill_death,
        nodes.dead_in_five_years,
    ];
    let samples = sample_chains(CHAINS, seed, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &recorded,
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            rng,
        )
        .map(|(samples, _)| samples)
    })?;
    Ok(MultiState {
        transitions,
        samples,
        nodes,
    })
}

/// The number of draws per chain (after as many adaptation steps).
pub const DEFAULT_SAMPLES: usize = 1000;

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let r = run(1, DEFAULT_SAMPLES)?;
    println!("{} observed transitions", r.transitions.len());
    let n = r.nodes;
    for (name, node, truth) in [
        ("q01 (falling ill)", n.illness, Some(TRUE_ILLNESS)),
        (
            "q02 (death, healthy)",
            n.healthy_death,
            Some(TRUE_HEALTHY_DEATH),
        ),
        ("q12 (death, ill)", n.ill_death, Some(TRUE_ILL_DEATH)),
        ("P(dead in 5 years)", n.dead_in_five_years, None),
    ] {
        let s = r.samples.summarize(node, &[])?;
        let truth = truth.map_or_else(String::new, |t| format!("  (true {t})"));
        println!(
            "{name:<21} mean {:.3}  90% interval [{:.3}, {:.3}]  R-hat {:.3}{truth}",
            s.mean, s.q5, s.q95, s.r_hat
        );
    }
    Ok(())
}
