//! How fast did influenza spread through an English boarding school in 1978, and how long were the boys ill?
//!
//! The SIR (susceptible, infected, recovered) model of an epidemic in a closed population of N people, with
//! transmission rate β and recovery rate γ:
//!
//! ```text
//! dS/dt = −β S I / N        dI/dt = β S I / N − γ I        dR/dt = γ I
//! ```
//!
//! starting from one infected person, S(0) = N − 1, I(0) = 1, R(0) = 0. The basic reproduction number R₀ = β / γ is
//! the number of people one infected person infects in a wholly susceptible population, and 1 / γ the mean time to
//! recovery. The daily number of boys confined to bed is a noisy count of I(t), negative binomial with mean I(t) and
//! dispersion φ (variance I + I² / φ), as in the Stan case study by Grinsztajn, Semenova, Margossian and Riou
//! (2021), "Bayesian workflow for disease transmission modeling in Stan", whose priors this example uses:
//!
//! ```text
//! β ~ N(2, 1) truncated to β > 0      γ ~ N(0.4, 0.5) truncated to γ > 0      1 / φ ~ Exponential(5)
//! cases(t) ~ NegativeBinomial(mean I(t), dispersion φ)    for t = 1, ..., 14 days
//! ```
//!
//! The state is the solution of the ODE at the 14 days, an [`ode`](keanu::graph::GraphBuilder::ode) node: the body
//! graph computes dy/dt from placeholders for the time, the state and the parameters, and NUTS takes gradients
//! through the solver by forward sensitivities.
//!
//! Data: the outbreak of influenza at a boys' boarding school in the north of England reported in the British
//! Medical Journal (anonymous, 1978, "Influenza in a boarding school", BMJ 1(6112), p. 587): 763 boys, of whom 512
//! were confined to bed at some point over 14 days, with the daily counts in [`BOARDING_SCHOOL`]. The posterior agrees
//! with the case study's (β about 1.73, γ about 0.54, R₀ about 3.2 and a recovery time of about 1.85 days). The
//! example's test also fits data simulated from the model with known β and γ ([`simulate`]) and checks that their
//! 90% posterior intervals cover them.
//!
//! Run with `cargo run --release --example sir`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::mcmc::chains::run_chains;
use keanu::graph::ode::OdeOptions;
use keanu::graph::op::DistributionKind;
use keanu::prelude::*;

/// The number of boys at the school.
pub const POPULATION: f64 = 763.0;

/// The number of boys confined to bed on each of days 1 to 14 (BMJ, 1978).
pub const BOARDING_SCHOOL: [i64; 14] = [3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4];

/// The number of chains.
pub const CHAINS: usize = 4;

/// The days of the counts, 1 to `days`.
fn days(days: usize) -> Vec<f64> {
    (1..=days).map(|t| t as f64).collect()
}

/// The number infected at each day, I(t) for t = 1, ..., `days`, as a function of β and γ: the infected column of the
/// ODE's solution.
fn infected<'b>(
    b: &'b GraphBuilder,
    beta: DoubleVar<'b>,
    gamma: DoubleVar<'b>,
    days: usize,
    options: OdeOptions,
) -> DoubleVar<'b> {
    let y0 = b.constant(DoubleTensor::vector(vec![POPULATION - 1.0, 1.0, 0.0]));
    let times = b.constant(DoubleTensor::vector(self::days(days)));
    let parameters = b.stack(0, &[beta, gamma]);
    // ANCHOR: sir_ode
    let states = b.ode(
        y0,
        times,
        b.scalar(0.0),
        &[parameters],
        options,
        |body, _t, y, theta| {
            let (s, i) = (y.take(&[0]), y.take(&[1]));
            let (beta, gamma) = (theta[0].take(&[0]), theta[0].take(&[1]));
            let infection = beta * s * i / POPULATION;
            let recovery = gamma * i;
            body.stack(0, &[-infection, infection - recovery, recovery])
        },
    );
    // ANCHOR_END: sir_ode
    states.slice(":, 1").label("infected")
}

/// Daily counts simulated from the model with transmission rate `beta`, recovery rate `gamma` and dispersion
/// `phi`, from a generator seeded with `seed`.
pub fn simulate(
    beta: f64,
    gamma: f64,
    phi: f64,
    days: usize,
    seed: u64,
) -> keanu::Result<Vec<i64>> {
    let b = GraphBuilder::new();
    let i = infected(
        &b,
        b.scalar(beta),
        b.scalar(gamma),
        days,
        OdeOptions::default(),
    );
    let cases = b.negative_binomial(i, phi).id();
    let (graph, mut state) = b.build()?;
    state.lazy_eval(&graph, &[cases], &mut KeanuRng::seed_from_u64(seed))?;
    Ok(state.i64(cases).unwrap().as_slice().to_vec())
}

/// A Gaussian truncated to positive values.
fn positive(b: &GraphBuilder, mu: f64, sigma: f64) -> DoubleVar<'_> {
    let parameters = [b.scalar(mu).id(), b.scalar(sigma).id()];
    b.truncated(DistributionKind::Gaussian, &parameters, 0.0, f64::INFINITY)
}

/// The quantities of the model, by node.
#[derive(Debug, Clone, Copy)]
pub struct Nodes {
    /// The transmission rate β (per day).
    pub beta: NodeId,
    /// The recovery rate γ (per day).
    pub gamma: NodeId,
    /// The basic reproduction number R₀ = β / γ.
    pub r0: NodeId,
    /// The mean recovery time 1 / γ (days).
    pub recovery_time: NodeId,
    /// The negative binomial dispersion φ.
    pub phi: NodeId,
    /// The number infected on each day, shape `[days]`.
    pub infected: NodeId,
}

/// The model of daily `cases`, with the ODE solved with `options`, its latents at the priors' central values.
pub fn model(cases: &[i64], options: OdeOptions) -> keanu::Result<(Model, State, Nodes)> {
    let b = GraphBuilder::new();
    let beta = positive(&b, 2.0, 1.0).label("beta");
    let gamma = positive(&b, 0.4, 0.5).label("gamma");
    let phi_inv = b.exponential(5.0).label("phi_inv");
    b.set_value(beta, DoubleTensor::scalar(2.0));
    b.set_value(gamma, DoubleTensor::scalar(0.4));
    b.set_value(phi_inv, DoubleTensor::scalar(0.2));
    let phi = phi_inv.reciprocal().label("phi");
    let i = infected(&b, beta, gamma, cases.len(), options);
    // ANCHOR: sir_likelihood
    let observed = b.negative_binomial(i, phi);
    b.observe(observed, IntTensor::vector(cases.to_vec()));
    // ANCHOR_END: sir_likelihood
    let r0 = (beta / gamma).label("r0");
    let recovery_time = gamma.reciprocal().label("recovery_time");
    let nodes = Nodes {
        beta: beta.id(),
        gamma: gamma.id(),
        r0: r0.id(),
        recovery_time: recovery_time.id(),
        phi: phi.id(),
        infected: i.id(),
    };
    let (graph, mut state) = b.build()?;
    let all: Vec<NodeId> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(0))?;
    Ok((Model::new(graph, &state)?, state, nodes))
}

/// The posterior draws and where to find each quantity in them.
#[derive(Debug)]
pub struct Sir {
    /// The draws of every chain.
    pub samples: ChainSamples,
    /// The number of divergent transitions after adaptation, over all chains.
    pub divergences: usize,
    /// The model's quantities.
    pub nodes: Nodes,
}

/// Fits the model to daily `cases` with `chains` NUTS chains of `draws` draws each, after as many adaptation steps.
pub fn fit(cases: &[i64], chains: usize, seed: u64, draws: usize) -> keanu::Result<Sir> {
    let (model, state, nodes) = model(cases, OdeOptions::default())?;
    let recorded = [
        nodes.beta,
        nodes.gamma,
        nodes.r0,
        nodes.recovery_time,
        nodes.phi,
        nodes.infected,
    ];
    let nuts = Nuts::default().with_adapt_count(draws);
    let chains = run_chains(chains, seed, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &recorded,
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            rng,
        )
    })?;
    let divergences = chains.iter().map(|(_, s)| s.divergence_count()).sum();
    let samples = ChainSamples::new(chains.into_iter().map(|(samples, _)| samples).collect())?;
    Ok(Sir {
        samples,
        divergences,
        nodes,
    })
}

/// Fits the model to the boarding school counts with `CHAINS` chains of `draws` draws each.
pub fn run(seed: u64, draws: usize) -> keanu::Result<Sir> {
    fit(&BOARDING_SCHOOL, CHAINS, seed, draws)
}

/// The number of draws per chain (after as many adaptation steps).
pub const DEFAULT_SAMPLES: usize = 1000;

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let r = run(1, DEFAULT_SAMPLES)?;
    println!("{} divergences", r.divergences);
    let n = r.nodes;
    for (name, node) in [
        ("beta", n.beta),
        ("gamma", n.gamma),
        ("R0", n.r0),
        ("recovery time", n.recovery_time),
        ("phi", n.phi),
    ] {
        let s = r.samples.summarize(node, &[])?;
        println!(
            "{name:<14} mean {:>6.3}  90% interval [{:.3}, {:.3}]  R-hat {:.3}",
            s.mean, s.q5, s.q95, s.r_hat
        );
    }
    println!("\nday  in bed  infected (posterior mean)");
    for (t, cases) in BOARDING_SCHOOL.iter().enumerate() {
        let s = r.samples.summarize(n.infected, &[t])?;
        println!("{:>3}  {cases:>6}  {:>8.1}", t + 1, s.mean);
    }
    Ok(())
}
