//! How quickly is an oral dose of a drug absorbed and cleared?
//!
//! The one-compartment model with first-order absorption, the standard model of an oral dose (as for the
//! theophylline data in R's `Theoph`): the dose D enters the gut at time 0, moves into the body (the central
//! compartment, of volume V) at the absorption rate kₐ, and is cleared from it at the elimination rate kₑ = CL / V,
//! for clearance CL:
//!
//! ```text
//! dA_gut/dt = −kₐ A_gut        dA_central/dt = kₐ A_gut − kₑ A_central        A_gut(0) = D, A_central(0) = 0
//! concentration C(t) = A_central(t) / V
//! ```
//!
//! Its solution has a closed form, C(t) = D kₐ / (V (kₐ − kₑ)) · (e^(−kₑ t) − e^(−kₐ t)), so the example fits the
//! model twice, once with the ODE solved numerically (an [`ode`](keanu::graph::GraphBuilder::ode) node, as for a
//! model with no closed form) and once with the formula, and its test checks that the two agree. The measured
//! concentrations have a proportional (log-normal) error:
//!
//! ```text
//! kₐ ~ LogNormal(ln 1, 0.5)    CL ~ LogNormal(ln 3, 0.5)    V ~ LogNormal(ln 30, 0.5)    σ ~ HalfNormal(0.5)
//! C_obs(tᵢ) ~ LogNormal(ln C(tᵢ), σ)
//! ```
//!
//! The priors keep kₐ above kₑ (absorption faster than elimination), which removes the "flip-flop" mode the formula
//! is symmetric under.
//!
//! Data: one subject's concentrations at the sampling times of the `Theoph` study, simulated from the model by
//! [`simulate`] with a 320 mg dose, kₐ = 1.5 per hour, CL = 2.8 L per hour, V = 35 L and σ = 0.1.
//!
//! Run with `cargo run --release --example pharmacokinetics`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::mcmc::chains::sample_chains;
use keanu::graph::ode::OdeOptions;
use keanu::prelude::*;

/// The dose (mg).
pub const DOSE: f64 = 320.0;
/// The sampling times (hours after the dose), those of the `Theoph` study.
pub const TIMES: [f64; 10] = [0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 9.0, 12.0, 24.0];
/// The absorption rate the data were simulated with (per hour).
pub const TRUE_KA: f64 = 1.5;
/// The clearance the data were simulated with (L per hour).
pub const TRUE_CL: f64 = 2.8;
/// The volume of distribution the data were simulated with (L).
pub const TRUE_V: f64 = 35.0;
/// The measurement error the data were simulated with (on the log scale).
pub const TRUE_SIGMA: f64 = 0.1;
/// The seed of the simulated data.
pub const DATA_SEED: u64 = 7;
/// The number of chains.
pub const CHAINS: usize = 4;

/// How the concentration is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Solution {
    /// By solving the ODE numerically.
    Ode,
    /// By the closed form.
    Analytic,
}

/// The concentration at [`TIMES`] given the absorption rate, the clearance and the volume.
fn concentration<'b>(
    b: &'b GraphBuilder,
    ka: DoubleVar<'b>,
    cl: DoubleVar<'b>,
    v: DoubleVar<'b>,
    solution: Solution,
) -> DoubleVar<'b> {
    let ke = cl / v;
    let times = b.constant(DoubleTensor::vector(TIMES.to_vec()));
    match solution {
        Solution::Ode => {
            // ANCHOR: pk_ode
            let rates = b.stack(0, &[ka, ke]);
            let amounts = b.ode(
                b.constant(DoubleTensor::vector(vec![DOSE, 0.0])),
                times,
                b.scalar(0.0),
                &[rates],
                OdeOptions::default(),
                |body, _t, y, theta| {
                    let (ka, ke) = (theta[0].take(&[0]), theta[0].take(&[1]));
                    let (gut, central) = (y.take(&[0]), y.take(&[1]));
                    body.stack(0, &[-(ka * gut), ka * gut - ke * central])
                },
            );
            // ANCHOR_END: pk_ode
            amounts.slice(":, 1") / v
        }
        Solution::Analytic => {
            let decay = |rate: DoubleVar<'b>| (-(rate * times)).exp();
            ka * DOSE / (v * (ka - ke)) * (decay(ke) - decay(ka))
        }
    }
}

/// Concentrations at [`TIMES`] simulated from the model with the true parameters, from a generator seeded with
/// `seed`.
pub fn simulate(seed: u64) -> keanu::Result<Vec<f64>> {
    let b = GraphBuilder::new();
    let c = concentration(
        &b,
        b.scalar(TRUE_KA),
        b.scalar(TRUE_CL),
        b.scalar(TRUE_V),
        Solution::Analytic,
    );
    let observed = b.log_normal(c.ln(), TRUE_SIGMA).id();
    let (graph, mut state) = b.build()?;
    state.lazy_eval(&graph, &[observed], &mut KeanuRng::seed_from_u64(seed))?;
    Ok(state.f64(observed).unwrap().as_slice().to_vec())
}

/// The model's parameters, by node.
#[derive(Debug, Clone, Copy)]
pub struct Nodes {
    /// The absorption rate kₐ (per hour).
    pub ka: NodeId,
    /// The clearance CL (L per hour).
    pub cl: NodeId,
    /// The volume of distribution V (L).
    pub v: NodeId,
    /// The measurement error σ.
    pub sigma: NodeId,
}

/// The model of the measured `concentrations`, with the concentration computed as `solution` says, its latents at
/// the priors' medians.
pub fn model(concentrations: &[f64], solution: Solution) -> keanu::Result<(Model, State, Nodes)> {
    let b = GraphBuilder::new();
    let ka = b.log_normal(1.0f64.ln(), 0.5).label("ka");
    let cl = b.log_normal(3.0f64.ln(), 0.5).label("cl");
    let v = b.log_normal(30.0f64.ln(), 0.5).label("v");
    let sigma = b.half_gaussian(0.5).label("sigma");
    b.set_value(ka, DoubleTensor::scalar(1.0));
    b.set_value(cl, DoubleTensor::scalar(3.0));
    b.set_value(v, DoubleTensor::scalar(30.0));
    b.set_value(sigma, DoubleTensor::scalar(0.3));
    let c = concentration(&b, ka, cl, v, solution);
    let observed = b.log_normal(c.ln(), sigma);
    b.observe(observed, DoubleTensor::vector(concentrations.to_vec()));
    let nodes = Nodes {
        ka: ka.id(),
        cl: cl.id(),
        v: v.id(),
        sigma: sigma.id(),
    };
    let (graph, mut state) = b.build()?;
    let all: Vec<NodeId> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(0))?;
    Ok((Model::new(graph, &state)?, state, nodes))
}

/// The posterior draws of one fit.
#[derive(Debug)]
pub struct Fit {
    /// The draws of every chain.
    pub samples: ChainSamples,
    /// The parameters.
    pub nodes: Nodes,
}

/// Fits the model to `concentrations` with `CHAINS` NUTS chains of `draws` draws each, after as many adaptation
/// steps.
pub fn fit(
    concentrations: &[f64],
    solution: Solution,
    seed: u64,
    draws: usize,
) -> keanu::Result<Fit> {
    let (model, state, nodes) = model(concentrations, solution)?;
    let nuts = Nuts::default().with_adapt_count(draws);
    let recorded = [nodes.ka, nodes.cl, nodes.v, nodes.sigma];
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
    Ok(Fit { samples, nodes })
}

/// Both fits of the simulated data.
#[derive(Debug)]
pub struct Results {
    /// The simulated concentrations.
    pub concentrations: Vec<f64>,
    /// The fit with the ODE solved numerically.
    pub ode: Fit,
    /// The fit with the closed form.
    pub analytic: Fit,
}

/// Simulates the data and fits both models with `draws` draws per chain.
pub fn run(seed: u64, draws: usize) -> keanu::Result<Results> {
    let concentrations = simulate(DATA_SEED)?;
    let ode = fit(&concentrations, Solution::Ode, seed, draws)?;
    let analytic = fit(&concentrations, Solution::Analytic, seed, draws)?;
    Ok(Results {
        concentrations,
        ode,
        analytic,
    })
}

/// The number of draws per chain (after as many adaptation steps).
pub const DEFAULT_SAMPLES: usize = 1000;

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let r = run(1, DEFAULT_SAMPLES)?;
    println!("hours  concentration (mg/L)");
    for (t, c) in TIMES.iter().zip(&r.concentrations) {
        println!("{t:>5}  {c:.3}");
    }
    println!("\n         ODE: mean (90% interval)      closed form: mean (90% interval)   true");
    let n = r.ode.nodes;
    for (name, node, truth) in [
        ("ka", n.ka, TRUE_KA),
        ("CL", n.cl, TRUE_CL),
        ("V", n.v, TRUE_V),
        ("sigma", n.sigma, TRUE_SIGMA),
    ] {
        let a = r.ode.samples.summarize(node, &[])?;
        let b = r.analytic.samples.summarize(node, &[])?;
        println!(
            "{name:<6} {:>7.3} ({:.3}, {:.3})      {:>7.3} ({:.3}, {:.3})      {truth}",
            a.mean, a.q5, a.q95, b.mean, b.q5, b.q95
        );
    }
    Ok(())
}
