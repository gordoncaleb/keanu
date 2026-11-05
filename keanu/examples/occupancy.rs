//! Which sites does a species occupy, when it is sometimes missed where it is present?
//!
//! A site-occupancy model (MacKenzie et al. 2002) on simulated surveys: each of N sites is occupied with probability
//! ψ, and each of J visits to an occupied site detects the species with probability p (an empty site is never a
//! detection). A site with no detections is either empty or occupied and missed every time, so ψ and p are only
//! identified together, through the pattern of repeated visits.
//!
//! The model is written three ways, which have the same posterior for ψ and p:
//!
//! - [`run_latent`] keeps each site's occupancy as a Bernoulli latent, which the default compound sampler updates with
//!   Gibbs steps between NUTS steps on ψ and p;
//! - [`run_marginalized`] sums the occupancies out with `marginalize`, leaving a model of ψ and p alone for NUTS, and
//!   recovers each site's posterior probability of being occupied with `recover_marginals`;
//! - [`run_zero_inflated`] writes the number of detections at each site directly as a zero-inflated binomial: 0 for
//!   an empty site (probability 1 − ψ), else Binomial(J, p).
//!
//! Run with `cargo run --release --example occupancy`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::marginal::recover_marginals;
use keanu::algorithms::mcmc::chains::sample_chains;
use keanu::graph::op::DistributionKind;
use keanu::prelude::*;

/// The occupancy probability the data are simulated with.
pub const TRUE_PSI: f64 = 0.65;
/// The detection probability the data are simulated with.
pub const TRUE_P: f64 = 0.4;
/// The number of sites.
pub const SITES: usize = 120;
/// The number of visits to each site.
pub const VISITS: usize = 4;

/// Simulated surveys: whether each site is occupied, and the detections, [sites, visits] in row-major order.
#[derive(Debug, Clone)]
pub struct Surveys {
    pub occupied: Vec<bool>,
    pub detections: Vec<bool>,
}

impl Surveys {
    /// The number of detections at each site.
    pub fn counts(&self) -> Vec<i64> {
        self.detections
            .chunks(VISITS)
            .map(|visits| visits.iter().filter(|&&d| d).count() as i64)
            .collect()
    }
}

/// Simulates [`SITES`] sites with [`VISITS`] visits each.
pub fn data(seed: u64) -> Surveys {
    let mut rng = KeanuRng::seed_from_u64(seed);
    let occupied: Vec<bool> = (0..SITES).map(|_| rng.next_f64() < TRUE_PSI).collect();
    let detections = occupied
        .iter()
        .flat_map(|&o| {
            (0..VISITS)
                .map(|_| o && rng.next_f64() < TRUE_P)
                .collect::<Vec<_>>()
        })
        .collect();
    Surveys {
        occupied,
        detections,
    }
}

/// A fit: the posterior draws of every chain, the nodes of ψ and p, and the posterior mean number of occupied sites.
#[derive(Debug)]
pub struct Fit {
    pub chains: ChainSamples,
    pub psi: NodeId,
    pub p: NodeId,
    pub occupied_sites: f64,
}

/// Fits with each site's occupancy a Bernoulli latent, sampled by the default compound sampler (Gibbs steps for the
/// occupancies, NUTS for ψ and p), with `chains` chains of `draws` draws after as many warm-up steps.
pub fn run_latent(surveys: &Surveys, chains: usize, draws: usize, seed: u64) -> keanu::Result<Fit> {
    let b = GraphBuilder::new();
    let psi = b.beta(1.0, 1.0).label("psi");
    let p = b.beta(1.0, 1.0).label("p");
    let occupied = b.bernoulli_with_shape(&[SITES], psi).label("occupied");
    // a visit detects the species with probability p at an occupied site and never at an empty one
    let detect = occupied.to_double_mask().reshape(&[SITES as isize, 1]) * p;
    let y = b.bernoulli_with_shape(&[SITES, VISITS], detect);
    b.observe(
        y,
        BoolTensor::new(&[SITES, VISITS], surveys.detections.clone())?,
    );
    // start with every site occupied, which every detection allows
    b.set_value(occupied, BoolTensor::full(&[SITES], true));
    b.set_value(psi, DoubleTensor::scalar(0.5));
    b.set_value(p, DoubleTensor::scalar(0.5));
    let (psi, p, occupied) = (psi.id(), p.id(), occupied.id());
    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;

    let Sampler::Compound(compound) = Sampler::for_model(&model, &state) else {
        unreachable!("a model with discrete and continuous latents gets the compound sampler");
    };
    let compound = compound.with_nuts_settings(Nuts::default().with_adapt_count(draws));
    let chains = sample_chains(chains, seed, &state, |state, rng| {
        compound
            .sample_with_statistics(
                &model,
                state,
                &[psi, p, occupied],
                2 * draws,
                SamplingOptions::default().with_drop(draws),
                rng,
            )
            .map(|(samples, _)| samples)
    })?;
    let mut total = 0.0;
    let mut count = 0;
    for chain in chains.chains() {
        for draw in chain.bool(occupied).unwrap().as_slice() {
            total += draw.as_slice().iter().filter(|&&o| o).count() as f64;
            count += 1;
        }
    }
    Ok(Fit {
        chains,
        psi,
        p,
        occupied_sites: total / count as f64,
    })
}

/// Fits the same model with the occupancies marginalised out, by NUTS on ψ and p alone; the number of occupied sites
/// comes from each site's recovered posterior probability of being occupied.
pub fn run_marginalized(
    surveys: &Surveys,
    chains: usize,
    draws: usize,
    seed: u64,
) -> keanu::Result<Fit> {
    let b = GraphBuilder::new();
    let psi = b.beta(1.0, 1.0).label("psi");
    let p = b.beta(1.0, 1.0).label("p");
    let occupied = b.bernoulli_with_shape(&[SITES], psi).label("occupied");
    let detect = occupied.to_double_mask().reshape(&[SITES as isize, 1]) * p;
    let y = b.bernoulli_with_shape(&[SITES, VISITS], detect);
    b.observe(
        y,
        BoolTensor::new(&[SITES, VISITS], surveys.detections.clone())?,
    );
    b.set_value(occupied, BoolTensor::full(&[SITES], true));
    b.set_value(psi, DoubleTensor::scalar(0.5));
    b.set_value(p, DoubleTensor::scalar(0.5));
    let (psi, p, occupied) = (psi.id(), p.id(), occupied.id());
    let (graph, state) = b.build()?;

    // each site's likelihood becomes ψ Π p^y (1 − p)^(1 − y) + (1 − ψ) [no detections], a factor of ψ and p
    let (graph, state, marginalized) = marginalize(&graph, &state, &[occupied])?;
    let (psi, p) = (
        marginalized.node(psi).expect("ψ is kept"),
        marginalized.node(p).expect("p is kept"),
    );
    let model = Model::new(graph, &state)?;
    let nuts = Nuts::default().with_adapt_count(draws);
    let chains = sample_chains(chains, seed, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &[psi, p],
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            rng,
        )
        .map(|(samples, _)| samples)
    })?;
    // the probability that each site is occupied given each draw, averaged over every chain's draws
    let mut occupied_sites = 0.0;
    for chain in chains.chains() {
        let sites = &recover_marginals(&model, &state, &marginalized, chain)?[0];
        let mean = sites.mean()?;
        let probabilities = mean.as_slice();
        let total: f64 = (0..SITES).map(|i| probabilities[2 * i + 1]).sum();
        occupied_sites += total / chains.num_chains() as f64;
    }
    Ok(Fit {
        chains,
        psi,
        p,
        occupied_sites,
    })
}

/// Fits the same model written directly: each site's number of detections is 0 with probability 1 − ψ (an empty
/// site), else Binomial(J, p). The count is sufficient for p, so this is the same likelihood up to a constant. The
/// number of occupied sites is the detected ones plus, for each site without a detection, its posterior probability
/// of being occupied, ψ(1 − p)^J / (ψ(1 − p)^J + 1 − ψ), averaged over the draws.
pub fn run_zero_inflated(
    surveys: &Surveys,
    chains: usize,
    draws: usize,
    seed: u64,
) -> keanu::Result<Fit> {
    let b = GraphBuilder::new();
    let psi = b.beta(1.0, 1.0).label("psi");
    let p = b.beta(1.0, 1.0).label("p");
    let visits = b.int_scalar(VISITS as i64);
    let detections: IntVar = b.zero_inflated_with_shape(
        &[SITES],
        DistributionKind::Binomial,
        &[p.id(), visits.id()],
        -psi + 1.0,
    );
    b.observe(detections, IntTensor::vector(surveys.counts()));
    b.set_value(psi, DoubleTensor::scalar(0.5));
    b.set_value(p, DoubleTensor::scalar(0.5));
    let (psi, p) = (psi.id(), p.id());
    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let nuts = Nuts::default().with_adapt_count(draws);
    let chains = sample_chains(chains, seed, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &[psi, p],
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            rng,
        )
        .map(|(samples, _)| samples)
    })?;
    let counts = surveys.counts();
    let detected = counts.iter().filter(|&&c| c > 0).count() as f64;
    let undetected = SITES as f64 - detected;
    let (mut total, mut count) = (0.0, 0);
    for chain in chains.chains() {
        let psis = chain.f64(psi).unwrap();
        let ps = chain.f64(p).unwrap();
        for (psi, p) in psis.as_slice().iter().zip(ps.as_slice()) {
            let (psi, p) = (psi.to_scalar().unwrap(), p.to_scalar().unwrap());
            let missed = psi * (1.0 - p).powi(VISITS as i32);
            total += detected + undetected * missed / (missed + 1.0 - psi);
            count += 1;
        }
    }
    Ok(Fit {
        chains,
        psi,
        p,
        occupied_sites: total / count as f64,
    })
}

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let surveys = data(3);
    let occupied = surveys.occupied.iter().filter(|&&o| o).count();
    let detected = surveys.counts().iter().filter(|&&c| c > 0).count();
    println!("{SITES} sites, {occupied} occupied, {detected} with a detection in {VISITS} visits");
    for (name, fit) in [
        (
            "occupancy latents (compound sampler)",
            run_latent(&surveys, 4, 500, 11)?,
        ),
        (
            "occupancy marginalised (NUTS)",
            run_marginalized(&surveys, 4, 500, 11)?,
        ),
        (
            "zero-inflated binomial (NUTS)",
            run_zero_inflated(&surveys, 4, 500, 11)?,
        ),
    ] {
        println!("{name}:");
        for (what, node) in [("psi", fit.psi), ("p", fit.p)] {
            let s = fit.chains.summarize(node, &[])?;
            println!(
                "  {what}: mean {:.3}, 90% interval [{:.3}, {:.3}], R-hat {:.3}",
                s.mean, s.q5, s.q95, s.r_hat
            );
        }
        println!("  occupied sites: {:.1}", fit.occupied_sites);
    }
    Ok(())
}
