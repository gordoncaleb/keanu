//! How much does indoor radon vary between Minnesota counties, and how much does a basement add?
//!
//! The varying-intercept model of Gelman & Hill, *Data Analysis Using Regression and Multilevel/Hierarchical
//! Models* (2007), ch. 12: house i in county j\[i\] has log radon
//!
//! ```text
//! log_radon_i ~ N(a[j[i]] + b·floor_i, sigma_y)    a_j ~ N(mu_a, sigma_a)
//! mu_a ~ N(0, 10)    b ~ N(0, 10)    sigma_a ~ Exponential(1)    sigma_y ~ Exponential(1)
//! ```
//!
//! where floor is 0 for a measurement in the basement and 1 on the ground floor. The priors are those of the PyMC
//! radon notebook. Each house's county intercept is picked out of the 85 with `gather` over the county index.
//!
//! The intercepts are sampled in the centred form above: with 919 houses the data pin sigma_a down well enough
//! that four NUTS chains sample it without divergences. (Eight schools, with far less data, is the classic case
//! where the non-centred form is needed.)
//!
//! Data: `data/radon.csv` holds the county index (0 to 84, alphabetical), county name, floor and log radon of the
//! 919 Minnesota houses. It is taken from `examples/data/radon.csv` in the PyMC examples repository
//! (<https://github.com/pymc-devs/pymc-examples>, MIT licence, Copyright (c) 2020 PyMC), which is the Minnesota
//! subset of the EPA's National Radon Survey distributed with Gelman & Hill's book (`srrs2.dat` and `cty.dat`).
//! Log radon is ln(activity), with an activity of 0 taken as 0.1, as in the book.
//!
//! Run with `cargo run --release --example radon`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::mcmc::chains::run_chains;
use keanu::prelude::*;

/// The number of counties.
pub const COUNTIES: usize = 85;

/// The number of chains.
pub const CHAINS: usize = 4;

/// The houses in the bundled CSV.
#[derive(Debug, Clone, Default)]
pub struct Houses {
    /// Each house's county index (0 to 84).
    pub county: Vec<i64>,
    /// 0 for a basement measurement, 1 for the ground floor.
    pub floor: Vec<f64>,
    /// ln of the radon activity (pCi/L).
    pub log_radon: Vec<f64>,
}

/// The houses from the bundled CSV.
pub fn data() -> Houses {
    let mut houses = Houses::default();
    for line in include_str!("data/radon.csv").lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        let [county, _name, floor, log_radon] = fields[..] else {
            panic!("expected county_code,county,floor,log_radon: {line}");
        };
        houses.county.push(county.trim().parse().unwrap());
        houses.floor.push(floor.trim().parse().unwrap());
        houses.log_radon.push(log_radon.trim().parse().unwrap());
    }
    houses
}

/// The posterior draws and where to find each parameter in them.
#[derive(Debug)]
pub struct Radon {
    /// The draws of every chain.
    pub samples: ChainSamples,
    /// The number of divergent transitions after adaptation, over all chains.
    pub divergences: usize,
    /// The mean county intercept.
    pub mu_a: NodeId,
    /// The spread of the county intercepts.
    pub sigma_a: NodeId,
    /// The ground floor effect.
    pub b: NodeId,
    /// The residual spread of log radon within a county.
    pub sigma_y: NodeId,
    /// The 85 county intercepts.
    pub a: NodeId,
}

/// Samples the model with `CHAINS` NUTS chains of `draws` draws each, after as many adaptation steps.
pub fn run(seed: u64, draws: usize) -> keanu::Result<Radon> {
    let houses = data();
    let n = houses.log_radon.len();

    let b = GraphBuilder::new();
    let mu_a = b.gaussian(0.0, 10.0);
    let sigma_a = b.exponential(1.0);
    let beta = b.gaussian(0.0, 10.0);
    let sigma_y = b.exponential(1.0);
    let a = b.gaussian_with_shape(&[COUNTIES], mu_a, sigma_a);
    b.set_value(mu_a, DoubleTensor::scalar(0.0));
    b.set_value(sigma_a, DoubleTensor::scalar(1.0));
    b.set_value(beta, DoubleTensor::scalar(0.0));
    b.set_value(sigma_y, DoubleTensor::scalar(1.0));
    b.set_value(a, DoubleTensor::vector(vec![0.0; COUNTIES]));
    let county = b.int_constant(IntTensor::vector(houses.county));
    let floor = b.constant(DoubleTensor::vector(houses.floor));
    // a[county[i]] for each house i
    let mean = a.gather(0, county) + beta * floor;
    let log_radon = b.gaussian_with_shape(&[n], mean, sigma_y);
    b.observe(log_radon, DoubleTensor::vector(houses.log_radon));
    let (mu_a, sigma_a, beta, sigma_y, a) =
        (mu_a.id(), sigma_a.id(), beta.id(), sigma_y.id(), a.id());

    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let adapt = draws;
    let nuts = Nuts::default().with_adapt_count(adapt);
    let chains = run_chains(CHAINS, seed, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &[mu_a, sigma_a, beta, sigma_y, a],
            adapt + draws,
            SamplingOptions::default().with_drop(adapt),
            rng,
        )
    })?;
    let divergences = chains.iter().map(|(_, s)| s.divergence_count()).sum();
    let samples = ChainSamples::new(chains.into_iter().map(|(samples, _)| samples).collect())?;
    Ok(Radon {
        samples,
        divergences,
        mu_a,
        sigma_a,
        b: beta,
        sigma_y,
        a,
    })
}

/// The number of draws per chain (after as many adaptation steps).
pub const DEFAULT_SAMPLES: usize = 1000;

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let r = run(1, DEFAULT_SAMPLES)?;
    println!("{} divergences", r.divergences);
    for (name, node) in [
        ("mu_a", r.mu_a),
        ("sigma_a", r.sigma_a),
        ("b", r.b),
        ("sigma_y", r.sigma_y),
    ] {
        let s = r.samples.summarize(node, &[])?;
        println!(
            "{name:<8} mean {:>7.3}  sd {:.3}  R-hat {:.3}  bulk ESS {:.0}",
            s.mean, s.sd, s.r_hat, s.ess_bulk
        );
    }
    let summary = r.samples.summary()?;
    let worst = summary
        .rows()
        .iter()
        .map(|row| row.stats.r_hat)
        .fold(f64::NEG_INFINITY, f64::max);
    println!("largest R-hat over all parameters and county intercepts: {worst:.3}");
    Ok(())
}
