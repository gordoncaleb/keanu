//! The stochastic SIR epidemic, whose likelihood is intractable, fitted by approximate Bayesian computation.
//!
//! The [ODE SIR example](sir.rs) follows the mean of an epidemic. In a small population chance matters: who infects
//! whom, and when, is random, and the same β and γ give visibly different outbreaks. The stochastic SIR model is
//! a continuous-time Markov chain over the counts (S, I, R) of N people: each infected person infects each
//! susceptible one at rate β / N and recovers at rate γ, so
//!
//! ```text
//! (S, I) → (S − 1, I + 1)   at rate β S I / N        (I, R) → (I − 1, R + 1)   at rate γ I.
//! ```
//!
//! It is easy to simulate exactly with Gillespie's algorithm ([`gillespie`]: wait an exponential time with the total
//! rate, then pick an event in proportion to its rate), but the probability of the observed daily counts sums over
//! every path between them, which is intractable. So the counts are a [simulator
//! node](keanu::graph::GraphBuilder::simulator), and SMC-ABC ([`Smc::sample_abc`] with an [`Abc`] likelihood)
//! compares outbreaks simulated at each particle's β and γ with the observed one through four summary statistics
//! ([`statistics`]): the final size and the peak prevalence (as fractions of N), the day of the peak, and the
//! person-days of infection per person, Σₜ I(t) / N. The final size is set by R₀ = β / γ and the person-days by the
//! recovery time 1 / γ, so together they identify both rates. Each statistic has its own scale ε, about its
//! spread between outbreaks simulated at the same rates ([`EPSILON`]).
//!
//! ```text
//! β ~ Uniform(0.2, 3)     γ ~ Uniform(0.05, 1)     (S(t), I(t)) for t = 1, ..., 40 days ~ stochastic SIR(β, γ)
//! ```
//!
//! The data are an outbreak simulated with β = 1 and γ = 0.3 (R₀ ≈ 3.3) in 200 people, 3 of them infected at first
//! ([`observed`]); the test checks that the ABC posterior covers both rates.
//!
//! Run with `cargo run --release --example abc_stochastic_sir`.

use keanu::algorithms::ChainSamples;
use keanu::prelude::*;

/// The population.
pub const POPULATION: i64 = 200;

/// The number infected at time 0.
pub const INITIALLY_INFECTED: i64 = 3;

/// The number of days observed.
pub const DAYS: usize = 40;

/// The rates the observed outbreak is simulated with: β = 1, γ = 0.3.
pub const TRUTH: (f64, f64) = (1.0, 0.3);

/// The scale ε of each summary statistic: final size, peak prevalence, peak day, person-days per person.
pub const EPSILON: [f64; 4] = [0.03, 0.03, 1.5, 0.3];

/// Simulates the epidemic with Gillespie's algorithm, returning the numbers susceptible and infected at the end of
/// each of days 1 to `DAYS`, as rows `[2, DAYS]`.
pub fn gillespie(beta: f64, gamma: f64, rng: &mut KeanuRng) -> Vec<i64> {
    let n = POPULATION as f64;
    let (mut s, mut i) = (POPULATION - INITIALLY_INFECTED, INITIALLY_INFECTED);
    let mut counts = vec![0; 2 * DAYS];
    let mut t = 0.0;
    let mut day = 0;
    while day < DAYS {
        let infection = beta * s as f64 * i as f64 / n;
        let recovery = gamma * i as f64;
        let total = infection + recovery;
        // the time of the next event, or never once nobody is infected
        t += if total > 0.0 {
            -(1.0 - rng.next_f64()).ln() / total
        } else {
            f64::INFINITY
        };
        // record the days that end before it
        while day < DAYS && (day + 1) as f64 <= t {
            counts[day] = s;
            counts[DAYS + day] = i;
            day += 1;
        }
        if total > 0.0 {
            if rng.next_f64() * total < infection {
                s -= 1;
                i += 1;
            } else {
                i -= 1;
            }
        }
    }
    counts
}

/// The summary statistics of an outbreak's daily counts (`[2, DAYS]`, as [`gillespie`] returns): the final size and
/// the peak prevalence as fractions of the population, the day of the peak, and the person-days of infection per
/// person.
pub fn statistics(counts: &[i64]) -> Vec<f64> {
    let n = POPULATION as f64;
    let (susceptible, infected) = counts.split_at(DAYS);
    let final_size = (POPULATION - susceptible[DAYS - 1]) as f64 / n;
    let (peak_day, &peak) = infected
        .iter()
        .enumerate()
        .max_by_key(|&(_, &i)| i)
        .expect("at least one day");
    let person_days = infected.iter().sum::<i64>() as f64 / n;
    vec![
        final_size,
        peak as f64 / n,
        (peak_day + 1) as f64,
        person_days,
    ]
}

/// The observed outbreak: simulated with [`TRUTH`] from a generator seeded with `seed`.
pub fn observed(seed: u64) -> Vec<i64> {
    gillespie(TRUTH.0, TRUTH.1, &mut KeanuRng::seed_from_u64(seed))
}

/// The model's nodes.
#[derive(Debug, Clone, Copy)]
pub struct Nodes {
    /// The transmission rate β (per day).
    pub beta: NodeId,
    /// The recovery rate γ (per day).
    pub gamma: NodeId,
    /// The basic reproduction number R₀ = β / γ.
    pub r0: NodeId,
    /// The daily counts, `[2, DAYS]`: susceptible, then infected.
    pub counts: NodeId,
}

/// The model of the outbreak `counts` (`[2, DAYS]`).
pub fn model(counts: &[i64]) -> keanu::Result<(Model, State, Nodes)> {
    let b = GraphBuilder::new();
    let beta = b.uniform(0.2, 3.0).label("beta");
    let gamma = b.uniform(0.05, 1.0).label("gamma");
    // ANCHOR: abc_simulator
    let outbreak: IntVar = b.simulator(
        &[beta.id(), gamma.id()],
        &[2, DAYS],
        |parameters: &[&Value], rng: &mut KeanuRng| {
            let rate = |k: usize| -> Result<f64, String> {
                Ok(parameters[k]
                    .as_f64()
                    .ok_or("the rates are f64")?
                    .as_slice()[0])
            };
            let counts = gillespie(rate(0)?, rate(1)?, rng);
            Ok(Value::I64(
                IntTensor::new(&[2, DAYS], counts).map_err(|e| e.to_string())?,
            ))
        },
    );
    b.observe(outbreak, IntTensor::new(&[2, DAYS], counts.to_vec())?);
    // ANCHOR_END: abc_simulator
    let r0 = (beta / gamma).label("r0");
    let nodes = Nodes {
        beta: beta.id(),
        gamma: gamma.id(),
        r0: r0.id(),
        counts: outbreak.id(),
    };
    let (graph, state) = b.build()?;
    Ok((Model::new(graph, &state)?, state, nodes))
}

/// The ABC likelihood: the Gaussian kernel on [`statistics`] with the scales [`EPSILON`].
pub fn abc() -> Abc {
    // ANCHOR: abc_likelihood
    let summary = SummaryStatistics::custom(|value: &Value| {
        Ok(statistics(
            value.as_i64().ok_or("counts are i64")?.as_slice(),
        ))
    });
    Abc::new(1.0)
        .with_epsilons(EPSILON.to_vec())
        .with_statistics(summary)
    // ANCHOR_END: abc_likelihood
}

/// The ABC posterior and where to find each quantity in it.
#[derive(Debug)]
pub struct Fit {
    /// The particles of every run.
    pub samples: ChainSamples,
    /// The model's nodes.
    pub nodes: Nodes,
}

/// Fits the model to `counts` by SMC-ABC with 4 runs of `particles` particles.
pub fn fit(counts: &[i64], particles: usize, seed: u64) -> keanu::Result<Fit> {
    let (model, state, nodes) = model(counts)?;
    // ANCHOR: abc_sample
    let result = Smc::default().with_particles(particles).sample_abc(
        &model,
        &state,
        &[nodes.beta, nodes.gamma, nodes.r0],
        &abc(),
        seed,
    )?;
    // ANCHOR_END: abc_sample
    Ok(Fit {
        samples: result.samples,
        nodes,
    })
}

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let counts = observed(1);
    let observed_statistics = statistics(&counts);
    println!(
        "observed: final size {:.2}, peak {:.2} on day {}, {:.2} person-days per person",
        observed_statistics[0],
        observed_statistics[1],
        observed_statistics[2],
        observed_statistics[3]
    );
    let fit = fit(&counts, 2000, 1)?;
    for (name, node, truth) in [
        ("beta", fit.nodes.beta, TRUTH.0),
        ("gamma", fit.nodes.gamma, TRUTH.1),
        ("R0", fit.nodes.r0, TRUTH.0 / TRUTH.1),
    ] {
        let s = fit.samples.summarize(node, &[])?;
        println!(
            "{name:<6} truth {truth:>5.2}  mean {:>5.2}  90% interval [{:.2}, {:.2}]  R-hat {:.3}",
            s.mean, s.q5, s.q95, s.r_hat
        );
    }
    Ok(())
}
