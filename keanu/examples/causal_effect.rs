//! Does a treatment work, when the people who take it differ from those who don't?
//!
//! A simulated observational study with a confounder: each of [`UNITS`] people has a severity score Z, sicker people
//! are more likely to be treated (X), and both the treatment and the severity raise the outcome Y:
//!
//! Z ~ N(μ_z, σ_z),  X ~ Bernoulli(logistic(α₀ + α₁ Z)),  Y ~ N(β₀ + τ X + β_z Z, σ_y).
//!
//! The treatment effect is τ: setting X to 1 rather than 0 raises Y by τ on average, whatever Z is. But comparing
//! the treated with the untreated, E[Y | X = 1] − E[Y | X = 0], also compares sicker people with healthier ones, so it
//! overstates the effect (the confounder Z opens a back-door path X ← Z → Y).
//!
//! The model is the causal graph above, fitted by NUTS. Two kinds of question are then asked of the posterior:
//!
//! - **Conditioning on X:** the posterior predictive of Y among the units that were treated, and among those that
//!   weren't. These are the confounded means E[Y | X = x].
//! - **Intervening on X:** [`intervene`] rewrites the graph with X set to x for everyone, do(X = x). X becomes a
//!   constant, its distribution and its causes' only path to the rest of the model (α₀, α₁ and the logits) are
//!   dropped, and Z keeps its distribution. The posterior predictive of Y on the rewritten graph, given the
//!   original posterior draws (`PredictiveSampler::interventional`), is the interventional distribution
//!   p(Y | do(X = x)), with mean β₀ + τx + β_z E[Z]. The average treatment effect is the difference of the two
//!   interventions' means, draw by draw.
//!
//! The data are simulated from known values, so both answers can be checked against the truth. Run with
//! `cargo run --release --example causal_effect`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::mcmc::chains::sample_chains;
use keanu::prelude::*;

/// The number of people in the study.
pub const UNITS: usize = 400;
/// The confounder's mean.
pub const TRUE_MU_Z: f64 = 1.0;
/// The confounder's standard deviation.
pub const TRUE_SIGMA_Z: f64 = 1.0;
/// The treatment's logistic regression on the confounder: intercept and slope.
pub const TRUE_ALPHA: [f64; 2] = [-1.0, 2.0];
/// The outcome's intercept.
pub const TRUE_BETA_0: f64 = 1.0;
/// The treatment effect τ.
pub const TRUE_EFFECT: f64 = 2.0;
/// The outcome's slope on the confounder.
pub const TRUE_BETA_Z: f64 = 3.0;
/// The outcome's noise.
pub const TRUE_SIGMA_Y: f64 = 1.0;
/// The seed the example's data are simulated with.
pub const DATA_SEED: u64 = 1;

/// A simulated study: each person's confounder, treatment and outcome.
#[derive(Debug, Clone)]
pub struct Study {
    pub z: Vec<f64>,
    pub x: Vec<bool>,
    pub y: Vec<f64>,
}

fn logistic(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

/// Simulates the study from the true values.
pub fn data(seed: u64) -> Study {
    let mut rng = KeanuRng::seed_from_u64(seed);
    let (mut z, mut x, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..UNITS {
        let zi = TRUE_MU_Z + TRUE_SIGMA_Z * rng.next_gaussian();
        let xi = rng.next_f64() < logistic(TRUE_ALPHA[0] + TRUE_ALPHA[1] * zi);
        let mean = TRUE_BETA_0 + TRUE_EFFECT * f64::from(u8::from(xi)) + TRUE_BETA_Z * zi;
        z.push(zi);
        x.push(xi);
        y.push(mean + TRUE_SIGMA_Y * rng.next_gaussian());
    }
    Study { z, x, y }
}

/// The true mean of Y under do(X = x): β₀ + τx + β_z E[Z].
pub fn interventional_mean(x: bool) -> f64 {
    TRUE_BETA_0 + TRUE_EFFECT * f64::from(u8::from(x)) + TRUE_BETA_Z * TRUE_MU_Z
}

/// The true mean of Y among the units with X = x: β₀ + τx + β_z E[Z | X = x], with E[Z | X = x] computed by
/// quadrature over Z's density times the probability of the treatment given Z.
pub fn conditional_mean(x: bool) -> f64 {
    let (steps, width) = (20_000, 10.0 * TRUE_SIGMA_Z);
    let (mut mass, mut first) = (0.0, 0.0);
    for i in 0..=steps {
        let z = TRUE_MU_Z - width + 2.0 * width * i as f64 / steps as f64;
        let u = (z - TRUE_MU_Z) / TRUE_SIGMA_Z;
        let p = logistic(TRUE_ALPHA[0] + TRUE_ALPHA[1] * z);
        let weight = (-0.5 * u * u).exp() * if x { p } else { 1.0 - p };
        mass += weight;
        first += weight * z;
    }
    TRUE_BETA_0 + TRUE_EFFECT * f64::from(u8::from(x)) + TRUE_BETA_Z * first / mass
}

/// A fit and the answers to its causal questions.
#[derive(Debug)]
pub struct Fit {
    pub chains: ChainSamples,
    /// The treatment effect τ and the other latents of the fitted model.
    pub effect: NodeId,
    pub latents: Vec<NodeId>,
    /// The posterior predictive mean of Y among the untreated and the treated: E[Y | X = 0] and E[Y | X = 1].
    pub conditional: [f64; 2],
    /// The posterior predictive mean of Y under do(X = 0) and do(X = 1).
    pub interventional: [f64; 2],
    /// The average treatment effect at each posterior draw, every chain's draws in turn: the mean of Y over the
    /// units under do(X = 1) less its mean under do(X = 0).
    pub average_effect: Vec<f64>,
    /// The labels of the nodes the intervention dropped, because they influenced the model only through X.
    pub dropped: Vec<String>,
}

/// Fits the causal model by NUTS (`chains` chains of `draws` draws after as many warm-up steps) and answers the
/// conditional and interventional questions by posterior predictive sampling.
pub fn run(study: &Study, chains: usize, draws: usize, seed: u64) -> keanu::Result<Fit> {
    let b = GraphBuilder::new();
    let mu_z = b.gaussian(0.0, 5.0).label("mu_z");
    let sigma_z = b.half_gaussian(2.0).label("sigma_z");
    let z = b.gaussian_with_shape(&[UNITS], mu_z, sigma_z).label("z");
    let alpha_0 = b.gaussian(0.0, 2.5).label("alpha_0");
    let alpha_1 = b.gaussian(0.0, 2.5).label("alpha_1");
    let logit = (alpha_0 + alpha_1 * z).label("treatment_logit");
    let x = b.bernoulli_logit_with_shape(&[UNITS], logit).label("x");
    let beta_0 = b.gaussian(0.0, 5.0).label("beta_0");
    let effect = b.gaussian(0.0, 5.0).label("effect");
    let beta_z = b.gaussian(0.0, 5.0).label("beta_z");
    let sigma_y = b.half_gaussian(2.0).label("sigma_y");
    let y = b
        .gaussian_with_shape(
            &[UNITS],
            beta_0 + effect * x.to_double_mask() + beta_z * z,
            sigma_y,
        )
        .label("y");
    b.observe(z, DoubleTensor::vector(study.z.clone()));
    b.observe(x, BoolTensor::vector(study.x.clone()));
    b.observe(y, DoubleTensor::vector(study.y.clone()));
    for (node, value) in [(sigma_z, 1.0), (sigma_y, 1.0)] {
        b.set_value(node, DoubleTensor::scalar(value));
    }
    for node in [mu_z, alpha_0, alpha_1, beta_0, effect, beta_z] {
        b.set_value(node, DoubleTensor::scalar(0.0));
    }
    let latents: Vec<NodeId> = [
        mu_z, sigma_z, alpha_0, alpha_1, beta_0, effect, beta_z, sigma_y,
    ]
    .iter()
    .map(|v| v.id())
    .collect();
    let (effect, x, y) = (effect.id(), x.id(), y.id());
    let (graph, state) = b.build()?;
    let model = Model::new(graph.clone(), &state)?;

    let nuts = Nuts::default().with_adapt_count(draws);
    let chains = sample_chains(chains, seed, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &latents,
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            rng,
        )
        .map(|(samples, _)| samples)
    })?;

    let mut rng = KeanuRng::seed_from_u64(seed);
    let predictive = PredictiveSampler::new();
    // conditioning: Y's posterior predictive among the units with each treatment
    let mut conditional = [(0.0, 0usize); 2];
    for chain in chains.chains() {
        let draws = predictive.posterior(&model, &state, chain, &[y], &mut rng)?;
        for draw in draws.f64(y).expect("Y is drawn").as_slice() {
            for (&value, &treated) in draw.as_slice().iter().zip(&study.x) {
                let (total, count) = &mut conditional[usize::from(treated)];
                *total += value;
                *count += 1;
            }
        }
    }

    // intervening: Y's posterior predictive with X set for everyone, and the effect draw by draw
    let mut means = Vec::with_capacity(2);
    let mut dropped = Vec::new();
    for treated in [false, true] {
        let (graph_do, state_do, intervened) = intervene(
            &graph,
            &state,
            &[(x, BoolTensor::full(&[UNITS], treated).into())],
        )?;
        dropped = intervened
            .dropped()
            .into_iter()
            .filter_map(|id| graph[id].label().map(str::to_owned))
            .collect();
        let model_do = Model::new(graph_do, &state_do)?;
        let y_do = intervened.node(y).expect("Y is kept");
        let mut unit_means = Vec::new();
        for chain in chains.chains() {
            let draws = predictive.interventional(
                &model_do,
                &state_do,
                &intervened,
                chain,
                &[y],
                &mut rng,
            )?;
            for draw in draws.f64(y_do).expect("Y is drawn").as_slice() {
                unit_means.push(draw.as_slice().iter().sum::<f64>() / UNITS as f64);
            }
        }
        means.push(unit_means);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let average_effect = means[1].iter().zip(&means[0]).map(|(a, b)| a - b).collect();
    Ok(Fit {
        chains,
        effect,
        latents,
        conditional: conditional.map(|(total, count)| total / count as f64),
        interventional: [mean(&means[0]), mean(&means[1])],
        average_effect,
        dropped,
    })
}

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let study = data(DATA_SEED);
    let treated = study.x.iter().filter(|&&x| x).count();
    println!("{UNITS} people, {treated} treated");
    let fit = run(&study, 4, 1000, 1)?;
    let s = fit.chains.summarize(fit.effect, &[])?;
    println!(
        "treatment effect τ: mean {:.3}, 90% interval [{:.3}, {:.3}] (truth {TRUE_EFFECT})",
        s.mean, s.q5, s.q95
    );
    println!("dropped by the intervention: {}", fit.dropped.join(", "));
    for (x, name) in [(false, "untreated"), (true, "treated")] {
        let i = usize::from(x);
        println!(
            "E[Y | X = {i}] ({name}): {:.3} (truth {:.3});  E[Y | do(X = {i})]: {:.3} (truth {:.3})",
            fit.conditional[i],
            conditional_mean(x),
            fit.interventional[i],
            interventional_mean(x)
        );
    }
    let mut effects = fit.average_effect.clone();
    effects.sort_by(f64::total_cmp);
    let quantile = |q: f64| effects[((effects.len() - 1) as f64 * q).round() as usize];
    println!(
        "comparing the treated with the untreated: {:.3}; average treatment effect: {:.3}, 90% interval \
         [{:.3}, {:.3}] (truth {TRUE_EFFECT})",
        fit.conditional[1] - fit.conditional[0],
        fit.interventional[1] - fit.interventional[0],
        quantile(0.05),
        quantile(0.95)
    );
    Ok(())
}
