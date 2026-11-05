//! What proportion of students cheated, from a privacy-preserving survey?
//!
//! Each student flips a coin: heads, they answer truthfully; tails, they flip again and answer "yes" on heads.
//! From "Bayesian Methods for Hackers", chapter 2. Run with `cargo run --release --example cheating_students`.

use keanu::prelude::*;

/// A posterior mean and its Monte Carlo standard error.
#[derive(Debug, Clone, Copy)]
pub struct Estimate {
    /// The posterior mean.
    pub mean: f64,
    /// Its Monte Carlo standard error, from the chain's effective sample size.
    pub mcse: f64,
}

/// Models every student's coin flips explicitly with Bernoulli nodes. The model has discrete latents (the flips)
/// and a continuous one (the probability of cheating), so the default sampler is the compound sampler: Gibbs for
/// the flips and NUTS for the probability. Runs `draws` draws after 1000 adaptation steps.
pub fn run_with_bernoulli(
    students: usize,
    yes_answers: f64,
    seed: u64,
    draws: usize,
) -> keanu::Result<Estimate> {
    let b = GraphBuilder::new();
    let probability_of_cheating = b.uniform(0.0, 1.0);
    let student_cheated = b.bernoulli_with_shape(&[students], probability_of_cheating);
    let answer_is_true = b.bernoulli_with_shape(&[students], 0.5);
    let random_answer = b.bernoulli_with_shape(&[students], 0.5);
    let (one, zero) = (b.scalar(1.0), b.scalar(0.0));
    let answer = b.where_(
        answer_is_true,
        b.where_(student_cheated, one, zero),
        b.where_(random_answer, one, zero),
    );
    let answer_total = b.gaussian(answer.sum(), 1.0);
    b.observe(answer_total, DoubleTensor::scalar(yes_answers));
    let probability_of_cheating = probability_of_cheating.id();

    let (graph, mut state) = b.build()?;
    let mut rng = KeanuRng::seed_from_u64(seed);
    let model = Model::new(graph, &state)?;
    model.probe_for_non_zero_probability(&mut state, 100, &mut rng)?;

    let warmup = 1000;
    let samples = Sampler::for_model(&model, &state).sample(
        &model,
        &mut state,
        &[probability_of_cheating],
        warmup + draws,
        SamplingOptions::default().with_drop(warmup),
        &mut rng,
    )?;
    let p = samples.f64(probability_of_cheating).unwrap();
    let mean = p.mean()?.to_scalar().unwrap();
    let sd = p.variance()?.to_scalar().unwrap().sqrt();
    Ok(Estimate {
        mean,
        mcse: sd / p.effective_sample_size(&[])?.sqrt(),
    })
}

/// The same model, collapsed: P(yes) = 0.5 p + 0.25, and the number of yes answers is Binomial.
pub fn run_with_binomial(
    students: i64,
    yes_answers: i64,
    seed: u64,
    sample_count: usize,
) -> keanu::Result<f64> {
    let b = GraphBuilder::new();
    let probability_of_cheating = b.uniform(0.0, 1.0);
    let p_yes = probability_of_cheating * 0.5 + 0.25;
    let answer_total = b.binomial(p_yes, students);
    b.observe(answer_total, IntTensor::scalar(yes_answers));
    let probability_of_cheating = probability_of_cheating.id();

    let (graph, mut state) = b.build()?;
    let mut rng = KeanuRng::seed_from_u64(seed);
    let model = Model::new(graph, &state)?;
    model.probe_for_non_zero_probability(&mut state, 100, &mut rng)?;

    let samples = MetropolisHastings::default().sample_with(
        &model,
        &mut state,
        &[probability_of_cheating],
        sample_count,
        SamplingOptions::default()
            .with_drop(sample_count / 10)
            .with_down_sample_interval(model.latents().len()),
        &mut rng,
    )?;
    Ok(samples
        .f64(probability_of_cheating)
        .unwrap()
        .mean()?
        .to_scalar()
        .unwrap())
}

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let estimate = run_with_bernoulli(100, 35.0, 1, 2000)?;
    println!(
        "Bernoulli model: {:.3} (± {:.3})",
        estimate.mean, estimate.mcse
    );
    println!("Binomial model: {}", run_with_binomial(100, 35, 1, 100)?);
    Ok(())
}
