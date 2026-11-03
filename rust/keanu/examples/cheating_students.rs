//! What proportion of students cheated, from a privacy-preserving survey?
//!
//! Each student flips a coin: heads, they answer truthfully; tails, they flip again and answer "yes" on heads.
//! From "Bayesian Methods for Hackers", chapter 2. Run with `cargo run --example cheating_students`.

use keanu::prelude::*;

/// Models every student's coin flips explicitly with Bernoulli nodes.
pub fn run_with_bernoulli(
    students: usize,
    yes_answers: f64,
    seed: u64,
    sample_count: usize,
) -> keanu::Result<f64> {
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

    let samples = MetropolisHastings::default().sample_with(
        &model,
        &mut state,
        &[probability_of_cheating],
        sample_count,
        SamplingOptions {
            drop: sample_count / 2,
            down_sample_interval: model.latents().len(),
        },
        &mut rng,
    )?;
    Ok(samples
        .f64(probability_of_cheating)
        .unwrap()
        .mean()?
        .to_scalar()
        .unwrap())
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
        SamplingOptions {
            drop: sample_count / 10,
            down_sample_interval: model.latents().len(),
        },
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
    println!(
        "Bernoulli model: {}",
        run_with_bernoulli(100, 35.0, 1, 2000)?
    );
    println!("Binomial model: {}", run_with_binomial(100, 35, 1, 100)?);
    Ok(())
}
