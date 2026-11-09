//! Hierarchical two-parameter logistic item response with sparse observations.
//!
//! A student's ability and an item's difficulty are on the same scale. The item's positive discrimination
//! controls how quickly the probability of a correct response increases with ability. A standard normal prior
//! fixes the ability scale. Difficulty and log discrimination have non-centred hierarchical priors.
//!
//! Run with `cargo run --release --example item_response`.

use keanu::prelude::*;

const STUDENTS: usize = 100;
const ITEMS: usize = 8;

pub fn run(draws: usize) -> keanu::Result<(f64, f64)> {
    let mut rng = KeanuRng::seed_from_u64(74);
    let abilities: Vec<_> = (0..STUDENTS).map(|_| rng.next_gaussian()).collect();
    let difficulty = [-1.8, -1.2, -0.6, -0.2, 0.3, 0.7, 1.2, 1.8];
    let discrimination = [1.4, 0.8, 1.0, 1.3, 0.8, 1.0, 1.4, 1.2];
    let (mut students, mut items, mut responses) = (Vec::new(), Vec::new(), Vec::new());
    let mut held_out = Vec::new();
    for (student, &ability) in abilities.iter().enumerate() {
        for item in 0..ITEMS {
            let p = 1.0 / (1.0 + (-(ability - difficulty[item]) * discrimination[item]).exp());
            let correct = rng.next_f64() < p;
            if (student + item) % ITEMS == 0 {
                held_out.push((student, item, correct));
            } else {
                students.push(student as i64);
                items.push(item as i64);
                responses.push(correct);
            }
        }
    }

    let b = GraphBuilder::new();
    let ability = b.gaussian_with_shape(&[STUDENTS], 0.0, 1.0);
    let difficulty_mean = b.gaussian(0.0, 1.0);
    let difficulty_scale = b.log_normal(0.0, 0.5);
    let difficulty_offset = b.gaussian_with_shape(&[ITEMS], 0.0, 1.0);
    let item_difficulty = difficulty_mean + difficulty_scale * difficulty_offset;
    let log_discrimination_mean = b.gaussian(0.0, 0.4);
    let log_discrimination_scale = b.log_normal(0.3_f64.ln(), 0.5);
    let discrimination_offset = b.gaussian_with_shape(&[ITEMS], 0.0, 1.0);
    let item_discrimination =
        (log_discrimination_mean + log_discrimination_scale * discrimination_offset).exp();
    let student_index = b.int_constant(IntTensor::vector(students));
    let item_index = b.int_constant(IntTensor::vector(items));
    let logits = item_discrimination.gather(0, item_index)
        * (ability.gather(0, student_index) - item_difficulty.gather(0, item_index));
    b.observe(b.bernoulli_logit(logits), BoolTensor::vector(responses));

    let held_students = b.int_constant(IntTensor::vector(
        held_out
            .iter()
            .map(|&(student, _, _)| student as i64)
            .collect(),
    ));
    let held_items = b.int_constant(IntTensor::vector(
        held_out.iter().map(|&(_, item, _)| item as i64).collect(),
    ));
    let held_logits = item_discrimination.gather(0, held_items)
        * (ability.gather(0, held_students) - item_difficulty.gather(0, held_items));
    let prediction = held_logits.sigmoid().id();
    let difficulty_id = item_difficulty.id();
    b.set_value(ability, DoubleTensor::zeros(&[STUDENTS]));
    b.set_value(difficulty_mean, DoubleTensor::scalar(0.0));
    b.set_value(difficulty_scale, DoubleTensor::scalar(1.0));
    b.set_value(difficulty_offset, DoubleTensor::zeros(&[ITEMS]));
    b.set_value(log_discrimination_mean, DoubleTensor::scalar(0.0));
    b.set_value(log_discrimination_scale, DoubleTensor::scalar(0.3));
    b.set_value(discrimination_offset, DoubleTensor::zeros(&[ITEMS]));
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let (samples, _) = Nuts::default()
        .with_adapt_count(draws)
        .sample_with_statistics(
            &model,
            &mut state,
            &[prediction, difficulty_id],
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            &mut rng,
        )?;
    let predictions = samples.f64(prediction).unwrap().mean()?;
    let brier = predictions
        .as_slice()
        .iter()
        .zip(&held_out)
        .map(|(&p, &(_, _, correct))| (p - f64::from(correct)).powi(2))
        .sum::<f64>()
        / held_out.len() as f64;
    let difficulties = samples.f64(difficulty_id).unwrap().mean()?;
    let item_rmse = (difficulties
        .as_slice()
        .iter()
        .zip(difficulty)
        .map(|(&a, b)| (a - b).powi(2))
        .sum::<f64>()
        / ITEMS as f64)
        .sqrt();
    Ok((brier, item_rmse))
}

fn main() -> keanu::Result<()> {
    let (brier, item_rmse) = run(600)?;
    println!("Held-out Brier score {brier:.3}. Item difficulty RMSE {item_rmse:.3}.");
    Ok(())
}
