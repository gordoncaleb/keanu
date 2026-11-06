//! Minibatch ADVI: the weighted batch log likelihood against the full data's, recovery of a large regression's
//! coefficients, alignment of covariates with observations, and invalid configurations.

use keanu_algorithms::AlgorithmError;
use keanu_algorithms::vi::{Advi, Minibatch, MinibatchGroup};
use keanu_distributions::KeanuRng;
use keanu_graph::{GraphBuilder, Model, NodeId, Parameterisation, State, node_log_prob};
use keanu_tensor::DoubleTensor;
use std::time::Instant;

fn rng(seed: u64) -> KeanuRng {
    KeanuRng::seed_from_u64(seed)
}

/// The nodes of a regression model built for `rows` rows.
#[derive(Clone, Copy)]
struct Regression {
    intercept: NodeId,
    slope: NodeId,
    x: NodeId,
    mu: NodeId,
    y: NodeId,
    factor: NodeId,
}

/// intercept, slope ~ N(0, 10); mu = intercept + slope · x; y ~ N(mu, 0.7) observed; and a factor
/// 0.1 · Σ slope · x (a term summed over the rows). `x` and `y` hold `x` and `y` (of `rows` rows).
fn regression(x: &DoubleTensor, y: &DoubleTensor) -> (Model, State, Regression) {
    let b = GraphBuilder::new();
    let intercept = b.gaussian(0.0, 10.0);
    let slope = b.gaussian(0.0, 10.0);
    b.set_value(intercept, DoubleTensor::scalar(0.3));
    b.set_value(slope, DoubleTensor::scalar(-0.8));
    let xs = b.constant(x.clone());
    let mu = intercept + slope * xs;
    let ys = b.gaussian(mu, 0.7);
    b.observe(ys, y.clone());
    let factor = b.factor((slope * xs).sum() * 0.1);
    let ids = Regression {
        intercept: intercept.id(),
        slope: slope.id(),
        x: xs.id(),
        mu: mu.id(),
        y: ys.id(),
        factor: factor.id(),
    };
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &graph.ids().collect::<Vec<_>>(), &mut rng(0))
        .unwrap();
    let model = Model::new(graph, &state).unwrap();
    (model, state, ids)
}

fn data(n: usize, seed: u64) -> (DoubleTensor, DoubleTensor) {
    let mut r = rng(seed);
    let x: Vec<f64> = (0..n).map(|_| r.next_gaussian()).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|x| 1.0 + 2.0 * x + 0.7 * r.next_gaussian())
        .collect();
    (DoubleTensor::vector(x), DoubleTensor::vector(y))
}

/// The weighted log likelihood (observations and factor) of the current batch: the weighted log probability minus
/// the latents' (unweighted) prior terms.
fn weighted_log_likelihood(
    param: &Parameterisation<'_>,
    state: &mut State,
    ids: Regression,
) -> f64 {
    let z = param.from_state(state).unwrap();
    let joint = param.log_prob(&z, false, state).unwrap();
    let graph = param.model().graph();
    joint
        - node_log_prob(graph, state, ids.intercept).unwrap()
        - node_log_prob(graph, state, ids.slope).unwrap()
}

fn group(b: usize, ids: Regression, x: &DoubleTensor, y: &DoubleTensor) -> MinibatchGroup {
    MinibatchGroup::new(b)
        .observed(ids.y, y.clone())
        .covariate(ids.x, x.clone())
        .factor(ids.factor)
}

#[test]
fn over_an_epoch_the_weighted_batches_average_to_the_full_log_likelihood() {
    let (n, b) = (60, 12);
    let (x, y) = data(n, 1);
    let (full, full_state, _) = regression(&x, &y);
    let full_log_likelihood = full.log_likelihood(&full_state).unwrap();

    let (model, mut state, ids) = regression(&x.slice(":12").unwrap(), &y.slice(":12").unwrap());
    let minibatch = Minibatch::from(group(b, ids, &x, &y));
    let mut batches = minibatch.batches(&model, &state).unwrap();
    assert_eq!(batches.weights(), &[(ids.y, 5.0), (ids.factor, 5.0)]);
    assert_eq!(batches.batches_per_epoch(0), 5);
    let param = Parameterisation::new(&model, &state)
        .unwrap()
        .with_weights(batches.weights().iter().copied())
        .unwrap();
    let mut r = rng(2);
    for epoch in 0..3 {
        let mut seen = Vec::new();
        let mut sum = 0.0;
        for _ in 0..5 {
            batches.next(&mut state, &mut r).unwrap();
            seen.extend_from_slice(batches.rows(0));
            sum += weighted_log_likelihood(&param, &mut state, ids);
        }
        // every row once per epoch
        seen.sort_unstable();
        assert_eq!(seen, (0..n).collect::<Vec<_>>(), "epoch {epoch}");
        let average = sum / 5.0;
        assert!(
            (average - full_log_likelihood).abs() < 1e-9 * full_log_likelihood.abs(),
            "epoch {epoch}: {average} vs {full_log_likelihood}"
        );
    }
}

#[test]
fn when_the_batch_size_does_not_divide_the_rows_the_weighted_batches_are_unbiased() {
    let (n, b) = (50, 12);
    let (x, y) = data(n, 3);
    let (full, full_state, _) = regression(&x, &y);
    let full_log_likelihood = full.log_likelihood(&full_state).unwrap();

    let (model, mut state, ids) = regression(&x.slice(":12").unwrap(), &y.slice(":12").unwrap());
    let minibatch = Minibatch::from(group(b, ids, &x, &y));
    let mut batches = minibatch.batches(&model, &state).unwrap();
    let param = Parameterisation::new(&model, &state)
        .unwrap()
        .with_weights(batches.weights().iter().copied())
        .unwrap();
    let mut r = rng(4);
    let draws = 20_000;
    let mut values = Vec::with_capacity(draws);
    let mut epoch: Vec<usize> = Vec::new();
    for i in 0..draws {
        batches.next(&mut state, &mut r).unwrap();
        epoch.extend_from_slice(batches.rows(0));
        // the 4 batches of an epoch are disjoint (the 2 rows left over are skipped)
        if i % 4 == 3 {
            epoch.sort_unstable();
            epoch.dedup();
            assert_eq!(epoch.len(), 48);
            epoch.clear();
        }
        values.push(weighted_log_likelihood(&param, &mut state, ids));
    }
    let mean = values.iter().sum::<f64>() / draws as f64;
    let sd = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (draws as f64 - 1.0)).sqrt();
    let standard_error = sd / (draws as f64).sqrt();
    assert!(
        (mean - full_log_likelihood).abs() < 4.0 * standard_error,
        "{mean} vs {full_log_likelihood} (standard error {standard_error})"
    );
}

/// Whether, in the state, every row of the batch has y = 3·x and mu = intercept + slope · x.
fn aligned(state: &State, ids: Regression) -> bool {
    let x = state.f64(ids.x).unwrap().as_slice();
    let y = state.f64(ids.y).unwrap().as_slice();
    let mu = state.f64(ids.mu).unwrap().as_slice();
    let intercept = state.f64(ids.intercept).unwrap().as_slice()[0];
    let slope = state.f64(ids.slope).unwrap().as_slice()[0];
    let recomputed = x
        .iter()
        .zip(mu)
        .all(|(x, mu)| (mu - (intercept + slope * x)).abs() < 1e-12);
    assert!(recomputed, "mu was not recomputed from the batch's x");
    x.iter().zip(y).all(|(x, y)| *y == 3.0 * x)
}

#[test]
fn covariates_and_observations_stay_aligned() {
    let n = 2000;
    let x = DoubleTensor::vector((0..n).map(|i| (i as f64 * 0.37).sin()).collect());
    let y = &x * 3.0;
    // the same observations, misaligned with the covariates by a random permutation of the rows
    let mut shuffled = y.as_slice().to_vec();
    let mut shuffle = rng(10);
    for i in (1..n).rev() {
        shuffled.swap(i, shuffle.next_below(i as u64 + 1) as usize);
    }
    let shuffled = DoubleTensor::vector(shuffled);
    let (model, mut state, ids) = regression(&x.slice(":100").unwrap(), &y.slice(":100").unwrap());
    let mut r = rng(5);

    let good = Minibatch::from(
        MinibatchGroup::new(100)
            .observed(ids.y, y.clone())
            .covariate(ids.x, x.clone())
            .factor(ids.factor),
    );
    let mut batches = good.batches(&model, &state).unwrap();
    for _ in 0..30 {
        batches.next(&mut state, &mut r).unwrap();
        assert!(aligned(&state, ids));
    }
    let bad = Minibatch::from(
        MinibatchGroup::new(100)
            .observed(ids.y, shuffled.clone())
            .covariate(ids.x, x.clone())
            .factor(ids.factor),
    );
    let mut batches = bad.batches(&model, &state).unwrap();
    batches.next(&mut state, &mut r).unwrap();
    assert!(!aligned(&state, ids));

    // fits: aligned data give the slope 3 (its posterior sd is about 0.02); misaligned data lose it. A fixed step
    // size scale keeps the test independent of the adaptation's grid search, whose 50-iteration comparisons can
    // pick η = 100 on this model, with or without minibatches.
    let fit = |minibatch: &Minibatch, state: &mut State| {
        let result = Advi::mean_field()
            .with_minibatch(minibatch.clone())
            .with_adapt(false)
            .with_max_iterations(3000)
            .fit(&model, state, &mut rng(6))
            .unwrap();
        let range = result.parameterisation().range(ids.slope).unwrap();
        result.mean()[range.start]
    };
    let slope = fit(&good, &mut state);
    assert!((slope - 3.0).abs() < 0.05, "{slope}");
    let slope = fit(&bad, &mut state);
    assert!(slope.abs() < 0.5, "{slope}");
}

#[test]
fn minibatch_advi_recovers_a_large_regression() {
    let n = 100_000;
    let batch = 1000;
    let mut r = rng(7);
    let x1: Vec<f64> = (0..n).map(|_| r.next_gaussian()).collect();
    let x2: Vec<f64> = (0..n).map(|_| r.next_gaussian()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 1.5 + 2.0 * x1[i] - 0.5 * x2[i] + r.next_gaussian())
        .collect();

    let b = GraphBuilder::new();
    let coefficients: Vec<_> = (0..3).map(|_| b.gaussian(0.0, 10.0)).collect();
    for c in &coefficients {
        b.set_value(*c, DoubleTensor::scalar(0.0));
    }
    let c1 = b.constant(DoubleTensor::zeros(&[batch]));
    let c2 = b.constant(DoubleTensor::zeros(&[batch]));
    let ys = b.gaussian(
        coefficients[0] + coefficients[1] * c1 + coefficients[2] * c2,
        1.0,
    );
    b.observe(ys, DoubleTensor::zeros(&[batch]));
    let ids: Vec<NodeId> = coefficients.iter().map(|c| c.id()).collect();
    let (c1, c2, ys) = (c1.id(), c2.id(), ys.id());
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    // the exact posterior: prior precision 0.01 on each coefficient, unit noise
    let mut precision = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]];
    let mut moment = [0.0; 3];
    for i in 0..n {
        let row = [1.0, x1[i], x2[i]];
        for a in 0..3 {
            moment[a] += row[a] * y[i];
            for c in 0..3 {
                precision[a][c] += row[a] * row[c];
            }
        }
    }
    let exact = solve3(precision, moment);
    let exact_sd: Vec<f64> = (0..3).map(|a| 1.0 / precision[a][a].sqrt()).collect();

    let minibatch = MinibatchGroup::new(batch)
        .observed(ys, DoubleTensor::vector(y))
        .covariate(c1, DoubleTensor::vector(x1))
        .covariate(c2, DoubleTensor::vector(x2));
    let start = Instant::now();
    let result = Advi::mean_field()
        .with_minibatch(minibatch)
        .with_tol_rel_obj(0.001)
        .fit(&model, &mut state, &mut rng(8))
        .unwrap();
    let elapsed = start.elapsed();
    // Stan's step sizes, η·k^(−1/2) per coordinate, leave the mean jittering by about 0.01 to 0.03 after
    // thousands of iterations, with or without minibatches (full-batch ADVI's error here is about 0.02, at 100
    // times the cost per iteration), so the means are checked to 0.05. The standard deviations show the weighting:
    // unweighted, the batch's log likelihood would give ones √(N / B) = 10 times too large.
    let mean = result.mean();
    let covariance = result.covariance();
    for a in 0..3 {
        let i = result.parameterisation().range(ids[a]).unwrap().start;
        let error = (mean[i] - exact[a]).abs();
        assert!(
            error < 0.05,
            "coefficient {a}: {} vs exact {}",
            mean[i],
            exact[a],
        );
        let sd = covariance.as_slice()[i * 3 + i].sqrt();
        let ratio = sd / exact_sd[a];
        assert!(
            (0.5..2.0).contains(&ratio),
            "coefficient {a}: sd {sd} vs exact {}",
            exact_sd[a]
        );
    }
    println!(
        "{} iterations, converged {}, {:?}; means {:?}, exact {exact:?}",
        result.iterations(),
        result.converged(),
        elapsed,
        mean
    );
    // the batched nodes get their values back
    assert_eq!(state.f64(ys).unwrap(), &DoubleTensor::zeros(&[batch]));
}

/// Solves a 3 × 3 linear system by Cramer's rule.
fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(a);
    let mut x = [0.0; 3];
    for (k, xk) in x.iter_mut().enumerate() {
        let mut m = a;
        for i in 0..3 {
            m[i][k] = b[i];
        }
        *xk = det(m) / d;
    }
    x
}

fn invalid(minibatch: impl Into<Minibatch>, model: &Model, state: &State) -> String {
    match minibatch.into().batches(model, state) {
        Err(AlgorithmError::InvalidArgument(message)) => message,
        other => panic!("expected an invalid argument, got {other:?}"),
    }
}

#[test]
fn invalid_minibatches_are_errors() {
    let (x, y) = data(40, 9);
    let (model, mut state, ids) = regression(&x.slice(":10").unwrap(), &y.slice(":10").unwrap());
    let x9 = x.slice(":39").unwrap();
    let ok = || group(10, ids, &x, &y);
    assert!(Minibatch::from(ok()).batches(&model, &state).is_ok());

    let message = invalid(Minibatch::new([]), &model, &state);
    assert!(message.contains("at least one group"), "{message}");
    let message = invalid(group(0, ids, &x, &y), &model, &state);
    assert!(message.contains("greater than 0"), "{message}");
    // the graph's batch is 10 rows, not 5
    let message = invalid(group(5, ids, &x, &y), &model, &state);
    assert!(message.contains("shape [10]"), "{message}");
    // data with different numbers of rows
    let message = invalid(group(10, ids, &x9, &y), &model, &state);
    assert!(message.contains("40 rows"), "{message}");
    // more rows in a batch than in the data
    let message = invalid(
        group(10, ids, &x.slice(":8").unwrap(), &y.slice(":8").unwrap()),
        &model,
        &state,
    );
    assert!(
        message.contains("shape [8]") || message.contains("more than"),
        "{message}"
    );
    // trailing shapes differ
    let wide = DoubleTensor::zeros(&[40, 2]);
    let message = invalid(
        MinibatchGroup::new(10)
            .observed(ids.y, wide)
            .covariate(ids.x, x.clone())
            .factor(ids.factor),
        &model,
        &state,
    );
    assert!(message.contains("shape"), "{message}");
    // a type mismatch
    let message = invalid(
        MinibatchGroup::new(10)
            .observed(ids.y, keanu_tensor::IntTensor::zeros(&[40]))
            .covariate(ids.x, x.clone())
            .factor(ids.factor),
        &model,
        &state,
    );
    assert!(message.contains("I64"), "{message}");
    // a latent is neither observed nor a covariate
    let message = invalid(
        MinibatchGroup::new(10)
            .observed(ids.y, y.clone())
            .covariate(ids.mu, x.clone())
            .factor(ids.factor),
        &model,
        &state,
    );
    assert!(
        message.contains("neither a constant nor an observed node"),
        "{message}"
    );
    let message = invalid(
        MinibatchGroup::new(10)
            .observed(ids.x, x.clone())
            .covariate(ids.x, x.clone()),
        &model,
        &state,
    );
    assert!(
        message.contains("not an observed probabilistic node"),
        "{message}"
    );
    let message = invalid(
        MinibatchGroup::new(10)
            .observed(ids.y, y.clone())
            .covariate(ids.x, x.clone())
            .factor(ids.y),
        &model,
        &state,
    );
    assert!(message.contains("not a factor"), "{message}");
    // a node twice
    let message = invalid(
        Minibatch::new([ok(), MinibatchGroup::new(10).observed(ids.y, y.clone())]),
        &model,
        &state,
    );
    assert!(message.contains("more than once"), "{message}");
    // the factor depends on the batched x but would not be weighted
    let message = invalid(
        MinibatchGroup::new(10)
            .observed(ids.y, y.clone())
            .covariate(ids.x, x.clone()),
        &model,
        &state,
    );
    assert!(message.contains("MinibatchGroup::factor"), "{message}");
    // the observations depend on the batched x but would not be weighted
    let message = invalid(
        MinibatchGroup::new(10)
            .covariate(ids.x, x.clone())
            .factor(ids.factor),
        &model,
        &state,
    );
    assert!(message.contains("would not be weighted"), "{message}");
    // a group with nothing to weigh
    let message = invalid(
        MinibatchGroup::new(10).covariate(ids.x, x.clone()),
        &model,
        &state,
    );
    assert!(message.contains("observed node or a factor"), "{message}");

    // a latent that depends on batched data (a local latent)
    let b = GraphBuilder::new();
    let slope = b.gaussian(0.0, 1.0);
    let xs = b.constant(DoubleTensor::zeros(&[10]));
    let local = b.gaussian(slope * xs, 1.0);
    let ys = b.gaussian(local, 1.0);
    b.observe(ys, DoubleTensor::zeros(&[10]));
    let (xs, ys) = (xs.id(), ys.id());
    let (graph, mut local_state) = b.build().unwrap();
    local_state
        .lazy_eval(&graph, &graph.ids().collect::<Vec<_>>(), &mut rng(0))
        .unwrap();
    let local_model = Model::new(graph, &local_state).unwrap();
    let message = invalid(
        MinibatchGroup::new(10)
            .observed(ys, y.clone())
            .covariate(xs, x.clone()),
        &local_model,
        &local_state,
    );
    assert!(message.contains("is a latent"), "{message}");

    // ADVI reports the same errors
    assert!(matches!(
        Advi::mean_field()
            .with_minibatch(group(0, ids, &x, &y))
            .fit(&model, &mut state, &mut rng(1)),
        Err(AlgorithmError::InvalidArgument(_))
    ));
}

/// A minibatch fit that fails after observing a batch gives the batched nodes their data back, as a fit that
/// succeeds does: here the initial approximation's draws make ln(intercept) NaN, a non-finite ELBO.
#[test]
fn a_failed_minibatch_fit_restores_the_batched_nodes() {
    let (n, b) = (60, 12);
    let (x, y) = data(n, 5);
    let (x12, y12) = (x.slice(":12").unwrap(), y.slice(":12").unwrap());
    let builder = GraphBuilder::new();
    let intercept = builder.gaussian(0.0, 10.0);
    let slope = builder.gaussian(0.0, 10.0);
    builder.set_value(intercept, DoubleTensor::scalar(0.3));
    builder.set_value(slope, DoubleTensor::scalar(-0.8));
    let xs = builder.constant(x12.clone());
    let ys = builder.gaussian(intercept + slope * xs, 0.7);
    builder.observe(ys, y12.clone());
    builder.factor(intercept.ln());
    let (xs, ys) = (xs.id(), ys.id());
    let (graph, mut state) = builder.build().unwrap();
    state
        .lazy_eval(&graph, &graph.ids().collect::<Vec<_>>(), &mut rng(0))
        .unwrap();
    let model = Model::new(graph, &state).unwrap();
    let minibatch = Minibatch::from(
        MinibatchGroup::new(b)
            .observed(ys, y.clone())
            .covariate(xs, x.clone()),
    );
    let result = Advi::mean_field()
        .with_minibatch(minibatch)
        .fit(&model, &mut state, &mut rng(6));
    assert!(
        matches!(result, Err(AlgorithmError::NonFiniteElbo(_))),
        "{:?}",
        result.err()
    );
    assert_eq!(state.f64(ys).unwrap(), &y12);
    assert_eq!(state.f64(xs).unwrap(), &x12);
}
