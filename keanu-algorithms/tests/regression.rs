//! Regression models: golden references from Java (golden-vectors/regression.json) and ports of Java's
//! e2e/regression tests.

use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::metropolis::{MetropolisHastings, RejectionStrategy, VariableSelector};
use keanu_algorithms::mcmc::proposal::GaussianProposal;
use keanu_algorithms::mcmc::select::Sampler;
use keanu_algorithms::optimizer::Fitness;
use keanu_algorithms::regression::{
    PredictiveModel, RegressionModel, Regularization, SamplingFitting, accuracy,
    coefficient_of_determination,
};
use keanu_distributions::KeanuRng;
use keanu_parity::{Arg, DType, GoldenTensor, load, number, parse_tensor};
use keanu_tensor::{BoolTensor, DoubleTensor};

mod common;
use common::assert_all_close_rel;

// ----- golden references from Java -----

fn f64_arg(arg: &Arg) -> DoubleTensor {
    match arg {
        Arg::Tensor(GoldenTensor {
            dtype: DType::F64,
            shape,
            data,
        }) => DoubleTensor::new(shape, data.clone()).unwrap(),
        other => panic!("expected an f64 tensor, got {other:?}"),
    }
}

fn bool_arg(arg: &Arg) -> BoolTensor {
    match arg {
        Arg::Tensor(GoldenTensor {
            dtype: DType::Bool,
            shape,
            data,
        }) => BoolTensor::new(shape, data.iter().map(|&x| x != 0.0).collect()).unwrap(),
        other => panic!("expected a bool tensor, got {other:?}"),
    }
}

fn regularization(name: &str) -> Regularization {
    match name {
        "NONE" => Regularization::None,
        "LASSO" => Regularization::Lasso,
        "RIDGE" => Regularization::Ridge,
        other => panic!("unknown regularization {other}"),
    }
}

/// Java's optimizer stops at a slightly different point. Rust must reach a fitness at least as good as Java's
/// parameters give, and its parameters must match Java's unless its fitness is strictly better. Returns whether
/// the parameters were compared by fitness alone.
fn assert_fit_matches<O: keanu_algorithms::regression::RegressionOutput>(
    name: &str,
    model: &RegressionModel<O>,
    reg: Regularization,
    java_weights: &GoldenTensor,
    java_intercept: f64,
) -> bool {
    let fitness = if reg == Regularization::None {
        Fitness::LogLikelihood
    } else {
        Fitness::LogProb
    };
    let java_w = DoubleTensor::new(&java_weights.shape, java_weights.data.clone()).unwrap();
    let rust_fitness = model
        .fitness_at(model.weights(), model.intercept(), fitness)
        .unwrap();
    let java_fitness = model.fitness_at(&java_w, java_intercept, fitness).unwrap();
    // Lasso optima can sit exactly on the Laplace prior's kink at 0 (here the intercept). Gradient optimizers
    // stall near a kink, in Java as in Rust, so neither reaches the exact optimum there: allow a small gap in log
    // probability instead of requiring the parameters to match.
    let on_kink = reg == Regularization::Lasso && java_intercept.abs() < 1e-6;
    let tolerance = if on_kink {
        0.05
    } else {
        1e-6 * java_fitness.abs().max(1.0)
    };
    assert!(
        rust_fitness >= java_fitness - tolerance,
        "{name}: Rust fitness {rust_fitness} is worse than Java's {java_fitness}"
    );
    if on_kink {
        return true;
    }
    let strictly_better = rust_fitness > java_fitness + tolerance;
    if !strictly_better {
        assert_all_close_rel(
            model.weights().as_slice(),
            &java_weights.data,
            2e-3,
            &format!("{name} weights"),
        );
        assert_all_close_rel(
            &[model.intercept()],
            &[java_intercept],
            2e-3,
            &format!("{name} intercept"),
        );
    }
    strictly_better
}

#[test]
fn every_regression_golden_case() {
    let cases = load("regression.json");
    let mut checked = 0;
    let mut better_than_java = 0;
    for case in &cases {
        let param = |name: &str| case.params.iter().find(|(n, _)| n == name).map(|(_, a)| a);
        let as_f64 = |a: &Arg| match a {
            Arg::Float(v) => *v,
            Arg::Int(v) => *v as f64,
            other => panic!("expected a number, got {other:?}"),
        };
        let settings = || -> (Regularization, Option<(f64, f64)>) {
            let reg = match param("regularization") {
                Some(Arg::Str(s)) => regularization(s),
                other => panic!("{}: missing regularization {other:?}", case.name),
            };
            let prior =
                param("priorMean").map(|m| (as_f64(m), as_f64(param("priorScale").unwrap())));
            (reg, prior)
        };
        match case.op.as_str() {
            "linearRegression" => {
                let (reg, prior) = settings();
                let (x, y, x_test, y_test) = (
                    f64_arg(&case.args[0]),
                    f64_arg(&case.args[1]),
                    f64_arg(&case.args[2]),
                    f64_arg(&case.args[3]),
                );
                let mut builder = RegressionModel::linear(&x, &y).unwrap().regularization(reg);
                if let Some((mean, scale)) = prior {
                    builder = builder.prior_on_weights_and_intercept(mean, scale);
                }
                let model = builder.fit().unwrap();
                let fit = &case.result["fit"];
                let weights = parse_tensor(&fit["weights"]).unwrap();
                assert_eq!(
                    model.weights().shape(),
                    weights.shape.as_slice(),
                    "{}",
                    case.name
                );
                let better = assert_fit_matches(
                    &case.name,
                    &model,
                    reg,
                    &weights,
                    parse_tensor(&fit["intercept"]).unwrap().data[0],
                );
                better_than_java += usize::from(better);
                let prediction = model.predict(&x_test).unwrap();
                let expected_prediction = parse_tensor(&fit["prediction"]).unwrap();
                assert_eq!(
                    prediction.shape(),
                    expected_prediction.shape.as_slice(),
                    "{}",
                    case.name
                );
                let r2 = coefficient_of_determination(&prediction, &y_test).unwrap();
                if !better {
                    assert_all_close_rel(
                        prediction.as_slice(),
                        &expected_prediction.data,
                        2e-3,
                        &format!("{} prediction", case.name),
                    );
                    assert_all_close_rel(
                        &[r2],
                        &[number(&fit["coefficientOfDetermination"])],
                        1e-3,
                        &format!("{} R²", case.name),
                    );
                }
            }
            "logisticRegression" => {
                let (reg, prior) = settings();
                let (x, y, x_test, y_test) = (
                    f64_arg(&case.args[0]),
                    bool_arg(&case.args[1]),
                    f64_arg(&case.args[2]),
                    bool_arg(&case.args[3]),
                );
                let mut builder = RegressionModel::logistic(&x, &y)
                    .unwrap()
                    .regularization(reg);
                if let Some((mean, scale)) = prior {
                    builder = builder.prior_on_weights_and_intercept(mean, scale);
                }
                let model = builder.fit().unwrap();
                let fit = &case.result["fit"];
                let weights = parse_tensor(&fit["weights"]).unwrap();
                let better = assert_fit_matches(
                    &case.name,
                    &model,
                    reg,
                    &weights,
                    parse_tensor(&fit["intercept"]).unwrap().data[0],
                );
                better_than_java += usize::from(better);
                let prediction = model.predict(&x_test).unwrap();
                let expected: Vec<bool> = parse_tensor(&fit["prediction"])
                    .unwrap()
                    .data
                    .iter()
                    .map(|&v| v != 0.0)
                    .collect();
                if !better {
                    assert_eq!(
                        prediction.as_slice(),
                        expected.as_slice(),
                        "{} prediction",
                        case.name
                    );
                    assert_all_close_rel(
                        &[accuracy(&prediction, &y_test).unwrap()],
                        &[number(&fit["accuracy"])],
                        1e-12,
                        &format!("{} accuracy", case.name),
                    );
                }
            }
            "coefficientOfDetermination" => {
                let r2 =
                    coefficient_of_determination(&f64_arg(&case.args[0]), &f64_arg(&case.args[1]))
                        .unwrap();
                assert_all_close_rel(&[r2], &[number(&case.result)], 1e-12, &case.name);
            }
            "accuracy" => {
                let a = accuracy(&bool_arg(&case.args[0]), &bool_arg(&case.args[1])).unwrap();
                assert_all_close_rel(&[a], &[number(&case.result)], 1e-12, &case.name);
            }
            other => panic!("{}: unmapped op {other}", case.name),
        }
        checked += 1;
    }
    assert_eq!(checked, cases.len());
    assert!(checked >= 27, "only {checked} cases");
    println!(
        "{checked} cases; {better_than_java} compared by fitness (lasso optima on the prior kink, or a better optimum than Java)"
    );
    // only lasso fits whose optimum sits on the prior's kink are compared by fitness alone
    assert!(
        better_than_java <= 4,
        "{better_than_java} cases differ from Java"
    );
}

// ----- ports of Java's e2e/regression tests -----

const EXPECTED_W1: f64 = 3.0;
const EXPECTED_W2: f64 = 7.0;
const EXPECTED_B: f64 = 20.0;

struct TestData {
    weights: Vec<f64>,
    intercept: f64,
    x: DoubleTensor,
    y: DoubleTensor,
}

fn uniform_column(rng: &mut KeanuRng, n: usize, min: f64, max: f64) -> Vec<f64> {
    (0..n).map(|_| rng.next_f64_between(min, max)).collect()
}

fn single_feature_data(n: usize, rng: &mut KeanuRng) -> TestData {
    let x = uniform_column(rng, n, 0.0, 10.0);
    let y: Vec<f64> = x
        .iter()
        .map(|&x| EXPECTED_W1 * x + EXPECTED_B + rng.next_gaussian())
        .collect();
    TestData {
        weights: vec![EXPECTED_W1],
        intercept: EXPECTED_B,
        x: DoubleTensor::new(&[n, 1], x).unwrap(),
        y: DoubleTensor::new(&[n, 1], y).unwrap(),
    }
}

fn two_feature_data(n: usize, rng: &mut KeanuRng) -> TestData {
    let x1 = uniform_column(rng, n, 0.0, 10.0);
    let x2 = uniform_column(rng, n, 50.0, 100.0);
    let mut x = Vec::with_capacity(2 * n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        x.extend([x1[i], x2[i]]);
        y.push(EXPECTED_W1 * x1[i] + EXPECTED_W2 * x2[i] + EXPECTED_B + rng.next_gaussian());
    }
    TestData {
        weights: vec![EXPECTED_W1, EXPECTED_W2],
        intercept: EXPECTED_B,
        x: DoubleTensor::new(&[n, 2], x).unwrap(),
        y: DoubleTensor::new(&[n, 1], y).unwrap(),
    }
}

/// x ~ U(0, 100)^features, y ~ N(x · w + 20, 1) with weights drawn by `weight`.
fn multi_feature_data(
    features: usize,
    rng: &mut KeanuRng,
    weight: impl Fn(&mut KeanuRng) -> f64,
) -> TestData {
    let n = 1000;
    let weights: Vec<f64> = (0..features).map(|_| weight(rng)).collect();
    let x: Vec<f64> = (0..n * features)
        .map(|_| rng.next_f64_between(0.0, 100.0))
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            (0..features)
                .map(|j| x[i * features + j] * weights[j])
                .sum::<f64>()
                + 20.0
                + rng.next_gaussian()
        })
        .collect();
    TestData {
        weights,
        intercept: 20.0,
        x: DoubleTensor::new(&[n, features], x).unwrap(),
        y: DoubleTensor::new(&[n, 1], y).unwrap(),
    }
}

fn laplace(rng: &mut KeanuRng, beta: f64) -> f64 {
    let u = rng.next_f64() - 0.5;
    -beta * u.signum() * (1.0 - 2.0 * u.abs()).ln()
}

/// Java's assertWeightsAndInterceptMatchTestData: intercept within 0.5, weights within 0.05.
fn assert_matches(model: &RegressionModel<f64>, data: &TestData) {
    assert!(
        (model.intercept() - data.intercept).abs() < 0.5,
        "intercept {} != {}",
        model.intercept(),
        data.intercept
    );
    for (i, (&w, &e)) in model
        .weights()
        .as_slice()
        .iter()
        .zip(&data.weights)
        .enumerate()
    {
        assert!((w - e).abs() < 0.05, "weight {i}: {w} != {e}");
    }
}

#[test]
fn linear_regression_finds_one_weight() {
    let data = single_feature_data(10_000, &mut KeanuRng::seed_from_u64(1));
    assert_matches(
        &RegressionModel::linear(&data.x, &data.y)
            .unwrap()
            .fit()
            .unwrap(),
        &data,
    );
}

#[test]
fn linear_regression_finds_two_weights() {
    let data = two_feature_data(10_000, &mut KeanuRng::seed_from_u64(2));
    assert_matches(
        &RegressionModel::linear(&data.x, &data.y)
            .unwrap()
            .fit()
            .unwrap(),
        &data,
    );
}

#[test]
fn linear_regression_finds_many_weights() {
    let data = multi_feature_data(20, &mut KeanuRng::seed_from_u64(3), |r| {
        r.next_f64_between(-10.0, 10.0)
    });
    assert_matches(
        &RegressionModel::linear(&data.x, &data.y)
            .unwrap()
            .fit()
            .unwrap(),
        &data,
    );
}

#[test]
fn ridge_regression_finds_params() {
    let mut rng = KeanuRng::seed_from_u64(4);
    let one = single_feature_data(10_000, &mut rng);
    let ridge = |d: &TestData| {
        RegressionModel::linear(&d.x, &d.y)
            .unwrap()
            .regularization(Regularization::Ridge)
            .prior_on_intercept(0.0, 40.0)
            .fit()
            .unwrap()
    };
    assert_matches(&ridge(&one), &one);
    let two = two_feature_data(10_000, &mut rng);
    assert_matches(&ridge(&two), &two);
    let many = multi_feature_data(20, &mut rng, |r| 5.0 * r.next_gaussian());
    assert_matches(&ridge(&many), &many);
}

#[test]
fn lasso_regression_finds_params() {
    let mut rng = KeanuRng::seed_from_u64(5);
    let lasso = |d: &TestData| {
        RegressionModel::linear(&d.x, &d.y)
            .unwrap()
            .regularization(Regularization::Lasso)
            .prior_on_intercept(0.0, 20.0)
            .fit()
            .unwrap()
    };
    let one = single_feature_data(10_000, &mut rng);
    assert_matches(&lasso(&one), &one);
    let two = two_feature_data(10_000, &mut rng);
    assert_matches(&lasso(&two), &two);
    let many = multi_feature_data(20, &mut rng, |r| laplace(r, 5.0));
    assert_matches(&lasso(&many), &many);
}

#[test]
fn narrower_priors_shrink_the_weights() {
    let mut rng = KeanuRng::seed_from_u64(6);
    let data = multi_feature_data(20, &mut rng, |r| 5.0 * r.next_gaussian());
    let fit = |reg, scale| {
        RegressionModel::linear(&data.x, &data.y)
            .unwrap()
            .regularization(reg)
            .prior_on_weights_and_intercept(0.0, scale)
            .fit()
            .unwrap()
    };
    let l2 = |m: &RegressionModel<f64>| m.weights().as_slice().iter().map(|w| w * w).sum::<f64>();
    assert!(l2(&fit(Regularization::Ridge, 1e-5)) < l2(&fit(Regularization::Ridge, 1e5)));
    let l1 = |m: &RegressionModel<f64>| m.weights().as_slice().iter().map(|w| w.abs()).sum::<f64>();
    let wide = RegressionModel::linear(&data.x, &data.y)
        .unwrap()
        .regularization(Regularization::Lasso)
        .prior_on_intercept(0.0, 1e5)
        .fit()
        .unwrap();
    assert!(l1(&fit(Regularization::Lasso, 1e-5)) < l1(&wide));
}

#[test]
fn lasso_brings_an_uncorrelated_weight_close_to_zero() {
    let mut rng = KeanuRng::seed_from_u64(7);
    let n = 10_000;
    let x1 = uniform_column(&mut rng, n, 0.0, 10.0);
    let x2 = uniform_column(&mut rng, n, 50.0, 100.0);
    let x3 = uniform_column(&mut rng, n, 50.0, 100.0);
    let mut x = Vec::with_capacity(3 * n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        x.extend([x1[i], x2[i], x3[i]]);
        y.push(EXPECTED_W1 * x1[i] + EXPECTED_W2 * x2[i] + EXPECTED_B + rng.next_gaussian());
    }
    let model = RegressionModel::linear(
        &DoubleTensor::new(&[n, 3], x).unwrap(),
        &DoubleTensor::new(&[n, 1], y).unwrap(),
    )
    .unwrap()
    .regularization(Regularization::Lasso)
    .fit()
    .unwrap();
    assert!(
        model.weights().as_slice()[2].abs() < 1e-2,
        "uncorrelated weight {}",
        model.weights().as_slice()[2]
    );
}

/// Ordinary least squares estimates and their standard errors for one feature (Java's
/// BivariateDataStatisticsCalculator).
fn least_squares(x: &[f64], y: &[f64]) -> (f64, f64, f64, f64) {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxx: f64 = x.iter().map(|&x| (x - mx) * (x - mx)).sum();
    let sxy: f64 = x.iter().zip(y).map(|(&x, &y)| (x - mx) * (y - my)).sum();
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    let residual: f64 = x
        .iter()
        .zip(y)
        .map(|(&x, &y)| (y - intercept - slope * x).powi(2))
        .sum();
    let se_slope = (residual / (n - 2.0) / sxx).sqrt();
    let se_intercept = se_slope * (x.iter().map(|x| x * x).sum::<f64>() / n).sqrt();
    (slope, intercept, se_slope, se_intercept)
}

#[test]
fn sampling_can_be_used_instead_of_optimization() {
    let data = single_feature_data(20, &mut KeanuRng::seed_from_u64(8));
    let count = 30_000;
    let mh = MetropolisHastings::new()
        .proposal(GaussianProposal::new(0.25))
        .selector(VariableSelector::Single)
        .rejection(RejectionStrategy::RollbackAndCascade);
    let sampling = SamplingFitting::new(Sampler::MetropolisHastings(mh), count, 9).with_options(
        SamplingOptions::default()
            .with_drop(count / 10)
            .with_down_sample_interval(2),
    );
    for reg in [
        Regularization::None,
        Regularization::Ridge,
        Regularization::Lasso,
    ] {
        let model = RegressionModel::linear(&data.x, &data.y)
            .unwrap()
            .regularization(reg)
            .prior_on_intercept(0.0, data.intercept)
            .prior_on_weights(0.0, data.weights[0])
            .sampling(sampling.clone())
            .fit()
            .unwrap();
        let samples = model.samples().expect("sampling records samples");
        let w = samples.f64(model.weights_node()).unwrap();
        let b = samples.f64(model.intercept_node()).unwrap();
        let (slope, intercept, se_slope, se_intercept) =
            least_squares(data.x.as_slice(), data.y.as_slice());
        let (w_mean, w_sd) = (
            w.mean().unwrap().as_slice()[0],
            w.variance().unwrap().as_slice()[0].sqrt(),
        );
        let (b_mean, b_sd) = (
            b.mean().unwrap().as_slice()[0],
            b.variance().unwrap().as_slice()[0].sqrt(),
        );
        // the posterior (with weak priors and unit observation noise) is close to the least squares sampling
        // distribution; Java's own tolerances only held for its particular seed, so these allow 3 standard errors
        assert!(
            (w_mean - slope).abs() < 3.0 * se_slope,
            "{reg:?} weight mean {w_mean} vs {slope} ± {se_slope}"
        );
        assert!(
            (b_mean - intercept).abs() < 3.0 * se_intercept,
            "{reg:?} intercept mean {b_mean} vs {intercept} ± {se_intercept}"
        );
        assert!(
            (w_sd / se_slope - 1.0).abs() < 0.5,
            "{reg:?} weight sd {w_sd} vs {se_slope}"
        );
        assert!(
            (b_sd / se_intercept - 1.0).abs() < 0.5,
            "{reg:?} intercept sd {b_sd} vs {se_intercept}"
        );
        // the fitted values are the most probable sample
        let best = samples.most_probable().unwrap();
        assert_eq!(best.f64(model.weights_node()).unwrap(), model.weights());
    }
}

#[test]
fn logistic_regression_classifies_held_out_data() {
    let true_weights = [0.5, -3.0, 1.5];
    let mut rng = KeanuRng::seed_from_u64(10);
    let mut generate = |n: usize| {
        let x: Vec<f64> = (0..n * 3).map(|_| rng.next_gaussian()).collect();
        let y: Vec<bool> = (0..n)
            .map(|i| {
                let z: f64 = (0..3).map(|j| x[i * 3 + j] * true_weights[j]).sum();
                rng.next_f64() < 1.0 / (1.0 + (-z).exp())
            })
            .collect();
        (
            DoubleTensor::new(&[n, 3], x).unwrap(),
            BoolTensor::new(&[n, 1], y).unwrap(),
        )
    };
    let (x_train, y_train) = generate(1250);
    let (x_test, y_test) = generate(200);
    let model = RegressionModel::logistic(&x_train, &y_train)
        .unwrap()
        .fit()
        .unwrap();
    assert!(accuracy(&model.predict(&x_test).unwrap(), &y_test).unwrap() > 0.75);
    for (w, e) in model.weights().as_slice().iter().zip(true_weights) {
        assert!((w - e).abs() < 0.3, "weight {w} != {e}");
    }
}

fn diabetes() -> (DoubleTensor, DoubleTensor) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/diabetes_standardized_training.csv");
    let text = std::fs::read_to_string(path).unwrap();
    let mut bmi = Vec::new();
    let mut y = Vec::new();
    for line in text.lines().skip(1) {
        let cells: Vec<f64> = line.split(',').map(|c| c.trim().parse().unwrap()).collect();
        bmi.push(cells[2]);
        y.push(cells[10]);
    }
    let n = bmi.len();
    (
        DoubleTensor::new(&[n, 1], bmi).unwrap(),
        DoubleTensor::new(&[n, 1], y).unwrap(),
    )
}

#[test]
fn diabetes_regression_on_bmi() {
    let (bmi, y) = diabetes();
    let mle = RegressionModel::linear(&bmi, &y).unwrap().fit().unwrap();
    assert!(
        (mle.weights().as_slice()[0] - 938.2378).abs() < 0.01,
        "weight {}",
        mle.weights().as_slice()[0]
    );
    assert!(
        (mle.intercept() - 152.9189).abs() < 0.01,
        "intercept {}",
        mle.intercept()
    );

    let ridge = RegressionModel::linear(&bmi, &y)
        .unwrap()
        .regularization(Regularization::Ridge)
        .prior_on_weights_and_intercept(0.0, 100.0)
        .fit()
        .unwrap();
    assert!((ridge.weights().as_slice()[0] - 938.2378).abs() < 0.5);
    assert!((ridge.intercept() - 152.9189).abs() < 0.5);

    // train on all but the last 100 and predict them
    let n = bmi.len();
    let train = n - 100;
    let (x_train, x_test) = (
        bmi.slice(&format!(":{train}")).unwrap(),
        bmi.slice(&format!("{train}:")).unwrap(),
    );
    let (y_train, y_test) = (
        y.slice(&format!(":{train}")).unwrap(),
        y.slice(&format!("{train}:")).unwrap(),
    );
    let model = RegressionModel::linear(&x_train, &y_train)
        .unwrap()
        .fit()
        .unwrap();
    assert!(coefficient_of_determination(&model.predict(&x_test).unwrap(), &y_test).unwrap() > 0.3);
}

#[test]
fn scores() {
    let mut rng = KeanuRng::seed_from_u64(11);
    let random = |r: &mut KeanuRng| {
        DoubleTensor::new(
            &[1, 1000],
            (0..1000)
                .map(|_| r.next_f64_between(-1000.0, 1000.0))
                .collect(),
        )
        .unwrap()
    };
    let (a, b) = (random(&mut rng), random(&mut rng));
    assert!(coefficient_of_determination(&a, &b).unwrap() < 0.0);
    assert!((coefficient_of_determination(&a, &a).unwrap() - 1.0).abs() < 1e-8);
    let noise: Vec<f64> = (0..1000).map(|_| 50.0 * rng.next_gaussian()).collect();
    let noisy = a
        .add(&DoubleTensor::new(&[1, 1000], noise).unwrap())
        .unwrap();
    assert!(coefficient_of_determination(&a, &noisy).unwrap() > 0.9);

    let t = BoolTensor::vector(vec![true, false, true, false]);
    assert_eq!(accuracy(&t, &t).unwrap(), 1.0);
    assert_eq!(accuracy(&t, &t.not()).unwrap(), 0.0);
    assert_eq!(
        accuracy(&BoolTensor::vector(vec![true; 4]), &t).unwrap(),
        0.5
    );
}

#[test]
fn shapes_and_errors() {
    // a vector of inputs is samples of a single feature
    let x = DoubleTensor::vector(vec![0.0, 1.0, 2.0, 3.0]);
    let y = &(&x * 2.0) + 1.0;
    let model = RegressionModel::linear(&x, &y).unwrap().fit().unwrap();
    assert_eq!(model.weights().shape(), &[1, 1]);
    assert_eq!(
        model
            .predict(&DoubleTensor::vector(vec![5.0, 6.0]))
            .unwrap()
            .shape(),
        &[2, 1]
    );
    assert_eq!(
        PredictiveModel::predict(&model, &DoubleTensor::scalar(1.0))
            .unwrap()
            .shape(),
        &[1, 1]
    );

    // one output per input row
    assert!(RegressionModel::linear(&x, &DoubleTensor::vector(vec![1.0, 2.0])).is_err());
    // inputs with the wrong number of features
    let two = RegressionModel::linear(
        &DoubleTensor::new(&[3, 2], vec![1.0, 2.0, 3.0, 5.0, 4.0, 1.0]).unwrap(),
        &DoubleTensor::vector(vec![1.0, 2.0, 3.0]),
    )
    .unwrap()
    .build()
    .unwrap();
    assert!(
        two.predict(&DoubleTensor::new(&[1, 3], vec![1.0, 2.0, 3.0]).unwrap())
            .is_err()
    );
    // unfitted models start at the prior means
    assert_eq!(two.weights().as_slice(), &[0.0, 0.0]);
}
