//! The Laplace approximation against exact Gaussian posteriors, on a positive latent, and its errors.

use keanu_algorithms::optimizer::Lbfgs;
use keanu_algorithms::{AlgorithmError, Laplace};
use keanu_distributions::KeanuRng;
use keanu_graph::{CustomDistribution, CustomGradient, GraphBuilder, GraphError, Model, Value};
use keanu_tensor::DoubleTensor;

const M0: f64 = 1.0;
const S0: f64 = 2.0;
const SIGMA: f64 = 1.5;
const YS: [f64; 4] = [2.1, 3.4, 1.7, 2.9];

/// mu ~ N(M0, S0), y_i ~ N(mu, SIGMA) observed at YS: the posterior of mu is Gaussian.
fn conjugate_gaussian() -> (Model, keanu_graph::State, keanu_graph::NodeId) {
    let b = GraphBuilder::new();
    let mu = b.gaussian(M0, S0).label("mu");
    b.set_value(mu, DoubleTensor::scalar(0.0));
    let y = b.gaussian_with_shape(&[YS.len()], mu, SIGMA);
    b.observe(y, DoubleTensor::vector(YS.to_vec()));
    let mu = mu.id();
    let (graph, state) = b.build().unwrap();
    (Model::new(graph, &state).unwrap(), state, mu)
}

#[test]
fn the_conjugate_gaussian_posterior_is_exact() {
    let (model, mut state, mu) = conjugate_gaussian();
    let precision = 1.0 / (S0 * S0) + YS.len() as f64 / (SIGMA * SIGMA);
    let mean = (M0 / (S0 * S0) + YS.iter().sum::<f64>() / (SIGMA * SIGMA)) / precision;

    let result = Laplace::new().fit(&model, &mut state).unwrap();
    assert!(result.converged());
    assert!(result.evaluations() > 0);
    assert_eq!(result.mode().len(), 1);
    assert!(
        (result.mode()[0] - mean).abs() < 1e-6,
        "{:?}",
        result.mode()
    );
    let value = result.get(mu).unwrap().to_scalar().unwrap();
    assert!((value - mean).abs() < 1e-6);
    // the state is left at the mode
    assert_eq!(state.f64(mu).unwrap().to_scalar(), Some(value));
    let covariance = result.covariance();
    assert_eq!(covariance.shape(), &[1, 1]);
    assert!((covariance.as_slice()[0] - 1.0 / precision).abs() < 1e-6);
    assert!((result.hessian().as_slice()[0] + precision).abs() < 1e-9);
    let factor = result.precision_factor().as_slice()[0];
    assert!((factor * factor - precision).abs() < 1e-9);
    assert!(result.log_density().is_finite());
}

#[test]
fn draws_have_the_posterior_mean_and_variance_and_are_reproducible() {
    let (model, mut state, mu) = conjugate_gaussian();
    let result = Laplace::new().fit(&model, &mut state).unwrap();
    let count = 4000;
    let samples = result
        .sample(&mut state, &[mu], count, &mut KeanuRng::seed_from_u64(7))
        .unwrap();
    assert_eq!(samples.len(), count);
    let draws: Vec<f64> = samples.f64(mu).unwrap().element(&[]).unwrap();
    let mean = draws.iter().sum::<f64>() / count as f64;
    let variance = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count as f64 - 1.0);
    let sd = result.covariance().as_slice()[0].sqrt();
    // 5 Monte Carlo standard errors of the mean, and of the variance (sd² √(2 / n))
    assert!((mean - result.mode()[0]).abs() < 5.0 * sd / (count as f64).sqrt());
    assert!((variance - sd * sd).abs() < 5.0 * sd * sd * (2.0 / count as f64).sqrt());

    let again = result
        .sample(&mut state, &[mu], count, &mut KeanuRng::seed_from_u64(7))
        .unwrap();
    assert_eq!(samples, again);
    let mut rng = KeanuRng::seed_from_u64(3);
    let first = result.draw(&mut rng);
    let mut rng = KeanuRng::seed_from_u64(3);
    assert_eq!(first, result.draw(&mut rng));
}

#[test]
fn a_correlated_posterior_has_the_exact_covariance() {
    // mu ~ MVN(m0, S) with a correlated S, and three observations y_i ~ N(mu, sigma) of both elements: the
    // posterior precision is S⁻¹ + 3 / sigma² I
    let m0 = [0.5, -1.0];
    let s = [2.0, 0.8, 0.8, 1.0];
    let sigma = 0.5;
    let ys = [1.0, 0.2, 1.4, -0.3, 0.7, 0.1];
    let b = GraphBuilder::new();
    let prior_mean = b.constant(DoubleTensor::vector(m0.to_vec()));
    let prior_cov = b.constant(DoubleTensor::new(&[2, 2], s.to_vec()).unwrap());
    let mu = b.multivariate_gaussian(prior_mean, prior_cov);
    b.set_value(mu, DoubleTensor::vector(vec![0.0, 0.0]));
    let y = b.gaussian_with_shape(&[3, 2], mu, sigma);
    b.observe(y, DoubleTensor::new(&[3, 2], ys.to_vec()).unwrap());
    let mu = mu.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let det = s[0] * s[3] - s[1] * s[2];
    let s_inv = [s[3] / det, -s[1] / det, -s[2] / det, s[0] / det];
    let n = 3.0 / (sigma * sigma);
    let p = [s_inv[0] + n, s_inv[1], s_inv[2], s_inv[3] + n];
    let p_det = p[0] * p[3] - p[1] * p[2];
    let cov = [p[3] / p_det, -p[1] / p_det, -p[2] / p_det, p[0] / p_det];
    let sums = [ys[0] + ys[2] + ys[4], ys[1] + ys[3] + ys[5]];
    let rhs = [
        s_inv[0] * m0[0] + s_inv[1] * m0[1] + sums[0] / (sigma * sigma),
        s_inv[2] * m0[0] + s_inv[3] * m0[1] + sums[1] / (sigma * sigma),
    ];
    let mean = [
        cov[0] * rhs[0] + cov[1] * rhs[1],
        cov[2] * rhs[0] + cov[3] * rhs[1],
    ];

    let result = Laplace::new().fit(&model, &mut state).unwrap();
    for (actual, expected) in result.mode().iter().zip(mean) {
        assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
    }
    assert_eq!(result.get(mu).unwrap().shape(), &[2]);
    let covariance = result.covariance();
    assert_eq!(covariance.shape(), &[2, 2]);
    for (actual, expected) in covariance.as_slice().iter().zip(cov) {
        assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
    }
}

#[test]
fn a_positive_latent_is_approximated_in_log_space() {
    // x ~ Gamma(theta, k): the density of z = log x is proportional to exp(k z - e^z / theta), whose mode is
    // log(k theta) and whose curvature there is -k
    let (theta, k) = (0.5, 3.0);
    let b = GraphBuilder::new();
    let x = b.gamma(theta, k);
    b.set_value(x, DoubleTensor::scalar(4.0));
    let x = x.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let result = Laplace::new().fit(&model, &mut state).unwrap();
    assert!((result.mode()[0] - (k * theta).ln()).abs() < 1e-6);
    // the constrained mode is exp of the unconstrained one, not the Gamma's own mode (k - 1) theta
    let value = result.get(x).unwrap().to_scalar().unwrap();
    assert!((value - k * theta).abs() < 1e-6);
    assert!((result.covariance().as_slice()[0] - 1.0 / k).abs() < 1e-6);
    let samples = result
        .sample(&mut state, &[x], 1000, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let draws = samples.f64(x).unwrap().element(&[]).unwrap();
    assert!(draws.iter().all(|&d| d > 0.0));
}

#[test]
fn a_flat_direction_is_an_error() {
    // only a + b is identified: with priors this wide, −H is singular to working precision
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1e10).label("a");
    let c = b.gaussian(0.0, 1e10).label("b");
    b.set_value(a, DoubleTensor::scalar(0.0));
    b.set_value(c, DoubleTensor::scalar(0.0));
    let y = b.gaussian(a + c, 1.0);
    b.observe(y, DoubleTensor::scalar(2.0));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    match Laplace::new().fit(&model, &mut state) {
        Err(AlgorithmError::IndefiniteHessian(message)) => {
            assert!(message.contains('a') && message.contains('b'), "{message}");
        }
        other => panic!("expected an indefinite Hessian, got {other:?}"),
    }
}

/// A density proportional to e^-x²/2 with no second-order rule.
#[derive(Debug)]
struct NoRule;

impl CustomDistribution for NoRule {
    fn log_prob(&self, value: &Value, _parameters: &[&Value]) -> Result<f64, String> {
        let x = value.as_f64().and_then(|v| v.to_scalar()).ok_or("scalar")?;
        Ok(-0.5 * x * x)
    }

    fn log_prob_gradient(
        &self,
        value: &Value,
        _parameters: &[&Value],
    ) -> Result<Option<CustomGradient>, String> {
        let x = value.as_f64().and_then(|v| v.to_scalar()).ok_or("scalar")?;
        Ok(Some(CustomGradient {
            value: Some(DoubleTensor::scalar(-x)),
            parameters: vec![],
        }))
    }

    fn sample(
        &self,
        _shape: &[usize],
        _parameters: &[&Value],
        rng: &mut KeanuRng,
    ) -> Result<Value, String> {
        Ok(Value::F64(DoubleTensor::scalar(rng.next_gaussian())))
    }
}

#[test]
fn a_distribution_without_a_second_order_rule_is_an_error() {
    let b = GraphBuilder::new();
    let x: keanu_graph::DoubleVar = b.custom(NoRule, &[], &[]);
    let x = x.label("custom_x");
    b.set_value(x, DoubleTensor::scalar(0.5));
    let y = b.gaussian(x, 1.0);
    b.observe(y, DoubleTensor::scalar(1.0));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    match Laplace::new().fit(&model, &mut state) {
        Err(AlgorithmError::Graph(GraphError::NoSecondOrderRule(message))) => {
            assert!(message.contains("custom_x"), "{message}");
        }
        other => panic!("expected no second-order rule, got {other:?}"),
    }
}

#[test]
fn settings_and_bad_models_are_checked() {
    let (model, mut state, _) = conjugate_gaussian();
    let invalid = Laplace::new().with_lbfgs(Lbfgs::default().with_history_size(0));
    assert!(matches!(
        invalid.fit(&model, &mut state),
        Err(AlgorithmError::InvalidSetting(_))
    ));
    // a tight L-BFGS configuration is used
    let result = Laplace::new()
        .with_lbfgs(Lbfgs::default().with_max_iterations(1))
        .fit(&model, &mut state);
    assert!(result.is_ok());

    let b = GraphBuilder::new();
    let y = b.gaussian(0.0, 1.0);
    b.observe(y, DoubleTensor::scalar(1.0));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert_eq!(
        Laplace::new().fit(&model, &mut state).unwrap_err(),
        AlgorithmError::NoLatents
    );

    let (model, mut state, mu) = conjugate_gaussian();
    let result = Laplace::new().fit(&model, &mut state).unwrap();
    let mut rng = KeanuRng::seed_from_u64(1);
    assert!(matches!(
        result.sample(&mut state, &[], 10, &mut rng),
        Err(AlgorithmError::InvalidArgument(_))
    ));
    assert!(
        result
            .sample(&mut state, &[mu], 0, &mut rng)
            .unwrap()
            .is_empty()
    );
}

/// The Laplace approximation of a model with a huge starting gradient: x ~ N(0, 1e-6) started at 10.
#[test]
fn a_huge_starting_gradient_converges() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1e-6);
    b.set_value(x, DoubleTensor::scalar(10.0));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let result = Laplace::new().fit(&model, &mut state).unwrap();
    assert!(result.converged());
    assert!(result.mode()[0].abs() < 1e-8, "{:?}", result.mode());
    let variance = result.covariance().as_slice()[0];
    assert!((variance / 1e-12 - 1.0).abs() < 1e-6, "{variance}");
}

/// The circular transform's auxiliary radial density is largest at the origin, where the angle is undefined, so
/// circular latents have no Laplace approximation: an error naming the latent.
#[test]
fn a_circular_latent_is_unsupported() {
    let b = GraphBuilder::new();
    let angle = b.von_mises(0.5, 2.0).label("angle");
    b.set_value(angle, DoubleTensor::scalar(0.1));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    match Laplace::new().fit(&model, &mut state) {
        Err(AlgorithmError::InvalidArgument(message)) => {
            assert!(message.contains("angle"), "{message}");
        }
        other => panic!("expected an unsupported circular latent, got {other:?}"),
    }
}

/// With observations exactly at the mean, the density of log sigma grows without bound as sigma goes to 0: the
/// mode is on the boundary.
#[test]
fn a_mode_on_the_boundary_is_an_error() {
    let b = GraphBuilder::new();
    let sigma = b.half_gaussian(1.0).label("sigma");
    b.set_value(sigma, DoubleTensor::scalar(1.0));
    let y = b.gaussian_with_shape(&[3], 0.0, sigma);
    b.observe(y, DoubleTensor::vector(vec![0.0; 3]));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    match Laplace::new().fit(&model, &mut state) {
        Err(AlgorithmError::ModeOnBoundary(latent)) => assert_eq!(latent, "sigma"),
        other => panic!("expected a mode on the boundary, got {other:?}"),
    }
}

/// A lone HalfGaussian(1) has its MAP on the boundary, but the density of log sigma, −e^{2z}/2 + z, has an
/// interior mode at z = 0: the Laplace approximation is fine.
#[test]
fn a_lone_half_gaussian_has_an_interior_mode_in_log_space() {
    let b = GraphBuilder::new();
    let sigma = b.half_gaussian(1.0).label("sigma");
    b.set_value(sigma, DoubleTensor::scalar(3.0));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let result = Laplace::new().fit(&model, &mut state).unwrap();
    assert!(result.mode()[0].abs() < 1e-6, "{:?}", result.mode());
}

#[test]
fn discrete_latents_and_impossible_starts_are_errors() {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0);
    b.set_value(mu, DoubleTensor::scalar(0.0));
    b.poisson(mu.exp()).label("count");
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    match Laplace::new().fit(&model, &mut state) {
        Err(AlgorithmError::DiscreteLatents(names)) => assert_eq!(names, vec!["count"]),
        other => panic!("expected discrete latents, got {other:?}"),
    }

    // a positive latent at a negative value has no unconstrained point
    let b = GraphBuilder::new();
    let sigma = b.half_gaussian(1.0);
    b.set_value(sigma, DoubleTensor::scalar(-1.0));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert_eq!(
        Laplace::new().fit(&model, &mut state).unwrap_err(),
        AlgorithmError::ImpossibleStart
    );

    // a real latent whose log probability at the start is impossible
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    b.set_value(x, DoubleTensor::scalar(0.0));
    let y = b.uniform(x, x + 1.0);
    b.observe(y, DoubleTensor::scalar(5.0));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert_eq!(
        Laplace::new().fit(&model, &mut state).unwrap_err(),
        AlgorithmError::ImpossibleStart
    );
}

/// A local level model (a random walk observed with noise) with latent level and observation noise scales: the
/// Laplace approximation needs the exact Hessian through the Kalman filter, which matches finite differences of
/// the exact gradient at the mode.
#[test]
fn a_kalman_filter_model_has_the_exact_hessian() {
    let b = GraphBuilder::new();
    let level_sd = b.half_gaussian(1.0).label("level_sd");
    let noise_sd = b.half_gaussian(1.0).label("noise_sd");
    b.set_value(level_sd, DoubleTensor::scalar(0.5));
    b.set_value(noise_sd, DoubleTensor::scalar(0.5));
    let q = (level_sd * level_sd).reshape(&[1, 1]);
    let r = (noise_sd * noise_sd).reshape(&[1, 1]);
    let one = || b.constant(DoubleTensor::new(&[1, 1], vec![1.0]).unwrap());
    let ys = [0.3, 0.9, 1.4, 1.1, 1.9, 2.6, 2.2, 3.1, 2.7, 3.5];
    let y = b.linear_gaussian_state_space(
        ys.len(),
        b.constant(DoubleTensor::vector(vec![0.0])),
        b.constant(DoubleTensor::new(&[1, 1], vec![4.0]).unwrap()),
        one(),
        q,
        one(),
        r,
    );
    b.observe(y, DoubleTensor::new(&[ys.len(), 1], ys.to_vec()).unwrap());
    let (level_sd, noise_sd) = (level_sd.id(), noise_sd.id());
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let result = Laplace::new().fit(&model, &mut state).unwrap();
    assert!(result.converged());
    let param = result.parameterisation();
    let mode = result.mode().to_vec();
    let n = mode.len();
    assert_eq!(n, 2);
    let mut gradient = |z: &[f64]| {
        let (_, g) = param.log_prob_and_gradient(z, true, &mut state).unwrap();
        g.unwrap()
    };
    // the gradient vanishes at the mode
    assert!(gradient(&mode).iter().all(|g| g.abs() < 1e-6));
    let hessian = result.hessian().as_slice();
    let h = 1e-5;
    for j in 0..n {
        let (mut plus, mut minus) = (mode.clone(), mode.clone());
        plus[j] += h;
        minus[j] -= h;
        let (gp, gm) = (gradient(&plus), gradient(&minus));
        for i in 0..n {
            let expected = (gp[i] - gm[i]) / (2.0 * h);
            let actual = hessian[i * n + j];
            assert!(
                (actual - expected).abs() <= 1e-6 * expected.abs().max(1.0),
                "H[{i}][{j}] = {actual}, finite differences give {expected}"
            );
        }
    }
    assert!((hessian[1] - hessian[n]).abs() < 1e-10);
    let draws = result
        .sample(
            &mut state,
            &[level_sd, noise_sd],
            100,
            &mut KeanuRng::seed_from_u64(2),
        )
        .unwrap();
    assert_eq!(draws.len(), 100);
}
