//! The code samples in the guide (`book/`). Each sample is the region between `ANCHOR: name` and
//! `ANCHOR_END: name`, which the book includes, so every sample in the book compiles and runs as a test.

use keanu::prelude::*;

#[test]
fn getting_started() -> keanu::Result<()> {
    // ANCHOR: describe
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    let b_var = b.gaussian(0.0, 1.0);
    let c = a + b_var;
    let (a, b_var, c) = (a.id(), b_var.id(), c.id());
    let (graph, mut state) = b.build()?;
    // ANCHOR_END: describe

    // ANCHOR: cascade
    // give B a value, then set A and recompute everything downstream of it
    state.set(&graph, b_var, DoubleTensor::scalar(2.0))?;
    state.set_and_cascade(&graph, a, DoubleTensor::scalar(1.0))?;
    assert_eq!(state.f64(c).unwrap().to_scalar(), Some(3.0));
    // ANCHOR_END: cascade

    // ANCHOR: lazy
    // set both, then compute C once
    state.set(&graph, a, DoubleTensor::scalar(5.0))?;
    state.set(&graph, b_var, DoubleTensor::scalar(6.0))?;
    state.cascade(&graph, &[a, b_var])?;
    assert_eq!(state.f64(c).unwrap().to_scalar(), Some(11.0));
    // ANCHOR_END: lazy

    // ANCHOR: observe
    state.observe(&graph, a, DoubleTensor::scalar(10.0))?;
    // an observed node keeps its value
    state.set(&graph, a, DoubleTensor::scalar(-1.0))?;
    assert_eq!(state.f64(a).unwrap().to_scalar(), Some(10.0));
    // ANCHOR_END: observe

    // ANCHOR: eval
    // nothing has a value yet: lazy_eval samples the probabilistic nodes it needs and computes the rest
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    let y = x * 2.0;
    let (x, y) = (x.id(), y.id());
    let (graph, mut state) = b.build()?;
    let mut rng = KeanuRng::seed_from_u64(1);
    state.lazy_eval(&graph, &[y], &mut rng)?;
    assert_eq!(
        state.f64(y).unwrap().to_scalar(),
        state.f64(x).unwrap().to_scalar().map(|v| v * 2.0)
    );
    // ANCHOR_END: eval
    Ok(())
}

#[test]
fn tensors() -> keanu::Result<()> {
    // ANCHOR: tensors
    let a = DoubleTensor::new(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let b = DoubleTensor::vector(vec![10.0, 20.0, 30.0]);

    // element-wise operations broadcast, as in numpy
    let c = a.add(&b)?;
    assert_eq!(c.as_slice(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);

    // operators work between a tensor and a scalar
    let doubled = &a * 2.0;
    assert_eq!(doubled.as_slice(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

    // reductions take (possibly negative) axes
    assert_eq!(a.sum(), 21.0);
    assert_eq!(a.sum_axes(&[-1])?.as_slice(), &[6.0, 15.0]);

    // numpy slicing syntax
    assert_eq!(a.slice("::-1, 0")?.as_slice(), &[4.0, 1.0]);
    assert_eq!(a.slice("0, 1:")?.as_slice(), &[2.0, 3.0]);

    // shapes
    assert_eq!(a.reshape(&[3, -1])?.shape(), &[3, 2]);
    assert_eq!(a.transpose()?.shape(), &[3, 2]);

    // linear algebra on the last two dimensions
    let product = a.matmul(&a.transpose()?)?;
    assert_eq!(product.as_slice(), &[14.0, 32.0, 32.0, 77.0]);
    let inverse = product.inverse()?;
    let identity = product.matmul(&inverse)?;
    assert!((identity.as_slice()[0] - 1.0).abs() < 1e-12);

    // integers and booleans
    let counts = IntTensor::vector(vec![3, -7, 4]);
    assert_eq!(counts.div(&IntTensor::scalar(2))?.as_slice(), &[1, -3, 2]); // truncates towards zero
    let big = a.gt(&DoubleTensor::scalar(3.5))?;
    assert_eq!(big.as_slice(), &[false, false, false, true, true, true]);
    // ANCHOR_END: tensors
    Ok(())
}

#[test]
fn building_models() -> keanu::Result<()> {
    // ANCHOR: building
    let b = GraphBuilder::new();

    // probabilistic nodes take other nodes or plain values as parameters
    let mu = b.gaussian(0.0, 10.0).label("mu");
    let sigma = b.half_gaussian(2.0).label("sigma");
    let x = b.gaussian_with_shape(&[3], mu, sigma).label("x");

    // deterministic nodes: operators, element-wise functions, reductions, shapes, ...
    let y = (x * 2.0 + 1.0).exp().sum();
    let positive = x.gt(0.0);
    let clipped = b.where_(positive, x, b.scalar(0.0));

    // integer and boolean nodes
    let k = b.poisson(mu.exp());
    let coin = b.bernoulli(0.5);

    let ids = [y.id(), clipped.id(), k.id(), coin.id()];
    let x = x.id();
    let (graph, _state) = b.build()?;
    assert_eq!(graph[x].shape(), &[3]);
    assert!(graph.find("sigma").is_some());
    assert_eq!(ids.len(), 4);
    // ANCHOR_END: building
    Ok(())
}

#[test]
fn indexing() -> keanu::Result<()> {
    // ANCHOR: indexing
    let b = GraphBuilder::new();

    // one mean per group, drawn around a shared mean
    let mu = b.gaussian(0.0, 10.0);
    let group_mu = b.gaussian_with_shape(&[3], mu, 1.0);

    // each observation's group, and its mean: group_mu[group]
    let group = b.int_constant(IntTensor::vector(vec![0, 0, 1, 2, 2, 2]));
    let y = b.gaussian(group_mu.gather(0, group), 0.5);
    b.observe(y, DoubleTensor::vector(vec![1.1, 0.9, 3.2, 5.0, 5.2, 4.8]));

    // the adjoint: per-group totals of the observations
    let totals = y.scatter_add(0, group, 3);
    // and stack builds a tensor from separate nodes
    let pair = b.stack(0, &[mu, group_mu.sum()]);

    let (group_mu, totals, pair) = (group_mu.id(), totals.id(), pair.id());
    let (graph, mut state) = b.build()?;
    state.lazy_eval(&graph, &[totals, pair], &mut KeanuRng::seed_from_u64(1))?;
    assert_eq!(graph[group_mu].shape(), &[3]);
    assert_eq!(graph[pair].shape(), &[2]);
    let totals = state.f64(totals).unwrap().as_slice().to_vec();
    assert!((totals[0] - 2.0).abs() < 1e-12 && (totals[2] - 15.0).abs() < 1e-12);
    // ANCHOR_END: indexing
    Ok(())
}

#[test]
fn new_distributions() -> keanu::Result<()> {
    // ANCHOR: new_distributions
    let b = GraphBuilder::new();

    // over-dispersed counts: mean mu, variance mu + mu² / alpha
    let mu = b.log_normal(1.0, 0.5);
    let alpha = b.gamma(1.0, 2.0);
    let counts = b.negative_binomial_with_shape(&[4], mu, alpha);
    b.observe(counts, IntTensor::vector(vec![3, 0, 7, 12]));

    // lifetimes
    let lifetime = b.weibull(1.5, 100.0);

    // a wind direction, near the ±π seam
    let direction = b.von_mises(std::f64::consts::PI - 0.1, 4.0);

    // ratings 0..4 from a latent score and four increasing cutpoints
    let score = b.gaussian(0.0, 1.0);
    let rating = b.ordered_logistic(score, DoubleTensor::vector(vec![-1.5, -0.5, 0.5, 1.5]));
    b.observe(rating, IntTensor::scalar(3));

    let (lifetime, direction) = (lifetime.id(), direction.id());
    let (graph, mut state) = b.build()?;
    state.lazy_eval(
        &graph,
        &[lifetime, direction],
        &mut KeanuRng::seed_from_u64(1),
    )?;
    let angle = state.f64(direction).unwrap().to_scalar().unwrap();
    assert!((-std::f64::consts::PI..std::f64::consts::PI).contains(&angle));
    // ANCHOR_END: new_distributions
    Ok(())
}

#[test]
fn truncated_and_censored() -> keanu::Result<()> {
    // ANCHOR: truncated_censored
    use keanu::graph::op::DistributionKind;

    let b = GraphBuilder::new();
    let (mu, sigma) = (b.gaussian(0.0, 5.0), b.half_gaussian(2.0));

    // a Gaussian restricted to [0, ∞) and renormalised: NUTS keeps it above 0
    let rate: DoubleVar = b.truncated(
        DistributionKind::Gaussian,
        &[mu.id(), sigma.id()],
        0.0,
        f64::INFINITY,
    );

    // readings from an instrument that saturates at 10: a reading of exactly 10 means "10 or more"
    let readings: DoubleVar = b.censored(
        DistributionKind::Gaussian,
        &[rate.id(), b.scalar(1.0).id()],
        f64::NEG_INFINITY,
        DoubleTensor::full(&[4], 10.0),
    );
    b.observe(readings, DoubleTensor::vector(vec![7.9, 9.4, 10.0, 10.0]));

    // a zero-truncated Poisson: counts of at least 1
    let count: IntVar = b.truncated(DistributionKind::Poisson, &[rate.id()], 1.0, f64::INFINITY);
    b.observe(count, IntTensor::scalar(3));

    let rate = rate.id();
    let (graph, mut state) = b.build()?;
    state.lazy_eval(&graph, &[rate], &mut KeanuRng::seed_from_u64(1))?;
    assert!(state.f64(rate).unwrap().to_scalar().unwrap() >= 0.0);
    // ANCHOR_END: truncated_censored
    Ok(())
}

#[test]
fn zero_inflated_hurdle_and_logit_counts() -> keanu::Result<()> {
    // ANCHOR: zero_inflated
    use keanu::graph::op::DistributionKind;

    let b = GraphBuilder::new();

    // fish caught per visit: some visitors never fish (structural zeros), the rest catch Poisson(λ)
    let lambda = b.gamma(2.0, 2.0);
    let never_fish = b.beta(1.0, 1.0);
    let catch: IntVar =
        b.zero_inflated_with_shape(&[6], DistributionKind::Poisson, &[lambda.id()], never_fish);
    b.observe(catch, IntTensor::vector(vec![0, 0, 3, 0, 5, 2]));

    // daily rainfall: dry with probability ψ, else a gamma amount (a value of exactly 0 is a dry day)
    let (scale, shape) = (b.half_gaussian(5.0), b.half_gaussian(5.0));
    let dry = b.beta(1.0, 1.0);
    let rain: DoubleVar = b.hurdle_with_shape(
        &[5],
        DistributionKind::Gamma,
        &[scale.id(), shape.id()],
        dry,
    );
    b.observe(rain, DoubleTensor::vector(vec![0.0, 4.2, 0.0, 11.5, 0.7]));

    // successes out of 20 with over-dispersed success probabilities
    let litter = b.beta_binomial_with_shape(&[3], 20, b.half_gaussian(5.0), b.half_gaussian(5.0));
    b.observe(litter, IntTensor::vector(vec![3, 15, 9]));

    // logistic and Poisson regressions straight from a linear predictor, accurate for any logit
    let (intercept, slope) = (b.gaussian(0.0, 2.0), b.gaussian(0.0, 2.0));
    let x = b.constant(DoubleTensor::vector(vec![-1.0, 0.0, 1.0, 2.0]));
    let linear = intercept + slope * x;
    let passed = b.bernoulli_logit(linear);
    b.observe(passed, BoolTensor::vector(vec![false, false, true, true]));
    let events = b.poisson_log(linear);
    b.observe(events, IntTensor::vector(vec![0, 1, 3, 6]));

    let (graph, mut state) = b.build()?;
    let nodes: Vec<NodeId> = graph.ids().collect();
    state.lazy_eval(&graph, &nodes, &mut KeanuRng::seed_from_u64(1))?;
    let model = Model::new(graph, &state)?;
    assert!(model.log_prob(&state)?.is_finite());
    // ANCHOR_END: zero_inflated
    Ok(())
}

#[test]
fn mixtures() -> keanu::Result<()> {
    // ANCHOR: mixture
    use keanu::graph::op::DistributionKind;

    let b = GraphBuilder::new();
    // three components: the last axis of the means, scales and weights is the component axis
    let means = b.constant(DoubleTensor::vector(vec![-5.0, 0.0, 5.0]));
    let sigmas = b.constant(DoubleTensor::vector(vec![1.0, 0.5, 2.0]));
    let weights = b.dirichlet(DoubleTensor::vector(vec![1.0, 1.0, 1.0]));
    let y: DoubleVar = b.mixture_with_shape(
        &[4],
        DistributionKind::Gaussian,
        &[means.id(), sigmas.id()],
        weights,
    );
    b.observe(y, DoubleTensor::vector(vec![-4.6, 0.2, 0.1, 6.3]));

    // a mixture of zero-truncated Poissons, sharing the bounds between components
    let rates = b.constant(DoubleTensor::vector(vec![1.0, 8.0]));
    let counts: IntVar = b.mixture_with_shape(
        &[3],
        DistributionKind::Truncated(std::sync::Arc::new(DistributionKind::Poisson)),
        &[rates.id(), b.scalar(1.0).id(), b.scalar(f64::INFINITY).id()],
        DoubleTensor::vector(vec![0.5, 0.5]),
    );
    b.observe(counts, IntTensor::vector(vec![1, 9, 2]));

    let weights = weights.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let samples = Nuts::default().sample(
        &model,
        &mut state,
        &[weights],
        500,
        &mut KeanuRng::seed_from_u64(1),
    )?;
    let mean = samples.f64(weights).unwrap().mean()?;
    assert!((mean.as_slice().iter().sum::<f64>() - 1.0).abs() < 1e-9);
    // ANCHOR_END: mixture
    Ok(())
}

#[test]
fn factors() -> keanu::Result<()> {
    // ANCHOR: factor
    let b = GraphBuilder::new();
    let effects = b.gaussian_with_shape(&[4], 0.0, 1.0);
    // a soft sum-to-zero constraint: log p += -(Σ effects / 0.05)² / 2
    let total = effects.sum() / 0.05;
    b.factor(total * total * -0.5).label("sum to zero");
    let effects = effects.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let samples = Nuts::default().sample(
        &model,
        &mut state,
        &[effects],
        500,
        &mut KeanuRng::seed_from_u64(2),
    )?;
    let draws = samples.f64(effects).unwrap();
    let columns: Vec<Vec<f64>> = (0..4)
        .map(|i| draws.element(&[i]))
        .collect::<Result<_, _>>()?;
    // after warm-up, every draw sums to within a few multiples of 0.05 of zero
    for draw in 250..columns[0].len() {
        let sum: f64 = columns.iter().map(|column| column[draw]).sum();
        assert!(sum.abs() < 0.3, "draw {draw} sums to {sum}");
    }
    // ANCHOR_END: factor
    Ok(())
}

#[test]
fn log_densities() -> keanu::Result<()> {
    // ANCHOR: log_density
    use keanu::graph::op::DistributionKind;

    let b = GraphBuilder::new();
    let rate = b.half_gaussian(5.0);
    // the counts a detector reports, of which only half are real: the real counts are the observations doubled,
    // which is a deterministic node, so its likelihood is a log density in a factor
    let reported = b.int_constant(IntTensor::vector(vec![3, 5, 4]));
    let real = reported * 2;
    let log_p = b.log_density(DistributionKind::Poisson, real, &[rate.id()]);
    b.factor(log_p);
    let rate = rate.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let samples = Nuts::default().sample(
        &model,
        &mut state,
        &[rate],
        500,
        &mut KeanuRng::seed_from_u64(3),
    )?;
    let mean = samples.f64(rate).unwrap().mean()?.to_scalar().unwrap();
    assert!((mean - 8.0).abs() < 1.5, "posterior mean rate {mean}");
    // ANCHOR_END: log_density
    Ok(())
}

#[test]
fn marginalising_discrete_latents() -> keanu::Result<()> {
    // ANCHOR: marginalize
    // yearly counts whose rate drops from `early` to `late` in an unknown year
    let counts = vec![5, 4, 6, 3, 5, 6, 4, 1, 0, 2, 1, 1, 0, 2, 1, 0];
    let years = counts.len() as i64;
    let b = GraphBuilder::new();
    let switch = b.uniform_int(0, years).label("switch");
    let early = b.exponential(0.5).label("early");
    let late = b.exponential(0.5).label("late");
    let year = b.int_constant(IntTensor::vector((0..years).collect()));
    let rate = b.where_(year.lt(switch), early, late);
    let observed = b.poisson(rate);
    b.observe(observed, IntTensor::vector(counts));
    b.set_value(switch, IntTensor::scalar(8));
    let switch = switch.id();
    let (graph, state) = b.build()?;

    // sum the switch year out: the counts' likelihood becomes a factor of the two rates
    let (graph, state, marginalized) = marginalize(&graph, &state, &[switch])?;
    let model = Model::new(graph, &state)?;
    assert!(matches!(
        Sampler::for_model(&model, &state),
        Sampler::Nuts(_)
    ));
    let rates = [model.find("early").unwrap(), model.find("late").unwrap()];
    let mut work = state.clone();
    let samples = Nuts::default().sample(
        &model,
        &mut work,
        &rates,
        1000,
        &mut KeanuRng::seed_from_u64(4),
    )?;

    // the switch year's posterior: its conditional distribution at each draw, averaged over the draws
    let switch = &recover_marginals(&model, &state, &marginalized, &samples.drop(500)?)?[0];
    let posterior = switch.mean()?;
    let year_of = |k: usize| switch.latent().values().start() + k as i64;
    let (most_likely, p) = posterior
        .as_slice()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    assert_eq!(year_of(most_likely), 7, "p = {p}");
    // ANCHOR_END: marginalize
    Ok(())
}

/// The thermometer example: a room temperature read by two noisy thermometers.
fn thermometers() -> keanu::Result<(Model, State, NodeId)> {
    // ANCHOR: thermometers
    let b = GraphBuilder::new();
    let temperature = b.uniform(20.0, 30.0).label("temperature");
    let first = b.gaussian(temperature, 2.5);
    let second = b.gaussian(temperature, 5.0);
    b.observe(first, DoubleTensor::scalar(25.0));
    b.observe(second, DoubleTensor::scalar(30.0));
    b.set_value(temperature, DoubleTensor::scalar(25.0));
    let temperature = temperature.id();

    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;
    // ANCHOR_END: thermometers
    Ok((model, state, temperature))
}

#[test]
fn maximum_a_posteriori() -> keanu::Result<()> {
    let (model, mut state, temperature) = thermometers()?;
    // ANCHOR: map
    // L-BFGS by default
    let result = GradientOptimizer::new(&model).max_a_posteriori(&mut state)?;
    let most_probable = result.get(temperature).unwrap().to_scalar().unwrap();
    assert!((most_probable - 26.0).abs() < 1e-3);
    // the state is left at the optimum
    assert_eq!(
        state.f64(temperature).unwrap().to_scalar(),
        Some(most_probable)
    );
    // ANCHOR_END: map

    // ANCHOR: map_settings
    let adam = Adam::default()
        .with_alpha(0.05)
        .with_max_evaluations(50_000);
    let result = GradientOptimizer::new(&model)
        .algorithm(adam)
        .max_a_posteriori(&mut state)?;
    assert!(result.converged);

    let lbfgs = Lbfgs::default().with_history_size(10).with_tol_grad(1e-10);
    let result = GradientOptimizer::new(&model)
        .algorithm(lbfgs)
        .max_a_posteriori(&mut state)?;
    assert!(result.converged);

    let conjugate_gradient = ConjugateGradient::default().with_max_evaluations(10_000);
    let result = GradientOptimizer::new(&model)
        .algorithm(conjugate_gradient)
        .max_likelihood(&mut state)?;
    let most_likely = result.get(temperature).unwrap().to_scalar().unwrap();
    assert!((most_likely - 26.0).abs() < 1e-3);
    // ANCHOR_END: map_settings

    // ANCHOR: bobyqa
    let bounds = OptimizerBounds::new().add_scalar(temperature, 20.0, 30.0);
    let result = NonGradientOptimizer::new(&model)
        .algorithm(Bobyqa::default().with_initial_trust_region_radius(1.0))
        .bounds(bounds)
        .max_a_posteriori(&mut state)?;
    let most_probable = result.get(temperature).unwrap().to_scalar().unwrap();
    assert!((most_probable - 26.0).abs() < 1e-6);
    // ANCHOR_END: bobyqa
    Ok(())
}

#[test]
fn map_in_unconstrained_space() -> keanu::Result<()> {
    // ANCHOR: map_transforms
    // sigma > 0 is optimized as log(sigma), but without the Jacobian: the MAP is the mode of the Gamma density,
    // (k - 1) theta = 0.5
    let b = GraphBuilder::new();
    let sigma = b.gamma(0.5, 2.0); // theta (scale) 0.5, k (shape) 2
    b.set_value(sigma, DoubleTensor::scalar(2.0));
    let sigma = sigma.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let result = GradientOptimizer::new(&model).max_a_posteriori(&mut state)?;
    let mode = result.get(sigma).unwrap().to_scalar().unwrap();
    assert!((mode - 0.5).abs() < 1e-3);

    // the opt-out: optimize the latents' own values
    let result = GradientOptimizer::new(&model)
        .with_space(Space::Constrained)
        .max_a_posteriori(&mut state)?;
    assert!((result.get(sigma).unwrap().to_scalar().unwrap() - 0.5).abs() < 1e-3);
    // ANCHOR_END: map_transforms

    // ANCHOR: map_boundary
    // with no data the mode of HalfGaussian(1) is 0, on the boundary of its support
    let b = GraphBuilder::new();
    let scale = b.half_gaussian(1.0).label("scale");
    b.set_value(scale, DoubleTensor::scalar(1.0));
    let scale = scale.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let result = GradientOptimizer::new(&model).max_a_posteriori(&mut state)?;
    assert_eq!(result.boundary, Some(scale));
    assert!(!result.converged);
    assert!(result.get(scale).unwrap().to_scalar().unwrap() < 1e-12);
    // ANCHOR_END: map_boundary
    Ok(())
}

#[test]
fn laplace_approximation() -> keanu::Result<()> {
    // ANCHOR: laplace
    // mu ~ N(1, 2), observed four times through y ~ N(mu, 1.5): the posterior of mu is Gaussian
    let b = GraphBuilder::new();
    let mu = b.gaussian(1.0, 2.0).label("mu");
    b.set_value(mu, DoubleTensor::scalar(0.0));
    let y = b.gaussian_with_shape(&[4], mu, 1.5);
    b.observe(y, DoubleTensor::vector(vec![2.1, 3.4, 1.7, 2.9]));
    let mu = mu.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;

    let result = Laplace::new()
        .with_lbfgs(Lbfgs::default().with_max_iterations(500))
        .fit(&model, &mut state)?;
    // for a Gaussian posterior the approximation is exact
    let precision = 1.0 / 4.0 + 4.0 / 2.25;
    let mean = (1.0 / 4.0 + 10.1 / 2.25) / precision;
    assert!((result.get(mu).unwrap().to_scalar().unwrap() - mean).abs() < 1e-6);
    assert!((result.covariance().as_slice()[0] - 1.0 / precision).abs() < 1e-9);

    // draws are mapped back to the latents' own values, as samples
    let mut rng = KeanuRng::seed_from_u64(1);
    let samples = result.sample(&mut state, &[mu], 2000, &mut rng)?;
    let sample_mean = samples.f64(mu).unwrap().mean()?.to_scalar().unwrap();
    assert!((sample_mean - mean).abs() < 0.1);
    // ANCHOR_END: laplace
    Ok(())
}

#[test]
fn posterior_sampling() -> keanu::Result<()> {
    // ANCHOR: sampling_model
    // A and B are around 20; a noisy measurement says A + B is 43. The posterior mean of each is 21.
    let b = GraphBuilder::new();
    let a = b.gaussian(20.0, 1.0);
    let b_var = b.gaussian(20.0, 1.0);
    let measurement = b.gaussian(a + b_var, 1.0);
    b.observe(measurement, DoubleTensor::scalar(43.0));
    let (a, b_var) = (a.id(), b_var.id());

    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let mut rng = KeanuRng::seed_from_u64(1);
    // sample the latents from their priors until the state is possible
    model.probe_for_non_zero_probability(&mut state, 100, &mut rng)?;
    // ANCHOR_END: sampling_model

    // ANCHOR: metropolis_hastings
    let samples = MetropolisHastings::default().sample_with(
        &model,
        &mut state,
        &[a, b_var],
        100_000,
        SamplingOptions::default()
            .with_drop(10_000)
            .with_down_sample_interval(2),
        &mut rng,
    )?;
    let mean_a = samples.f64(a).unwrap().mean()?.to_scalar().unwrap();
    let mean_b = samples.f64(b_var).unwrap().mean()?.to_scalar().unwrap();
    assert!((mean_a - 21.0).abs() < 0.1 && (mean_b - 21.0).abs() < 0.1);
    // ANCHOR_END: metropolis_hastings

    // ANCHOR: nuts
    let nuts = Nuts::default().with_adapt_count(500);
    let samples = nuts.sample(&model, &mut state, &[a, b_var], 2000, &mut rng)?;
    let mean_a = samples.f64(a).unwrap().mean()?.to_scalar().unwrap();
    assert!((mean_a - 21.0).abs() < 0.2);
    // ANCHOR_END: nuts

    // ANCHOR: nuts_dense
    use keanu::algorithms::mcmc::nuts::{Metric, QuadraticPotential};
    // a strongly correlated posterior: x ~ MVN(0, [[1, 0.95], [0.95, 1]])
    let b = GraphBuilder::new();
    let x = b.multivariate_gaussian(
        DoubleTensor::vector(vec![0.0, 0.0]),
        DoubleTensor::new(&[2, 2], vec![1.0, 0.95, 0.95, 1.0])?,
    );
    b.set_value(x, DoubleTensor::vector(vec![0.0, 0.0]));
    let x = x.id();
    let (graph, mut correlated_state) = b.build()?;
    let correlated = Model::new(graph, &correlated_state)?;
    let dense = Nuts::default()
        .with_adapt_count(500)
        .with_potential(QuadraticPotential::default().with_metric(Metric::Dense));
    let (_, statistics) = dense.sample_with_statistics(
        &correlated,
        &mut correlated_state,
        &[x],
        1500,
        SamplingOptions::default().with_drop(500),
        &mut rng,
    )?;
    // the adapted inverse mass matrix estimates the posterior covariance, correlation included
    let metric = statistics.dense_inverse_metric.unwrap();
    let m = metric.as_slice();
    assert!(m[1] / (m[0] * m[3]).sqrt() > 0.9);
    // ANCHOR_END: nuts_dense

    // ANCHOR: nuts_transforms
    // sigma ~ HalfGaussian(1), observed through y ~ N(0, sigma): NUTS moves log(sigma), so every draw is positive
    let b = GraphBuilder::new();
    let sigma = b.half_gaussian(1.0);
    let y = b.gaussian_with_shape(&[3], 0.0, sigma);
    b.observe(y, DoubleTensor::vector(vec![0.4, -1.1, 0.7]));
    b.set_value(sigma, DoubleTensor::scalar(1.0));
    let sigma = sigma.id();
    let (graph, mut scale_state) = b.build()?;
    let scale_model = Model::new(graph, &scale_state)?;
    let nuts = Nuts::default().with_adapt_count(200);
    let samples = nuts.sample(&scale_model, &mut scale_state, &[sigma], 500, &mut rng)?;
    let draws = samples.f64(sigma).unwrap();
    assert!(
        draws
            .as_slice()
            .iter()
            .all(|s| s.to_scalar().unwrap() > 0.0)
    );

    // the opt-out: NUTS on the latents' own values
    let constrained = nuts.with_space(Space::Constrained);
    constrained.sample(&scale_model, &mut scale_state, &[sigma], 500, &mut rng)?;
    // ANCHOR_END: nuts_transforms

    // ANCHOR: choose_sampler
    // here every latent is continuous and differentiable, so NUTS
    let sampler = Sampler::for_model(&model, &state);
    assert!(matches!(sampler, Sampler::Nuts(_)));
    let samples = sampler.sample(
        &model,
        &mut state,
        &[a],
        1000,
        SamplingOptions::default(),
        &mut rng,
    )?;
    assert_eq!(samples.len(), 1000);
    // ANCHOR_END: choose_sampler

    // ANCHOR: autocorrelation
    let chain = samples.f64(a).unwrap();
    let autocorrelation = chain.autocorrelation(&[])?; // the element index; [] for a scalar node
    assert_eq!(autocorrelation[0], 1.0);
    let effective = chain.effective_sample_size(&[])?;
    assert!(effective > 0.0 && effective <= 1000.0 * 3.0);
    // ANCHOR_END: autocorrelation
    Ok(())
}

#[test]
fn compound_sampling() -> keanu::Result<()> {
    use keanu::algorithms::mcmc::chains::sample_chains;
    use keanu::algorithms::mcmc::compound::CompoundStep;
    // ANCHOR: compound
    // a change point: counts with one rate before an unknown switch time and another after it
    let counts = vec![4, 5, 4, 6, 3, 1, 0, 2, 1, 0, 1, 2];
    let b = GraphBuilder::new();
    let switch = b.uniform_int(0, 13).label("switch");
    let early = b.exponential(1.0).label("early");
    let late = b.exponential(1.0).label("late");
    let time = b.int_constant(IntTensor::vector((0..12).collect()));
    let y = b.poisson(b.where_(switch.gt(time), early, late));
    b.observe(y, IntTensor::vector(counts));
    let (switch, early, late) = (switch.id(), early.id(), late.id());
    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;

    // discrete and continuous latents: Gibbs for the switch, NUTS for the rates
    let sampler = Sampler::for_model(&model, &state);
    let Sampler::Compound(compound) = &sampler else {
        unreachable!("a model with both kinds of latent gets the compound sampler")
    };
    assert!(matches!(
        compound.steps(),
        [CompoundStep::Gibbs(_), CompoundStep::Nuts { .. }]
    ));

    // or spelt out, with shorter NUTS adaptation
    let compound = Compound::new()
        .gibbs(&[switch])
        .nuts_with(Nuts::default().with_adapt_count(500), &[early, late]);
    let chains = sample_chains(4, 1, &state, |state, rng| {
        let (samples, statistics) = compound.sample_with_statistics(
            &model,
            state,
            &[switch, early, late],
            1500,
            SamplingOptions::default().with_drop(500),
            rng,
        )?;
        assert_eq!(statistics.nuts[0].divergence_count(), 0);
        Ok(samples)
    })?;
    // R-hat works for the discrete switch too
    for node in [switch, early, late] {
        assert!(chains.rhat(node, &[])? < 1.01);
    }
    // ANCHOR_END: compound
    let switch_mean = chains.summarize(switch, &[])?.mean;
    assert!((3.0..=7.0).contains(&switch_mean), "{switch_mean}");

    // ANCHOR: gibbs
    // an all-discrete model gets Gibbs sampling: each element drawn from its exact conditional distribution
    let b = GraphBuilder::new();
    let z = b.bernoulli_with_shape(&[3], 0.3);
    let y = b.gaussian_with_shape(&[3], z.to_double_mask() * 2.0, 1.0);
    b.observe(y, DoubleTensor::vector(vec![1.8, 0.1, 2.4]));
    let z = z.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let sampler = Sampler::for_model(&model, &state);
    assert!(matches!(sampler, Sampler::Gibbs(_)));
    let mut rng = KeanuRng::seed_from_u64(2);
    let samples = Gibbs::new(&[z]).sample(&model, &mut state, &[z], 2000, &mut rng)?;
    let first_is_true = samples.bool(z).unwrap().probability(|z| z.as_slice()[0]);
    // ANCHOR_END: gibbs
    assert!(
        first_is_true > 0.4 && first_is_true < 0.8,
        "{first_is_true}"
    );
    Ok(())
}

#[test]
fn nuts_divergences_and_energy() -> keanu::Result<()> {
    use keanu::algorithms::diagnostics::{self, E_BFMI_THRESHOLD};

    // ANCHOR: nuts_diagnostics
    // Neal's funnel: log_scale ~ N(0, 1.5), x ~ N(0, exp(log_scale)). Its narrow neck is hard to integrate through.
    let b = GraphBuilder::new();
    let log_scale = b.gaussian(0.0, 1.5);
    let x = b.gaussian_with_shape(&[5], 0.0, log_scale.exp());
    b.set_value(log_scale, DoubleTensor::scalar(0.0));
    b.set_value(x, DoubleTensor::vector(vec![0.0; 5]));
    let log_scale = log_scale.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let mut rng = KeanuRng::seed_from_u64(3);

    let adapt = 500;
    let (samples, statistics) = Nuts::default()
        .with_adapt_count(adapt)
        .sample_with_statistics(
            &model,
            &mut state,
            &[log_scale],
            adapt + 1000,
            SamplingOptions::default().with_drop(adapt),
            &mut rng,
        )?;
    // divergences are always recorded; any at all means the draws may be biased
    assert!(statistics.divergence_count() > 0);
    let flags = statistics.divergent_flags(); // one per step: draw i is step adapt + i here
    assert_eq!(flags[adapt..].len(), samples.len());

    // the Hamiltonian of each step's sample, for the energy Bayesian fraction of missing information
    let e_bfmi = diagnostics::e_bfmi(&statistics.energy[adapt..]);
    println!("E-BFMI {e_bfmi:.2} (below {E_BFMI_THRESHOLD} is a warning)");
    // ANCHOR_END: nuts_diagnostics
    assert!(e_bfmi.is_finite());
    Ok(())
}

#[test]
fn pointwise_log_likelihood() -> keanu::Result<()> {
    // ANCHOR: pointwise_log_likelihood
    use keanu::algorithms::mcmc::chains::sample_chains;

    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 5.0);
    let sigma = b.half_gaussian(2.0);
    let y = b.gaussian_with_shape(&[4], mu, sigma).label("y");
    b.observe(y, DoubleTensor::vector(vec![1.2, 0.4, 2.2, 1.0]));
    let (mu, sigma) = (mu.id(), sigma.id());
    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let nuts = Nuts::default().with_adapt_count(200);
    let chains = sample_chains(4, 1, &state, |state, rng| {
        nuts.sample(&model, state, &[mu, sigma], 250, rng)
    })?;

    // log p(y[i] | mu, sigma) for each observation at each draw of each chain
    let loglik = PointwiseLogLik::from_chains(&model, &state, &chains)?;
    assert_eq!(loglik.names(), &["y[0]", "y[1]", "y[2]", "y[3]"]);
    assert_eq!((loglik.num_chains(), loglik.draws()), (4, 250));
    let first: f64 = loglik.chain(0).unwrap()[0].iter().sum(); // the first draw's log likelihood
    // ANCHOR_END: pointwise_log_likelihood
    assert!(first.is_finite());
    Ok(())
}

#[test]
fn cmdstan_export() -> keanu::Result<()> {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 5.0).label("mu");
    let theta = b.gaussian_with_shape(&[3], mu, 1.0).label("theta");
    let y = b.gaussian(theta, 1.0).label("y");
    b.observe(y, DoubleTensor::vector(vec![1.2, 0.4, 2.2]));
    let latents = [mu.id(), theta.id()];
    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let directory = std::env::temp_dir().join(format!("keanu-book-{}", std::process::id()));
    std::fs::create_dir_all(&directory)
        .map_err(|e| keanu::algorithms::AlgorithmError::Io(e.to_string()))?;

    // ANCHOR: cmdstan_export
    use keanu::algorithms::ChainSamples;
    use keanu::algorithms::mcmc::chains::run_chains;

    // keep NUTS's per-step statistics, and record the draws after 500 warm-up steps
    let nuts = Nuts::default()
        .with_adapt_count(500)
        .with_save_statistics(true);
    let options = SamplingOptions::default().with_drop(500);
    let runs = run_chains(4, 1, &state, |state, rng| {
        nuts.sample_with_statistics(&model, state, &latents, 1500, options, rng)
    })?;
    let (samples, statistics): (Vec<_>, Vec<_>) = runs.into_iter().unzip();
    let chains = ChainSamples::new(samples)?;
    let loglik = PointwiseLogLik::from_chains(&model, &state, &chains)?;

    // output_1.csv to output_4.csv, with lp__, the NUTS columns, mu, theta.1 to theta.3 and log_lik.1 to log_lik.3
    let paths = CmdStanCsv::new(model.graph())
        .with_model_name("hierarchy")
        .with_nuts(&nuts, &statistics, options)
        .with_log_lik(&loglik)
        .write_files(&chains, &directory, "output")?;
    // ANCHOR_END: cmdstan_export
    assert_eq!(paths.len(), 4);
    let text = std::fs::read_to_string(&paths[0]).unwrap();
    let header = text.lines().find(|l| !l.starts_with('#')).unwrap();
    assert!(header.ends_with("mu,theta.1,theta.2,theta.3,log_lik.1,log_lik.2,log_lik.3"));
    std::fs::remove_dir_all(&directory).unwrap();
    Ok(())
}

#[test]
fn predictive_sampling() -> keanu::Result<()> {
    // ANCHOR: predictive
    // the rate of a Poisson process, with a Gamma prior, and three observed counts
    let b = GraphBuilder::new();
    let rate = b.gamma(2.0, 3.0); // scale 2, shape 3: prior mean 6
    let counts: Vec<NodeId> = [4, 7, 5]
        .into_iter()
        .map(|count| {
            let y = b.poisson(rate);
            b.observe(y, IntTensor::scalar(count));
            y.id()
        })
        .collect();
    let rate = rate.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let mut rng = KeanuRng::seed_from_u64(1);
    model.probe_for_non_zero_probability(&mut state, 100, &mut rng)?;

    // prior predictive: counts the model generates before seeing any data
    let prior = PredictiveSampler::new().prior(&model, &state, &counts, 1000, &mut rng)?;
    assert_eq!(prior.len(), 1000);

    // posterior predictive: one draw of every count for each of 200 evenly spaced posterior samples
    let posterior =
        MetropolisHastings::default().sample(&model, &mut state, &[rate], 5000, &mut rng)?;
    let replicated = PredictiveSampler::new()
        .with_draws(200)
        .posterior(&model, &state, &posterior, &counts, &mut rng)?;
    let first = replicated.i64(counts[0]).unwrap();
    assert_eq!(first.len(), 200);
    // the observations themselves are unchanged
    assert_eq!(state.i64(counts[0]).unwrap().to_scalar(), Some(4));
    // ANCHOR_END: predictive
    Ok(())
}

#[test]
fn multi_chain_diagnostics() -> keanu::Result<()> {
    let b = GraphBuilder::new();
    let a = b.gaussian(20.0, 1.0);
    let b_var = b.gaussian(20.0, 1.0);
    let measurement = b.gaussian(a + b_var, 1.0);
    b.observe(measurement, DoubleTensor::scalar(43.0));
    let a = a.id();
    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;

    // ANCHOR: multi_chain
    use keanu::algorithms::diagnostics::{MIN_ESS, RHAT_THRESHOLD};
    use keanu::algorithms::mcmc::chains::sample_chains;

    // four chains, each with its own copy of the state and its own random number generator
    let nuts = Nuts::default().with_adapt_count(500);
    let chains = sample_chains(4, 42, &state, |state, rng| {
        nuts.sample(&model, state, &[a], 1000, rng)
    })?;
    println!("{}", chains.summary()?); // a row per element: mean, sd, quantiles, MCSE, ESS, R-hat

    let a_stats = chains.summarize(a, &[])?;
    assert!(a_stats.r_hat < RHAT_THRESHOLD);
    assert!(a_stats.ess_bulk > MIN_ESS && a_stats.ess_tail > MIN_ESS);
    assert!((a_stats.mean - 21.0).abs() < 4.0 * a_stats.mcse_mean + 0.05);
    // ANCHOR_END: multi_chain
    Ok(())
}

#[test]
fn sequences() -> keanu::Result<()> {
    // ANCHOR: sequence
    // a Gaussian random walk x[t] ~ N(x[t-1], 1), observed with noise as y[t] ~ N(x[t], 0.5)
    use keanu::graph::sequence::SequenceBuilder;

    let observations = [0.3, 1.1, 0.9, 1.8, 2.4];
    let b = GraphBuilder::new();
    let sequence = SequenceBuilder::new(&b)
        .with_initial_state("x", b.scalar(0.0))
        .from_iter(observations, |item, observed| {
            let previous = item.double_proxy_for("x");
            let x = b.gaussian(previous, 1.0);
            let y = b.gaussian(x, 0.5);
            b.observe(y, DoubleTensor::scalar(observed));
            b.set_value(x, DoubleTensor::scalar(observed));
            item.add("x", x);
            item.add("y", y);
        })?;
    let (graph, mut state) = b.build()?;

    // each item's nodes are labelled in its own namespace
    let last_x = sequence.last_item().unwrap().get("x").unwrap();
    assert!(graph[last_x].label().unwrap().ends_with(".x"));

    let model = Model::new(graph, &state)?;
    assert_eq!(model.latents().len(), observations.len());
    // compute the values that aren't set yet (the proxies), then find the most probable walk
    model.probe_for_non_zero_probability(&mut state, 10, &mut KeanuRng::seed_from_u64(1))?;
    GradientOptimizer::new(&model).max_a_posteriori(&mut state)?;
    // ANCHOR_END: sequence
    Ok(())
}

#[test]
fn hidden_markov_models() -> keanu::Result<()> {
    // ANCHOR: hmm
    use keanu::graph::op::DistributionKind;

    let observations = vec![
        -2.1, -1.8, -2.4, 2.9, 3.2, 3.1, 2.7, -1.9, -2.2, 3.0, 3.3, 2.8,
    ];
    let b = GraphBuilder::new();
    // two regimes with unknown means (the trailing axis is the state axis) and a shared noise scale
    let means = b.gaussian(
        DoubleTensor::vector(vec![-1.0, 1.0]),
        DoubleTensor::vector(vec![2.0, 2.0]),
    );
    let sigma = b.half_gaussian(1.0);
    // each row of the transition matrix is on the simplex: one Dirichlet per row, stacked
    let rows = [
        b.dirichlet(DoubleTensor::vector(vec![5.0, 1.0])),
        b.dirichlet(DoubleTensor::vector(vec![1.0, 5.0])),
    ];
    let transition = b.stack(0, &rows);
    let initial = DoubleTensor::vector(vec![0.5, 0.5]);
    let y: DoubleVar = b.hidden_markov_model(
        observations.len(),
        DistributionKind::Gaussian,
        &[means.id(), sigma.id()],
        initial,
        transition,
    );
    b.observe(y, DoubleTensor::vector(observations));

    let (means, stay) = (means.id(), rows[0].id());
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let mut rng = KeanuRng::seed_from_u64(1);
    model.probe_for_non_zero_probability(&mut state, 10, &mut rng)?;
    // the log likelihood sums over every state path exactly, so NUTS samples the continuous parameters directly
    let samples = Nuts::default().sample(&model, &mut state, &[means, stay], 300, &mut rng)?;
    let regime_means = samples.f64(means).unwrap().mean()?;
    assert!(regime_means.as_slice()[0] < 0.0 && regime_means.as_slice()[1] > 0.0);
    // ANCHOR_END: hmm
    Ok(())
}

#[test]
fn smoothing_and_forecasting() -> keanu::Result<()> {
    // ANCHOR: smoothing
    use keanu::algorithms::timeseries::StateSpacePosterior;

    // the local level model, with the level noise scale sampled by NUTS
    let b = GraphBuilder::new();
    let matrix = |v: f64| DoubleTensor::new(&[1, 1], vec![v]);
    let level_noise = b.half_gaussian(1.0);
    let y = b.linear_gaussian_state_space(
        6,
        DoubleTensor::vector(vec![0.0]),
        matrix(10.0)?,
        matrix(1.0)?,
        (level_noise * level_noise).reshape(&[1, 1]),
        matrix(1.0)?,
        matrix(0.25)?,
    );
    b.observe(
        y,
        DoubleTensor::new(&[6, 1], vec![0.3, 0.9, 1.4, 1.1, 2.0, 2.6])?,
    );
    b.set_value(level_noise, DoubleTensor::scalar(0.5));
    let (level_noise, y) = (level_noise.id(), y.id());
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let samples = Nuts::default().sample(
        &model,
        &mut state,
        &[level_noise],
        300,
        &mut KeanuRng::seed_from_u64(1),
    )?;

    // the hidden levels and the next 4 observations, at each draw of the noise scale
    let posterior = StateSpacePosterior::new(&model, &state, y, &samples)?;
    let levels = posterior.mean_states()?; // E[level_t | y], [6, 1], averaged over the draws
    let forecasts = posterior.forecast(4)?; // each draw's predictive means and covariances
    let paths = posterior.sample_forecast(4, &mut KeanuRng::seed_from_u64(2))?; // one future path per draw
    assert_eq!(levels.shape(), &[6, 1]);
    assert_eq!(forecasts.len(), samples.len());
    assert_eq!(paths[0].observations.shape(), &[4, 1]);
    // ANCHOR_END: smoothing
    Ok(())
}

#[test]
fn structural_time_series() -> keanu::Result<()> {
    // ANCHOR: sts
    use keanu::algorithms::sts::StructuralPosterior;
    use keanu::graph::sts::{Component, StructuralModel};

    // four years of quarterly sales: a trend, a seasonal cycle and a promotion covariate
    let promotion: Vec<f64> = (0..20).map(|t| f64::from(u8::from(t % 5 == 2))).collect();
    let sales: Vec<f64> = (0..20)
        .map(|t| 10.0 + 0.3 * t as f64 + [1.5, -0.5, -2.0, 1.0][t % 4] + 2.0 * promotion[t])
        .collect();
    let b = GraphBuilder::new();
    let sts = StructuralModel::new(&b, &sales[..16])
        .with(Component::local_linear_trend())
        .with(Component::seasonal(4))
        .with(Component::regression(DoubleTensor::new(
            &[16, 1],
            promotion[..16].to_vec(),
        )?))
        .build()?;
    let nodes: Vec<NodeId> = sts.parameters().into_iter().map(|(_, id)| id).collect();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let samples = Nuts::default().sample(
        &model,
        &mut state,
        &nodes,
        300,
        &mut KeanuRng::seed_from_u64(1),
    )?;

    let posterior = StructuralPosterior::new(&model, &state, &sts, &samples)?;
    // each component's part of the 16 observed quarters, with 90% intervals
    let parts = posterior.decompose(0.9)?;
    let seasonal = parts.component("seasonal").unwrap();
    // the next 4 quarters, given the promotion's future values
    let future = DoubleTensor::new(&[4, 1], promotion[16..].to_vec())?;
    let forecast = posterior.forecast(4, &[future], 0.9)?;
    assert_eq!(seasonal.mean.len(), 16);
    assert_eq!(forecast.observations.upper.len(), 4);
    // ANCHOR_END: sts
    Ok(())
}

#[test]
fn time_series_processes() -> keanu::Result<()> {
    // ANCHOR: processes
    use keanu::distributions::Garch11;

    // 500 simulated daily returns with volatility clustering
    let truth = Garch11::new(
        DoubleTensor::scalar(0.05),
        DoubleTensor::scalar(0.1),
        DoubleTensor::scalar(0.8),
        DoubleTensor::scalar(1.0),
    );
    let returns = truth.sample(&[500], &mut KeanuRng::seed_from_u64(3))?;

    let b = GraphBuilder::new();
    // (alpha, beta, 1 − alpha − beta) on the simplex: the process stays stationary under NUTS
    let weights = b.dirichlet(DoubleTensor::vector(vec![1.0, 1.0, 1.0]));
    let omega = b.half_gaussian(1.0);
    let r = b.garch11(500, omega, weights.slice("0"), weights.slice("1"), 1.0);
    b.observe(r, returns);
    // a smooth trend as a Gaussian random walk prior, and an AR(2) series, side by side
    let trend = b.gaussian_random_walk(100, 0.0, 0.1, 0.0, 1.0);
    let rho = b.uniform_with_shape(&[2], -0.5, 0.5);
    let ar = b.autoregressive(100, 0.0, rho, 1.0, 0.0, 1.0);
    assert_eq!((trend.shape(), ar.shape()), (vec![100], vec![100]));

    b.set_value(weights, DoubleTensor::vector(vec![0.2, 0.5, 0.3]));
    b.set_value(omega, DoubleTensor::scalar(0.2));
    let (weights, omega) = (weights.id(), omega.id());
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let samples = Nuts::default().sample(
        &model,
        &mut state,
        &[weights, omega],
        400,
        &mut KeanuRng::seed_from_u64(1),
    )?;
    let alpha_beta = samples.f64(weights).unwrap().mean()?;
    let persistence = alpha_beta.as_slice()[0] + alpha_beta.as_slice()[1];
    assert!(persistence > 0.5 && persistence < 1.0);
    // ANCHOR_END: processes
    Ok(())
}

#[test]
fn kalman_filters() -> keanu::Result<()> {
    // ANCHOR: kalman
    // a local level model: a random walk level observed with noise, with an unknown level noise scale
    let b = GraphBuilder::new();
    let matrix = |v: f64| DoubleTensor::new(&[1, 1], vec![v]);
    let level_noise = b.half_gaussian(1.0);
    let level_variance = (level_noise * level_noise).reshape(&[1, 1]);
    let y = b.linear_gaussian_state_space(
        6,
        DoubleTensor::vector(vec![0.0]), // initial mean
        matrix(10.0)?,                   // initial covariance
        matrix(1.0)?,                    // transition matrix
        level_variance,                  // transition noise covariance
        matrix(1.0)?,                    // observation matrix
        matrix(0.25)?,                   // observation noise covariance
    );
    b.observe(
        y,
        DoubleTensor::new(&[6, 1], vec![0.3, 0.9, 1.4, 1.1, 2.0, 2.6])?,
    );

    let level_noise = level_noise.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    model.probe_for_non_zero_probability(&mut state, 10, &mut KeanuRng::seed_from_u64(1))?;
    // the Kalman filter gives the exact marginal likelihood and its gradient, so the noise scale is optimised
    // directly with the hidden levels integrated out
    GradientOptimizer::new(&model).max_a_posteriori(&mut state)?;
    let scale = state.f64(level_noise).unwrap().to_scalar().unwrap();
    assert!(scale > 0.0 && scale.is_finite());
    // ANCHOR_END: kalman
    Ok(())
}

#[test]
fn scan() -> keanu::Result<()> {
    // ANCHOR: scan
    // an AR(1) process x[t] = phi x[t-1] + sigma eps[t], observed with noise as y[t] ~ N(x[t], 0.5)
    let observations = [0.3, 1.1, 0.9, 1.8, 2.4, 1.7, 1.2, 0.4];
    let b = GraphBuilder::new();
    let phi = b.uniform(-1.0, 1.0);
    let sigma = b.half_gaussian(1.0);
    // the per-step noise is one vector latent, outside the loop
    let eps = b.gaussian_with_shape(&[observations.len()], 0.0, 1.0);
    // the body sees placeholders: the carry, step t of each per-step input, and the constants
    let x = b.scan(b.scalar(0.0), &[eps], &[phi, sigma], |_body, x, eps, c| {
        c[0] * x + c[1] * eps[0]
    });
    assert_eq!(x.shape(), vec![observations.len()]);
    let y = b.gaussian(x, 0.5);
    b.observe(y, DoubleTensor::vector(observations.to_vec()));
    let phi = phi.id();
    let (graph, mut state) = b.build()?;

    // the graph holds the body once: its size doesn't grow with the number of steps
    assert!(graph.len() < 20);
    let model = Model::new(graph, &state)?;
    let mut rng = KeanuRng::seed_from_u64(1);
    model.probe_for_non_zero_probability(&mut state, 10, &mut rng)?;
    let samples =
        Nuts::default()
            .with_adapt_count(200)
            .sample(&model, &mut state, &[phi], 500, &mut rng)?;
    let phi_samples = samples.f64(phi).unwrap().element(&[])?;
    assert!(phi_samples.iter().all(|p| (-1.0..=1.0).contains(p)));
    // ANCHOR_END: scan
    Ok(())
}

#[cfg(feature = "json")]
#[test]
fn save_and_load() -> keanu::Result<()> {
    let (model, state, _) = thermometers()?;
    // ANCHOR: save_load
    use keanu::io::{Metadata, load_json, save_json};

    let metadata = Metadata::from([("author".to_owned(), "me".to_owned())]);
    let json = save_json(model.graph(), &state, &metadata)?;

    let loaded = load_json(&json)?;
    let temperature = loaded.graph.find("temperature").unwrap();
    assert_eq!(
        loaded.state.f64(temperature).unwrap().to_scalar(),
        Some(25.0)
    );
    assert_eq!(loaded.metadata["author"], "me");
    // ANCHOR_END: save_load
    Ok(())
}

#[test]
fn particle_filter() -> keanu::Result<()> {
    // ANCHOR: particle_filter
    use keanu::algorithms::particle_filter::ParticleFilter;

    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    let y = b.gaussian(x, 0.5);
    b.observe(y, DoubleTensor::scalar(2.0));
    let x = x.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;

    let filter = ParticleFilter::default().with_particles(500);
    let particles = filter.run(&model, &mut state, &mut KeanuRng::seed_from_u64(1))?;
    // the particles are sorted, most probable first, and the state is left at the best one
    let best = particles[0]
        .value(x)
        .unwrap()
        .as_f64()
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!((best - 1.6).abs() < 0.15);
    // ANCHOR_END: particle_filter
    Ok(())
}

#[test]
fn custom_distributions() -> keanu::Result<()> {
    // ANCHOR: custom
    /// A density proportional to e^-|x| (a Laplace distribution with scale 1) over a scalar.
    #[derive(Debug)]
    struct DoubleExponential;

    impl CustomDistribution for DoubleExponential {
        fn log_prob(&self, value: &Value, _parameters: &[&Value]) -> Result<f64, String> {
            let x = value
                .as_f64()
                .ok_or("expected f64")?
                .to_scalar()
                .ok_or("expected a scalar")?;
            Ok(-x.abs() - std::f64::consts::LN_2)
        }

        fn log_prob_gradient(
            &self,
            value: &Value,
            _parameters: &[&Value],
        ) -> Result<Option<CustomGradient>, String> {
            let x = value
                .as_f64()
                .ok_or("expected f64")?
                .to_scalar()
                .ok_or("expected a scalar")?;
            Ok(Some(CustomGradient {
                value: Some(DoubleTensor::scalar(-x.signum())),
                parameters: vec![],
            }))
        }

        fn sample(
            &self,
            _shape: &[usize],
            _parameters: &[&Value],
            rng: &mut KeanuRng,
        ) -> Result<Value, String> {
            let u = rng.next_f64() - 0.5;
            Ok(Value::F64(DoubleTensor::scalar(
                -u.signum() * (1.0 - 2.0 * u.abs()).ln(),
            )))
        }
    }

    let b = GraphBuilder::new();
    let x: DoubleVar = b.custom(DoubleExponential, &[], &[]);
    let y = b.gaussian(x, 1.0);
    b.observe(y, DoubleTensor::scalar(1.0));
    let x = x.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let mut rng = KeanuRng::seed_from_u64(1);
    model.probe_for_non_zero_probability(&mut state, 10, &mut rng)?;
    let samples = Nuts::default().sample(&model, &mut state, &[x], 2000, &mut rng)?;
    // ANCHOR_END: custom
    let mean = samples.f64(x).unwrap().mean()?.to_scalar().unwrap();
    assert!(mean > 0.0 && mean < 1.0, "{mean}");
    Ok(())
}

#[test]
fn regression() -> keanu::Result<()> {
    // ANCHOR: regression
    // y = 3 x1 - 2 x2 + 1, plus a little noise
    let mut rng = KeanuRng::seed_from_u64(3);
    let n = 200;
    let x = DoubleTensor::new(
        &[n, 2],
        (0..2 * n)
            .map(|_| rng.next_f64_between(-5.0, 5.0))
            .collect(),
    )?;
    let xs = x.as_slice();
    let y = DoubleTensor::vector(
        (0..n)
            .map(|i| 3.0 * xs[2 * i] - 2.0 * xs[2 * i + 1] + 1.0 + 0.1 * rng.next_gaussian())
            .collect(),
    );

    let model = RegressionModel::linear(&x, &y)?.fit()?;
    println!(
        "weights {:?}, intercept {}",
        model.weights().as_slice(),
        model.intercept()
    );

    let ridge = RegressionModel::linear(&x, &y)?
        .regularization(Regularization::Ridge)
        .prior_on_weights_and_intercept(0.0, 0.1)
        .fit()?;

    let x_new = DoubleTensor::new(&[1, 2], vec![1.0, 1.0])?;
    let prediction = model.predict(&x_new)?; // about 3 - 2 + 1 = 2
    let r2 = coefficient_of_determination(&model.predict(&x)?, &y)?;
    // ANCHOR_END: regression
    assert!((model.weights().as_slice()[0] - 3.0).abs() < 0.05);
    assert!((model.intercept() - 1.0).abs() < 0.05);
    assert!(ridge.weights().as_slice()[0].abs() < model.weights().as_slice()[0].abs());
    assert!((prediction.as_slice()[0] - 2.0).abs() < 0.1);
    assert!(r2 > 0.99);
    Ok(())
}

#[test]
fn regression_sampling() -> keanu::Result<()> {
    let x = DoubleTensor::vector(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let y = DoubleTensor::vector(vec![1.2, 2.9, 5.1, 7.0, 8.8, 11.1, 13.0, 15.1]);
    // ANCHOR: regression_sampling
    let mh = MetropolisHastings::new().proposal(GaussianProposal::new(0.25));
    let model = RegressionModel::linear(&x, &y)?
        .prior_on_weights_and_intercept(0.0, 10.0)
        .sampling(SamplingFitting::new(
            Sampler::MetropolisHastings(mh),
            20_000,
            1,
        ))
        .fit()?;
    let samples = model.samples().expect("fitted by sampling");
    let weight = samples
        .f64(model.weights_node())
        .expect("weights are recorded");
    println!(
        "weight {} ± {}",
        weight.mean()?.as_slice()[0],
        weight.variance()?.as_slice()[0].sqrt()
    );
    // ANCHOR_END: regression_sampling
    let mean = weight.mean()?.as_slice()[0];
    assert!((mean - 2.0).abs() < 0.2, "{mean}");
    Ok(())
}

#[test]
fn variational_inference() -> keanu::Result<()> {
    // ANCHOR: advi
    // mu ~ N(0, 10) and sigma ~ HalfGaussian(5), observed through y ~ N(mu, sigma)
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 10.0).label("mu");
    let sigma = b.half_gaussian(5.0).label("sigma");
    b.set_value(mu, DoubleTensor::scalar(0.0));
    b.set_value(sigma, DoubleTensor::scalar(1.0));
    let y = b.gaussian_with_shape(&[6], mu, sigma);
    b.observe(y, DoubleTensor::vector(vec![4.1, 5.3, 3.8, 6.0, 4.9, 5.4]));
    let (mu, sigma) = (mu.id(), sigma.id());
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;

    let mut rng = KeanuRng::seed_from_u64(1);
    let result = Advi::mean_field()
        .with_tol_rel_obj(0.001)
        .fit(&model, &mut state, &mut rng)?;
    // the Gaussian's mean and covariance are in unconstrained space: sigma's coordinate is log(sigma)
    let log_sigma = result.parameterisation().range(sigma).unwrap().start;
    assert!(result.mean()[log_sigma].exp() > 0.3);
    assert!(result.elbo_trace().iter().all(|elbo| elbo.is_finite()));

    // draws are mapped back to the latents' own values, so every sigma is positive
    let samples = result.sample(&mut state, &[mu, sigma], 2000, &mut rng)?;
    let mean_mu = samples.f64(mu).unwrap().mean()?.to_scalar().unwrap();
    assert!((mean_mu - 4.9).abs() < 0.5);
    let sigmas = samples.f64(sigma).unwrap();
    assert!(
        sigmas
            .as_slice()
            .iter()
            .all(|s| s.to_scalar().unwrap() > 0.0)
    );
    // ANCHOR_END: advi

    // ANCHOR: advi_full_rank
    // a full covariance captures the correlation between mu and log(sigma), at a cost quadratic in the dimension
    let full = Advi::full_rank()
        .with_tol_rel_obj(0.001)
        .fit(&model, &mut state, &mut rng)?;
    assert_eq!(full.covariance().shape(), &[2, 2]);
    // ANCHOR_END: advi_full_rank

    // ANCHOR: pathfinder
    // four paths from random starting points, their draws resampled by Pareto-smoothed importance weights
    let pathfinder = Pathfinder::default().multi_path(&model, &mut state, &mut rng)?;
    assert_eq!(pathfinder.draws().len(), 1000);
    // the Pareto k̂ of the importance weights: below about 0.7 the approximation is trustworthy
    println!("k̂ = {:.2}", pathfinder.pareto_k());
    let samples = pathfinder.sample(&mut state, &[mu, sigma])?;
    let mean_mu = samples.f64(mu).unwrap().mean()?.to_scalar().unwrap();
    assert!((mean_mu - 4.9).abs() < 0.5);
    // a single path from the latents' current values
    let single = Pathfinder::default().single_path(&model, &mut state, &mut rng)?;
    let path = &single.paths()[0];
    assert!(path.elbo().is_finite() && path.best_iteration() <= path.iterations());
    // ANCHOR_END: pathfinder

    // ANCHOR: pathfinder_nuts
    use keanu::algorithms::mcmc::chains::sample_chains_from;
    // start four NUTS chains from four of the draws, near the posterior, with a short warm-up
    let starts = pathfinder.initial_states(&state, 4, &mut rng)?;
    let nuts = Nuts::default().with_adapt_count(200);
    let chains = sample_chains_from(&starts, 7, |state, rng| {
        let options = SamplingOptions::default().with_drop(200);
        let (samples, _) =
            nuts.sample_with_statistics(&model, state, &[mu, sigma], 1200, options, rng)?;
        Ok(samples)
    })?;
    assert!(chains.summarize(mu, &[])?.r_hat < 1.05);
    // ANCHOR_END: pathfinder_nuts
    Ok(())
}

#[test]
fn minibatch_variational_inference() -> keanu::Result<()> {
    // 20 000 rows of a linear regression
    let mut rng = KeanuRng::seed_from_u64(3);
    let n = 20_000;
    let xs: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
    let ys: Vec<f64> = xs
        .iter()
        .map(|x| 1.0 + 2.0 * x + 0.5 * rng.next_gaussian())
        .collect();
    let (xs, ys) = (DoubleTensor::vector(xs), DoubleTensor::vector(ys));

    // ANCHOR: advi_minibatch
    // the model is built for one batch of 500 rows: x and y have leading dimension 500
    let batch = 500;
    let b = GraphBuilder::new();
    let intercept = b.gaussian(0.0, 10.0).label("intercept");
    let slope = b.gaussian(0.0, 10.0).label("slope");
    let sigma = b.half_gaussian(2.0).label("sigma");
    b.set_value(intercept, DoubleTensor::scalar(0.0));
    b.set_value(slope, DoubleTensor::scalar(0.0));
    b.set_value(sigma, DoubleTensor::scalar(1.0));
    let x = b.constant(xs.slice(":500")?);
    let y = b.gaussian(intercept + slope * x, sigma);
    b.observe(y, ys.slice(":500")?);
    let (slope, x, y) = (slope.id(), x.id(), y.id());
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;

    // each iteration observes y and sets x to the same 500 random rows of the data, weighting y's log
    // likelihood by 20 000 / 500 = 40
    let minibatch = MinibatchGroup::new(batch).observed(y, ys).covariate(x, xs);
    let result = Advi::mean_field()
        .with_minibatch(minibatch)
        .with_max_iterations(5000)
        .fit(&model, &mut state, &mut rng)?;
    let slope_mean = result.mean()[result.parameterisation().range(slope).unwrap().start];
    assert!((slope_mean - 2.0).abs() < 0.05);
    // ANCHOR_END: advi_minibatch
    Ok(())
}

#[test]
fn gaussian_processes() -> keanu::Result<()> {
    // ANCHOR: gaussian_processes
    use keanu::graph::gp::{GaussianProcess, Kernel};

    // noisy observations of sin(x)
    let mut rng = KeanuRng::seed_from_u64(5);
    let x: Vec<f64> = (0..15).map(|i| 0.5 * i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|x| x.sin() + 0.1 * rng.next_gaussian())
        .collect();
    let (x, y) = (DoubleTensor::vector(x), DoubleTensor::vector(y));
    let grid = DoubleTensor::vector((0..7).map(|i| 0.3 + i as f64).collect());

    let b = GraphBuilder::new();
    let amplitude = b.half_gaussian(2.0).label("amplitude");
    let length_scale = b.half_gaussian(2.0).label("length_scale");
    b.set_value(amplitude, DoubleTensor::scalar(1.0));
    b.set_value(length_scale, DoubleTensor::scalar(1.0));
    let gp = GaussianProcess::new(Kernel::exp_quad(&b, amplitude, length_scale)).with_noise(0.1);
    // the observed marginal likelihood p(y | amplitude, length scale)
    gp.marginal(x.clone(), y.clone());
    // the latent function on the grid, given the data
    let prediction = gp.predict(x, y, grid.clone());
    let (mean, sd) = (prediction.mean.id(), prediction.variance().sqrt().id());
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;

    // fit the hyperparameters by MAP, then evaluate the predictive at the fitted values
    GradientOptimizer::new(&model).max_a_posteriori(&mut state)?;
    state.eval(model.graph(), &[mean, sd], &mut rng)?;
    let (mean, sd) = (state.f64(mean).unwrap(), state.f64(sd).unwrap());
    // ANCHOR_END: gaussian_processes
    for ((m, s), x) in mean
        .as_slice()
        .iter()
        .zip(sd.as_slice())
        .zip(grid.as_slice())
    {
        assert!(
            (m - x.sin()).abs() < 3.0 * s + 0.05,
            "{m} ± {s} vs sin({x})"
        );
        assert!(*s < 0.2, "sd {s}");
    }
    Ok(())
}

#[test]
fn zero_sum() -> keanu::Result<()> {
    // ANCHOR: zero_sum
    use keanu::algorithms::mcmc::chains::sample_chains;

    // 20 measurements in each of 4 groups whose effects are -0.5, 0, 0.2 and 0.3 around a mean of 1
    let mut rng = KeanuRng::seed_from_u64(8);
    let effects = [-0.5, 0.0, 0.2, 0.3];
    let group: Vec<i64> = (0..80).map(|i| i / 20).collect();
    let y: Vec<f64> = group
        .iter()
        .map(|&g| 1.0 + effects[g as usize] + 0.3 * rng.next_gaussian())
        .collect();

    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 5.0).label("mu");
    let tau = b.half_gaussian(1.0).label("tau");
    let sigma = b.half_gaussian(1.0).label("sigma");
    b.set_value(tau, DoubleTensor::scalar(1.0));
    b.set_value(sigma, DoubleTensor::scalar(1.0));
    // one effect per group, summing to zero: mu is the grand mean
    let a = b.zero_sum_normal(4, tau).label("a");
    b.set_value(a, DoubleTensor::zeros(&[4]));
    let group = b.int_constant(IntTensor::vector(group));
    let observed = b.gaussian(mu + a.gather(0, group), sigma);
    b.observe(observed, DoubleTensor::vector(y));
    let (a, mu) = (a.id(), mu.id());
    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;

    // 4 chains of 300 draws after 300 adaptation steps
    let nuts = Nuts::default().with_adapt_count(300);
    let samples = sample_chains(4, 8, &state, |state, rng| {
        let options = SamplingOptions::default().with_drop(300);
        let (samples, _) =
            nuts.sample_with_statistics(&model, state, &[a, mu], 600, options, rng)?;
        Ok(samples)
    })?;
    let a_mean = samples.summarize(a, &[0])?.mean;
    // ANCHOR_END: zero_sum
    assert!((a_mean + 0.5).abs() < 0.2, "{a_mean}");
    assert!((samples.summarize(mu, &[])?.mean - 1.0).abs() < 0.15);
    Ok(())
}

#[test]
fn hsgp() -> keanu::Result<()> {
    // ANCHOR: hsgp
    use keanu::graph::gp::{Hsgp, Kernel};

    // 500 noisy observations of sin(x) on [0, 10]
    let mut rng = KeanuRng::seed_from_u64(6);
    let x: Vec<f64> = (0..500).map(|i| i as f64 / 49.9).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|x| x.sin() + 0.2 * rng.next_gaussian())
        .collect();
    let grid = DoubleTensor::vector((0..7).map(|i| 0.3 + i as f64).collect());

    let b = GraphBuilder::new();
    let length_scale = b.half_gaussian(2.0).label("length_scale");
    b.set_value(length_scale, DoubleTensor::scalar(1.0));
    let kernel = Kernel::exp_quad(&b, 1.0, length_scale);
    // 25 basis functions on the inputs' range extended by half its width at each end
    let hsgp = Hsgp::new(&kernel, 25, 1.5)?.prior(DoubleTensor::vector(x));
    // the weights β ~ N(0, 1) are latents too, here starting at 0
    b.set_value(hsgp.beta, DoubleTensor::zeros(&[25]));
    let observed = b.gaussian(hsgp.f, 0.2);
    b.observe(observed, DoubleTensor::vector(y));
    // the same function, with the same weights, on the grid
    let on_grid = hsgp.predict(grid.clone()).id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;

    // fit the length scale and the weights by MAP
    GradientOptimizer::new(&model).max_a_posteriori(&mut state)?;
    state.eval(model.graph(), &[on_grid], &mut rng)?;
    let f = state.f64(on_grid).unwrap();
    // ANCHOR_END: hsgp
    for (f, x) in f.as_slice().iter().zip(grid.as_slice()) {
        assert!((f - x.sin()).abs() < 0.1, "{f} vs sin({x})");
    }
    Ok(())
}

#[test]
fn model_comparison() -> keanu::Result<()> {
    // ANCHOR: loo_fit
    use keanu::algorithms::mcmc::chains::sample_chains;

    let x = vec![-1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4, 1.8];
    let y = vec![-1.9, -1.2, -0.1, 0.3, 1.4, 1.7, 2.9, 3.4];
    // y[i] ~ N(a + slope·x[i], sigma), or, with `with_slope` false, y[i] ~ N(a, sigma)
    let fit = |with_slope: bool| -> keanu::Result<Loo> {
        let b = GraphBuilder::new();
        let a = b.gaussian(0.0, 5.0);
        let sigma = b.half_gaussian(2.0);
        b.set_value(sigma, DoubleTensor::scalar(1.0));
        let mut latents = vec![a.id(), sigma.id()];
        let mean = if with_slope {
            let slope = b.gaussian(0.0, 5.0);
            latents.push(slope.id());
            a + slope * b.constant(DoubleTensor::vector(x.clone()))
        } else {
            a * b.constant(DoubleTensor::vector(vec![1.0; x.len()]))
        };
        let observed = b.gaussian(mean, sigma).label("y");
        b.observe(observed, DoubleTensor::vector(y.clone()));
        let (graph, state) = b.build()?;
        let model = Model::new(graph, &state)?;
        let nuts = Nuts::default().with_adapt_count(500);
        let chains = sample_chains(4, 7, &state, |state, rng| {
            nuts.sample(&model, state, &latents, 1000, rng)?.drop(500)
        })?;
        // the log likelihood of each observation at each draw: points y[0] to y[7]
        let loglik = PointwiseLogLik::from_chains(&model, &state, &chains)?;
        let estimate = loo(&loglik)?;
        println!("{estimate}"); // elpd_loo, p_loo and looic with standard errors, and the k̂ summary
        Ok(estimate)
    };
    let (line, flat) = (fit(true)?, fit(false)?);
    assert!(line.warnings.is_empty(), "{:?}", line.warnings);
    for point in &line.points {
        println!(
            "{}: elpd {:.2}, k̂ {:.2}",
            point.name, point.elpd, point.pareto_k
        );
    }
    // ANCHOR_END: loo_fit

    // ANCHOR: loo_compare
    use keanu::algorithms::loo::compare;

    let table = compare(&[("flat", &flat), ("line", &line)])?;
    println!("{table}");
    let best = table.best();
    assert_eq!(best.name, "line");
    let other = table.get("flat").unwrap();
    // the flat model predicts much worse, by more than a few standard errors of the difference
    assert!(other.elpd_diff < -4.0 * other.se_diff);
    assert!(best.weight > 0.9);
    // ANCHOR_END: loo_compare

    // ANCHOR: loo_weights
    use keanu::algorithms::loo::{pseudo_bma_weights, stacking_weights};

    let stacking = stacking_weights(&[&line, &flat])?;
    let pseudo_bma = pseudo_bma_weights(&[&line, &flat], &mut KeanuRng::seed_from_u64(1))?;
    assert!((stacking.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(pseudo_bma[0] > 0.9);
    // ANCHOR_END: loo_weights
    Ok(())
}

#[test]
fn differential_equations() -> keanu::Result<()> {
    // ANCHOR: ode_logistic
    use keanu::graph::ode::OdeOptions;

    // logistic growth dN/dt = r N (1 − N / K) from N(0) = 5, counted at six times
    let times = vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0];
    let counts = vec![12.0, 27.0, 49.0, 71.0, 92.0, 98.0];
    let b = GraphBuilder::new();
    let r = b.log_normal(0.0, 1.0).label("r");
    let capacity = b.log_normal(100.0f64.ln(), 0.5).label("capacity");
    b.set_value(r, DoubleTensor::scalar(1.0));
    b.set_value(capacity, DoubleTensor::scalar(100.0));
    let n = b.ode(
        b.constant(DoubleTensor::vector(vec![5.0])), // the initial state, [n]
        b.constant(DoubleTensor::vector(times)),     // the output times, [T]
        b.scalar(0.0),                               // the initial time
        &[r, capacity],                              // the parameters the body sees
        OdeOptions::default(),
        // the body: dN/dt from placeholders for the time, the state and the parameters
        |_body, _t, n, theta| {
            let (r, capacity) = (theta[0], theta[1]);
            r * n * (1.0 - n / capacity)
        },
    );
    assert_eq!(n.shape(), vec![6, 1]); // [T, n]: the state at each output time
    let observed = b.gaussian(n.reshape(&[6]), 3.0);
    b.observe(observed, DoubleTensor::vector(counts));
    let (r, capacity) = (r.id(), capacity.id());
    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;
    // NUTS takes gradients through the solver
    let samples = Nuts::default().with_adapt_count(300).sample(
        &model,
        &mut state.clone(),
        &[r, capacity],
        600,
        &mut KeanuRng::seed_from_u64(2),
    )?;
    let samples = samples.drop(300)?;
    let mean = |id| {
        samples
            .f64(id)
            .unwrap()
            .mean()
            .unwrap()
            .to_scalar()
            .unwrap()
    };
    println!("r ≈ {:.2}, K ≈ {:.0}", mean(r), mean(capacity));
    // ANCHOR_END: ode_logistic
    assert!((mean(r) - 0.9).abs() < 0.15, "r {}", mean(r));
    assert!((mean(capacity) - 100.0).abs() < 6.0, "K {}", mean(capacity));

    // ANCHOR: ode_options
    use keanu::graph::ode::OdeSolver;

    // Robertson's chemical kinetics, a classic stiff problem: the stiff solver takes few steps where the explicit
    // one would need millions
    let b = GraphBuilder::new();
    let rates = b.constant(DoubleTensor::vector(vec![0.04, 1e4, 3e7]));
    let options = OdeOptions::new(OdeSolver::Stiff)
        .with_tolerances(1e-8, 1e-10)
        .with_max_steps(10_000);
    let y = b.ode(
        b.constant(DoubleTensor::vector(vec![1.0, 0.0, 0.0])),
        b.constant(DoubleTensor::vector(vec![40.0])),
        b.scalar(0.0),
        &[rates],
        options,
        |body, _t, y, k| {
            let (y1, y2, y3) = (y.take(&[0]), y.take(&[1]), y.take(&[2]));
            let k = |i: usize| k[0].take(&[i]);
            let a = k(0) * y1 - k(1) * y2 * y3;
            let c = k(2) * y2 * y2;
            body.stack(0, &[-a, a - c, c])
        },
    );
    let y = y.id();
    let (graph, mut state) = b.build()?;
    state.eval(&graph, &[y], &mut KeanuRng::seed_from_u64(0))?;
    // y₁(40) from the LSODE documentation
    assert!((state.f64(y).unwrap().as_slice()[0] - 0.7158271).abs() < 1e-6);
    // ANCHOR_END: ode_options

    // ANCHOR: ode_gradient
    use keanu::graph::reverse_partials;

    // dy/dt = −k y from y(t0) = y0: y(t) = y0 e^(−k (t − t0)); every parent is differentiable
    let b = GraphBuilder::new();
    let (k, y0, t0) = (
        b.uniform(0.0, 2.0),
        b.uniform_with_shape(&[1], 0.0, 5.0),
        b.uniform(-1.0, 1.0),
    );
    b.set_value(k, DoubleTensor::scalar(0.5));
    b.set_value(y0, DoubleTensor::vector(vec![2.0]));
    b.set_value(t0, DoubleTensor::scalar(0.0));
    let options = OdeOptions::default().with_tolerances(1e-10, 1e-10);
    let times = b.constant(DoubleTensor::vector(vec![1.0, 3.0]));
    let y = b.ode(y0, times, t0, &[k], options, |_, _, y, theta| {
        -(theta[0] * y)
    });
    let (k, y0, t0, y) = (k.id(), y0.id(), t0.id(), y.id());
    let (graph, mut state) = b.build()?;
    state.eval(&graph, &[y], &mut KeanuRng::seed_from_u64(0))?;
    let partials = reverse_partials(&graph, &state, y, &[k, y0, t0])?;
    // ∂y(3)/∂k = −3 y(3), ∂y(3)/∂y0 = e^(−3k) and ∂y(3)/∂t0 = k y(3)
    let y3 = 2.0 * (-1.5f64).exp();
    let d = |id| partials.get(id).unwrap().as_slice()[1];
    assert!((d(k) + 3.0 * y3).abs() < 1e-8);
    assert!((d(y0) - (-1.5f64).exp()).abs() < 1e-8);
    assert!((d(t0) - 0.5 * y3).abs() < 1e-8);
    // ANCHOR_END: ode_gradient

    // ANCHOR: ode_failure
    use keanu::graph::GraphError;

    // dy/dt = k y² from y(0) = 1 blows up at t = 1 / k: for k ≥ 1 there is no solution at t = 1
    let b = GraphBuilder::new();
    let k = b.uniform(0.0, 2.0);
    b.set_value(k, DoubleTensor::scalar(0.5));
    let times = b.constant(DoubleTensor::vector(vec![1.0]));
    let one = b.constant(DoubleTensor::vector(vec![1.0]));
    let y = b.ode(
        one,
        times,
        b.scalar(0.0),
        &[k],
        OdeOptions::default(),
        |_, _, y, theta| theta[0] * y * y,
    );
    let observed = b.gaussian(y, 1.0);
    b.observe(observed, DoubleTensor::new(&[1, 1], vec![2.0])?);
    let (k, y) = (k.id(), y.id());
    let (graph, mut state) = b.build()?;
    state.eval(&graph, &[y], &mut KeanuRng::seed_from_u64(0))?;
    let model = Model::new(graph, &state)?;
    let before = model.log_prob(&state)?;
    // evaluating there fails with a solver error, which the model reports as an impossible point
    let proposal = [(k, Value::from(DoubleTensor::scalar(1.5)))];
    assert_eq!(
        model.log_prob_after(&mut state, &proposal, before)?,
        f64::NEG_INFINITY
    );
    let error = state
        .set_and_cascade(model.graph(), k, DoubleTensor::scalar(1.5))
        .unwrap_err();
    assert!(matches!(error, GraphError::SolverFailed(_)) && error.is_impossible_point());
    // ANCHOR_END: ode_failure

    // ANCHOR: matrix_exp
    // a two-state continuous-time Markov chain with rates a (0 → 1) and b (1 → 0): P(t) = exp(Q t)
    let builder = GraphBuilder::new();
    let (a, b) = (0.3, 0.1);
    let q = builder.constant(DoubleTensor::new(&[2, 2], vec![-a, a, b, -b])?);
    let p = (q * 2.0).matrix_exp().id();
    let (graph, mut state) = builder.build()?;
    state.eval(&graph, &[p], &mut KeanuRng::seed_from_u64(0))?;
    let p = state.f64(p).unwrap().as_slice().to_vec();
    // each row sums to 1, and P₀₁(t) = a / (a + b) · (1 − e^(−(a + b) t))
    assert!((p[0] + p[1] - 1.0).abs() < 1e-12 && (p[2] + p[3] - 1.0).abs() < 1e-12);
    assert!((p[1] - a / (a + b) * (1.0 - (-(a + b) * 2.0f64).exp())).abs() < 1e-12);
    // ANCHOR_END: matrix_exp
    Ok(())
}

#[test]
fn ode_solvers_on_slices() -> Result<(), Box<dyn std::error::Error>> {
    // ANCHOR: ode_integrate
    use keanu::graph::ode::{OdeOptions, integrate};

    // the solvers also work on plain slices, outside any graph: the harmonic oscillator y'' = −y
    let states = integrate(
        |_t, y, dydt| {
            dydt[0] = y[1];
            dydt[1] = -y[0];
        },
        |_t, _y, dfdy, dfdt| {
            dfdy.copy_from_slice(&[0.0, 1.0, -1.0, 0.0]);
            dfdt.fill(0.0);
        },
        0.0,
        &[1.0, 0.0],
        &[std::f64::consts::PI],
        &OdeOptions::default().with_tolerances(1e-10, 1e-10),
    )?;
    assert!((states[0][0] + 1.0).abs() < 1e-8);
    // ANCHOR_END: ode_integrate
    Ok(())
}

#[test]
fn sequential_monte_carlo() -> keanu::Result<()> {
    // ANCHOR: smc
    // counts with one rate throughout, or with a change point
    let counts = vec![4, 5, 4, 6, 3, 1, 0, 2, 1, 0, 1, 2];
    let constant = {
        let b = GraphBuilder::new();
        let rate = b.exponential(1.0).label("rate");
        let y = b.poisson_with_shape(&[12], rate);
        b.observe(y, IntTensor::vector(counts.clone()));
        let rate = rate.id();
        let (graph, state) = b.build()?;
        (Model::new(graph, &state)?, state, rate)
    };
    let change_point = {
        let b = GraphBuilder::new();
        let switch = b.uniform_int(0, 13).label("switch");
        let early = b.exponential(1.0).label("early");
        let late = b.exponential(1.0).label("late");
        let time = b.int_constant(IntTensor::vector((0..12).collect()));
        let y = b.poisson(b.where_(switch.gt(time), early, late));
        b.observe(y, IntTensor::vector(counts));
        let switch = switch.id();
        let (graph, state) = b.build()?;
        (Model::new(graph, &state)?, state, switch)
    };

    // 4 independent runs of 500 particles each, at most 10 mutation steps per stage; the switch is mutated by
    // Gibbs steps at each temperature
    let smc = Smc::default().with_particles(500).with_max_steps(10);
    let one_rate = smc.sample(&constant.0, &constant.1, &[constant.2], 1)?;
    let switching = smc.sample(&change_point.0, &change_point.1, &[change_point.2], 2)?;
    assert!(switching.samples.rhat(change_point.2, &[])? < 1.01);

    // the log Bayes factor: the data strongly favour a change in rate
    let log_bayes_factor =
        switching.combined_log_marginal_likelihood() - one_rate.combined_log_marginal_likelihood();
    assert!(log_bayes_factor > 2.0, "{log_bayes_factor}"); // about 2.8
    // ANCHOR_END: smc
    // the one-rate model's evidence in closed form: a Gamma(1, 1) prior and Poisson counts
    let (n, sum) = (12.0, 29.0);
    let log_factorials: f64 = [4, 5, 4, 6, 3, 1, 0, 2, 1, 0, 1, 2]
        .iter()
        .map(|&k: &i32| (1..=k).map(|i| f64::from(i).ln()).sum::<f64>())
        .sum();
    let exact = keanu::tensor::special::ln_gamma(sum + 1.0)
        - (sum + 1.0) * (n + 1.0f64).ln()
        - log_factorials;
    assert!((one_rate.combined_log_marginal_likelihood() - exact).abs() < 0.1);
    Ok(())
}
