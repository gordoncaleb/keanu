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
        .space(Space::Constrained)
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
    // NUTS if every latent is continuous and differentiable, otherwise Metropolis–Hastings
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
    // compute every node once, including the parts of the kernel that depend only on the constant inputs
    let all: Vec<NodeId> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng)?;
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
