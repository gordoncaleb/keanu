//! The code samples in the guide (`rust/book`). Each sample is the region between `ANCHOR: name` and
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
    assert_eq!(graph.node(x).shape(), &[3]);
    assert!(graph.find("sigma").is_some());
    assert_eq!(ids.len(), 4);
    // ANCHOR_END: building
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
    // conjugate gradient by default
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
    let adam = Adam {
        alpha: 0.05,
        max_evaluations: 50_000,
        ..Adam::default()
    };
    let result = GradientOptimizer::new(&model)
        .algorithm(adam)
        .max_a_posteriori(&mut state)?;
    assert!(result.converged);

    let conjugate_gradient = ConjugateGradient {
        max_evaluations: 10_000,
        ..ConjugateGradient::default()
    };
    let result = GradientOptimizer::new(&model)
        .algorithm(conjugate_gradient)
        .max_likelihood(&mut state)?;
    let most_likely = result.get(temperature).unwrap().to_scalar().unwrap();
    assert!((most_likely - 26.0).abs() < 1e-3);
    // ANCHOR_END: map_settings
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
        SamplingOptions {
            drop: 10_000,
            down_sample_interval: 2,
        },
        &mut rng,
    )?;
    let mean_a = samples.f64(a).unwrap().mean()?.to_scalar().unwrap();
    let mean_b = samples.f64(b_var).unwrap().mean()?.to_scalar().unwrap();
    assert!((mean_a - 21.0).abs() < 0.1 && (mean_b - 21.0).abs() < 0.1);
    // ANCHOR_END: metropolis_hastings

    // ANCHOR: nuts
    let nuts = Nuts {
        adapt_count: 500,
        ..Nuts::default()
    };
    let samples = nuts.sample(&model, &mut state, &[a, b_var], 2000, &mut rng)?;
    let mean_a = samples.f64(a).unwrap().mean()?.to_scalar().unwrap();
    assert!((mean_a - 21.0).abs() < 0.2);
    // ANCHOR_END: nuts

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
    assert!(graph.node(last_x).label().unwrap().ends_with(".x"));

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

    let filter = ParticleFilter {
        particles: 500,
        ..ParticleFilter::default()
    };
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
