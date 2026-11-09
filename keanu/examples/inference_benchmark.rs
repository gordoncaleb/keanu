//! Run with `cargo run --release --example inference_benchmark`.

use keanu::algorithms::statistics::effective_sample_size;
use keanu::prelude::*;
use std::time::Instant;

const WARMUP: usize = 500;
const DRAWS: usize = 3000;

fn report_sampler(
    name: &str,
    id: NodeId,
    mut run: impl FnMut(&mut KeanuRng) -> (NetworkSamples, usize),
) {
    let mut rates = Vec::new();
    let mut costs = Vec::new();
    for seed in 1..=5 {
        let start = Instant::now();
        let (samples, evaluations) = run(&mut KeanuRng::seed_from_u64(seed));
        let seconds = start.elapsed().as_secs_f64();
        let series = samples.f64(id).unwrap().element(&[]).unwrap();
        rates.push(effective_sample_size(&series) / seconds);
        costs.push(evaluations);
    }
    rates.sort_by(f64::total_cmp);
    costs.sort_unstable();
    println!(
        "{name:>8} {:>10.0} ESS/s {:>7} evaluations",
        rates[2], costs[2]
    );
}

fn samplers(name: &str, model: &Model, state: &State, smooth: bool) {
    println!("\n{name} sampling");
    let nodes = model.latents();
    let id = nodes[0];
    let count = DRAWS + WARMUP;
    let options = SamplingOptions::default().with_drop(WARMUP);
    if smooth {
        report_sampler("NUTS", id, |rng| {
            let (samples, stats) = Nuts::default()
                .with_adapt_count(WARMUP)
                .with_save_statistics(true)
                .sample_with_statistics(model, &mut state.clone(), nodes, count, options, rng)
                .unwrap();
            (samples, stats.tree_size.iter().sum())
        });
        report_sampler("HMC", id, |rng| {
            let (samples, stats) = Hmc::default()
                .with_adapt_count(WARMUP)
                .with_leapfrog_steps(5)
                .sample_with_statistics(model, &mut state.clone(), nodes, count, options, rng)
                .unwrap();
            (samples, stats.gradient_evaluations)
        });
    }
    report_sampler("Slice", id, |rng| {
        let mut state = state.clone();
        let mut sampler = Slice::default()
            .sampler(model, &mut state, nodes, rng)
            .unwrap();
        let samples = generate(&mut sampler, &mut state, count, options, rng).unwrap();
        (samples, sampler.evaluations())
    });
    report_sampler("DE", id, |rng| {
        let mut states = vec![state.clone(); 8];
        for (i, state) in states.iter_mut().enumerate() {
            let x = (i as f64 - 3.5) * 0.25;
            state
                .set(model.graph(), id, DoubleTensor::scalar(x))
                .unwrap();
        }
        let (walkers, stats) = DifferentialEvolutionMetropolis::default()
            .sample_with_statistics(model, &mut states, nodes, count, options, rng)
            .unwrap();
        (walkers.into_iter().next().unwrap(), stats.evaluations)
    });
}

fn report_optimizer(name: &str, mut run: impl FnMut() -> OptimizedResult) {
    let result = run();
    let mut times = Vec::new();
    for _ in 0..21 {
        let start = Instant::now();
        let result = run();
        times.push(start.elapsed().as_secs_f64() * 1e6);
        std::hint::black_box(result);
    }
    times.sort_by(f64::total_cmp);
    println!(
        "{name:>12} {:>9.1} µs {:>5} evaluations {:>5} HVPs  log density {:>10.6}",
        times[10], result.evaluations, result.hessian_vector_products, result.fitness
    );
}

fn optimizers(name: &str, model: &Model, state: &State) {
    println!("\n{name} optimization");
    let optimizer = GradientOptimizer::new(model).with_space(Space::Constrained);
    let lbfgs = optimizer.clone().algorithm(Lbfgs::default());
    let newton = optimizer.algorithm(TrustRegion::default());
    let bobyqa = NonGradientOptimizer::new(model).algorithm(Bobyqa::default());
    for (name, optimizer) in [("L-BFGS", lbfgs), ("Trust region", newton)] {
        report_optimizer(name, || {
            optimizer.max_a_posteriori(&mut state.clone()).unwrap()
        });
    }
    report_optimizer("BOBYQA", || {
        bobyqa.max_a_posteriori(&mut state.clone()).unwrap()
    });
}

fn main() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    let y = b.gaussian(0.0, 1.0);
    b.set_value(x, DoubleTensor::scalar(0.8));
    b.set_value(y, DoubleTensor::scalar(-0.2));
    let z = b.gaussian(x + y, 0.5);
    b.observe(z, DoubleTensor::scalar(0.0));
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    samplers("Correlated Gaussian posterior", &model, &state, true);
    optimizers("Correlated Gaussian posterior", &model, &state);

    let b = GraphBuilder::new();
    let x = b.uniform(-2.0, 2.0);
    b.set_value(x, DoubleTensor::scalar(0.25));
    b.factor(b.where_(x.gt(0.0), b.scalar(0.0), b.scalar(-3.0_f64.ln())));
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    samplers("Discontinuous density", &model, &state, false);

    let b = GraphBuilder::new();
    let x = b.uniform(-5.0, 5.0);
    let y = b.uniform(-5.0, 5.0);
    b.set_value(x, DoubleTensor::scalar(-1.2));
    b.set_value(y, DoubleTensor::scalar(1.0));
    let left = 1.0 - x;
    let right = y - x * x;
    b.factor(-left * left - 100.0 * right * right);
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    optimizers("Rosenbrock valley", &model, &state);
}
