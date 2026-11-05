//! Model level benchmarks: the Java JMH NUTSBenchmark, OptimizerBenchmark and CompiledBackEnds (graph evaluation
//! only; the compiled backend isn't ported), plus log prob, gradient and Metropolis–Hastings on the golden models
//! (`keanu-parity/fixtures/golden-vectors/models`), which the Java `ModelBenchmark` (at the `java-final` tag) ran on
//! the same files.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use keanu::algorithms::optimizer::{ConvergenceChecker, Norm};
use keanu::prelude::*;
use std::hint::black_box;

fn nuts(c: &mut Criterion) {
    // A, B ~ N(0, 1); D ~ N(A² + B², 0.03) observed at 0.5; 1000 samples, adapting for all of them
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    let b_var = b.gaussian(0.0, 1.0);
    let d = b.gaussian(a * a + b_var * b_var, 0.03);
    b.observe(d, DoubleTensor::scalar(0.5));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    model
        .probe_for_non_zero_probability(&mut state, 100, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let mut state = state;
    let latents = model.latents().to_vec();
    let mut rng = KeanuRng::seed_from_u64(1);
    // as in Java, each run continues from the state the previous run left
    c.bench_function("NUTSBenchmark/takeSamples", |bench| {
        bench.iter(|| {
            let nuts = Nuts::default().with_adapt_count(1000);
            black_box(
                nuts.sample(&model, &mut state, &latents, 1000, &mut rng)
                    .unwrap(),
            )
        })
    });
}

fn optimizers(c: &mut Criterion) {
    // A ~ N(10, 0.1), B ~ N(5, 1), C ~ N(A × B, 0.1) observed at 30; MAP from A = B = 0
    let b = GraphBuilder::new();
    let a = b.gaussian(10.0, 0.1);
    let b_var = b.gaussian(5.0, 1.0);
    let c_var = b.gaussian(a * b_var, 0.1);
    b.set_value(a, DoubleTensor::scalar(0.0));
    b.set_value(b_var, DoubleTensor::scalar(0.0));
    b.observe(c_var, DoubleTensor::scalar(30.0));
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let mut group = c.benchmark_group("OptimizerBenchmark");
    let adam = Adam::default().with_convergence(ConvergenceChecker::Absolute {
        norm: Norm::Max,
        threshold: 1e-5,
    });
    let cg = ConjugateGradient::default().with_absolute_threshold(1e-5);
    for (name, algorithm) in [
        (
            "ADAM_OPTIMIZER",
            keanu::algorithms::optimizer::Algorithm::from(adam),
        ),
        ("CONJUGATE_GRADIENT", cg.into()),
        ("LBFGS", Lbfgs::default().into()),
    ] {
        let optimizer = GradientOptimizer::new(&model).algorithm(algorithm);
        // same check as the Java benchmark's teardown
        let result = optimizer.max_a_posteriori(&mut state.clone()).unwrap();
        let mut warm = state.clone();
        optimizer.max_a_posteriori(&mut warm).unwrap();
        let warm_evaluations = optimizer.max_a_posteriori(&mut warm).unwrap().evaluations;
        println!(
            "{name}: {} evaluations from A = B = 0, {warm_evaluations} from the previous optimum",
            result.evaluations
        );
        assert!(
            (result.fitness - -0.1496).abs() < 1e-2,
            "{name} fitness {}",
            result.fitness
        );
        // as Java's benchmark: the graph keeps its values between runs, so after the first run each one starts at
        // the optimum the previous run found
        let mut warm = state.clone();
        group.bench_function(format!("{name}/from previous optimum (as Java)"), |bench| {
            bench.iter(|| black_box(optimizer.max_a_posteriori(&mut warm).unwrap()))
        });
        // every run starts from A = B = 0
        group.bench_function(format!("{name}/from A = B = 0"), |bench| {
            bench.iter(|| black_box(optimizer.max_a_posteriori(&mut state.clone()).unwrap()))
        });
    }
    // Java's BOBYQA.builder().build() defaults
    let bobyqa = NonGradientOptimizer::new(&model).algorithm(Bobyqa::default());
    let result = bobyqa.max_a_posteriori(&mut state.clone()).unwrap();
    assert!(
        (result.fitness - -0.1496).abs() < 1e-2,
        "BOBYQA fitness {}",
        result.fitness
    );
    let mut warm = state.clone();
    group.bench_function(
        "BOBYQA_OPTIMIZER/from previous optimum (as Java)",
        |bench| bench.iter(|| black_box(bobyqa.max_a_posteriori(&mut warm).unwrap())),
    );
    group.bench_function("BOBYQA_OPTIMIZER/from A = B = 0", |bench| {
        bench.iter(|| black_box(bobyqa.max_a_posteriori(&mut state.clone()).unwrap()))
    });
    group.finish();
}

fn graph_evaluation(c: &mut Criterion) {
    // CompiledBackEnds' KEANU_GRAPH case: a chain of `links` plus/minus/times links, evaluated from new inputs
    let mut group = c.benchmark_group("CompiledBackEnds");
    for links in [0usize, 100] {
        let b = GraphBuilder::new();
        let a = b.gaussian(0.0, 1.0);
        let b_var = b.gaussian(0.0, 1.0);
        let mut out = a * b_var;
        let (mut left, mut right) = (a, b_var);
        for _ in 0..links {
            left = out + left;
            right = out - right;
            out = left * right;
        }
        let (a, b_var, out) = (a.id(), b_var.id(), out.id());
        b.set_value(b.double_var(a), DoubleTensor::scalar(0.0));
        b.set_value(b.double_var(b_var), DoubleTensor::scalar(0.5));
        let (graph, mut state) = b.build().unwrap();
        state
            .lazy_eval(&graph, &[out], &mut KeanuRng::seed_from_u64(1))
            .unwrap();
        // as Java's sweepValues: 10,000 evaluations per run, incrementing both inputs by 0.1 each time
        group.bench_with_input(BenchmarkId::new("KEANU_GRAPH", links), &(), |bench, _| {
            bench.iter(|| {
                let (mut va, mut vb) = (0.0, 0.5);
                let mut result = None;
                for _ in 0..10_000 {
                    state.set(&graph, a, DoubleTensor::scalar(va)).unwrap();
                    state.set(&graph, b_var, DoubleTensor::scalar(vb)).unwrap();
                    state.cascade(&graph, &[a, b_var]).unwrap();
                    result = state.f64(out).unwrap().to_scalar();
                    va += 0.1;
                    vb += 0.1;
                }
                black_box(result)
            })
        });
    }
    group.finish();
}

#[cfg(feature = "protobuf")]
fn golden_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("ModelBenchmark");
    for name in [
        "thermometers",
        "hierarchicalGaussian",
        "linearRegression",
        "gammaPoisson",
    ] {
        let path = format!(
            "{}/../keanu-parity/fixtures/golden-vectors/models/{name}.pb",
            env!("CARGO_MANIFEST_DIR")
        );
        let loaded = keanu::io::load_protobuf(&std::fs::read(path).unwrap()).unwrap();
        let (graph, mut state) = (loaded.graph, loaded.state);
        let all: Vec<NodeId> = graph.ids().collect();
        state
            .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(1))
            .unwrap();
        let model = Model::new(graph, &state).unwrap();
        // fitness and gradient evaluations of each gradient optimizer to the MAP from the stored values
        for (algorithm, choice) in [
            (
                "CONJUGATE_GRADIENT",
                keanu::algorithms::optimizer::Algorithm::from(ConjugateGradient::default()),
            ),
            ("LBFGS", Lbfgs::default().into()),
        ] {
            match GradientOptimizer::new(&model)
                .algorithm(choice)
                .max_a_posteriori(&mut state.clone())
            {
                Ok(result) => println!(
                    "MAP {name} {algorithm}: {} evaluations, fitness {}, converged {}",
                    result.evaluations, result.fitness, result.converged
                ),
                Err(e) => println!("MAP {name} {algorithm}: {e}"),
            }
        }
        group.bench_with_input(BenchmarkId::new("logProb", name), &(), |b, _| {
            b.iter(|| black_box(model.log_prob(&state).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("logProbGradient", name), &(), |b, _| {
            b.iter(|| black_box(model.log_prob_gradient(&state).unwrap()))
        });
        let mut rng = KeanuRng::seed_from_u64(1);
        let latents = model.latents().to_vec();
        // as Java: each chain continues from where the previous one ended
        let mut chain_state = state.clone();
        group.bench_with_input(
            BenchmarkId::new("metropolisHastings1000", name),
            &(),
            |b, _| {
                b.iter(|| {
                    black_box(
                        MetropolisHastings::default()
                            .sample(&model, &mut chain_state, &latents, 1000, &mut rng)
                            .unwrap(),
                    )
                })
            },
        );
    }
    group.finish();
}

#[cfg(not(feature = "protobuf"))]
fn golden_models(_: &mut Criterion) {
    eprintln!("ModelBenchmark needs `--features protobuf`");
}

fn autodiff(c: &mut Criterion) {
    // reverse mode gradients of sum(op(x)) through ops with closed-form rules (no dense Jacobians)
    let mut group = c.benchmark_group("Autodiff");
    type Build = for<'b> fn(&'b GraphBuilder, &[DoubleVar<'b>]) -> DoubleVar<'b>;
    let cases: [(&str, &[&[usize]], Build); 5] = [
        ("tensorMultiply30x30", &[&[30, 30], &[30, 30]], |_, v| {
            v[0].tensor_multiply(v[1], &[1], &[0])
        }),
        ("matmul30x30", &[&[30, 30], &[30, 30]], |_, v| {
            v[0].matmul(v[1])
        }),
        ("cumsum10000", &[&[10_000]], |_, v| v[0].cumsum(0)),
        ("product10000", &[&[10_000]], |_, v| v[0].product()),
        ("cumprod10000", &[&[10_000]], |_, v| v[0].cumprod(0)),
    ];
    for (name, shapes, build) in cases {
        let b = GraphBuilder::new();
        let inputs: Vec<DoubleVar<'_>> = shapes
            .iter()
            .map(|&shape| {
                let n: usize = shape.iter().product();
                b.constant(
                    DoubleTensor::new(shape, (0..n).map(|i| 1.0 + 1e-4 * (i % 7) as f64).collect())
                        .unwrap(),
                )
            })
            .collect();
        let out = build(&b, &inputs).sum().id();
        let wrt: Vec<NodeId> = inputs.iter().map(|v| v.id()).collect();
        let (graph, mut state) = b.build().unwrap();
        state
            .lazy_eval(&graph, &[out], &mut KeanuRng::seed_from_u64(1))
            .unwrap();
        group.bench_with_input(BenchmarkId::new("reverse", name), &(), |bench, _| {
            bench.iter(|| {
                black_box(keanu::graph::reverse_partials(&graph, &state, out, &wrt).unwrap())
            })
        });
    }
    group.finish();
}

fn scan_build(c: &mut Criterion) {
    // building the AR(1) recursion x[t] = phi x[t-1] + eps[t] over T steps: one scan node (constant in T) against an
    // unrolled loop (linear in T)
    let mut group = c.benchmark_group("ScanBuild");
    for length in [100usize, 1000, 10_000] {
        group.bench_with_input(BenchmarkId::new("scan", length), &length, |bench, &n| {
            bench.iter(|| {
                let b = GraphBuilder::new();
                let phi = b.gaussian(0.0, 1.0);
                let eps = b.gaussian_with_shape(&[n], 0.0, 1.0);
                b.scan(b.scalar(0.0), &[eps], &[phi], |_, x, e, c| c[0] * x + e[0]);
                black_box(b.build().unwrap())
            })
        });
        group.bench_with_input(
            BenchmarkId::new("unrolled", length),
            &length,
            |bench, &n| {
                bench.iter(|| {
                    let b = GraphBuilder::new();
                    let phi = b.gaussian(0.0, 1.0);
                    let eps = b.gaussian_with_shape(&[n], 0.0, 1.0);
                    let mut x = b.scalar(0.0);
                    let items: Vec<DoubleVar<'_>> = (0..n)
                        .map(|t| {
                            x = phi * x + eps.take(&[t]);
                            x
                        })
                        .collect();
                    b.stack(0, &items);
                    black_box(b.build().unwrap())
                })
            },
        );
    }
    group.finish();
}

fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(3))
        .sample_size(10)
}

criterion_group!(name = benches; config = config(); targets = nuts, optimizers, graph_evaluation, golden_models, autodiff, scan_build);
criterion_main!(benches);
