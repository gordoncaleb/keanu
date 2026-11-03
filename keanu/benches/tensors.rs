//! Tensor benchmarks mirroring the Java JMH benchmarks in `benchmarks/` (BroadcastBinaryDoubleOperations,
//! ElementwiseBinaryDoubleOperations, UnaryDoubleOperations) with the same shapes and operations.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use keanu::prelude::*;
use std::hint::black_box;

fn shape(text: &str) -> Vec<usize> {
    if text.is_empty() {
        vec![]
    } else {
        text.split('x').map(|d| d.parse().unwrap()).collect()
    }
}

fn broadcast(c: &mut Criterion) {
    let mut group = c.benchmark_group("BroadcastBinaryDoubleOperations");
    let mut rng = KeanuRng::seed_from_u64(1);
    for dims in [
        ",2x100",
        "2x100,",
        "1x1,2x10",
        "2x10,1x1",
        "1x10,2x10",
        "2x10,1x10",
        "2x2x2,2x2",
        "2x2,2x2x2",
        "20x200,20x20x200",
        "20x1x20,20x20x20",
    ] {
        let (l, r) = dims.split_once(',').unwrap();
        let left = rng.gaussian(&shape(l));
        let right = rng.gaussian(&shape(r));
        group.bench_with_input(BenchmarkId::new("TIMES", dims), &(), |b, _| {
            b.iter(|| black_box(left.mul(&right).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("DIVIDE", dims), &(), |b, _| {
            b.iter(|| black_box(left.div(&right).unwrap()))
        });
    }
    group.finish();
}

fn elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("ElementwiseBinaryDoubleOperations");
    // Java also runs 10240; a 10240² matrix multiply takes minutes per iteration, so it's left out of both runs
    for n in [1usize, 10, 100, 1000] {
        let left = DoubleTensor::arange(0.0, (n * n) as f64)
            .reshape(&[n as isize, n as isize])
            .unwrap();
        let right = DoubleTensor::arange((n * n) as f64, (2 * n * n) as f64)
            .reshape(&[n as isize, n as isize])
            .unwrap();
        group.bench_with_input(BenchmarkId::new("TIMES", n), &(), |b, _| {
            b.iter(|| black_box(left.mul(&right).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("MATRIX_MULTIPLY", n), &(), |b, _| {
            b.iter(|| black_box(left.matmul(&right).unwrap()))
        });
    }
    group.finish();
}

type UnaryOp<'a> = Box<dyn Fn() -> DoubleTensor + 'a>;

fn unary(c: &mut Criterion) {
    // each iteration applies the operation 100 times, as the Java benchmark does
    let mut group = c.benchmark_group("UnaryDoubleOperations");
    for n in [2usize, 5, 10, 50] {
        let mut rng = KeanuRng::seed_from_u64(1);
        let a = rng.uniform(&[n, n]);
        let tensor = &a.matmul(&a.transpose().unwrap()).unwrap() * 0.5;
        // the Java benchmark's matrix is only positive semi-definite for some sizes; add to the diagonal so the
        // Cholesky decomposition always succeeds (the cost is the same)
        let spd = tensor.add(&DoubleTensor::eye(n)).unwrap();
        let ops: [(&str, UnaryOp); 6] = [
            ("CHOLESKY", Box::new(|| spd.cholesky().unwrap())),
            ("DETERMINANT", Box::new(|| tensor.determinant().unwrap())),
            ("COS", Box::new(|| tensor.cos())),
            ("INVERSE", Box::new(|| spd.inverse().unwrap())),
            ("LOG", Box::new(|| tensor.ln())),
            ("TRANSPOSE", Box::new(|| tensor.transpose().unwrap())),
        ];
        for (name, op) in &ops {
            group.bench_with_input(BenchmarkId::new(*name, n), &(), |b, _| {
                b.iter(|| {
                    let mut result = None;
                    for _ in 0..100 {
                        result = Some(op());
                    }
                    black_box(result)
                })
            });
        }
    }
    group.finish();
}

fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(2))
        .sample_size(20)
}

criterion_group!(name = benches; config = config(); targets = broadcast, elementwise, unary);
criterion_main!(benches);
