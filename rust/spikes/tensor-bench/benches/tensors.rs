use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

fn data(n: usize, seed: u64) -> Vec<f64> {
    // deterministic pseudo-random values in [-1, 1)
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn spd(n: usize) -> Vec<f64> {
    let a = data(n * n, 7);
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = (0..n).map(|k| a[i * n + k] * a[j * n + k]).sum::<f64>() + if i == j { n as f64 } else { 0.0 };
        }
    }
    out
}

macro_rules! suite {
    ($c:expr, $label:literal, $T:ty) => {{
        type T = $T;
        let big = T::new(&[1000, 1000], data(1_000_000, 1)).unwrap();
        let big2 = T::new(&[1000, 1000], data(1_000_000, 2)).unwrap();
        let row = T::new(&[1000], data(1000, 3)).unwrap();
        let cube = T::new(&[100, 100, 100], data(1_000_000, 4)).unwrap();
        let m256 = T::new(&[256, 256], data(65536, 5)).unwrap();
        let m256b = T::new(&[256, 256], data(65536, 6)).unwrap();
        let small_batch = T::new(&[1000, 4, 4], data(16000, 8)).unwrap();
        let small = T::new(&[4, 4], data(16, 9)).unwrap();
        let chol = T::new(&[200, 200], spd(200)).unwrap();
        let scalars: Vec<T> = (0..1000).map(|i| T::scalar(i as f64 * 0.001)).collect();
        let tiny: Vec<T> = (0..1000).map(|i| T::new(&[3], vec![i as f64, 1.0, 2.0]).unwrap()).collect();

        $c.bench_with_input(BenchmarkId::new("broadcast_add_1000x1000_by_1000", $label), &(), |b, _| b.iter(|| black_box(big.plus(&row).unwrap())));
        $c.bench_with_input(BenchmarkId::new("times_1000x1000", $label), &(), |b, _| b.iter(|| black_box(big.times(&big2).unwrap())));
        $c.bench_with_input(BenchmarkId::new("exp_1000x1000", $label), &(), |b, _| b.iter(|| black_box(big.map(f64::exp))));
        $c.bench_with_input(BenchmarkId::new("sum_axis0_1000x1000", $label), &(), |b, _| b.iter(|| black_box(big.sum_dims(&[0]).unwrap())));
        $c.bench_with_input(BenchmarkId::new("cumsum_axis1_1000x1000", $label), &(), |b, _| b.iter(|| black_box(big.cum_sum(1).unwrap())));
        $c.bench_with_input(BenchmarkId::new("slice_reverse_step_1000x1000", $label), &(), |b, _| b.iter(|| black_box(big.slice("::-1, ::2").unwrap())));
        $c.bench_with_input(BenchmarkId::new("permute_100^3", $label), &(), |b, _| b.iter(|| black_box(cube.permute(&[2, 0, 1]).unwrap())));
        $c.bench_with_input(BenchmarkId::new("matmul_256", $label), &(), |b, _| b.iter(|| black_box(m256.matmul(&m256b, false, false).unwrap())));
        $c.bench_with_input(BenchmarkId::new("batched_matmul_1000x4x4", $label), &(), |b, _| b.iter(|| black_box(small_batch.matmul(&small, false, false).unwrap())));
        $c.bench_with_input(BenchmarkId::new("cholesky_200", $label), &(), |b, _| b.iter(|| black_box(chol.cholesky().unwrap())));
        $c.bench_with_input(BenchmarkId::new("1000_scalar_ops", $label), &(), |b, _| b.iter(|| {
            let mut acc = T::scalar(0.0);
            for s in &scalars { acc = acc.plus(s).unwrap().times(s).unwrap(); }
            black_box(acc)
        }));
        $c.bench_with_input(BenchmarkId::new("1000_tiny_vector_ops", $label), &(), |b, _| b.iter(|| {
            let mut acc = T::new(&[3], vec![0.0; 3]).unwrap();
            for t in &tiny { acc = acc.plus(t).unwrap().map(f64::sin); }
            black_box(acc)
        }));
    }};
}

fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor");
    group.sample_size(20).measurement_time(std::time::Duration::from_secs(2)).warm_up_time(std::time::Duration::from_millis(500));
    suite!(group, "inhouse", tensor_inhouse::Tensor);
    suite!(group, "ndarray", keanu_tensor_ndarray_spike::Tensor);
    group.finish();
}

criterion_group!(tensor_benches, benches);
criterion_main!(tensor_benches);
