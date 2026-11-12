//! Large-n gradients through the checkpointed Poisson-binomial reverse sweep.

use keanu_distributions::PoissonBinomial;
use keanu_tensor::{DoubleTensor, IntTensor};

fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance * b.abs().max(1.0), "{a} vs {b}");
}

/// With identical trials the score has the closed form k / (n p) − (n − k) / (n (1 − p)).
#[test]
fn identical_trials_match_the_binomial_score() {
    // ⌈√5001⌉ = 71 does not divide 5001, so the final replayed block is short.
    let (n, k, q) = (5001_usize, 2400_i64, 0.47);
    let dist = PoissonBinomial::new(DoubleTensor::full(&[n], q)).unwrap();
    let gradient = dist.d_log_prob(&IntTensor::scalar(k)).unwrap().p;
    let (nf, kf) = (n as f64, k as f64);
    let expected = kf / (nf * q) - (nf - kf) / (nf * (1.0 - q));
    for &g in gradient.as_slice() {
        close(g, expected, 1e-10);
    }
}

#[test]
fn heterogeneous_trials_match_finite_differences() {
    let n = 5000;
    let p: Vec<f64> = (0..n)
        .map(|i| 0.05 + 0.9 * ((i * 37 + 11) % 997) as f64 / 997.0)
        .collect();
    let mean = p.iter().sum::<f64>().round() as i64;
    let x = IntTensor::scalar(mean);
    let k = IntTensor::scalar(mean - 40);
    let dist = PoissonBinomial::new(DoubleTensor::vector(p.clone())).unwrap();
    let mass = dist.d_log_prob(&x).unwrap().p;
    let cdf = dist.d_log_cdf(&k).unwrap().p;
    let ccdf = dist.d_log_ccdf(&k).unwrap().p;
    let h = 1e-6;
    // First, last, and coordinates on and either side of block boundaries (interval 71).
    for i in [0, 70, 71, 72, 2556, 4969, 4970, 4999] {
        let shifted = |delta: f64| {
            let mut q = p.clone();
            q[i] += delta;
            PoissonBinomial::new(DoubleTensor::vector(q)).unwrap()
        };
        let (up, down) = (shifted(h), shifted(-h));
        let fd = |f: &dyn Fn(&PoissonBinomial) -> f64| (f(&up) - f(&down)) / (2.0 * h);
        close(
            mass.as_slice()[i],
            fd(&|d| d.log_prob(&x).unwrap().as_slice()[0]),
            1e-6,
        );
        close(
            cdf.as_slice()[i],
            fd(&|d| d.log_cdf(&k).unwrap().as_slice()[0]),
            1e-6,
        );
        close(
            ccdf.as_slice()[i],
            fd(&|d| d.log_ccdf(&k).unwrap().as_slice()[0]),
            1e-6,
        );
    }
}

/// Batched rows share one workspace; each row must match its own unbatched gradient.
#[test]
fn batched_rows_match_individual_rows() {
    let rows: Vec<Vec<f64>> = (0..3)
        .map(|r| {
            (0..130)
                .map(|i| ((i * (r + 3) + r) % 29) as f64 / 29.0)
                .collect()
        })
        .collect();
    let counts = [5_i64, 64, 120];
    let batch = DoubleTensor::new(&[3, 130], rows.concat()).unwrap();
    let gradient = PoissonBinomial::new(&batch)
        .unwrap()
        .d_log_prob(&IntTensor::vector(counts.to_vec()))
        .unwrap()
        .p;
    for (r, (row, &count)) in rows.iter().zip(&counts).enumerate() {
        let single = PoissonBinomial::new(DoubleTensor::vector(row.clone()))
            .unwrap()
            .d_log_prob(&IntTensor::scalar(count))
            .unwrap()
            .p;
        assert_eq!(
            &gradient.as_slice()[r * 130..(r + 1) * 130],
            single.as_slice()
        );
    }
}
