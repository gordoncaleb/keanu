use keanu_algorithms::statistics::{autocorrelation, effective_sample_size, summarize};
use keanu_distributions::KeanuRng;

fn ar1(phi: f64, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = KeanuRng::seed_from_u64(seed);
    let mut x = 0.0;
    (0..n)
        .map(|_| {
            x = phi * x + rng.next_gaussian();
            x
        })
        .collect()
}

#[test]
fn fft_autocorrelation_matches_the_direct_sum() {
    let series = ar1(0.3, 257, 3);
    let n = series.len();
    let mean = series.iter().sum::<f64>() / n as f64;
    let acov = |k: usize| {
        (0..n - k)
            .map(|t| (series[t] - mean) * (series[t + k] - mean))
            .sum::<f64>()
            / n as f64
    };
    let direct: Vec<f64> = (0..n).map(|k| acov(k) / acov(0)).collect();
    let fft = autocorrelation(&series);
    for (a, b) in fft.iter().zip(&direct) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
    assert_eq!(fft[0], 1.0);
}

#[test]
fn ar1_autocorrelation_decays_geometrically() {
    let phi = 0.6;
    let rho = autocorrelation(&ar1(phi, 200_000, 7));
    for (k, &r) in rho.iter().enumerate().take(5).skip(1) {
        assert!((r - phi.powi(k as i32)).abs() < 0.02, "lag {k}: {}", r);
    }
}

#[test]
fn effective_sample_size_of_independent_and_correlated_chains() {
    let n = 100_000;
    let independent = effective_sample_size(&ar1(0.0, n, 11));
    assert!((independent / n as f64 - 1.0).abs() < 0.1, "{independent}");

    let phi = 0.6;
    let expected = n as f64 * (1.0 - phi) / (1.0 + phi);
    let correlated = effective_sample_size(&ar1(phi, n, 13));
    assert!(
        (correlated / expected - 1.0).abs() < 0.15,
        "{correlated} vs {expected}"
    );
}

#[test]
fn edge_cases() {
    assert!(autocorrelation(&[]).is_empty());
    assert!(autocorrelation(&[2.0, 2.0, 2.0]).iter().all(|x| x.is_nan()));
    assert_eq!(effective_sample_size(&[]), 0.0);
    assert_eq!(effective_sample_size(&[2.0, 2.0, 2.0]), 3.0);
}

#[test]
fn summary_statistics() {
    let s = summarize(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!((s.count, s.mean, s.min, s.max), (4, 2.5, 1.0, 4.0));
    assert!((s.variance - 5.0 / 3.0).abs() < 1e-15);
    assert!((s.std_dev - (5.0f64 / 3.0).sqrt()).abs() < 1e-15);
}
