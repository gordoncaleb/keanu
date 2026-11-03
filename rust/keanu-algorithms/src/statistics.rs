//! Statistics of sample chains: autocorrelation, effective sample size and summaries.

/// The autocorrelation of a series at lags 0, 1, ..., n - 1: the autocovariance at each lag (with a 1/n
/// normalisation) divided by the variance. Computed with an FFT of the mean-centred, zero-padded series.
///
/// A constant series has zero variance, so its autocorrelation is NaN (as in Java). An empty series gives an
/// empty result.
pub fn autocorrelation(series: &[f64]) -> Vec<f64> {
    let n = series.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = series.iter().sum::<f64>() / n as f64;
    // zero padding to at least 2n + 1 stops the circular correlation wrapping around
    let size = (2 * n + 1).next_power_of_two();
    let mut re: Vec<f64> = series
        .iter()
        .map(|x| x - mean)
        .chain(std::iter::repeat_n(0.0, size - n))
        .collect();
    let mut im = vec![0.0; size];

    fft(&mut re, &mut im, false);
    // multiply by the complex conjugate: |z|^2
    for (r, i) in re.iter_mut().zip(im.iter_mut()) {
        *r = *r * *r + *i * *i;
        *i = 0.0;
    }
    fft(&mut re, &mut im, true);

    let autocovariance: Vec<f64> = re[..n].iter().map(|x| x / size as f64 / n as f64).collect();
    let variance = autocovariance[0];
    autocovariance.iter().map(|x| x / variance).collect()
}

/// In-place iterative radix-2 Cooley–Tukey FFT. `inverse` uses the conjugate twiddles and does not scale.
fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    // bit reversal permutation
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let angle = sign * 2.0 * std::f64::consts::PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (s, c) = (angle * k as f64).sin_cos();
                let (a, b) = (start + k, start + k + len / 2);
                let tr = re[b] * c - im[b] * s;
                let ti = re[b] * s + im[b] * c;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

/// The effective sample size of a series, using Geyer's initial monotone sequence estimator: sums of
/// consecutive pairs of autocorrelations are accumulated while they stay positive, and forced to be
/// non-increasing. The result is n / τ with τ = -1 + 2 Σ Γₖ, capped at n.
///
/// Returns 0 for an empty series and n for a constant one.
pub fn effective_sample_size(series: &[f64]) -> f64 {
    let n = series.len();
    if n == 0 {
        return 0.0;
    }
    let rho = autocorrelation(series);
    if rho[0].is_nan() {
        return n as f64;
    }
    let mut sum = 0.0;
    let mut previous = f64::INFINITY;
    let mut k = 0;
    while k + 1 < n {
        let pair = rho[k] + rho[k + 1];
        if pair <= 0.0 {
            break;
        }
        let pair = pair.min(previous);
        sum += pair;
        previous = pair;
        k += 2;
    }
    let tau = -1.0 + 2.0 * sum;
    if tau <= 0.0 {
        n as f64
    } else {
        (n as f64 / tau).min(n as f64)
    }
}

/// Summary statistics of a series.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Summary {
    /// The number of values.
    pub count: usize,
    /// Their mean.
    pub mean: f64,
    /// Sample variance (n - 1 denominator).
    pub variance: f64,
    /// Their sample standard deviation.
    pub std_dev: f64,
    /// The smallest value.
    pub min: f64,
    /// The largest value.
    pub max: f64,
}

/// Count, mean, sample variance, standard deviation, min and max of a series. NaN statistics for an empty series
/// (and a NaN variance for a single value).
pub fn summarize(series: &[f64]) -> Summary {
    let n = series.len();
    let mean = series.iter().sum::<f64>() / n as f64;
    let variance = series.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n as f64 - 1.0);
    let (min, max) = if n == 0 {
        (f64::NAN, f64::NAN)
    } else {
        series
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                (lo.min(x), hi.max(x))
            })
    };
    Summary {
        count: n,
        mean,
        variance,
        std_dev: variance.sqrt(),
        min,
        max,
    }
}
