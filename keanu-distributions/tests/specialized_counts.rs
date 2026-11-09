use keanu_distributions::{DiscreteWeibull, KeanuRng, PoissonBinomial};
use keanu_tensor::{DoubleTensor, IntTensor};

fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{a} vs {b}");
}

/// Enumerate the mass polynomial and differentiate its factors, including at zero or one.
fn enumerate(
    p: &[f64],
    matches: impl Fn(u32) -> bool,
    direction: &[f64],
) -> (f64, Vec<f64>, Vec<f64>) {
    let mut mass = 0.0;
    let mut gradient = vec![0.0; p.len()];
    let mut hvp = vec![0.0; p.len()];
    for mask in (0..1_usize << p.len()).filter(|mask| matches(mask.count_ones())) {
        let sign = |i| if (mask >> i) & 1 == 1 { 1.0 } else { -1.0 };
        let product = |skip, skip2| {
            p.iter()
                .enumerate()
                .filter(|&(i, _)| i != skip && i != skip2)
                .map(|(i, p)| if sign(i) > 0.0 { *p } else { 1.0 - p })
                .product::<f64>()
        };
        mass += product(p.len(), p.len());
        for (i, g) in gradient.iter_mut().enumerate() {
            *g += sign(i) * product(i, i);
            for (j, &t) in direction.iter().enumerate().filter(|&(j, _)| j != i) {
                hvp[i] += sign(i) * sign(j) * product(i, j) * t;
            }
        }
    }
    let score = gradient
        .iter()
        .zip(direction)
        .map(|(g, t)| g * t / mass)
        .sum::<f64>();
    for (g, h) in gradient.iter_mut().zip(&mut hvp) {
        *g /= mass;
        *h = *h / mass - *g * score;
    }
    (mass.ln(), gradient, hvp)
}

#[test]
fn poisson_binomial_matches_enumeration_and_derivatives() {
    let direction = [0.4, -0.2, 0.6, -0.1];
    let tangent = DoubleTensor::new(&[1, 4], direction.to_vec()).unwrap();
    for p in [[0.1, 0.4, 0.8, 0.3], [0.0, 0.4, 1.0, 0.3]] {
        let distribution = PoissonBinomial::new(DoubleTensor::vector(p.to_vec())).unwrap();
        for k in 1..=3 {
            let count = IntTensor::scalar(i64::from(k));
            let (lp, expected_g, expected_h) = enumerate(&p, |count| count == k, &direction);
            let (actual, gradient) = distribution.log_prob_and_gradient(&count).unwrap();
            close(actual.to_scalar().unwrap(), lp);
            for (&a, b) in gradient.p.as_slice().iter().zip(expected_g) {
                close(a, b);
            }
            for (&a, b) in distribution
                .hvp(&count, &tangent)
                .unwrap()
                .as_slice()
                .iter()
                .zip(expected_h)
            {
                close(a, b);
            }
        }
    }
}

#[test]
fn poisson_binomial_keeps_rare_masses_and_broadcasts_independent_draws() {
    let dist = PoissonBinomial::new(DoubleTensor::full(&[180], 0.001)).unwrap();
    let count = IntTensor::scalar(180);
    let lp = dist.log_prob(&count).unwrap().to_scalar().unwrap();
    close(lp, 180.0 * 0.001_f64.ln());
    assert_eq!(lp.exp(), 0.0);
    let p = DoubleTensor::new(&[2, 3], vec![0.0, 0.4, 1.0, 0.2, 0.5, 0.9]).unwrap();
    let dist = PoissonBinomial::new(&p).unwrap();
    let samples = dist
        .sample(&[4000, 2], &mut KeanuRng::seed_from_u64(4))
        .unwrap();
    assert_eq!(dist.log_prob(&samples).unwrap().shape(), &[4000, 2]);
    for (j, mean) in [1.4, 1.6].into_iter().enumerate() {
        let average = samples
            .as_slice()
            .chunks_exact(2)
            .map(|r| r[j])
            .sum::<i64>() as f64
            / 4000.0;
        assert!((average - mean).abs() < 0.04);
    }
    assert!(PoissonBinomial::new(DoubleTensor::zeros(&[0])).is_err());
    let invalid = PoissonBinomial::new(DoubleTensor::vector(vec![f64::NAN])).unwrap();
    assert!(
        invalid
            .log_prob(&IntTensor::scalar(0))
            .unwrap()
            .to_scalar()
            .unwrap()
            .is_nan()
    );
    assert!(
        invalid
            .sample(&[], &mut KeanuRng::seed_from_u64(5))
            .is_err()
    );
}

#[test]
fn bounded_poisson_binomial_matches_enumerated_mass_and_derivatives() {
    let direction = [0.4, -0.2, 0.6, -0.1, 0.5];
    let tangent = DoubleTensor::new(&[1, 5], direction.to_vec()).unwrap();
    let (lo, hi) = (2, 4);
    let lower = DoubleTensor::scalar(f64::from(lo));
    let upper = DoubleTensor::scalar(f64::from(hi));
    for p in [[0.1, 0.4, 0.8, 0.3, 0.9], [0.0, 0.4, 1.0, 0.3, 0.9]] {
        let dist = PoissonBinomial::new(DoubleTensor::vector(p.to_vec())).unwrap();
        let truncated = dist.truncated(&lower, &upper);
        let censored = dist.censored(&lower, &upper);
        let (log_z, dz, hz) = enumerate(&p, |c| (lo..=hi).contains(&c), &direction);
        for k in lo..=hi {
            let count = IntTensor::scalar(i64::from(k));
            let (mass, dm, hm) = enumerate(&p, |c| c == k, &direction);
            let (clipped, dc, hc) = enumerate(&p, |c| c.clamp(lo, hi) == k, &direction);
            let (actual, gt) = truncated.log_prob_and_gradient(&count).unwrap();
            let (clip_actual, gc) = censored.log_prob_and_gradient(&count).unwrap();
            close(actual.to_scalar().unwrap(), mass - log_z);
            close(clip_actual.to_scalar().unwrap(), clipped);
            let ht = truncated.hvp(&count, &tangent).unwrap();
            let hclip = censored.hvp(&count, &tangent).unwrap();
            for i in 0..p.len() {
                close(gt.p.as_slice()[i], dm[i] - dz[i]);
                close(ht.as_slice()[i], hm[i] - hz[i]);
                close(gc.p.as_slice()[i], dc[i]);
                close(hclip.as_slice()[i], hc[i]);
            }
        }
        assert_eq!(
            truncated
                .log_prob(&IntTensor::scalar(1))
                .unwrap()
                .as_slice(),
            &[f64::NEG_INFINITY]
        );
    }
}

#[test]
fn rare_poisson_binomial_truncation_scores_and_samples_a_conditional_law() {
    let (n, p) = (180, 0.001_f64);
    let d = PoissonBinomial::new(DoubleTensor::full(&[n], p)).unwrap();
    let lower = DoubleTensor::scalar(n as f64 - 1.0);
    let upper = DoubleTensor::scalar(f64::INFINITY);
    let t = d.truncated(&lower, &upper);
    let odds = p / (n as f64 * (1.0 - p));
    close(
        t.log_prob(&IntTensor::scalar(n as i64)).unwrap().as_slice()[0],
        (odds / (1.0 + odds)).ln(),
    );
    close(
        t.log_prob(&IntTensor::scalar(n as i64 - 1))
            .unwrap()
            .as_slice()[0],
        -odds.ln_1p(),
    );
    let samples = t.sample(&[32], &mut KeanuRng::seed_from_u64(72)).unwrap();
    assert!(
        samples
            .as_slice()
            .iter()
            .all(|&x| (n as i64 - 1..=n as i64).contains(&x))
    );

    // Force the log-space fallback on a rare interval with two appreciable conditional masses.
    let d = PoissonBinomial::new(DoubleTensor::vector(vec![0.1; 8])).unwrap();
    let lower = DoubleTensor::scalar(5.0);
    let upper = DoubleTensor::scalar(6.0);
    let samples = d
        .truncated(&lower, &upper)
        .sample(&[4000], &mut KeanuRng::seed_from_u64(73))
        .unwrap();
    let upper_fraction = samples.as_slice().iter().filter(|&&x| x == 6).count() as f64 / 4000.0;
    // The mass ratio is C(8,6)/C(8,5) times p/(1-p), i.e. 1/18.
    assert!(
        (upper_fraction - 1.0 / 19.0).abs() < 0.012,
        "{upper_fraction}"
    );

    let impossible = DoubleTensor::scalar(9.0);
    assert!(
        d.truncated(&impossible, &upper)
            .sample(&[], &mut KeanuRng::seed_from_u64(74))
            .is_err()
    );
    assert!(
        d.truncated(&impossible, &DoubleTensor::scalar(10.0))
            .sample(&[], &mut KeanuRng::seed_from_u64(75))
            .is_err()
    );
    let deterministic = PoissonBinomial::new(DoubleTensor::zeros(&[8])).unwrap();
    assert!(
        deterministic
            .truncated(&lower, &upper)
            .sample(&[], &mut KeanuRng::seed_from_u64(76))
            .is_err()
    );
}

#[test]
fn discrete_weibull_matches_closed_form_masses_and_tails() {
    let (q, beta) = (0.78_f64, 1.35);
    let dist = DiscreteWeibull::new(DoubleTensor::scalar(q), DoubleTensor::scalar(beta));
    let x = IntTensor::vector(vec![0, 1, 3, 10]);
    let mass = dist.log_prob(&x).unwrap();
    let cdf = dist.log_cdf(&x).unwrap();
    let survival = dist.log_ccdf(&x).unwrap();
    for (i, &k) in x.as_slice().iter().enumerate() {
        let k = k as f64;
        let at = q.powf(k.powf(beta));
        let after = q.powf((k + 1.0).powf(beta));
        close(mass.as_slice()[i], (at - after).ln());
        close(cdf.as_slice()[i], (-after).ln_1p());
        close(survival.as_slice()[i], after.ln());
    }
    assert_eq!(
        dist.log_prob(&IntTensor::scalar(-1)).unwrap().to_scalar(),
        Some(f64::NEG_INFINITY)
    );
    assert_eq!(
        dist.log_ccdf(&IntTensor::scalar(-1)).unwrap().to_scalar(),
        Some(0.0)
    );
    let samples = dist
        .sample(&[6000], &mut KeanuRng::seed_from_u64(8))
        .unwrap();
    let zero_fraction = samples.as_slice().iter().filter(|&&x| x == 0).count() as f64 / 6000.0;
    assert!((zero_fraction - (1.0 - q)).abs() < 0.02);
}

#[test]
fn discrete_weibull_does_not_subtract_rounded_equal_tails() {
    let q = 1.0 - f64::EPSILON;
    let dist = DiscreteWeibull::new(DoubleTensor::scalar(q), DoubleTensor::scalar(0.5));
    let k = i64::MAX as f64;
    let gap = 1.0 / (k.sqrt() + (k + 1.0).sqrt());
    let expected = q.ln() * k.sqrt() + (-(q.ln() * gap).exp_m1()).ln();
    close(
        dist.log_prob(&IntTensor::scalar(i64::MAX))
            .unwrap()
            .to_scalar()
            .unwrap(),
        expected,
    );
}

#[test]
fn poisson_binomial_tails_match_enumeration_and_derivatives() {
    let direction = [0.4, -0.2, 0.6, -0.1];
    let tangent = DoubleTensor::new(&[1, 4], direction.to_vec()).unwrap();
    for p in [[0.1, 0.4, 0.8, 0.3], [0.0, 0.4, 1.0, 0.3]] {
        let dist = PoissonBinomial::new(DoubleTensor::vector(p.to_vec())).unwrap();
        for k in [1, 2] {
            let threshold = IntTensor::scalar(i64::from(k));
            for upper in [false, true] {
                let (log, grad, hvp) = if upper {
                    (
                        dist.log_ccdf(&threshold),
                        dist.d_log_ccdf(&threshold),
                        dist.log_ccdf_hvp(&threshold, &tangent),
                    )
                } else {
                    (
                        dist.log_cdf(&threshold),
                        dist.d_log_cdf(&threshold),
                        dist.log_cdf_hvp(&threshold, &tangent),
                    )
                };
                let (value, gradient, product) =
                    enumerate(&p, |count| (count > k) == upper, &direction);
                close(log.unwrap().to_scalar().unwrap(), value);
                for (&a, b) in grad.unwrap().p.as_slice().iter().zip(gradient) {
                    close(a, b);
                }
                for (&a, b) in hvp.unwrap().as_slice().iter().zip(product) {
                    close(a, b);
                }
            }
        }
    }
}

#[test]
fn poisson_binomial_tails_remain_finite_below_probability_underflow() {
    let p = 0.001_f64;
    let n = 200;
    let high = PoissonBinomial::new(DoubleTensor::full(&[n], p)).unwrap();
    let threshold = IntTensor::scalar(n as i64 - 1);
    close(
        high.log_ccdf(&threshold).unwrap().to_scalar().unwrap(),
        n as f64 * p.ln(),
    );
    assert_eq!(
        high.log_cdf(&threshold).unwrap().to_scalar().unwrap().exp(),
        1.0
    );
    let low = PoissonBinomial::new(DoubleTensor::full(&[n], 1.0 - p)).unwrap();
    close(
        low.log_cdf(&IntTensor::scalar(0))
            .unwrap()
            .to_scalar()
            .unwrap(),
        n as f64 * (1.0 - (1.0 - p)).ln(),
    );
    assert_eq!(
        high.log_ccdf(&IntTensor::scalar(-1)).unwrap().to_scalar(),
        Some(0.0)
    );
    assert_eq!(
        high.log_ccdf(&IntTensor::scalar(n as i64))
            .unwrap()
            .to_scalar(),
        Some(f64::NEG_INFINITY)
    );
}
