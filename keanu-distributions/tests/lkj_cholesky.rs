//! The LKJ distribution over Cholesky factors of correlation matrices: its normalising constant against the
//! analytic K = 2 density and a quadrature for K = 3, its gradients against finite differences, and its onion
//! sampler against the known marginal distribution of each correlation.

use keanu_distributions::*;
use keanu_tensor::DoubleTensor;
use keanu_tensor::special::log_beta;

mod common;
use common::{assert_close_rel, s};

const H: f64 = 1e-6;

fn matrix(k: usize, values: &[f64]) -> DoubleTensor {
    DoubleTensor::new(&[k, k], values.to_vec()).unwrap()
}

/// The correlation Cholesky factor of a 2 × 2 correlation matrix with correlation r.
fn factor_2(r: f64) -> DoubleTensor {
    matrix(2, &[1.0, 0.0, r, (1.0 - r * r).sqrt()])
}

/// The correlation Cholesky factor of each canonical partial correlation in `z` (row-major strictly lower
/// triangle), as the tanh transform builds it: `Lᵢⱼ = zᵢⱼ √(1 − Σₘ₍ⱼ Lᵢₘ²)`, `Lᵢᵢ = √(1 − Σₘ₍ᵢ Lᵢₘ²)`.
fn factor_from_partial_correlations(k: usize, z: &[f64]) -> Vec<f64> {
    let mut l = vec![0.0; k * k];
    let mut next = 0;
    for i in 0..k {
        let mut remaining = 1.0_f64;
        for j in 0..i {
            l[i * k + j] = z[next] * remaining.sqrt();
            remaining -= l[i * k + j] * l[i * k + j];
            next += 1;
        }
        l[i * k + i] = remaining.sqrt();
    }
    l
}

/// A valid K = 4 factor from fixed partial correlations.
fn factor_4() -> DoubleTensor {
    matrix(
        4,
        &factor_from_partial_correlations(4, &[0.3, -0.5, 0.2, 0.6, -0.1, 0.4]),
    )
}

#[test]
fn k2_log_density_is_the_analytic_correlation_density() {
    // For K = 2 the free coordinate of L is r = L[1, 0] itself (L[1, 1] = √(1 − r²) is determined), so the
    // Jacobian between L and r is 1 and the density of L is the density of r: (1 − r²)^(η − 1) / (2^(2η − 1) B(η, η))
    // on (−1, 1).
    for eta in [0.5, 1.0, 2.5, 7.0] {
        for r in [-0.9_f64, -0.3, 0.0, 0.45, 0.99] {
            let expected = (eta - 1.0) * (1.0 - r * r).ln()
                - (2.0 * eta - 1.0) * 2f64.ln()
                - log_beta(eta, eta);
            let actual = LkjCholesky::new(s(eta)).log_prob(&factor_2(r)).unwrap();
            assert_close_rel(actual, expected, 1e-12, &format!("eta {eta}, r {r}"));
        }
    }
}

#[test]
fn k3_density_integrates_to_one() {
    // Integrate over the unconstrained coordinates y of the tanh transform, z = tanh y, where the integrand decays
    // like e^(−2|y|) and the trapezoid rule converges geometrically. The Jacobian from y to the strictly lower
    // triangle of L is Π (1 − zᵢⱼ²) (from z = tanh y) times √(1 − z₂₀²) (from L₂₁ = z₂₁ √(1 − z₂₀²)).
    let (half_width, points) = (10.0, 121);
    let h = 2.0 * half_width / (points - 1) as f64;
    let grid: Vec<f64> = (0..points).map(|i| -half_width + i as f64 * h).collect();
    for eta in [1.0, 1.7, 4.0] {
        let lkj = LkjCholesky::new(s(eta));
        let mut integral = 0.0;
        for &y10 in &grid {
            for &y20 in &grid {
                for &y21 in &grid {
                    let z = [y10.tanh(), y20.tanh(), y21.tanh()];
                    let l = matrix(3, &factor_from_partial_correlations(3, &z));
                    let jacobian =
                        z.iter().map(|z| 1.0 - z * z).product::<f64>() * (1.0 - z[1] * z[1]).sqrt();
                    integral += (lkj.log_prob(&l).unwrap()).exp() * jacobian;
                }
            }
        }
        integral *= h * h * h;
        assert_close_rel(integral, 1.0, 1e-6, &format!("eta {eta}"));
    }
}

#[test]
fn k1_is_the_point_mass_at_one() {
    let lkj = LkjCholesky::new(s(2.0));
    assert_eq!(lkj.log_prob(&matrix(1, &[1.0])).unwrap(), 0.0);
    let sample = lkj
        .sample(&[1, 1], &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    assert_eq!(sample.as_slice(), &[1.0]);
}

/// Central differences of `log_prob` at `at`, element by element, against `analytic`, for the listed elements.
fn check(
    name: &str,
    at: &DoubleTensor,
    analytic: &DoubleTensor,
    elements: impl Iterator<Item = usize>,
    log_prob: impl Fn(&DoubleTensor) -> f64,
) {
    assert_eq!(analytic.shape(), at.shape(), "{name}: gradient shape");
    for i in elements {
        let mut up = at.as_slice().to_vec();
        let mut down = up.clone();
        up[i] += H;
        down[i] -= H;
        let numeric = (log_prob(&DoubleTensor::new(at.shape(), up).unwrap())
            - log_prob(&DoubleTensor::new(at.shape(), down).unwrap()))
            / (2.0 * H);
        assert_close_rel(
            analytic.as_slice()[i],
            numeric,
            1e-6,
            &format!("{name}[{i}]"),
        );
    }
}

#[test]
fn gradients_match_finite_differences() {
    let x = factor_4();
    for eta in [0.6, 1.0, 3.2] {
        let g = LkjCholesky::new(s(eta)).d_log_prob(&x).unwrap();
        check(&format!("eta {eta}"), &s(eta), &g.eta, 0..1, |e| {
            LkjCholesky::new(e.clone()).log_prob(&x).unwrap()
        });
        // the lower triangle (perturbing the upper triangle leaves the support); the density depends on the
        // diagonal only, so the rest of the gradient is zero
        let lower = (0..16).filter(|i| i % 4 <= i / 4);
        check(&format!("x, eta {eta}"), &x, &g.x, lower, |x| {
            LkjCholesky::new(s(eta)).log_prob(x).unwrap()
        });
        for i in (0..16).filter(|i| i % 4 > i / 4) {
            assert_eq!(g.x.as_slice()[i], 0.0);
        }
    }
}

#[test]
fn batches_sum_and_reduce_the_eta_gradient_to_its_shape() {
    let a = factor_4();
    let b = matrix(
        4,
        &factor_from_partial_correlations(4, &[-0.7, 0.1, 0.3, -0.2, 0.5, 0.8]),
    );
    let batch = DoubleTensor::concat(
        0,
        &[
            &a.reshape(&[1, 4, 4]).unwrap(),
            &b.reshape(&[1, 4, 4]).unwrap(),
        ],
    )
    .unwrap();
    let separate = |eta_a: f64, eta_b: f64| {
        LkjCholesky::new(s(eta_a)).log_prob(&a).unwrap()
            + LkjCholesky::new(s(eta_b)).log_prob(&b).unwrap()
    };

    // a scalar eta is shared by every matrix; its gradient is the sum over the batch
    let shared = LkjCholesky::new(s(1.5));
    assert_close_rel(
        shared.log_prob(&batch).unwrap(),
        separate(1.5, 1.5),
        1e-12,
        "shared eta",
    );
    let g = shared.d_log_prob(&batch).unwrap();
    assert_eq!(g.eta.shape(), &[] as &[usize]);
    assert_eq!(g.x.shape(), &[2, 4, 4]);
    check("shared eta", &s(1.5), &g.eta, 0..1, |e| {
        LkjCholesky::new(e.clone()).log_prob(&batch).unwrap()
    });

    // one eta per matrix
    let etas = DoubleTensor::vector(vec![0.8, 2.5]);
    let each = LkjCholesky::new(etas.clone());
    assert_close_rel(
        each.log_prob(&batch).unwrap(),
        separate(0.8, 2.5),
        1e-12,
        "eta per matrix",
    );
    let g = each.d_log_prob(&batch).unwrap();
    check("eta per matrix", &etas, &g.eta, 0..2, |e| {
        LkjCholesky::new(e.clone()).log_prob(&batch).unwrap()
    });
    let lower = (0..32).filter(|i| i % 4 <= (i % 16) / 4);
    check("x per matrix", &batch, &g.x, lower, |x| {
        LkjCholesky::new(etas.clone()).log_prob(x).unwrap()
    });
}

#[test]
fn outside_the_support_and_invalid_parameters() {
    let lkj = LkjCholesky::new(s(2.0));
    // a non-positive diagonal, a row that is not unit length, a non-zero upper triangle
    for bad in [
        matrix(2, &[1.0, 0.0, 0.6, -0.8]),
        matrix(2, &[1.0, 0.0, 0.6, 0.9]),
        matrix(2, &[1.0, 0.1, 0.6, 0.8]),
    ] {
        assert_eq!(lkj.log_prob(&bad).unwrap(), f64::NEG_INFINITY, "{bad:?}");
    }
    // a non-positive or NaN eta has no density
    for eta in [0.0, -1.0, f64::NAN] {
        assert!(
            LkjCholesky::new(s(eta))
                .log_prob(&factor_2(0.3))
                .unwrap()
                .is_nan()
        );
        assert!(matches!(
            LkjCholesky::new(s(eta)).sample(&[2, 2], &mut KeanuRng::seed_from_u64(1)),
            Err(DistributionError::InvalidParameter(_))
        ));
    }
    // values and sample shapes that are not square matrices, an empty matrix, and an eta that doesn't broadcast
    // to the batch
    let mut rng = KeanuRng::seed_from_u64(1);
    for shape in [&[3][..], &[2, 3], &[0, 0], &[]] {
        let x = DoubleTensor::new(shape, vec![0.0; shape.iter().product()]).unwrap();
        assert!(lkj.log_prob(&x).is_err(), "{shape:?}");
        assert!(lkj.d_log_prob(&x).is_err(), "{shape:?}");
        assert!(lkj.sample(shape, &mut rng).is_err(), "{shape:?}");
    }
    let three = LkjCholesky::new(DoubleTensor::vector(vec![1.0, 2.0, 3.0]));
    assert!(three.log_prob(&factor_2(0.1)).is_err());
    assert!(three.sample(&[2, 2, 2], &mut rng).is_err());
}

/// The off-diagonal correlations Ω = L Lᵀ of each sampled K × K factor, and checks each factor is a valid
/// correlation Cholesky factor.
fn correlations(samples: &DoubleTensor, k: usize) -> Vec<Vec<f64>> {
    let mut by_pair = vec![Vec::new(); k * (k - 1) / 2];
    for l in samples.as_slice().chunks(k * k) {
        for i in 0..k {
            let row = &l[i * k..(i + 1) * k];
            assert!(row[i] > 0.0, "diagonal {row:?}");
            assert!(row[i + 1..].iter().all(|&x| x == 0.0), "upper {row:?}");
            let length: f64 = row.iter().map(|x| x * x).sum();
            assert!((length - 1.0).abs() < 1e-12, "row length {length}");
        }
        let mut pair = 0;
        for i in 1..k {
            for j in 0..i {
                let r: f64 = (0..k).map(|m| l[i * k + m] * l[j * k + m]).sum();
                by_pair[pair].push(r);
                pair += 1;
            }
        }
    }
    by_pair
}

#[test]
fn eta_1_k_2_correlations_are_uniform() {
    let n = 100_000;
    let samples = LkjCholesky::new(s(1.0))
        .sample(&[n, 2, 2], &mut KeanuRng::seed_from_u64(42))
        .unwrap();
    let mut r = correlations(&samples, 2).remove(0);
    // Kolmogorov–Smirnov against Uniform(−1, 1) at the 1% level
    r.sort_by(f64::total_cmp);
    let count = r.len() as f64;
    let d = r
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let f = (x + 1.0) / 2.0;
            (f - i as f64 / count)
                .abs()
                .max(((i + 1) as f64 / count - f).abs())
        })
        .fold(0.0, f64::max);
    assert!(d < 1.63 / count.sqrt(), "KS statistic {d}");
}

#[test]
fn sampled_correlations_have_the_lkj_marginal() {
    // Lewandowski, Kurowicka & Joe (2009), section 3.2: under LKJ(η) in dimension K every off-diagonal correlation
    // has (r + 1) / 2 ~ Beta(β, β) with β = η − 1 + K / 2. So r has mean 0 and variance
    // 4 · β² / ((2β)² (2β + 1)) = 1 / (2β + 1) = 1 / (2η + K − 1), and fourth moment
    // 3 / ((2β + 1)(2β + 3)) (the kurtosis distinguishes the onion construction from a merely right variance).
    let n = 100_000;
    for (eta, k) in [(2.0, 3), (0.7, 4), (5.0, 5)] {
        let samples = LkjCholesky::new(s(eta))
            .sample(&[n, k, k], &mut KeanuRng::seed_from_u64(7))
            .unwrap();
        let beta = eta - 1.0 + k as f64 / 2.0;
        let variance = 1.0 / (2.0 * beta + 1.0);
        let fourth = 3.0 / ((2.0 * beta + 1.0) * (2.0 * beta + 3.0));
        for (pair, r) in correlations(&samples, k).iter().enumerate() {
            let mean = r.iter().sum::<f64>() / n as f64;
            let m2 = r.iter().map(|r| r * r).sum::<f64>() / n as f64;
            let m4 = r.iter().map(|r| r.powi(4)).sum::<f64>() / n as f64;
            // 5 standard errors: Var(r) / n for the mean, Var(r²) / n for the second moment and Var(r⁴) / n
            // bounded by E[r⁸] ≤ E[r⁴] for the fourth
            let what = format!("eta {eta}, K {k}, pair {pair}");
            assert!(
                mean.abs() < 5.0 * (variance / n as f64).sqrt(),
                "{what}: mean {mean}"
            );
            assert!(
                (m2 - variance).abs() < 5.0 * ((fourth - variance * variance) / n as f64).sqrt(),
                "{what}: variance {m2} vs {variance}"
            );
            assert!(
                (m4 - fourth).abs() < 5.0 * (fourth / n as f64).sqrt(),
                "{what}: fourth moment {m4} vs {fourth}"
            );
        }
    }
}

#[test]
fn eta_broadcasts_over_a_sample_batch() {
    let etas = DoubleTensor::vector(vec![1.0, 50.0]);
    let samples = LkjCholesky::new(etas)
        .sample(&[2, 3, 3], &mut KeanuRng::seed_from_u64(3))
        .unwrap();
    assert_eq!(samples.shape(), &[2, 3, 3]);
    correlations(&samples, 3);
}
