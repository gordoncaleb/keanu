//! The discrete and multivariate families of the distribution tail: the ordered probit, hypergeometric, Skellam,
//! Dirichlet-multinomial and Plackett–Luce probabilities sum to 1; the multivariate Student-t, Wishart, inverse
//! Wishart (and their Cholesky forms), matrix normal and von Mises–Fisher densities match dense closed forms (and
//! the von Mises–Fisher normalises); gradients and Hessian-vector products match finite differences; samplers match
//! their moments.

use keanu_distributions::*;
use keanu_tensor::{DoubleTensor, IntTensor};

mod common;
use common::{assert_close, assert_close_rel, s, t, v};

/// Central differences of `f` at `inputs` with respect to input `which`, entry `index` (and its mirror entry when
/// the input is a symmetric `k` × `k` matrix, `symmetric` = Some(k)).
fn finite_difference(
    f: &dyn Fn(&[DoubleTensor]) -> f64,
    inputs: &[DoubleTensor],
    which: usize,
    index: usize,
    symmetric: Option<usize>,
) -> f64 {
    let h = 1e-6 * inputs[which].as_slice()[index].abs().max(1.0);
    let shifted = |sign: f64| {
        let mut moved = inputs.to_vec();
        let mut data = moved[which].as_slice().to_vec();
        data[index] += sign * h;
        if let Some(k) = symmetric {
            let (i, j) = ((index % (k * k)) / k, index % k);
            if i != j {
                data[index - (i * k + j) + j * k + i] += sign * h;
            }
        }
        moved[which] = DoubleTensor::new(inputs[which].shape(), data).unwrap();
        f(&moved)
    };
    (shifted(1.0) - shifted(-1.0)) / (2.0 * h)
}

/// Asserts that `gradient[which]` matches finite differences of `f` in every entry (for a symmetric input the sum
/// of the two mirrored entries, the derivative along a symmetric perturbation).
fn check_gradient(
    name: &str,
    f: &dyn Fn(&[DoubleTensor]) -> f64,
    inputs: &[DoubleTensor],
    gradient: &[DoubleTensor],
    symmetric: &[Option<usize>],
    skip: &[usize],
) {
    check_gradient_lower(name, f, inputs, gradient, symmetric, skip, &[]);
}

/// [`check_gradient`], skipping the upper triangles of the `lower` inputs (Cholesky factors whose upper triangle
/// must stay zero).
fn check_gradient_lower(
    name: &str,
    f: &dyn Fn(&[DoubleTensor]) -> f64,
    inputs: &[DoubleTensor],
    gradient: &[DoubleTensor],
    symmetric: &[Option<usize>],
    skip: &[usize],
    lower: &[usize],
) {
    for (which, g) in gradient.iter().enumerate() {
        if skip.contains(&which) {
            continue;
        }
        let k = inputs[which].shape().last().copied().unwrap_or(1);
        let g = g.as_slice();
        for index in 0..g.len() {
            if lower.contains(&which) && index % k > (index / k) % k {
                continue;
            }
            let expected = finite_difference(f, inputs, which, index, symmetric[which]);
            let actual = match symmetric[which] {
                Some(k) => {
                    let (i, j) = ((index % (k * k)) / k, index % k);
                    if i == j {
                        g[index]
                    } else {
                        g[index] + g[index - (i * k + j) + j * k + i]
                    }
                }
                None => g[index],
            };
            assert_close_rel(
                actual,
                expected,
                2e-5,
                &format!("{name}: d/d input {which} [{index}]"),
            );
        }
    }
}

/// A direction for each input (symmetric for symmetric inputs, lower triangular for `lower` inputs).
fn directions(
    inputs: &[DoubleTensor],
    symmetric: &[Option<usize>],
    lower: &[usize],
) -> Vec<DoubleTensor> {
    inputs
        .iter()
        .enumerate()
        .map(|(which, input)| {
            let mut data: Vec<f64> = (0..input.len())
                .map(|i| 0.3 * ((i + 3 * which) as f64 * 1.37 + 0.4).sin())
                .collect();
            let k = input.shape().last().copied().unwrap_or(1);
            if let Some(k) = symmetric[which] {
                for m in 0..data.len() / (k * k) {
                    for i in 0..k {
                        for j in 0..i {
                            data[m * k * k + j * k + i] = data[m * k * k + i * k + j];
                        }
                    }
                }
            }
            if lower.contains(&which) {
                for m in 0..data.len() / (k * k) {
                    for i in 0..k {
                        for j in i + 1..k {
                            data[m * k * k + i * k + j] = 0.0;
                        }
                    }
                }
            }
            DoubleTensor::new(input.shape(), data).unwrap()
        })
        .collect()
}

/// Hessian-vector products of inputs along tangents.
type HvpFn<'a> = dyn Fn(&[DoubleTensor], &[DoubleTensor]) -> Vec<DoubleTensor> + 'a;

/// Asserts that the Hessian-vector products along `tangents` (one direction) match central differences of the
/// gradient along them.
fn check_hvp(
    name: &str,
    gradient: &dyn Fn(&[DoubleTensor]) -> Vec<DoubleTensor>,
    hvp: &HvpFn<'_>,
    inputs: &[DoubleTensor],
    tangents: &[DoubleTensor],
) {
    let h = 1e-6;
    let moved = |sign: f64| -> Vec<DoubleTensor> {
        inputs
            .iter()
            .zip(tangents)
            .map(|(x, t)| x.add(&(t * (sign * h))).unwrap())
            .collect()
    };
    let (plus, minus) = (gradient(&moved(1.0)), gradient(&moved(-1.0)));
    let products = hvp(inputs, tangents);
    for (which, ((p, m), product)) in plus.iter().zip(&minus).zip(&products).enumerate() {
        for (index, ((p, m), actual)) in p
            .as_slice()
            .iter()
            .zip(m.as_slice())
            .zip(product.as_slice())
            .enumerate()
        {
            let expected = (p - m) / (2.0 * h);
            assert_close_rel(
                *actual,
                expected,
                2e-5,
                &format!("{name}: HVP input {which} [{index}]"),
            );
        }
    }
}

/// [b = 1, shape...] tangents.
fn batched(tangents: &[DoubleTensor]) -> Vec<DoubleTensor> {
    tangents
        .iter()
        .map(|t| {
            let shape: Vec<isize> = std::iter::once(1)
                .chain(t.shape().iter().map(|&d| d as isize))
                .collect();
            t.reshape(&shape).unwrap()
        })
        .collect()
}

/// Strips the leading [1] of each Hessian-vector product.
fn unbatched(products: Vec<DoubleTensor>, inputs: &[DoubleTensor]) -> Vec<DoubleTensor> {
    products
        .into_iter()
        .zip(inputs)
        .map(|(p, x)| {
            let shape: Vec<isize> = x.shape().iter().map(|&d| d as isize).collect();
            p.reshape(&shape).unwrap()
        })
        .collect()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// A symmetric positive definite matrix from a lower triangular factor given row by row.
fn spd(k: usize, lower: &[f64]) -> DoubleTensor {
    let l = cholesky_factor(k, lower);
    l.matmul_transposed(&l, false, true).unwrap()
}

/// A K × K lower triangular matrix from its lower triangle, row by row.
fn cholesky_factor(k: usize, lower: &[f64]) -> DoubleTensor {
    let mut data = vec![0.0; k * k];
    let mut next = lower.iter();
    for i in 0..k {
        for j in 0..=i {
            data[i * k + j] = *next.next().unwrap();
        }
    }
    t(&[k, k], &data)
}

/// ln |A| of a symmetric positive definite matrix, from its Cholesky factor.
fn log_det(a: &DoubleTensor) -> f64 {
    2.0 * a.cholesky().unwrap().diag_part().unwrap().ln().sum()
}

/// A⁻¹ of a symmetric positive definite matrix.
fn inverse(a: &DoubleTensor) -> DoubleTensor {
    a.cholesky().unwrap().cholesky_inverse().unwrap()
}

/// tr(A B).
fn trace_product(a: &DoubleTensor, b: &DoubleTensor) -> f64 {
    a.matmul(b).unwrap().diag_part().unwrap().sum()
}

/// ln Γ_K(a).
fn log_multivariate_gamma(k: usize, a: f64) -> f64 {
    let pi = std::f64::consts::PI;
    k as f64 * (k as f64 - 1.0) / 4.0 * pi.ln()
        + (0..k)
            .map(|j| keanu_tensor::special::ln_gamma(a - j as f64 / 2.0))
            .sum::<f64>()
}

// ---------------------------------------------------------------------------------------------------------------
// Discrete families

#[test]
fn ordered_probit_probabilities_sum_to_one_and_are_normal_cdf_differences() {
    let cutpoints = v(&[-1.2, -0.1, 0.7, 2.0]);
    for eta in [-3.0, -0.4, 0.0, 0.9, 4.0] {
        let d = OrderedProbit::new(s(eta), &cutpoints);
        let k = IntTensor::vector((0..5).collect());
        let probabilities: Vec<f64> = d.log_prob(&k).unwrap().exp().into_vec();
        assert_close(probabilities.iter().sum(), 1.0, 1e-14, "sum");
        let phi = keanu_tensor::special::ndtr;
        let c = cutpoints.as_slice();
        let expected = [
            1.0 - phi(eta - c[0]),
            phi(eta - c[0]) - phi(eta - c[1]),
            phi(eta - c[1]) - phi(eta - c[2]),
            phi(eta - c[2]) - phi(eta - c[3]),
            phi(eta - c[3]),
        ];
        for (p, e) in probabilities.iter().zip(expected) {
            assert_close(*p, e, 1e-14, "probability");
        }
        let total: f64 = d.probabilities().unwrap().as_slice().iter().sum();
        assert_close(total, 1.0, 1e-14, "probabilities");
    }
    // far in either tail the middle categories keep their relative precision: Φ(a) − Φ(b) for a = −38, b = −39
    // is φ(−38)/38 to leading order, about e^−725.6
    let d = OrderedProbit::new(s(-38.0), v(&[0.0, 1.0]));
    let lp = d.log_prob(&IntTensor::vector(vec![1])).unwrap().as_slice()[0];
    let expected = keanu_tensor::special::log_ndtr(-38.0);
    assert_close_rel(lp, expected, 1e-12, "lower tail");
    let d = OrderedProbit::new(s(40.0), v(&[0.0, 1.0]));
    let lp = d.log_prob(&IntTensor::vector(vec![1])).unwrap().as_slice()[0];
    assert_close_rel(
        lp,
        keanu_tensor::special::log_ndtr(-39.0),
        1e-12,
        "upper tail",
    );
    // cutpoints that don't increase are impossible, and an index outside 0..K is too
    let d = OrderedProbit::new(s(0.0), v(&[1.0, 0.5]));
    assert_eq!(
        d.log_prob(&IntTensor::scalar(1)).unwrap().as_slice()[0],
        f64::NEG_INFINITY
    );
    let d = OrderedProbit::new(s(0.0), v(&[0.0, 0.5]));
    assert_eq!(
        d.log_prob(&IntTensor::scalar(3)).unwrap().as_slice()[0],
        f64::NEG_INFINITY
    );
}

#[test]
fn ordered_probit_derivatives_match_finite_differences() {
    let cutpoints = [-1.0, 0.3, 1.1];
    for eta in [-2.5, 0.2, 1.7] {
        for k in 0..=3 {
            let d = OrderedProbit::category_derivatives(k, eta, &cutpoints);
            // inputs (η, c_(k−1), c_k), the missing one at the ends not moving
            let f = |x: [f64; 3]| {
                let mut c = cutpoints;
                if k > 0 {
                    c[k as usize - 1] = x[1];
                }
                if k < 3 {
                    c[k as usize] = x[2];
                }
                OrderedProbit::category_derivatives(k, x[0], &c)
            };
            let lo = if k > 0 {
                cutpoints[k as usize - 1]
            } else {
                0.0
            };
            let hi = if k < 3 { cutpoints[k as usize] } else { 0.0 };
            let point = [eta, lo, hi];
            for i in 0..3 {
                let h = 1e-6;
                let (mut plus, mut minus) = (point, point);
                plus[i] += h;
                minus[i] -= h;
                let (fp, fm) = (f(plus), f(minus));
                assert_close_rel(
                    d.gradient[i],
                    (fp.value - fm.value) / (2.0 * h),
                    1e-6,
                    "gradient",
                );
                for j in 0..3 {
                    let fd = (fp.gradient[j] - fm.gradient[j]) / (2.0 * h);
                    assert_close_rel(d.hessian[i][j], fd, 1e-6, "hessian");
                }
            }
        }
    }
    // the tensor gradient agrees
    let d = OrderedProbit::new(v(&[0.4, -0.3]), v(&[-1.0, 0.3, 1.1]));
    let g = d.d_log_prob(&IntTensor::vector(vec![1, 3])).unwrap();
    let e = OrderedProbit::category_derivatives(1, 0.4, &[-1.0, 0.3, 1.1]);
    assert_close(g.eta.as_slice()[0], e.gradient[0], 1e-15, "eta");
    assert_close(g.cutpoints.as_slice()[0], e.gradient[1], 1e-15, "c0");
    assert_close(g.cutpoints.as_slice()[1], e.gradient[2], 1e-15, "c1");
}

#[test]
fn ordered_probit_samples_match_the_probabilities() {
    let d = OrderedProbit::new(s(0.3), v(&[-0.8, 0.2, 1.0]));
    let draws = d
        .sample(&[40_000], &mut KeanuRng::seed_from_u64(3))
        .unwrap();
    let probabilities = d.probabilities().unwrap();
    for (k, p) in probabilities.as_slice().iter().enumerate() {
        let f = draws.as_slice().iter().filter(|&&x| x == k as i64).count() as f64 / 40_000.0;
        assert_close(f, *p, 0.01, "frequency");
    }
}

#[test]
fn hypergeometric_probabilities_sum_to_one_with_the_right_mean() {
    for (n, k, d) in [
        (20, 7, 5),
        (50, 30, 40),
        (10, 0, 3),
        (10, 10, 10),
        (1, 1, 1),
    ] {
        let dist = HyperGeometric::new(
            IntTensor::scalar(n),
            IntTensor::scalar(k),
            IntTensor::scalar(d),
        );
        let range = HyperGeometric::range(n, k, d);
        let values: Vec<i64> = (range.start() - 2..=range.end() + 2).collect();
        let lp = dist.log_prob(&IntTensor::vector(values.clone())).unwrap();
        let p: Vec<f64> = lp.exp().into_vec();
        assert_close(p.iter().sum(), 1.0, 1e-12, "sum");
        let mean: f64 = values.iter().zip(&p).map(|(&x, p)| x as f64 * p).sum();
        assert_close(mean, d as f64 * k as f64 / n as f64, 1e-10, "mean");
        for (&x, &lp) in values.iter().zip(lp.as_slice()) {
            if !range.contains(&x) {
                assert_eq!(lp, f64::NEG_INFINITY);
            }
        }
        let draws = dist
            .sample(&[20_000], &mut KeanuRng::seed_from_u64(4))
            .unwrap();
        let sample_mean = draws.as_slice().iter().sum::<i64>() as f64 / 20_000.0;
        assert_close(sample_mean, mean, 0.05 * mean.max(1.0), "sample mean");
    }
    // invalid parameters: NaN, and an error when sampling
    let dist = HyperGeometric::new(
        IntTensor::scalar(5),
        IntTensor::scalar(6),
        IntTensor::scalar(2),
    );
    assert!(dist.log_prob(&IntTensor::scalar(1)).unwrap().as_slice()[0].is_nan());
    assert!(dist.sample(&[1], &mut KeanuRng::seed_from_u64(1)).is_err());
}

#[test]
fn skellam_is_the_difference_of_poissons() {
    let (mu1, mu2) = (2.3, 1.4);
    let d = Skellam::new(s(mu1), s(mu2));
    let values: Vec<i64> = (-40..=40).collect();
    let p: Vec<f64> = d
        .log_prob(&IntTensor::vector(values.clone()))
        .unwrap()
        .exp()
        .into_vec();
    assert_close(p.iter().sum(), 1.0, 1e-13, "sum");
    // against the convolution of the two Poisson mass functions
    let poisson = |mu: f64, k: i64| {
        (k as f64 * mu.ln() - mu - keanu_tensor::special::ln_gamma(k as f64 + 1.0)).exp()
    };
    for (&k, &pk) in values.iter().zip(&p).filter(|(k, _)| k.abs() <= 8) {
        let convolution: f64 = (0..80)
            .filter(|&j| j - k >= 0)
            .map(|j| poisson(mu1, j) * poisson(mu2, j - k))
            .sum();
        assert_close_rel(pk, convolution, 1e-12, "convolution");
    }
    let mean: f64 = values.iter().zip(&p).map(|(&k, p)| k as f64 * p).sum();
    let var: f64 = values
        .iter()
        .zip(&p)
        .map(|(&k, p)| (k as f64 - mean).powi(2) * p)
        .sum();
    assert_close(mean, mu1 - mu2, 1e-12, "mean");
    assert_close(var, mu1 + mu2, 1e-11, "variance");
    let draws = d
        .sample(&[40_000], &mut KeanuRng::seed_from_u64(5))
        .unwrap();
    let draws: Vec<f64> = draws.as_slice().iter().map(|&k| k as f64).collect();
    assert_close(mean_of(&draws), mu1 - mu2, 0.04, "sample mean");
    // large, nearly equal means stay finite and accurate
    let big = Skellam::new(s(5000.0), s(4990.0));
    let lp = big.log_prob(&IntTensor::scalar(10)).unwrap().as_slice()[0];
    // the mode's mass is close to the normal approximation 1/√(2π(μ₁ + μ₂))
    assert_close_rel(
        lp,
        -(2.0 * std::f64::consts::PI * 9990.0).sqrt().ln(),
        1e-3,
        "large means",
    );
}

fn mean_of(values: &[f64]) -> f64 {
    mean(values)
}

#[test]
fn skellam_derivatives_match_finite_differences() {
    for k in [-3, 0, 1, 5] {
        for (m1, m2) in [(0.7, 2.1), (3.5, 0.4), (1e-3, 2e-3)] {
            let d = Skellam::log_prob_derivatives(k as f64, [m1, m2]);
            let scale = d
                .hessian
                .iter()
                .flatten()
                .fold(1.0f64, |m, h| m.max(h.abs()));
            for i in 1..3 {
                let h = 1e-5 * [0.0, m1, m2][i];
                let at = |delta: f64| {
                    let mut p = [m1, m2];
                    p[i - 1] += delta;
                    Skellam::log_prob_derivatives(k as f64, p)
                };
                let (plus, minus) = (at(h), at(-h));
                assert_close_rel(
                    d.gradient[i],
                    (plus.value - minus.value) / (2.0 * h),
                    1e-6,
                    "gradient",
                );
                for j in 1..3 {
                    let fd = (plus.gradient[j] - minus.gradient[j]) / (2.0 * h);
                    assert_close(d.hessian[i][j], fd, 1e-6 * scale, "hessian");
                }
            }
            assert_eq!(d.gradient[0], 0.0);
        }
    }
    assert!(
        Skellam::new(s(-1.0), s(1.0))
            .log_prob(&IntTensor::scalar(0))
            .unwrap()
            .as_slice()[0]
            .is_nan()
    );
}

/// Every vector of `k` non-negative counts summing to `n`.
fn compositions(n: i64, k: usize) -> Vec<Vec<i64>> {
    if k == 1 {
        return vec![vec![n]];
    }
    (0..=n)
        .flat_map(|first| {
            compositions(n - first, k - 1)
                .into_iter()
                .map(move |mut rest| {
                    rest.insert(0, first);
                    rest
                })
        })
        .collect()
}

#[test]
fn dirichlet_multinomial_probabilities_sum_to_one() {
    let alpha = v(&[0.7, 2.0, 1.3]);
    let d = DirichletMultinomial::new(IntTensor::scalar(6), &alpha);
    let all = compositions(6, 3);
    let total: f64 = all
        .iter()
        .map(|x| d.log_prob(&IntTensor::vector(x.clone())).unwrap().exp())
        .sum();
    assert_close(total, 1.0, 1e-12, "sum");
    // n = 1 is the categorical with probabilities α/α₀
    let one = DirichletMultinomial::new(IntTensor::scalar(1), &alpha);
    let lp = one.log_prob(&IntTensor::vector(vec![0, 1, 0])).unwrap();
    assert_close(lp, (2.0f64 / 4.0).ln(), 1e-14, "n = 1");
    // counts that don't sum to n, or are negative, are impossible
    assert_eq!(
        d.log_prob(&IntTensor::vector(vec![1, 2, 2])).unwrap(),
        f64::NEG_INFINITY
    );
    assert_eq!(
        d.log_prob(&IntTensor::vector(vec![-1, 4, 3])).unwrap(),
        f64::NEG_INFINITY
    );
    // a batch sums its members, and n broadcasts
    let batch = DirichletMultinomial::new(IntTensor::vector(vec![6, 2]), &alpha);
    let x = IntTensor::new(&[2, 3], vec![1, 2, 3, 0, 2, 0]).unwrap();
    let two = DirichletMultinomial::new(IntTensor::scalar(2), &alpha);
    let expected = d.log_prob(&IntTensor::vector(vec![1, 2, 3])).unwrap()
        + two.log_prob(&IntTensor::vector(vec![0, 2, 0])).unwrap();
    assert_close(batch.log_prob(&x).unwrap(), expected, 1e-12, "batch");
    // the sampler's mean is nα/α₀
    let draws = d
        .sample(&[20_000, 3], &mut KeanuRng::seed_from_u64(6))
        .unwrap();
    for c in 0..3 {
        let m = draws.as_slice().iter().skip(c).step_by(3).sum::<i64>() as f64 / 20_000.0;
        assert_close(m, 6.0 * alpha.as_slice()[c] / 4.0, 0.05, "sample mean");
    }
    assert!(
        draws
            .as_slice()
            .chunks(3)
            .all(|x| x.iter().sum::<i64>() == 6)
    );
}

#[test]
fn dirichlet_multinomial_gradients_and_hvps_match_finite_differences() {
    let x = IntTensor::new(&[2, 3], vec![1, 2, 3, 0, 4, 2]).unwrap();
    let n = IntTensor::vector(vec![6, 6]);
    let alpha = t(&[2, 3], &[0.7, 2.0, 1.3, 1.1, 0.4, 3.0]);
    let f = |inputs: &[DoubleTensor]| {
        DirichletMultinomial::new(&n, &inputs[0])
            .log_prob(&x)
            .unwrap()
    };
    let g = DirichletMultinomial::new(&n, &alpha)
        .d_log_prob(&x)
        .unwrap();
    check_gradient(
        "dirichlet multinomial",
        &f,
        std::slice::from_ref(&alpha),
        &[g.alpha],
        &[None],
        &[],
    );
    let gradient = |inputs: &[DoubleTensor]| {
        vec![
            DirichletMultinomial::new(&n, &inputs[0])
                .d_log_prob(&x)
                .unwrap()
                .alpha,
        ]
    };
    let hvp = |inputs: &[DoubleTensor], tangents: &[DoubleTensor]| {
        let t = batched(tangents);
        let h = DirichletMultinomial::new(&n, &inputs[0])
            .hvp(&x, Some(&t[0]), 1)
            .unwrap();
        unbatched(vec![h.alpha], inputs)
    };
    let inputs = [alpha];
    let tangents = directions(&inputs, &[None], &[]);
    check_hvp("dirichlet multinomial", &gradient, &hvp, &inputs, &tangents);
}

/// Every permutation of 0..k.
fn permutations(k: usize) -> Vec<Vec<i64>> {
    if k == 0 {
        return vec![vec![]];
    }
    let mut all = Vec::new();
    for p in permutations(k - 1) {
        for position in 0..=p.len() {
            let mut q = p.clone();
            q.insert(position, (k - 1) as i64);
            all.push(q);
        }
    }
    all
}

#[test]
fn plackett_luce_probabilities_sum_to_one() {
    let worths = v(&[0.5, 2.0, 1.0, 3.5]);
    let d = PlackettLuce::new(&worths);
    let total: f64 = permutations(4)
        .into_iter()
        .map(|x| d.log_prob(&IntTensor::vector(x)).unwrap().exp())
        .sum();
    assert_close(total, 1.0, 1e-13, "sum");
    // the first choice is in proportion to the worths
    let lp = d.log_prob(&IntTensor::vector(vec![3, 1, 2, 0])).unwrap();
    let expected = (3.5f64 / 7.0).ln() + (2.0f64 / 3.5).ln() + (1.0f64 / 1.5).ln();
    assert_close(lp, expected, 1e-14, "ranking");
    for bad in [vec![0, 1, 1, 2], vec![0, 1, 2, 4], vec![-1, 0, 1, 2]] {
        assert_eq!(
            d.log_prob(&IntTensor::vector(bad)).unwrap(),
            f64::NEG_INFINITY
        );
    }
    let draws = d
        .sample(&[30_000, 4], &mut KeanuRng::seed_from_u64(7))
        .unwrap();
    for item in 0..4 {
        let first = draws.as_slice().chunks(4).filter(|r| r[0] == item).count() as f64 / 30_000.0;
        assert_close(
            first,
            worths.as_slice()[item as usize] / 7.0,
            0.01,
            "first choice",
        );
    }
    // the second choice too: P(1 then 3) = 2/7 · 3.5/5
    let pair = draws
        .as_slice()
        .chunks(4)
        .filter(|r| r[0] == 1 && r[1] == 3)
        .count() as f64
        / 30_000.0;
    assert_close(pair, 2.0 / 7.0 * 3.5 / 5.0, 0.01, "second choice");
}

#[test]
fn plackett_luce_gradients_and_hvps_match_finite_differences() {
    let x = IntTensor::new(&[2, 4], vec![2, 0, 3, 1, 1, 3, 0, 2]).unwrap();
    let worths = v(&[0.5, 2.0, 1.0, 3.5]);
    let f = |inputs: &[DoubleTensor]| PlackettLuce::new(&inputs[0]).log_prob(&x).unwrap();
    let g = PlackettLuce::new(&worths).d_log_prob(&x).unwrap();
    check_gradient(
        "plackett luce",
        &f,
        std::slice::from_ref(&worths),
        &[g.worths],
        &[None],
        &[],
    );
    let gradient = |inputs: &[DoubleTensor]| {
        vec![PlackettLuce::new(&inputs[0]).d_log_prob(&x).unwrap().worths]
    };
    let hvp = |inputs: &[DoubleTensor], tangents: &[DoubleTensor]| {
        let t = batched(tangents);
        unbatched(
            vec![
                PlackettLuce::new(&inputs[0])
                    .hvp(&x, Some(&t[0]), 1)
                    .unwrap()
                    .worths,
            ],
            inputs,
        )
    };
    let inputs = [worths];
    let tangents = directions(&inputs, &[None], &[]);
    check_hvp("plackett luce", &gradient, &hvp, &inputs, &tangents);
}

// ---------------------------------------------------------------------------------------------------------------
// Multivariate Student-t

/// The multivariate Student-t log density by the dense formula.
fn dense_student_t(x: &DoubleTensor, nu: f64, mu: &DoubleTensor, scale: &DoubleTensor) -> f64 {
    let k = x.len() as f64;
    let r = x.sub(mu).unwrap();
    let column = r.reshape(&[-1, 1]).unwrap();
    let q = column
        .matmul_transposed(&inverse(scale).matmul(&column).unwrap(), true, false)
        .unwrap()
        .sum();
    let ln_gamma = keanu_tensor::special::ln_gamma;
    ln_gamma((nu + k) / 2.0)
        - ln_gamma(nu / 2.0)
        - k / 2.0 * (nu * std::f64::consts::PI).ln()
        - 0.5 * log_det(scale)
        - (nu + k) / 2.0 * (1.0 + q / nu).ln()
}

#[test]
fn multivariate_student_t_matches_the_dense_formula_and_its_cholesky_form() {
    let scale = spd(3, &[1.2, 0.3, 0.9, -0.4, 0.2, 0.7]);
    let factor = scale.cholesky().unwrap();
    let mu = v(&[0.1, -0.5, 1.0]);
    let x = t(&[2, 3], &[0.4, 0.3, -0.2, 2.0, -1.0, 0.5]);
    let nu = 4.5;
    let d = MultivariateStudentT::new(s(nu), &mu, &scale);
    let c = MultivariateStudentTCholesky::new(s(nu), &mu, &factor);
    let expected = dense_student_t(&x.slice("0").unwrap(), nu, &mu, &scale)
        + dense_student_t(&x.slice("1").unwrap(), nu, &mu, &scale);
    assert_close_rel(d.log_prob(&x).unwrap(), expected, 1e-13, "dense");
    assert_close_rel(c.log_prob(&x).unwrap(), expected, 1e-13, "cholesky");
    // as ν grows it approaches the Gaussian
    let big = MultivariateStudentT::new(s(1e6), &mu, &scale)
        .log_prob(&x)
        .unwrap();
    let gaussian = MultivariateGaussian::new(&mu, &scale).log_prob(&x).unwrap();
    assert_close_rel(big, gaussian, 1e-4, "gaussian limit");
    // invalid parameters give NaN
    assert!(
        MultivariateStudentT::new(s(-1.0), &mu, &scale)
            .log_prob(&x)
            .unwrap()
            .is_nan()
    );
    let not_pd = t(&[3, 3], &[1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    assert!(
        MultivariateStudentT::new(s(nu), &mu, &not_pd)
            .log_prob(&x)
            .unwrap()
            .is_nan()
    );
}

#[test]
fn multivariate_student_t_gradients_and_hvps_match_finite_differences() {
    let scale = spd(3, &[1.2, 0.3, 0.9, -0.4, 0.2, 0.7]);
    let factor = scale.cholesky().unwrap();
    let inputs = [
        t(&[2, 3], &[0.4, 0.3, -0.2, 2.0, -1.0, 0.5]),
        s(4.5),
        v(&[0.1, -0.5, 1.0]),
        scale,
    ];
    let symmetric = [None, None, None, Some(3)];
    let f = |i: &[DoubleTensor]| {
        MultivariateStudentT::new(&i[1], &i[2], &i[3])
            .log_prob(&i[0])
            .unwrap()
    };
    let gradient = |i: &[DoubleTensor]| {
        let g = MultivariateStudentT::new(&i[1], &i[2], &i[3])
            .d_log_prob(&i[0])
            .unwrap();
        vec![g.x, g.nu, g.mu, g.scale]
    };
    check_gradient(
        "student t",
        &f,
        &inputs,
        &gradient(&inputs),
        &symmetric,
        &[],
    );
    let hvp = |i: &[DoubleTensor], tangents: &[DoubleTensor]| {
        let t = batched(tangents);
        let h = MultivariateStudentT::new(&i[1], &i[2], &i[3])
            .hvp(
                &i[0],
                Some(&t[0]),
                &[Some(&t[1]), Some(&t[2]), Some(&t[3])],
                1,
            )
            .unwrap();
        unbatched(vec![h.x, h.nu, h.mu, h.scale], i)
    };
    let tangents = directions(&inputs, &symmetric, &[]);
    check_hvp("student t", &gradient, &hvp, &inputs, &tangents);

    // the Cholesky form, in the factor's lower triangle
    let mut inputs = inputs;
    inputs[3] = factor;
    let none = [None; 4];
    let f = |i: &[DoubleTensor]| {
        MultivariateStudentTCholesky::new(&i[1], &i[2], &i[3])
            .log_prob(&i[0])
            .unwrap()
    };
    let gradient = |i: &[DoubleTensor]| {
        let g = MultivariateStudentTCholesky::new(&i[1], &i[2], &i[3])
            .d_log_prob(&i[0])
            .unwrap();
        vec![g.x, g.nu, g.mu, g.scale_cholesky]
    };
    check_gradient(
        "student t cholesky",
        &f,
        &inputs,
        &gradient(&inputs),
        &none,
        &[],
    );
    let hvp = |i: &[DoubleTensor], tangents: &[DoubleTensor]| {
        let t = batched(tangents);
        let h = MultivariateStudentTCholesky::new(&i[1], &i[2], &i[3])
            .hvp(
                &i[0],
                Some(&t[0]),
                &[Some(&t[1]), Some(&t[2]), Some(&t[3])],
                1,
            )
            .unwrap();
        unbatched(vec![h.x, h.nu, h.mu, h.scale_cholesky], i)
    };
    let tangents = directions(&inputs, &none, &[3]);
    check_hvp("student t cholesky", &gradient, &hvp, &inputs, &tangents);
}

#[test]
fn multivariate_student_t_samples_have_the_right_moments() {
    let scale = spd(2, &[1.0, 0.6, 0.8]);
    let mu = v(&[1.0, -2.0]);
    let nu = 10.0;
    let draws = MultivariateStudentT::new(s(nu), &mu, &scale)
        .sample(&[60_000, 2], &mut KeanuRng::seed_from_u64(8))
        .unwrap();
    let rows: Vec<&[f64]> = draws.as_slice().chunks(2).collect();
    let m: Vec<f64> = (0..2)
        .map(|c| mean(&rows.iter().map(|r| r[c]).collect::<Vec<_>>()))
        .collect();
    for (m, mu) in m.iter().zip(mu.as_slice()) {
        assert_close(*m, *mu, 0.02, "mean");
    }
    // the covariance is νΣ/(ν − 2)
    for (i, j) in [(0, 0), (0, 1), (1, 1)] {
        let cov = mean(
            &rows
                .iter()
                .map(|r| (r[i] - m[i]) * (r[j] - m[j]))
                .collect::<Vec<_>>(),
        );
        let expected = nu / (nu - 2.0) * scale.as_slice()[i * 2 + j];
        assert_close(cov, expected, 0.05 * expected.abs().max(1.0), "covariance");
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Wishart and inverse Wishart

fn dense_wishart(w: &DoubleTensor, nu: f64, scale: &DoubleTensor) -> f64 {
    let k = w.shape()[0];
    let kf = k as f64;
    (nu - kf - 1.0) / 2.0 * log_det(w)
        - 0.5 * trace_product(&inverse(scale), w)
        - nu * kf / 2.0 * std::f64::consts::LN_2
        - nu / 2.0 * log_det(scale)
        - log_multivariate_gamma(k, nu / 2.0)
}

fn dense_inverse_wishart(w: &DoubleTensor, nu: f64, scale: &DoubleTensor) -> f64 {
    let k = w.shape()[0];
    let kf = k as f64;
    nu / 2.0 * log_det(scale)
        - nu * kf / 2.0 * std::f64::consts::LN_2
        - log_multivariate_gamma(k, nu / 2.0)
        - (nu + kf + 1.0) / 2.0 * log_det(w)
        - 0.5 * trace_product(scale, &inverse(w))
}

/// K ln 2 + Σ (K − k + 1) ln L_kk (one-based k): the log-Jacobian of L ↦ L Lᵀ.
fn cholesky_log_jacobian(l: &DoubleTensor) -> f64 {
    let k = l.shape()[0];
    k as f64 * std::f64::consts::LN_2
        + (0..k)
            .map(|i| (k - i) as f64 * l.as_slice()[i * k + i].ln())
            .sum::<f64>()
}

#[test]
fn wishart_densities_match_dense_formulas_and_their_cholesky_forms() {
    let scale = spd(3, &[1.1, 0.4, 0.8, -0.3, 0.2, 1.3]);
    let w = spd(3, &[2.0, 0.5, 1.5, 0.1, -0.6, 0.9]);
    let (ls, lw) = (scale.cholesky().unwrap(), w.cholesky().unwrap());
    let nu = 5.5;
    let expected = dense_wishart(&w, nu, &scale);
    assert_close_rel(
        Wishart::new(s(nu), &scale).log_prob(&w).unwrap(),
        expected,
        1e-13,
        "wishart",
    );
    assert_close_rel(
        WishartCholesky::new(s(nu), &ls).log_prob(&lw).unwrap(),
        expected + cholesky_log_jacobian(&lw),
        1e-13,
        "wishart cholesky",
    );
    let expected = dense_inverse_wishart(&w, nu, &scale);
    assert_close_rel(
        InverseWishart::new(s(nu), &scale).log_prob(&w).unwrap(),
        expected,
        1e-13,
        "inverse",
    );
    assert_close_rel(
        InverseWishartCholesky::new(s(nu), &ls)
            .log_prob(&lw)
            .unwrap(),
        expected + cholesky_log_jacobian(&lw),
        1e-13,
        "inverse cholesky",
    );
    // K = 1: Wishart(ν, s) is Gamma(shape ν/2, scale 2s), and the inverse Wishart the inverse gamma
    let (nu, sc, x) = (3.7, 0.8, 1.9);
    let one = |v: f64| t(&[1, 1], &[v]);
    let gamma = Gamma::new(s(2.0 * sc), s(nu / 2.0))
        .log_prob(&s(x))
        .unwrap()
        .as_slice()[0];
    assert_close_rel(
        Wishart::new(s(nu), one(sc)).log_prob(&one(x)).unwrap(),
        gamma,
        1e-13,
        "K = 1",
    );
    let inverse_gamma = InverseGamma::new(s(nu / 2.0), s(sc / 2.0))
        .log_prob(&s(x))
        .unwrap()
        .as_slice()[0];
    assert_close_rel(
        InverseWishart::new(s(nu), one(sc))
            .log_prob(&one(x))
            .unwrap(),
        inverse_gamma,
        1e-13,
        "K = 1 inverse",
    );
    // outside the support, and invalid parameters
    let not_pd = t(&[2, 2], &[1.0, 2.0, 2.0, 1.0]);
    let asymmetric = t(&[2, 2], &[1.0, 0.2, 0.3, 1.0]);
    let eye = t(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
    for bad in [&not_pd, &asymmetric] {
        assert_eq!(
            Wishart::new(s(3.0), &eye).log_prob(bad).unwrap(),
            f64::NEG_INFINITY
        );
        assert_eq!(
            InverseWishart::new(s(3.0), &eye).log_prob(bad).unwrap(),
            f64::NEG_INFINITY
        );
    }
    assert_eq!(
        WishartCholesky::new(s(3.0), &eye)
            .log_prob(&asymmetric)
            .unwrap(),
        f64::NEG_INFINITY
    );
    assert!(Wishart::new(s(0.9), &eye).log_prob(&eye).unwrap().is_nan());
    assert!(
        Wishart::new(s(3.0), &not_pd)
            .log_prob(&eye)
            .unwrap()
            .is_nan()
    );
    assert!(
        WishartCholesky::new(s(3.0), t(&[2, 2], &[-1.0, 0.0, 0.0, 1.0]))
            .log_prob(&eye)
            .unwrap()
            .is_nan()
    );
}

#[test]
fn the_cholesky_jacobian_matches_finite_differences() {
    // the map from the lower triangle of L to the lower triangle of L Lᵀ: its log |det J| is the term the Cholesky
    // forms add to the matrix forms
    let l = cholesky_factor(3, &[1.3, 0.4, 0.9, -0.7, 0.2, 1.6]);
    let positions: Vec<(usize, usize)> =
        (0..3).flat_map(|i| (0..=i).map(move |j| (i, j))).collect();
    let map = |l: &DoubleTensor| -> Vec<f64> {
        let w = l.matmul_transposed(l, false, true).unwrap();
        positions
            .iter()
            .map(|&(i, j)| w.as_slice()[i * 3 + j])
            .collect()
    };
    let n = positions.len();
    let mut jacobian = DoubleTensor::zeros(&[n, n]);
    let mut data = jacobian.as_slice().to_vec();
    for (c, &(i, j)) in positions.iter().enumerate() {
        let h = 1e-6;
        let mut plus = l.as_slice().to_vec();
        plus[i * 3 + j] += h;
        let mut minus = l.as_slice().to_vec();
        minus[i * 3 + j] -= h;
        let (p, m) = (map(&t(&[3, 3], &plus)), map(&t(&[3, 3], &minus)));
        for r in 0..n {
            data[r * n + c] = (p[r] - m[r]) / (2.0 * h);
        }
    }
    jacobian = t(&[n, n], &data);
    // |det J| from an LU-free route: det(J)² = det(Jᵀ J)
    let gram = jacobian.matmul_transposed(&jacobian, true, false).unwrap();
    let log_abs_det = 0.5 * log_det(&gram);
    assert_close(log_abs_det, cholesky_log_jacobian(&l), 1e-7, "log-Jacobian");
}

#[test]
fn wishart_gradients_and_hvps_match_finite_differences() {
    let scale = spd(3, &[1.1, 0.4, 0.8, -0.3, 0.2, 1.3]);
    let w = spd(3, &[2.0, 0.5, 1.5, 0.1, -0.6, 0.9]);
    let symmetric = [Some(3), None, Some(3)];
    for inverse in [false, true] {
        let name = if inverse {
            "inverse wishart"
        } else {
            "wishart"
        };
        let log_prob = move |i: &[DoubleTensor]| {
            if inverse {
                InverseWishart::new(&i[1], &i[2]).log_prob(&i[0]).unwrap()
            } else {
                Wishart::new(&i[1], &i[2]).log_prob(&i[0]).unwrap()
            }
        };
        let gradient = move |i: &[DoubleTensor]| {
            if inverse {
                let g = InverseWishart::new(&i[1], &i[2]).d_log_prob(&i[0]).unwrap();
                vec![g.x, g.nu, g.scale]
            } else {
                let g = Wishart::new(&i[1], &i[2]).d_log_prob(&i[0]).unwrap();
                vec![g.x, g.nu, g.scale]
            }
        };
        let hvp = move |i: &[DoubleTensor], tangents: &[DoubleTensor]| {
            let t = batched(tangents);
            let parts = if inverse {
                let h = InverseWishart::new(&i[1], &i[2])
                    .hvp(&i[0], Some(&t[0]), &[Some(&t[1]), Some(&t[2])], 1)
                    .unwrap();
                vec![h.x, h.nu, h.scale]
            } else {
                let h = Wishart::new(&i[1], &i[2])
                    .hvp(&i[0], Some(&t[0]), &[Some(&t[1]), Some(&t[2])], 1)
                    .unwrap();
                vec![h.x, h.nu, h.scale]
            };
            unbatched(parts, i)
        };
        let inputs = [w.clone(), s(5.5), scale.clone()];
        check_gradient(
            name,
            &log_prob,
            &inputs,
            &gradient(&inputs),
            &symmetric,
            &[],
        );
        let tangents = directions(&inputs, &symmetric, &[]);
        check_hvp(name, &gradient, &hvp, &inputs, &tangents);
    }
    // the Cholesky forms, in the lower triangles
    let (ls, lw) = (scale.cholesky().unwrap(), w.cholesky().unwrap());
    let none = [None; 3];
    for inverse in [false, true] {
        let name = if inverse {
            "inverse wishart cholesky"
        } else {
            "wishart cholesky"
        };
        let log_prob = move |i: &[DoubleTensor]| {
            if inverse {
                InverseWishartCholesky::new(&i[1], &i[2])
                    .log_prob(&i[0])
                    .unwrap()
            } else {
                WishartCholesky::new(&i[1], &i[2]).log_prob(&i[0]).unwrap()
            }
        };
        let gradient = move |i: &[DoubleTensor]| {
            if inverse {
                let g = InverseWishartCholesky::new(&i[1], &i[2])
                    .d_log_prob(&i[0])
                    .unwrap();
                vec![g.x, g.nu, g.scale_cholesky]
            } else {
                let g = WishartCholesky::new(&i[1], &i[2])
                    .d_log_prob(&i[0])
                    .unwrap();
                vec![g.x, g.nu, g.scale_cholesky]
            }
        };
        let hvp = move |i: &[DoubleTensor], tangents: &[DoubleTensor]| {
            let t = batched(tangents);
            let parts = if inverse {
                let h = InverseWishartCholesky::new(&i[1], &i[2])
                    .hvp(&i[0], Some(&t[0]), &[Some(&t[1]), Some(&t[2])], 1)
                    .unwrap();
                vec![h.x, h.nu, h.scale_cholesky]
            } else {
                let h = WishartCholesky::new(&i[1], &i[2])
                    .hvp(&i[0], Some(&t[0]), &[Some(&t[1]), Some(&t[2])], 1)
                    .unwrap();
                vec![h.x, h.nu, h.scale_cholesky]
            };
            unbatched(parts, i)
        };
        let inputs = [lw.clone(), s(5.5), ls.clone()];
        // only the lower triangles are free (the upper triangle of the value must stay zero)
        check_gradient_lower(
            name,
            &log_prob,
            &inputs,
            &gradient(&inputs),
            &none,
            &[],
            &[0],
        );
        let tangents = directions(&inputs, &none, &[0, 2]);
        check_hvp(name, &gradient, &hvp, &inputs, &tangents);
    }
}

#[test]
fn wishart_samples_have_the_right_means() {
    let scale = spd(3, &[1.1, 0.4, 0.8, -0.3, 0.2, 1.3]);
    let nu = 7.0;
    let count = 20_000;
    let rng = &mut KeanuRng::seed_from_u64(9);
    let mean_matrix = |draws: &DoubleTensor| -> Vec<f64> {
        (0..9)
            .map(|e| draws.as_slice().iter().skip(e).step_by(9).sum::<f64>() / count as f64)
            .collect()
    };
    let w = Wishart::new(s(nu), &scale)
        .sample(&[count, 3, 3], rng)
        .unwrap();
    for (m, s) in mean_matrix(&w).iter().zip(scale.as_slice()) {
        assert_close(*m, nu * s, 0.06 * nu, "Wishart mean νS");
    }
    // the Cholesky form's draws are factors of Wishart draws
    let l = WishartCholesky::new(s(nu), scale.cholesky().unwrap())
        .sample(&[count, 3, 3], rng)
        .unwrap();
    let products = l.matmul_transposed(&l, false, true).unwrap();
    for (m, s) in mean_matrix(&products).iter().zip(scale.as_slice()) {
        assert_close(*m, nu * s, 0.06 * nu, "Wishart Cholesky mean");
    }
    assert!(
        l.as_slice()
            .chunks(9)
            .all(|f| f[1] == 0.0 && f[2] == 0.0 && f[5] == 0.0 && f[0] > 0.0)
    );
    // the inverse Wishart's mean is S/(ν − K − 1)
    let nu = 9.0;
    let w = InverseWishart::new(s(nu), &scale)
        .sample(&[count, 3, 3], rng)
        .unwrap();
    for (m, s) in mean_matrix(&w).iter().zip(scale.as_slice()) {
        assert_close(*m, s / (nu - 4.0), 0.02, "inverse Wishart mean");
    }
    let l = InverseWishartCholesky::new(s(nu), scale.cholesky().unwrap())
        .sample(&[count, 3, 3], rng)
        .unwrap();
    let products = l.matmul_transposed(&l, false, true).unwrap();
    for (m, s) in mean_matrix(&products).iter().zip(scale.as_slice()) {
        assert_close(*m, s / (nu - 4.0), 0.02, "inverse Wishart Cholesky mean");
    }
    assert!(Wishart::new(s(1.5), &scale).sample(&[3, 3], rng).is_err());
}

// ---------------------------------------------------------------------------------------------------------------
// Matrix normal

#[test]
fn matrix_normal_is_a_multivariate_normal_with_kronecker_covariance() {
    let (n, p) = (3, 2);
    let u = spd(n, &[1.0, 0.3, 0.8, -0.2, 0.4, 1.2]);
    let col = spd(p, &[0.9, -0.5, 0.7]);
    let mu = t(&[n, p], &[0.1, 0.2, -0.3, 0.4, 0.5, -0.6]);
    let x = t(&[n, p], &[0.4, -0.1, 0.0, 1.2, -0.7, 0.3]);
    // vec(X) row-major: entry (i, a) at i·p + a, with covariance U_ij V_ab
    let mut kron = vec![0.0; n * p * n * p];
    for i in 0..n {
        for a in 0..p {
            for j in 0..n {
                for b in 0..p {
                    kron[(i * p + a) * n * p + j * p + b] =
                        u.as_slice()[i * n + j] * col.as_slice()[a * p + b];
                }
            }
        }
    }
    let expected = MultivariateGaussian::new(mu.reshape(&[-1]).unwrap(), t(&[n * p, n * p], &kron))
        .log_prob(&x.reshape(&[-1]).unwrap())
        .unwrap();
    let d = MatrixNormal::new(&mu, &u, &col);
    assert_close_rel(d.log_prob(&x).unwrap(), expected, 1e-13, "kronecker");

    let inputs = [x, mu.clone(), u.clone(), col.clone()];
    let symmetric = [None, None, Some(n), Some(p)];
    let f = |i: &[DoubleTensor]| {
        MatrixNormal::new(&i[1], &i[2], &i[3])
            .log_prob(&i[0])
            .unwrap()
    };
    let gradient = |i: &[DoubleTensor]| {
        let g = MatrixNormal::new(&i[1], &i[2], &i[3])
            .d_log_prob(&i[0])
            .unwrap();
        vec![g.x, g.mu, g.row_covariance, g.column_covariance]
    };
    check_gradient(
        "matrix normal",
        &f,
        &inputs,
        &gradient(&inputs),
        &symmetric,
        &[],
    );
    let hvp = |i: &[DoubleTensor], tangents: &[DoubleTensor]| {
        let t = batched(tangents);
        let h = MatrixNormal::new(&i[1], &i[2], &i[3])
            .hvp(
                &i[0],
                Some(&t[0]),
                &[Some(&t[1]), Some(&t[2]), Some(&t[3])],
                1,
            )
            .unwrap();
        unbatched(vec![h.x, h.mu, h.row_covariance, h.column_covariance], i)
    };
    let tangents = directions(&inputs, &symmetric, &[]);
    check_hvp("matrix normal", &gradient, &hvp, &inputs, &tangents);

    // draws have mean M and the Kronecker covariance
    let draws = d
        .sample(&[40_000, n, p], &mut KeanuRng::seed_from_u64(10))
        .unwrap();
    let rows: Vec<&[f64]> = draws.as_slice().chunks(n * p).collect();
    let m: Vec<f64> = (0..n * p)
        .map(|e| mean(&rows.iter().map(|r| r[e]).collect::<Vec<_>>()))
        .collect();
    for (m, e) in m.iter().zip(inputs[1].as_slice()) {
        assert_close(*m, *e, 0.02, "mean");
    }
    for (e1, e2) in [(0, 0), (0, 1), (1, 4), (5, 2)] {
        let cov = mean(
            &rows
                .iter()
                .map(|r| (r[e1] - m[e1]) * (r[e2] - m[e2]))
                .collect::<Vec<_>>(),
        );
        assert_close(cov, kron[e1 * n * p + e2], 0.03, "covariance");
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Von Mises-Fisher

#[test]
fn von_mises_fisher_normalises_and_matches_the_von_mises_in_two_dimensions() {
    // p = 3: ∫ C₃(κ) e^(κ cos θ) 2π sin θ dθ over [0, π] = 1, by Simpson's rule
    for kappa in [0.0, 0.3, 2.5, 40.0] {
        let d = VonMisesFisher::new(v(&[0.0, 0.0, 1.0]), s(kappa));
        let steps = 4000;
        let h = std::f64::consts::PI / steps as f64;
        let mut total = 0.0;
        for i in 0..=steps {
            let theta = i as f64 * h;
            let x = v(&[theta.sin(), 0.0, theta.cos()]);
            let density = d.log_prob(&x).unwrap().exp() * 2.0 * std::f64::consts::PI * theta.sin();
            let weight = if i == 0 || i == steps {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            total += weight * density;
        }
        assert_close(total * h / 3.0, 1.0, 1e-8, "normalisation");
    }
    // p = 2: the von Mises density of the angle
    let (mu, kappa) = (0.7f64, 1.9);
    let d = VonMisesFisher::new(v(&[mu.cos(), mu.sin()]), s(kappa));
    for theta in [-2.0f64, 0.1, 1.5] {
        let lp = d.log_prob(&v(&[theta.cos(), theta.sin()])).unwrap();
        let expected = VonMises::new(s(mu), s(kappa))
            .log_prob(&s(theta))
            .unwrap()
            .as_slice()[0];
        assert_close_rel(lp, expected, 1e-13, "von Mises");
    }
    // off the sphere is impossible, a non-unit mean direction or negative concentration invalid
    let d = VonMisesFisher::new(v(&[1.0, 0.0, 0.0]), s(1.0));
    assert_eq!(d.log_prob(&v(&[1.0, 0.1, 0.0])).unwrap(), f64::NEG_INFINITY);
    assert!(
        VonMisesFisher::new(v(&[1.0, 1.0, 0.0]), s(1.0))
            .log_prob(&v(&[1.0, 0.0, 0.0]))
            .unwrap()
            .is_nan()
    );
    assert!(
        VonMisesFisher::new(v(&[1.0, 0.0, 0.0]), s(-1.0))
            .log_prob(&v(&[1.0, 0.0, 0.0]))
            .unwrap()
            .is_nan()
    );
}

#[test]
fn von_mises_fisher_gradients_and_hvps_match_finite_differences() {
    let norm = |values: &[f64]| {
        let length = values.iter().map(|v| v * v).sum::<f64>().sqrt();
        values.iter().map(|v| v / length).collect::<Vec<_>>()
    };
    let x = t(
        &[2, 4],
        &[norm(&[0.3, -0.2, 0.9, 0.1]), norm(&[-0.5, 0.5, 0.2, 0.7])].concat(),
    );
    let mu = v(&norm(&[0.2, 0.1, 0.8, -0.3]));
    // the log density is linear in x and μ, so finite differences off the sphere are fine for the gradient and
    // Hessian (the support check is not part of the gradient)
    for kappa in [0.0, 1e-5, 0.8, 30.0] {
        let inputs = [x.clone(), mu.clone(), s(kappa)];
        let gradient = |i: &[DoubleTensor]| {
            let g = VonMisesFisher::new(&i[1], &i[2]).d_log_prob(&i[0]).unwrap();
            vec![g.x, g.mu, g.kappa]
        };
        // the κ derivative against finite differences of the log density (κ is the only input with a support check
        // that a step keeps)
        if kappa > 0.0 {
            let f = |i: &[DoubleTensor]| VonMisesFisher::new(&i[1], &i[2]).log_prob(&i[0]).unwrap();
            check_gradient("vmf", &f, &inputs, &gradient(&inputs), &[None; 3], &[0, 1]);
        }
        let hvp = |i: &[DoubleTensor], tangents: &[DoubleTensor]| {
            let t = batched(tangents);
            let h = VonMisesFisher::new(&i[1], &i[2])
                .hvp(&i[0], Some(&t[0]), &[Some(&t[1]), Some(&t[2])], 1)
                .unwrap();
            unbatched(vec![h.x, h.mu, h.kappa], i)
        };
        let mut tangents = directions(&inputs, &[None; 3], &[]);
        if kappa == 0.0 {
            // a positive step in κ, to stay in the support
            tangents[2] = s(0.3);
        }
        let tangents = tangents;
        if kappa > 0.0 {
            check_hvp("vmf", &gradient, &hvp, &inputs, &tangents);
        }
        // at κ = 0 the Hessian in κ is −A'(0) = −1/p
        if kappa == 0.0 {
            let h = hvp(
                &inputs,
                &[
                    DoubleTensor::zeros(&[2, 4]),
                    DoubleTensor::zeros(&[4]),
                    s(1.0),
                ],
            );
            assert_close(h[2].as_slice()[0], -2.0 / 4.0, 1e-12, "κ = 0 curvature");
        }
    }
}

#[test]
fn von_mises_fisher_samples_have_the_mean_resultant_length() {
    let mu = v(&[0.0, 0.6, 0.8]);
    for kappa in [0.0, 0.5, 4.0, 200.0] {
        let draws = VonMisesFisher::new(&mu, s(kappa))
            .sample(&[20_000, 3], &mut KeanuRng::seed_from_u64(11))
            .unwrap();
        let rows: Vec<&[f64]> = draws.as_slice().chunks(3).collect();
        assert!(
            rows.iter()
                .all(|r| (r.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() < 1e-12)
        );
        let resultant = mean(
            &rows
                .iter()
                .map(|r| r.iter().zip(mu.as_slice()).map(|(x, m)| x * m).sum::<f64>())
                .collect::<Vec<_>>(),
        );
        let expected = keanu_tensor::special::bessel_i_ratio(0.5, kappa);
        assert_close(resultant, expected, 0.01, "mean resultant length");
        // the mean direction is μ: the component orthogonal to μ averages to 0
        let orthogonal = mean(&rows.iter().map(|r| r[0]).collect::<Vec<_>>());
        assert_close(orthogonal, 0.0, 0.015, "orthogonal mean");
    }
}
