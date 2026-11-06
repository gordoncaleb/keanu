//! The continuous families written as one scalar formula (skew normal, ex-Gaussian, Gumbel, generalized extreme
//! value, generalized Pareto, inverse Gaussian, Kumaraswamy, logit-normal, half Student-t, asymmetric Laplace,
//! log-logistic, Rice, generalized gamma): each density normalises (tanh-sinh quadrature), each CDF and survival
//! function matches the integrated density, the inverse CDFs invert, gradients and Hessians match finite
//! differences, samplers match the moments (from quadrature) and pass a Kolmogorov–Smirnov test, and values outside
//! the support are impossible. Also the generalized extreme value distribution's limit ξ → 0 (the Gumbel).

use keanu_distributions::*;
use keanu_tensor::DoubleTensor;

mod common;
use common::{s, v};

type Derivatives = (f64, Vec<f64>, Vec<Vec<f64>>);
/// A function of x and the parameters.
type OfX<T> = Box<dyn Fn(f64, &[f64]) -> T>;
/// A function of x, the parameters and whether it is the upper tail.
type OfTail<T> = Box<dyn Fn(f64, &[f64], bool) -> T>;
/// A function of a tensor of x and the parameters (and whether it is the upper tail).
type OfTensor = Box<dyn Fn(&DoubleTensor, &[f64]) -> DoubleTensor>;
type OfTensorTail = Box<dyn Fn(&DoubleTensor, &[f64], bool) -> DoubleTensor>;
type Sampler = Box<dyn Fn(&[f64], usize, &mut KeanuRng) -> Vec<f64>>;

fn parts<const N: usize>(d: ElementDerivatives<N>) -> Derivatives {
    (
        d.value,
        d.gradient.to_vec(),
        d.hessian.iter().map(|row| row.to_vec()).collect(),
    )
}

/// A family at fixed parameters, as scalar functions.
struct Family {
    name: &'static str,
    params: Vec<f64>,
    /// The support's ends.
    lower: f64,
    upper: f64,
    /// Points where the density has a kink, which the quadrature splits at.
    breaks: Vec<f64>,
    /// Points inside the support where derivatives are checked.
    points: Vec<f64>,
    /// Points outside the support (or on its boundary where the density is 0).
    outside: Vec<f64>,
    density: OfX<Derivatives>,
    tensor_log_prob: OfTensor,
    cdf: Option<OfTail<Derivatives>>,
    tensor_log_cdf: Option<OfTensorTail>,
    inverse: Option<OfTail<f64>>,
    sample: Sampler,
}

macro_rules! family {
    ($name:literal, $ty:ident, |$p:ident| $new:expr, [$($param:expr),+], ($lower:expr, $upper:expr),
     points = [$($point:expr),*], outside = [$($out:expr),*] $(, $extra:ident)*) => {{
        #[allow(unused_mut)]
        let mut f = Family {
            name: $name,
            params: vec![$($param),+],
            lower: $lower,
            upper: $upper,
            breaks: vec![],
            points: vec![$($point),*],
            outside: vec![$($out),*],
            density: Box::new(|x, p: &[f64]| parts($ty::log_prob_derivatives(x, p.try_into().unwrap()))),
            tensor_log_prob: Box::new(|x, $p: &[f64]| $new.log_prob(x).unwrap()),
            cdf: None,
            tensor_log_cdf: None,
            inverse: None,
            sample: Box::new(|$p: &[f64], n, rng| $new.sample(&[n], rng).unwrap().into_vec()),
        };
        $(family!(@$extra f, $ty, |$p| $new);)*
        f
    }};
    (@cdf $f:ident, $ty:ident, |$p:ident| $new:expr) => {
        $f.cdf = Some(Box::new(|x, p: &[f64], upper| {
            parts($ty::log_cdf_derivatives(x, p.try_into().unwrap(), upper))
        }));
        $f.tensor_log_cdf = Some(Box::new(|x, $p: &[f64], upper| {
            let d = $new;
            if upper { d.log_ccdf(x).unwrap() } else { d.log_cdf(x).unwrap() }
        }));
    };
    (@inverse $f:ident, $ty:ident, |$p:ident| $new:expr) => {
        $f.inverse = Some(Box::new(|q, $p: &[f64], upper| {
            let d = $new;
            let q = s(q);
            let x = if upper { d.inverse_ccdf(&q) } else { d.inverse_cdf(&q) };
            x.unwrap().as_slice()[0]
        }));
    };
}

fn families() -> Vec<Family> {
    let inf = f64::INFINITY;
    vec![
        family!(
            "skew normal",
            SkewNormal,
            |p| SkewNormal::new(s(p[0]), s(p[1]), s(p[2])),
            [0.5, 1.3, 2.5],
            (-inf, inf),
            points = [-1.0, 0.4, 2.2, 4.0],
            outside = [],
            cdf
        ),
        family!(
            "skew normal (left)",
            SkewNormal,
            |p| SkewNormal::new(s(p[0]), s(p[1]), s(p[2])),
            [-0.3, 0.8, -1.5],
            (-inf, inf),
            points = [-2.0, -0.5, 0.3],
            outside = [],
            cdf
        ),
        family!(
            "ex-Gaussian",
            ExGaussian,
            |p| ExGaussian::new(s(p[0]), s(p[1]), s(p[2])),
            [0.4, 0.6, 1.5],
            (-inf, inf),
            points = [-0.5, 0.5, 1.4, 3.0],
            outside = [],
            cdf
        ),
        family!(
            "Gumbel",
            Gumbel,
            |p| Gumbel::new(s(p[0]), s(p[1])),
            [0.7, 1.4],
            (-inf, inf),
            points = [-1.5, 0.5, 3.0],
            outside = [],
            cdf,
            inverse
        ),
        family!(
            "GEV (Fréchet)",
            GeneralizedExtremeValue,
            |p| GeneralizedExtremeValue::new(s(p[0]), s(p[1]), s(p[2])),
            [0.5, 1.2, 0.2],
            (0.5 - 1.2 / 0.2, inf),
            points = [-1.0, 0.4, 2.0, 6.0],
            outside = [-5.6, -6.0, -50.0],
            cdf,
            inverse
        ),
        family!(
            "GEV (Weibull)",
            GeneralizedExtremeValue,
            |p| GeneralizedExtremeValue::new(s(p[0]), s(p[1]), s(p[2])),
            [0.5, 1.2, -0.3],
            (-inf, 0.5 + 1.2 / 0.3),
            points = [-2.0, 0.4, 2.0, 4.0],
            outside = [4.5, 4.6, 10.0],
            cdf,
            inverse
        ),
        family!(
            "GEV (Gumbel)",
            GeneralizedExtremeValue,
            |p| GeneralizedExtremeValue::new(s(p[0]), s(p[1]), s(p[2])),
            [0.5, 1.2, 0.0],
            (-inf, inf),
            points = [-2.0, 0.4, 3.0],
            outside = [],
            cdf,
            inverse
        ),
        family!(
            "generalized Pareto",
            GeneralizedPareto,
            |p| GeneralizedPareto::new(s(p[0]), s(p[1]), s(p[2])),
            [1.0, 0.8, 0.2],
            (1.0, inf),
            points = [1.1, 1.8, 4.0],
            outside = [0.99, -1.0],
            cdf,
            inverse
        ),
        family!(
            "generalized Pareto (bounded)",
            GeneralizedPareto,
            |p| GeneralizedPareto::new(s(p[0]), s(p[1]), s(p[2])),
            [1.0, 0.8, -0.4],
            (1.0, 3.0),
            points = [1.1, 1.8, 2.7],
            outside = [0.9, 3.0, 3.5],
            cdf,
            inverse
        ),
        family!(
            "generalized Pareto (exponential)",
            GeneralizedPareto,
            |p| GeneralizedPareto::new(s(p[0]), s(p[1]), s(p[2])),
            [-1.0, 0.8, 0.0],
            (-1.0, inf),
            points = [-0.5, 1.0],
            outside = [-1.1],
            cdf,
            inverse
        ),
        family!(
            "inverse Gaussian",
            InverseGaussian,
            |p| InverseGaussian::new(s(p[0]), s(p[1])),
            [1.3, 2.5],
            (0.0, inf),
            points = [0.3, 1.0, 2.5, 6.0],
            outside = [0.0, -1.0],
            cdf
        ),
        family!(
            "Kumaraswamy",
            Kumaraswamy,
            |p| Kumaraswamy::new(s(p[0]), s(p[1])),
            [2.5, 3.5],
            (0.0, 1.0),
            points = [0.1, 0.4, 0.8],
            outside = [0.0, 1.0, -0.2, 1.3],
            cdf,
            inverse
        ),
        family!(
            "logit-normal",
            LogitNormal,
            |p| LogitNormal::new(s(p[0]), s(p[1])),
            [0.4, 0.9],
            (0.0, 1.0),
            points = [0.1, 0.5, 0.9],
            outside = [0.0, 1.0, -0.2, 1.3],
            cdf,
            inverse
        ),
        family!(
            "half Student-t",
            HalfStudentT,
            |p| HalfStudentT::new(s(p[0]), s(p[1])),
            [4.5, 1.3],
            (0.0, inf),
            points = [0.2, 1.0, 4.0],
            outside = [-0.1, -3.0],
            cdf
        ),
        Family {
            breaks: vec![0.3],
            ..family!(
                "asymmetric Laplace",
                AsymmetricLaplace,
                |p| AsymmetricLaplace::new(s(p[0]), s(p[1]), s(p[2])),
                [0.3, 1.2, 0.3],
                (-inf, inf),
                points = [-2.0, 0.1, 0.8, 3.0],
                outside = [],
                cdf,
                inverse
            )
        },
        family!(
            "log-logistic",
            LogLogistic,
            |p| LogLogistic::new(s(p[0]), s(p[1])),
            [1.5, 4.0],
            (0.0, inf),
            points = [0.3, 1.4, 3.0],
            outside = [0.0, -1.0],
            cdf,
            inverse
        ),
        family!(
            "Rice",
            Rice,
            |p| Rice::new(s(p[0]), s(p[1])),
            [1.5, 0.8],
            (0.0, inf),
            points = [0.3, 1.4, 3.0],
            outside = [0.0, -1.0]
        ),
        family!(
            "Rayleigh (Rice with ν = 0)",
            Rice,
            |p| Rice::new(s(p[0]), s(p[1])),
            [0.0, 0.8],
            (0.0, inf),
            points = [0.3, 1.4],
            outside = [0.0, -1.0]
        ),
        family!(
            "generalized gamma",
            GeneralizedGamma,
            |p| GeneralizedGamma::new(s(p[0]), s(p[1]), s(p[2])),
            [1.4, 2.2, 1.7],
            (0.0, inf),
            points = [0.5, 1.6, 3.5],
            outside = [0.0, -1.0],
            cdf
        ),
    ]
}

// ----- quadrature -----

/// ∫ f over (a, b) by tanh-sinh quadrature, with infinite ends mapped to finite ones.
fn integrate(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    match (a.is_finite(), b.is_finite()) {
        (true, true) => tanh_sinh(f, a, b),
        (true, false) => tanh_sinh(|u| f(a + u / (1.0 - u)) / ((1.0 - u) * (1.0 - u)), 0.0, 1.0),
        (false, true) => tanh_sinh(|u| f(b - u / (1.0 - u)) / ((1.0 - u) * (1.0 - u)), 0.0, 1.0),
        (false, false) => tanh_sinh(
            |u| {
                let d = 1.0 - u * u;
                f(u / d) * (1.0 + u * u) / (d * d)
            },
            -1.0,
            1.0,
        ),
    }
}

fn tanh_sinh(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let (c, d) = (0.5 * (a + b), 0.5 * (b - a));
    let h = 1.0 / 64.0;
    let half_pi = std::f64::consts::FRAC_PI_2;
    let mut total = 0.0;
    for k in -192..=192 {
        let t = k as f64 * h;
        let u = half_pi * t.sinh();
        let x = u.tanh();
        let w = half_pi * t.cosh() / (u.cosh() * u.cosh());
        let point = c + d * x;
        if point <= a || point >= b {
            continue;
        }
        let value = f(point);
        if value.is_finite() {
            total += w * value;
        }
    }
    total * d * h
}

fn density(f: &Family, x: f64) -> f64 {
    (f.density)(x, &f.params).0.exp()
}

/// ∫ g p over (a, b), split at the family's kinks.
fn expect(f: &Family, g: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let mut ends = vec![a];
    ends.extend(f.breaks.iter().copied().filter(|&k| k > a && k < b));
    ends.push(b);
    ends.windows(2)
        .map(|w| integrate(|x| g(x) * density(f, x), w[0], w[1]))
        .sum()
}

#[test]
fn densities_normalise_and_match_the_tensor_api() {
    for f in families() {
        let total = expect(&f, |_| 1.0, f.lower, f.upper);
        assert!((total - 1.0).abs() < 1e-8, "{}: ∫ p = {total}", f.name);
        let xs = v(&f.points);
        let tensor = (f.tensor_log_prob)(&xs, &f.params);
        for (i, &x) in f.points.iter().enumerate() {
            let scalar = (f.density)(x, &f.params).0;
            assert!(
                (tensor.as_slice()[i] - scalar).abs() < 1e-14 * scalar.abs().max(1.0),
                "{}: tensor log_prob {} vs {scalar}",
                f.name,
                tensor.as_slice()[i]
            );
        }
    }
}

#[test]
fn values_outside_the_support_are_impossible() {
    for f in families() {
        for &x in &f.outside {
            let lp = (f.density)(x, &f.params).0;
            assert_eq!(lp, f64::NEG_INFINITY, "{} at {x}", f.name);
        }
    }
}

#[test]
fn cdfs_match_the_integrated_density() {
    for f in families() {
        let Some(cdf) = &f.cdf else { continue };
        for &x in &f.points {
            let below = expect(&f, |_| 1.0, f.lower, x);
            let above = expect(&f, |_| 1.0, x, f.upper);
            let log_cdf = cdf(x, &f.params, false).0;
            let log_ccdf = cdf(x, &f.params, true).0;
            assert!(
                (log_cdf.exp() - below).abs() < 1e-8,
                "{} at {x}: CDF {} vs ∫ {below}",
                f.name,
                log_cdf.exp()
            );
            assert!(
                (log_ccdf.exp() - above).abs() < 1e-8,
                "{} at {x}: survival {} vs ∫ {above}",
                f.name,
                log_ccdf.exp()
            );
            let tensor_cdf = f.tensor_log_cdf.as_ref().unwrap();
            assert_eq!(tensor_cdf(&s(x), &f.params, false).as_slice()[0], log_cdf);
            assert_eq!(tensor_cdf(&s(x), &f.params, true).as_slice()[0], log_ccdf);
        }
        // the limits, and outside the support
        let lower_limit = cdf(f.lower.min(-1e300), &f.params, false).0;
        assert_eq!(
            cdf(f64::NEG_INFINITY, &f.params, false).0,
            f64::NEG_INFINITY
        );
        assert_eq!(cdf(f64::INFINITY, &f.params, true).0, f64::NEG_INFINITY);
        assert_eq!(cdf(f64::INFINITY, &f.params, false).0, 0.0);
        assert_eq!(lower_limit, f64::NEG_INFINITY, "{}", f.name);
        for &x in &f.outside {
            let (lower, upper) = (cdf(x, &f.params, false).0, cdf(x, &f.params, true).0);
            assert!(
                (lower == f64::NEG_INFINITY && upper == 0.0)
                    || (lower == 0.0 && upper == f64::NEG_INFINITY),
                "{} at {x}: {lower}, {upper}",
                f.name
            );
        }
    }
}

#[test]
fn inverse_cdfs_invert() {
    for f in families() {
        let (Some(inverse), Some(cdf)) = (&f.inverse, &f.cdf) else {
            continue;
        };
        for q in [1e-12, 1e-4, 0.1, 0.5, 0.9, 0.9999] {
            for upper in [false, true] {
                let x = inverse(q, &f.params, upper);
                let back = cdf(x, &f.params, upper).0.exp();
                assert!(
                    (back - q).abs() < 1e-9 * q.max(1e-3),
                    "{} (upper {upper}): {q} -> {x} -> {back}",
                    f.name
                );
            }
        }
    }
}

/// Checks a function's gradient and Hessian against central differences of its value and gradient.
fn check_derivatives(what: &str, f: &dyn Fn(f64, &[f64]) -> Derivatives, x: f64, params: &[f64]) {
    let mut inputs = vec![x];
    inputs.extend_from_slice(params);
    let at = |inputs: &[f64]| f(inputs[0], &inputs[1..]);
    let (value, gradient, hessian) = at(&inputs);
    for i in 0..inputs.len() {
        let h = 1e-5 * inputs[i].abs().max(0.1);
        let mut plus = inputs.clone();
        let mut minus = inputs.clone();
        plus[i] += h;
        minus[i] -= h;
        let (vp, gp, _) = at(&plus);
        let (mut vm, mut gm, _) = at(&minus);
        // one-sided at the edge of a parameter's range (ν = 0 for the Rice)
        let mut width = 2.0 * h;
        if vm.is_nan() {
            (vm, gm, width) = (value, gradient.clone(), h);
        }
        let numeric = (vp - vm) / width;
        assert!(
            (gradient[i] - numeric).abs() < 1e-6 * numeric.abs().max(1.0),
            "{what}: d/d input {i} = {}, finite difference {numeric}",
            gradient[i]
        );
        for j in 0..inputs.len() {
            let numeric = (gp[j] - gm[j]) / width;
            assert!(
                (hessian[i][j] - numeric).abs() < 1e-5 * numeric.abs().max(1.0),
                "{what}: d²/d input {i} d input {j} = {}, finite difference {numeric}",
                hessian[i][j]
            );
            assert_eq!(hessian[i][j], hessian[j][i], "{what}: symmetric");
        }
    }
}

#[test]
fn gradients_and_hessians_match_finite_differences() {
    for f in families() {
        for &x in &f.points {
            check_derivatives(
                &format!("{} density at {x}", f.name),
                &f.density,
                x,
                &f.params,
            );
            if let Some(cdf) = &f.cdf {
                for upper in [false, true] {
                    check_derivatives(
                        &format!("{} log CDF (upper {upper}) at {x}", f.name),
                        &|x, p| cdf(x, p, upper),
                        x,
                        &f.params,
                    );
                }
            }
        }
    }
}

#[test]
fn samplers_match_the_moments_and_the_cdf() {
    let n = 20_000;
    for (i, f) in families().iter().enumerate() {
        let mut rng = KeanuRng::seed_from_u64(100 + i as u64);
        let mut draws = (f.sample)(&f.params, n, &mut rng);
        assert!(
            draws.iter().all(|&x| x >= f.lower && x <= f.upper),
            "{}: a draw outside the support",
            f.name
        );
        let mean = expect(f, |x| x, f.lower, f.upper);
        let second = expect(f, |x| x * x, f.lower, f.upper);
        let variance = second - mean * mean;
        let sample_mean = draws.iter().sum::<f64>() / n as f64;
        let sample_variance =
            draws.iter().map(|x| (x - sample_mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        assert!(
            (sample_mean - mean).abs() < 5.0 * (variance / n as f64).sqrt(),
            "{}: sample mean {sample_mean}, mean {mean}",
            f.name
        );
        assert!(
            (sample_variance / variance - 1.0).abs() < 0.1,
            "{}: sample variance {sample_variance}, variance {variance}",
            f.name
        );
        // Kolmogorov–Smirnov: the critical value at the 0.1% level is 1.95/√n
        draws.sort_by(f64::total_cmp);
        let cdf_at = |x: f64| match &f.cdf {
            Some(cdf) => cdf(x, &f.params, false).0.exp(),
            None => expect(f, |_| 1.0, f.lower, x),
        };
        // every draw with a CDF, else 200 evenly spaced order statistics
        let step = if f.cdf.is_some() { 1 } else { n / 200 };
        let statistic = (0..n)
            .step_by(step)
            .map(|k| {
                let c = cdf_at(draws[k]);
                (c - k as f64 / n as f64)
                    .abs()
                    .max((c - (k + 1) as f64 / n as f64).abs())
            })
            .fold(0.0, f64::max);
        assert!(
            statistic < 1.95 / (n as f64).sqrt(),
            "{}: KS statistic {statistic}",
            f.name
        );
    }
}

#[test]
fn sampling_rejects_invalid_parameters() {
    let mut rng = KeanuRng::seed_from_u64(1);
    assert!(
        SkewNormal::new(s(0.0), s(-1.0), s(1.0))
            .sample(&[2], &mut rng)
            .is_err()
    );
    assert!(
        ExGaussian::new(s(0.0), s(1.0), s(0.0))
            .sample(&[2], &mut rng)
            .is_err()
    );
    assert!(Gumbel::new(s(0.0), s(0.0)).sample(&[2], &mut rng).is_err());
    assert!(
        GeneralizedExtremeValue::new(s(0.0), s(-1.0), s(0.1))
            .sample(&[2], &mut rng)
            .is_err()
    );
    assert!(
        GeneralizedPareto::new(s(0.0), s(1.0), s(f64::NAN))
            .sample(&[2], &mut rng)
            .is_err()
    );
    assert!(
        InverseGaussian::new(s(-1.0), s(1.0))
            .sample(&[2], &mut rng)
            .is_err()
    );
    assert!(
        Kumaraswamy::new(s(1.0), s(0.0))
            .sample(&[2], &mut rng)
            .is_err()
    );
    assert!(
        LogitNormal::new(s(0.0), s(0.0))
            .sample(&[2], &mut rng)
            .is_err()
    );
    assert!(
        HalfStudentT::new(s(0.0), s(1.0))
            .sample(&[2], &mut rng)
            .is_err()
    );
    assert!(
        AsymmetricLaplace::new(s(0.0), s(1.0), s(1.0))
            .sample(&[2], &mut rng)
            .is_err()
    );
    assert!(
        LogLogistic::new(s(1.0), s(-1.0))
            .sample(&[2], &mut rng)
            .is_err()
    );
    assert!(Rice::new(s(-1.0), s(1.0)).sample(&[2], &mut rng).is_err());
    assert!(
        GeneralizedGamma::new(s(1.0), s(1.0), s(0.0))
            .sample(&[2], &mut rng)
            .is_err()
    );
    // invalid parameters make the density NaN
    assert!(
        Gumbel::new(s(0.0), s(-1.0))
            .log_prob(&s(0.0))
            .unwrap()
            .as_slice()[0]
            .is_nan()
    );
}

#[test]
fn parameters_broadcast_with_the_value() {
    let gev = GeneralizedExtremeValue::new(v(&[0.0, 1.0]), s(2.0), v(&[0.1, -0.1]));
    let lp = gev.log_prob(&v(&[0.5, 0.5])).unwrap();
    for (i, (mu, xi)) in [(0.0, 0.1), (1.0, -0.1)].into_iter().enumerate() {
        let single = GeneralizedExtremeValue::log_prob_derivatives(0.5, [mu, 2.0, xi]).value;
        assert_eq!(lp.as_slice()[i], single);
    }
    let g = gev.d_log_prob(&v(&[0.5, 0.5])).unwrap();
    assert_eq!(g.xi.shape(), &[2]);
    let sample = gev
        .sample(&[3, 2], &mut KeanuRng::seed_from_u64(2))
        .unwrap();
    assert_eq!(sample.shape(), &[3, 2]);
}

// ----- AE4: the generalized extreme value distribution as ξ → 0 -----

#[test]
fn gev_tends_to_the_gumbel_as_the_shape_tends_to_zero() {
    let (mu, sigma) = (0.3, 1.7);
    for xi in [0.0, 1e-12, -1e-12, 1e-10, -1e-10] {
        for x in [-3.0, -0.5, 0.3, 2.0, 8.0] {
            let gev = GeneralizedExtremeValue::log_prob_derivatives(x, [mu, sigma, xi]);
            let gumbel = Gumbel::log_prob_derivatives(x, [mu, sigma]);
            assert!((gev.value - gumbel.value).abs() < 1e-8, "ξ = {xi}, x = {x}");
            for i in 0..3 {
                assert!(
                    (gev.gradient[i] - gumbel.gradient[i]).abs() < 1e-8,
                    "gradient {i}, ξ = {xi}, x = {x}"
                );
                for j in 0..3 {
                    assert!((gev.hessian[i][j] - gumbel.hessian[i][j]).abs() < 1e-8);
                }
            }
            for upper in [false, true] {
                let gev = GeneralizedExtremeValue::log_cdf_derivatives(x, [mu, sigma, xi], upper);
                let gumbel = Gumbel::log_cdf_derivatives(x, [mu, sigma], upper);
                assert!((gev.value - gumbel.value).abs() < 1e-8 * gumbel.value.abs().max(1.0));
                for i in 0..3 {
                    assert!((gev.gradient[i] - gumbel.gradient[i]).abs() < 1e-8);
                }
            }
        }
    }
    // the shape derivative is continuous through 0: at ξ = 0 it is the limit of the closed form's
    for x in [-1.0, 0.5, 3.0] {
        let at_zero = GeneralizedExtremeValue::log_prob_derivatives(x, [mu, sigma, 0.0]);
        let near = |xi: f64| GeneralizedExtremeValue::log_prob_derivatives(x, [mu, sigma, xi]);
        let numeric = (near(1e-4).value - near(-1e-4).value) / 2e-4;
        assert!((at_zero.gradient[3] - numeric).abs() < 1e-6, "x = {x}");
        let numeric = (near(1e-4).gradient[3] - near(-1e-4).gradient[3]) / 2e-4;
        assert!((at_zero.hessian[3][3] - numeric).abs() < 1e-5, "x = {x}");
    }
}

#[test]
fn gev_log_cdf_matches_the_closed_form() {
    let (mu, sigma) = (0.3, 1.7);
    for xi in [0.4_f64, -0.25, 1e-3] {
        for x in [-1.0, 0.3, 2.0] {
            let t = (1.0 + xi * (x - mu) / sigma).powf(-1.0 / xi);
            let gev = GeneralizedExtremeValue::new(s(mu), s(sigma), s(xi));
            let log_cdf = gev.log_cdf(&s(x)).unwrap().as_slice()[0];
            let log_ccdf = gev.log_ccdf(&s(x)).unwrap().as_slice()[0];
            assert!(
                (log_cdf + t).abs() < 1e-12 * t.max(1.0),
                "ξ = {xi}, x = {x}"
            );
            assert!((log_ccdf - (-(-t).exp_m1()).ln()).abs() < 1e-12);
        }
    }
}

#[test]
fn tails_stay_finite_and_monotone() {
    // the skew normal's CDF is Φ(z) − 2T(z, α), not in logs (as in Stan), so it underflows in its tails
    for f in families()
        .into_iter()
        .filter(|f| !f.name.starts_with("skew normal"))
    {
        let Some(cdf) = &f.cdf else { continue };
        // far into each infinite tail: the log probability keeps decreasing instead of reaching −∞ or NaN
        let tails: [(bool, Vec<f64>); 2] = [
            (
                true,
                if f.upper.is_infinite() {
                    vec![1e1, 1e2, 1e3]
                } else {
                    vec![]
                },
            ),
            (
                false,
                if f.lower.is_infinite() {
                    vec![-1e1, -3e1, -1e2]
                } else {
                    vec![]
                },
            ),
        ];
        for (upper, xs) in tails {
            let values: Vec<f64> = xs.iter().map(|&x| cdf(x, &f.params, upper).0).collect();
            for (w, x) in values.windows(2).zip(&xs) {
                assert!(
                    w[1].is_finite() && w[1] < w[0],
                    "{} (upper {upper}) beyond {x}: {values:?}",
                    f.name
                );
            }
        }
    }
}
