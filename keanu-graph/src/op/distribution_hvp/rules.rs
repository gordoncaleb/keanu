//! Element-wise second derivatives of univariate log densities and log CDFs, as scalar rules.
//!
//! Each rule fills the Hessian of one element's log density (or log CDF or survival function) with respect to its
//! inputs: index 0 is the value (or, for a CDF, the point it is evaluated at) and index `j + 1` the j-th parameter,
//! in the distribution's parameter order. Integer parameters and discrete values are passed as f64s; their rows are
//! never read. Each rule is derived from the first-order gradient next to it in `keanu-distributions`, and where
//! the log density is −∞ the result is unspecified, as the gradient's is.
//!
//! Most rules are compositions `g(z(inputs))` of a standard function with a transformed coordinate `z` (the
//! standardised value of a location-scale family, `x / θ` for the gamma, `(x / β)^α` for the Weibull), whose
//! Hessian is `g''(z) ∇z ∇zᵀ + g'(z) ∇²z` ([`chain`]). For a log CDF `λ = ln F(z)` with standard log density
//! `ln f`, `λ' = f / F` and `λ'' = λ' (ψ − λ')` where `ψ = (ln f)'`; the same holds for the log survival function
//! with `λ' = −f / S`. The first derivative `λ'` is recovered from the first-order gradient, so the two agree.

use super::super::distribution::DistributionKind;
use keanu_distributions as d;
use keanu_tensor::special::{
    digamma, log_bessel_i0, log_bessel_i1, log_gamma_p_da2, log_gamma_q_da2, log_inc_beta_da2,
    log_inc_beta_dadb, log_inc_beta_db2, trigamma,
};

/// The most inputs (value and parameters, a truncated distribution's value, base parameters and bounds, or a
/// zero-inflated or hurdle truncated count's, with ψ) any element-wise rule has.
pub(crate) const MAX_INPUTS: usize = 7;

/// A symmetric Hessian of one element's log density with respect to up to [`MAX_INPUTS`] inputs.
#[derive(Debug, Clone)]
pub(crate) struct Hessian {
    n: usize,
    h: [f64; MAX_INPUTS * MAX_INPUTS],
}

impl Hessian {
    /// A zero Hessian over `n` inputs.
    pub(crate) fn new(n: usize) -> Self {
        debug_assert!(n <= MAX_INPUTS);
        Self {
            n,
            h: [0.0; MAX_INPUTS * MAX_INPUTS],
        }
    }

    pub(crate) fn clear(&mut self) {
        self.h = [0.0; MAX_INPUTS * MAX_INPUTS];
    }

    pub(crate) fn get(&self, i: usize, j: usize) -> f64 {
        self.h[i * self.n + j]
    }

    /// Adds `v` to the (i, j) and (j, i) entries (once on the diagonal).
    pub(crate) fn add(&mut self, i: usize, j: usize, v: f64) {
        self.h[i * self.n + j] += v;
        if i != j {
            self.h[j * self.n + i] += v;
        }
    }
}

/// Adds the Hessian of `g(z)` given `g'(z)`, `g''(z)`, the nonzero entries of ∇z as `(input, ∂z)` and of the upper
/// triangle of ∇²z as `(input, input, ∂²z)`.
fn chain(h: &mut Hessian, g1: f64, g2: f64, dz: &[(usize, f64)], d2z: &[(usize, usize, f64)]) {
    for (a, &(i, di)) in dz.iter().enumerate() {
        for &(j, dj) in &dz[a..] {
            h.add(i, j, g2 * di * dj);
        }
    }
    if g1 != 0.0 {
        for &(i, j, d) in d2z {
            h.add(i, j, g1 * d);
        }
    }
}

/// Adds the Hessian of `g((x − l) / s)` with respect to the value (input 0), the location (input `location`, if
/// it is one) and the scale (input `scale`).
fn location_scale(
    h: &mut Hessian,
    z: f64,
    s: f64,
    location: Option<usize>,
    scale: usize,
    g1: f64,
    g2: f64,
) {
    let s2 = s * s;
    match location {
        Some(l) => chain(
            h,
            g1,
            g2,
            &[(0, 1.0 / s), (l, -1.0 / s), (scale, -z / s)],
            &[
                (0, scale, -1.0 / s2),
                (l, scale, 1.0 / s2),
                (scale, scale, 2.0 * z / s2),
            ],
        ),
        None => chain(
            h,
            g1,
            g2,
            &[(0, 1.0 / s), (scale, -z / s)],
            &[(0, scale, -1.0 / s2), (scale, scale, 2.0 * z / s2)],
        ),
    }
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn sign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `count / q²`, taken as 0 for a zero count (as the first-order rules take `count / q`).
fn count_over_square(count: f64, q: f64) -> f64 {
    if count == 0.0 { 0.0 } else { count / (q * q) }
}

/// The first and second derivatives (g', g'') of a standard location-scale log density g(z).
type Standard = fn(f64) -> (f64, f64);

/// ln φ(z) = −z²/2 + c.
fn standard_gaussian(z: f64) -> (f64, f64) {
    (-z, -1.0)
}

/// −ln(1 + z²) + c.
fn standard_cauchy(z: f64) -> (f64, f64) {
    let q = 1.0 + z * z;
    (-2.0 * z / q, -2.0 * (1.0 - z * z) / (q * q))
}

/// −|z| + c: its second derivative is 0 everywhere but at 0, where the density has a kink and no second
/// derivative; the rule takes the almost-everywhere value 0 there too (and the first-order rule takes the
/// subgradient 0).
fn standard_laplace(z: f64) -> (f64, f64) {
    (-sign(z), 0.0)
}

/// −z − 2 ln(1 + e^−z).
fn standard_logistic(z: f64) -> (f64, f64) {
    let p = sigmoid(z);
    (1.0 - 2.0 * p, -2.0 * p * (1.0 - p))
}

/// The Hessian of a location-scale density `g((x − l) / s) − ln s` in (x, l, s), or in (x, s) for a half
/// distribution (location 0).
fn location_scale_density(
    h: &mut Hessian,
    x: f64,
    l: f64,
    s: f64,
    location: Option<usize>,
    g: Standard,
) {
    let scale = location.map_or(1, |l| l + 1);
    let z = (x - l) / s;
    let (g1, g2) = g(z);
    location_scale(h, z, s, location, scale, g1, g2);
    h.add(scale, scale, 1.0 / (s * s));
}

/// The rule for a log density: fills the Hessian of one element given the value and the parameters.
pub(crate) type DensityRule = fn(x: f64, p: &[f64], h: &mut Hessian);

/// The log density rule of an element-wise kind, None for a kind that isn't element-wise (or is a wrapper).
pub(crate) fn density_rule(kind: &DistributionKind) -> Option<DensityRule> {
    use DistributionKind as K;
    let rule: DensityRule = match kind {
        K::Gaussian => {
            |x, p, h| location_scale_density(h, x, p[0], p[1], Some(1), standard_gaussian)
        }
        K::HalfGaussian => {
            |x, p, h| location_scale_density(h, x, 0.0, p[0], None, standard_gaussian)
        }
        K::Cauchy => |x, p, h| location_scale_density(h, x, p[0], p[1], Some(1), standard_cauchy),
        K::HalfCauchy => |x, p, h| location_scale_density(h, x, 0.0, p[0], None, standard_cauchy),
        K::Laplace => |x, p, h| location_scale_density(h, x, p[0], p[1], Some(1), standard_laplace),
        K::Logistic => {
            |x, p, h| location_scale_density(h, x, p[0], p[1], Some(1), standard_logistic)
        }
        K::LogNormal => log_normal,
        K::Beta => beta,
        K::ChiSquared => |x, p, h| h.add(0, 0, -(p[0] / 2.0 - 1.0) / (x * x)),
        K::Exponential => |_, p, h| {
            // ln r − x r
            h.add(0, 1, -1.0);
            h.add(1, 1, -1.0 / (p[0] * p[0]));
        },
        K::Gamma => gamma,
        K::InverseGamma => inverse_gamma,
        K::Pareto => pareto,
        K::StudentT => |t, p, h| {
            // −(v + 1)/2 ln(1 + t²/v)
            let v = p[0];
            let q = t * t + v;
            h.add(0, 0, -(v + 1.0) * (v - t * t) / (q * q));
        },
        K::Triangular => triangular,
        // the log density is constant inside the support and, as in first order, has no derivative with respect
        // to the bounds (and the hypergeometric's parameters are all integers)
        K::Uniform | K::UniformInt | K::HyperGeometric => |_, _, _| {},
        K::Weibull => weibull,
        K::VonMises => von_mises,
        K::Bernoulli => |x, p, h| {
            let p = p[0];
            if p > 0.0 && p <= 1.0 {
                let q = if x != 0.0 { p } else { p - 1.0 };
                h.add(1, 1, -1.0 / (q * q));
            }
        },
        K::Binomial => |k, p, h| {
            let (q, n) = (p[0], p[1]);
            h.add(
                1,
                1,
                -count_over_square(k, q) - count_over_square(n - k, 1.0 - q),
            );
        },
        K::Geometric => |k, p, h| {
            if k >= 1.0 {
                let q = p[0];
                h.add(1, 1, -1.0 / (q * q) - (k - 1.0) / ((1.0 - q) * (1.0 - q)));
            }
        },
        K::Poisson => |k, p, h| h.add(1, 1, -k / (p[0] * p[0])),
        K::NegativeBinomial => negative_binomial,
        K::BetaBinomial => beta_binomial,
        // ln σ(±η): −σ(η)σ(−η)
        K::BernoulliLogit => |_, p, h| h.add(1, 1, -sigmoid(p[0]) * sigmoid(-p[0])),
        // k ln σ(η) + (n − k) ln σ(−η): −nσ(η)σ(−η)
        K::BinomialLogit => |k, p, h| {
            let (eta, n) = (p[0], p[1]);
            if k >= 0.0 && k <= n {
                h.add(1, 1, -n * sigmoid(eta) * sigmoid(-eta));
            }
        },
        // kη − e^η
        K::PoissonLog => |k, p, h| {
            if k >= 0.0 {
                h.add(1, 1, -p[0].exp());
            }
        },
        // the families written as one scalar formula, differentiated to second order in forward mode
        K::SkewNormal => |x, p, h| derived(h, d::SkewNormal::log_prob_derivatives(x, at(p))),
        K::ExGaussian => |x, p, h| derived(h, d::ExGaussian::log_prob_derivatives(x, at(p))),
        K::Gumbel => |x, p, h| derived(h, d::Gumbel::log_prob_derivatives(x, at(p))),
        K::GeneralizedExtremeValue => |x, p, h| {
            derived(
                h,
                d::GeneralizedExtremeValue::log_prob_derivatives(x, at(p)),
            )
        },
        K::GeneralizedPareto => {
            |x, p, h| derived(h, d::GeneralizedPareto::log_prob_derivatives(x, at(p)))
        }
        K::InverseGaussian => {
            |x, p, h| derived(h, d::InverseGaussian::log_prob_derivatives(x, at(p)))
        }
        K::Kumaraswamy => |x, p, h| derived(h, d::Kumaraswamy::log_prob_derivatives(x, at(p))),
        K::LogitNormal => |x, p, h| derived(h, d::LogitNormal::log_prob_derivatives(x, at(p))),
        K::HalfStudentT => |x, p, h| derived(h, d::HalfStudentT::log_prob_derivatives(x, at(p))),
        K::AsymmetricLaplace => {
            |x, p, h| derived(h, d::AsymmetricLaplace::log_prob_derivatives(x, at(p)))
        }
        K::LogLogistic => |x, p, h| derived(h, d::LogLogistic::log_prob_derivatives(x, at(p))),
        K::Rice => |x, p, h| derived(h, d::Rice::log_prob_derivatives(x, at(p))),
        K::GeneralizedGamma => {
            |x, p, h| derived(h, d::GeneralizedGamma::log_prob_derivatives(x, at(p)))
        }
        K::Skellam => |x, p, h| derived(h, d::Skellam::log_prob_derivatives(x, at(p))),
        _ => return None,
    };
    Some(rule)
}

/// The first `P` parameters as an array.
fn at<const P: usize>(p: &[f64]) -> [f64; P] {
    std::array::from_fn(|i| p[i])
}

/// Adds the Hessian of a family written as one scalar formula, from its forward-mode derivatives (see
/// [`keanu_distributions::ElementDerivatives`]).
fn derived<const N: usize>(h: &mut Hessian, d: keanu_distributions::ElementDerivatives<N>) {
    for i in 0..N {
        for j in i..N {
            h.add(i, j, d.hessian[i][j]);
        }
    }
}

/// ln C(n, k) + ln B(k + α, n − k + β) − ln B(α, β) (inputs k, n, α, β), with t = ψ₁(α + β) − ψ₁(n + α + β):
/// ψ₁(k + α) − ψ₁(α) + t, t and ψ₁(n − k + β) − ψ₁(β) + t.
fn beta_binomial(k: f64, p: &[f64], h: &mut Hessian) {
    let (n, a, b) = (p[0], p[1], p[2]);
    if k < 0.0 || k > n {
        return;
    }
    let shared = trigamma(a + b) - trigamma(n + a + b);
    h.add(2, 2, trigamma(k + a) - trigamma(a) + shared);
    h.add(2, 3, shared);
    h.add(3, 3, trigamma(n - k + b) - trigamma(b) + shared);
}

/// −(ln x − m)²/(2s²) − ln s − ln x, with z = (ln x − m)/s.
fn log_normal(x: f64, p: &[f64], h: &mut Hessian) {
    let (m, s) = (p[0], p[1]);
    let z = (x.ln() - m) / s;
    let s2 = s * s;
    chain(
        h,
        -z,
        -1.0,
        &[(0, 1.0 / (s * x)), (1, -1.0 / s), (2, -z / s)],
        &[
            (0, 0, -1.0 / (s * x * x)),
            (0, 2, -1.0 / (s2 * x)),
            (1, 2, 1.0 / s2),
            (2, 2, 2.0 * z / s2),
        ],
    );
    h.add(2, 2, 1.0 / s2);
    h.add(0, 0, 1.0 / (x * x));
}

/// (a − 1) ln x + (b − 1) ln(1 − x) − ln B(a, b), on [0, 1].
fn beta(x: f64, p: &[f64], h: &mut Hessian) {
    let (a, b) = (p[0], p[1]);
    let y = 1.0 - x;
    let both = trigamma(a + b);
    h.add(0, 0, -(a - 1.0) / (x * x) - (b - 1.0) / (y * y));
    h.add(0, 1, 1.0 / x);
    h.add(0, 2, -1.0 / y);
    h.add(1, 1, both - trigamma(a));
    h.add(1, 2, both);
    h.add(2, 2, both - trigamma(b));
}

/// (k − 1) ln x − ln Γ(k) − x/θ − k ln θ.
fn gamma(x: f64, p: &[f64], h: &mut Hessian) {
    let (theta, k) = (p[0], p[1]);
    let t2 = theta * theta;
    h.add(0, 0, -(k - 1.0) / (x * x));
    h.add(0, 1, 1.0 / t2);
    h.add(0, 2, 1.0 / x);
    h.add(1, 1, -2.0 * x / (t2 * theta) + k / t2);
    h.add(1, 2, -1.0 / theta);
    h.add(2, 2, -trigamma(k));
}

/// a ln b − (a + 1) ln x − ln Γ(a) − b/x.
fn inverse_gamma(x: f64, p: &[f64], h: &mut Hessian) {
    let (a, b) = (p[0], p[1]);
    let x2 = x * x;
    h.add(0, 0, (a + 1.0) / x2 - 2.0 * b / (x2 * x));
    h.add(0, 1, -1.0 / x);
    h.add(0, 2, 1.0 / x2);
    h.add(1, 1, -trigamma(a));
    h.add(1, 2, 1.0 / b);
    h.add(2, 2, -a / (b * b));
}

/// ln s + s ln l − (s + 1) ln x, with l the location (minimum) and s the scale (shape).
fn pareto(x: f64, p: &[f64], h: &mut Hessian) {
    let (l, s) = (p[0], p[1]);
    h.add(0, 0, (s + 1.0) / (x * x));
    h.add(0, 2, -1.0 / x);
    h.add(1, 1, -s / (l * l));
    h.add(1, 2, 1.0 / l);
    h.add(2, 2, -1.0 / (s * s));
}

/// Piecewise: ln 2 + ln(x − a) − ln(b − a) − ln(c − a) below the mode c, ln 2 − ln(b − a) at it, and
/// ln 2 + ln(b − x) − ln(b − a) − ln(b − c) above it (inputs x, a, b, c). Each piece is smooth; the density's kinks
/// at a, b and c have no second derivative and take the adjoining piece's (as the first-order rule does).
fn triangular(x: f64, p: &[f64], h: &mut Hessian) {
    let (a, b, c) = (p[0], p[1], p[2]);
    let range = 1.0 / ((b - a) * (b - a));
    if x >= a && x < c {
        let from_a = if x - a > 0.0 {
            1.0 / ((x - a) * (x - a))
        } else {
            0.0
        };
        let mode = 1.0 / ((c - a) * (c - a));
        h.add(0, 0, -from_a);
        h.add(0, 1, from_a);
        h.add(1, 1, -from_a + range + mode);
        h.add(1, 2, -range);
        h.add(1, 3, -mode);
        h.add(2, 2, range);
        h.add(3, 3, mode);
    } else if x == c {
        h.add(1, 1, range);
        h.add(1, 2, -range);
        h.add(2, 2, range);
    } else if x > c && x <= b {
        let to_b = if b - x > 0.0 {
            1.0 / ((b - x) * (b - x))
        } else {
            0.0
        };
        let mode = 1.0 / ((b - c) * (b - c));
        h.add(0, 0, -to_b);
        h.add(0, 2, to_b);
        h.add(1, 1, range);
        h.add(1, 2, -range);
        h.add(2, 2, -to_b + range + mode);
        h.add(2, 3, -mode);
        h.add(3, 3, mode);
    }
}

/// ln a − ln b + (a − 1) L − t with L = ln(x/b) and t = (x/b)^a (inputs x, a, b).
fn weibull(x: f64, p: &[f64], h: &mut Hessian) {
    let (a, b) = (p[0], p[1]);
    let l = (x / b).ln();
    let t = (x / b).powf(a);
    let x2 = x * x;
    h.add(0, 0, -(a - 1.0) * (1.0 + a * t) / x2);
    h.add(0, 1, (1.0 - t - a * t * l) / x);
    h.add(0, 2, a * a * t / (x * b));
    h.add(1, 1, -1.0 / (a * a) - t * l * l);
    h.add(1, 2, (t + a * t * l - 1.0) / b);
    h.add(2, 2, -a * (a * t + t - 1.0) / (b * b));
}

/// κ cos(x − μ) − ln 2π − ln I₀(κ) (inputs x, μ, κ). With A = I₁/I₀ (so d ln I₀/dκ = A) and I₁' = I₀ − I₁/κ,
/// d²/dκ² ln I₀(κ) = A' = 1 − A/κ − A².
fn von_mises(x: f64, p: &[f64], h: &mut Hessian) {
    let (mu, kappa) = (p[0], p[1]);
    let (sin, cos) = (x - mu).sin_cos();
    let a = (log_bessel_i1(kappa) - log_bessel_i0(kappa)).exp();
    // A/κ → 1/2 as κ → 0, where the ratio loses precision: A = κ/2 − κ³/16 + O(κ⁵)
    let a_over_kappa = if kappa < 1e-4 {
        0.5 - kappa * kappa / 16.0
    } else {
        a / kappa
    };
    h.add(0, 0, -kappa * cos);
    h.add(0, 1, kappa * cos);
    h.add(1, 1, -kappa * cos);
    h.add(0, 2, -sin);
    h.add(1, 2, sin);
    h.add(2, 2, -(1.0 - a_over_kappa - a * a));
}

/// With T = α + μ: ln C(k, α) − α ln(1 + μ/α) + k ln(μ/T) (inputs k, μ, α).
fn negative_binomial(k: f64, p: &[f64], h: &mut Hessian) {
    if k < 0.0 {
        return;
    }
    let (mu, alpha) = (p[0], p[1]);
    let total = alpha + mu;
    let t2 = total * total;
    h.add(1, 1, -k / (mu * mu) + (k + alpha) / t2);
    h.add(1, 2, (k - mu) / t2);
    h.add(
        2,
        2,
        trigamma(k + alpha) - trigamma(alpha) - 1.0 / total + 1.0 / alpha - (mu - k) / t2,
    );
}

/// The rule for a log CDF (`upper` false) or log survival function (`upper` true): fills the Hessian of one element
/// given the point, the parameters and the first-order gradient there (the point's derivative first, 0 where there
/// is none).
pub(crate) type CdfRule = fn(point: f64, p: &[f64], upper: bool, gradient: &[f64], h: &mut Hessian);

/// The log CDF rule of a kind with a CDF. Every rule gives the second derivatives with respect to the point and
/// every parameter with a first-order gradient; those of CDFs built on the regularised incomplete gamma and beta
/// functions with respect to their shapes come from the shape derivatives of those functions in
/// [`keanu_tensor::special`], and the mixed ones with the point from the shape derivative of the density.
pub(crate) fn cdf_rule(kind: &DistributionKind) -> Option<CdfRule> {
    use DistributionKind as K;
    let rule: CdfRule = match kind {
        K::Gaussian => |x, p, _, g, h| standard_cdf(h, x, p, g, |z| -z),
        K::Cauchy => |x, p, _, g, h| standard_cdf(h, x, p, g, |z| -2.0 * z / (1.0 + z * z)),
        K::Laplace => |x, p, _, g, h| standard_cdf(h, x, p, g, |z| -sign(z)),
        K::Logistic => |x, p, _, g, h| standard_cdf(h, x, p, g, |z| 1.0 - 2.0 * sigmoid(z)),
        K::LogNormal => log_normal_cdf,
        K::Exponential => exponential_cdf,
        K::Weibull => weibull_cdf,
        K::Uniform => uniform_cdf,
        K::Gamma => gamma_cdf,
        K::Beta => beta_cdf,
        // the degrees of freedom are an integer, without derivatives
        K::StudentT => |t, p, _, g, h| {
            let v = p[0];
            let psi = -(v + 1.0) * t / (v + t * t);
            h.add(0, 0, g[0] * (psi - g[0]));
        },
        K::Poisson => |k, p, _, g, h| {
            if k >= 0.0 {
                h.add(1, 1, g[1] * (k / p[0] - 1.0 - g[1]));
            }
        },
        K::Binomial => binomial_cdf,
        K::NegativeBinomial => negative_binomial_cdf,
        // the tail is a finite sum, whose Hessian in the shapes is summed with it (inputs k, n, α, β)
        K::SkewNormal => {
            |x, p, u, _, h| derived(h, d::SkewNormal::log_cdf_derivatives(x, at(p), u))
        }
        K::ExGaussian => {
            |x, p, u, _, h| derived(h, d::ExGaussian::log_cdf_derivatives(x, at(p), u))
        }
        K::Gumbel => |x, p, u, _, h| derived(h, d::Gumbel::log_cdf_derivatives(x, at(p), u)),
        K::GeneralizedExtremeValue => |x, p, u, _, h| {
            derived(
                h,
                d::GeneralizedExtremeValue::log_cdf_derivatives(x, at(p), u),
            )
        },
        K::GeneralizedPareto => {
            |x, p, u, _, h| derived(h, d::GeneralizedPareto::log_cdf_derivatives(x, at(p), u))
        }
        K::InverseGaussian => {
            |x, p, u, _, h| derived(h, d::InverseGaussian::log_cdf_derivatives(x, at(p), u))
        }
        K::Kumaraswamy => {
            |x, p, u, _, h| derived(h, d::Kumaraswamy::log_cdf_derivatives(x, at(p), u))
        }
        K::LogitNormal => {
            |x, p, u, _, h| derived(h, d::LogitNormal::log_cdf_derivatives(x, at(p), u))
        }
        K::HalfStudentT => {
            |x, p, u, _, h| derived(h, d::HalfStudentT::log_cdf_derivatives(x, at(p), u))
        }
        K::AsymmetricLaplace => {
            |x, p, u, _, h| derived(h, d::AsymmetricLaplace::log_cdf_derivatives(x, at(p), u))
        }
        K::LogLogistic => {
            |x, p, u, _, h| derived(h, d::LogLogistic::log_cdf_derivatives(x, at(p), u))
        }
        K::GeneralizedGamma => {
            |x, p, u, _, h| derived(h, d::GeneralizedGamma::log_cdf_derivatives(x, at(p), u))
        }
        K::BetaBinomial => |k, p, upper, _, h| {
            let tail = keanu_distributions::BetaBinomial::log_tail(
                k as i64,
                p[0] as i64,
                p[1],
                p[2],
                upper,
            );
            h.add(2, 2, tail.hessian[0]);
            h.add(2, 3, tail.hessian[1]);
            h.add(3, 3, tail.hessian[2]);
        },
        _ => return None,
    };
    Some(rule)
}

/// A location-scale log CDF λ(z), z = (x − l)/s, in (x, l, s), given ψ(z), the derivative of the standard log
/// density: λ' = s ∂λ/∂x from the gradient.
fn standard_cdf(h: &mut Hessian, x: f64, p: &[f64], g: &[f64], psi: fn(f64) -> f64) {
    let (l, s) = (p[0], p[1]);
    let z = (x - l) / s;
    let d1 = s * g[0];
    location_scale(h, z, s, Some(1), 2, d1, d1 * (psi(z) - d1));
}

/// The log-normal: the Gaussian's in z = (ln x − m)/s, with λ' = −s ∂λ/∂m.
fn log_normal_cdf(x: f64, p: &[f64], _: bool, g: &[f64], h: &mut Hessian) {
    if x <= 0.0 {
        return;
    }
    let (m, s) = (p[0], p[1]);
    let z = (x.ln() - m) / s;
    let d1 = -s * g[1];
    let s2 = s * s;
    chain(
        h,
        d1,
        d1 * (-z - d1),
        &[(0, 1.0 / (s * x)), (1, -1.0 / s), (2, -z / s)],
        &[
            (0, 0, -1.0 / (s * x * x)),
            (0, 2, -1.0 / (s2 * x)),
            (1, 2, 1.0 / s2),
            (2, 2, 2.0 * z / s2),
        ],
    );
}

/// The standard exponential's log CDF ln(1 − e^−t) or log survival function −t: (λ', λ'').
fn standard_exponential(t: f64, upper: bool) -> (f64, f64) {
    if upper {
        (-1.0, 0.0)
    } else {
        let d1 = 1.0 / t.exp_m1();
        (d1, d1 * (-1.0 - d1))
    }
}

/// In t = r x (inputs x, r).
fn exponential_cdf(x: f64, p: &[f64], upper: bool, _: &[f64], h: &mut Hessian) {
    if x <= 0.0 {
        return;
    }
    let r = p[0];
    let (d1, d2) = standard_exponential(r * x, upper);
    chain(h, d1, d2, &[(0, r), (1, x)], &[(0, 1, 1.0)]);
}

/// In t = (x/b)^a (inputs x, a, b), with L = ln(x/b).
fn weibull_cdf(x: f64, p: &[f64], upper: bool, _: &[f64], h: &mut Hessian) {
    if x <= 0.0 {
        return;
    }
    let (a, b) = (p[0], p[1]);
    let l = (x / b).ln();
    let t = (x / b).powf(a);
    let (d1, d2) = standard_exponential(t, upper);
    chain(
        h,
        d1,
        d2,
        &[(0, a * t / x), (1, t * l), (2, -a * t / b)],
        &[
            (0, 0, a * (a - 1.0) * t / (x * x)),
            (0, 1, t * (1.0 + a * l) / x),
            (0, 2, -a * a * t / (x * b)),
            (1, 1, t * l * l),
            (1, 2, -t * (1.0 + a * l) / b),
            (2, 2, a * (a + 1.0) * t / (b * b)),
        ],
    );
}

/// ln(x − a) − ln(b − a), or ln(b − x) − ln(b − a) for the survival function, between the bounds (inputs x, a, b).
fn uniform_cdf(x: f64, p: &[f64], upper: bool, _: &[f64], h: &mut Hessian) {
    let (a, b) = (p[0], p[1]);
    if x <= a || x >= b {
        return;
    }
    let range = 1.0 / ((b - a) * (b - a));
    if upper {
        let to_b = 1.0 / ((b - x) * (b - x));
        h.add(0, 0, -to_b);
        h.add(0, 2, to_b);
        h.add(2, 2, -to_b + range);
        h.add(1, 2, -range);
        h.add(1, 1, range);
    } else {
        let from_a = 1.0 / ((x - a) * (x - a));
        h.add(0, 0, -from_a);
        h.add(0, 1, from_a);
        h.add(1, 1, -from_a + range);
        h.add(1, 2, -range);
        h.add(2, 2, range);
    }
}

/// The regularised incomplete gamma P(k, y) (or Q) in y = x/θ, with ψ = (k − 1)/y − 1 and λ_y = θ ∂λ/∂x (inputs
/// x, θ, k). In the shape, λ_kk is the incomplete gamma function's own second shape derivative and
/// λ_ky = λ_y (∂ ln p/∂k − λ_k), with p(y) = y^(k−1) e^(−y) / Γ(k) the density P and Q differentiate to (±) in y,
/// so ∂ ln p/∂k = ln y − ψ(k).
fn gamma_cdf(x: f64, p: &[f64], upper: bool, g: &[f64], h: &mut Hessian) {
    if x <= 0.0 || x == f64::INFINITY {
        return;
    }
    let (theta, k) = (p[0], p[1]);
    let y = x / theta;
    let d1 = theta * g[0];
    let t2 = theta * theta;
    chain(
        h,
        d1,
        d1 * ((k - 1.0) / y - 1.0 - d1),
        &[(0, 1.0 / theta), (1, -x / t2)],
        &[(0, 1, -1.0 / t2), (1, 1, 2.0 * x / (t2 * theta))],
    );
    let d_ky = d1 * (y.ln() - digamma(k) - g[2]);
    h.add(0, 2, d_ky / theta);
    h.add(1, 2, -d_ky * x / t2);
    h.add(
        2,
        2,
        if upper {
            log_gamma_q_da2(k, y)
        } else {
            log_gamma_p_da2(k, y)
        },
    );
}

/// The regularised incomplete beta I_x(a, b), or its complement I_(1−x)(b, a) (inputs x, a, b). In the shapes,
/// λ's second derivatives are the incomplete beta function's own, and the mixed ones with the point are
/// λ_xa = λ_x (∂ ln f/∂a − λ_a) with f the Beta(a, b) density, whose log has derivatives ln x − ψ(a) + ψ(a + b)
/// and ln(1 − x) − ψ(b) + ψ(a + b).
fn beta_cdf(x: f64, p: &[f64], upper: bool, g: &[f64], h: &mut Hessian) {
    if x <= 0.0 || x >= 1.0 {
        return;
    }
    let (a, b) = (p[0], p[1]);
    let y = 1.0 - x;
    let psi = (a - 1.0) / x - (b - 1.0) / y;
    h.add(0, 0, g[0] * (psi - g[0]));
    let both = digamma(a + b);
    h.add(0, 1, g[0] * (x.ln() - digamma(a) + both - g[1]));
    h.add(0, 2, g[0] * (y.ln() - digamma(b) + both - g[2]));
    let (aa, ab, bb) = if upper {
        (
            log_inc_beta_db2(b, a, y),
            log_inc_beta_dadb(b, a, y),
            log_inc_beta_da2(b, a, y),
        )
    } else {
        (
            log_inc_beta_da2(a, b, x),
            log_inc_beta_dadb(a, b, x),
            log_inc_beta_db2(a, b, x),
        )
    };
    h.add(1, 1, aa);
    h.add(1, 2, ab);
    h.add(2, 2, bb);
}

/// In p, whose derivative of the CDF is minus the Beta(k + 1, n − k) density at p (inputs k, p, n).
fn binomial_cdf(k: f64, p: &[f64], _: bool, g: &[f64], h: &mut Hessian) {
    let (q, n) = (p[0], p[1]);
    if k < 0.0 || k >= n {
        return;
    }
    let ratio = |count: f64, r: f64| if count == 0.0 { 0.0 } else { count / r };
    let psi = ratio(k, q) - ratio(n - k - 1.0, 1.0 - q);
    h.add(1, 1, g[1] * (psi - g[1]));
}

/// I_p(α, k + 1), or its complement I_(1−p)(k + 1, α), in α and p = α/(α + μ), whose derivative in p is (±) the
/// Beta(α, k + 1) density f at p (inputs k, μ, α). With λ_α the derivative at fixed p (the gradient's less
/// λ_p p_α), λ_αα the incomplete beta function's own second shape derivative and λ_αp = λ_p (∂ ln f/∂α − λ_α),
/// ∂ ln f/∂α = ln p − ψ(α) + ψ(α + k + 1):
/// λ_μμ = λ_pp p_μ² + λ_p p_μμ, λ_μα = λ_pp p_μ p_α + λ_p p_μα + λ_αp p_μ and
/// λ_αα (total) = λ_αα + 2 λ_αp p_α + λ_pp p_α² + λ_p p_αα.
fn negative_binomial_cdf(k: f64, p: &[f64], upper: bool, g: &[f64], h: &mut Hessian) {
    if k < 0.0 {
        return;
    }
    let (mu, alpha) = (p[0], p[1]);
    let total = alpha + mu;
    let dp = -alpha / (total * total);
    let d2p = 2.0 * alpha / (total * total * total);
    let (q_p, q_q) = (alpha / total, mu / total);
    let d1 = g[1] / dp;
    let psi = (alpha - 1.0) / q_p - k / q_q;
    let d2 = d1 * (psi - d1);
    h.add(1, 1, d2 * dp * dp + d1 * d2p);
    // p_α = μ/T², p_αα = −2μ/T³ and p_μα = (α − μ)/T³
    let t3 = total * total * total;
    let (dp_a, d2p_a, d2p_ma) = (mu / (total * total), -2.0 * mu / t3, (alpha - mu) / t3);
    let count = k + 1.0;
    let d_alpha = g[2] - d1 * dp_a;
    let d_ap = d1 * (q_p.ln() - digamma(alpha) + digamma(alpha + count) - d_alpha);
    let d_aa = if upper {
        log_inc_beta_db2(count, alpha, q_q)
    } else {
        log_inc_beta_da2(alpha, count, q_p)
    };
    h.add(1, 2, d2 * dp * dp_a + d1 * d2p_ma + d_ap * dp);
    h.add(
        2,
        2,
        d_aa + 2.0 * d_ap * dp_a + d2 * dp_a * dp_a + d1 * d2p_a,
    );
}
