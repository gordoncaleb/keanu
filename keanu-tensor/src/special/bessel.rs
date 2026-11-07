//! Modified Bessel functions of real order ν: K_ν(x) and I_ν(x) (in logs and as ratios), with their derivatives
//! in x.
//!
//! The algorithms follow published descriptions (no code is copied):
//!
//! - K_μ and K_{μ+1} for |μ| <= ½ by Temme's series for x <= 2 (Temme 1975, as described in Numerical Recipes
//!   §6.7) and Steed's continued fraction CF2 for x > 2 (ibid.), then the forward recurrence
//!   K_{μ+k+1} = K_{μ+k−1} + 2(μ+k)/x K_{μ+k} (stable for K), carried as ratios and logs so it never overflows;
//! - the ratio I_{ν+1}/I_ν by its continued fraction 1/(2(ν+1)/x + 1/(2(ν+2)/x + …)) (DLMF 10.33.1), and ln I_ν by
//!   its power series (DLMF 10.25.2) for x <= 30, the Hankel expansion (DLMF 10.40.1) for x >= max(30, ν²), and
//!   otherwise the Wronskian I_ν K_{ν+1} + I_{ν+1} K_ν = 1/x (DLMF 10.28.2);
//! - for ν >= 30, Debye's uniform asymptotic expansions (DLMF 10.41.3, 10.41.4) of I_ν, K_ν and their derivatives,
//!   with fourteen terms, whose truncation error is below 1e-17 there.

use super::{Jet, continued_fraction, ln_gamma};
use std::f64::consts::{LN_2, PI};

/// At and above this order, the Debye expansions are used.
const DEBYE_MIN_ORDER: f64 = 30.0;

/// Below this argument (and order [`DEBYE_MIN_ORDER`]), ln I_ν is summed from its power series.
const SERIES_MAX_X: f64 = 30.0;

/// At and above max(this, ν²), ln I_ν and the ratio I_{ν+1}/I_ν come from the Hankel expansion.
const HANKEL_MIN_X: f64 = 30.0;

/// The most terms of Steed's continued fraction and Temme's series.
const MAX_TERMS: usize = 100_000;
const RECIPROCAL_GAMMA_1P: [f64; 29] = [
    1.0,
    0.5772156649015329,
    -0.6558780715202539,
    -0.04200263503409524,
    0.16653861138229148,
    -0.04219773455554433,
    -0.009621971527876973,
    0.0072189432466631,
    -0.0011651675918590652,
    -0.00021524167411495098,
    0.0001280502823881162,
    -2.013485478078824e-05,
    -1.2504934821426706e-06,
    1.133027231981696e-06,
    -2.056338416977607e-07,
    6.116095104481416e-09,
    5.002007644469223e-09,
    -1.18127457048702e-09,
    1.0434267116911005e-10,
    7.782263439905071e-12,
    -3.696805618642206e-12,
    5.100370287454476e-13,
    -2.0583260535665066e-14,
    -5.348122539423018e-15,
    1.2267786282382608e-15,
    -1.1812593016974588e-16,
    1.1866922547516004e-18,
    1.4123806553180319e-18,
    -2.29874568443537e-19,
];
/// The Debye polynomials u_k(p) (DLMF 10.41.10), k = 0..=13, as p^k Σ_j c_j (p²)^j: the coefficients c_j.
const DEBYE_U: [&[f64]; 14] = [
    &[1.0],
    &[0.125, -0.20833333333333334],
    &[0.0703125, -0.4010416666666667, 0.3342013888888889],
    &[
        0.0732421875,
        -0.8912109375,
        1.8464626736111112,
        -1.0258125964506173,
    ],
    &[
        0.112152099609375,
        -2.3640869140625,
        8.78912353515625,
        -11.207002616222994,
        4.669584423426247,
    ],
    &[
        0.22710800170898438,
        -7.368794359479632,
        42.53499874538846,
        -91.81824154324002,
        84.63621767460073,
        -28.212072558200244,
    ],
    &[
        0.5725014209747314,
        -26.491430486951554,
        218.1905117442116,
        -699.5796273761325,
        1059.9904525279999,
        -765.2524681411817,
        212.57013003921713,
    ],
    &[
        1.7277275025844574,
        -108.09091978839466,
        1200.9029132163525,
        -5305.646978613403,
        11655.393336864534,
        -13586.550006434138,
        8061.722181737309,
        -1919.457662318407,
    ],
    &[
        6.074042001273483,
        -493.915304773088,
        7109.514302489364,
        -41192.65496889755,
        122200.46498301746,
        -203400.17728041555,
        192547.00123253153,
        -96980.59838863752,
        20204.29133096615,
    ],
    &[
        24.380529699556064,
        -2499.8304818112097,
        45218.76898136273,
        -331645.1724845636,
        1268365.2733216248,
        -2813563.226586534,
        3763271.297656404,
        -2998015.9185381066,
        1311763.6146629772,
        -242919.18790055133,
    ],
    &[
        110.01714026924674,
        -13886.08975371704,
        308186.4046126624,
        -2785618.1280864547,
        13288767.166421818,
        -37567176.66076335,
        66344512.27472903,
        -74105148.21153265,
        50952602.49266464,
        -19706819.118432228,
        3284469.853072038,
    ],
    &[
        551.3358961220206,
        -84005.43360302408,
        2243768.1779224495,
        -24474062.72573873,
        142062907.7975331,
        -495889784.2750303,
        1106842816.8230145,
        -1621080552.1083372,
        1553596899.57058,
        -939462359.6815784,
        325573074.18576574,
        -49329253.66450996,
    ],
    &[
        3038.090510922384,
        -549842.3275722887,
        17395107.553978164,
        -225105661.88941526,
        1559279864.8792574,
        -6563293792.619285,
        17954213731.1556,
        -33026599749.800724,
        41280185579.753975,
        -34632043388.158775,
        18688207509.295826,
        -5866481492.051847,
        814789096.1183121,
    ],
    &[
        18257.755474293175,
        -3871833.442572613,
        143157876.71888897,
        -2167164983.223795,
        17634730606.83497,
        -87867072178.02327,
        287900649906.1506,
        -645364869245.3765,
        1008158106865.3821,
        -1098375156081.2233,
        819218669548.5773,
        -399096175224.4665,
        114498237732.0258,
        -14679261247.695616,
    ],
];
/// The Debye polynomials v_k(p) (DLMF 10.41.11), k = 0..=13, as p^k Σ_j c_j (p²)^j: the coefficients c_j.
const DEBYE_V: [&[f64]; 14] = [
    &[1.0],
    &[-0.375, 0.2916666666666667],
    &[-0.1171875, 0.515625, -0.3949652777777778],
    &[
        -0.1025390625,
        1.0892578125,
        -2.1305338541666665,
        1.1464964313271604,
    ],
    &[
        -0.144195556640625,
        2.7939208984375,
        -9.961006673177083,
        12.386687102141204,
        -5.0756352428546165,
    ],
    &[
        -0.2775764465332031,
        8.502455030168806,
        -47.53911624484592,
        100.56283597592954,
        -91.40711508856879,
        30.15773273462785,
    ],
    &[
        -0.6765925884246826,
        30.023621218545095,
        -241.15793403307597,
        760.412638452318,
        -1138.5082638263702,
        814.6235951180321,
        -224.71699461288668,
    ],
    &[
        -1.993531733751297,
        120.80749858702931,
        -1315.2746192369575,
        5730.098736902475,
        -12459.213566993121,
        14409.977279551358,
        -8497.490948317705,
        2013.0897434071098,
    ],
    &[
        -6.883914268109947,
        545.9063894860446,
        -7727.732937488438,
        44243.96274437144,
        -130084.36594966374,
        215023.04455358215,
        -202421.2064239434,
        101491.32389508576,
        -21064.0484088796,
    ],
    &[
        -27.248827311268542,
        2737.909575317039,
        -48836.270499871745,
        354517.25334556797,
        -1345235.895947178,
        2965647.7253209413,
        -3946845.50729818,
        3131261.0704731336,
        -1365304.9866900374,
        252085.9497081193,
    ],
    &[
        -121.59789187653587,
        15093.575819257652,
        -331015.02717656334,
        2965335.4266726775,
        -14048125.290217351,
        39493698.5408025,
        -69430303.54332107,
        77258558.77372554,
        -52950743.766886786,
        20423430.722738855,
        -3395807.814193124,
    ],
    &[
        -603.8440767050702,
        90725.86829126602,
        -2398510.810882618,
        25957339.25457138,
        -149741983.89469704,
        520079529.84942204,
        -1156035830.9040372,
        1687247105.2556162,
        -1612223197.667583,
        972425951.249353,
        -336247601.20824987,
        50847076.85418719,
    ],
    &[
        -3302.2722944808525,
        590571.3888739398,
        -18517372.55746063,
        237968842.56881043,
        -1639242934.8730657,
        6868563271.345762,
        -18718222826.098392,
        34321760524.30271,
        -42781283237.19957,
        35806010960.63873,
        -19281483938.162357,
        6041600342.560858,
        -837740901.6427716,
    ],
    &[
        -19718.37591223663,
        4138856.438612103,
        -151834111.67154893,
        2284309036.3710275,
        -18494961368.143993,
        91772275385.93541,
        -299651696841.0955,
        669718260537.6549,
        -1043532075527.3253,
        1134387456280.6077,
        -844425397842.3798,
        410664180303.4365,
        -117635175752.08131,
        15060540760.622776,
    ],
];

/// Σ cⱼ qʲ by Horner's rule.
fn polynomial(coefficients: &[f64], q: f64) -> f64 {
    coefficients.iter().rev().fold(0.0, |sum, &c| sum * q + c)
}

/// The Debye expansions at order ν >= [`DEBYE_MIN_ORDER`] and argument x > 0.
struct Debye {
    /// ln I_ν(x), or ln I_ν(x) − x if scaled.
    ln_i: f64,
    /// ln K_ν(x), or ln K_ν(x) + x if scaled.
    ln_k: f64,
    /// I'_ν(x) / I_ν(x).
    i_log_derivative: f64,
    /// −K'_ν(x) / K_ν(x).
    k_log_derivative_negated: f64,
}

fn debye(nu: f64, x: f64, scaled: bool) -> Debye {
    let z = x / nu;
    let s = z.hypot(1.0);
    let p = 1.0 / s;
    let q = p * p;
    // η = √(1 + z²) + ln(z / (1 + √(1 + z²))), or η − z with √(1 + z²) − z = 1 / (z + √(1 + z²)), so that
    // νη ∓ x is not rounded twice
    // ln(z / (1 + s)) = −ln(1 + (1 + s − z)/z), accurate also where it is near 0 (large z)
    let log_part = -((1.0 + 1.0 / (z + s)) / z).ln_1p();
    let eta = if scaled {
        1.0 / (z + s) + log_part
    } else {
        s + log_part
    };
    let (mut u_plus, mut u_minus, mut v_plus, mut v_minus) = (0.0, 0.0, 0.0, 0.0);
    // (p/ν)^k
    let mut power = 1.0;
    for (k, (u, v)) in DEBYE_U.iter().zip(DEBYE_V.iter()).enumerate() {
        let (u, v) = (power * polynomial(u, q), power * polynomial(v, q));
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        u_plus += u;
        u_minus += sign * u;
        v_plus += v;
        v_minus += sign * v;
        power *= p / nu;
    }
    // (1 + z²)^¼ = √s
    let half_ln_s = 0.5 * s.ln();
    Debye {
        ln_i: nu * eta - 0.5 * (2.0 * PI * nu).ln() - half_ln_s + u_plus.ln(),
        ln_k: -nu * eta + 0.5 * (PI / (2.0 * nu)).ln() - half_ln_s + u_minus.ln(),
        i_log_derivative: s / z * (v_plus / u_plus),
        k_log_derivative_negated: s / z * (v_minus / u_minus),
    }
}

/// (1/Γ(1 + μ), 1/Γ(1 − μ), (1/Γ(1 − μ) − 1/Γ(1 + μ)) / 2μ, (1/Γ(1 − μ) + 1/Γ(1 + μ)) / 2) for |μ| <= ½, from
/// the Taylor series of 1/Γ(1 + μ), so that the third has no cancellation near μ = 0.
fn temme_gammas(mu: f64) -> (f64, f64, f64, f64) {
    let (mut odd, mut even) = (0.0, 0.0);
    // odd = Σ_{j odd} cⱼ μ^{j−1}, even = Σ_{j even} cⱼ μʲ, both series in μ²
    let mu2 = mu * mu;
    for (j, &c) in RECIPROCAL_GAMMA_1P.iter().enumerate().rev() {
        if j % 2 == 0 {
            even = even * mu2 + c;
        } else {
            odd = odd * mu2 + c;
        }
    }
    let plus = even + mu * odd;
    let minus = even - mu * odd;
    (plus, minus, -odd, even)
}

/// (ln K_μ(x), K_{μ+1}(x) / K_μ(x)) for |μ| <= ½ and 0 < x <= 2, by Temme's series.
fn temme(mu: f64, x: f64) -> (f64, f64) {
    let half = 0.5 * x;
    let pi_mu = PI * mu;
    let fact = if mu == 0.0 { 1.0 } else { pi_mu / pi_mu.sin() };
    // d = ln(2/x), e = μ d
    let d = -half.ln();
    let e = mu * d;
    let fact2 = if e == 0.0 { 1.0 } else { e.sinh() / e };
    let (gamma_plus, gamma_minus, gamma1, gamma2) = temme_gammas(mu);
    let mut f = fact * (gamma1 * e.cosh() + gamma2 * fact2 * d);
    let exp_e = e.exp();
    let mut p = 0.5 * exp_e / gamma_plus;
    let mut q = 0.5 / (exp_e * gamma_minus);
    let mut c = 1.0;
    let square = half * half;
    let mut sum = f;
    let mut sum1 = p;
    let mu2 = mu * mu;
    for i in 1..MAX_TERMS {
        let i = i as f64;
        f = (i * f + p + q) / (i * i - mu2);
        c *= square / i;
        p /= i - mu;
        q /= i + mu;
        let term = c * f;
        sum += term;
        sum1 += c * (p - i * f);
        if term.abs() < sum.abs() * f64::EPSILON {
            break;
        }
    }
    (sum.ln(), sum1 / sum * (2.0 / x))
}

/// (ln K_μ(x) + x, K_{μ+1}(x) / K_μ(x)) for |μ| <= ½ and x > 2, by Steed's algorithm for the continued fraction CF2.
fn steed(mu: f64, x: f64) -> (f64, f64) {
    let mut b = 2.0 * (1.0 + x);
    let mut d = 1.0 / b;
    let mut h = d;
    let mut delta_h = d;
    let (mut q1, mut q2) = (0.0, 1.0);
    let a1 = 0.25 - mu * mu;
    let mut q = a1;
    let mut c = a1;
    let mut a = -a1;
    let mut s = 1.0 + q * delta_h;
    for i in 2..MAX_TERMS {
        let i = i as f64;
        a -= 2.0 * (i - 1.0);
        c = -a * c / i;
        let q_next = (q1 - b * q2) / a;
        q1 = q2;
        q2 = q_next;
        q += c * q_next;
        b += 2.0;
        d = 1.0 / (b + a * d);
        delta_h *= b * d - 1.0;
        h += delta_h;
        let delta_s = q * delta_h;
        s += delta_s;
        if (delta_s / s).abs() < f64::EPSILON {
            break;
        }
    }
    let h = a1 * h;
    let ln_k_scaled = 0.5 * (PI / (2.0 * x)).ln() - s.ln();
    (ln_k_scaled, (mu + x + 0.5 - h) / x)
}

/// (ln K_ν(x), or ln K_ν(x) + x if scaled, and K_{ν+1}(x) / K_ν(x)) for ν >= 0 and 0 < x < ∞.
fn k_core(nu: f64, x: f64, scaled: bool) -> (f64, f64) {
    if nu >= DEBYE_MIN_ORDER {
        let d = debye(nu, x, scaled);
        // K_{ν+1} / K_ν = ν/x − K'_ν/K_ν (DLMF 10.29.2)
        return (d.ln_k, nu / x + d.k_log_derivative_negated);
    }
    let n = (nu + 0.5).floor();
    let mu = nu - n;
    let (mut ln_k, mut ratio) = if x <= 2.0 {
        let (ln_k, ratio) = temme(mu, x);
        (if scaled { ln_k + x } else { ln_k }, ratio)
    } else {
        let (ln_k, ratio) = steed(mu, x);
        (if scaled { ln_k } else { ln_k - x }, ratio)
    };
    // K_{j+1} = K_{j−1} + 2j/x K_j, as ratios: K_{j+1}/K_j = K_{j−1}/K_j + 2j/x
    for i in 1..=n as usize {
        ln_k += ratio.ln();
        ratio = 1.0 / ratio + 2.0 * (mu + i as f64) / x;
    }
    (ln_k, ratio)
}

/// The Hankel sum Σ_k (−1)^k a_k(ν) / x^k of I_ν(x) ~ eˣ / √(2πx) Σ … (DLMF 10.40.1), for x >= max(30, ν²).
fn hankel_sum(nu: f64, x: f64) -> f64 {
    let mu = 4.0 * nu * nu;
    let mut term = 1.0f64;
    let mut sum = 1.0;
    for k in 1..200 {
        let odd = (2 * k - 1) as f64;
        term *= -(mu - odd * odd) / (8.0 * k as f64 * x);
        sum += term;
        if term.abs() <= 1e-17 * sum.abs() {
            break;
        }
    }
    sum
}

/// I_{ν+1}(x) / I_ν(x) for ν >= 0 and 0 < x < ∞.
fn i_ratio_core(nu: f64, x: f64) -> f64 {
    if nu >= DEBYE_MIN_ORDER && x > nu {
        // I_{ν+1} / I_ν = I'_ν/I_ν − ν/x (DLMF 10.29.2); no cancellation for x > ν
        return debye(nu, x, false).i_log_derivative - nu / x;
    }
    if nu < DEBYE_MIN_ORDER && x >= HANKEL_MIN_X.max(nu * nu) {
        return hankel_sum(nu + 1.0, x) / hankel_sum(nu, x);
    }
    // 1 / (I_{ν+1}/I_ν) = 2(ν+1)/x + I_{ν+2}/I_{ν+1}, a continued fraction of positive terms
    let b = |n: usize| Jet::<0>::constant(2.0 * (nu + n as f64) / x);
    match continued_fraction(b(1), |n| (Jet::constant(1.0), b(n + 1))) {
        Some(f) => 1.0 / f.v,
        None => f64::NAN,
    }
}

/// ln I_ν(x), or ln I_ν(x) − x if scaled, for ν >= 0 and 0 < x < ∞.
fn log_i_core(nu: f64, x: f64, scaled: bool) -> f64 {
    if nu >= DEBYE_MIN_ORDER {
        return debye(nu, x, scaled).ln_i;
    }
    let shift = if scaled { x } else { 0.0 };
    if x <= SERIES_MAX_X {
        // ln of the first term, (x/2)^ν / Γ(ν + 1), plus ln(1 + the rest relative to it)
        let q = 0.25 * x * x;
        let mut term = 1.0;
        let mut rest = 0.0;
        for j in 1..1000 {
            let j = j as f64;
            term *= q / (j * (j + nu));
            rest += term;
            if term <= 1e-17 * (1.0 + rest) {
                break;
            }
        }
        let first = if nu == 0.0 {
            0.0
        } else {
            nu * (0.5 * x).ln() - ln_gamma(nu + 1.0)
        };
        return first + f64::ln_1p(rest) - shift;
    }
    if x >= nu * nu {
        return (x - shift) - 0.5 * (2.0 * PI * x).ln() + hankel_sum(nu, x).ln();
    }
    // the Wronskian I_ν K_{ν+1} + I_{ν+1} K_ν = 1/x (with the scaled K for the scaled I)
    let (ln_k, k_ratio) = k_core(nu, x, scaled);
    -x.ln() - ln_k - (k_ratio + i_ratio_core(nu, x)).ln()
}

/// Below this x, ln K_ν(x) and ln I_ν(x) are given by their leading terms as x → 0, whose next terms are smaller
/// by a factor of about x².
const TINY_X: f64 = 1e-150;

/// The Euler–Mascheroni constant γ.
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// ln K_ν(x) for ν >= 0 and 0 < x < [`TINY_X`], from K₀(x) ~ −ln(x/2) − γ and, for ν > 0,
/// K_ν(x) ~ ½Γ(ν)(x/2)^{−ν} (1 − Γ(1 − ν)/Γ(1 + ν) (x/2)^{2ν}) (the bracket's second term matters only for ν < 1).
/// ln(x/2) is formed as ln x − ln 2, since x/2 underflows for the least subnormal x.
fn log_bessel_k_tiny(nu: f64, x: f64) -> f64 {
    let log_half_x = x.ln() - LN_2;
    if nu == 0.0 {
        return (-log_half_x - EULER_GAMMA).ln();
    }
    let leading = ln_gamma(nu) - LN_2 - nu * log_half_x;
    if nu >= 1.0 {
        return leading;
    }
    let exponent = ln_gamma_reflection_ratio(nu) + 2.0 * nu * log_half_x;
    leading + (-exponent.exp_m1()).ln()
}

/// ln Γ(1 − ν) − ln Γ(1 + ν) for 0 <= ν < 1. For small ν it is about 2γν, and the bracket of
/// [`log_bessel_k_tiny`] cancels down to about 2ν(−ln(x/2) − γ), so the difference of the two ln Γ (each with an
/// absolute error near the rounding of 1) would lose the relative accuracy; there it is summed from the series
/// 2(γν + Σ_{odd k >= 3} ζ(k) νᵏ / k) (from DLMF 5.7.3).
fn ln_gamma_reflection_ratio(nu: f64) -> f64 {
    if nu >= 0.1 {
        return ln_gamma(1.0 - nu) - ln_gamma(1.0 + nu);
    }
    // ζ(3), ζ(5), …, ζ(19); beyond, ζ(k) = 1 + 2⁻ᵏ to well within the terms' size
    const ZETA_ODD: [f64; 9] = [
        1.202_056_903_159_594_3,
        1.036_927_755_143_37,
        1.008_349_277_381_922_8,
        1.002_008_392_826_082_2,
        1.000_494_188_604_119_5,
        1.000_122_713_347_578_5,
        1.000_030_588_236_307,
        1.000_007_637_197_637_9,
        1.000_001_908_212_716_6,
    ];
    let nu2 = nu * nu;
    let mut power = nu;
    let mut sum = EULER_GAMMA * nu;
    for k in (3..40).step_by(2) {
        power *= nu2;
        let zeta = ZETA_ODD
            .get((k - 3) / 2)
            .copied()
            .unwrap_or_else(|| 1.0 + 0.5f64.powi(k as i32));
        let term = zeta * power / k as f64;
        sum += term;
        if term < 1e-18 * sum {
            break;
        }
    }
    2.0 * sum
}

/// ln K_ν(x), the log of the modified Bessel function of the second kind, for real ν (K_{−ν} = K_ν) and x >= 0:
/// ∞ at x = 0, −∞ at x = ∞ and NaN for x < 0 or NaN arguments. It never overflows or underflows.
///
/// Accuracy: an error within 1e-14 × max(1, |ln K_ν|) (so a relative error of about 1e-14 in K_ν where the log is
/// moderate), checked against 50-digit values for ν from 0 to 2000 and x from the least positive float (about
/// 5e-324) to 1e10. It costs a few dozen operations plus, for |ν| < 30, one per unit of |ν|.
pub fn log_bessel_k(nu: f64, x: f64) -> f64 {
    if nu.is_nan() || x.is_nan() || x < 0.0 || nu.is_infinite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x.is_infinite() {
        return f64::NEG_INFINITY;
    }
    if x < TINY_X {
        return log_bessel_k_tiny(nu.abs(), x);
    }
    k_core(nu.abs(), x, false).0
}

/// e^{scaled} e^{shift} without losing the accuracy of e^{shift} (for an exact `shift`) to rounding their sum.
fn exp_shifted(scaled: f64, shift: f64) -> f64 {
    if shift.abs() < 700.0 && scaled.abs() < 700.0 {
        scaled.exp() * shift.exp()
    } else {
        (scaled + shift).exp()
    }
}

/// K_ν(x), the modified Bessel function of the second kind, for real ν (K_{−ν} = K_ν) and x >= 0 (see
/// [`log_bessel_k`]; this is its exponential, so it overflows to ∞ for small x and large ν and underflows to 0
/// for x beyond about 700). Its relative error is within 3e-14, or 6e-16 times the larger of |ln K_ν| and (for
/// |ν| >= 30) √(ν² + x²), the rounding of the exponent, if that is larger.
pub fn bessel_k(nu: f64, x: f64) -> f64 {
    let scaled = bessel_k_scaled_log(nu, x);
    if scaled.is_nan() || x.is_infinite() {
        return log_bessel_k(nu, x).exp();
    }
    exp_shifted(scaled, -x)
}

/// ln(eˣ K_ν(x)), or NaN outside the domain or at its ends.
fn bessel_k_scaled_log(nu: f64, x: f64) -> f64 {
    if nu.is_nan() || x.is_nan() || x <= 0.0 || nu.is_infinite() || x.is_infinite() {
        return f64::NAN;
    }
    k_core(nu.abs(), x, true).0
}

/// eˣ K_ν(x), the exponentially scaled modified Bessel function of the second kind (see [`log_bessel_k`]), which
/// stays representable for large x.
pub fn bessel_k_scaled(nu: f64, x: f64) -> f64 {
    if x.is_infinite() && x > 0.0 {
        return if nu.is_finite() { 0.0 } else { f64::NAN };
    }
    if x == 0.0 && nu.is_finite() {
        return f64::INFINITY;
    }
    bessel_k_scaled_log(nu, x).exp()
}

/// K_{ν+1}(x) / K_ν(x) for real ν >= 0 and x >= 0 (∞ at 0, 1 at ∞): the ratio from which the derivatives of
/// K_ν follow (DLMF 10.29.2). Within 1e-14 relative error (checked as [`log_bessel_k`]).
pub fn bessel_k_ratio(nu: f64, x: f64) -> f64 {
    if nu.is_nan() || x.is_nan() || x < 0.0 || nu < 0.0 || nu.is_infinite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x.is_infinite() {
        return 1.0;
    }
    k_core(nu, x, false).1
}

/// d/dx ln K_ν(x) = K'_ν(x) / K_ν(x) = ν/x − K_{ν+1}(x)/K_ν(x) (DLMF 10.29.2), for real ν and x > 0 (−∞ at
/// x = 0, −1 at ∞), always negative. ν is a constant here: no derivative in ν is provided.
pub fn log_bessel_k_dx(nu: f64, x: f64) -> f64 {
    let nu = nu.abs();
    if x == 0.0 && !nu.is_nan() {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() && x > 0.0 && nu.is_finite() {
        return -1.0;
    }
    nu / x - bessel_k_ratio(nu, x)
}

/// K'_ν(x) = K_ν(x) (ν/x − K_{ν+1}(x)/K_ν(x)), for real ν and x > 0.
pub fn bessel_k_dx(nu: f64, x: f64) -> f64 {
    let k = bessel_k(nu, x);
    if k == 0.0 {
        return -0.0;
    }
    k * log_bessel_k_dx(nu, x)
}

/// ln I_ν(x), the log of the modified Bessel function of the first kind, for ν >= 0 and x >= 0: 0 at x = 0 for
/// ν = 0 and −∞ for ν > 0, ∞ at x = ∞, and NaN for negative or NaN arguments (I_ν of a negative argument is real
/// only for integer ν). It never overflows or underflows.
///
/// Accuracy: an error within 1e-14 × max(1, |ln I_ν|), checked against 50-digit values for ν from 0 to 2000 and x
/// from the least positive float (about 5e-324) to 1e10. For ν < 30 and 30 < x < ν² it costs a continued fraction
/// of about x terms.
pub fn log_bessel_i(nu: f64, x: f64) -> f64 {
    if nu.is_nan() || x.is_nan() || x < 0.0 || nu < 0.0 || nu.is_infinite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return if nu == 0.0 { 0.0 } else { f64::NEG_INFINITY };
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    if x < TINY_X {
        // I_ν(x) ~ (x/2)^ν / Γ(ν + 1), with ln I₀(x) ~ x²/4
        return if nu == 0.0 {
            0.25 * x * x
        } else {
            nu * (x.ln() - LN_2) - ln_gamma(nu + 1.0)
        };
    }
    log_i_core(nu, x, false)
}

/// I_ν(x), the modified Bessel function of the first kind, for ν >= 0 and x >= 0 (see [`log_bessel_i`]; this
/// is its exponential, so it overflows to ∞ for x beyond about 700), with the relative accuracy of
/// [`bessel_k`].
pub fn bessel_i(nu: f64, x: f64) -> f64 {
    if x > 0.0 && x.is_finite() && nu >= 0.0 && nu.is_finite() {
        return exp_shifted(log_i_core(nu, x, true), x);
    }
    log_bessel_i(nu, x).exp()
}

/// e⁻ˣ I_ν(x), the exponentially scaled modified Bessel function of the first kind (see [`log_bessel_i`]).
pub fn bessel_i_scaled(nu: f64, x: f64) -> f64 {
    if x > 0.0 && x.is_finite() && nu >= 0.0 && nu.is_finite() {
        return log_i_core(nu, x, true).exp();
    }
    if x.is_infinite() && x > 0.0 && nu >= 0.0 && nu.is_finite() {
        return 0.0;
    }
    log_bessel_i(nu, x).exp()
}

/// I_{ν+1}(x) / I_ν(x) for ν >= 0 and x >= 0 (0 at 0, 1 at ∞), in (0, 1) between: for example the mean
/// resultant length I_{d/2}(κ) / I_{d/2−1}(κ) of a von Mises–Fisher distribution. Within 1e-14 relative error
/// (checked as [`log_bessel_i`]).
pub fn bessel_i_ratio(nu: f64, x: f64) -> f64 {
    if nu.is_nan() || x.is_nan() || x < 0.0 || nu < 0.0 || nu.is_infinite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x.is_infinite() {
        return 1.0;
    }
    i_ratio_core(nu, x)
}

/// d/dx ln I_ν(x) = I'_ν(x) / I_ν(x) = I_{ν+1}(x)/I_ν(x) + ν/x (DLMF 10.29.2), for ν >= 0 and x >= 0 (0 at x = 0
/// for ν = 0 and ∞ for ν > 0, 1 at ∞). ν is a constant here: no derivative in ν is provided.
pub fn log_bessel_i_dx(nu: f64, x: f64) -> f64 {
    if x == 0.0 && nu >= 0.0 {
        return if nu == 0.0 { 0.0 } else { f64::INFINITY };
    }
    if x.is_infinite() && x > 0.0 && nu >= 0.0 && nu.is_finite() {
        return 1.0;
    }
    bessel_i_ratio(nu, x) + nu / x
}
