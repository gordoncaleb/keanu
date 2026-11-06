//! Values and x-derivatives of the [`BesselOp`] functions (see [`Op::Bessel`](super::Op::Bessel)).

use super::BesselOp;
use keanu_tensor::special;
use std::f64::consts::LN_2;

/// None if ν is a valid order for the function, else why not.
pub(crate) fn invalid_order(kind: BesselOp, nu: f64) -> Option<String> {
    let valid = nu.is_finite()
        && match kind {
            BesselOp::K | BesselOp::LogK => true,
            BesselOp::LogI => nu >= 0.0,
            BesselOp::Matern => nu > 0.0,
        };
    (!valid).then(|| {
        let range = match kind {
            BesselOp::K | BesselOp::LogK => "finite",
            BesselOp::LogI => "finite and non-negative",
            BesselOp::Matern => "finite and positive",
        };
        format!("the order of {kind:?} must be {range}, not {nu}")
    })
}

/// The function at x.
pub(crate) fn value(kind: BesselOp, nu: f64, x: f64) -> f64 {
    match kind {
        BesselOp::K => special::bessel_k(nu, x),
        BesselOp::LogK => special::log_bessel_k(nu, x),
        BesselOp::LogI => special::log_bessel_i(nu, x),
        BesselOp::Matern => matern(nu, x),
    }
}

/// The first and second derivatives in x at x, with y the value there.
pub(crate) fn derivatives(kind: BesselOp, nu: f64, x: f64, y: f64) -> (f64, f64) {
    // from the modified Bessel equation x² w'' + x w' − (x² + ν²) w = 0, satisfied by I_ν and K_ν
    let ratio = 1.0 + nu * nu / (x * x);
    match kind {
        BesselOp::K => {
            if y == 0.0 {
                return (-0.0, 0.0);
            }
            let d1 = y * special::log_bessel_k_dx(nu, x);
            (d1, ratio * y - d1 / x)
        }
        // with L = ln w: L'' = w''/w − L'² = 1 + ν²/x² − L'/x − L'²
        BesselOp::LogK | BesselOp::LogI => {
            let d1 = if kind == BesselOp::LogK {
                special::log_bessel_k_dx(nu, x)
            } else {
                special::log_bessel_i_dx(nu, x)
            };
            (d1, ratio - d1 / x - d1 * d1)
        }
        BesselOp::Matern => matern_derivatives(nu, x, y),
    }
}

/// f(x) = 2^{1−ν}/Γ(ν) x^ν K_ν(x), with f(0) = 1.
fn matern(nu: f64, x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x.is_infinite() {
        return 0.0;
    }
    ((1.0 - nu) * LN_2 - special::ln_gamma(nu) + nu * x.ln() + special::log_bessel_k(nu, x)).exp()
}

/// (f', f'') for the Matérn correlation f: f' = −2^{1−ν}/Γ(ν) x^ν K_{ν−1}(x) = −f K_{ν−1}/K_ν (DLMF 10.29.4) and
/// f'' = f + (2ν − 1) f'/x, with their limits at x = 0.
fn matern_derivatives(nu: f64, x: f64, f: f64) -> (f64, f64) {
    if x == 0.0 {
        // f = 1 − c x^{2ν} + … for ν < 1 (and x² ln x at ν = 1), 1 − x²/(4(ν − 1)) + … for ν > 1
        let d1 = if nu > 0.5 {
            0.0
        } else if nu == 0.5 {
            -1.0
        } else {
            f64::NEG_INFINITY
        };
        let d2 = if nu > 1.0 {
            -0.5 / (nu - 1.0)
        } else if nu > 0.5 {
            f64::NEG_INFINITY
        } else if nu == 0.5 {
            1.0
        } else {
            f64::INFINITY
        };
        return (d1, d2);
    }
    if f == 0.0 || x.is_infinite() {
        return (-0.0, 0.0);
    }
    // K_{ν−1}/K_ν, from the ratio K_ν/K_{ν−1} for ν >= 1 and through K_{ν−1} = K_{1−ν} otherwise
    let below = if nu >= 1.0 {
        1.0 / special::bessel_k_ratio(nu - 1.0, x)
    } else {
        (special::log_bessel_k(1.0 - nu, x) - special::log_bessel_k(nu, x)).exp()
    };
    let d1 = -f * below;
    (d1, f + (2.0 * nu - 1.0) * d1 / x)
}
