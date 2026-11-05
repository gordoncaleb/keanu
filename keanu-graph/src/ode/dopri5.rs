//! The explicit Dormand–Prince 5(4) Runge–Kutta method (Dormand and Prince, 1980) with its fourth-order dense
//! output, after Hairer, Nørsett and Wanner, *Solving Ordinary Differential Equations I* (II.4–II.6): a seven stage,
//! first-same-as-last pair, the fifth-order solution propagated, the fourth-order one for the error estimate, and a
//! PI step-size controller.

use super::{
    FailureReason, Integration, OdeOptions, OdeStatistics, Problem, error_norm, failed,
    initial_step,
};

const C: [f64; 7] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];

/// The lower triangle of the Butcher matrix, row by row (the last row is the fifth-order weights).
const A: [&[f64]; 7] = [
    &[],
    &[1.0 / 5.0],
    &[3.0 / 40.0, 9.0 / 40.0],
    &[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
    &[
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
    ],
    &[
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
    ],
    &[
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    ],
];

/// The fifth- minus the fourth-order weights, for the error estimate.
const E: [f64; 7] = [
    71.0 / 57600.0,
    0.0,
    -71.0 / 16695.0,
    71.0 / 1920.0,
    -17253.0 / 339200.0,
    22.0 / 525.0,
    -1.0 / 40.0,
];

/// The coefficients of the dense output's last term (Hairer, Nørsett and Wanner, II.6, after Shampine, 1986).
const D: [f64; 7] = [
    -12715105075.0 / 11282082432.0,
    0.0,
    87487479700.0 / 32700410799.0,
    -10690763975.0 / 1880347072.0,
    701980252875.0 / 199316789632.0,
    -1453857185.0 / 822651844.0,
    69997945.0 / 29380423.0,
];

/// PI control: the exponents of the current and previous error, the safety factor and the bounds of the step
/// change (Hairer's `dopri5` defaults).
const BETA: f64 = 0.04;
const EXPONENT: f64 = 0.2 - BETA * 0.75;
const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;

pub(super) fn integrate<P: Problem>(
    problem: &mut P,
    t0: f64,
    z0: &[f64],
    times: &[f64],
    options: &OdeOptions,
) -> Integration<P::Error> {
    let n = z0.len();
    let t_end = *times.last().expect("at least one output time");
    let span = t_end - t0;
    let mut stats = OdeStatistics::default();
    let mut out = Vec::with_capacity(times.len());
    let mut next = 0;

    let mut t = t0;
    let mut z = z0.to_vec();
    let mut k: Vec<Vec<f64>> = vec![vec![0.0; n]; 7];
    problem.rhs(t, &z, &mut k[0])?;
    stats.rhs_evaluations += 1;
    if k[0].iter().any(|x| !x.is_finite()) {
        return Err(failed(t, FailureReason::NonFinite));
    }
    let mut h = initial_step(problem, t, &z, &k[0], 5, span, options)?;
    stats.rhs_evaluations += 1;
    let mut previous_error: f64 = 1e-4;
    let mut rejected_last = false;
    let mut stage = vec![0.0; n];
    let mut z_new = vec![0.0; n];
    let mut error = vec![0.0; n];

    loop {
        if stats.accepted_steps + stats.rejected_steps >= options.max_steps {
            return Err(failed(t, FailureReason::MaxSteps(options.max_steps)));
        }
        if h.is_nan() || 0.1 * h.abs() <= t.abs() * f64::EPSILON {
            return Err(failed(t, FailureReason::StepSizeUnderflow(h)));
        }
        let last = t + 1.01 * h >= t_end;
        if last {
            h = t_end - t;
        }

        // stages 2 to 7; the seventh is f at the new solution
        let mut finite = true;
        for i in 1..7 {
            let target: &mut [f64] = if i == 6 { &mut z_new } else { &mut stage };
            for (j, value) in target.iter_mut().enumerate() {
                *value = z[j] + h * A[i].iter().zip(&k).map(|(a, kk)| a * kk[j]).sum::<f64>();
            }
            let input = if i == 6 { &z_new } else { &stage };
            problem.rhs(t + C[i] * h, input, &mut k[i])?;
            stats.rhs_evaluations += 1;
            if k[i].iter().any(|x| !x.is_finite()) {
                finite = false;
                break;
            }
        }
        let norm = if finite {
            for (j, e) in error.iter_mut().enumerate() {
                *e = h * E.iter().zip(&k).map(|(c, kk)| c * kk[j]).sum::<f64>();
            }
            error_norm(&error, &z, &z_new, options)
        } else {
            f64::NAN
        };
        if !norm.is_finite() {
            // a non-finite stage: retry with a much smaller step
            stats.rejected_steps += 1;
            rejected_last = true;
            h *= MIN_FACTOR;
            continue;
        }

        let factor_current = norm.powf(EXPONENT);
        if norm <= 1.0 {
            stats.accepted_steps += 1;
            let factor = (factor_current / previous_error.powf(BETA) / SAFETY)
                .clamp(1.0 / MAX_FACTOR, 1.0 / MIN_FACTOR);
            let mut h_new = h / factor;
            previous_error = norm.max(1e-4);
            let t_new = if last { t_end } else { t + h };
            // dense output at the output times this step passes
            while next < times.len() && (times[next] <= t_new || last) {
                if times[next] == t_new || last && next + 1 == times.len() {
                    out.push(z_new.clone());
                } else {
                    let theta = (times[next] - t) / h;
                    out.push(dense(&z, &z_new, &k, h, theta));
                }
                next += 1;
            }
            if next == times.len() {
                return Ok((out, stats));
            }
            t = t_new;
            std::mem::swap(&mut z, &mut z_new);
            k.swap(0, 6);
            if rejected_last {
                h_new = h_new.min(h);
            }
            rejected_last = false;
            h = h_new.min(span);
        } else {
            stats.rejected_steps += 1;
            rejected_last = true;
            h /= (factor_current / SAFETY).min(1.0 / MIN_FACTOR);
        }
    }
}

/// The dense output at t + θh within an accepted step from z0 to z1 with stages `k`.
fn dense(z0: &[f64], z1: &[f64], k: &[Vec<f64>], h: f64, theta: f64) -> Vec<f64> {
    let theta1 = 1.0 - theta;
    (0..z0.len())
        .map(|j| {
            let difference = z1[j] - z0[j];
            let b = h * k[0][j] - difference;
            let c = difference - h * k[6][j] - b;
            let d = h * D.iter().zip(k).map(|(d, kk)| d * kk[j]).sum::<f64>();
            z0[j] + theta * (difference + theta1 * (b + theta * (c + theta1 * d)))
        })
        .collect()
}
