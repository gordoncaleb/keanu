//! A linearly implicit Rosenbrock method of order 4 with an embedded order 3 solution, with the coefficients of
//! Rodas4 from Hairer and Wanner, *Solving Ordinary Differential Equations II* (IV.7, Table 7.2's method in the
//! transformed form (7.25)). Each stage solves a linear system with the matrix I / (hγ) − J, with J the exact
//! Jacobian, factored once per step. The method is L-stable and stiffly accurate; its error estimate is the last
//! stage. Steps end exactly at the output times.

use super::{
    FailureReason, Integration, OdeError, OdeOptions, OdeStatistics, Problem, error_norm, failed,
    initial_step,
};

/// γ, the diagonal of the method.
const GAMMA: f64 = 0.25;

/// The stage times αᵢ (stages 5 and 6 are at the end of the step).
const ALPHA: [f64; 6] = [0.0, 0.386, 0.21, 0.63, 1.0, 1.0];

/// The time-derivative weights γᵢ = Σⱼ γᵢⱼ (zero for stages 5 and 6).
const GAMMA_SUM: [f64; 6] = [0.25, -0.1043, 0.1035, -0.362e-1, 0.0, 0.0];

/// The transformed coefficients aᵢⱼ of the stage arguments, rows 2 to 5 (stage 6's argument is stage 5's plus
/// its own vector, and the solution stage 6's argument plus its vector).
const A: [&[f64]; 5] = [
    &[],
    &[1.544],
    &[0.9466785280815826, 0.2557011698983284],
    &[3.314825187068521, 2.896124015972201, 0.9986419139977817],
    &[
        1.221224509226641,
        6.019134481288629,
        12.53708332932087,
        -0.687886036105895,
    ],
];

/// The transformed coefficients cᵢⱼ (scaled by 1 / h) coupling the stage vectors, rows 2 to 6.
const C: [&[f64]; 6] = [
    &[],
    &[-5.6688],
    &[-2.430093356833875, -0.2063599157091915],
    &[-0.1073529058151375, -9.594562251023355, -20.47028614809616],
    &[
        7.496443313967647,
        -10.24680431464352,
        -33.99990352819905,
        11.7089089320616,
    ],
    &[
        8.083246795921522,
        -7.981132988064893,
        -31.52159432874371,
        16.31930543123136,
        -6.058818238834054,
    ],
];

/// Step-size control (Hairer and Wanner's `rodas` defaults): the safety factor, the bounds of the step change and
/// the predictive (Gustafsson) controller.
const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 6.0;

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
    let mut f0 = vec![0.0; n];
    problem.rhs(t, &z, &mut f0)?;
    stats.rhs_evaluations += 1;
    if f0.iter().any(|x| !x.is_finite()) {
        return Err(failed(t, FailureReason::NonFinite));
    }
    let mut h = initial_step(problem, t, &z, &f0, 4, span, options)?;
    stats.rhs_evaluations += 1;
    let mut dfdt = vec![0.0; n];
    let mut linearised = false;
    let mut k: Vec<Vec<f64>> = vec![vec![0.0; n]; 6];
    let mut z_next = vec![0.0; n];
    let mut rejected_last = false;
    // the previous accepted step and its error, for the predictive controller
    let mut previous: Option<(f64, f64)> = None;

    loop {
        if stats.accepted_steps + stats.rejected_steps >= options.max_steps {
            return Err(failed(t, FailureReason::MaxSteps(options.max_steps)));
        }
        if h.is_nan() || 0.1 * h.abs() <= t.abs() * f64::EPSILON {
            return Err(failed(t, FailureReason::StepSizeUnderflow(h)));
        }
        let target = times[next];
        let hits = t + 1.01 * h >= target;
        if hits {
            h = target - t;
        }
        if !linearised {
            problem.linearise(t, &z, &mut dfdt)?;
            stats.jacobian_evaluations += 1;
            linearised = true;
        }
        if !problem.factor(1.0 / (h * GAMMA)) {
            // a singular stage matrix: retry with a smaller step
            stats.rejected_steps += 1;
            rejected_last = true;
            h *= 0.5;
            continue;
        }

        let norm = match attempt(problem, t, &z, &f0, &dfdt, h, &mut k, &mut stats)? {
            Some(z_new) => {
                let norm = error_norm(&k[5], &z, &z_new, options);
                z_next = z_new;
                norm
            }
            None => f64::NAN,
        };
        if !norm.is_finite() {
            stats.rejected_steps += 1;
            rejected_last = true;
            h *= MIN_FACTOR;
            continue;
        }

        let factor = (norm.powf(0.25) / SAFETY).clamp(1.0 / MAX_FACTOR, 1.0 / MIN_FACTOR);
        let mut h_new = h / factor;
        if norm <= 1.0 {
            stats.accepted_steps += 1;
            if let Some((h_previous, error_previous)) = previous {
                let predictive = (h_previous / h * (norm * norm / error_previous).powf(0.25)
                    / SAFETY)
                    .clamp(1.0 / MAX_FACTOR, 1.0 / MIN_FACTOR);
                h_new = h / factor.max(predictive);
            }
            previous = Some((h, norm.max(1e-2)));
            t = if hits { target } else { t + h };
            z.copy_from_slice(&z_next);
            if hits {
                out.push(z.clone());
                next += 1;
                if next == times.len() {
                    return Ok((out, stats));
                }
            }
            problem.rhs(t, &z, &mut f0)?;
            stats.rhs_evaluations += 1;
            linearised = false;
            if rejected_last {
                h_new = h_new.min(h);
            }
            rejected_last = false;
            h = h_new.min(span);
        } else {
            stats.rejected_steps += 1;
            rejected_last = true;
            h = h_new;
        }
    }
}

/// One step of size h from (t, z), with f0 = f(t, z) and the problem linearised and factored there: the new solution,
/// with the error estimate left in k[5], or None if a stage is not finite.
#[allow(clippy::too_many_arguments)]
fn attempt<P: Problem>(
    problem: &mut P,
    t: f64,
    z: &[f64],
    f0: &[f64],
    dfdt: &[f64],
    h: f64,
    k: &mut [Vec<f64>],
    stats: &mut OdeStatistics,
) -> Result<Option<Vec<f64>>, OdeError<P::Error>> {
    let n = z.len();
    let mut argument = z.to_vec();
    let mut evaluation = f0.to_vec();
    for i in 0..6 {
        if i > 0 {
            // the stage's argument: z + Σ aᵢⱼ kⱼ (stage 6: stage 5's argument plus k₅)
            if i < 5 {
                for (j, a) in argument.iter_mut().enumerate() {
                    *a = z[j] + A[i].iter().zip(&*k).map(|(c, kk)| c * kk[j]).sum::<f64>();
                }
            } else {
                argument.iter_mut().zip(&k[4]).for_each(|(a, kk)| *a += kk);
            }
            problem.rhs(t + ALPHA[i] * h, &argument, &mut evaluation)?;
            stats.rhs_evaluations += 1;
        }
        let (done, rest) = k.split_at_mut(i);
        let stage = &mut rest[0];
        for j in 0..n {
            stage[j] = evaluation[j]
                + C[i]
                    .iter()
                    .zip(done.iter())
                    .map(|(c, kk)| c * kk[j])
                    .sum::<f64>()
                    / h
                + h * GAMMA_SUM[i] * dfdt[j];
        }
        problem.solve(stage)?;
        if stage.iter().any(|x| !x.is_finite()) {
            return Ok(None);
        }
    }
    // the solution is stage 6's argument plus k₆, and k₆ is the error estimate
    Ok(Some(
        argument.iter().zip(&k[5]).map(|(a, e)| a + e).collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ode::{FnSystem, Plain};

    /// A non-linear, non-autonomous system with a full Jacobian, so that every order condition matters.
    #[allow(clippy::type_complexity)]
    fn system() -> FnSystem<
        impl FnMut(f64, &[f64], &mut [f64]),
        impl FnMut(f64, &[f64], &mut [f64], &mut [f64]),
    > {
        FnSystem::new(
            |t, y, dy| {
                dy[0] = -y[1] * y[0] + t.cos();
                dy[1] = 0.5 * y[0] * y[0] - y[1] + (2.0 * t).sin();
            },
            |t, y, j, dt| {
                j.copy_from_slice(&[-y[1], -y[0], y[0], -1.0]);
                dt.copy_from_slice(&[-t.sin(), 2.0 * (2.0 * t).cos()]);
            },
        )
    }

    /// The solution and embedded solution after one step of size h from y(0) = (1, 0.5).
    fn one_step(h: f64) -> (Vec<f64>, Vec<f64>) {
        let mut system = system();
        let mut problem = Plain {
            system: &mut system,
            n: 2,
            jacobian: vec![0.0; 4],
            lu: None,
        };
        let z = [1.0, 0.5];
        let (mut f0, mut dfdt) = (vec![0.0; 2], vec![0.0; 2]);
        problem.rhs(0.0, &z, &mut f0).unwrap();
        problem.linearise(0.0, &z, &mut dfdt).unwrap();
        assert!(problem.factor(1.0 / (h * GAMMA)));
        let mut k = vec![vec![0.0; 2]; 6];
        let mut stats = OdeStatistics::default();
        let z1 = attempt(&mut problem, 0.0, &z, &f0, &dfdt, h, &mut k, &mut stats)
            .unwrap()
            .unwrap();
        let embedded = z1.iter().zip(&k[5]).map(|(z, e)| z - e).collect();
        (z1, embedded)
    }

    /// The local errors of the solution (global order 4) and of the embedded solution (order 3) fall by 2⁵ and 2⁴
    /// when the step is halved, which checks the coefficients against the order conditions.
    #[test]
    fn the_local_errors_have_orders_five_and_four() {
        let local = |h: f64| {
            let (z, e) = one_step(h);
            let options =
                OdeOptions::new(crate::ode::OdeSolver::Rk45).with_tolerances(1e-14, 1e-14);
            let exact = crate::ode::solve(&mut system(), 0.0, &[1.0, 0.5], &[h], &options).unwrap();
            let exact = &exact.states[0];
            let distance = |x: &[f64]| (x[0] - exact[0]).abs().max((x[1] - exact[1]).abs());
            (distance(&z), distance(&e))
        };
        let errors: Vec<(f64, f64)> = [0.1, 0.05, 0.025].into_iter().map(local).collect();
        for pair in errors.windows(2) {
            let (solution, embedded) = (pair[0].0 / pair[1].0, pair[0].1 / pair[1].1);
            assert!(
                (28.0..40.0).contains(&solution),
                "solution ratio {solution}"
            );
            assert!(
                (14.0..18.0).contains(&embedded),
                "embedded ratio {embedded}"
            );
        }
    }
}
