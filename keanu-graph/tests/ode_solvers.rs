//! The ODE integrators against closed forms, conserved quantities, published reference values and finite
//! differences.

use std::convert::Infallible;

use keanu_graph::ode::{
    FailureReason, FnSystem, OdeError, OdeOptions, OdeSolver, OdeSystem, SensitivitySystem,
    SolverFailed, integrate, solve, solve_sensitivities,
};

const SOLVERS: [OdeSolver; 2] = [OdeSolver::Rk45, OdeSolver::Stiff];

fn tight(solver: OdeSolver) -> OdeOptions {
    OdeOptions::new(solver).with_tolerances(1e-8, 1e-10)
}

/// Output times that don't line up with any step.
fn times(end: f64, count: usize) -> Vec<f64> {
    (1..=count)
        .map(|i| end * (i as f64 / count as f64).powf(1.1))
        .collect()
}

#[test]
fn exponential_decay_matches_its_closed_form() {
    for solver in SOLVERS {
        let rate = 1.3;
        let ts = times(5.0, 40);
        let states = integrate(
            |_, y, dy| dy[0] = -rate * y[0],
            |_, _, j, dt| {
                j[0] = -rate;
                dt[0] = 0.0;
            },
            0.0,
            &[2.0],
            &ts,
            &tight(solver),
        )
        .unwrap();
        let worst = ts
            .iter()
            .zip(&states)
            .map(|(t, y)| ((y[0] - 2.0 * (-rate * t).exp()) / (2.0 * (-rate * t).exp())).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-7, "{solver:?}: relative error {worst}");
    }
}

#[test]
fn a_harmonic_oscillator_matches_its_closed_form_over_ten_periods() {
    for solver in SOLVERS {
        let omega = 3.0;
        let period = 2.0 * std::f64::consts::PI / omega;
        let ts = times(10.0 * period, 97);
        let states = integrate(
            |_, y, dy| {
                dy[0] = y[1];
                dy[1] = -omega * omega * y[0];
            },
            |_, _, j, dt| {
                j.copy_from_slice(&[0.0, 1.0, -omega * omega, 0.0]);
                dt.fill(0.0);
            },
            0.0,
            &[1.0, 0.0],
            &ts,
            &tight(solver),
        )
        .unwrap();
        let worst = ts
            .iter()
            .zip(&states)
            .map(|(t, y)| {
                let exact = [(omega * t).cos(), -omega * (omega * t).sin()];
                (y[0] - exact[0]).abs().max((y[1] - exact[1]).abs() / omega)
            })
            .fold(0.0, f64::max);
        assert!(worst < 5e-7, "{solver:?}: error {worst}");
    }
}

/// Lotka–Volterra predator–prey equations with parameters (α, β, γ, δ).
struct LotkaVolterra([f64; 4]);

impl OdeSystem for LotkaVolterra {
    type Error = Infallible;

    fn rhs(&mut self, _: f64, y: &[f64], dy: &mut [f64]) -> Result<(), Infallible> {
        let [a, b, c, d] = self.0;
        dy[0] = a * y[0] - b * y[0] * y[1];
        dy[1] = d * y[0] * y[1] - c * y[1];
        Ok(())
    }

    fn jacobian(
        &mut self,
        _: f64,
        y: &[f64],
        j: &mut [f64],
        dt: &mut [f64],
    ) -> Result<(), Infallible> {
        let [a, b, c, d] = self.0;
        j.copy_from_slice(&[a - b * y[1], -b * y[0], d * y[1], d * y[0] - c]);
        dt.fill(0.0);
        Ok(())
    }
}

impl SensitivitySystem for LotkaVolterra {
    /// Sensitivities with respect to (y0₁, y0₂, α, β, γ, δ).
    fn sensitivity_rhs(
        &mut self,
        t: f64,
        y: &[f64],
        s: &[f64],
        ds: &mut [f64],
    ) -> Result<(), Infallible> {
        let (mut j, mut dt) = ([0.0; 4], [0.0; 2]);
        self.jacobian(t, y, &mut j, &mut dt)?;
        let p = s.len() / 2;
        for i in 0..2 {
            for k in 0..p {
                ds[i * p + k] = j[i * 2] * s[k] + j[i * 2 + 1] * s[p + k];
            }
        }
        // ∂f/∂(α, β, γ, δ)
        let forcing = [
            [y[0], -y[0] * y[1], 0.0, 0.0],
            [0.0, 0.0, -y[1], y[0] * y[1]],
        ];
        for i in 0..2 {
            for (k, f) in forcing[i].iter().enumerate() {
                ds[i * p + 2 + k] += f;
            }
        }
        Ok(())
    }
}

const LOTKA_VOLTERRA: [f64; 4] = [1.5, 1.0, 3.0, 1.0];

#[test]
fn lotka_volterra_conserves_its_invariant_and_matches_a_high_precision_reference() {
    let ts = times(15.0, 60);
    let y0 = [1.0, 1.0];
    let reference = solve(
        &mut LotkaVolterra(LOTKA_VOLTERRA),
        0.0,
        &y0,
        &ts,
        &OdeOptions::new(OdeSolver::Rk45).with_tolerances(1e-13, 1e-13),
    )
    .unwrap()
    .states;
    let [a, b, c, d] = LOTKA_VOLTERRA;
    let invariant = |y: &[f64]| d * y[0] - c * y[0].ln() + b * y[1] - a * y[1].ln();
    for solver in SOLVERS {
        let states = solve(
            &mut LotkaVolterra(LOTKA_VOLTERRA),
            0.0,
            &y0,
            &ts,
            &tight(solver),
        )
        .unwrap()
        .states;
        let (mut drift, mut error) = (0.0f64, 0.0f64);
        for (y, r) in states.iter().zip(&reference) {
            drift = drift.max((invariant(y) - invariant(&y0)).abs());
            error = error.max((y[0] - r[0]).abs().max((y[1] - r[1]).abs()));
        }
        assert!(drift < 1e-6, "{solver:?}: invariant drift {drift}");
        assert!(error < 1e-6, "{solver:?}: error {error}");
    }
}

/// Robertson's problem (Hairer and Wanner, IV.10) at t = 40 and 10¹¹, against the reference values of Hindmarsh's
/// LSODE and SUNDIALS documentation (t = 40) and of Mazzia and Magherini's Test Set for IVP Solvers (t = 10¹¹).
#[test]
fn robertsons_stiff_problem_matches_the_reference_values() {
    let options = OdeOptions::new(OdeSolver::Stiff).with_tolerances(1e-8, 1e-14);
    let solution = solve(
        &mut Robertson(0.04),
        0.0,
        &[1.0, 0.0, 0.0],
        &[40.0, 1e11],
        &options,
    )
    .unwrap();
    let references = [
        [0.7158270687, 9.185534764e-6, 0.2841637457],
        [
            2.083340149701255e-8,
            8.333360770334713e-14,
            0.999999979166505,
        ],
    ];
    for (state, reference) in solution.states.iter().zip(&references) {
        for (y, r) in state.iter().zip(reference) {
            assert!((y - r).abs() <= 1e-7 * r.abs(), "{y} != {r}");
        }
    }
    // the stiff solver needs only about a thousand steps over thirteen decades of time
    assert!(
        solution.statistics.accepted_steps < 3000,
        "{:?}",
        solution.statistics
    );
}

/// The Prothero–Robinson problem y' = λ (y − g(t)) + g'(t), y(0) = g(0), whose solution is g, with a very stiff λ:
/// it checks the stiff solver's use of ∂f/∂t.
#[test]
fn a_stiff_non_autonomous_problem_follows_its_solution() {
    let lambda = -1e6;
    let ts = times(10.0, 50);
    let states = integrate(
        |t, y, dy| dy[0] = lambda * (y[0] - t.sin()) + t.cos(),
        |t, _, j, dt| {
            j[0] = lambda;
            dt[0] = -lambda * t.cos() - t.sin();
        },
        0.0,
        &[0.0],
        &ts,
        &tight(OdeSolver::Stiff),
    )
    .unwrap();
    for (t, y) in ts.iter().zip(&states) {
        assert!((y[0] - t.sin()).abs() < 1e-8, "{} != {}", y[0], t.sin());
    }
}

#[test]
fn the_non_stiff_solver_runs_out_of_steps_on_a_stiff_problem() {
    let options = OdeOptions::new(OdeSolver::Rk45).with_max_steps(10_000);
    let result = solve(
        &mut Robertson(0.04),
        0.0,
        &[1.0, 0.0, 0.0],
        &[1e5],
        &options,
    );
    assert!(
        matches!(
            result,
            Err(OdeError::SolverFailed(SolverFailed {
                reason: FailureReason::MaxSteps(10_000),
                ..
            }))
        ),
        "{result:?}"
    );
}

fn invalid(result: Result<Vec<Vec<f64>>, OdeError>) -> String {
    match result {
        Err(OdeError::InvalidProblem(message)) => message,
        other => panic!("expected an invalid problem, got {other:?}"),
    }
}

fn decay(times: &[f64], options: &OdeOptions) -> Result<Vec<Vec<f64>>, OdeError> {
    integrate(
        |_, y, dy| dy[0] = -y[0],
        |_, _, j, dt| {
            j[0] = -1.0;
            dt[0] = 0.0;
        },
        1.0,
        &[1.0],
        times,
        options,
    )
}

#[test]
fn output_times_must_be_finite_strictly_increasing_and_after_the_initial_time() {
    for solver in SOLVERS {
        let options = OdeOptions::new(solver);
        let message = invalid(decay(&[1.0, 2.0], &options));
        assert!(message.contains("time 0 is 1"), "{message}");
        assert!(invalid(decay(&[0.5], &options)).contains("time 0"));
        assert!(invalid(decay(&[2.0, 3.0, 3.0], &options)).contains("time 2"));
        assert!(invalid(decay(&[2.0, 1.5], &options)).contains("time 1"));
        assert!(invalid(decay(&[2.0, f64::NAN], &options)).contains("time 1"));
        assert!(invalid(decay(&[f64::INFINITY], &options)).contains("time 0"));
        // no output times: nothing to do
        assert_eq!(decay(&[], &options).unwrap(), Vec::<Vec<f64>>::new());
    }
    let bad_initial_time = integrate(
        |_, _, _| {},
        |_, _, _, _| {},
        f64::NAN,
        &[1.0],
        &[1.0],
        &OdeOptions::default(),
    );
    assert!(invalid(bad_initial_time).contains("initial time"));
}

#[test]
fn options_are_checked() {
    for options in [
        OdeOptions::default().with_tolerances(0.0, 0.0),
        OdeOptions::default().with_tolerances(-1e-6, 1e-6),
        OdeOptions::default().with_tolerances(1e-6, f64::NAN),
        OdeOptions::default().with_max_steps(0),
    ] {
        assert!(options.validate().is_err());
        invalid(decay(&[2.0], &options));
    }
    let defaults = OdeOptions::default();
    assert_eq!(defaults.solver, OdeSolver::Rk45);
    assert_eq!(
        (defaults.rtol, defaults.atol, defaults.max_steps),
        (1e-6, 1e-6, 1_000_000)
    );
}

fn failure<E: std::fmt::Debug>(result: Result<impl std::fmt::Debug, OdeError<E>>) -> SolverFailed {
    match result {
        Err(OdeError::SolverFailed(failed)) => failed,
        other => panic!("expected a solver failure, got {other:?}"),
    }
}

#[test]
fn non_finite_states_and_blow_ups_fail_without_panicking() {
    for solver in SOLVERS {
        let options = OdeOptions::new(solver);
        let nan = integrate(
            |_, y, dy| dy[0] = -y[0],
            |_, _, j, dt| {
                j[0] = -1.0;
                dt[0] = 0.0;
            },
            0.0,
            &[f64::NAN],
            &[1.0],
            &options,
        );
        assert_eq!(
            failure(nan),
            SolverFailed {
                t: 0.0,
                reason: FailureReason::NonFinite
            }
        );
        // f is not finite at the initial state
        let singular = integrate(
            |_, y, dy| dy[0] = 1.0 / y[0],
            |_, y, j, dt| {
                j[0] = -1.0 / (y[0] * y[0]);
                dt[0] = 0.0;
            },
            0.0,
            &[0.0],
            &[1.0],
            &options,
        );
        assert_eq!(failure(singular).reason, FailureReason::NonFinite);

        // y' = y², y(0) = 1 is 1 / (1 - t): accurate before the blow-up at t = 1, a failure after it
        let blow_up = |times: &[f64]| {
            integrate(
                |_, y, dy| dy[0] = y[0] * y[0],
                |_, y, j, dt| {
                    j[0] = 2.0 * y[0];
                    dt[0] = 0.0;
                },
                0.0,
                &[1.0],
                times,
                &options.with_tolerances(1e-10, 1e-10),
            )
        };
        let before = blow_up(&[0.5, 0.9]).unwrap();
        assert!((before[0][0] - 2.0).abs() < 1e-8 && (before[1][0] - 10.0).abs() < 1e-6);
        let failed = failure(blow_up(&[2.0]));
        assert!((failed.t - 1.0).abs() < 1e-2, "{failed:?}");
        assert!(
            matches!(
                failed.reason,
                FailureReason::StepSizeUnderflow(_) | FailureReason::MaxSteps(_)
            ),
            "{failed:?}"
        );
    }
}

/// A system whose evaluation fails after t = 0.5.
struct Failing;

impl OdeSystem for Failing {
    type Error = String;

    fn rhs(&mut self, t: f64, y: &[f64], dy: &mut [f64]) -> Result<(), String> {
        if t > 0.5 {
            return Err(format!("no rate at t = {t}"));
        }
        dy[0] = -y[0];
        Ok(())
    }

    fn jacobian(&mut self, _: f64, _: &[f64], j: &mut [f64], dt: &mut [f64]) -> Result<(), String> {
        j[0] = -1.0;
        dt[0] = 0.0;
        Ok(())
    }
}

#[test]
fn system_errors_are_passed_on() {
    for solver in SOLVERS {
        let result = solve(
            &mut Failing,
            0.0,
            &[1.0],
            &[0.25, 1.0],
            &OdeOptions::new(solver),
        );
        match result {
            Err(OdeError::System(message)) => assert!(message.starts_with("no rate"), "{message}"),
            other => panic!("{other:?}"),
        }
        let error = OdeError::<String>::System("no rate".into());
        assert_eq!(error.to_string(), "evaluating the ODE failed: no rate");
    }
    let failed = SolverFailed {
        t: 3.0,
        reason: FailureReason::MaxSteps(10),
    };
    assert_eq!(failed.to_string(), "the ODE solver took 10 steps by t = 3");
}

#[test]
fn an_empty_state_needs_no_integration() {
    let states = integrate(
        |_, _, _| {},
        |_, _, _, _| {},
        0.0,
        &[],
        &[1.0, 2.0],
        &OdeOptions::default(),
    )
    .unwrap();
    assert_eq!(states, vec![Vec::<f64>::new(); 2]);
}

// ---------------------------------------------------------------------------------------------------------------
// forward sensitivities

#[test]
fn sensitivities_of_exponential_decay_match_their_closed_forms() {
    let (rate, y0, t0) = (0.8, 2.0, 0.5);
    // with respect to (y0, rate, t0): S(t0) = (1, 0, -f(t0, y0))
    let mut system = FnSystem::with_sensitivities(
        |_, y, dy| dy[0] = -rate * y[0],
        |_, _, j, dt| {
            j[0] = -rate;
            dt[0] = 0.0;
        },
        |_, y, s, ds| {
            for k in 0..3 {
                ds[k] = -rate * s[k];
            }
            ds[1] -= y[0];
        },
    );
    let ts = times(6.0, 25)
        .into_iter()
        .map(|t| t + t0)
        .collect::<Vec<_>>();
    for solver in SOLVERS {
        let solution = solve_sensitivities(
            &mut system,
            t0,
            &[y0],
            &[1.0, 0.0, rate * y0],
            &ts,
            &tight(solver),
        )
        .unwrap();
        for ((t, y), s) in ts.iter().zip(&solution.states).zip(&solution.sensitivities) {
            let decay = (-rate * (t - t0)).exp();
            let expected = [decay, -(t - t0) * y0 * decay, rate * y0 * decay];
            assert!((y[0] - y0 * decay).abs() < 1e-8);
            for (a, e) in s.iter().zip(expected) {
                assert!(
                    (a - e).abs() < 1e-7 * e.abs().max(1.0),
                    "{solver:?}: {a} != {e}"
                );
            }
        }
    }
}

/// ∂y(tᵢ)/∂pₖ by central differences of tight solves, row-major [time][n × p].
fn finite_difference_sensitivities(
    solve_at: impl Fn(&[f64]) -> Vec<Vec<f64>>,
    parameters: &[f64],
    n: usize,
) -> Vec<Vec<f64>> {
    let p = parameters.len();
    let base = solve_at(parameters);
    let mut out = vec![vec![0.0; n * p]; base.len()];
    for k in 0..p {
        let h = 1e-5 * parameters[k].abs().max(1.0);
        let shifted = |delta: f64| {
            let mut q = parameters.to_vec();
            q[k] += delta;
            solve_at(&q)
        };
        let (up, down) = (shifted(h), shifted(-h));
        for (i, row) in out.iter_mut().enumerate() {
            for j in 0..n {
                row[j * p + k] = (up[i][j] - down[i][j]) / (2.0 * h);
            }
        }
    }
    out
}

fn assert_sensitivities_close(
    actual: &[Vec<f64>],
    expected: &[Vec<f64>],
    tolerance: f64,
    what: &str,
) {
    let scale = expected
        .iter()
        .flatten()
        .fold(1.0f64, |m, x| m.max(x.abs()));
    for (a, e) in actual.iter().flatten().zip(expected.iter().flatten()) {
        assert!((a - e).abs() <= tolerance * scale, "{what}: {a} != {e}");
    }
}

#[test]
fn lotka_volterra_sensitivities_match_finite_differences() {
    let ts = times(10.0, 30);
    let y0 = [1.0, 1.0];
    // parameters: (y0₁, y0₂, α, β, γ, δ)
    let parameters: Vec<f64> = y0.iter().chain(&LOTKA_VOLTERRA).copied().collect();
    let reference_options = OdeOptions::new(OdeSolver::Rk45).with_tolerances(1e-12, 1e-12);
    let expected = finite_difference_sensitivities(
        |q| {
            let mut system = LotkaVolterra([q[2], q[3], q[4], q[5]]);
            solve(&mut system, 0.0, &q[..2], &ts, &reference_options)
                .unwrap()
                .states
        },
        &parameters,
        2,
    );
    let mut s0 = vec![0.0; 12];
    s0[0] = 1.0;
    s0[6 + 1] = 1.0;
    for solver in SOLVERS {
        let solution = solve_sensitivities(
            &mut LotkaVolterra(LOTKA_VOLTERRA),
            0.0,
            &y0,
            &s0,
            &ts,
            &tight(solver),
        )
        .unwrap();
        assert_sensitivities_close(
            &solution.sensitivities,
            &expected,
            1e-6,
            &format!("{solver:?}"),
        );
    }
}

/// Robertson's chemical kinetics, the classic stiff test problem, with its first rate constant (0.04) as a
/// parameter, and its sensitivities with respect to it.
struct Robertson(f64);

impl OdeSystem for Robertson {
    type Error = Infallible;

    fn rhs(&mut self, _: f64, y: &[f64], dy: &mut [f64]) -> Result<(), Infallible> {
        dy[0] = -self.0 * y[0] + 1e4 * y[1] * y[2];
        dy[2] = 3e7 * y[1] * y[1];
        dy[1] = -dy[0] - dy[2];
        Ok(())
    }

    fn jacobian(
        &mut self,
        _: f64,
        y: &[f64],
        j: &mut [f64],
        dt: &mut [f64],
    ) -> Result<(), Infallible> {
        j.copy_from_slice(&[
            -self.0,
            1e4 * y[2],
            1e4 * y[1],
            self.0,
            -1e4 * y[2] - 6e7 * y[1],
            -1e4 * y[1],
            0.0,
            6e7 * y[1],
            0.0,
        ]);
        dt.fill(0.0);
        Ok(())
    }
}

impl SensitivitySystem for Robertson {
    fn sensitivity_rhs(
        &mut self,
        t: f64,
        y: &[f64],
        s: &[f64],
        ds: &mut [f64],
    ) -> Result<(), Infallible> {
        let (mut j, mut dt) = ([0.0; 9], [0.0; 3]);
        self.jacobian(t, y, &mut j, &mut dt)?;
        for i in 0..3 {
            ds[i] = (0..3).map(|m| j[i * 3 + m] * s[m]).sum();
        }
        ds[0] -= y[0];
        ds[1] += y[0];
        Ok(())
    }
}

#[test]
fn stiff_sensitivities_match_finite_differences() {
    let ts = [0.4, 4.0, 40.0];
    let reference_options = OdeOptions::new(OdeSolver::Stiff).with_tolerances(1e-12, 1e-16);
    let expected = finite_difference_sensitivities(
        |q| {
            solve(
                &mut Robertson(q[0]),
                0.0,
                &[1.0, 0.0, 0.0],
                &ts,
                &reference_options,
            )
            .unwrap()
            .states
        },
        &[0.04],
        3,
    );
    let options = OdeOptions::new(OdeSolver::Stiff).with_tolerances(1e-8, 1e-12);
    let solution = solve_sensitivities(
        &mut Robertson(0.04),
        0.0,
        &[1.0, 0.0, 0.0],
        &[0.0; 3],
        &ts,
        &options,
    )
    .unwrap();
    assert_sensitivities_close(&solution.sensitivities, &expected, 1e-7, "Robertson");
}

#[test]
fn initial_sensitivities_must_match_the_state() {
    let result = solve_sensitivities(
        &mut LotkaVolterra(LOTKA_VOLTERRA),
        0.0,
        &[1.0, 1.0],
        &[1.0, 0.0, 0.0],
        &[1.0],
        &OdeOptions::default(),
    );
    assert!(
        matches!(result, Err(OdeError::InvalidProblem(_))),
        "{result:?}"
    );
}

/// y' = −k (y − sin t) + cos t, y(0) = 1 has y = sin t + e^(−kt) and ∂y/∂k = −t e^(−kt); its sensitivity equation
/// depends on t, which the stiff solver differentiates.
#[test]
fn sensitivities_of_a_non_autonomous_problem_match_their_closed_form() {
    let k = 50.0;
    let mut system = FnSystem::with_sensitivities(
        |t, y, dy| dy[0] = -k * (y[0] - t.sin()) + t.cos(),
        |t, _, j, dt| {
            j[0] = -k;
            dt[0] = k * t.cos() - t.sin();
        },
        |t, y, s, ds| ds[0] = -k * s[0] - (y[0] - t.sin()),
    );
    let ts = times(2.0, 20);
    for solver in SOLVERS {
        let solution =
            solve_sensitivities(&mut system, 0.0, &[1.0], &[0.0], &ts, &tight(solver)).unwrap();
        for (t, s) in ts.iter().zip(&solution.sensitivities) {
            let expected = -t * (-k * t).exp();
            assert!(
                (s[0] - expected).abs() < 1e-8,
                "{solver:?}: {} != {expected}",
                s[0]
            );
        }
    }
}

/// At a large initial time a zero initial state no longer gives a first step too small to advance t:
/// y' = cos(10⁻¹² t), y(10¹²) = 0, whose solution is 10¹² (sin(10⁻¹² t) − sin 1).
#[test]
fn a_zero_state_at_a_large_initial_time_takes_a_step_that_advances_time() {
    let t0 = 1e12;
    for solver in SOLVERS {
        let ts = [t0 + 10.0, t0 + 1000.0];
        let states = integrate(
            |t, _, dy| dy[0] = (1e-12 * t).cos(),
            |t, _, j, dt| {
                j[0] = 0.0;
                dt[0] = -1e-12 * (1e-12 * t).sin();
            },
            t0,
            &[0.0],
            &ts,
            &OdeOptions::new(solver),
        )
        .unwrap_or_else(|e| panic!("{solver:?}: {e}"));
        for (t, y) in ts.iter().zip(&states) {
            // 10¹² (sin(10⁻¹² t) − sin 1) ≈ cos(1) (t − t0) for t − t0 ≪ 10¹²; times near 10¹² are only
            // resolved to about 10⁻⁴
            let expected = 1f64.cos() * (t - t0);
            assert!(
                (y[0] - expected).abs() < 1e-3 + 1e-6 * expected,
                "{solver:?}: y({t}) = {}, expected {expected}",
                y[0]
            );
        }
    }
}
