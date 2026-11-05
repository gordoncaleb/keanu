//! Integrators for ordinary differential equations dy/dt = f(t, y), on plain `&[f64]` states: an explicit
//! Dormand–Prince 5(4) method with dense output for non-stiff problems ([`OdeSolver::Rk45`]) and a linearly implicit
//! Rosenbrock method of order 4 with an embedded order 3 estimate (Hairer and Wanner's Rodas4) for stiff ones
//! ([`OdeSolver::Stiff`]). Both adapt their step size to keep each step's estimated local error below
//! `atol + rtol · |y|` per component, and report the solution at the requested output times.
//!
//! The quickest entry point is [`integrate`], with closures for f and its Jacobian. Systems whose evaluation can
//! fail implement [`OdeSystem`] and call [`solve`]. Forward sensitivities S = ∂y/∂p with respect to any parameters p
//! (initial values, parameters of f, the initial time) are integrated alongside the state by [`solve_sensitivities`]
//! for a [`SensitivitySystem`], whose sensitivity equations are dS/dt = J S + ∂f/∂p with J = ∂f/∂y; the step-size
//! control then covers the sensitivities as well as the state.
//!
//! ```
//! use keanu_graph::ode::{OdeOptions, OdeSolver, integrate};
//!
//! // dy/dt = -2 y, y(0) = 1
//! let options = OdeOptions::new(OdeSolver::Rk45).with_tolerances(1e-10, 1e-10);
//! let states = integrate(
//!     |_t, y, dydt| dydt[0] = -2.0 * y[0],
//!     |_t, _y, dfdy, dfdt| {
//!         dfdy[0] = -2.0;
//!         dfdt[0] = 0.0;
//!     },
//!     0.0,
//!     &[1.0],
//!     &[0.5, 1.0],
//!     &options,
//! )?;
//! assert!((states[1][0] - (-2.0f64).exp()).abs() < 1e-9);
//! # Ok::<(), keanu_graph::ode::OdeError>(())
//! ```

mod dopri5;
mod rosenbrock;

use std::convert::Infallible;
use std::fmt;

use keanu_tensor::LuFactorization;

/// The integration method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OdeSolver {
    /// The explicit Dormand–Prince 5(4) Runge–Kutta method (as MATLAB's `ode45`, SciPy's `RK45` and Stan's
    /// `ode_rk45`), with its fourth-order dense output between steps. For non-stiff problems; it needs no Jacobian.
    #[default]
    Rk45,
    /// A linearly implicit (Rosenbrock) method of order 4 with an embedded order 3 error estimate, with the
    /// coefficients of Hairer and Wanner's Rodas4. It is L-stable and stiffly accurate, for stiff problems. It needs
    /// the exact Jacobian ∂f/∂y and time derivative ∂f/∂t, and solves a linear system with them at every stage.
    Stiff,
}

/// Options of an ODE solve: the method, tolerances and step limit. The defaults are Stan's: relative and absolute
/// tolerances of 10⁻⁶ and at most 10⁶ steps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OdeOptions {
    /// The method.
    pub solver: OdeSolver,
    /// The relative tolerance (≥ 0).
    pub rtol: f64,
    /// The absolute tolerance (≥ 0, and not 0 together with `rtol`).
    pub atol: f64,
    /// The largest number of steps (accepted or rejected) over the whole integration (> 0).
    pub max_steps: usize,
}

impl Default for OdeOptions {
    fn default() -> Self {
        Self::new(OdeSolver::default())
    }
}

impl OdeOptions {
    /// The default relative tolerance, 10⁻⁶.
    pub const DEFAULT_RTOL: f64 = 1e-6;
    /// The default absolute tolerance, 10⁻⁶.
    pub const DEFAULT_ATOL: f64 = 1e-6;
    /// The default step limit, 10⁶.
    pub const DEFAULT_MAX_STEPS: usize = 1_000_000;

    /// The default options with the given method.
    pub fn new(solver: OdeSolver) -> Self {
        Self {
            solver,
            rtol: Self::DEFAULT_RTOL,
            atol: Self::DEFAULT_ATOL,
            max_steps: Self::DEFAULT_MAX_STEPS,
        }
    }

    /// With the given relative and absolute tolerances.
    pub fn with_tolerances(self, rtol: f64, atol: f64) -> Self {
        Self { rtol, atol, ..self }
    }

    /// With the given step limit.
    pub fn with_max_steps(self, max_steps: usize) -> Self {
        Self { max_steps, ..self }
    }

    /// Checks the tolerances and step limit, as every solve does first.
    pub fn validate(&self) -> Result<(), String> {
        let valid = |x: f64| x.is_finite() && x >= 0.0;
        if !valid(self.rtol) || !valid(self.atol) || self.rtol + self.atol == 0.0 {
            return Err(format!(
                "the tolerances must be finite and non-negative, and not both zero; got rtol {} and atol {}",
                self.rtol, self.atol
            ));
        }
        if self.max_steps == 0 {
            return Err("the step limit must be positive".into());
        }
        Ok(())
    }
}

/// Why an integration failed (see [`SolverFailed`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FailureReason {
    /// The step limit ([`OdeOptions::max_steps`]) was reached: the problem is stiff for [`OdeSolver::Rk45`], or the
    /// tolerances are too tight.
    MaxSteps(usize),
    /// The step size became negligible relative to the time (it is given), usually near a singularity of the
    /// solution, or because the right-hand side keeps giving non-finite values.
    StepSizeUnderflow(f64),
    /// The initial state, or the right-hand side or Jacobian there, is not finite.
    NonFinite,
}

/// An integration that failed partway, at time `t`: the problem (or this point of a model's parameters) can't be
/// solved with these options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverFailed {
    /// The time the integration had reached.
    pub t: f64,
    /// Why it stopped.
    pub reason: FailureReason,
}

impl fmt::Display for SolverFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.reason {
            FailureReason::MaxSteps(n) => {
                write!(f, "the ODE solver took {n} steps by t = {}", self.t)
            }
            FailureReason::StepSizeUnderflow(h) => write!(
                f,
                "the ODE solver's step size fell to {h:e} at t = {}",
                self.t
            ),
            FailureReason::NonFinite => {
                write!(
                    f,
                    "the ODE's state or derivatives are not finite at t = {}",
                    self.t
                )
            }
        }
    }
}

impl std::error::Error for SolverFailed {}

/// An error of an ODE solve. `E` is the error type of the system's own evaluations ([`OdeSystem::Error`]).
#[derive(Debug, Clone, PartialEq)]
pub enum OdeError<E = Infallible> {
    /// The problem is malformed: output times not finite, not strictly increasing or not after the initial time,
    /// invalid options, or inconsistent dimensions.
    InvalidProblem(String),
    /// The integrator failed (see [`SolverFailed`]).
    SolverFailed(SolverFailed),
    /// Evaluating the system failed.
    System(E),
}

impl<E: fmt::Display> fmt::Display for OdeError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OdeError::InvalidProblem(message) => write!(f, "invalid ODE problem: {message}"),
            OdeError::SolverFailed(failed) => failed.fmt(f),
            OdeError::System(e) => write!(f, "evaluating the ODE failed: {e}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for OdeError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OdeError::SolverFailed(failed) => Some(failed),
            OdeError::System(e) => Some(e),
            OdeError::InvalidProblem(_) => None,
        }
    }
}

/// An ODE dy/dt = f(t, y) with n-dimensional states. Matrices are row-major n × n.
pub trait OdeSystem {
    /// The error of an evaluation.
    type Error;

    /// Writes f(t, y) into `dydt`.
    fn rhs(&mut self, t: f64, y: &[f64], dydt: &mut [f64]) -> Result<(), Self::Error>;

    /// Writes the Jacobian ∂f/∂y (`dfdy`, row-major n × n: element (i, j) is ∂fᵢ/∂yⱼ) and ∂f/∂t (`dfdt`) at (t, y).
    /// Only [`OdeSolver::Stiff`] calls it.
    fn jacobian(
        &mut self,
        t: f64,
        y: &[f64],
        dfdy: &mut [f64],
        dfdt: &mut [f64],
    ) -> Result<(), Self::Error>;
}

/// An [`OdeSystem`] with forward sensitivities S = ∂y/∂p with respect to p parameters, held row-major n × p
/// (element (i, k) is ∂yᵢ/∂pₖ). They satisfy dS/dt = J S + ∂f/∂p, where J = ∂f/∂y and ∂f/∂p is zero for
/// parameters that only enter the initial state. Initial values S(t₀) are given to [`solve_sensitivities`]: the
/// identity columns for the initial state, zero for parameters of f, and −f(t₀, y₀) for the initial time.
pub trait SensitivitySystem: OdeSystem {
    /// Writes dS/dt = J S + ∂f/∂p at (t, y, S) into `ds` (both n × p).
    fn sensitivity_rhs(
        &mut self,
        t: f64,
        y: &[f64],
        s: &[f64],
        ds: &mut [f64],
    ) -> Result<(), Self::Error>;

    /// Writes the derivative of [`Self::sensitivity_rhs`] with respect to y along `x` (an n-vector), holding t and S
    /// fixed, into `out` (n × p): its column k is (∂/∂y)(J sₖ + ∂f/∂pₖ) · x, which holds second derivatives of f.
    /// [`OdeSolver::Stiff`] needs it for the exact Jacobian of the combined system. The default is a central
    /// difference of [`Self::sensitivity_rhs`] (accurate to about 10⁻¹⁰ relative).
    fn sensitivity_jacobian_product(
        &mut self,
        t: f64,
        y: &[f64],
        s: &[f64],
        x: &[f64],
        out: &mut [f64],
    ) -> Result<(), Self::Error> {
        let size = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if size == 0.0 {
            out.fill(0.0);
            return Ok(());
        }
        let scale = y.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        let delta = DIFFERENCE_STEP * scale / size;
        let shifted = |sign: f64| -> Vec<f64> {
            y.iter().zip(x).map(|(y, x)| y + sign * delta * x).collect()
        };
        let mut down = vec![0.0; out.len()];
        self.sensitivity_rhs(t, &shifted(1.0), s, out)?;
        self.sensitivity_rhs(t, &shifted(-1.0), s, &mut down)?;
        out.iter_mut()
            .zip(&down)
            .for_each(|(o, d)| *o = (*o - d) / (2.0 * delta));
        Ok(())
    }

    /// Writes the derivative of [`Self::sensitivity_rhs`] with respect to t, holding y and S fixed, into `out`
    /// (n × p), for [`OdeSolver::Stiff`]. The default is a central difference; an autonomous system's is zero.
    fn sensitivity_time_derivative(
        &mut self,
        t: f64,
        y: &[f64],
        s: &[f64],
        out: &mut [f64],
    ) -> Result<(), Self::Error> {
        let delta = DIFFERENCE_STEP * t.abs().max(1.0);
        let mut down = vec![0.0; out.len()];
        self.sensitivity_rhs(t + delta, y, s, out)?;
        self.sensitivity_rhs(t - delta, y, s, &mut down)?;
        out.iter_mut()
            .zip(&down)
            .for_each(|(o, d)| *o = (*o - d) / (2.0 * delta));
        Ok(())
    }
}

/// The relative step of the default central differences: about the cube root of the unit roundoff.
const DIFFERENCE_STEP: f64 = 6e-6;

/// An [`OdeSystem`] (and [`SensitivitySystem`]) from closures that can't fail: f(t, y, dydt), the Jacobian
/// (t, y, dfdy, dfdt) and, optionally, the sensitivity right-hand side (t, y, S, dS).
pub struct FnSystem<F, J, G = fn(f64, &[f64], &[f64], &mut [f64])> {
    rhs: F,
    jacobian: J,
    sensitivities: G,
}

impl<F, J, G> fmt::Debug for FnSystem<F, J, G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FnSystem").finish_non_exhaustive()
    }
}

impl<F, J> FnSystem<F, J>
where
    F: FnMut(f64, &[f64], &mut [f64]),
    J: FnMut(f64, &[f64], &mut [f64], &mut [f64]),
{
    /// A system from f and its Jacobian, without sensitivities.
    pub fn new(rhs: F, jacobian: J) -> Self {
        Self {
            rhs,
            jacobian,
            sensitivities: |_, _, _, ds| ds.fill(0.0),
        }
    }
}

impl<F, J, G> FnSystem<F, J, G>
where
    F: FnMut(f64, &[f64], &mut [f64]),
    J: FnMut(f64, &[f64], &mut [f64], &mut [f64]),
    G: FnMut(f64, &[f64], &[f64], &mut [f64]),
{
    /// A system from f, its Jacobian and the sensitivity right-hand side dS/dt = J S + ∂f/∂p.
    pub fn with_sensitivities(rhs: F, jacobian: J, sensitivities: G) -> Self {
        Self {
            rhs,
            jacobian,
            sensitivities,
        }
    }
}

impl<F, J, G> OdeSystem for FnSystem<F, J, G>
where
    F: FnMut(f64, &[f64], &mut [f64]),
    J: FnMut(f64, &[f64], &mut [f64], &mut [f64]),
{
    type Error = Infallible;

    fn rhs(&mut self, t: f64, y: &[f64], dydt: &mut [f64]) -> Result<(), Infallible> {
        (self.rhs)(t, y, dydt);
        Ok(())
    }

    fn jacobian(
        &mut self,
        t: f64,
        y: &[f64],
        dfdy: &mut [f64],
        dfdt: &mut [f64],
    ) -> Result<(), Infallible> {
        (self.jacobian)(t, y, dfdy, dfdt);
        Ok(())
    }
}

impl<F, J, G> SensitivitySystem for FnSystem<F, J, G>
where
    F: FnMut(f64, &[f64], &mut [f64]),
    J: FnMut(f64, &[f64], &mut [f64], &mut [f64]),
    G: FnMut(f64, &[f64], &[f64], &mut [f64]),
{
    fn sensitivity_rhs(
        &mut self,
        t: f64,
        y: &[f64],
        s: &[f64],
        ds: &mut [f64],
    ) -> Result<(), Infallible> {
        (self.sensitivities)(t, y, s, ds);
        Ok(())
    }
}

/// Counts of an integration's work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OdeStatistics {
    /// Accepted steps.
    pub accepted_steps: usize,
    /// Rejected steps (too large an error estimate, a non-finite stage or a singular stage matrix).
    pub rejected_steps: usize,
    /// Evaluations of the right-hand side (of the state and sensitivities together, when integrated).
    pub rhs_evaluations: usize,
    /// Evaluations of the Jacobian.
    pub jacobian_evaluations: usize,
}

/// The solution at the output times.
#[derive(Debug, Clone, PartialEq)]
pub struct OdeSolution {
    /// The state y(tᵢ) at each output time.
    pub states: Vec<Vec<f64>>,
    /// The sensitivities S(tᵢ) = ∂y(tᵢ)/∂p (row-major n × p) at each output time; empty without sensitivities.
    pub sensitivities: Vec<Vec<f64>>,
    /// The work done.
    pub statistics: OdeStatistics,
}

/// Solves dy/dt = f(t, y), y(t₀) = y₀, giving y at each of the output `times`, which must be finite, strictly
/// increasing and after t₀. `rhs` writes f(t, y) into its last argument; `jacobian` writes ∂f/∂y (row-major n × n)
/// and ∂f/∂t, and is only called by [`OdeSolver::Stiff`]. See [`solve`].
pub fn integrate<F, J>(
    rhs: F,
    jacobian: J,
    t0: f64,
    y0: &[f64],
    times: &[f64],
    options: &OdeOptions,
) -> Result<Vec<Vec<f64>>, OdeError>
where
    F: FnMut(f64, &[f64], &mut [f64]),
    J: FnMut(f64, &[f64], &mut [f64], &mut [f64]),
{
    Ok(solve(&mut FnSystem::new(rhs, jacobian), t0, y0, times, options)?.states)
}

/// Solves dy/dt = f(t, y), y(t₀) = y₀ for a system, giving y at each of the output `times` (finite, strictly
/// increasing and after t₀).
///
/// Fails with [`OdeError::InvalidProblem`] for malformed times or options, [`OdeError::SolverFailed`] when the
/// integration can't continue (a non-finite initial state, the step limit, or a step size underflow), and
/// [`OdeError::System`] when the system's evaluation fails. A step whose stages are not finite is retried with a
/// smaller step.
pub fn solve<S: OdeSystem>(
    system: &mut S,
    t0: f64,
    y0: &[f64],
    times: &[f64],
    options: &OdeOptions,
) -> Result<OdeSolution, OdeError<S::Error>> {
    validate(t0, times, options)?;
    let mut problem = Plain {
        system,
        n: y0.len(),
        jacobian: vec![0.0; y0.len() * y0.len()],
        lu: None,
    };
    let (states, statistics) = run(&mut problem, t0, y0, times, options)?;
    Ok(OdeSolution {
        states,
        sensitivities: Vec::new(),
        statistics,
    })
}

/// Solves dy/dt = f(t, y), y(t₀) = y₀ together with the sensitivities S = ∂y/∂p, S(t₀) = `s0` (row-major n × p, so
/// its length is a multiple of n), giving both at each output time. The step-size control covers the state and the
/// sensitivities (as CVODES with `errconS`). Errors as [`solve`].
pub fn solve_sensitivities<S: SensitivitySystem>(
    system: &mut S,
    t0: f64,
    y0: &[f64],
    s0: &[f64],
    times: &[f64],
    options: &OdeOptions,
) -> Result<OdeSolution, OdeError<S::Error>> {
    validate(t0, times, options)?;
    let n = y0.len();
    if (n == 0 && !s0.is_empty()) || (n > 0 && s0.len() % n != 0) {
        return Err(OdeError::InvalidProblem(format!(
            "the initial sensitivities of a {n}-dimensional state must be n × p, got {} values",
            s0.len()
        )));
    }
    let p = s0.len().checked_div(n).unwrap_or(0);
    let mut problem = Augmented {
        system,
        n,
        p,
        jacobian: vec![0.0; n * n],
        lu: None,
        t: t0,
        point: Vec::new(),
    };
    let z0: Vec<f64> = y0.iter().chain(s0).copied().collect();
    let (combined, statistics) = run(&mut problem, t0, &z0, times, options)?;
    let (states, sensitivities) = combined
        .into_iter()
        .map(|mut z| {
            let s = z.split_off(n);
            (z, s)
        })
        .unzip();
    Ok(OdeSolution {
        states,
        sensitivities,
        statistics,
    })
}

fn validate<E>(t0: f64, times: &[f64], options: &OdeOptions) -> Result<(), OdeError<E>> {
    options.validate().map_err(OdeError::InvalidProblem)?;
    if !t0.is_finite() {
        return Err(OdeError::InvalidProblem(format!(
            "the initial time must be finite, got {t0}"
        )));
    }
    let mut previous = t0;
    for (i, &t) in times.iter().enumerate() {
        if !t.is_finite() || t <= previous {
            return Err(OdeError::InvalidProblem(format!(
                "the output times must be finite, strictly increasing and after the initial time {t0}; time {i} is {t}"
            )));
        }
        previous = t;
    }
    Ok(())
}

fn run<P: Problem>(
    problem: &mut P,
    t0: f64,
    z0: &[f64],
    times: &[f64],
    options: &OdeOptions,
) -> Integration<P::Error> {
    if times.is_empty() || z0.is_empty() {
        return Ok((vec![z0.to_vec(); times.len()], OdeStatistics::default()));
    }
    if z0.iter().any(|z| !z.is_finite()) {
        return Err(failed(t0, FailureReason::NonFinite));
    }
    match options.solver {
        OdeSolver::Rk45 => dopri5::integrate(problem, t0, z0, times, options),
        OdeSolver::Stiff => rosenbrock::integrate(problem, t0, z0, times, options),
    }
}

/// The states at the output times and the work done, or why the integration failed.
pub(crate) type Integration<E> = Result<(Vec<Vec<f64>>, OdeStatistics), OdeError<E>>;

pub(crate) fn failed<E>(t: f64, reason: FailureReason) -> OdeError<E> {
    OdeError::SolverFailed(SolverFailed { t, reason })
}

/// The system the integrators step, on its full state z (an ODE's state, then its sensitivities if any).
pub(crate) trait Problem {
    type Error;

    /// dz/dt at (t, z).
    fn rhs(&mut self, t: f64, z: &[f64], dz: &mut [f64]) -> Result<(), OdeError<Self::Error>>;

    /// Evaluates and keeps the Jacobian at (t, z), writing ∂(dz/dt)/∂t into `dfdt`.
    fn linearise(
        &mut self,
        t: f64,
        z: &[f64],
        dfdt: &mut [f64],
    ) -> Result<(), OdeError<Self::Error>>;

    /// Factors c I − J for the kept Jacobian J; false if it is singular.
    fn factor(&mut self, c: f64) -> bool;

    /// Solves (c I − J) x = r in place with the last factorisation.
    fn solve(&mut self, r: &mut [f64]) -> Result<(), OdeError<Self::Error>>;
}

/// A plain ODE system.
struct Plain<'a, S> {
    system: &'a mut S,
    n: usize,
    jacobian: Vec<f64>,
    lu: Option<LuFactorization>,
}

/// Factors c I − J for a row-major n × n J; None if singular.
fn factor_shifted(jacobian: &[f64], n: usize, c: f64) -> Option<LuFactorization> {
    let mut matrix: Vec<f64> = jacobian.iter().map(|j| -j).collect();
    for i in 0..n {
        matrix[i * n + i] += c;
    }
    LuFactorization::new(&matrix, n).ok()
}

impl<S: OdeSystem> Problem for Plain<'_, S> {
    type Error = S::Error;

    fn rhs(&mut self, t: f64, z: &[f64], dz: &mut [f64]) -> Result<(), OdeError<S::Error>> {
        self.system.rhs(t, z, dz).map_err(OdeError::System)
    }

    fn linearise(&mut self, t: f64, z: &[f64], dfdt: &mut [f64]) -> Result<(), OdeError<S::Error>> {
        self.system
            .jacobian(t, z, &mut self.jacobian, dfdt)
            .map_err(OdeError::System)?;
        if self.jacobian.iter().chain(&*dfdt).any(|x| !x.is_finite()) {
            return Err(failed(t, FailureReason::NonFinite));
        }
        Ok(())
    }

    fn factor(&mut self, c: f64) -> bool {
        self.lu = factor_shifted(&self.jacobian, self.n, c);
        self.lu.is_some()
    }

    fn solve(&mut self, r: &mut [f64]) -> Result<(), OdeError<S::Error>> {
        self.lu
            .as_ref()
            .expect("factored before solving")
            .solve_in_place(r);
        Ok(())
    }
}

/// A system with its sensitivities: z = [y, S]. Its Jacobian is block lower triangular, [[J, 0], [M, I ⊗ J]] with
/// M x = (∂/∂y)(J S + ∂f/∂p) · x, so its solves need only J's factorisation and one product with M.
struct Augmented<'a, S> {
    system: &'a mut S,
    n: usize,
    p: usize,
    jacobian: Vec<f64>,
    lu: Option<LuFactorization>,
    /// the point (t, z) of the last linearisation
    t: f64,
    point: Vec<f64>,
}

impl<S: SensitivitySystem> Problem for Augmented<'_, S> {
    type Error = S::Error;

    fn rhs(&mut self, t: f64, z: &[f64], dz: &mut [f64]) -> Result<(), OdeError<S::Error>> {
        let (y, s) = z.split_at(self.n);
        let (dy, ds) = dz.split_at_mut(self.n);
        self.system.rhs(t, y, dy).map_err(OdeError::System)?;
        self.system
            .sensitivity_rhs(t, y, s, ds)
            .map_err(OdeError::System)
    }

    fn linearise(&mut self, t: f64, z: &[f64], dfdt: &mut [f64]) -> Result<(), OdeError<S::Error>> {
        let (y, s) = z.split_at(self.n);
        let (dy, ds) = dfdt.split_at_mut(self.n);
        self.system
            .jacobian(t, y, &mut self.jacobian, dy)
            .map_err(OdeError::System)?;
        self.system
            .sensitivity_time_derivative(t, y, s, ds)
            .map_err(OdeError::System)?;
        if self.jacobian.iter().chain(&*dfdt).any(|x| !x.is_finite()) {
            return Err(failed(t, FailureReason::NonFinite));
        }
        self.t = t;
        self.point.clear();
        self.point.extend_from_slice(z);
        Ok(())
    }

    fn factor(&mut self, c: f64) -> bool {
        self.lu = factor_shifted(&self.jacobian, self.n, c);
        self.lu.is_some()
    }

    fn solve(&mut self, r: &mut [f64]) -> Result<(), OdeError<S::Error>> {
        let lu = self.lu.as_ref().expect("factored before solving");
        let (ry, rs) = r.split_at_mut(self.n);
        lu.solve_in_place(ry);
        if self.p > 0 {
            // (c I − J) xₖ = rₖ + Mₖ x_y for every sensitivity column at once
            let (y, s) = self.point.split_at(self.n);
            let mut product = vec![0.0; self.n * self.p];
            self.system
                .sensitivity_jacobian_product(self.t, y, s, ry, &mut product)
                .map_err(OdeError::System)?;
            rs.iter_mut().zip(&product).for_each(|(r, m)| *r += m);
            lu.solve_in_place(rs);
        }
        Ok(())
    }
}

/// The weighted root-mean-square norm of an error estimate, with weights atol + rtol · max(|z₀|, |z₁|).
pub(crate) fn error_norm(error: &[f64], z0: &[f64], z1: &[f64], options: &OdeOptions) -> f64 {
    let sum: f64 = error
        .iter()
        .zip(z0.iter().zip(z1))
        .map(|(e, (a, b))| {
            let scale = options.atol + options.rtol * a.abs().max(b.abs());
            (e / scale).powi(2)
        })
        .sum();
    (sum / error.len() as f64).sqrt()
}

/// The initial step size by Hairer, Nørsett and Wanner's heuristic (Solving ODEs I, II.4): a step that makes an
/// explicit Euler step's error estimate small relative to the tolerances, for a method of the given order.
pub(crate) fn initial_step<P: Problem>(
    problem: &mut P,
    t0: f64,
    z0: &[f64],
    f0: &[f64],
    order: i32,
    span: f64,
    options: &OdeOptions,
) -> Result<f64, OdeError<P::Error>> {
    let norm = |v: &[f64]| error_norm(v, z0, z0, options);
    let (d0, d1) = (norm(z0), norm(f0));
    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    }
    .min(span);
    let z1: Vec<f64> = z0.iter().zip(f0).map(|(z, f)| z + h0 * f).collect();
    let mut f1 = vec![0.0; z0.len()];
    problem.rhs(t0 + h0, &z1, &mut f1)?;
    let difference: Vec<f64> = f1.iter().zip(f0).map(|(a, b)| a - b).collect();
    let d2 = norm(&difference) / h0;
    let largest = d1.max(d2);
    let h1 = if !largest.is_finite() {
        h0 * 1e-3
    } else if largest <= 1e-15 {
        (h0 * 1e-3).max(1e-6)
    } else {
        (0.01 / largest).powf(1.0 / order as f64)
    };
    Ok((100.0 * h0).min(h1).min(span))
}
