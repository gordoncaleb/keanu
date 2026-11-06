use super::Objective;
use crate::error::{AlgorithmError, Result};
use crate::support::{dot, norm};
use keanu_graph::{GraphError, HessianMethod, Parameterisation, State};

/// Trust-region Newton (Nocedal & Wright, *Numerical Optimization*, 2nd ed., Algorithm 4.1), with each step from
/// the Steihaug–Toint truncated conjugate gradient method (Algorithm 7.2) on Hessian-vector products.
///
/// Each iteration minimises the quadratic model m(p) = gᵀp + ½ pᵀBp of the negated fitness (B its Hessian) within a
/// ball of radius Δ by conjugate gradient, which needs only products Bv: it stops when the model's residual is below
/// min(0.5, √‖g‖)·‖g‖, when a step reaches the edge of the ball, after as many iterations as there are coordinates,
/// or when it meets a direction of zero or negative curvature, which it follows to the edge (so an indefinite
/// Hessian, at a saddle or far from the optimum, gives a step to the trust region's boundary rather than towards a
/// maximum of the model). The step is accepted if the ratio ρ of the actual to the predicted decrease is above `eta`;
/// Δ is quartered (to a quarter of the step's length) when ρ < ¼ and doubled, up to `max_radius`, when ρ > ¾ and the
/// step reached the edge. A trial point whose fitness or gradient is not finite (an impossible point) has ρ = −∞.
///
/// The products are of the Hessian in unconstrained space without the Jacobian, as for the other optimizers, by
/// `hessian_method` (default [`HessianMethod::Auto`]: exact by automatic differentiation unless the model has a
/// lambda, custom distribution or ODE without a second-order rule, then by central differences of the gradient; see
/// [`Parameterisation::hvp_with`]). An exact product costs about as much as two gradient evaluations (a forward
/// sweep of tangents and a reverse sweep carrying them); a finite-difference one is two gradient evaluations.
/// [`OptimizedResult::hessian_vector_products`](super::OptimizedResult::hessian_vector_products) counts them,
/// separately from `evaluations` (two per trial point: the fitness and its gradient).
///
/// It converges when the gradient's norm is below `tol_grad`, when an accepted step changes the fitness by less than
/// `tol_rel_obj` × ε relative to its magnitude (ε being machine epsilon) or moves the point less than `tol_param`, or
/// when Δ shrinks below `tol_param` (no step longer than that improves the fitness). It stops without converging
/// after `max_iterations` iterations, when a Hessian-vector product is not finite, and, like the others, at the first
/// accepted point past the boundary threshold of a bounded latent that is heading for the boundary (see the
/// [module documentation](super)).
///
/// A starting point whose fitness is not finite is an [`AlgorithmError::ImpossibleStart`], even without the
/// initial-condition check.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct TrustRegion {
    /// The initial trust-region radius Δ₀ (default 1).
    pub initial_radius: f64,
    /// The largest radius (default 1e3).
    pub max_radius: f64,
    /// The smallest ratio of actual to predicted decrease for which a step is accepted, η (default 1e-4).
    pub eta: f64,
    /// The maximum number of iterations (default 1000).
    pub max_iterations: usize,
    /// Norm of the gradient (default 1e-8).
    pub tol_grad: f64,
    /// Relative change in the fitness at an accepted step, in units of machine epsilon (default 1e4).
    pub tol_rel_obj: f64,
    /// Length of an accepted step, and the smallest radius (default 1e-8).
    pub tol_param: f64,
    /// How the Hessian-vector products are computed (default [`HessianMethod::Auto`]).
    pub hessian_method: HessianMethod,
}

impl Default for TrustRegion {
    fn default() -> Self {
        Self {
            initial_radius: 1.0,
            max_radius: 1e3,
            eta: 1e-4,
            max_iterations: 1000,
            tol_grad: 1e-8,
            tol_rel_obj: 1e4,
            tol_param: 1e-8,
            hessian_method: HessianMethod::Auto,
        }
    }
}

/// Setters for each setting, for use as `TrustRegion::default().with_...(...)`.
impl TrustRegion {
    /// Sets [`initial_radius`](Self::initial_radius).
    #[must_use]
    pub fn with_initial_radius(mut self, initial_radius: f64) -> Self {
        self.initial_radius = initial_radius;
        self
    }

    /// Sets [`max_radius`](Self::max_radius).
    #[must_use]
    pub fn with_max_radius(mut self, max_radius: f64) -> Self {
        self.max_radius = max_radius;
        self
    }

    /// Sets [`eta`](Self::eta).
    #[must_use]
    pub fn with_eta(mut self, eta: f64) -> Self {
        self.eta = eta;
        self
    }

    /// Sets [`max_iterations`](Self::max_iterations).
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets [`tol_grad`](Self::tol_grad).
    #[must_use]
    pub fn with_tol_grad(mut self, tol_grad: f64) -> Self {
        self.tol_grad = tol_grad;
        self
    }

    /// Sets [`tol_rel_obj`](Self::tol_rel_obj).
    #[must_use]
    pub fn with_tol_rel_obj(mut self, tol_rel_obj: f64) -> Self {
        self.tol_rel_obj = tol_rel_obj;
        self
    }

    /// Sets [`tol_param`](Self::tol_param).
    #[must_use]
    pub fn with_tol_param(mut self, tol_param: f64) -> Self {
        self.tol_param = tol_param;
        self
    }

    /// Sets [`hessian_method`](Self::hessian_method).
    #[must_use]
    pub fn with_hessian_method(mut self, hessian_method: HessianMethod) -> Self {
        self.hessian_method = hessian_method;
        self
    }
}

/// Where a minimisation ended.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Minimum {
    pub(crate) point: Vec<f64>,
    pub(crate) value: f64,
    pub(crate) converged: bool,
    pub(crate) iterations: usize,
    pub(crate) hessian_vector_products: usize,
}

/// A step of the Steihaug–Toint subproblem: the step p, the product Bp, and whether it reached the edge of the
/// trust region. `None` if a product was not finite.
type Step = Option<(Vec<f64>, Vec<f64>, bool)>;

fn axpy(a: f64, x: &[f64], y: &[f64]) -> Vec<f64> {
    x.iter().zip(y).map(|(xi, yi)| a * xi + yi).collect()
}

/// The τ ≥ 0 at which ‖z + τd‖ = Δ, for ‖z‖ ≤ Δ.
fn to_edge(z: &[f64], d: &[f64], radius: f64) -> f64 {
    let a = dot(d, d);
    let b = 2.0 * dot(z, d);
    let c = dot(z, z) - radius * radius;
    let root = (b * b - 4.0 * a * c).max(0.0).sqrt();
    // the positive root, written to avoid cancellation
    if b >= 0.0 {
        (-2.0 * c) / (b + root)
    } else {
        (root - b) / (2.0 * a)
    }
}

/// Approximately minimises gᵀp + ½ pᵀBp over ‖p‖ ≤ Δ by the Steihaug–Toint conjugate gradient method, with `hvp`
/// giving Bv.
pub(crate) fn steihaug(
    g: &[f64],
    radius: f64,
    mut hvp: impl FnMut(&[f64]) -> Result<Vec<f64>>,
) -> Result<Step> {
    let n = g.len();
    let g_norm = norm(g);
    let tolerance = g_norm.sqrt().min(0.5) * g_norm;
    let mut z = vec![0.0; n];
    let mut bz = vec![0.0; n];
    let mut r = g.to_vec();
    let mut d: Vec<f64> = g.iter().map(|x| -x).collect();
    let mut rr = dot(&r, &r);
    for _ in 0..n.max(1) {
        let bd = hvp(&d)?;
        let curvature = dot(&d, &bd);
        if !curvature.is_finite() {
            return Ok(None);
        }
        if curvature <= 0.0 {
            // zero or negative curvature: the model decreases without bound along d, so go to the edge
            let tau = to_edge(&z, &d, radius);
            return Ok(Some((axpy(tau, &d, &z), axpy(tau, &bd, &bz), true)));
        }
        let alpha = rr / curvature;
        let next = axpy(alpha, &d, &z);
        if norm(&next) >= radius {
            let tau = to_edge(&z, &d, radius);
            return Ok(Some((axpy(tau, &d, &z), axpy(tau, &bd, &bz), true)));
        }
        z = next;
        bz = axpy(alpha, &bd, &bz);
        r = axpy(alpha, &bd, &r);
        let rr_next = dot(&r, &r);
        if rr_next.sqrt() < tolerance {
            break;
        }
        let beta = rr_next / rr;
        d = d.iter().zip(&r).map(|(di, ri)| beta * di - ri).collect();
        rr = rr_next;
    }
    Ok(Some((z, bz, false)))
}

impl TrustRegion {
    fn validate(&self) -> Result<()> {
        let positive = [self.initial_radius, self.max_radius];
        if positive.iter().any(|r| !(r.is_finite() && *r > 0.0))
            || self.initial_radius > self.max_radius
        {
            return Err(AlgorithmError::InvalidSetting(
                "the trust-region radii must be positive and finite, the initial one no larger than the maximum"
                    .into(),
            ));
        }
        if !(0.0..0.25).contains(&self.eta) {
            return Err(AlgorithmError::InvalidSetting(
                "the trust-region eta must be in [0, 0.25)".into(),
            ));
        }
        if self.max_iterations == 0 {
            return Err(AlgorithmError::InvalidSetting(
                "max_iterations must be positive".into(),
            ));
        }
        let tolerances = [self.tol_grad, self.tol_rel_obj, self.tol_param];
        if tolerances.iter().any(|t| t.is_nan() || *t < 0.0) {
            return Err(AlgorithmError::InvalidSetting(
                "the trust-region tolerances must be zero or positive".into(),
            ));
        }
        Ok(())
    }

    /// Maximizes the objective from `start`, with Hessian-vector products of `curvature` (the parameterisation of
    /// the objective's fitness, without the Jacobian). `known_start`, if given, is the fitness and gradient at
    /// `start`.
    pub(crate) fn maximize(
        &self,
        objective: &Objective<'_>,
        curvature: &Parameterisation<'_>,
        state: &mut State,
        start: Vec<f64>,
        known_start: Option<(f64, Vec<f64>)>,
    ) -> Result<Minimum> {
        self.validate()?;
        let (fitness, gradient) = match known_start {
            Some(known) => known,
            None => objective.value_and_gradient_flat(state, &start)?,
        };
        if !fitness.is_finite() {
            return Err(AlgorithmError::ImpossibleStart);
        }
        let mut method = self.hessian_method;
        let mut minimum = self.minimize(
            state,
            start,
            -fitness,
            negate(&gradient),
            |state, x| {
                let (value, gradient) = objective.value_and_gradient_flat(state, x)?;
                Ok((-value, negate(&gradient)))
            },
            |state, x, v| Ok(negate(&hvp(curvature, x, v, &mut method, state)?)),
            |state, x| Ok(objective.boundary(state, x)?.is_some()),
        )?;
        minimum.value = -minimum.value;
        Ok(minimum)
    }

    /// Minimizes f from `x`, where f is `f` (finite) and its gradient `g`. `evaluate` returns f and its gradient at
    /// a point, `hvp` the product of f's Hessian at a point with a direction, and `stop` says whether to stop
    /// (without converging) at an accepted point.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn minimize<C>(
        &self,
        context: &mut C,
        mut x: Vec<f64>,
        mut f: f64,
        mut g: Vec<f64>,
        mut evaluate: impl FnMut(&mut C, &[f64]) -> Result<(f64, Vec<f64>)>,
        mut hvp: impl FnMut(&mut C, &[f64], &[f64]) -> Result<Vec<f64>>,
        mut stop: impl FnMut(&mut C, &[f64]) -> Result<bool>,
    ) -> Result<Minimum> {
        let mut products = 0;
        let done = |point, value, converged, iterations, products| Minimum {
            point,
            value,
            converged,
            iterations,
            hessian_vector_products: products,
        };
        if !g.iter().all(|gi| gi.is_finite()) {
            return Ok(done(x, f, false, 0, 0));
        }
        if norm(&g) <= self.tol_grad {
            return Ok(done(x, f, true, 0, 0));
        }
        let mut radius = self.initial_radius;
        for iteration in 1..=self.max_iterations {
            let step = steihaug(&g, radius, |v| {
                products += 1;
                hvp(context, &x, v)
            })?;
            let Some((p, bp, at_edge)) = step else {
                return Ok(done(x, f, false, iteration, products));
            };
            let predicted = -(dot(&g, &p) + 0.5 * dot(&p, &bp));
            let trial: Vec<f64> = x.iter().zip(&p).map(|(a, b)| a + b).collect();
            let (f_trial, g_trial) = evaluate(context, &trial)?;
            let possible = f_trial.is_finite() && g_trial.iter().all(|gi| gi.is_finite());
            let rho = if possible && predicted > 0.0 {
                (f - f_trial) / predicted
            } else {
                f64::NEG_INFINITY
            };
            let length = norm(&p);
            if rho < 0.25 {
                radius = 0.25 * length.min(radius);
            } else if rho > 0.75 && at_edge {
                radius = (2.0 * radius).min(self.max_radius);
            }
            if rho > self.eta {
                let change = (f - f_trial).abs();
                let scale = f.abs().max(f_trial.abs()).max(1.0);
                x = trial;
                f = f_trial;
                g = g_trial;
                if stop(context, &x)? {
                    return Ok(done(x, f, false, iteration, products));
                }
                if norm(&g) <= self.tol_grad
                    || change / scale < self.tol_rel_obj * f64::EPSILON
                    || length < self.tol_param
                {
                    return Ok(done(x, f, true, iteration, products));
                }
            } else if radius < self.tol_param {
                return Ok(done(x, f, true, iteration, products));
            }
        }
        Ok(done(x, f, false, self.max_iterations, products))
    }
}

fn negate(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| -x).collect()
}

/// The Hessian-vector product of `param`'s log probability (without the Jacobian) at `x` along `v` by `method`,
/// resolving [`HessianMethod::Auto`] on the first call: to [`HessianMethod::Exact`], or, if a second-order rule is
/// missing, to [`HessianMethod::FiniteDifference`] for the rest of the run.
fn hvp(
    param: &Parameterisation<'_>,
    x: &[f64],
    v: &[f64],
    method: &mut HessianMethod,
    state: &mut State,
) -> Result<Vec<f64>> {
    if *method == HessianMethod::Auto {
        match param.hvp_with(x, v, false, HessianMethod::Exact, state) {
            Err(GraphError::NoSecondOrderRule(_)) => *method = HessianMethod::FiniteDifference,
            other => {
                *method = HessianMethod::Exact;
                return Ok(other?);
            }
        }
    }
    Ok(param.hvp_with(x, v, false, *method, state)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generalised Rosenbrock function, its gradient and its Hessian times a direction.
    fn rosenbrock(x: &[f64]) -> (f64, Vec<f64>) {
        let n = x.len();
        let mut f = 0.0;
        let mut g = vec![0.0; n];
        for i in 0..n - 1 {
            let a = x[i + 1] - x[i] * x[i];
            let b = 1.0 - x[i];
            f += 100.0 * a * a + b * b;
            g[i] += -400.0 * a * x[i] - 2.0 * b;
            g[i + 1] += 200.0 * a;
        }
        (f, g)
    }

    fn rosenbrock_hvp(x: &[f64], v: &[f64]) -> Vec<f64> {
        let n = x.len();
        let mut hv = vec![0.0; n];
        for i in 0..n - 1 {
            let hii = 1200.0 * x[i] * x[i] - 400.0 * x[i + 1] + 2.0;
            let hij = -400.0 * x[i];
            hv[i] += hii * v[i] + hij * v[i + 1];
            hv[i + 1] += hij * v[i] + 200.0 * v[i + 1];
        }
        hv
    }

    fn minimize_rosenbrock(start: Vec<f64>) -> (Minimum, usize) {
        let (f, g) = rosenbrock(&start);
        let mut evaluations = 0;
        let minimum = TrustRegion::default()
            .minimize(
                &mut (),
                start,
                f,
                g,
                |_, x| {
                    evaluations += 1;
                    Ok(rosenbrock(x))
                },
                |_, x, v| Ok(rosenbrock_hvp(x, v)),
                |_, _| Ok(false),
            )
            .unwrap();
        (minimum, evaluations)
    }

    #[test]
    fn rosenbrock_converges_to_all_ones() {
        let mut ten = [-1.2, 1.0].repeat(5);
        ten[9] = 1.0;
        for start in [vec![-1.2, 1.0], ten, vec![0.0; 10], vec![0.0, 1.0]] {
            let n = start.len();
            let (minimum, evaluations) = minimize_rosenbrock(start.clone());
            assert!(minimum.converged, "from {start:?}");
            assert!(
                minimum.point.iter().all(|xi| (xi - 1.0).abs() < 1e-6),
                "{n} dimensions: {:?}",
                minimum.point
            );
            assert!(evaluations < 200, "{n} dimensions: {evaluations}");
        }
    }

    /// At (0, 1) Rosenbrock's Hessian is diag(−398, 200), indefinite (the full run from there converges, in
    /// `rosenbrock_converges_to_all_ones`). Along a direction of negative curvature, as −g for the model
    /// x·1 − x² + y² at the origin, the step goes to the edge of the trust region.
    #[test]
    fn negative_curvature_steps_to_the_edge_of_the_trust_region() {
        assert_eq!(rosenbrock_hvp(&[0.0, 1.0], &[1.0, 0.0]), vec![-398.0, 0.0]);
        let saddle = |v: &[f64]| Ok(vec![-2.0 * v[0], 2.0 * v[1]]);
        let (p, bp, at_edge) = steihaug(&[1.0, 0.0], 0.5, saddle).unwrap().unwrap();
        assert!(at_edge);
        assert!((p[0] + 0.5).abs() < 1e-15 && p[1] == 0.0, "{p:?}");
        assert_eq!(bp, vec![1.0, 0.0]);
        // a positive definite model whose minimum is inside the region is solved exactly (in n steps)
        let b = |v: &[f64]| Ok(vec![2.0 * v[0] + v[1], v[0] + 3.0 * v[1]]);
        let (p, _, at_edge) = steihaug(&[1.0, -2.0], 10.0, b).unwrap().unwrap();
        assert!(!at_edge);
        // B p = −g: p = (−1, 1)
        assert!(
            (p[0] + 1.0).abs() < 1e-12 && (p[1] - 1.0).abs() < 1e-12,
            "{p:?}"
        );
        // and one outside it is cut at the edge
        let (p, _, at_edge) = steihaug(&[1.0, -2.0], 0.5, b).unwrap().unwrap();
        assert!(at_edge);
        assert!((norm(&p) - 0.5).abs() < 1e-12);
        // a non-finite product ends the step
        assert!(
            steihaug(&[1.0], 1.0, |_| Ok(vec![f64::NAN]))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn the_edge_of_the_region_is_found_from_inside() {
        let tau = to_edge(&[0.3, 0.0], &[1.0, 1.0], 1.0);
        let p = [0.3 + tau, tau];
        assert!(tau > 0.0 && (norm(&p) - 1.0).abs() < 1e-14);
        let tau = to_edge(&[0.3, 0.0], &[-1.0, 0.0], 1.0);
        assert!((tau - 1.3).abs() < 1e-14);
    }
}
