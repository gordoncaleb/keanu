//! Exact Hessian-vector products of the joint log probability (forward-over-reverse), checked against finite
//! differences of the exact gradient.
//!
//! The table-driven test builds one small model per op family, with Gaussian latents set to chosen points and every
//! deterministic output observed through a Gaussian likelihood, so each op's second-order rule (and the first-order
//! rule applied to the tangent channel) is exercised. It checks that the table covers every op of the graph.
//!
//! A second table does the same for every distribution kind: each case makes the value (where it is continuous) and
//! every continuous parameter depend on latents, through simple transforms that keep them valid, and the test checks
//! that the cases cover every kind (and every base of a truncated or censored distribution).

use keanu_distributions::KeanuRng;
use keanu_graph::op::{DistributionKind, Op};
use keanu_graph::transform::{
    Bijector, Circular, CorrelationCholesky, Identity, Interval, LowerBound, Ordered, Simplex,
    UpperBound,
};
use keanu_graph::{
    DoubleVar, GraphBuilder, GraphError, IntVar, Model, NodeId, Parameterisation, Space, State,
    Value,
};
use keanu_tensor::{DoubleTensor, IntTensor};
use std::collections::BTreeSet;
use std::sync::Arc;

// ---------------------------------------------------------------------------------------------------------------
// the finite-difference checker

fn t(shape: &[usize], data: &[f64]) -> DoubleTensor {
    DoubleTensor::new(shape, data.to_vec()).unwrap()
}

fn model_of(build: impl FnOnce(&GraphBuilder)) -> (Model, State) {
    named_model("a model", build)
}

fn named_model(name: &str, build: impl FnOnce(&GraphBuilder)) -> (Model, State) {
    let b = GraphBuilder::new();
    build(&b);
    let (graph, mut state) = b.build().unwrap_or_else(|e| panic!("{name}: {e}"));
    let all: Vec<NodeId> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(3))
        .unwrap();
    let model = Model::new(graph, &state).unwrap();
    (model, state)
}

/// A latent with a standard Gaussian prior, set to `value`.
fn latent<'b>(b: &'b GraphBuilder, value: DoubleTensor) -> DoubleVar<'b> {
    let x = b.gaussian_with_shape(value.shape(), 0.0, 1.0);
    b.set_value(x, value);
    x
}

fn latent_vector<'b>(b: &'b GraphBuilder, values: &[f64]) -> DoubleVar<'b> {
    latent(b, DoubleTensor::vector(values.to_vec()))
}

/// Observes `out` through a Gaussian likelihood at fixed data that differs from it, so the likelihood's cotangent
/// is not zero and the op's second derivatives matter.
fn observe(b: &GraphBuilder, out: DoubleVar<'_>) {
    let shape = out.shape();
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n)
        .map(|i| 0.9 * ((i as f64) * 1.3 + 0.4).sin())
        .collect();
    let y = b.gaussian(out, 1.3);
    b.observe(y, DoubleTensor::new(&shape, data).unwrap());
}

fn gradient(
    param: &Parameterisation<'_>,
    z: &[f64],
    jacobian: bool,
    state: &mut State,
) -> Vec<f64> {
    let (_, g) = param.log_prob_and_gradient(z, jacobian, state).unwrap();
    g.expect("a possible point")
}

/// The Hessian by central differences of the exact gradient: column j is (∇f(z + h eⱼ) − ∇f(z − h eⱼ)) / 2h.
fn fd_hessian(
    param: &Parameterisation<'_>,
    z: &[f64],
    jacobian: bool,
    state: &mut State,
) -> Vec<Vec<f64>> {
    let n = z.len();
    let mut columns = Vec::with_capacity(n);
    for j in 0..n {
        let h = 1e-5 * z[j].abs().max(1.0);
        let mut plus = z.to_vec();
        plus[j] += h;
        let mut minus = z.to_vec();
        minus[j] -= h;
        let gp = gradient(param, &plus, jacobian, state);
        let gm = gradient(param, &minus, jacobian, state);
        columns.push(
            gp.iter()
                .zip(&gm)
                .map(|(p, m)| (p - m) / (2.0 * h))
                .collect(),
        );
    }
    columns
}

/// Checks the exact Hessian against finite differences, and a Hessian-vector product against the Hessian.
fn check_hessian(name: &str, model: &Model, state: &mut State, space: Space, jacobian: bool) {
    let param = Parameterisation::with_space(model, state, space).unwrap();
    let z = param.from_state(state).unwrap();
    let n = z.len();
    let hessian = param
        .hessian(&z, jacobian, state)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(hessian.shape(), &[n, n], "{name}");
    let h = hessian.as_slice();
    let fd = fd_hessian(&param, &z, jacobian, state);
    for (j, column) in fd.iter().enumerate() {
        for (i, &expected) in column.iter().enumerate() {
            let actual = h[i * n + j];
            let scale = expected.abs().max(actual.abs()).max(1.0);
            assert!(
                (actual - expected).abs() <= 2e-6 * scale,
                "{name}: H[{i}][{j}] = {actual}, finite differences give {expected}"
            );
        }
    }
    let v: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.7 + 0.3).sin()).collect();
    let hv = param.hvp(&z, &v, jacobian, state).unwrap();
    for (i, &actual) in hv.iter().enumerate() {
        let expected: f64 = (0..n).map(|j| h[i * n + j] * v[j]).sum();
        assert!(
            (actual - expected).abs() <= 1e-10 * expected.abs().max(1.0),
            "{name}: (Hv)[{i}] = {actual}, H·v = {expected}"
        );
    }
}

/// What a case expects of the Hessian.
enum Expect {
    /// It matches finite differences.
    Matches,
    /// It fails with [`GraphError::NoSecondOrderRule`] whose message contains this text.
    NoRule(&'static str),
}

type Case = (&'static str, fn(&GraphBuilder), Expect);

fn run(case: &Case, covered: &mut BTreeSet<&'static str>) {
    let (name, build, expect) = case;
    let (model, mut state) = named_model(name, build);
    for id in model.graph().ids() {
        covered.insert(model.graph()[id].op().name());
    }
    match expect {
        Expect::Matches => {
            check_hessian(name, &model, &mut state, Space::Constrained, false);
        }
        Expect::NoRule(text) => {
            let param = Parameterisation::with_space(&model, &state, Space::Constrained).unwrap();
            let z = param.from_state(&state).unwrap();
            match param.hessian(&z, false, &mut state) {
                Err(GraphError::NoSecondOrderRule(message)) => assert!(
                    message.contains(text),
                    "{name}: the error {message:?} does not mention {text:?}"
                ),
                other => panic!("{name}: expected a no-second-order-rule error, got {other:?}"),
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// the op table

type UnaryCase = (&'static str, fn(DoubleVar<'_>) -> DoubleVar<'_>, [f64; 3]);

const IN_UNIT: [f64; 3] = [0.3, 0.55, 0.8];
const MIXED: [f64; 3] = [-0.6, 0.35, 1.2];

fn unary_cases() -> Vec<UnaryCase> {
    vec![
        ("abs", |x| x.abs(), MIXED),
        ("neg", |x| -x, MIXED),
        ("reciprocal", |x| x.reciprocal(), MIXED),
        ("exp", |x| x.exp(), MIXED),
        ("exp2", |x| x.exp2(), MIXED),
        ("exp_m1", |x| x.exp_m1(), MIXED),
        ("ln", |x| x.ln(), IN_UNIT),
        ("ln_1p", |x| x.ln_1p(), MIXED),
        ("log2", |x| x.log2(), IN_UNIT),
        ("log10", |x| x.log10(), IN_UNIT),
        ("sqrt", |x| x.sqrt(), IN_UNIT),
        ("sin", |x| x.sin(), MIXED),
        ("cos", |x| x.cos(), MIXED),
        ("tan", |x| x.tan(), MIXED),
        ("asin", |x| x.asin(), [-0.6, 0.35, 0.8]),
        ("acos", |x| x.acos(), [-0.6, 0.35, 0.8]),
        ("atan", |x| x.atan(), MIXED),
        ("sinh", |x| x.sinh(), MIXED),
        ("cosh", |x| x.cosh(), MIXED),
        ("tanh", |x| x.tanh(), MIXED),
        ("asinh", |x| x.asinh(), MIXED),
        ("acosh", |x| x.acosh(), [1.3, 2.0, 3.5]),
        ("atanh", |x| x.atanh(), [-0.6, 0.35, 0.8]),
        ("sigmoid", |x| x.sigmoid(), MIXED),
        ("ln_gamma", |x| x.ln_gamma(), [0.4, 1.7, 6.0]),
        ("digamma", |x| x.digamma(), [0.4, 1.7, 6.0]),
        ("trigamma", |x| x.trigamma(), [0.4, 1.7, 12.5]),
        ("erf", |x| x.erf(), MIXED),
        ("erfc", |x| x.erfc(), MIXED),
        ("erf_inv", |x| x.erf_inv(), [-0.6, 0.35, 0.8]),
        ("log_ndtr", |x| x.log_ndtr(), [-4.0, -0.3, 1.5]),
        ("ndtri", |x| x.ndtri(), IN_UNIT),
        ("softplus", |x| x.softplus(), MIXED),
        ("logit", |x| x.logit(), IN_UNIT),
        ("log1mexp", |x| x.log1mexp(), [-2.0, -0.7, -0.1]),
        // not differentiable: they contribute nothing, so only the smooth factor's derivatives remain
        ("floor", |x| x.floor() * x.sin(), [0.3, 1.55, -0.8]),
        ("ceil", |x| x.ceil() * x.sin(), [0.3, 1.55, -0.8]),
        ("round", |x| x.round() * x.sin(), [0.3, 1.55, -0.8]),
        ("sign", |x| x.sign() * x.exp(), MIXED),
    ]
}

type BinaryCase = (
    &'static str,
    for<'b> fn(DoubleVar<'b>, DoubleVar<'b>) -> DoubleVar<'b>,
    [f64; 3],
    [f64; 3],
);

fn binary_cases() -> Vec<BinaryCase> {
    vec![
        ("add", |l, r| (l + r).sin(), MIXED, IN_UNIT),
        ("sub", |l, r| (l - r).sin(), MIXED, IN_UNIT),
        ("mul", |l, r| l * r, MIXED, IN_UNIT),
        ("div", |l, r| l / r, MIXED, IN_UNIT),
        ("pow", |l, r| l.pow(r), IN_UNIT, MIXED),
        ("max", |l, r| l.maximum(r) * l, MIXED, IN_UNIT),
        ("min", |l, r| l.minimum(r) * r, MIXED, IN_UNIT),
        ("atan2", |l, r| l.atan2(r), MIXED, IN_UNIT),
        ("log_add_exp", |l, r| l.log_add_exp(r), MIXED, IN_UNIT),
        ("log_add_exp2", |l, r| l.log_add_exp2(r), MIXED, IN_UNIT),
        ("safe_ln_times", |l, r| l.safe_ln_times(r), IN_UNIT, MIXED),
        ("log_beta", |l, r| l.log_beta(r), [0.4, 1.7, 3.0], IN_UNIT),
    ]
}

fn general_cases() -> Vec<Case> {
    vec![
        (
            "compare, mask, logic and predicates",
            |b| {
                let x = latent_vector(b, &[0.4, -0.7, 1.1]);
                let y = latent_vector(b, &[0.2, 0.5, -0.3]);
                let both = x.gt(y).and(!x.approx_eq(y, 1e-3));
                let any_but_not_all = both.any().and(!both.all());
                let selected = b.where_(both, x * y.sin(), x.exp() * y);
                let masked = selected * (x.gt_mask(0.0) + 1.0);
                let nan_free = b.where_(x.is_nan(), x, y * y);
                observe(b, masked + nan_free);
                observe(b, b.where_(any_but_not_all, x * x, y));
            },
            Expect::Matches,
        ),
        (
            "rem, casts, arg reductions and multiplexers",
            |b| {
                let x = latent_vector(b, &[0.4, -0.7, 1.1]);
                let remainder: IntVar<'_> = b.int_constant(IntTensor::vector(vec![5, 7, 9])) % 4;
                observe(b, remainder.to_double() * x.sin() + x.to_int().to_double());
                observe(b, b.multiplexer(x.argmax(), &[x * x, x.sin(), x.exp()]));
            },
            Expect::Matches,
        ),
        (
            "reductions",
            |b| {
                let x = latent(b, t(&[2, 3], &[0.4, -0.7, 1.1, 0.9, 1.6, -0.2]));
                observe(b, x.exp().sum());
                observe(b, x.sum_axes(&[0]) * x.sum_axes(&[0]));
                observe(b, x.product());
                observe(b, x.product_axes(&[1]));
                observe(b, x.mean_axes(&[0]).sin());
                observe(b, x.mean());
                observe(b, x.max() * x.min());
                observe(b, x.std_dev());
                observe(b, x.log_sum_exp());
                observe(b, x.log_sum_exp_axes(&[1]));
            },
            Expect::Matches,
        ),
        (
            "products with zero elements",
            |b| {
                let x = latent(b, t(&[2, 3], &[0.0, -0.7, 1.1, 0.0, 0.0, -0.2]));
                observe(b, x.product());
                observe(b, x.product_axes(&[1]));
                observe(b, x.cumprod(1));
            },
            Expect::Matches,
        ),
        (
            "standardize, scans and softmax",
            |b| {
                let x = latent_vector(b, &[0.4, -0.7, 1.1, 0.25]);
                observe(b, x.standardize());
                observe(b, x.cumsum(0).sin());
                observe(b, x.cumprod(0));
                observe(b, x.softmax(0));
                observe(b, x.log_softmax(-1));
                let m = latent(b, t(&[2, 3], &[0.4, -0.7, 1.1, 0.9, 1.6, -0.2]));
                observe(b, m.cumprod(1));
                observe(b, m.softmax(0));
                observe(b, m.log_softmax(1));
            },
            Expect::Matches,
        ),
        (
            "scan",
            |b| {
                let init = latent_vector(b, &[0.3, -0.4]);
                let xs = latent(b, t(&[3, 2], &[0.5, -0.1, 0.2, 0.9, -0.6, 0.4]));
                let w = latent(b, t(&[2, 2], &[0.8, -0.3, 0.2, 0.6]));
                observe(
                    b,
                    b.scan(init, &[xs], &[w], |_, h, x, c| {
                        (c[0].matmul(h.reshape(&[2, 1])).reshape(&[2]) + x[0]).tanh() * h.exp()
                    }),
                );
            },
            Expect::Matches,
        ),
        (
            "shape and indexing ops",
            |b| {
                let x = latent(b, t(&[2, 3], &[0.4, -0.7, 1.1, 0.9, 1.6, -0.2]));
                let e = x.exp();
                observe(b, e.reshape(&[3, 2]));
                observe(b, e.permute(&[1, 0]) * x.transpose());
                observe(b, e.broadcast_to(&[2, 2, 3]));
                observe(b, e.slice("0:1, 1:"));
                observe(b, b.concat(0, &[e, x * x]));
                observe(b, e.take(&[1, 2]) * x.take(&[0, 1]));
                let indices = IntTensor::vector(vec![2, 0, 2]);
                observe(b, e.gather(1, indices.clone()));
                observe(b, e.scatter_add(1, indices, 4));
                observe(b, b.stack(0, &[e, x.sin()]));
            },
            Expect::Matches,
        ),
        (
            "triangles and diagonals",
            |b| {
                let m = latent(
                    b,
                    t(&[3, 3], &[0.4, -0.7, 1.1, 0.9, 1.6, -0.2, 0.3, 0.8, -1.2]),
                );
                let e = m.exp();
                observe(b, e.diag_part());
                observe(b, e.diag_part().diag());
                observe(b, e.tril(0));
                observe(b, e.triu(1));
                let v = latent_vector(b, &[0.4, -0.7, 1.1, 0.9, 1.6, -0.2]);
                observe(b, v.exp().fill_triangular(true, true));
                observe(b, v.sin().fill_triangular(false, true));
                observe(b, e.triangle_part(false));
                observe(b, e.triangle_part(true));
            },
            Expect::Matches,
        ),
        (
            "matrix products and contractions",
            |b| {
                let a = latent(b, t(&[2, 3], &[0.4, -0.7, 1.1, 0.9, 1.6, -0.2]));
                let m = latent(b, t(&[3, 2], &[0.3, 0.8, -1.2, 0.5, 0.1, -0.4]));
                observe(b, a.matmul(m));
                observe(b, a.matmul_transposed(a, false, true));
                observe(b, a.matmul_transposed(a, true, false));
                observe(b, m.matmul_transposed(a, true, true));
                let batch = latent(
                    b,
                    t(
                        &[2, 2, 3],
                        &[
                            0.2, 0.1, -0.3, 0.7, 0.5, 0.6, -0.8, 0.9, 0.4, 0.2, -0.1, 0.3,
                        ],
                    ),
                );
                observe(b, batch.matmul(m));
                observe(b, a.tensor_multiply(m, &[1], &[0]));
                observe(b, a.tensor_multiply(m, &[0, 1], &[1, 0]));
                observe(b, batch.tensor_multiply(a, &[2], &[1]));
            },
            Expect::Matches,
        ),
        (
            "inverses, determinants and Cholesky factors",
            |b| {
                let m = latent(
                    b,
                    t(&[3, 3], &[1.4, -0.7, 0.3, 0.9, 1.6, -0.2, 0.3, 0.8, 1.2]),
                );
                let eye = b.constant(DoubleTensor::eye(3).unwrap());
                let spd = m.matmul_transposed(m, false, true) + eye * 0.5;
                observe(b, spd.inverse());
                observe(b, m.inverse());
                observe(b, m.determinant());
                observe(b, spd.determinant());
                observe(b, spd.cholesky());
                observe(b, spd.cholesky().cholesky_inverse());
                observe(b, m.cholesky_inverse());
                observe(b, m.log_abs_determinant());
                observe(b, (-m).log_abs_determinant() * m.determinant_sign());
            },
            Expect::Matches,
        ),
        (
            "solves",
            |b| {
                let m = latent(
                    b,
                    t(&[3, 3], &[1.4, -0.7, 0.3, 0.9, 1.6, -0.2, 0.3, 0.8, 1.2]),
                );
                let rhs = latent(b, t(&[3, 2], &[0.3, 0.8, -1.2, 0.5, 0.1, -0.4]));
                let v = latent_vector(b, &[0.6, -0.1, 0.9]);
                for (lower, transpose) in
                    [(true, false), (true, true), (false, false), (false, true)]
                {
                    observe(b, m.solve_triangular(rhs, lower, transpose));
                    observe(b, m.solve_triangular(v, lower, transpose));
                }
                observe(b, m.solve(rhs));
                observe(b, m.solve(v));
                let batch = latent(
                    b,
                    t(
                        &[2, 3, 3],
                        &[
                            1.2, 0.1, -0.3, 0.7, 1.5, 0.6, -0.8, 0.9, 1.4, 1.1, -0.1, 0.3, 0.2,
                            0.9, -0.5, 0.4, 0.6, 1.3,
                        ],
                    ),
                );
                observe(b, batch.solve(rhs));
                observe(b, batch.solve_triangular(v, true, false));
            },
            Expect::Matches,
        ),
        (
            "symmetric eigendecompositions",
            |b| {
                let m = latent(
                    b,
                    t(&[3, 3], &[1.4, -0.7, 0.3, 0.9, 1.6, -0.2, 0.3, 0.8, 1.2]),
                );
                let s = m.matmul_transposed(m, false, true);
                observe(b, s.eigvalsh());
                observe(b, s.eigh_vectors());
                // only the lower triangle is read
                observe(b, m.eigvalsh());
                observe(b, m.eigh_vectors());
            },
            Expect::Matches,
        ),
        (
            "selections, masks, proxies and assertions",
            |b| {
                let x = latent_vector(b, &[0.5, -0.5, 2.0]);
                let s = latent(b, DoubleTensor::scalar(0.7));
                let positive = x.sum().gt(0.0);
                observe(b, b.cpt(&[positive.id()], &[(vec![1], x * x * s)], x.sin()));
                let mask = b.constant(DoubleTensor::vector(vec![1.0, 0.0, 1.0]));
                observe(b, x.exp().set_with_mask(mask, s * s));
                // NaN where x < 0, replaced with 0
                observe(b, x.ln().replace_nan(0.0) * s);
                let square = x * x;
                let proxy = b.double_proxy(&[3]);
                b.connect(proxy, square);
                observe(b, proxy.sin());
                x.gt(-10.0).all().assert(Some("x is not too small"));
            },
            Expect::Matches,
        ),
        (
            "a lambda without a second-order rule",
            |b| {
                let x = latent_vector(b, &[0.5, -0.5, 2.0]);
                let id = b.lambda_with_jacobian(
                    "square",
                    &[x.id()],
                    &[3],
                    |v| Ok(Value::F64(v[0].as_f64().unwrap().map(|x| x * x))),
                    |v| {
                        let x = v[0].as_f64().unwrap();
                        Ok(vec![Some(x.map(|x| 2.0 * x).diag().unwrap())])
                    },
                );
                b.set_label_of(id, "my_lambda").unwrap();
                observe(b, b.double_var(id));
            },
            Expect::NoRule("my_lambda"),
        ),
        (
            "a Gaussian in its value, mu and sigma",
            |b| {
                let mu = latent(b, DoubleTensor::scalar(0.3));
                let s = latent(b, DoubleTensor::scalar(0.2));
                let sigma = s.exp();
                let x = b.gaussian(mu, sigma);
                b.set_value(x, DoubleTensor::scalar(0.9));
                let y = b.gaussian(x * mu, sigma + 0.5);
                b.observe(y, DoubleTensor::scalar(0.4));
                // a scalar mu broadcast against a vector sigma
                let ones = b.constant(DoubleTensor::vector(vec![1.0, 2.0, 0.5]));
                let w = b.gaussian(mu * s, sigma * ones);
                b.observe(w, DoubleTensor::vector(vec![0.1, -0.4, 0.8]));
            },
            Expect::Matches,
        ),
        (
            "a multivariate Gaussian in its value, mu and covariance",
            |b| {
                let mu = latent_vector(b, &[0.3, -0.2, 0.5]);
                let l = latent(
                    b,
                    t(&[3, 3], &[1.1, 0.0, 0.0, 0.4, 0.9, 0.0, -0.3, 0.2, 1.3]),
                );
                let eye = b.constant(DoubleTensor::eye(3).unwrap());
                let cov = l.matmul_transposed(l, false, true) + eye * 0.5;
                let x = b.multivariate_gaussian(mu, cov);
                b.set_value(x, DoubleTensor::vector(vec![0.6, 0.1, -0.4]));
                let y = b.multivariate_gaussian(x * mu, cov);
                b.observe(y, DoubleTensor::vector(vec![0.2, -0.5, 0.9]));
                // a batch of two observations sharing the covariance
                let offset = b.constant(t(&[2, 3], &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]));
                let z = b.multivariate_gaussian(mu + offset, cov * 2.0);
                b.observe(z, t(&[2, 3], &[0.2, -0.5, 0.9, 1.0, 0.3, -0.1]));
            },
            Expect::Matches,
        ),
        (
            "a custom distribution without a second-order rule",
            |b| {
                let mu = latent(b, DoubleTensor::scalar(0.3));
                let y: DoubleVar<'_> = b.custom(UnitGaussian, &[], &[mu.id()]);
                b.observe(y, DoubleTensor::scalar(0.4));
            },
            Expect::NoRule("custom"),
        ),
    ]
}

#[test]
fn every_op_matches_finite_differences_of_the_gradient() {
    let mut covered = BTreeSet::new();
    for (name, f, values) in unary_cases() {
        let case_build = move |b: &GraphBuilder| {
            let x = latent_vector(b, &values);
            observe(b, f(x));
        };
        let (model, mut state) = named_model(name, case_build);
        for id in model.graph().ids() {
            covered.insert(model.graph()[id].op().name());
        }
        check_hessian(name, &model, &mut state, Space::Constrained, false);
    }
    for (name, f, left, right) in binary_cases() {
        let (model, mut state) = named_model(name, |b| {
            let l = latent_vector(b, &left);
            let r = latent_vector(b, &right);
            observe(b, f(l, r));
            // broadcasting: a [2, 3] left operand against the [3] right one, and a scalar right operand
            let wide = latent(
                b,
                t(
                    &[2, 3],
                    &[left[0], left[1], left[2], left[2], left[0], left[1]],
                ),
            );
            observe(b, f(wide, r));
            let scalar = latent(b, DoubleTensor::scalar(right[1]));
            observe(b, f(l, scalar));
        });
        check_hessian(name, &model, &mut state, Space::Constrained, false);
    }
    for case in &general_cases() {
        run(case, &mut covered);
    }
    let missing: Vec<&str> = Op::NAMES
        .iter()
        .copied()
        .filter(|op| !covered.contains(op))
        .collect();
    assert!(missing.is_empty(), "ops without a case: {missing:?}");
}

// ---------------------------------------------------------------------------------------------------------------
// properties of the Hessian

/// Gaussians, a multivariate Gaussian and nonlinear deterministic links.
fn mixed_model() -> (Model, State) {
    model_of(|b| {
        let mu = latent(b, DoubleTensor::scalar(0.3));
        let s = latent(b, DoubleTensor::scalar(-0.4));
        let w = latent_vector(b, &[0.5, -1.0, 0.25]);
        let sigma = s.exp();
        let y = b.gaussian(mu * w.sum(), sigma);
        b.observe(y, DoubleTensor::scalar(0.7));
        let rates = (w * mu).softmax(0);
        let z = b.gaussian(rates.ln(), sigma + 0.1);
        b.observe(z, DoubleTensor::vector(vec![-1.0, -0.5, -2.0]));
        let eye = b.constant(DoubleTensor::eye(3).unwrap());
        let cov = (w.diag() * sigma).matmul_transposed(w.diag(), false, true) + eye;
        let u = b.multivariate_gaussian(w * mu, cov);
        b.observe(u, DoubleTensor::vector(vec![0.4, 0.5, 1.1]));
    })
}

#[test]
fn the_hessian_of_a_mixed_model_is_symmetric() {
    let (model, mut state) = mixed_model();
    let param = Parameterisation::new(&model, &state).unwrap();
    let z = param.from_state(&state).unwrap();
    let n = z.len();
    let hessian = param.hessian(&z, true, &mut state).unwrap();
    let h = hessian.as_slice();
    for i in 0..n {
        for j in 0..i {
            let (a, b) = (h[i * n + j], h[j * n + i]);
            assert!(
                (a - b).abs() <= 1e-8 * a.abs().max(1.0),
                "H[{i}][{j}] = {a} but H[{j}][{i}] = {b}"
            );
        }
    }
    check_hessian(
        "mixed model",
        &model,
        &mut state,
        Space::Unconstrained,
        true,
    );
}

#[test]
fn the_hessian_of_a_multivariate_gaussian_log_density_is_minus_the_precision() {
    let covariance = t(&[3, 3], &[2.0, 0.6, -0.4, 0.6, 1.5, 0.3, -0.4, 0.3, 1.0]);
    let (model, mut state) = model_of(|b| {
        let x = b.multivariate_gaussian(
            DoubleTensor::vector(vec![0.1, -0.2, 0.3]),
            covariance.clone(),
        );
        b.set_value(x, DoubleTensor::vector(vec![0.7, 0.2, -1.1]));
    });
    let param = Parameterisation::new(&model, &state).unwrap();
    let z = param.from_state(&state).unwrap();
    let hessian = param.hessian(&z, true, &mut state).unwrap();
    let precision = covariance.inverse().unwrap();
    for (h, p) in hessian.as_slice().iter().zip(precision.as_slice()) {
        assert!((h + p).abs() < 1e-12, "{h} vs {}", -p);
    }
}

#[test]
fn a_hessian_vector_product_checks_its_arguments() {
    let (model, mut state) = mixed_model();
    let param = Parameterisation::new(&model, &state).unwrap();
    let z = param.from_state(&state).unwrap();
    assert!(matches!(
        param.hvp(&z, &[1.0], false, &mut state),
        Err(GraphError::InvalidValue(_))
    ));
    assert!(matches!(
        param.hvp(&[1.0], &z, false, &mut state),
        Err(GraphError::InvalidValue(_))
    ));
    assert!(matches!(
        param.hessian(&[1.0], false, &mut state),
        Err(GraphError::InvalidValue(_))
    ));
}

#[test]
fn a_model_without_continuous_latents_has_an_empty_hessian() {
    let (model, mut state) = model_of(|b| {
        let y = b.gaussian(0.0, 1.0);
        b.observe(y, DoubleTensor::scalar(0.4));
    });
    let param = Parameterisation::new(&model, &state).unwrap();
    assert_eq!(
        param.hvp(&[], &[], true, &mut state).unwrap(),
        Vec::<f64>::new()
    );
    assert_eq!(
        param.hessian(&[], true, &mut state).unwrap().shape(),
        &[0, 0]
    );
}

#[test]
fn a_distribution_needs_a_rule_only_where_it_depends_on_the_latents() {
    // a custom distribution whose parameter depends on a continuous latent: its log density is a function of the
    // latent, so it needs a rule
    let (model, mut state) = model_of(|b| {
        let x = latent(b, DoubleTensor::scalar(0.3));
        let y: DoubleVar<'_> = b.custom(UnitGaussian, &[], &[x.exp().id()]);
        b.observe(y, DoubleTensor::scalar(0.4));
    });
    let param = Parameterisation::new(&model, &state).unwrap();
    let z = param.from_state(&state).unwrap();
    match param.hessian(&z, true, &mut state) {
        Err(GraphError::NoSecondOrderRule(message)) => assert!(message.contains("custom")),
        other => panic!("expected a no-second-order-rule error, got {other:?}"),
    }
    // the same distribution with a constant parameter doesn't depend on the continuous latent, so needs no rule
    let (model, mut state) = model_of(|b| {
        let x = latent(b, DoubleTensor::scalar(0.3));
        let y: DoubleVar<'_> = b.custom(UnitGaussian, &[], &[b.scalar(2.0).id()]);
        b.observe(y, DoubleTensor::scalar(0.4));
        let w = b.gaussian(x, 1.0);
        b.observe(w, DoubleTensor::scalar(0.1));
    });
    check_hessian(
        "constant custom distribution",
        &model,
        &mut state,
        Space::Unconstrained,
        true,
    );
}

// ---------------------------------------------------------------------------------------------------------------
// bijector second derivatives

fn bijectors() -> Vec<(Box<dyn Bijector>, Vec<f64>)> {
    vec![
        (Box::new(Identity::new(&[2])), vec![0.3, -0.8]),
        (
            Box::new(LowerBound::new(&DoubleTensor::scalar(1.0), &[2]).unwrap()),
            vec![0.3, -0.8],
        ),
        (
            Box::new(UpperBound::new(&DoubleTensor::scalar(1.0), &[2]).unwrap()),
            vec![0.3, -0.8],
        ),
        (
            Box::new(
                Interval::new(
                    &DoubleTensor::scalar(-1.0),
                    &DoubleTensor::scalar(3.0),
                    &[3],
                )
                .unwrap(),
            ),
            vec![0.3, -0.8, 2.5],
        ),
        // element-wise mixed bound kinds
        (
            Box::new(
                LowerBound::new(&DoubleTensor::vector(vec![1.0, f64::NEG_INFINITY]), &[2]).unwrap(),
            ),
            vec![0.3, -0.8],
        ),
        (
            Box::new(
                UpperBound::new(&DoubleTensor::vector(vec![f64::INFINITY, 1.0]), &[2]).unwrap(),
            ),
            vec![0.3, -0.8],
        ),
        (
            Box::new(
                Interval::new(
                    &DoubleTensor::vector(vec![f64::NEG_INFINITY, 0.0, f64::NEG_INFINITY, -1.0]),
                    &DoubleTensor::vector(vec![f64::INFINITY, f64::INFINITY, 2.0, 3.0]),
                    &[4],
                )
                .unwrap(),
            ),
            vec![0.3, -0.8, 2.5, -1.1],
        ),
        (
            Box::new(Ordered::new(&[2, 3]).unwrap()),
            vec![0.3, -0.8, 0.1, 1.2, 0.4, -0.5],
        ),
        (Box::new(Simplex::new(&[4]).unwrap()), vec![0.3, -0.8, 0.6]),
        (
            Box::new(CorrelationCholesky::new(&[3, 3]).unwrap()),
            vec![0.3, -0.8, 0.6],
        ),
        (Box::new(Circular::new(&[2])), vec![0.3, -0.8, 1.1, 0.4]),
    ]
}

#[test]
fn bijector_second_derivatives_match_finite_differences() {
    for (bijector, y) in bijectors() {
        let n = y.len();
        let tangent: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.9 + 0.2).cos()).collect();
        let shape = bijector.constrained_shape().to_vec();
        let len: usize = shape.iter().product();
        let cotangent = DoubleTensor::new(
            &shape,
            (0..len).map(|i| ((i as f64) * 0.6 + 0.5).sin()).collect(),
        )
        .unwrap();
        let h = 1e-6;
        let shifted =
            |s: f64| -> Vec<f64> { y.iter().zip(&tangent).map(|(y, t)| y + s * t).collect() };

        let exact = bijector.forward_vjp_jvp(&y, &tangent, &cotangent).unwrap();
        let plus = bijector.forward_vjp(&shifted(h), &cotangent).unwrap();
        let minus = bijector.forward_vjp(&shifted(-h), &cotangent).unwrap();
        for i in 0..n {
            let fd = (plus[i] - minus[i]) / (2.0 * h);
            assert!(
                (exact[i] - fd).abs() <= 1e-7 * fd.abs().max(1.0),
                "{bijector:?}: d vjp[{i}] = {} vs {fd}",
                exact[i]
            );
        }

        let exact = bijector.log_det_jacobian_hvp(&y, &tangent).unwrap();
        let plus = bijector.log_det_jacobian_gradient(&shifted(h)).unwrap();
        let minus = bijector.log_det_jacobian_gradient(&shifted(-h)).unwrap();
        for i in 0..n {
            let fd = (plus[i] - minus[i]) / (2.0 * h);
            assert!(
                (exact[i] - fd).abs() <= 1e-7 * fd.abs().max(1.0),
                "{bijector:?}: log |J| Hv[{i}] = {} vs {fd}",
                exact[i]
            );
        }
    }
}

#[test]
fn the_log_jacobian_hessian_is_zero_for_a_lower_bound_and_matches_differences_for_an_interval() {
    let lower = LowerBound::new(&DoubleTensor::scalar(0.5), &[3]).unwrap();
    let y = [0.3, -2.0, 4.0];
    for k in 0..3 {
        let mut e = [0.0; 3];
        e[k] = 1.0;
        assert_eq!(lower.log_det_jacobian_hvp(&y, &e).unwrap(), vec![0.0; 3]);
    }

    let interval = Interval::new(
        &DoubleTensor::scalar(-1.0),
        &DoubleTensor::scalar(2.0),
        &[3],
    )
    .unwrap();
    let h = 1e-5;
    for k in 0..3 {
        let mut e = [0.0; 3];
        e[k] = 1.0;
        let column = interval.log_det_jacobian_hvp(&y, &e).unwrap();
        for (i, &c) in column.iter().enumerate() {
            let at = |s: f64| {
                let mut p = y;
                p[k] += s;
                interval.log_det_jacobian_gradient(&p).unwrap()[i]
            };
            let fd = (at(h) - at(-h)) / (2.0 * h);
            assert!((c - fd).abs() < 1e-8, "H[{i}][{k}] = {c} vs {fd}");
            if i != k {
                assert_eq!(c, 0.0);
            }
        }
    }
}

#[test]
fn bijector_second_derivatives_check_their_arguments() {
    let interval =
        Interval::new(&DoubleTensor::scalar(0.0), &DoubleTensor::scalar(1.0), &[2]).unwrap();
    assert!(matches!(
        interval.log_det_jacobian_hvp(&[0.1], &[0.1, 0.2]),
        Err(GraphError::Transform(_))
    ));
    assert!(matches!(
        interval.forward_vjp_jvp(&[0.1, 0.2], &[0.1], &DoubleTensor::vector(vec![1.0, 2.0])),
        Err(GraphError::Transform(_))
    ));
    assert!(matches!(
        interval.forward_vjp_jvp(&[0.1, 0.2], &[0.1, 0.2], &DoubleTensor::scalar(1.0)),
        Err(GraphError::Transform(_))
    ));
}

/// A bijector that implements only the first-order methods.
#[derive(Debug)]
struct FirstOrderOnly(Identity);

impl Bijector for FirstOrderOnly {
    fn constrained_shape(&self) -> &[usize] {
        self.0.constrained_shape()
    }
    fn unconstrained_len(&self) -> usize {
        self.0.unconstrained_len()
    }
    fn forward(&self, y: &[f64]) -> keanu_graph::Result<DoubleTensor> {
        self.0.forward(y)
    }
    fn inverse(&self, x: &DoubleTensor) -> keanu_graph::Result<Vec<f64>> {
        self.0.inverse(x)
    }
    fn log_det_jacobian(&self, y: &[f64]) -> keanu_graph::Result<f64> {
        self.0.log_det_jacobian(y)
    }
    fn log_det_jacobian_gradient(&self, y: &[f64]) -> keanu_graph::Result<Vec<f64>> {
        self.0.log_det_jacobian_gradient(y)
    }
    fn forward_jvp(&self, y: &[f64], tangent: &[f64]) -> keanu_graph::Result<DoubleTensor> {
        self.0.forward_jvp(y, tangent)
    }
    fn forward_vjp(&self, y: &[f64], cotangent: &DoubleTensor) -> keanu_graph::Result<Vec<f64>> {
        self.0.forward_vjp(y, cotangent)
    }
}

#[test]
fn a_bijector_without_second_order_methods_reports_no_rule() {
    let b = FirstOrderOnly(Identity::new(&[1]));
    assert!(matches!(
        b.log_det_jacobian_hvp(&[0.0], &[1.0]),
        Err(GraphError::NoSecondOrderRule(m)) if m.contains("FirstOrderOnly")
    ));
    assert!(matches!(
        b.forward_vjp_jvp(&[0.0], &[1.0], &DoubleTensor::vector(vec![1.0])),
        Err(GraphError::NoSecondOrderRule(_))
    ));
}

// ---------------------------------------------------------------------------------------------------------------
// the distribution table

/// A Gaussian with unit scale, written as a custom distribution (which has no second-order rule).
#[derive(Debug)]
struct UnitGaussian;

impl keanu_graph::CustomDistribution for UnitGaussian {
    fn log_prob(&self, value: &Value, parameters: &[&Value]) -> Result<f64, String> {
        let (x, mu) = (
            value.as_f64().unwrap().as_slice()[0],
            parameters[0].as_f64().unwrap().as_slice()[0],
        );
        Ok(-0.5 * (x - mu) * (x - mu))
    }

    fn log_prob_gradient(
        &self,
        value: &Value,
        parameters: &[&Value],
    ) -> Result<Option<keanu_graph::CustomGradient>, String> {
        let (x, mu) = (
            value.as_f64().unwrap().as_slice()[0],
            parameters[0].as_f64().unwrap().as_slice()[0],
        );
        Ok(Some(keanu_graph::CustomGradient {
            value: Some(DoubleTensor::scalar(mu - x)),
            parameters: vec![Some(DoubleTensor::scalar(x - mu))],
        }))
    }
}

fn scalar_latent(b: &GraphBuilder, value: f64) -> DoubleVar<'_> {
    latent(b, DoubleTensor::scalar(value))
}

/// A positive scalar that depends on a latent: exp of a latent set to ln `value`.
fn positive(b: &GraphBuilder, value: f64) -> DoubleVar<'_> {
    scalar_latent(b, value.ln()).exp()
}

fn positive_vector<'b>(b: &'b GraphBuilder, values: &[f64]) -> DoubleVar<'b> {
    let logs: Vec<f64> = values.iter().map(|v| v.ln()).collect();
    latent_vector(b, &logs).exp()
}

/// Probabilities that depend on latents: the sigmoid of latents set to the logits of `values`.
fn probabilities<'b>(b: &'b GraphBuilder, values: &[f64]) -> DoubleVar<'b> {
    let logits: Vec<f64> = values.iter().map(|p| (p / (1.0 - p)).ln()).collect();
    latent_vector(b, &logits).sigmoid()
}

fn vector(values: &[f64]) -> DoubleTensor {
    DoubleTensor::vector(values.to_vec())
}

fn ints(values: &[i64]) -> IntTensor {
    IntTensor::vector(values.to_vec())
}

/// Bounds that depend on latents: `lower` and `lower + width`.
fn bounds(b: &GraphBuilder, lower: f64, width: f64) -> (DoubleVar<'_>, DoubleVar<'_>) {
    let l = scalar_latent(b, lower);
    (l, l + positive(b, width))
}

/// Positive bounds that depend on latents: `lower` and `lower + width`.
fn positive_bounds(b: &GraphBuilder, lower: f64, width: f64) -> (DoubleVar<'_>, DoubleVar<'_>) {
    let l = positive(b, lower);
    (l, l + positive(b, width))
}

/// A distribution case: a model, the space its Hessian is checked in, and what is expected.
struct DistributionCase {
    name: &'static str,
    build: fn(&GraphBuilder),
    space: Space,
    expect: Expect,
}

fn case(name: &'static str, build: fn(&GraphBuilder)) -> DistributionCase {
    DistributionCase {
        name,
        build,
        space: Space::Constrained,
        expect: Expect::Matches,
    }
}

/// A case checked in unconstrained space (with the Jacobian), for values whose support a finite difference step in
/// constrained space would leave.
fn unconstrained(name: &'static str, build: fn(&GraphBuilder)) -> DistributionCase {
    DistributionCase {
        space: Space::Unconstrained,
        ..case(name, build)
    }
}

/// A truncated location-scale base whose parameters and bounds depend on latents: a latent value between the
/// bounds, an observation, and an observation with an infinite upper bound.
fn truncated_location_scale(b: &GraphBuilder, base: DistributionKind) {
    let location = scalar_latent(b, 0.2);
    let scale = positive(b, 0.8);
    let (lower, upper) = bounds(b, -0.4, 2.0);
    let params = [location.id(), scale.id()];
    let x: DoubleVar<'_> = b.truncated(base.clone(), &params, lower, upper);
    b.set_value(x, DoubleTensor::scalar(0.5));
    let shifted = [(location * 0.5).id(), (scale + 0.3).id()];
    let y: DoubleVar<'_> = b.truncated(base.clone(), &shifted, lower, upper);
    b.observe(y, DoubleTensor::scalar(-0.2));
    let z: DoubleVar<'_> = b.truncated(base, &params, lower, f64::INFINITY);
    b.observe(z, DoubleTensor::scalar(2.5));
}

/// A censored location-scale base: observations at the lower bound, inside and at the upper bound (fixed bounds,
/// so that the values stay at them), and a latent value inside bounds that depend on latents.
fn censored_location_scale(b: &GraphBuilder, base: DistributionKind) {
    let location = scalar_latent(b, 0.2);
    let scale = positive(b, 0.8);
    let params = [location.id(), scale.id()];
    let lower = b.constant(vector(&[-0.4, -0.4, -0.4]));
    let y: DoubleVar<'_> = b.censored(base.clone(), &params, lower, 1.6);
    b.observe(y, vector(&[-0.4, 0.3, 1.6]));
    let (l, u) = bounds(b, -0.4, 2.0);
    let x: DoubleVar<'_> = b.censored(base, &params, l, u);
    b.set_value(x, DoubleTensor::scalar(0.5));
}

/// A truncated positive base with two positive parameters.
fn truncated_positive(b: &GraphBuilder, base: DistributionKind) {
    let params = [positive(b, 1.6).id(), positive(b, 1.2).id()];
    let (lower, upper) = positive_bounds(b, 0.5, 2.0);
    let x: DoubleVar<'_> = b.truncated(base.clone(), &params, lower, upper);
    b.set_value(x, DoubleTensor::scalar(1.2));
    let y: DoubleVar<'_> = b.truncated(base.clone(), &params, lower, upper);
    b.observe(y, DoubleTensor::scalar(2.1));
    let z: DoubleVar<'_> = b.truncated(base, &params, lower, f64::INFINITY);
    b.observe(z, DoubleTensor::scalar(3.0));
}

/// A censored positive base with two positive parameters.
fn censored_positive(b: &GraphBuilder, base: DistributionKind) {
    let params = [positive(b, 1.6).id(), positive(b, 1.2).id()];
    let lower = b.constant(vector(&[0.5, 0.5, 0.5]));
    let y: DoubleVar<'_> = b.censored(base.clone(), &params, lower, 2.5);
    b.observe(y, vector(&[0.5, 1.3, 2.5]));
    let (l, u) = positive_bounds(b, 0.5, 2.0);
    let x: DoubleVar<'_> = b.censored(base, &params, l, u);
    b.set_value(x, DoubleTensor::scalar(1.2));
}

fn distribution_cases() -> Vec<DistributionCase> {
    use DistributionKind as K;
    vec![
        case("half_gaussian", |b| {
            let sigma = positive(b, 0.8);
            let x = b.half_gaussian(sigma);
            b.set_value(x, DoubleTensor::scalar(0.6));
            let y = b.half_gaussian_with_shape(&[2], sigma * 1.5);
            b.observe(y, vector(&[0.3, 1.1]));
        }),
        case("cauchy", |b| {
            let location = scalar_latent(b, 0.2);
            let scale = positive(b, 0.7);
            let x = b.cauchy(location, scale);
            b.set_value(x, DoubleTensor::scalar(0.9));
            let y = b.cauchy_with_shape(&[3], location * scale, scale);
            b.observe(y, vector(&[-0.5, 0.4, 2.0]));
        }),
        case("half_cauchy", |b| {
            let scale = positive(b, 0.9);
            let x = b.half_cauchy(scale);
            b.set_value(x, DoubleTensor::scalar(0.4));
            let y = b.half_cauchy_with_shape(&[2], scale + 0.3);
            b.observe(y, vector(&[0.2, 1.7]));
        }),
        case("beta", |b| {
            let (alpha, beta) = (positive(b, 1.7), positive(b, 2.4));
            let x = b.beta(alpha, beta);
            b.set_value(x, DoubleTensor::scalar(0.35));
            let y = b.beta_with_shape(&[2], alpha * 0.5, beta);
            b.observe(y, vector(&[0.2, 0.8]));
        }),
        case("chi_squared", |b| {
            let x = b.chi_squared(3);
            b.set_value(x, DoubleTensor::scalar(1.7));
        }),
        case("exponential", |b| {
            let rate = positive(b, 1.3);
            let x = b.exponential(rate);
            b.set_value(x, DoubleTensor::scalar(0.8));
            let y = b.exponential_with_shape(&[2], rate * 2.0);
            b.observe(y, vector(&[0.3, 2.0]));
        }),
        case("gamma", |b| {
            let (theta, k) = (positive(b, 0.9), positive(b, 2.3));
            let x = b.gamma(theta, k);
            b.set_value(x, DoubleTensor::scalar(1.4));
            let y = b.gamma_with_shape(&[2], theta, k * 0.5);
            b.observe(y, vector(&[0.5, 3.0]));
        }),
        case("inverse_gamma", |b| {
            let (alpha, beta) = (positive(b, 2.1), positive(b, 1.2));
            let x = b.inverse_gamma(alpha, beta);
            b.set_value(x, DoubleTensor::scalar(0.7));
            let y = b.inverse_gamma_with_shape(&[2], alpha, beta * 2.0);
            b.observe(y, vector(&[0.4, 1.9]));
        }),
        case("laplace", |b| {
            let (mu, beta) = (scalar_latent(b, 0.1), positive(b, 0.8));
            let x = b.laplace(mu, beta);
            b.set_value(x, DoubleTensor::scalar(0.9));
            let y = b.laplace_with_shape(&[3], mu, beta);
            b.observe(y, vector(&[-0.7, 0.5, 1.3]));
        }),
        case("logistic", |b| {
            let (mu, s) = (scalar_latent(b, 0.1), positive(b, 0.8));
            let x = b.logistic(mu, s);
            b.set_value(x, DoubleTensor::scalar(0.9));
            let y = b.logistic_with_shape(&[3], mu, s * 1.5);
            b.observe(y, vector(&[-0.7, 0.5, 3.3]));
        }),
        case("log_normal", |b| {
            let (mu, sigma) = (scalar_latent(b, 0.2), positive(b, 0.6));
            let x = b.log_normal(mu, sigma);
            b.set_value(x, DoubleTensor::scalar(1.3));
            let y = b.log_normal_with_shape(&[2], mu, sigma);
            b.observe(y, vector(&[0.5, 2.2]));
        }),
        case("pareto", |b| {
            let (location, scale) = (positive(b, 0.8), positive(b, 1.7));
            let x = b.pareto(location, scale);
            b.set_value(x, DoubleTensor::scalar(1.5));
            let y = b.pareto_with_shape(&[2], location, scale * 0.7);
            b.observe(y, vector(&[1.0, 2.5]));
        }),
        case("student_t", |b| {
            let x = b.student_t(4);
            b.set_value(x, DoubleTensor::scalar(0.7));
        }),
        case("triangular", |b| {
            let x_min = scalar_latent(b, -0.5);
            let width = positive(b, 2.0);
            let x_max = x_min + width;
            let c = x_min + width * scalar_latent(b, 0.1).sigmoid();
            // one value below the mode and one above it
            let x = b.triangular_with_shape(&[2], x_min, x_max, c);
            b.set_value(x, vector(&[-0.1, 1.0]));
            let y = b.triangular_with_shape(&[2], x_min, x_max, c);
            b.observe(y, vector(&[1.3, 0.2]));
        }),
        case("uniform", |b| {
            let x_min = scalar_latent(b, -0.3);
            let x_max = x_min + positive(b, 1.5);
            let x = b.uniform(x_min, x_max);
            b.set_value(x, DoubleTensor::scalar(0.4));
            let y = b.uniform_with_shape(&[2], x_min, x_max);
            b.observe(y, vector(&[0.0, 1.0]));
        }),
        case("weibull", |b| {
            let (alpha, beta) = (positive(b, 1.6), positive(b, 1.2));
            let x = b.weibull(alpha, beta);
            b.set_value(x, DoubleTensor::scalar(0.9));
            let y = b.weibull_with_shape(&[2], alpha, beta);
            b.observe(y, vector(&[0.4, 2.1]));
        }),
        case("von_mises", |b| {
            let (mu, kappa) = (scalar_latent(b, 0.3), positive(b, 1.8));
            let x = b.von_mises(mu, kappa);
            b.set_value(x, DoubleTensor::scalar(0.9));
            let y = b.von_mises_with_shape(&[2], mu, kappa);
            b.observe(y, vector(&[-1.0, 2.0]));
            // a small concentration
            let z = b.von_mises_with_shape(&[2], mu, kappa * 0.01);
            b.observe(z, vector(&[-1.0, 2.0]));
        }),
        unconstrained("dirichlet", |b| {
            let concentration = positive_vector(b, &[1.5, 2.5, 0.8]);
            let x = b.dirichlet(concentration);
            b.set_value(x, vector(&[0.2, 0.5, 0.3]));
            let y = b.dirichlet(concentration * 2.0);
            b.observe(y, vector(&[0.1, 0.6, 0.3]));
        }),
        case("kde", |b| {
            let kde = keanu_distributions::Kde::new(vec![-0.5, 0.2, 0.4, 1.5], 0.6).unwrap();
            let x = b.kde(kde, &[2]);
            b.set_value(x, vector(&[0.1, 0.9]));
        }),
        unconstrained("lkj_cholesky", |b| {
            let factor = |r10: f64, r20: f64, r21: f64| {
                let d1 = (1.0 - r10 * r10).sqrt();
                let d2 = (1.0 - r20 * r20 - r21 * r21).sqrt();
                t(&[3, 3], &[1.0, 0.0, 0.0, r10, d1, 0.0, r20, r21, d2])
            };
            let eta = positive(b, 1.7);
            let l = b.lkj_cholesky(3, eta);
            b.set_value(l, factor(0.3, -0.2, 0.4));
            let y = b.lkj_cholesky(3, eta * 2.0);
            b.observe(y, factor(-0.5, 0.1, 0.3));
        }),
        case("bernoulli", |b| {
            let p = probabilities(b, &[0.3, 0.7]);
            let y = b.bernoulli(p);
            b.observe(y, keanu_tensor::BoolTensor::vector(vec![true, false]));
        }),
        case("binomial", |b| {
            let p = probabilities(b, &[0.3, 0.6]);
            let y = b.binomial(p, 5);
            b.observe(y, ints(&[2, 5]));
        }),
        case("geometric", |b| {
            let p = probabilities(b, &[0.3, 0.6]);
            let y = b.geometric(p);
            b.observe(y, ints(&[1, 4]));
        }),
        case("poisson", |b| {
            let rate = positive_vector(b, &[1.3, 2.0]);
            let y = b.poisson(rate);
            b.observe(y, ints(&[0, 3]));
        }),
        case("negative_binomial", |b| {
            let mu = positive_vector(b, &[1.5, 3.0]);
            let alpha = positive(b, 2.0);
            let y = b.negative_binomial(mu, alpha);
            b.observe(y, ints(&[0, 4]));
        }),
        case("categorical", |b| {
            let p = [positive(b, 0.2), positive(b, 0.5), positive(b, 0.3)];
            let y = b.categorical(&[2], &p);
            b.observe(y, ints(&[2, 0]));
        }),
        case("multinomial", |b| {
            let p = positive_vector(b, &[0.2, 0.5, 0.3]);
            let y = b.multinomial(6, p);
            b.observe(y, ints(&[1, 3, 2]));
            let q = latent_vector(b, &[0.4, -0.3, 0.1]).softmax(0);
            let z = b.multinomial(4, q);
            b.observe(z, ints(&[0, 4, 0]));
        }),
        case("ordered_logistic", |b| {
            let eta = latent_vector(b, &[0.2, -0.5, 1.0, 0.4]);
            let raw = latent_vector(b, &[-0.5, 0.8f64.ln(), 1.2f64.ln()]);
            let cutpoints = b
                .concat(0, &[raw.slice("0:1"), raw.slice("1:").exp()])
                .cumsum(0);
            // the first category, the two in the middle and the last
            let y = b.ordered_logistic(eta, cutpoints);
            b.observe(y, ints(&[0, 1, 2, 3]));
        }),
        case("uniform_int (no continuous parameters)", |b| {
            let y = b.uniform_int(0, 5);
            b.observe(y, IntTensor::scalar(2));
            observe(b, scalar_latent(b, 0.3).sin());
        }),
        // time series
        case("a hidden Markov model with Gaussian emissions", |b| {
            let params = [
                latent_vector(b, &[-0.5, 1.0]).id(),
                positive_vector(b, &[0.7, 1.3]).id(),
            ];
            let initial = latent_vector(b, &[0.3, -0.2]).softmax(0);
            let transition = latent(b, t(&[2, 2], &[0.8, -0.3, 0.1, 0.6])).softmax(1);
            let y: DoubleVar<'_> =
                b.hidden_markov_model(4, K::Gaussian, &params, initial, transition);
            b.observe(y, vector(&[-0.8, 0.2, 1.4, 0.9]));
            // a latent series, with two observations per step
            let x: DoubleVar<'_> = b.hidden_markov_model_with_shape(
                &[3, 2],
                K::Gaussian,
                &params,
                initial,
                transition,
            );
            b.set_value(x, t(&[3, 2], &[0.1, -0.4, 1.2, 0.8, -0.6, 0.3]));
        }),
        case("a hidden Markov model with Poisson emissions", |b| {
            let rates = positive_vector(b, &[1.0, 4.0]);
            let initial = latent_vector(b, &[0.3, -0.2]).softmax(0);
            let transition = latent(b, t(&[2, 2], &[0.8, -0.3, 0.1, 0.6])).softmax(1);
            let y: IntVar<'_> =
                b.hidden_markov_model(5, K::Poisson, &[rates.id()], initial, transition);
            b.observe(y, ints(&[0, 2, 5, 3, 1]));
        }),
        case(
            "a linear Gaussian state space model in every parameter",
            |b| {
                let eye = b.constant(DoubleTensor::eye(2).unwrap());
                // each covariance L Lᵀ + I/2 with a latent L
                let covariance = |values: &[f64]| {
                    let l = latent(b, t(&[2, 2], values));
                    l.matmul_transposed(l, false, true) + eye * 0.5
                };
                let y = b.linear_gaussian_state_space(
                    4,
                    latent_vector(b, &[0.2, -0.4]),
                    covariance(&[0.9, 0.0, 0.3, 0.7]),
                    latent(b, t(&[2, 2], &[0.8, 0.2, -0.1, 0.7])),
                    covariance(&[0.5, 0.0, -0.2, 0.4]),
                    latent(b, t(&[2, 2], &[1.0, 0.3, -0.2, 0.9])),
                    covariance(&[0.6, 0.0, 0.1, 0.5]),
                );
                b.observe(y, t(&[4, 2], &[0.1, 0.5, -0.2, 0.9, 0.4, -0.3, 1.1, 0.2]));
            },
        ),
        case(
            "a linear Gaussian state space model in its observations",
            |b| {
                // a state of dimension 2 observed in dimension 1, with latent observations, dynamics and noise
                let r = positive(b, 0.3).reshape(&[1, 1]);
                let q = positive_vector(b, &[0.4, 0.2]).diag();
                let y = b.linear_gaussian_state_space(
                    4,
                    DoubleTensor::vector(vec![0.0, 0.5]),
                    DoubleTensor::eye(2).unwrap(),
                    latent(b, t(&[2, 2], &[0.9, 0.1, -0.2, 0.6])),
                    q,
                    latent(b, t(&[1, 2], &[1.0, 0.4])),
                    r,
                );
                b.set_value(y, t(&[4, 1], &[0.3, -0.1, 0.8, 0.5]));
            },
        ),
        // mixtures
        case("a mixture of Gaussians", |b| {
            let params = [
                latent_vector(b, &[-0.5, 1.0]).id(),
                positive_vector(b, &[0.7, 1.3]).id(),
            ];
            let weights = latent_vector(b, &[0.3, -0.2]).softmax(0);
            let x: DoubleVar<'_> = b.mixture(K::Gaussian, &params, weights);
            b.set_value(x, DoubleTensor::scalar(0.4));
            let y: DoubleVar<'_> = b.mixture_with_shape(&[3], K::Gaussian, &params, weights);
            b.observe(y, vector(&[-1.0, 0.2, 1.5]));
        }),
        case("a mixture of gammas sharing their scale", |b| {
            let params = [positive(b, 0.9).id(), positive_vector(b, &[1.5, 4.0]).id()];
            let weights = latent_vector(b, &[0.3, -0.2]).softmax(0);
            let x: DoubleVar<'_> = b.mixture(K::Gamma, &params, weights);
            b.set_value(x, DoubleTensor::scalar(1.2));
            let y: DoubleVar<'_> = b.mixture_with_shape(&[2], K::Gamma, &params, weights);
            b.observe(y, vector(&[0.5, 4.0]));
        }),
        case("a mixture of Poissons", |b| {
            let rates = positive_vector(b, &[1.0, 4.0, 2.5]);
            let weights = latent_vector(b, &[0.3, -0.2, 0.1]).softmax(0);
            let y: IntVar<'_> = b.mixture_with_shape(&[3], K::Poisson, &[rates.id()], weights);
            b.observe(y, ints(&[0, 2, 5]));
        }),
        case("a mixture of truncated Gaussians", |b| {
            let (lower, upper) = bounds(b, -1.0, 3.0);
            let params = [
                latent_vector(b, &[-0.5, 1.0]).id(),
                positive_vector(b, &[0.7, 1.3]).id(),
                lower.id(),
                upper.id(),
            ];
            let weights = latent_vector(b, &[0.3, -0.2]).softmax(0);
            let kind = K::Truncated(Arc::new(K::Gaussian));
            let x: DoubleVar<'_> = b.mixture(kind, &params, weights);
            b.set_value(x, DoubleTensor::scalar(0.4));
        }),
        case("a mixture of truncated gammas with a latent shape", |b| {
            let params = [
                positive(b, 0.9).id(),
                positive_vector(b, &[1.5, 4.0]).id(),
                b.scalar(0.5).id(),
                b.scalar(3.0).id(),
            ];
            let weights = b.constant(vector(&[0.4, 0.6]));
            let kind = K::Truncated(Arc::new(K::Gamma));
            let y: DoubleVar<'_> = b.mixture(kind, &params, weights);
            b.observe(y, DoubleTensor::scalar(1.2));
        }),
        // truncated distributions, one for each base with a CDF
        case("truncated gaussian", |b| {
            truncated_location_scale(b, K::Gaussian)
        }),
        case("truncated cauchy", |b| {
            truncated_location_scale(b, K::Cauchy)
        }),
        case("truncated laplace", |b| {
            truncated_location_scale(b, K::Laplace)
        }),
        case("truncated logistic", |b| {
            truncated_location_scale(b, K::Logistic)
        }),
        case("truncated log_normal", |b| {
            let params = [scalar_latent(b, 0.2).id(), positive(b, 0.6).id()];
            let (lower, upper) = positive_bounds(b, 0.5, 2.0);
            let x: DoubleVar<'_> = b.truncated(K::LogNormal, &params, lower, upper);
            b.set_value(x, DoubleTensor::scalar(1.2));
            let z: DoubleVar<'_> = b.truncated(K::LogNormal, &params, lower, f64::INFINITY);
            b.observe(z, DoubleTensor::scalar(3.0));
        }),
        case("truncated exponential", |b| {
            let rate = [positive(b, 1.3).id()];
            let (lower, upper) = positive_bounds(b, 0.5, 2.0);
            let x: DoubleVar<'_> = b.truncated(K::Exponential, &rate, lower, upper);
            b.set_value(x, DoubleTensor::scalar(1.2));
            let z: DoubleVar<'_> = b.truncated(K::Exponential, &rate, lower, f64::INFINITY);
            b.observe(z, DoubleTensor::scalar(3.0));
            let w: DoubleVar<'_> = b.truncated(K::Exponential, &rate, 0.0, upper);
            b.observe(w, DoubleTensor::scalar(0.3));
        }),
        case("truncated weibull", |b| truncated_positive(b, K::Weibull)),
        case("truncated uniform", |b| {
            // the uniform has no derivatives with respect to its own bounds, so they are fixed here
            let (lower, upper) = bounds(b, -0.4, 2.0);
            let params = [b.scalar(-1.0).id(), b.scalar(2.0).id()];
            let x: DoubleVar<'_> = b.truncated(K::Uniform, &params, lower, upper);
            b.set_value(x, DoubleTensor::scalar(0.5));
        }),
        case("truncated gamma with a fixed shape", |b| {
            let params = [positive(b, 0.9).id(), b.scalar(2.3).id()];
            let (lower, upper) = positive_bounds(b, 0.5, 2.0);
            let x: DoubleVar<'_> = b.truncated(K::Gamma, &params, lower, upper);
            b.set_value(x, DoubleTensor::scalar(1.2));
            let z: DoubleVar<'_> = b.truncated(K::Gamma, &params, lower, f64::INFINITY);
            b.observe(z, DoubleTensor::scalar(3.0));
        }),
        case("truncated beta with fixed shapes", |b| {
            let lower = scalar_latent(b, -1.4).sigmoid();
            let upper = lower + (-lower + 1.0) * scalar_latent(b, 1.0).sigmoid();
            let params = [b.scalar(1.7).id(), b.scalar(2.4).id()];
            let x: DoubleVar<'_> = b.truncated(K::Beta, &params, lower, upper);
            b.set_value(x, DoubleTensor::scalar(0.5));
        }),
        case("truncated student_t", |b| {
            let v = [b.int_constant(IntTensor::scalar(4)).id()];
            let (lower, upper) = bounds(b, -0.4, 2.0);
            let x: DoubleVar<'_> = b.truncated(K::StudentT, &v, lower, upper);
            b.set_value(x, DoubleTensor::scalar(0.5));
            let z: DoubleVar<'_> = b.truncated(K::StudentT, &v, lower, f64::INFINITY);
            b.observe(z, DoubleTensor::scalar(2.5));
        }),
        case("truncated poisson", |b| {
            let rate = [positive_vector(b, &[2.5, 2.5, 2.5]).id()];
            let y: IntVar<'_> = b.truncated(K::Poisson, &rate, 1.0, 6.0);
            b.observe(y, ints(&[1, 3, 6]));
            let z: IntVar<'_> = b.truncated(K::Poisson, &rate, 1.0, f64::INFINITY);
            b.observe(z, ints(&[1, 3, 9]));
        }),
        case("truncated binomial", |b| {
            let p = probabilities(b, &[0.4, 0.4, 0.4]);
            let params = [p.id(), b.int_constant(IntTensor::scalar(8)).id()];
            let y: IntVar<'_> = b.truncated(K::Binomial, &params, 1.0, 6.0);
            b.observe(y, ints(&[1, 4, 6]));
        }),
        case("truncated negative_binomial with a fixed dispersion", |b| {
            let params = [
                positive_vector(b, &[2.5, 2.5, 2.5]).id(),
                b.scalar(2.0).id(),
            ];
            let y: IntVar<'_> = b.truncated(K::NegativeBinomial, &params, 0.0, 7.0);
            b.observe(y, ints(&[0, 3, 7]));
        }),
        // the shapes of CDFs built on the incomplete gamma and beta functions
        case("truncated gamma with a latent shape", |b| {
            truncated_positive(b, K::Gamma)
        }),
        case("truncated gamma with a large latent shape", |b| {
            // where the incomplete gamma function comes from its asymptotic expansion
            let params = [positive(b, 0.01).id(), positive(b, 150.0).id()];
            let (lower, upper) = positive_bounds(b, 1.3, 0.4);
            let x: DoubleVar<'_> = b.truncated(K::Gamma, &params, lower, upper);
            b.set_value(x, DoubleTensor::scalar(1.5));
        }),
        case("truncated beta with latent shapes", |b| {
            let lower = scalar_latent(b, -1.4).sigmoid();
            let upper = lower + (-lower + 1.0) * scalar_latent(b, 1.0).sigmoid();
            let params = [positive(b, 1.7).id(), positive(b, 2.4).id()];
            let x: DoubleVar<'_> = b.truncated(K::Beta, &params, lower, upper);
            b.set_value(x, DoubleTensor::scalar(0.5));
            let y: DoubleVar<'_> = b.truncated(K::Beta, &params, 0.05, upper);
            b.observe(y, DoubleTensor::scalar(0.3));
        }),
        case(
            "truncated negative_binomial with a latent dispersion",
            |b| {
                let params = [
                    positive_vector(b, &[2.5, 2.5, 2.5]).id(),
                    positive(b, 2.0).id(),
                ];
                let y: IntVar<'_> = b.truncated(K::NegativeBinomial, &params, 0.0, 7.0);
                b.observe(y, ints(&[0, 3, 7]));
                let z: IntVar<'_> = b.truncated(K::NegativeBinomial, &params, 2.0, f64::INFINITY);
                b.observe(z, ints(&[2, 4, 11]));
            },
        ),
        // censored distributions, one for each base with a CDF
        case("censored gaussian", |b| {
            censored_location_scale(b, K::Gaussian)
        }),
        case("censored cauchy", |b| censored_location_scale(b, K::Cauchy)),
        case("censored laplace", |b| {
            censored_location_scale(b, K::Laplace)
        }),
        case("censored logistic", |b| {
            censored_location_scale(b, K::Logistic)
        }),
        case("censored log_normal", |b| {
            censored_positive(b, K::LogNormal)
        }),
        case("censored weibull", |b| censored_positive(b, K::Weibull)),
        case("censored exponential", |b| {
            let rate = [positive(b, 1.3).id()];
            let lower = b.constant(vector(&[0.5, 0.5, 0.5]));
            let y: DoubleVar<'_> = b.censored(K::Exponential, &rate, lower, 2.5);
            b.observe(y, vector(&[0.5, 1.3, 2.5]));
        }),
        case("censored uniform", |b| {
            let params = [b.scalar(-1.0).id(), b.scalar(2.0).id()];
            let lower = b.constant(vector(&[-0.4, -0.4, -0.4]));
            let y: DoubleVar<'_> = b.censored(K::Uniform, &params, lower, 1.6);
            b.observe(y, vector(&[-0.4, 0.3, 1.6]));
            let (l, u) = bounds(b, -0.4, 2.0);
            let x: DoubleVar<'_> = b.censored(K::Uniform, &params, l, u);
            b.set_value(x, DoubleTensor::scalar(0.5));
        }),
        case("censored gamma with a fixed shape", |b| {
            let params = [positive(b, 0.9).id(), b.scalar(2.3).id()];
            let lower = b.constant(vector(&[0.5, 0.5, 0.5]));
            let y: DoubleVar<'_> = b.censored(K::Gamma, &params, lower, 2.5);
            b.observe(y, vector(&[0.5, 1.3, 2.5]));
        }),
        case(
            "censored gamma with a latent shape and no value at a bound",
            |b| {
                let params = [positive(b, 0.9).id(), positive(b, 2.3).id()];
                let lower = b.constant(vector(&[0.5, 0.5]));
                let y: DoubleVar<'_> = b.censored(K::Gamma, &params, lower, 2.5);
                b.observe(y, vector(&[0.7, 1.3]));
            },
        ),
        case("censored gamma with a latent shape", |b| {
            censored_positive(b, K::Gamma)
        }),
        case("censored beta with fixed shapes", |b| {
            let params = [b.scalar(1.7).id(), b.scalar(2.4).id()];
            let lower = b.constant(vector(&[0.2, 0.2, 0.2]));
            let y: DoubleVar<'_> = b.censored(K::Beta, &params, lower, 0.8);
            b.observe(y, vector(&[0.2, 0.5, 0.8]));
            let (l, u) = (scalar_latent(b, 0.2), scalar_latent(b, 0.8));
            let x: DoubleVar<'_> = b.censored(K::Beta, &params, l, u);
            b.set_value(x, DoubleTensor::scalar(0.5));
        }),
        case("censored beta with latent shapes", |b| {
            let params = [positive(b, 1.7).id(), positive(b, 2.4).id()];
            let lower = b.constant(vector(&[0.2, 0.2, 0.2]));
            let y: DoubleVar<'_> = b.censored(K::Beta, &params, lower, 0.8);
            b.observe(y, vector(&[0.2, 0.5, 0.8]));
        }),
        case("censored student_t", |b| {
            let v = [b.int_constant(IntTensor::scalar(4)).id()];
            let lower = b.constant(vector(&[-0.4, -0.4, -0.4]));
            let y: DoubleVar<'_> = b.censored(K::StudentT, &v, lower, 1.6);
            b.observe(y, vector(&[-0.4, 0.3, 1.6]));
            let (l, u) = bounds(b, -0.4, 2.0);
            let x: DoubleVar<'_> = b.censored(K::StudentT, &v, l, u);
            b.set_value(x, DoubleTensor::scalar(0.5));
        }),
        case("censored poisson", |b| {
            let rate = [positive_vector(b, &[2.5, 2.5, 2.5]).id()];
            let y: IntVar<'_> = b.censored(K::Poisson, &rate, 1.0, 6.0);
            b.observe(y, ints(&[1, 3, 6]));
        }),
        case("censored binomial", |b| {
            let p = probabilities(b, &[0.4, 0.4, 0.4]);
            let params = [p.id(), b.int_constant(IntTensor::scalar(8)).id()];
            let y: IntVar<'_> = b.censored(K::Binomial, &params, 1.0, 6.0);
            b.observe(y, ints(&[1, 4, 6]));
        }),
        case("censored negative_binomial with a fixed dispersion", |b| {
            let params = [
                positive_vector(b, &[2.5, 2.5, 2.5]).id(),
                b.scalar(2.0).id(),
            ];
            let y: IntVar<'_> = b.censored(K::NegativeBinomial, &params, 0.0, 7.0);
            b.observe(y, ints(&[0, 3, 7]));
        }),
        case("censored negative_binomial with a latent dispersion", |b| {
            let params = [
                positive_vector(b, &[2.5, 2.5, 2.5]).id(),
                positive(b, 2.0).id(),
            ];
            let y: IntVar<'_> = b.censored(K::NegativeBinomial, &params, 0.0, 7.0);
            b.observe(y, ints(&[0, 3, 7]));
        }),
    ]
}

/// The distribution kinds of a model's probabilistic nodes, by [`DistributionKind::name`] ("truncated"), and for a
/// kind built around another one also with the inner kind's name ("truncated(gaussian)").
fn kinds_of(model: &Model, kinds: &mut BTreeSet<String>) {
    let graph = model.graph();
    for id in graph.ids() {
        if let Op::Distribution(d) = graph[id].op() {
            kinds.insert(d.kind.name().to_string());
            if let Some((_, inner)) = d.kind.nested() {
                kinds.insert(format!("{}({})", d.kind.name(), inner.name()));
            }
        }
    }
}

fn assert_symmetric(name: &str, model: &Model, state: &mut State, space: Space, jacobian: bool) {
    let param = Parameterisation::with_space(model, state, space).unwrap();
    let z = param.from_state(state).unwrap();
    let n = z.len();
    let hessian = param.hessian(&z, jacobian, state).unwrap();
    let h = hessian.as_slice();
    for i in 0..n {
        for j in 0..i {
            let (a, b) = (h[i * n + j], h[j * n + i]);
            assert!(
                (a - b).abs() <= 1e-8 * a.abs().max(1.0),
                "{name}: H[{i}][{j}] = {a} but H[{j}][{i}] = {b}"
            );
        }
    }
}

#[test]
fn every_distribution_kind_matches_finite_differences_of_the_gradient() {
    let mut kinds = BTreeSet::new();
    for case in distribution_cases() {
        let (model, mut state) = named_model(case.name, case.build);
        kinds_of(&model, &mut kinds);
        let jacobian = case.space == Space::Unconstrained;
        match case.expect {
            Expect::Matches => {
                check_hessian(case.name, &model, &mut state, case.space, jacobian);
                assert_symmetric(case.name, &model, &mut state, case.space, jacobian);
            }
            Expect::NoRule(text) => {
                let param = Parameterisation::with_space(&model, &state, case.space).unwrap();
                let z = param.from_state(&state).unwrap();
                match param.hessian(&z, jacobian, &mut state) {
                    Err(GraphError::NoSecondOrderRule(message)) => assert!(
                        message.contains(text),
                        "{}: the error {message:?} does not mention {text:?}",
                        case.name
                    ),
                    other => panic!(
                        "{}: expected a no-second-order-rule error, got {other:?}",
                        case.name
                    ),
                }
            }
        }
    }
    // the Gaussian and multivariate Gaussian cases are in the op table
    for (name, build, _) in &general_cases() {
        let (model, _) = named_model(name, build);
        kinds_of(&model, &mut kinds);
    }
    // every kind but the custom one, which has no second-order rule of its own (see
    // a_distribution_needs_a_rule_only_where_it_depends_on_the_latents)
    let missing: Vec<&str> = DistributionKind::NAMES
        .iter()
        .copied()
        .filter(|&k| k != "custom" && !kinds.contains(k))
        .collect();
    assert!(
        missing.is_empty(),
        "distribution kinds without a case: {missing:?}"
    );
    for wrapper in ["truncated", "censored"] {
        let missing: Vec<&str> = DistributionKind::CDF_NAMES
            .iter()
            .copied()
            .filter(|base| !kinds.contains(&format!("{wrapper}({base})")))
            .collect();
        assert!(
            missing.is_empty(),
            "{wrapper} bases without a case: {missing:?}"
        );
    }
}
