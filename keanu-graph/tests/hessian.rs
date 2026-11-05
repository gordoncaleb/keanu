//! Exact Hessian-vector products of the joint log probability (forward-over-reverse), checked against finite
//! differences of the exact gradient.
//!
//! The table-driven test builds one small model per op family, with Gaussian latents set to chosen points and every
//! deterministic output observed through a Gaussian likelihood, so each op's second-order rule (and the first-order
//! rule applied to the tangent channel) is exercised. It checks that the table covers every op of the graph.

use keanu_distributions::KeanuRng;
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
            "a distribution without a second-order rule",
            |b| {
                let g = b.gamma(1.5, 2.0);
                b.set_value(g, DoubleTensor::scalar(0.8));
                let y = b.gaussian(g, 1.0);
                b.observe(y, DoubleTensor::scalar(0.4));
            },
            Expect::NoRule("gamma"),
        ),
    ]
}

/// Every op of the graph, by [`keanu_graph::op::Op::name`].
const EVERY_OP: &[&str] = &[
    "constant",
    "unary",
    "binary",
    "compare",
    "mask",
    "approx_eq",
    "logic",
    "not",
    "all",
    "any",
    "predicate",
    "reduce",
    "standardize",
    "scan",
    "arg_reduce",
    "softmax",
    "log_softmax",
    "reshape",
    "permute",
    "transpose",
    "broadcast",
    "slice",
    "concat",
    "take",
    "gather",
    "scatter_add",
    "stack",
    "diag",
    "diag_part",
    "tril",
    "triu",
    "fill_triangular",
    "triangle_part",
    "matmul",
    "tensor_multiply",
    "inverse",
    "determinant",
    "cholesky",
    "cholesky_inverse",
    "solve_triangular",
    "solve",
    "log_abs_determinant",
    "determinant_sign",
    "eigvalsh",
    "eigh_vectors",
    "cast",
    "where",
    "multiplexer",
    "cpt",
    "set_with_mask",
    "replace_nan",
    "proxy",
    "lambda",
    "assert",
    "distribution",
];

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
    let missing: Vec<&str> = EVERY_OP
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
    // a discrete Poisson latent whose rate depends on a continuous latent: its log density is a function of the
    // continuous latent through the rate, so it needs a rule
    let (model, mut state) = model_of(|b| {
        let x = latent(b, DoubleTensor::scalar(0.3));
        let k = b.poisson(x.exp());
        b.set_value(k, keanu_tensor::IntTensor::scalar(2));
    });
    let param = Parameterisation::new(&model, &state).unwrap();
    let z = param.from_state(&state).unwrap();
    match param.hessian(&z, true, &mut state) {
        Err(GraphError::NoSecondOrderRule(message)) => assert!(message.contains("poisson")),
        other => panic!("expected a no-second-order-rule error, got {other:?}"),
    }
    // the same Poisson with a constant rate doesn't depend on the continuous latent, so needs no rule
    let (model, mut state) = model_of(|b| {
        let x = latent(b, DoubleTensor::scalar(0.3));
        let k = b.poisson(2.0);
        b.set_value(k, keanu_tensor::IntTensor::scalar(2));
        let y = b.gaussian(x, 1.0);
        b.observe(y, DoubleTensor::scalar(0.1));
    });
    check_hessian(
        "constant Poisson",
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
