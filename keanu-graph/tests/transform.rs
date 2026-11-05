//! Bijectors between unconstrained flat vectors and constrained values: round trips, log-Jacobians and their
//! gradients against finite differences, Jacobian-vector products, extremes and invalid input.

use keanu_distributions::KeanuRng;
use keanu_graph::transform::{self, Bijector};
use keanu_graph::{GraphError, Support};
use keanu_tensor::DoubleTensor;

/// Extracts the constrained coordinates the log-Jacobian is taken with respect to.
type Free = fn(&[usize], &DoubleTensor) -> Vec<f64>;

struct Case {
    name: String,
    bijector: Box<dyn Bijector>,
    /// The free coordinates of a constrained value, as many as the unconstrained length.
    free: Free,
}

fn all(_: &[usize], x: &DoubleTensor) -> Vec<f64> {
    x.as_slice().to_vec()
}

/// Every element of each last-axis row except the last, which is fixed by the others.
fn simplex_free(shape: &[usize], x: &DoubleTensor) -> Vec<f64> {
    let k = *shape.last().unwrap();
    x.as_slice()
        .chunks_exact(k)
        .flat_map(|row| row[..k - 1].to_vec())
        .collect()
}

/// The strictly lower triangle of each matrix, row by row.
fn lower_free(shape: &[usize], x: &DoubleTensor) -> Vec<f64> {
    let k = *shape.last().unwrap();
    let mut out = Vec::new();
    for m in x.as_slice().chunks_exact(k * k) {
        for i in 1..k {
            for j in 0..i {
                out.push(m[i * k + j]);
            }
        }
    }
    out
}

fn t(shape: &[usize], data: &[f64]) -> DoubleTensor {
    DoubleTensor::new(shape, data.to_vec()).unwrap()
}

fn bijector(support: &Support, shape: &[usize]) -> Box<dyn Bijector> {
    transform::bijector_for(support, shape).unwrap()
}

/// Every square bijector (every one except the circular transform) over several shapes.
fn square_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for shape in [vec![], vec![1], vec![3], vec![2, 3]] {
        let n: usize = shape.iter().product();
        let lower = DoubleTensor::from_fn(&shape, |i| i as f64 - 1.0);
        let upper = DoubleTensor::from_fn(&shape, |i| 2.0 * i as f64 + 0.5);
        let supports = [
            ("identity", Support::Real),
            ("lower_bound", Support::greater_than(0.0)),
            ("lower_bound_tensor", Support::GreaterThan(lower.clone())),
            ("upper_bound", Support::less_than(2.0)),
            ("upper_bound_tensor", Support::LessThan(upper.clone())),
            ("interval", Support::interval(-1.0, 3.0)),
            ("interval_tensor", Support::Interval(lower, upper)),
        ];
        for (name, support) in supports {
            cases.push(Case {
                name: format!("{name} {shape:?}"),
                bijector: bijector(&support, &shape),
                free: all,
            });
        }
        assert_eq!(cases.last().unwrap().bijector.unconstrained_len(), n);
    }
    for shape in [vec![1], vec![3], vec![2, 3]] {
        cases.push(Case {
            name: format!("ordered {shape:?}"),
            bijector: bijector(&Support::Ordered, &shape),
            free: all,
        });
    }
    for shape in [vec![1], vec![2], vec![3], vec![5], vec![2, 3]] {
        cases.push(Case {
            name: format!("simplex {shape:?}"),
            bijector: bijector(&Support::Simplex, &shape),
            free: simplex_free,
        });
    }
    for shape in [
        vec![1, 1],
        vec![2, 2],
        vec![3, 3],
        vec![4, 4],
        vec![2, 3, 3],
    ] {
        cases.push(Case {
            name: format!("correlation_cholesky {shape:?}"),
            bijector: bijector(&Support::CorrelationCholesky, &shape),
            free: lower_free,
        });
    }
    cases
}

fn random_y(rng: &mut KeanuRng, n: usize, scale: f64) -> Vec<f64> {
    (0..n).map(|_| scale * rng.next_gaussian()).collect()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: lengths");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol * (1.0 + e.abs()),
            "{what}: element {i}: {a} vs {e}"
        );
    }
}

/// The forward Jacobian of the free coordinates by central differences, row-major (free x unconstrained).
fn fd_jacobian(case: &Case, y: &[f64]) -> Vec<Vec<f64>> {
    let shape = case.bijector.constrained_shape().to_vec();
    let h = 1e-6;
    let n = y.len();
    let mut columns = Vec::new();
    for j in 0..n {
        let mut plus = y.to_vec();
        let mut minus = y.to_vec();
        plus[j] += h;
        minus[j] -= h;
        let fp = (case.free)(&shape, &case.bijector.forward(&plus).unwrap());
        let fm = (case.free)(&shape, &case.bijector.forward(&minus).unwrap());
        columns.push(
            fp.iter()
                .zip(&fm)
                .map(|(p, m)| (p - m) / (2.0 * h))
                .collect::<Vec<f64>>(),
        );
    }
    (0..n)
        .map(|i| (0..n).map(|j| columns[j][i]).collect())
        .collect()
}

/// log |det a| by Gaussian elimination with partial pivoting.
fn log_abs_det(mut a: Vec<Vec<f64>>) -> f64 {
    let n = a.len();
    let mut log_det = 0.0;
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&p, &q| a[p][col].abs().total_cmp(&a[q][col].abs()))
            .unwrap();
        a.swap(col, pivot);
        let p = a[col][col];
        log_det += p.abs().ln();
        let (top, bottom) = a.split_at_mut(col + 1);
        let pivot_row = &top[col];
        for row in bottom {
            let factor = row[col] / p;
            for (r, q) in row[col..].iter_mut().zip(&pivot_row[col..]) {
                *r -= factor * q;
            }
        }
    }
    log_det
}

#[test]
fn inverse_of_forward_is_the_identity() {
    let mut rng = KeanuRng::seed_from_u64(11);
    for case in square_cases() {
        for _ in 0..5 {
            let y = random_y(&mut rng, case.bijector.unconstrained_len(), 1.5);
            let x = case.bijector.forward(&y).unwrap();
            assert_eq!(
                x.shape(),
                case.bijector.constrained_shape(),
                "{}",
                case.name
            );
            let back = case.bijector.inverse(&x).unwrap();
            assert_close(&back, &y, 1e-8, &case.name);
        }
    }
}

#[test]
fn forward_of_inverse_is_the_identity() {
    let mut rng = KeanuRng::seed_from_u64(12);
    for case in square_cases() {
        let y = random_y(&mut rng, case.bijector.unconstrained_len(), 1.0);
        let x = case.bijector.forward(&y).unwrap();
        let again = case
            .bijector
            .forward(&case.bijector.inverse(&x).unwrap())
            .unwrap();
        assert_close(again.as_slice(), x.as_slice(), 1e-10, &case.name);
    }
}

#[test]
fn inverse_of_hand_written_valid_values() {
    let lower = bijector(&Support::greater_than(1.0), &[3]);
    assert_close(
        &lower.inverse(&t(&[3], &[2.0, 1.5, 4.0])).unwrap(),
        &[0.0, 0.5f64.ln(), 3.0f64.ln()],
        1e-12,
        "lower bound",
    );
    let interval = bijector(&Support::interval(0.0, 4.0), &[]);
    assert_close(
        &interval.inverse(&DoubleTensor::scalar(2.0)).unwrap(),
        &[0.0],
        1e-12,
        "interval midpoint",
    );
    let ordered = bijector(&Support::Ordered, &[3]);
    assert_close(
        &ordered.inverse(&t(&[3], &[-1.0, 0.0, 2.0])).unwrap(),
        &[-1.0, 0.0, 2.0f64.ln()],
        1e-12,
        "ordered",
    );
    let simplex = bijector(&Support::Simplex, &[3]);
    let x = t(&[3], &[0.2, 0.3, 0.5]);
    let y = simplex.inverse(&x).unwrap();
    assert_eq!(y.len(), 2);
    assert_close(
        simplex.forward(&y).unwrap().as_slice(),
        x.as_slice(),
        1e-12,
        "simplex",
    );
    let uniform = simplex.inverse(&t(&[3], &[1.0 / 3.0; 3])).unwrap();
    assert_close(&uniform, &[0.0, 0.0], 1e-12, "uniform simplex");
    let corr = bijector(&Support::CorrelationCholesky, &[3, 3]);
    let identity = t(&[3, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    assert_close(
        &corr.inverse(&identity).unwrap(),
        &[0.0; 3],
        1e-12,
        "identity",
    );
    assert_close(
        corr.forward(&[0.0; 3]).unwrap().as_slice(),
        identity.as_slice(),
        1e-15,
        "forward of zero",
    );
}

#[test]
fn log_det_jacobian_matches_finite_differences() {
    let mut rng = KeanuRng::seed_from_u64(13);
    for case in square_cases() {
        for _ in 0..3 {
            let y = random_y(&mut rng, case.bijector.unconstrained_len(), 1.0);
            let expected = log_abs_det(fd_jacobian(&case, &y));
            let actual = case.bijector.log_det_jacobian(&y).unwrap();
            assert!(
                (actual - expected).abs() <= 1e-6 * (1.0 + expected.abs()),
                "{}: {actual} vs {expected} at {y:?}",
                case.name
            );
        }
    }
}

#[test]
fn log_det_jacobian_gradient_matches_finite_differences() {
    let mut rng = KeanuRng::seed_from_u64(14);
    let h = 1e-6;
    for case in square_cases() {
        let y = random_y(&mut rng, case.bijector.unconstrained_len(), 1.0);
        let grad = case.bijector.log_det_jacobian_gradient(&y).unwrap();
        let fd: Vec<f64> = (0..y.len())
            .map(|j| {
                let mut plus = y.clone();
                let mut minus = y.clone();
                plus[j] += h;
                minus[j] -= h;
                (case.bijector.log_det_jacobian(&plus).unwrap()
                    - case.bijector.log_det_jacobian(&minus).unwrap())
                    / (2.0 * h)
            })
            .collect();
        assert_close(&grad, &fd, 1e-6, &case.name);
    }
}

#[test]
fn jacobian_vector_products_match_finite_differences_and_each_other() {
    let mut rng = KeanuRng::seed_from_u64(15);
    let h = 1e-6;
    for case in square_cases() {
        let b = &case.bijector;
        let n = b.unconstrained_len();
        let y = random_y(&mut rng, n, 1.0);
        let tangent = random_y(&mut rng, n, 1.0);
        let jvp = b.forward_jvp(&y, &tangent).unwrap();
        assert_eq!(jvp.shape(), b.constrained_shape(), "{}", case.name);
        let plus: Vec<f64> = y.iter().zip(&tangent).map(|(y, t)| y + h * t).collect();
        let minus: Vec<f64> = y.iter().zip(&tangent).map(|(y, t)| y - h * t).collect();
        let (fp, fm) = (b.forward(&plus).unwrap(), b.forward(&minus).unwrap());
        let fd: Vec<f64> = fp
            .as_slice()
            .iter()
            .zip(fm.as_slice())
            .map(|(p, m)| (p - m) / (2.0 * h))
            .collect();
        assert_close(jvp.as_slice(), &fd, 1e-6, &case.name);

        // <vjp(c), t> = <c, jvp(t)>
        let cotangent = DoubleTensor::from_fn(b.constrained_shape(), |_| rng.next_gaussian());
        let vjp = b.forward_vjp(&y, &cotangent).unwrap();
        assert_eq!(vjp.len(), n, "{}", case.name);
        let left: f64 = vjp.iter().zip(&tangent).map(|(a, b)| a * b).sum();
        let right: f64 = cotangent
            .as_slice()
            .iter()
            .zip(jvp.as_slice())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            (left - right).abs() <= 1e-10 * (1.0 + right.abs()),
            "{}: {left} vs {right}",
            case.name
        );
    }
}

#[test]
fn simplex_stays_on_the_simplex_at_extremes() {
    for k in [2usize, 3, 5, 10] {
        let b = bijector(&Support::Simplex, &[k]);
        let patterns: Vec<Vec<f64>> = vec![
            vec![50.0; k - 1],
            vec![-50.0; k - 1],
            (0..k - 1)
                .map(|i| if i % 2 == 0 { 50.0 } else { -50.0 })
                .collect(),
        ];
        for y in patterns {
            let x = b.forward(&y).unwrap();
            let sum: f64 = x.as_slice().iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "k={k}, y={y:?}: sum {sum}");
            assert!(
                x.as_slice().iter().all(|&v| v > 0.0 && v.is_finite()),
                "k={k}, y={y:?}: {x:?}"
            );
            assert!(b.log_det_jacobian(&y).unwrap().is_finite());
            assert!(
                b.log_det_jacobian_gradient(&y)
                    .unwrap()
                    .iter()
                    .all(|g| g.is_finite())
            );
        }
    }
}

#[test]
fn simplex_log_det_is_sum_of_log_x_plus_half_log_k() {
    let b = bijector(&Support::Simplex, &[4]);
    let y = [0.3, -1.2, 0.7];
    let x = b.forward(&y).unwrap();
    let expected: f64 = x.as_slice().iter().map(|v| v.ln()).sum::<f64>() + 0.5 * 4f64.ln();
    assert!((b.log_det_jacobian(&y).unwrap() - expected).abs() < 1e-12);
}

#[test]
fn correlation_cholesky_has_unit_rows_and_positive_diagonal() {
    let mut rng = KeanuRng::seed_from_u64(16);
    for k in [2usize, 3, 5] {
        let b = bijector(&Support::CorrelationCholesky, &[k, k]);
        let n = k * (k - 1) / 2;
        assert_eq!(b.unconstrained_len(), n);
        for scale in [1.0, 5.0, 30.0] {
            let y = random_y(&mut rng, n, scale);
            let l = b.forward(&y).unwrap();
            let m = l.as_slice();
            for i in 0..k {
                let norm: f64 = (0..k).map(|j| m[i * k + j] * m[i * k + j]).sum();
                assert!((norm - 1.0).abs() < 1e-12, "k={k} row {i}: {norm}");
                assert!(m[i * k + i] > 0.0, "k={k} diagonal {i}");
                assert!((i + 1..k).all(|j| m[i * k + j] == 0.0), "upper triangle");
            }
            assert!(b.log_det_jacobian(&y).unwrap().is_finite());
        }
    }
}

#[test]
fn circular_transform() {
    let b = bijector(&Support::Circular, &[3]);
    assert_eq!(b.unconstrained_len(), 6);
    assert_eq!(b.constrained_shape(), &[3]);
    let y = [1.0, 1.0, -2.0, 0.0, 0.0, -0.5];
    let theta = b.forward(&y).unwrap();
    assert_close(
        theta.as_slice(),
        &[
            std::f64::consts::FRAC_PI_4,
            std::f64::consts::PI,
            -std::f64::consts::FRAC_PI_2,
        ],
        1e-15,
        "angles",
    );
    let r2: f64 = y.iter().map(|v| v * v).sum();
    assert!((b.log_det_jacobian(&y).unwrap() + 0.5 * r2).abs() < 1e-15);
    let negated: Vec<f64> = y.iter().map(|v| -v).collect();
    assert_close(
        &b.log_det_jacobian_gradient(&y).unwrap(),
        &negated,
        0.0,
        "gradient",
    );

    // The inverse picks the point on the unit circle; forward recovers the angle.
    let angles = t(&[3], &[0.3, -2.0, 3.0]);
    let back = b.inverse(&angles).unwrap();
    assert_close(
        &back,
        &[
            0.3f64.cos(),
            0.3f64.sin(),
            (-2.0f64).cos(),
            (-2.0f64).sin(),
            3.0f64.cos(),
            3.0f64.sin(),
        ],
        1e-15,
        "inverse",
    );
    assert_close(
        b.forward(&back).unwrap().as_slice(),
        angles.as_slice(),
        1e-14,
        "round trip",
    );
    // An angle outside (-π, π] comes back wrapped.
    let wrapped = b
        .forward(&b.inverse(&t(&[3], &[7.0, 0.0, -4.0])).unwrap())
        .unwrap();
    let tau = std::f64::consts::TAU;
    assert_close(
        wrapped.as_slice(),
        &[7.0 - tau, 0.0, -4.0 + tau],
        1e-14,
        "wrapped",
    );

    // Products: the angle's derivative along (du, dv) is (u dv - v du) / r².
    let mut rng = KeanuRng::seed_from_u64(17);
    let y = random_y(&mut rng, 6, 1.0);
    let tangent = random_y(&mut rng, 6, 1.0);
    let jvp = b.forward_jvp(&y, &tangent).unwrap();
    let h = 1e-6;
    let plus: Vec<f64> = y.iter().zip(&tangent).map(|(y, t)| y + h * t).collect();
    let minus: Vec<f64> = y.iter().zip(&tangent).map(|(y, t)| y - h * t).collect();
    let fd: Vec<f64> = b
        .forward(&plus)
        .unwrap()
        .as_slice()
        .iter()
        .zip(b.forward(&minus).unwrap().as_slice())
        .map(|(p, m)| (p - m) / (2.0 * h))
        .collect();
    assert_close(jvp.as_slice(), &fd, 1e-6, "circular jvp");
    let cotangent = t(&[3], &[0.5, -1.0, 2.0]);
    let vjp = b.forward_vjp(&y, &cotangent).unwrap();
    let left: f64 = vjp.iter().zip(&tangent).map(|(a, b)| a * b).sum();
    let right: f64 = cotangent
        .as_slice()
        .iter()
        .zip(jvp.as_slice())
        .map(|(a, b)| a * b)
        .sum();
    assert!((left - right).abs() < 1e-12);

    // At the origin the angle has no derivative.
    assert!(matches!(
        b.forward_jvp(&[0.0; 6], &[1.0; 6]),
        Err(GraphError::Transform(_))
    ));
}

#[test]
fn supports_map_to_bijectors_of_the_right_size() {
    let cases: Vec<(Support, Vec<usize>, usize)> = vec![
        (Support::Real, vec![2, 3], 6),
        (Support::greater_than(0.0), vec![], 1),
        (Support::interval(0.0, 1.0), vec![4], 4),
        (Support::Ordered, vec![2, 3], 6),
        (Support::Simplex, vec![2, 3], 4),
        (Support::Simplex, vec![1], 0),
        (Support::CorrelationCholesky, vec![3, 3], 3),
        (Support::CorrelationCholesky, vec![2, 4, 4], 12),
        (Support::Circular, vec![3], 6),
        (Support::Circular, vec![], 2),
    ];
    for (support, shape, len) in cases {
        let b = bijector(&support, &shape);
        assert_eq!(b.unconstrained_len(), len, "{support:?} {shape:?}");
        assert_eq!(b.constrained_shape(), shape.as_slice());
    }
}

#[test]
fn tensor_bounds_apply_element_wise() {
    let lower = t(&[2], &[0.0, 10.0]);
    let upper = t(&[2], &[1.0, 20.0]);
    let b = bijector(&Support::Interval(lower, upper), &[3, 2]);
    let x = b.forward(&[0.0; 6]).unwrap();
    assert_eq!(x.as_slice(), &[0.5, 15.0, 0.5, 15.0, 0.5, 15.0]);
    let b = bijector(&Support::GreaterThan(t(&[2, 1], &[-1.0, 5.0])), &[2, 2]);
    assert_eq!(
        b.forward(&[0.0; 4]).unwrap().as_slice(),
        &[0.0, 0.0, 6.0, 6.0]
    );
}

fn is_transform_error<T: std::fmt::Debug>(result: Result<T, GraphError>) -> bool {
    matches!(result, Err(GraphError::Transform(_)))
}

#[test]
fn invalid_supports_are_rejected() {
    let nan = f64::NAN;
    let invalid: Vec<(Support, Vec<usize>)> = vec![
        (Support::interval(2.0, 2.0), vec![]),
        (Support::interval(3.0, 2.0), vec![3]),
        (Support::interval(nan, 1.0), vec![]),
        (Support::interval(0.0, nan), vec![]),
        (Support::interval(0.0, f64::INFINITY), vec![]),
        (
            Support::Interval(t(&[2], &[0.0, 1.0]), t(&[2], &[1.0, 1.0])),
            vec![2],
        ),
        (Support::greater_than(nan), vec![2]),
        (Support::greater_than(f64::NEG_INFINITY), vec![]),
        (Support::GreaterThan(t(&[3], &[0.0; 3])), vec![2]),
        (Support::GreaterThan(t(&[3], &[0.0; 3])), vec![]),
        (Support::Discrete, vec![]),
        (Support::Simplex, vec![]),
        (Support::Simplex, vec![0]),
        (Support::Ordered, vec![]),
        (Support::CorrelationCholesky, vec![3]),
        (Support::CorrelationCholesky, vec![2, 3]),
    ];
    for (support, shape) in invalid {
        assert!(
            is_transform_error(transform::bijector_for(&support, &shape)),
            "{support:?} {shape:?}"
        );
    }
}

#[test]
fn inverse_of_values_outside_or_on_the_boundary_errors() {
    let invalid: Vec<(Support, DoubleTensor)> = vec![
        (Support::greater_than(0.0), DoubleTensor::scalar(0.0)),
        (Support::greater_than(0.0), t(&[2], &[1.0, -1.0])),
        (Support::greater_than(0.0), DoubleTensor::scalar(f64::NAN)),
        (
            Support::greater_than(0.0),
            DoubleTensor::scalar(f64::INFINITY),
        ),
        (Support::interval(0.0, 1.0), DoubleTensor::scalar(0.0)),
        (Support::interval(0.0, 1.0), DoubleTensor::scalar(1.0)),
        (Support::interval(0.0, 1.0), DoubleTensor::scalar(f64::NAN)),
        (Support::Real, DoubleTensor::scalar(f64::NAN)),
        (Support::Ordered, t(&[3], &[0.0, 1.0, 1.0])),
        (Support::Ordered, t(&[2], &[1.0, 0.0])),
        (Support::Simplex, t(&[3], &[0.0, 0.5, 0.5])),
        (Support::Simplex, t(&[2], &[0.5, 0.6])),
        (Support::Simplex, t(&[2], &[f64::NAN, 0.5])),
        (
            Support::CorrelationCholesky,
            t(&[2, 2], &[1.0, 0.0, 1.0, 0.0]),
        ),
        (
            Support::CorrelationCholesky,
            t(&[2, 2], &[1.0, 0.0, 0.6, -0.8]),
        ),
        (
            Support::CorrelationCholesky,
            t(&[2, 2], &[1.0, 0.0, 0.5, 0.5]),
        ),
        (
            Support::CorrelationCholesky,
            t(&[2, 2], &[1.0, 0.1, 0.6, 0.8]),
        ),
        (Support::Circular, DoubleTensor::scalar(f64::INFINITY)),
    ];
    for (support, x) in invalid {
        let b = bijector(&support, x.shape());
        assert!(
            is_transform_error(b.inverse(&x)),
            "{support:?} inverse of {x:?}"
        );
    }
}

#[test]
fn wrong_sizes_are_rejected() {
    let b = bijector(&Support::Simplex, &[3]);
    assert!(is_transform_error(b.forward(&[0.0; 3])));
    assert!(is_transform_error(b.log_det_jacobian(&[0.0])));
    assert!(is_transform_error(b.log_det_jacobian_gradient(&[])));
    assert!(is_transform_error(b.forward_jvp(&[0.0; 2], &[0.0; 3])));
    assert!(is_transform_error(
        b.forward_vjp(&[0.0; 2], &DoubleTensor::scalar(1.0))
    ));
    assert!(is_transform_error(b.inverse(&t(&[2], &[0.5, 0.5]))));
}
