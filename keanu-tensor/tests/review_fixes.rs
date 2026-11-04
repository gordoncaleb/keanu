//! Regression tests for code review findings: special function accuracy, and edge-case inputs that used to
//! panic, loop for ever or give wrong answers. Reference values are from mpmath (40 digits), rounded to f64.

use keanu_tensor::{DoubleTensor, IntTensor, TensorError};
use std::time::{Duration, Instant};

fn digamma(x: f64) -> f64 {
    DoubleTensor::scalar(x).digamma().to_scalar().unwrap()
}

fn trigamma(x: f64) -> f64 {
    DoubleTensor::scalar(x).trigamma().to_scalar().unwrap()
}

fn assert_rel(actual: f64, expected: f64, tolerance: f64) {
    let error = ((actual - expected) / expected).abs();
    assert!(
        error <= tolerance,
        "{actual} vs {expected}: relative error {error:e}"
    );
}

#[test]
fn digamma_matches_mpmath() {
    // ψ(1) = -γ was off by ~2.9e-9 from a sign error in the asymptotic series
    for (x, expected) in [
        (1.0, -0.5772156649015329),
        (0.5, -1.9635100260214235),
        (10.0, 2.251752589066721),
        (49.0, 3.881581510162586),
        (60.5, 4.094356135733628),
        (1000.0, 6.907255195648812),
        (-0.5, 0.03648997397857652),
        (-2.5, 1.103156640645243),
        (-10.3, 4.662403493582097),
        (-123_456.25, 14.865238824878356),
    ] {
        assert_rel(digamma(x), expected, 1e-13);
    }
}

#[test]
fn trigamma_matches_mpmath() {
    // the 1/(42 z⁷) term of the asymptotic series had the wrong sign
    for (x, expected) in [
        (1.0, 1.6449340668482264),
        (0.5, 4.934802200544679),
        (10.0, 0.10516633568168575),
        (49.0, 0.020617826354560515),
        (60.5, 0.016666280901697868),
        (1000.0, 0.0010005001666666333),
        (-0.5, 8.934802200544679),
        (-2.5, 9.539246644989124),
        (-10.3, 14.986887065722687),
        (-123_456.25, 19.739200702176085),
    ] {
        assert_rel(trigamma(x), expected, 1e-12);
    }
}

#[test]
fn digamma_and_trigamma_of_large_negative_arguments_are_fast() {
    let start = Instant::now();
    // these used to loop ~|x| times (and for ever from about 1e16, where x + 1 == x)
    assert_rel(digamma(-1000000000000000.5), 34.538776394910684, 1e-12);
    assert_rel(trigamma(-1000000000000000.5), 9.869604401089358, 1e-12);
    for x in [-1e16, -1e300, f64::MIN] {
        // every float this large is an integer, so a pole
        assert_eq!(digamma(x), f64::NEG_INFINITY);
        assert_eq!(trigamma(x), f64::INFINITY);
    }
    assert!(start.elapsed() < Duration::from_secs(1));
}

#[test]
fn digamma_and_trigamma_poles_are_unchanged() {
    for x in [0.0, -1.0, -2.0, -50.0] {
        assert_eq!(digamma(x), f64::NEG_INFINITY, "digamma({x})");
        assert_eq!(trigamma(x), f64::INFINITY, "trigamma({x})");
    }
    assert_eq!(digamma(-0.0), f64::INFINITY);
}

#[test]
fn shapes_whose_length_overflows_are_errors() {
    let huge = [usize::MAX, 2];
    assert!(matches!(
        DoubleTensor::new(&huge, vec![]),
        Err(TensorError::LengthMismatch { .. })
    ));
    // a wrapping product used to make this look like a 2 element shape
    let wraps = [1usize << 63, 2, 2];
    assert!(matches!(
        DoubleTensor::new(&wraps, vec![0.0; 0]),
        Err(TensorError::LengthMismatch { .. })
    ));
    // the error message doesn't overflow either
    let message = DoubleTensor::new(&huge, vec![]).unwrap_err().to_string();
    assert!(message.contains("usize"), "{message}");
    // a shape with a zero dimension has no elements however large the others are
    assert_eq!(
        DoubleTensor::new(&[usize::MAX, 2, 0], vec![])
            .unwrap()
            .len(),
        0
    );

    let t = DoubleTensor::vector(vec![1.0, 2.0, 3.0, 4.0]);
    for target in [
        vec![isize::MAX, isize::MAX, -1],
        vec![isize::MAX, 4],
        vec![1 << 62, 4, 4],
    ] {
        assert!(
            matches!(t.reshape(&target), Err(TensorError::InvalidReshape { .. })),
            "{target:?}"
        );
    }

    // broadcasting [n, 1] with [1, n] where n * n overflows
    let n = 1usize << 33;
    let left = DoubleTensor::new(&[n, 1, 0], vec![]).unwrap();
    assert!(left.broadcast_to(&[n, n, 0]).is_ok());
    let a = DoubleTensor::new(&[0, n, 1], vec![]).unwrap();
    assert!(a.reshape(&[0, 1, n as isize]).is_ok());
    let big = DoubleTensor::scalar(1.0);
    assert!(matches!(
        big.broadcast_to(&[n, n]),
        Err(TensorError::IncompatibleShapes { .. })
    ));
}

#[test]
fn slicing_with_a_huge_step_takes_one_element() {
    let v = DoubleTensor::vector((0..10).map(f64::from).collect());
    let sliced = v.slice("0:10:9223372036854775807").unwrap();
    assert_eq!(sliced.as_slice(), &[0.0]);
    let sliced = v.slice("9:0:-9223372036854775808").unwrap();
    assert_eq!(sliced.as_slice(), &[9.0]);
    let sliced = v.slice("::-9223372036854775807").unwrap();
    assert_eq!(sliced.as_slice(), &[9.0]);
    // a huge step on a dimension with a large stride
    let m = DoubleTensor::new(&[3, 2], (0..6).map(f64::from).collect()).unwrap();
    let sliced = m.slice("1::9223372036854775807, :").unwrap();
    assert_eq!(sliced.shape(), &[1, 2]);
    assert_eq!(sliced.as_slice(), &[2.0, 3.0]);
}

#[test]
fn argmax_and_argmin_along_an_empty_axis_are_errors() {
    let empty = DoubleTensor::new(&[2, 0], vec![]).unwrap();
    assert!(empty.argmax_axis(1).is_err());
    assert!(empty.argmin_axis(-1).is_err());
    let ints = IntTensor::new(&[0, 3], vec![]).unwrap();
    assert!(ints.argmax_axis(0).is_err());
    // along a non-empty axis of an empty tensor there is simply nothing to return
    assert_eq!(ints.argmax_axis(1).unwrap().shape(), &[0]);
}

#[test]
fn arange_and_eye_reject_impossible_sizes() {
    assert!(DoubleTensor::arange(0.0, f64::INFINITY).is_err());
    assert!(DoubleTensor::arange(f64::NAN, 1.0).is_err());
    assert!(DoubleTensor::arange(0.0, 1e300).is_err());
    assert!(DoubleTensor::arange(-f64::MAX, f64::MAX).is_err());
    assert_eq!(
        DoubleTensor::arange(1.0, 4.0).unwrap().as_slice(),
        &[1.0, 2.0, 3.0]
    );
    assert_eq!(DoubleTensor::arange(4.0, 1.0).unwrap().len(), 0);

    assert!(DoubleTensor::eye(usize::MAX).is_err());
    assert!(DoubleTensor::eye(1 << 32).is_err());
    assert_eq!(
        DoubleTensor::eye(2).unwrap().as_slice(),
        &[1.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(DoubleTensor::eye(0).unwrap().shape(), &[0, 0]);
}

#[test]
fn tensor_multiply_over_empty_axes_gives_zeros() {
    let a = DoubleTensor::new(&[2, 0], vec![]).unwrap();
    let b = DoubleTensor::new(&[0, 3], vec![]).unwrap();
    let product = a.tensor_multiply(&b, &[1], &[0]).unwrap();
    assert_eq!(product.shape(), &[2, 3]);
    assert!(product.as_slice().iter().all(|&x| x == 0.0));
    // the same as matmul
    assert_eq!(product, a.matmul(&b).unwrap());

    // and with a non-empty contraction, results are unchanged
    let a = DoubleTensor::new(&[2, 3], (0..6).map(f64::from).collect()).unwrap();
    let b = DoubleTensor::new(&[3, 2], (0..6).map(f64::from).collect()).unwrap();
    assert_eq!(
        a.tensor_multiply(&b, &[1], &[0]).unwrap(),
        a.matmul(&b).unwrap()
    );
    // an empty kept dimension
    let c = DoubleTensor::new(&[0, 3], vec![]).unwrap();
    assert_eq!(c.tensor_multiply(&b, &[1], &[0]).unwrap().shape(), &[0, 2]);
}

#[test]
fn matmul_whose_result_length_overflows_is_an_error() {
    let n = 1usize << 33;
    let a = DoubleTensor::new(&[n, 0], vec![]).unwrap();
    let b = DoubleTensor::new(&[0, n], vec![]).unwrap();
    assert!(matches!(
        a.matmul(&b),
        Err(TensorError::IncompatibleShapes { .. })
    ));
    assert!(a.tensor_multiply(&b, &[1], &[0]).is_err());
}

#[test]
fn owned_scalar_operators_reuse_the_buffer() {
    let t = DoubleTensor::vector((0..100).map(f64::from).collect());
    let expected = -&(&(&(&(&t + 1.0) * 2.0) - 3.0) / 4.0);
    let pointer = t.as_slice().as_ptr();
    let result = -((((t + 1.0) * 2.0) - 3.0) / 4.0);
    assert_eq!(result, expected);
    assert_eq!(result.as_slice().as_ptr(), pointer);

    let i = IntTensor::vector((0..100).collect());
    let expected = -&(&(&(&i + 1) * 2) - 3);
    let pointer = i.as_slice().as_ptr();
    let result = -(((i + 1) * 2) - 3);
    assert_eq!(result, expected);
    assert_eq!(result.as_slice().as_ptr(), pointer);
    // integer arithmetic still wraps
    assert_eq!((IntTensor::scalar(i64::MAX) + 1).as_slice(), &[i64::MIN]);
    assert_eq!((-IntTensor::scalar(i64::MIN)).as_slice(), &[i64::MIN]);
}

#[test]
fn integer_division_results_and_errors_are_unchanged() {
    let a = IntTensor::new(&[2, 3], vec![7, -7, 8, 9, 10, i64::MIN]).unwrap();
    // same shape, broadcast and scalar operands
    let same = IntTensor::new(&[2, 3], vec![2, 2, 3, 4, 5, -1]).unwrap();
    assert_eq!(
        a.div(&same).unwrap().as_slice(),
        &[3, -3, 2, 2, 2, i64::MIN]
    );
    assert_eq!(a.rem(&same).unwrap().as_slice(), &[1, -1, 2, 1, 0, 0]);
    let row = IntTensor::vector(vec![2, 3, 4]);
    assert_eq!(
        a.div(&row).unwrap().as_slice(),
        &[3, -2, 2, 4, 3, i64::MIN / 4]
    );
    assert_eq!(
        a.pow(&IntTensor::scalar(2)).unwrap().as_slice(),
        &[49, 49, 64, 81, 100, 0]
    );
    // a zero divisor anywhere is an error, whatever the shapes
    let zero_last = IntTensor::new(&[2, 3], vec![1, 1, 1, 1, 1, 0]).unwrap();
    for divisor in [
        zero_last,
        IntTensor::vector(vec![1, 0, 1]),
        IntTensor::scalar(0),
    ] {
        assert_eq!(a.div(&divisor), Err(TensorError::DivisionByZero));
        assert_eq!(a.rem(&divisor), Err(TensorError::DivisionByZero));
    }
    assert_eq!(
        IntTensor::vector(vec![2, 0]).pow(&IntTensor::scalar(-1)),
        Err(TensorError::DivisionByZero)
    );
    // shape errors come before division errors
    assert!(matches!(
        a.div(&IntTensor::vector(vec![0, 0])),
        Err(TensorError::IncompatibleShapes { .. })
    ));
    // the first error in row-major order is the one returned
    let x = DoubleTensor::vector(vec![1.0, 2.0, 3.0]);
    let result = x.try_zip_with(&IntTensor::vector(vec![2, 0, 0]), |x, d| {
        if d == 0 {
            Err(TensorError::InvalidArgument(format!("{x}")))
        } else {
            Ok(x)
        }
    });
    assert_eq!(result, Err(TensorError::InvalidArgument("2".into())));
    let result = x.try_zip_with(&IntTensor::scalar(0), |x, _| {
        Err::<f64, _>(TensorError::InvalidArgument(format!("{x}")))
    });
    assert_eq!(result, Err(TensorError::InvalidArgument("1".into())));
}

#[test]
fn matmul_and_tensor_multiply_results_are_unchanged() {
    let a = DoubleTensor::new(&[2, 3, 4], (0..24).map(f64::from).collect()).unwrap();
    let b = DoubleTensor::new(&[4, 3, 5], (0..60).map(|i| f64::from(i % 7)).collect()).unwrap();
    let product = a.tensor_multiply(&b, &[1, 2], &[1, 0]).unwrap();
    assert_eq!(product.shape(), &[2, 5]);
    for i in 0..2 {
        for k in 0..5 {
            let mut expected = 0.0;
            for j in 0..3 {
                for l in 0..4 {
                    expected += a.get(&[i, j, l]).unwrap() * b.get(&[l, j, k]).unwrap();
                }
            }
            assert_eq!(product.get(&[i, k]).unwrap(), expected);
        }
    }
    // small batched products (the plain loop), with and without transposition
    let m = DoubleTensor::new(&[3, 2, 2], (0..12).map(f64::from).collect()).unwrap();
    let n = DoubleTensor::new(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(
        m.matmul(&n).unwrap().as_slice(),
        &[
            3.0, 4.0, 11.0, 16.0, 19.0, 28.0, 27.0, 40.0, 35.0, 52.0, 43.0, 64.0
        ]
    );
    assert_eq!(
        m.matmul_transposed(&n, true, true).unwrap(),
        m.transpose()
            .unwrap()
            .matmul(&n.transpose().unwrap())
            .unwrap()
    );
    // an empty contraction is zero
    let e = DoubleTensor::new(&[2, 0], vec![]).unwrap();
    let f = DoubleTensor::new(&[0, 3], vec![]).unwrap();
    assert_eq!(e.matmul(&f).unwrap(), DoubleTensor::zeros(&[2, 3]));
}

#[test]
fn triangle_part_of_batched_and_empty_matrices() {
    let m = DoubleTensor::new(&[2, 2, 2], (0..8).map(f64::from).collect()).unwrap();
    assert_eq!(
        m.triangle_part(true).unwrap().as_slice(),
        &[0.0, 1.0, 3.0, 4.0, 5.0, 7.0]
    );
    assert_eq!(
        m.triangle_part(false).unwrap().as_slice(),
        &[0.0, 2.0, 3.0, 4.0, 6.0, 7.0]
    );
    let empty = DoubleTensor::new(&[3, 0, 0], vec![]).unwrap();
    assert_eq!(empty.triangle_part(true).unwrap().shape(), &[3, 0]);
    let no_batch = DoubleTensor::new(&[0, 2, 2], vec![]).unwrap();
    assert_eq!(no_batch.triangle_part(true).unwrap().shape(), &[0, 3]);
}

#[test]
fn slices_allow_spaces_around_colons_and_a_trailing_comma() {
    use keanu_tensor::ops::shape_ops::{SliceItem, parse_slice};
    let t = DoubleTensor::new(&[2, 4], (0..8).map(f64::from).collect()).unwrap();
    assert_eq!(t.slice("1 : 3").unwrap(), t.slice("1:3").unwrap());
    assert_eq!(
        t.slice(" : , 1 : : 2 ").unwrap(),
        t.slice(":, 1::2").unwrap()
    );
    assert_eq!(t.slice("1,").unwrap(), t.slice("1").unwrap());
    assert_eq!(t.slice("1,").unwrap().shape(), &[4]);
    assert_eq!(t.slice(":, -1 ,").unwrap(), t.slice(":, -1").unwrap());
    assert_eq!(
        parse_slice(" 1 : 3 , ").unwrap(),
        parse_slice("1:3").unwrap()
    );
    assert_eq!(
        parse_slice("::-1,").unwrap(),
        vec![SliceItem::Range {
            start: None,
            stop: None,
            step: Some(-1)
        }]
    );
    // still errors: empty items other than one trailing one, an empty string, and spaces inside a number
    for bad in ["", ",", "1,,", ",1", "1, ,2", "1 2", "- 1"] {
        assert!(
            matches!(t.slice(bad), Err(TensorError::InvalidSlice(_))),
            "{bad:?}"
        );
    }
}
