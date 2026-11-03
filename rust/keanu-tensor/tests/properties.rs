//! Property tests for the shape logic: broadcasting, slicing, permuting and reducing.

use keanu_tensor::{DoubleTensor, broadcast_shape};
use proptest::prelude::*;

fn shape(max_rank: usize, max_dim: usize) -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(1..=max_dim, 0..=max_rank)
}

fn tensor_with_shape(shape: Vec<usize>) -> impl Strategy<Value = DoubleTensor> {
    let len: usize = shape.iter().product();
    prop::collection::vec(-100i32..100, len).prop_map(move |v| {
        DoubleTensor::new(&shape, v.into_iter().map(f64::from).collect()).unwrap()
    })
}

/// Two shapes that broadcast together: start from a common shape and replace some trailing dims with 1 or drop
/// leading dims.
fn broadcastable_pair() -> impl Strategy<Value = (Vec<usize>, Vec<usize>)> {
    (
        shape(4, 4),
        prop::collection::vec(any::<bool>(), 4),
        0usize..=4,
    )
        .prop_map(|(full, ones, drop)| {
            let other: Vec<usize> = full
                .iter()
                .zip(&ones)
                .map(|(&d, &one)| if one { 1 } else { d })
                .collect();
            let drop = drop.min(other.len());
            (full, other[drop..].to_vec())
        })
}

proptest! {
    #[test]
    fn broadcast_add_matches_explicitly_broadcast_operands((left, right) in broadcastable_pair(), seed in 0u64..1000) {
        let a = DoubleTensor::new(&left, (0..left.iter().product::<usize>()).map(|i| (i as f64) + seed as f64).collect()).unwrap();
        let b = DoubleTensor::new(&right, (0..right.iter().product::<usize>()).map(|i| (i as f64) * 0.5).collect()).unwrap();
        let out = broadcast_shape(&left, &right).unwrap();

        let direct = a.add(&b).unwrap();
        let expanded = a.broadcast_to(&out).unwrap().add(&b.broadcast_to(&out).unwrap()).unwrap();
        prop_assert_eq!(direct, expanded);

        // broadcasting is symmetric in shape
        let reversed = b.add(&a).unwrap();
        prop_assert_eq!(reversed.shape(), out.as_slice());
    }

    #[test]
    fn slicing_a_vector_matches_a_naive_implementation(len in 0usize..12, start in -15isize..15, stop in -15isize..15, step in prop::sample::select(vec![-3isize, -2, -1, 1, 2, 3]), use_start: bool, use_stop: bool) {
        let v = DoubleTensor::arange(0.0, len as f64);
        let spec = format!("{}:{}:{}", if use_start { start.to_string() } else { String::new() }, if use_stop { stop.to_string() } else { String::new() }, step);
        let sliced = v.slice(&spec).unwrap();

        // python semantics, written out the slow way
        let n = len as isize;
        let norm = |x: isize| if x < 0 { x + n } else { x };
        let (lo, hi) = if step > 0 { (0, n) } else { (-1, n - 1) };
        let first = if use_start { norm(start).clamp(lo, hi) } else if step > 0 { 0 } else { n - 1 };
        let end = if use_stop { norm(stop).clamp(lo, hi) } else if step > 0 { n } else { -1 };
        let mut expected = Vec::new();
        let mut i = first;
        while (step > 0 && i < end) || (step < 0 && i > end) {
            expected.push(i as f64);
            i += step;
        }
        prop_assert_eq!(sliced.as_slice(), expected.as_slice(), "slice {}", spec);
    }

    #[test]
    fn permute_then_inverse_permute_is_identity(t in shape(4, 4).prop_flat_map(tensor_with_shape), seed in any::<u64>()) {
        let rank = t.rank();
        let mut permutation: Vec<usize> = (0..rank).collect();
        // deterministic shuffle from the seed
        for i in (1..rank).rev() {
            permutation.swap(i, (seed as usize).wrapping_mul(i + 7) % (i + 1));
        }
        let mut inverse = vec![0; rank];
        for (i, &p) in permutation.iter().enumerate() {
            inverse[p] = i;
        }
        prop_assert_eq!(t.permute(&permutation).unwrap().permute(&inverse).unwrap(), t);
    }

    #[test]
    fn summing_every_axis_equals_the_total(t in shape(4, 4).prop_flat_map(tensor_with_shape)) {
        let axes: Vec<isize> = (0..t.rank() as isize).collect();
        let by_axes = t.sum_axes(&axes).unwrap();
        prop_assert_eq!(by_axes.shape(), &[] as &[usize]);
        prop_assert_eq!(by_axes.as_slice()[0], t.sum());
    }

    #[test]
    fn reshape_round_trips(t in shape(4, 4).prop_flat_map(tensor_with_shape)) {
        let flat = t.reshape(&[-1]).unwrap();
        let back: Vec<isize> = t.shape().iter().map(|&d| d as isize).collect();
        prop_assert_eq!(flat.reshape(&back).unwrap(), t);
    }

    #[test]
    fn concat_then_slice_recovers_the_parts(a in tensor_with_shape(vec![2, 3]), b in tensor_with_shape(vec![1, 3])) {
        let joined = DoubleTensor::concat(0, &[&a, &b]).unwrap();
        prop_assert_eq!(joined.slice("0:2").unwrap(), a);
        prop_assert_eq!(joined.slice("2:3").unwrap(), b);
    }

    #[test]
    fn triangle_part_inverts_fill_triangular(n in 1usize..5, upper: bool, batch in 1usize..3) {
        let m = n * (n + 1) / 2;
        let v = DoubleTensor::new(&[batch, m], (0..batch * m).map(|i| i as f64 + 1.0).collect()).unwrap();
        let filled = v.fill_triangular(upper, !upper).unwrap();
        prop_assert_eq!(filled.triangle_part(upper).unwrap(), v);
    }
}
