//! Property tests for the shape logic: broadcasting, slicing, permuting and reducing.

use keanu_tensor::{DoubleTensor, IntTensor, broadcast_shape};
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
        let v = DoubleTensor::arange(0.0, len as f64).unwrap();
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

    /// Scatter-add is the adjoint of gather: <gather(x), u> = <x, scatter_add(u)> for any axis and indices,
    /// including repeated ones.
    #[test]
    fn scatter_add_is_the_adjoint_of_gather(
        x in shape(3, 3).prop_filter("rank >= 1", |s| !s.is_empty()).prop_flat_map(tensor_with_shape),
        axis_seed in 0usize..3,
        index_seed in prop::collection::vec(0usize..100, 0..4),
        index_rank in 0usize..3,
    ) {
        let axis = axis_seed % x.rank();
        let n = x.shape()[axis];
        let mut idx: Vec<i64> = index_seed.iter().map(|&i| (i % n) as i64).collect();
        let index_shape: Vec<usize> = match index_rank {
            0 => { idx.truncate(1); if idx.is_empty() { idx.push(0); } vec![] }
            1 => vec![idx.len()],
            _ => vec![1, idx.len()],
        };
        let indices = IntTensor::new(&index_shape, idx).unwrap();
        let gathered = x.gather(axis as isize, &indices).unwrap();
        let u = DoubleTensor::new(gathered.shape(), (0..gathered.len()).map(|i| (i as f64) * 0.5 - 1.0).collect()).unwrap();
        let scattered = u.scatter_add(axis as isize, &indices, n).unwrap();
        prop_assert_eq!(scattered.shape(), x.shape());
        let left: f64 = gathered.mul(&u).unwrap().sum();
        let right: f64 = x.mul(&scattered).unwrap().sum();
        prop_assert!((left - right).abs() <= 1e-9 * left.abs().max(1.0), "{} != {}", left, right);
    }

    #[test]
    fn stack_then_slice_recovers_the_parts(a in tensor_with_shape(vec![2, 3]), b in tensor_with_shape(vec![2, 3])) {
        let stacked = DoubleTensor::stack(1, &[&a, &b]).unwrap();
        prop_assert_eq!(stacked.shape(), &[2, 2, 3]);
        prop_assert_eq!(stacked.slice(":, 0").unwrap(), a.clone());
        prop_assert_eq!(stacked.slice(":, 1").unwrap(), b.clone());
        let last = DoubleTensor::stack(-1, &[&a, &b]).unwrap();
        prop_assert_eq!(last.shape(), &[2, 3, 2]);
        prop_assert_eq!(last.slice("..., 0").unwrap(), a);
    }
}

#[test]
fn gather_selects_along_an_axis() {
    let x = DoubleTensor::new(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let idx = IntTensor::vector(vec![2, 0, 2, 1]);
    let g = x.gather(1, &idx).unwrap();
    assert_eq!(g.shape(), &[2, 4]);
    assert_eq!(g.as_slice(), &[3.0, 1.0, 3.0, 2.0, 6.0, 4.0, 6.0, 5.0]);
    let rows = x
        .gather(-2, &IntTensor::new(&[1, 2], vec![1, 1]).unwrap())
        .unwrap();
    assert_eq!(rows.shape(), &[1, 2, 3]);
    let empty = x.gather(0, &IntTensor::vector(vec![])).unwrap();
    assert_eq!(empty.shape(), &[0, 3]);
    assert!(x.gather(1, &IntTensor::vector(vec![3])).is_err());
    assert!(x.gather(1, &IntTensor::vector(vec![-1])).is_err());
    assert!(x.gather(2, &idx).is_err());
    assert!(DoubleTensor::scalar(1.0).gather(0, &idx).is_err());
}

#[test]
fn scatter_add_accumulates_repeated_indices() {
    let u = DoubleTensor::vector(vec![1.0, 2.0, 3.0, 4.0]);
    let idx = IntTensor::vector(vec![0, 2, 2, 1]);
    assert_eq!(
        u.scatter_add(0, &idx, 3).unwrap().as_slice(),
        &[1.0, 4.0, 5.0]
    );
    assert!(u.scatter_add(0, &idx, 2).is_err());
    assert!(u.scatter_add(0, &IntTensor::vector(vec![0, 1]), 3).is_err());
    assert!(u.scatter_add(1, &idx, 3).is_err());
    let ints = IntTensor::vector(vec![1, 1]);
    assert_eq!(
        ints.scatter_add(0, &IntTensor::vector(vec![1, 1]), 2)
            .unwrap()
            .as_slice(),
        &[0, 2]
    );
}

#[test]
fn stack_needs_matching_shapes() {
    let a = DoubleTensor::vector(vec![1.0, 2.0]);
    let b = DoubleTensor::vector(vec![1.0, 2.0, 3.0]);
    assert!(DoubleTensor::stack(0, &[&a, &b]).is_err());
    assert!(DoubleTensor::stack(0, &[]).is_err());
    assert!(DoubleTensor::stack(2, &[&a]).is_err());
    let s = DoubleTensor::stack(0, &[&a, &a, &a]).unwrap();
    assert_eq!(s.shape(), &[3, 2]);
}
