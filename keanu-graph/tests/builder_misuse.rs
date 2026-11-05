//! Builder misuse is reported as an error, never a panic or a silently wrong graph: handles from another builder,
//! failed untyped construction, and ops with the wrong number of parents.

use keanu_distributions::KeanuRng;
use keanu_graph::op::{BinaryOp, DistributionKind, Op, Predicate, UnaryOp};
use keanu_graph::{DType, GraphBuilder, GraphError, Value};
use keanu_tensor::DoubleTensor;

#[test]
fn a_handle_from_another_builder_fails_the_build() {
    // the handles share the lifetime 'b, so this compiles; it used to wire b1's node #0 (x) as the right operand
    // and evaluate to 11
    let (b1, b2) = (GraphBuilder::new(), GraphBuilder::new());
    let x = b1.scalar(10.0);
    let y = b2.scalar(100.0);
    let _sum = x + y;
    assert_eq!(b1.error(), Some(GraphError::WrongBuilder(y.id())));
    assert!(matches!(b1.build(), Err(GraphError::WrongBuilder(_))));
    // the other builder is untouched
    assert!(b2.build().is_ok());
}

#[test]
fn a_foreign_handle_out_of_range_is_an_error_not_a_panic() {
    let (b1, b2) = (GraphBuilder::new(), GraphBuilder::new());
    for _ in 0..5 {
        b2.scalar(1.0);
    }
    let far = b2.scalar(2.0);
    let flag = b2.bool_scalar(true);
    let x = b1.scalar(1.0);
    let _ = x * far;
    let _ = b1.gaussian(far, 1.0);
    let _ = b1.where_(flag, x, x);
    let _ = b1.multiplexer(b1.int_scalar(0), &[x, far]);
    let _ = b1.concat(0, &[x.reshape(&[1]), far.reshape(&[1])]);
    b1.set_value(far, DoubleTensor::scalar(3.0));
    b1.observe(flag, keanu_tensor::BoolTensor::scalar(true));
    let proxy = b1.double_proxy(&[]);
    b1.connect(proxy, far);
    assert!(matches!(b1.build(), Err(GraphError::WrongBuilder(_))));
}

#[test]
fn every_handle_taking_method_checks_the_builder() {
    type Misuse = fn(&GraphBuilder, &GraphBuilder);
    let cases: &[(&str, Misuse)] = &[
        ("binary", |a, b| {
            let _ = a.scalar(1.0) - b.scalar(1.0);
        }),
        ("distribution", |a, b| {
            let _ = a.gaussian(0.0, b.scalar(1.0));
        }),
        ("where", |a, b| {
            let _ = a.where_(a.bool_scalar(true), a.scalar(1.0), b.scalar(2.0));
        }),
        ("cpt", |a, b| {
            let c = a.bool_scalar(true).id();
            let _ = a.cpt(&[c], &[(vec![1], b.scalar(1.0))], a.scalar(2.0));
        }),
        ("set_value", |a, b| {
            a.set_value(b.scalar(1.0), DoubleTensor::scalar(2.0))
        }),
        ("logic", |a, b| {
            let _ = a.bool_scalar(true) & b.bool_scalar(false);
        }),
    ];
    for (name, misuse) in cases {
        let (a, b) = (GraphBuilder::new(), GraphBuilder::new());
        b.scalar(0.0);
        misuse(&a, &b);
        assert!(
            matches!(a.build(), Err(GraphError::WrongBuilder(_))),
            "{name}"
        );
    }
}

#[test]
fn a_failed_add_distribution_leaves_no_node() {
    let b = GraphBuilder::new();
    let mu = b.constant(DoubleTensor::vector(vec![1.0, 2.0, 3.0])).id();
    let sigma = b.constant(DoubleTensor::vector(vec![1.0, 2.0])).id();
    let before = b.node_count();
    assert!(
        b.add_distribution(DistributionKind::Gaussian, None, &[mu, sigma])
            .is_err()
    );
    assert_eq!(b.node_count(), before, "no placeholder node is left behind");
    let ok = b
        .add_distribution(DistributionKind::Gaussian, None, &[mu, mu])
        .unwrap();
    assert_eq!(ok.index(), before, "ids are not shifted");
    let (graph, _) = b.build().unwrap();
    assert_eq!(graph.len(), before + 1);
}

#[test]
fn untyped_errors_are_returned_even_when_an_earlier_error_is_pending() {
    let b = GraphBuilder::new();
    let a = b.constant(DoubleTensor::vector(vec![1.0, 2.0, 3.0]));
    let c = b.constant(DoubleTensor::vector(vec![1.0, 2.0]));
    let _ = a + c; // records a deferred error
    let before = b.node_count();
    assert!(
        b.add_distribution(DistributionKind::Gaussian, None, &[a.id(), c.id()])
            .is_err()
    );
    assert_eq!(b.node_count(), before);
    assert!(matches!(
        b.set_value_of(a.id(), Value::F64(DoubleTensor::scalar(1.0))),
        Err(GraphError::InvalidValue(_))
    ));
    assert!(matches!(
        b.observe_of(
            a.id(),
            Value::F64(DoubleTensor::vector(vec![1.0, 2.0, 3.0]))
        ),
        Err(GraphError::NotObservable(_))
    ));
    let proxy = b.add_proxy(DType::F64, &[]).unwrap();
    assert!(matches!(
        b.connect_ids(proxy, a.id()),
        Err(GraphError::Proxy(_))
    ));
    // ... and the earlier error still fails the build
    assert!(matches!(b.build(), Err(GraphError::Tensor(_))));
}

#[test]
fn untyped_calls_with_unknown_nodes_are_errors() {
    let b = GraphBuilder::new();
    let x = b.scalar(1.0).id();
    let unknown = keanu_graph::NodeId::from_index(10);
    assert!(
        b.set_value_of(unknown, Value::F64(DoubleTensor::scalar(1.0)))
            .is_err()
    );
    assert!(
        b.observe_of(unknown, Value::F64(DoubleTensor::scalar(1.0)))
            .is_err()
    );
    assert!(b.connect_ids(unknown, x).is_err());
    assert!(b.build().is_ok());
}

#[test]
fn add_node_checks_the_number_of_parents() {
    let b = GraphBuilder::new();
    let x = b.scalar(1.0).id();
    let flag = b.bool_scalar(true).id();
    let count = b.node_count();
    let no_parents = [
        Op::Proxy,
        Op::Unary(UnaryOp::Exp),
        Op::Not,
        Op::All,
        Op::Any,
        Op::Predicate(Predicate::IsNan),
        Op::Standardize,
        Op::Transpose,
        Op::Reshape(vec![1]),
        Op::Slice("0".into()),
        Op::Take(vec![]),
        Op::Inverse,
        Op::Determinant,
        Op::Cast(DType::F64),
        Op::ReplaceNan(0.0),
        Op::Assert(None),
        Op::Concat(0),
        Op::Binary(BinaryOp::Add),
        Op::Where,
        Op::SetWithMask,
        Op::Multiplexer,
    ];
    for op in no_parents {
        let name = op.name();
        assert!(
            matches!(b.add_node(op, &[]), Err(GraphError::InvalidNode { .. })),
            "{name} with no parents"
        );
    }
    let extra_parents = [
        (Op::Unary(UnaryOp::Exp), vec![x, x]),
        (Op::Not, vec![flag, flag]),
        (Op::All, vec![flag, flag]),
        (Op::Predicate(Predicate::IsNan), vec![x, x]),
        (Op::Cast(DType::I64), vec![x, x]),
        (Op::ReplaceNan(0.0), vec![x, x]),
        (Op::Assert(None), vec![flag, flag]),
        (Op::Binary(BinaryOp::Add), vec![x, x, x]),
        (Op::Where, vec![flag, x, x, x]),
        (Op::Constant(Value::F64(DoubleTensor::scalar(1.0))), vec![x]),
    ];
    for (op, parents) in extra_parents {
        let name = op.name();
        assert!(
            matches!(
                b.add_node(op, &parents),
                Err(GraphError::InvalidNode { .. })
            ),
            "{name} with {} parents",
            parents.len()
        );
    }
    assert_eq!(b.node_count(), count);
    // the right number still works
    let y = b.add_node(Op::Unary(UnaryOp::Exp), &[x]).unwrap();
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &[y], &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    assert_eq!(state.f64(y).unwrap().as_slice(), &[1f64.exp()]);
}

#[test]
fn duplicate_parents_give_a_single_child_entry() {
    let b = GraphBuilder::new();
    let x = b.scalar(2.0);
    let square = x * x;
    let flag = b.bool_scalar(true);
    let chosen = b.where_(flag, x, x);
    let cube = b.concat(0, &[x.reshape(&[1]), x.reshape(&[1]), x.reshape(&[1])]);
    let (x, square, chosen, cube) = (x.id(), square.id(), chosen.id(), cube.id());
    let (graph, _) = b.build().unwrap();
    let children = graph.children(x).unwrap();
    assert_eq!(children.len(), 5, "{children:?}");
    assert_eq!(children.iter().filter(|&&c| c == square).count(), 1);
    assert_eq!(children.iter().filter(|&&c| c == chosen).count(), 1);
    assert!(!children.contains(&cube));
    assert!(children.windows(2).all(|w| w[0] < w[1]));
}

#[test]
fn a_cpt_with_more_conditions_than_parents_is_an_error_not_a_panic() {
    let b = GraphBuilder::new();
    let x = b.scalar(1.0).id();
    // usize::MAX conditions used to overflow the arity check (and wrap to zero parents in release builds)
    for (conditions, parents) in [(usize::MAX, vec![]), (usize::MAX, vec![x]), (3, vec![x])] {
        let op = Op::Cpt {
            conditions,
            keys: vec![],
        };
        assert!(
            matches!(
                b.add_node(op, &parents),
                Err(GraphError::InvalidNode { .. })
            ),
            "{conditions} conditions, {} parents",
            parents.len()
        );
    }
}

#[test]
fn shapes_too_large_to_allocate_are_errors_not_panics() {
    let b = GraphBuilder::new();
    let x = b.scalar(1.0).id();
    let index = b
        .int_constant(keanu_tensor::IntTensor::vector(vec![0]))
        .id();
    let v = b.constant(DoubleTensor::vector(vec![1.0, 2.0, 3.0])).id();
    // 2^61 f64s are 2^64 bytes, more than isize::MAX
    let too_large = [
        (Op::Broadcast(vec![1 << 61]), vec![x]),
        (Op::Broadcast(vec![1 << 31, 1 << 30]), vec![x]),
        (
            Op::ScatterAdd {
                axis: 0,
                size: 1 << 61,
            },
            vec![v, index],
        ),
    ];
    for (op, parents) in too_large {
        let name = op.name();
        assert!(
            matches!(
                b.add_node(op, &parents),
                Err(GraphError::InvalidNode { .. })
            ),
            "{name}"
        );
    }
    // a scan of 2^62 steps over a scalar carry
    let body = GraphBuilder::new();
    let carry = body.double_proxy(&[]);
    let out = carry * 2.0;
    let (carry, out) = (carry.id(), out.id());
    let (graph, _) = body.build().unwrap();
    assert!(matches!(
        keanu_graph::op::ScanBody::new(graph, 1 << 62, carry, vec![], vec![], out),
        Err(GraphError::InvalidNode { .. })
    ));
}

#[test]
fn explicit_shapes_over_the_element_cap_are_errors() {
    // [2^31, 2^31] has 2^62 elements: it fits in a usize but no value of that size can be allocated, so the builder
    // rejects it as it does inferred shapes
    let huge = [1usize << 31, 1 << 31];
    let b = GraphBuilder::new();
    let mu = b.scalar(0.0).id();
    let sigma = b.scalar(1.0).id();
    assert!(matches!(
        b.add_distribution(DistributionKind::Gaussian, Some(&huge), &[mu, sigma]),
        Err(GraphError::InvalidNode { .. })
    ));
    assert!(matches!(
        b.add_proxy(DType::F64, &huge),
        Err(GraphError::InvalidNode { .. })
    ));
    // on error nothing is added, and shapes under the cap are still accepted
    assert_eq!(b.node_count(), 2);
    assert!(b.add_proxy(DType::F64, &[2, 3]).is_ok());
    assert!(b.build().is_ok());
    // the typed proxies record the error for build
    let b = GraphBuilder::new();
    let _ = b.double_proxy(&huge);
    assert!(matches!(b.build(), Err(GraphError::InvalidNode { .. })));
}
