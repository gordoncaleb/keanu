//! Load limits: a small file can describe a graph that is enormous (huge shapes) or would take practically forever
//! to evaluate (a scan of 2^40 steps). Loading checks every node against [`LoadOptions`], whose defaults protect
//! against untrusted files, and the limits can be raised for trusted ones.

#![cfg(feature = "protobuf")]

use keanu_graph::GraphBuilder;
use keanu_graph::op::Op;
use keanu_io::{IoError, LoadOptions, Metadata, load_protobuf, load_protobuf_with, save_protobuf};

/// The protobuf file of the graph `build` makes (nothing is evaluated).
fn file(build: impl FnOnce(&GraphBuilder)) -> Vec<u8> {
    let b = GraphBuilder::new();
    build(&b);
    let (graph, state) = b.build().unwrap();
    save_protobuf(&graph, &state, &Metadata::new()).unwrap()
}

fn rejected(result: Result<keanu_io::Loaded, IoError>) -> bool {
    matches!(result, Err(IoError::InvalidNode { .. }))
}

#[test]
fn a_scan_with_too_many_steps_is_rejected() {
    // a few bytes describing 2^40 steps over a scalar carry: within the element cap, but evaluating it would loop
    // for hours
    let bytes = file(|b| {
        b.scan_steps(1 << 40, b.scalar(1.0), &[], |_, x, _| x * 2.0);
    });
    assert!(bytes.len() < 1000);
    assert!(rejected(load_protobuf(&bytes)));
    // the default allows 2^24 steps
    let ok = file(|b| {
        b.scan_steps(1 << 24, b.scalar(1.0), &[], |_, x, _| x * 2.0);
    });
    assert!(load_protobuf(&ok).is_ok());
    // a trusted file can raise the limits
    // (the scan's output [2^40] also exceeds the default per-node limit)
    let mut options = LoadOptions::default();
    options.max_scan_steps = 1 << 40;
    options.max_scan_work = 1 << 50;
    options.max_elements = 1 << 40;
    assert!(load_protobuf_with(&bytes, &options).is_ok());
    options.max_scan_steps = (1 << 24) - 1;
    assert!(rejected(load_protobuf_with(&ok, &options)));
}

#[test]
fn a_scan_doing_too_much_work_is_rejected() {
    // 2^20 steps (under the step limit) of a body over [2^13] tensors: about 2^34 element operations
    let bytes = file(|b| {
        let init = b.constant(keanu_tensor::DoubleTensor::zeros(&[1 << 13]));
        b.scan_steps(1 << 20, init, &[], |_, x, _| (x * 2.0).exp());
    });
    assert!(rejected(load_protobuf(&bytes)));
    // (the scan's output [2^20, 2^13] also exceeds the default per-node limit)
    let mut options = LoadOptions::default();
    options.max_scan_work = 1 << 40;
    options.max_elements = 1 << 33;
    assert!(load_protobuf_with(&bytes, &options).is_ok());
    // nested scans multiply: 2^12 outer steps, each running 2^12 inner steps of a three node scalar body, is about
    // 3 × 2^24 work
    let nested = file(|b| {
        b.scan_steps(1 << 12, b.scalar(1.0), &[], |b, x, _| {
            let inner = b.scan_steps(1 << 12, x, &[], |_, y, _| y + 1.0);
            inner.slice("-1")
        });
    });
    assert!(load_protobuf(&nested).is_ok());
    options.max_scan_work = 1 << 20;
    assert!(rejected(load_protobuf_with(&nested, &options)));
}

#[test]
fn nodes_with_too_many_elements_are_rejected() {
    // a broadcast to [2^32] (32 GiB) is within the hard cap but over the default per-node limit of 2^31 elements
    let bytes = file(|b| {
        let x = b.scalar(1.0).id();
        b.add_node(Op::Broadcast(vec![1 << 32]), &[x]).unwrap();
    });
    assert!(rejected(load_protobuf(&bytes)));
    let mut options = LoadOptions::default();
    options.max_elements = 1 << 32;
    assert!(load_protobuf_with(&bytes, &options).is_ok());
    // the limit applies inside scan bodies too
    let scan = file(|b| {
        b.scan_steps(2, b.scalar(1.0), &[], |b, x, _| {
            let wide = b.double_var(b.add_node(Op::Broadcast(vec![1 << 20]), &[x.id()]).unwrap());
            wide.sum() + x
        });
    });
    assert!(load_protobuf(&scan).is_ok());
    options.max_elements = 1 << 19;
    assert!(rejected(load_protobuf_with(&scan, &options)));
}
