//! Auto diff where the "of" or "with respect to" node is empty (shape with a 0 dimension). Found by the no-panic
//! property test: the partials are empty rather than a division by zero.

use keanu_distributions::KeanuRng;
use keanu_graph::{GraphBuilder, forward_partials, reverse_partials};
use keanu_tensor::DoubleTensor;

#[test]
fn partials_of_an_empty_output_through_a_broadcast() {
    let b = GraphBuilder::new();
    let empty = b.constant(DoubleTensor::new(&[0], vec![]).unwrap());
    let latent = b.gaussian(0.0, 1.0);
    // max broadcasts the scalar latent to the empty shape [0]
    let out = latent.reshape(&[]).maximum(empty);
    let (out, latent) = (out.id(), latent.id());
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<_> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(1))
        .unwrap();

    let reverse = reverse_partials(&graph, &state, out, &[latent]).unwrap();
    assert_eq!(reverse.get(latent).unwrap().shape(), &[0]);
    let forward = forward_partials(&graph, &state, latent, &[out]).unwrap();
    assert_eq!(forward.get(out).unwrap().shape(), &[0]);
}
