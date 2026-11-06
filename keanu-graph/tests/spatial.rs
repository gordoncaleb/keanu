//! ICAR, proper CAR and BYM2 nodes: their log probabilities are the distributions', their draws are in their
//! supports, and misuse is an error.

use keanu_distributions::{Adjacency, Car, Icar, KeanuRng};
use keanu_graph::{GraphBuilder, GraphError, Model, Support, node_support};
use keanu_tensor::DoubleTensor;
use std::sync::Arc;

fn evaluate(graph: &keanu_graph::Graph, state: &mut keanu_graph::State, seed: u64) {
    let all: Vec<_> = graph.ids().collect();
    state
        .lazy_eval(graph, &all, &mut KeanuRng::seed_from_u64(seed))
        .unwrap();
}

#[test]
fn icar_and_car_nodes_have_the_distributions_log_probabilities_and_draws_in_their_supports() {
    let islands = Arc::new(Adjacency::from_edges(6, &[(0, 1), (1, 2), (3, 4)]).unwrap());
    let ring = Arc::new(Adjacency::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]).unwrap());
    let b = GraphBuilder::new();
    let sigma = b.half_gaussian(1.0);
    b.set_value(sigma, DoubleTensor::scalar(0.7));
    let phi = b.icar(islands.clone(), sigma);
    let car = b.car(ring.clone(), 2.0, 0.4).unwrap();
    assert_eq!(phi.shape(), vec![6]);
    assert_eq!(car.shape(), vec![4]);
    let (phi, car) = (phi.id(), car.id());
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 2);
    let x = state.f64(phi).unwrap().clone();
    assert!((x.as_slice()[0] + x.as_slice()[1] + x.as_slice()[2]).abs() < 1e-12);
    assert!((x.as_slice()[3] + x.as_slice()[4]).abs() < 1e-12);
    assert_eq!(
        node_support(&graph, &state, phi).unwrap(),
        Support::SumToZeroComponents(vec![vec![0, 1, 2], vec![3, 4]].into())
    );
    assert_eq!(node_support(&graph, &state, car).unwrap(), Support::Real);

    let model = Model::new(graph.clone(), &state).unwrap();
    let expected = keanu_distributions::HalfGaussian::new(DoubleTensor::scalar(1.0))
        .log_prob(&DoubleTensor::scalar(0.7))
        .unwrap()
        .as_slice()[0]
        + Icar::new(&islands, DoubleTensor::scalar(0.7))
            .log_prob(&x)
            .unwrap()
        + Car::new(&ring, DoubleTensor::scalar(2.0), DoubleTensor::scalar(0.4))
            .log_prob(state.f64(car).unwrap())
            .unwrap();
    let log_prob = model.log_prob(&state).unwrap();
    assert!(
        (log_prob - expected).abs() < 1e-12,
        "{log_prob} != {expected}"
    );

    // a value off the ICAR's support is impossible
    let mut off = x.into_vec();
    off[0] += 0.5;
    state
        .set_and_cascade(&graph, phi, DoubleTensor::vector(off))
        .unwrap();
    assert_eq!(model.log_prob(&state).unwrap(), f64::NEG_INFINITY);
}

#[test]
fn bym2_combines_a_scaled_icar_and_independent_normals() {
    let adjacency = Arc::new(Adjacency::lattice(3, 4).unwrap());
    let b = GraphBuilder::new();
    let bym2 = b.bym2(adjacency.clone(), 0.8, 0.3).unwrap();
    let (spatial, unstructured, effect) =
        (bym2.spatial.id(), bym2.unstructured.id(), bym2.effect.id());
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 5);
    let phi = state.f64(spatial).unwrap().as_slice();
    let theta = state.f64(unstructured).unwrap().as_slice();
    let s = adjacency.scaling_factors().unwrap();
    for i in 0..12 {
        let expected = 0.8 * ((0.3 / s[i]).sqrt() * phi[i] + 0.7f64.sqrt() * theta[i]);
        assert!((state.f64(effect).unwrap().as_slice()[i] - expected).abs() < 1e-12);
    }
}

#[test]
fn misuse_is_an_error() {
    // a proper CAR needs every area to have a neighbour
    let b = GraphBuilder::new();
    let islands = Adjacency::from_edges(3, &[(0, 1)]).unwrap();
    assert!(matches!(
        b.car(islands, 1.0, 0.5),
        Err(GraphError::Distribution(_))
    ));
    // the parameters are scalars
    let b = GraphBuilder::new();
    b.icar(
        Adjacency::lattice(2, 2).unwrap(),
        DoubleTensor::vector(vec![1.0, 2.0]),
    );
    assert!(matches!(
        b.build(),
        Err(GraphError::InvalidNode { op: "icar", .. })
    ));
    // a value of the wrong shape is rejected by the node
    let b = GraphBuilder::new();
    let phi = b.icar(Adjacency::lattice(2, 2).unwrap(), 1.0);
    b.set_value(phi, DoubleTensor::zeros(&[3]));
    assert!(b.build().is_err());
}
