//! Sequences, mirroring the semantics tested by Java's SequenceBuilderTest.

use keanu_graph::sequence::{SequenceBuilder, proxy_label_for};
use keanu_graph::{GraphBuilder, GraphError, Model, joint_log_prob};
use keanu_tensor::DoubleTensor;

#[test]
fn items_are_labelled_in_their_namespace_and_proxies_link_to_the_previous_item() {
    let b = GraphBuilder::new();
    let initial = b.scalar(1.0);
    let sequence = SequenceBuilder::new(&b)
        .with_initial_state("x", initial)
        .count(3, |item| {
            let previous = item.double_proxy_for("x");
            let x = b.exponential(previous);
            item.add("x", x);
            let y = b.poisson(x);
            item.add("y", y);
        })
        .unwrap();
    let initial = initial.id();
    let (graph, _) = b.build().unwrap();

    assert_eq!(sequence.len(), 3);
    assert_eq!(sequence.identifier(), 1);
    let mut previous = initial;
    for item in sequence.items() {
        let i = item.index();
        let proxy = item.get(&proxy_label_for("x")).unwrap();
        let x = item.get("x").unwrap();
        let y = item.get("y").unwrap();
        assert_eq!(
            graph[proxy].label(),
            Some(format!("Sequence_Item_{i}.1.proxy_for.x").as_str())
        );
        assert_eq!(
            graph[x].label(),
            Some(format!("Sequence_Item_{i}.1.x").as_str())
        );
        assert_eq!(graph[proxy].parents(), &[previous]);
        assert_eq!(graph[x].parents(), &[proxy]);
        assert_eq!(graph[y].parents(), &[x]);
        // the full label also works
        assert_eq!(item.get(&format!("Sequence_Item_{i}.1.x")), Some(x));
        previous = x;
    }
    assert_eq!(sequence.last_item().unwrap().get("x"), Some(previous));
}

#[test]
fn named_sequences_have_an_outer_namespace_and_identifiers_count_per_builder() {
    let b = GraphBuilder::new();
    let first = SequenceBuilder::new(&b)
        .count(1, |item| {
            item.add("x", b.scalar(1.0));
        })
        .unwrap();
    let second = SequenceBuilder::new(&b)
        .named("weather")
        .count(2, |item| {
            item.add("x", b.scalar(2.0));
        })
        .unwrap();
    let (graph, _) = b.build().unwrap();
    assert_eq!((first.identifier(), second.identifier()), (1, 2));
    assert_eq!(second.name(), Some("weather"));
    assert_eq!(
        graph[second.item(1).unwrap().get("x").unwrap()].label(),
        Some("weather.Sequence_Item_1.2.x")
    );
    // a second builder starts again from 1, so labels are reproducible
    let other = GraphBuilder::new();
    assert_eq!(
        SequenceBuilder::new(&other)
            .count(1, |_| {})
            .unwrap()
            .identifier(),
        1
    );
}

#[test]
fn sequences_can_be_built_from_data_with_a_transition_mapping() {
    let b = GraphBuilder::new();
    let start = b.scalar(0.0).label("start");
    let observations = [0.3, 0.9, 0.4];
    let sequence = SequenceBuilder::new(&b)
        .with_initial_state("position", start)
        .with_transition(&proxy_label_for("previousPosition"), "position")
        .from_iter(observations, |item, observation| {
            let previous = item.double_proxy_for("previousPosition");
            let position = b.gaussian(previous, 0.5);
            item.add("position", position);
            let reading = b.gaussian(position, 1.0);
            b.observe(reading, DoubleTensor::scalar(observation));
            item.add("reading", reading);
        })
        .unwrap();
    let (graph, mut state) = b.build().unwrap();
    assert_eq!(sequence.len(), 3);
    let positions: Vec<_> = sequence
        .items()
        .iter()
        .map(|i| i.get("position").unwrap())
        .collect();
    for (i, &p) in positions.iter().enumerate() {
        state
            .set(&graph, p, DoubleTensor::scalar(0.1 * i as f64))
            .unwrap();
    }
    let all: Vec<_> = graph.ids().collect();
    state
        .lazy_eval(
            &graph,
            &all,
            &mut keanu_distributions::KeanuRng::seed_from_u64(0),
        )
        .unwrap();
    let model = Model::new(graph.clone(), &state).unwrap();
    assert_eq!(model.latents(), positions.as_slice());
    assert!(
        (model.log_prob(&state).unwrap() - joint_log_prob(&graph, &state).unwrap()).abs() < 1e-12
    );
}

#[test]
fn construction_errors() {
    // no initial state for the first item's proxy
    let b = GraphBuilder::new();
    let result = SequenceBuilder::new(&b).count(2, |item| {
        let previous = item.double_proxy_for("x");
        item.add("x", previous + 1.0);
    });
    assert!(matches!(result, Err(GraphError::Sequence(_))));

    // the same label twice in one item
    let b = GraphBuilder::new();
    let result = SequenceBuilder::new(&b).count(1, |item| {
        item.add("x", b.scalar(1.0));
        item.add("x", b.scalar(2.0));
    });
    assert!(matches!(result, Err(GraphError::Sequence(_))));

    // a node already labelled by a sequence item
    let b = GraphBuilder::new();
    let result = SequenceBuilder::new(&b).count(1, |item| {
        let x = b.scalar(1.0).label("Sequence_Item_0.7.x");
        item.add("Sequence_Item_0.7.x", x);
    });
    assert!(matches!(result, Err(GraphError::Sequence(_))));
}
