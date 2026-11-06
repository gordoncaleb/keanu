//! NUTS on the spatial random effects: the ICAR's hard sum-to-zero constraint holds on every draw, and a BYM2 model
//! recovers its scale and spatial proportion.

use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::chains::sample_chains;
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_distributions::{Adjacency, Icar, KeanuRng};
use keanu_graph::{GraphBuilder, Model};
use keanu_tensor::DoubleTensor;
use std::sync::Arc;

/// AE1: NUTS draws of an ICAR on a 10 × 10 lattice, with a second component and an island, sum to zero within
/// each connected component.
#[test]
fn nuts_draws_of_an_icar_sum_to_zero_within_each_component() {
    // the lattice (areas 0..100), a path 100 - 101 - 102 and an island 103
    let lattice = Adjacency::lattice(10, 10).unwrap();
    let mut edges = lattice.edges().to_vec();
    edges.extend([(100, 101), (101, 102)]);
    let adjacency = Arc::new(Adjacency::from_edges(104, &edges).unwrap());
    let mut rng = KeanuRng::seed_from_u64(3);
    let truth = Icar::new(&adjacency, DoubleTensor::scalar(0.5))
        .sample(&[104], &mut rng)
        .unwrap();
    let y: Vec<f64> = truth
        .as_slice()
        .iter()
        .map(|v| v + 0.3 * rng.next_gaussian())
        .collect();

    let b = GraphBuilder::new();
    let phi = b.icar(adjacency.clone(), 0.5);
    b.set_value(phi, DoubleTensor::zeros(&[104]));
    let observed = b.gaussian(phi, 0.3);
    b.observe(observed, DoubleTensor::vector(y));
    let phi = phi.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let nuts = Nuts::default().with_adapt_count(150);
    let samples = sample_chains(2, 7, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &[phi],
            250,
            SamplingOptions::default().with_drop(150),
            rng,
        )
        .map(|(samples, _)| samples)
    })
    .unwrap();
    for chain in samples.chains() {
        for draw in chain.iter() {
            let x = draw.f64(phi).unwrap().as_slice();
            for group in adjacency.connected_groups() {
                let sum: f64 = group.iter().map(|&i| x[i]).sum();
                assert!(sum.abs() < 1e-10, "a component sums to {sum}");
            }
            // the island is free
            assert!(x[103] != 0.0);
        }
    }
}

/// A BYM2 model of Gaussian observations on a 12 × 12 lattice, with data simulated from known σ and ρ: NUTS
/// recovers both within their 90% intervals.
#[test]
fn nuts_recovers_the_scale_and_spatial_proportion_of_a_bym2_effect() {
    let adjacency = Arc::new(Adjacency::lattice(12, 12).unwrap());
    let n = adjacency.len();
    let (sigma_true, rho_true, noise) = (1.0, 0.7, 0.2);
    let mut rng = KeanuRng::seed_from_u64(11);
    let phi = Icar::new(&adjacency, DoubleTensor::scalar(1.0))
        .sample(&[n], &mut rng)
        .unwrap();
    let scaling = adjacency.scaling_factors().unwrap();
    let effect: Vec<f64> = (0..n)
        .map(|i| {
            sigma_true
                * ((rho_true / scaling[i]).sqrt() * phi.as_slice()[i]
                    + (1.0 - rho_true).sqrt() * rng.next_gaussian())
        })
        .collect();
    // two observations per area separate the effect from the noise
    let y: Vec<f64> = effect
        .iter()
        .flat_map(|&e| [e, e])
        .map(|e| e + noise * rng.next_gaussian())
        .collect();

    let b = GraphBuilder::new();
    let sigma = b.half_gaussian(1.0);
    let rho = b.beta(1.0, 1.0);
    b.set_value(sigma, DoubleTensor::scalar(0.5));
    b.set_value(rho, DoubleTensor::scalar(0.5));
    let bym2 = b.bym2(adjacency.clone(), sigma, rho).unwrap();
    b.set_value(bym2.spatial, DoubleTensor::zeros(&[n]));
    b.set_value(bym2.unstructured, DoubleTensor::zeros(&[n]));
    let mean = bym2.effect.reshape(&[n as isize, 1]).broadcast_to(&[n, 2]);
    let observed = b.gaussian(mean, noise);
    b.observe(observed, DoubleTensor::new(&[n, 2], y).unwrap());
    let (sigma, rho) = (sigma.id(), rho.id());
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let nuts = Nuts::default().with_adapt_count(400);
    let samples = sample_chains(4, 21, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &[sigma, rho],
            1000,
            SamplingOptions::default().with_drop(400),
            rng,
        )
        .map(|(samples, _)| samples)
    })
    .unwrap();
    for (id, truth, name) in [(sigma, sigma_true, "sigma"), (rho, rho_true, "rho")] {
        let s = samples.summarize(id, &[]).unwrap();
        assert!(s.r_hat < 1.02, "R-hat of {name} is {}", s.r_hat);
        assert!(
            s.q5 < truth && truth < s.q95,
            "{name}: 90% interval [{}, {}] misses {truth}",
            s.q5,
            s.q95
        );
    }
}
