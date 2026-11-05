//! Sampling marginalised models and recovering the marginalised latents: NUTS on a mixture whose assignments were
//! marginalised automatically, and the assignments' conditional probabilities against the exact ones.

mod mcmc_common;

use keanu_algorithms::marginal::recover_marginals;
use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::chains::sample_chains;
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_algorithms::mcmc::select::Sampler;
use keanu_algorithms::{AlgorithmError, ChainSamples};
use keanu_graph::{GraphBuilder, Marginalized, Model, NodeId, State, marginalize};
use keanu_tensor::{DoubleTensor, IntTensor};
use mcmc_common::*;

/// Draws from 0.4 N(-2, 1) + 0.6 N(3, 1).
fn mixture_data(count: usize, seed: u64) -> Vec<f64> {
    let mut r = rng(seed);
    (0..count)
        .map(|_| {
            let mean = if r.next_f64() < 0.4 { -2.0 } else { 3.0 };
            mean + r.next_gaussian()
        })
        .collect()
}

/// The nodes of the marginalised mixture.
struct Fit {
    model: Model,
    state: State,
    marginalized: Marginalized,
    /// The latents: the first mean, the gap to the second, and the weights.
    latents: [NodeId; 3],
    means: NodeId,
}

/// A two-component mixture with unit scales, ordered means and Dirichlet(2, 2) weights, written with an explicit
/// categorical assignment per point and marginalised automatically.
fn marginalised_mixture(data: &[f64]) -> Fit {
    let b = GraphBuilder::new();
    let first = b.gaussian_with_shape(&[1], 0.0, 10.0);
    let gap = b.half_gaussian_with_shape(&[1], 10.0);
    let means = b.concat(0, &[first, gap]).cumsum(0);
    let weights = b.dirichlet(DoubleTensor::vector(vec![2.0, 2.0]));
    let n = data.len();
    let z = b.categorical(&[n], &[weights.take(&[0]), weights.take(&[1])]);
    b.set_value(z, IntTensor::vector(vec![0; n]));
    let y = b.gaussian_with_shape(&[n], means.gather(0, z), 1.0);
    b.observe(y, DoubleTensor::vector(data.to_vec()));
    b.set_value(first, DoubleTensor::vector(vec![0.0]));
    b.set_value(gap, DoubleTensor::vector(vec![1.0]));
    b.set_value(weights, DoubleTensor::vector(vec![0.5, 0.5]));
    let (first, gap, weights, means, z) = (first.id(), gap.id(), weights.id(), means.id(), z.id());
    let (graph, state) = b.build().unwrap();
    let (graph, state, marginalized) = marginalize(&graph, &state, &[z]).unwrap();
    let map = |id: NodeId| marginalized.node(id).unwrap();
    let latents = [map(first), map(gap), map(weights)];
    let means = map(means);
    let model = Model::new(graph, &state).unwrap();
    Fit {
        model,
        state,
        marginalized,
        latents,
        means,
    }
}

fn normal_density(x: f64, mean: f64) -> f64 {
    (-(x - mean) * (x - mean) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// NUTS is the default sampler of the marginalised mixture (every latent is continuous), converges (R-hat below
/// 1.01), and the recovered assignment probabilities sum to 1 and equal the exact conditional posteriors
/// w_k N(yᵢ | μ_k, 1) / Σⱼ w_j N(yᵢ | μ_j, 1) at every draw; draws of the assignments follow them.
#[test]
fn nuts_samples_the_marginalised_mixture_and_the_assignments_are_recovered() {
    let data = mixture_data(60, 3);
    let fit = marginalised_mixture(&data);
    assert!(matches!(
        Sampler::for_model(&fit.model, &fit.state),
        Sampler::Nuts(_)
    ));
    let mut recorded = fit.latents.to_vec();
    recorded.push(fit.means);
    let nuts = Nuts::default().with_adapt_count(500);
    let options = SamplingOptions::default().with_drop(500);
    let chains: ChainSamples = sample_chains(4, 7, &fit.state, |state, r| {
        nuts.sample_with_statistics(&fit.model, state, &recorded, 1500, options, r)
            .map(|(samples, _)| samples)
    })
    .unwrap();
    for c in 0..2 {
        let rhat = chains.rhat(fit.means, &[c]).unwrap();
        assert!(rhat < 1.01, "mean {c}: R-hat {rhat}");
    }
    let (low, high) = (
        chains.summarize(fit.means, &[0]).unwrap(),
        chains.summarize(fit.means, &[1]).unwrap(),
    );
    assert!((low.mean + 2.0).abs() < 4.0 * low.sd, "{}", low.mean);
    assert!((high.mean - 3.0).abs() < 4.0 * high.sd, "{}", high.mean);

    let chain = &chains.chains()[0];
    let marginals = recover_marginals(&fit.model, &fit.state, &fit.marginalized, chain).unwrap();
    assert_eq!(marginals.len(), 1);
    let assignments = &marginals[0];
    assert_eq!(assignments.len(), chain.len());
    let means = chain.f64(fit.means).unwrap();
    let weights = chain.f64(fit.latents[2]).unwrap();
    for (draw, p) in assignments.probabilities().iter().enumerate() {
        assert_eq!(p.shape(), &[60, 2]);
        let mu = means.as_slice()[draw].as_slice();
        let w = weights.as_slice()[draw].as_slice();
        for (i, &y) in data.iter().enumerate() {
            let row = &p.as_slice()[2 * i..2 * i + 2];
            let (a, b) = (
                w[0] * normal_density(y, mu[0]),
                w[1] * normal_density(y, mu[1]),
            );
            assert!((row[0] + row[1] - 1.0).abs() < 1e-12);
            assert!(
                (row[0] - a / (a + b)).abs() < 1e-10,
                "draw {draw} point {i}"
            );
        }
    }
    // draws of the assignments: each element's frequency of component 1 against its mean probability
    let mean = assignments.mean().unwrap();
    let mut r = rng(11);
    let repeats = 20;
    let mut ones = vec![0.0; 60];
    for _ in 0..repeats {
        for z in assignments.draw(&mut r).unwrap() {
            for (count, &k) in ones.iter_mut().zip(z.as_i64().unwrap().as_slice()) {
                assert!(k == 0 || k == 1);
                *count += k as f64;
            }
        }
    }
    let trials = (repeats * assignments.len()) as f64;
    for (i, &count) in ones.iter().enumerate() {
        let p = mean.as_slice()[2 * i + 1];
        let tolerance = 5.0 * (p * (1.0 - p) / trials).sqrt() + 1e-9;
        assert!(
            (count / trials - p).abs() <= tolerance,
            "point {i}: frequency {} vs probability {p}",
            count / trials
        );
    }
}

/// Recovering needs every latent the conditional probabilities depend on.
#[test]
fn recovering_without_the_latents_is_an_error() {
    let data = mixture_data(10, 4);
    let fit = marginalised_mixture(&data);
    let mut samples =
        keanu_algorithms::NetworkSamples::new(fit.model.graph(), &[fit.means]).unwrap();
    let mut state = fit.state.clone();
    state
        .eval(fit.model.graph(), &[fit.means], &mut rng(1))
        .unwrap();
    samples.record(&state, 0.0).unwrap();
    let error = recover_marginals(&fit.model, &fit.state, &fit.marginalized, &samples).unwrap_err();
    assert!(
        matches!(&error, AlgorithmError::InvalidArgument(m) if m.contains("without samples")),
        "{error}"
    );
}
