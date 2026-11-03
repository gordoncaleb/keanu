//! Running several independent chains over one model.

use crate::error::Result;
use keanu_distributions::KeanuRng;
use keanu_graph::State;

/// Runs `chains` independent chains, each with its own copy of `state` and its own random number generator seeded
/// from `seed` and the chain's index, and returns their results in chain order.
///
/// The model and graph are shared (they are `Send + Sync`). With the `parallel` feature the chains run on rayon's
/// thread pool; otherwise one after another. Either way the results are the same for the same seed.
///
/// ```
/// use keanu_algorithms::mcmc::chains::run_chains;
/// use keanu_algorithms::mcmc::nuts::Nuts;
/// use keanu_graph::{GraphBuilder, Model};
/// use keanu_tensor::DoubleTensor;
///
/// let b = GraphBuilder::new();
/// let mu = b.gaussian(0.0, 1.0);
/// let x = b.gaussian(mu, 1.0);
/// b.observe(x, DoubleTensor::scalar(1.0));
/// let mu = mu.id();
/// let (graph, state) = b.build()?;
/// let model = Model::new(graph, &state)?;
/// let nuts = Nuts::default().with_adapt_count(50);
/// let means = run_chains(4, 7, &state, |state, rng| {
///     let samples = nuts.sample(&model, state, &[mu], 200, rng)?;
///     Ok(samples.f64(mu).unwrap().mean()?.to_scalar().unwrap())
/// })?;
/// assert_eq!(means.len(), 4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn run_chains<T, F>(chains: usize, seed: u64, state: &State, chain: F) -> Result<Vec<T>>
where
    T: Send,
    F: Fn(&mut State, &mut KeanuRng) -> Result<T> + Sync,
{
    let run = |index: usize| -> Result<T> {
        let mut state = state.clone();
        let mut rng = KeanuRng::seed_from_u64(chain_seed(seed, index));
        chain(&mut state, &mut rng)
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..chains).into_par_iter().map(run).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..chains).map(run).collect()
    }
}

/// A well spread seed for each chain (splitmix64 of the seed and index).
fn chain_seed(seed: u64, index: usize) -> u64 {
    let mut z = seed.wrapping_add((index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
