//! A NUTS chain that advances a few transitions per call, so a browser worker can report progress between calls.

use keanu::algorithms::mcmc::SamplingAlgorithm;
use keanu::algorithms::mcmc::nuts::NutsSampler;
use keanu::prelude::*;
use std::mem::ManuallyDrop;

/// One NUTS chain over a model it owns: `warmup` adaptation transitions (discarded), then `draws` recorded ones.
pub struct ChainRun {
    // Declared before the model it borrows; `Drop` drops it first, then frees the model.
    sampler: ManuallyDrop<NutsSampler<'static>>,
    model: *mut Model,
    state: State,
    rng: KeanuRng,
    samples: NetworkSamples,
    warmup: usize,
    draws: usize,
    done: usize,
}

impl std::fmt::Debug for ChainRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChainRun")
            .field("warmup", &self.warmup)
            .field("draws", &self.draws)
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

impl ChainRun {
    /// Starts a chain recording `nodes`, adapting NUTS over the first `warmup` transitions.
    pub fn new(
        model: Model,
        mut state: State,
        nodes: &[NodeId],
        nuts: Nuts,
        warmup: usize,
        draws: usize,
        seed: u64,
    ) -> keanu::Result<Self> {
        let model: *mut Model = Box::into_raw(Box::new(model));
        // SAFETY: the model lives on the heap until `Drop`, which drops the sampler borrowing it first. It is never
        // mutated or moved while the sampler exists.
        let model_ref: &'static Model = unsafe { &*model };
        let mut rng = KeanuRng::seed_from_u64(seed);
        let sampler = match nuts
            .with_adapt_count(warmup)
            .sampler(model_ref, &mut state, nodes, &mut rng)
        {
            Ok(sampler) => sampler,
            Err(error) => {
                // SAFETY: nothing borrows the model.
                drop(unsafe { Box::from_raw(model) });
                return Err(error.into());
            }
        };
        let samples = NetworkSamples::new(model_ref.graph(), nodes)?;
        Ok(Self {
            sampler: ManuallyDrop::new(sampler),
            model,
            state,
            rng,
            samples,
            warmup,
            draws,
            done: 0,
        })
    }

    /// Runs up to `steps` more transitions. Returns whether the chain has finished.
    pub fn advance(&mut self, steps: usize) -> keanu::Result<bool> {
        for _ in 0..steps {
            if self.finished() {
                break;
            }
            self.sampler.step(&mut self.state, &mut self.rng)?;
            self.done += 1;
            if self.done > self.warmup {
                self.sampler.record(&self.state, &mut self.samples)?;
            }
        }
        Ok(self.finished())
    }

    /// Whether every transition has run.
    pub fn finished(&self) -> bool {
        self.done >= self.warmup + self.draws
    }

    /// The fraction of transitions run, from 0 to 1.
    pub fn progress(&self) -> f64 {
        self.done as f64 / (self.warmup + self.draws) as f64
    }

    /// Whether the chain is still adapting.
    pub fn in_warmup(&self) -> bool {
        self.done < self.warmup
    }

    /// The recorded draws so far.
    pub fn samples(&self) -> &NetworkSamples {
        &self.samples
    }

    /// Divergent transitions after adaptation.
    pub fn divergences(&self) -> usize {
        self.sampler.statistics().divergence_count()
    }

    /// The model.
    pub fn model(&self) -> &Model {
        // SAFETY: see `new`.
        unsafe { &*self.model }
    }

    /// A copy of the chain's generator, for posterior predictive simulation that leaves the chain, and the next
    /// simulation, unchanged.
    pub fn rng_copy(&self) -> KeanuRng {
        self.rng.clone()
    }
}

impl Drop for ChainRun {
    fn drop(&mut self) {
        // SAFETY: the sampler is dropped exactly once, here, before the model it borrows is freed.
        unsafe {
            ManuallyDrop::drop(&mut self.sampler);
            drop(Box::from_raw(self.model));
        }
    }
}

/// Each recorded draw of an f64 node, flattened.
pub fn draws(samples: &NetworkSamples, node: NodeId) -> Vec<Vec<f64>> {
    samples
        .f64(node)
        .map(|s| s.as_slice().iter().map(|t| t.as_slice().to_vec()).collect())
        .unwrap_or_default()
}

/// Each recorded draw of a scalar f64 node.
pub fn scalar_draws(samples: &NetworkSamples, node: NodeId) -> Vec<f64> {
    draws(samples, node).into_iter().map(|d| d[0]).collect()
}

/// The q-quantile (0 ≤ q ≤ 1) of `values`, by linear interpolation between order statistics.
pub fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let at = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (at.floor() as usize, at.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (at - lo as f64)
}

/// The mean of `values`.
pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// A summary of draws: the mean and a central 90% interval.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Interval {
    /// The mean.
    pub mean: f64,
    /// The 5% quantile.
    pub low: f64,
    /// The 95% quantile.
    pub high: f64,
}

impl Interval {
    /// The summary of `values`.
    pub fn of(values: &[f64]) -> Self {
        Self {
            mean: mean(values),
            low: quantile(values, 0.05),
            high: quantile(values, 0.95),
        }
    }
}

/// A histogram of `values` over `bins` equal bins from their minimum to maximum: the bin edges' lower ends and the
/// fraction of values in each.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Histogram {
    /// The lower edge of each bin.
    pub start: Vec<f64>,
    /// The bin width.
    pub width: f64,
    /// The fraction of values in each bin.
    pub fraction: Vec<f64>,
}

impl Histogram {
    /// The histogram of whole-number `counts`: one bin of width 1 for each count from 0 to the largest.
    pub fn of_counts(counts: &[usize]) -> Self {
        let largest = counts.iter().copied().max().unwrap_or(0);
        let mut bins = vec![0usize; largest + 1];
        for &c in counts {
            bins[c] += 1;
        }
        Self {
            start: (0..=largest).map(|c| c as f64).collect(),
            width: 1.0,
            fraction: bins
                .into_iter()
                .map(|b| b as f64 / counts.len().max(1) as f64)
                .collect(),
        }
    }

    /// The histogram of `values`.
    pub fn of(values: &[f64], bins: usize) -> Self {
        let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let width = if hi > lo {
            (hi - lo) / bins as f64
        } else {
            1.0
        };
        let mut counts = vec![0usize; bins];
        for &v in values {
            let i = (((v - lo) / width) as usize).min(bins - 1);
            counts[i] += 1;
        }
        Self {
            start: (0..bins).map(|i| lo + i as f64 * width).collect(),
            width,
            fraction: counts
                .into_iter()
                .map(|c| c as f64 / values.len() as f64)
                .collect(),
        }
    }
}
