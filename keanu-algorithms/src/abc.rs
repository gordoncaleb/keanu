//! Approximate Bayesian computation (ABC): inference for models with
//! [simulator nodes](keanu_graph::GraphBuilder::simulator), whose data can be simulated but whose likelihood can't be
//! evaluated, by sequential Monte Carlo (SMC-ABC).
//!
//! [`Abc`] is a [`TemperedLikelihood`] for [`Smc`]. At each particle it simulates every observed simulator node
//! from its parameters' values, reduces the simulated and the observed data to summary statistics s (the data
//! itself by default), and replaces the node's log likelihood by a kernel of their difference scaled by ε:
//!
//! ```text
//! Gaussian:  −½ ‖(s(simulated) − s(observed)) / ε‖²        Laplace:  −‖(s(simulated) − s(observed)) / ε‖₁
//! ```
//!
//! The observed nodes that aren't simulators and the factors keep their exact log likelihood, which is added. This
//! follows PyMC's `pm.Simulator` with `pm.sample_smc` (where the kernel is called the distance): ε is fixed and the
//! tempering raises β from 0 to 1 as for any likelihood, so the stage at β targets the prior times the kernel to the
//! power β, which for the Gaussian kernel is the kernel with ε/√β. Each particle keeps the kernel value of the
//! simulation it was accepted with, and a move is accepted on a fresh simulation at the proposed point, so the
//! mutation is a pseudo-marginal Metropolis–Hastings step targeting the ABC posterior
//!
//! ```text
//! p_ε(θ | y) ∝ p(θ) · E_{x ~ simulator(θ)}[ K((s(x) − s(y)) / ε) ],
//! ```
//!
//! which approaches the exact posterior given s(y) as ε shrinks (and the exact posterior if the statistics are
//! sufficient). A smaller ε is more accurate and needs more stages and particles. The log marginal likelihood SMC
//! reports is the log of the ABC evidence, the prior times the expected kernel integrated over θ, with the kernels unnormalised as PyMC's: it
//! compares models only under the same kernel, ε and statistics.
//!
//! ```
//! use keanu_algorithms::abc::{Abc, SummaryStatistics};
//! use keanu_algorithms::smc::Smc;
//! use keanu_distributions::KeanuRng;
//! use keanu_graph::{DoubleVar, GraphBuilder, Model, Value};
//! use keanu_tensor::DoubleTensor;
//!
//! // mu ~ N(0, 1), and 20 points simulated from N(mu, 1), whose mean is observed to be 1
//! let b = GraphBuilder::new();
//! let mu = b.gaussian(0.0, 1.0);
//! let data: DoubleVar = b.simulator(&[mu.id()], &[20], |parameters: &[&Value], rng: &mut KeanuRng| {
//!     let mu = parameters[0].as_f64().ok_or("mu is f64")?.as_slice()[0];
//!     Ok(Value::F64(DoubleTensor::vector((0..20).map(|_| mu + rng.next_gaussian()).collect())))
//! });
//! b.observe(data, DoubleTensor::vector(vec![1.0; 20]));
//! let mu = mu.id();
//! let (graph, state) = b.build()?;
//! let model = Model::new(graph, &state)?;
//!
//! let abc = Abc::new(0.1).with_statistics(SummaryStatistics::Mean);
//! let result = Smc::default().with_particles(500).with_chains(2).sample_abc(&model, &state, &[mu], &abc, 1)?;
//! // the exact posterior given the mean is N(20/21, 1/21)
//! let mean = result.samples.chains()[0].f64(mu).unwrap().mean()?.to_scalar().unwrap();
//! assert!((mean - 20.0 / 21.0).abs() < 0.1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{AlgorithmError, Result};
use crate::smc::{Smc, SmcResult, TemperedLikelihood};
use keanu_distributions::KeanuRng;
use keanu_graph::op::{DistributionKind, Op};
use keanu_graph::{GraphError, Model, NodeId, State, Value};
use std::fmt;
use std::sync::Arc;

/// The ABC kernel: how the scaled difference between the simulated and the observed summary statistics becomes a
/// log likelihood (PyMC's `distance`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum AbcKernel {
    /// −½ ‖(s(simulated) − s(observed)) / ε‖², the default (PyMC's `"gaussian"`).
    #[default]
    Gaussian,
    /// −‖(s(simulated) − s(observed)) / ε‖₁ (PyMC's `"laplace"`), which penalises large differences less.
    Laplace,
}

/// A user function from a simulated or observed value to its summary statistics.
pub type SummaryFn = dyn Fn(&Value) -> std::result::Result<Vec<f64>, String> + Send + Sync;

/// The summary statistics ABC compares, computed from a simulator node's value (integers and booleans as numbers)
/// over all its elements (PyMC's `sum_stat`).
#[derive(Clone, Default)]
#[non_exhaustive]
pub enum SummaryStatistics {
    /// The data itself, its elements in row-major order (the default).
    #[default]
    Identity,
    /// The elements sorted in increasing order, which ignores their order (for exchangeable data).
    Sort,
    /// The mean of the elements.
    Mean,
    /// The median of the elements.
    Median,
    /// The variance of the elements (dividing by their number).
    Variance,
    /// A user function of the value (see [`custom`](Self::custom)). It must give the same number of statistics
    /// for the simulated and the observed value.
    Custom(Arc<SummaryFn>),
}

impl SummaryStatistics {
    /// Statistics computed by a user function of the value, which must be thread safe (parallel runs call it at
    /// once). A statistic of NaN makes the particle impossible.
    pub fn custom(
        function: impl Fn(&Value) -> std::result::Result<Vec<f64>, String> + Send + Sync + 'static,
    ) -> Self {
        Self::Custom(Arc::new(function))
    }

    /// The statistics of a value.
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidArgument`] if a user function fails.
    pub fn compute(&self, value: &Value) -> Result<Vec<f64>> {
        let data = || value.to_double_tensor().into_vec();
        Ok(match self {
            Self::Identity => data(),
            Self::Sort => {
                let mut x = data();
                x.sort_by(f64::total_cmp);
                x
            }
            Self::Mean => vec![mean(&data())],
            Self::Median => {
                let mut x = data();
                x.sort_by(f64::total_cmp);
                let n = x.len();
                vec![if n == 0 {
                    f64::NAN
                } else if n % 2 == 1 {
                    x[n / 2]
                } else {
                    0.5 * (x[n / 2 - 1] + x[n / 2])
                }]
            }
            Self::Variance => {
                let x = data();
                let m = mean(&x);
                vec![x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64]
            }
            Self::Custom(function) => function(value).map_err(|message| {
                AlgorithmError::InvalidArgument(format!(
                    "an ABC summary statistic failed: {message}"
                ))
            })?,
        })
    }
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

impl fmt::Debug for SummaryStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Identity => "Identity",
            Self::Sort => "Sort",
            Self::Mean => "Mean",
            Self::Median => "Median",
            Self::Variance => "Variance",
            Self::Custom(_) => "Custom",
        })
    }
}

/// The approximate likelihood of SMC-ABC (see the [module documentation](self)): a kernel of the difference between
/// summary statistics of data simulated at a particle and of the observations, for every observed simulator node,
/// plus the exact log likelihood of the model's other observed nodes and factors.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Abc {
    /// The kernel (default [`AbcKernel::Gaussian`]).
    pub kernel: AbcKernel,
    /// The scale ε of the differences: one for every statistic, or one per statistic (each simulator node must
    /// then have that many statistics). Each must be positive and finite.
    pub epsilon: Vec<f64>,
    /// The summary statistics (default [`SummaryStatistics::Identity`]).
    pub statistics: SummaryStatistics,
}

impl Abc {
    /// ABC with the Gaussian kernel, scale ε for every statistic, and the data itself as its statistics.
    pub fn new(epsilon: f64) -> Self {
        Self {
            kernel: AbcKernel::Gaussian,
            epsilon: vec![epsilon],
            statistics: SummaryStatistics::Identity,
        }
    }

    /// Sets [`kernel`](Self::kernel).
    #[must_use]
    pub fn with_kernel(mut self, kernel: AbcKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Sets [`epsilon`](Self::epsilon) to one scale for every statistic.
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = vec![epsilon];
        self
    }

    /// Sets [`epsilon`](Self::epsilon) to a scale per statistic, for statistics of different sizes.
    #[must_use]
    pub fn with_epsilons(mut self, epsilon: Vec<f64>) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Sets [`statistics`](Self::statistics).
    #[must_use]
    pub fn with_statistics(mut self, statistics: SummaryStatistics) -> Self {
        self.statistics = statistics;
        self
    }

    /// The kernel's log value for the statistics of one simulated and one observed value.
    fn kernel_value(&self, simulated: &[f64], observed: &[f64], node: &str) -> Result<f64> {
        if simulated.len() != observed.len() {
            return Err(AlgorithmError::InvalidArgument(format!(
                "the summary statistics of {node} have {} values for the simulated data and {} for the observed",
                simulated.len(),
                observed.len()
            )));
        }
        if self.epsilon.len() != 1 && self.epsilon.len() != simulated.len() {
            return Err(AlgorithmError::InvalidSetting(format!(
                "ABC has {} scales ε, but {node} has {} summary statistics",
                self.epsilon.len(),
                simulated.len()
            )));
        }
        let epsilon = |i: usize| self.epsilon[if self.epsilon.len() == 1 { 0 } else { i }];
        let scaled = simulated
            .iter()
            .zip(observed)
            .enumerate()
            .map(|(i, (s, o))| (s - o) / epsilon(i));
        Ok(match self.kernel {
            AbcKernel::Gaussian => -0.5 * scaled.map(|d| d * d).sum::<f64>(),
            AbcKernel::Laplace => -scaled.map(f64::abs).sum::<f64>(),
        })
    }
}

/// The model's observed nodes that aren't simulators, and its factors: the terms ABC evaluates exactly.
fn exact_terms(model: &Model) -> Vec<NodeId> {
    let graph = model.graph();
    model
        .observed()
        .iter()
        .chain(model.factors())
        .copied()
        .filter(|&id| !graph[id].is_simulator())
        .collect()
}

impl TemperedLikelihood for Abc {
    fn log_likelihood(&self, model: &Model, state: &mut State, rng: &mut KeanuRng) -> Result<f64> {
        let graph = model.graph();
        let mut total = model.local_log_prob(state, &exact_terms(model))?;
        if total == f64::NEG_INFINITY || total.is_nan() {
            return Ok(f64::NEG_INFINITY);
        }
        for &id in model.simulators() {
            let node = &graph[id];
            let Op::Distribution(op) = node.op() else {
                unreachable!("a simulator node is probabilistic")
            };
            let DistributionKind::Simulator(simulator) = &op.kind else {
                unreachable!("a simulator node has a simulator distribution")
            };
            let parameters: Vec<&Value> = node
                .parents()
                .iter()
                .map(|&p| state.value(p).ok_or(GraphError::MissingValue(p)))
                .collect::<std::result::Result<_, _>>()?;
            let simulated = simulator.simulate(op.shape.as_slice(), &parameters, rng)?;
            let observed = state.value(id).ok_or(GraphError::MissingValue(id))?;
            total += self.kernel_value(
                &self.statistics.compute(&simulated)?,
                &self.statistics.compute(observed)?,
                &graph.describe(id),
            )?;
        }
        Ok(total)
    }

    /// Checks the settings, and that the model has at least one simulator node and every one is observed.
    fn check(&self, model: &Model) -> Result<()> {
        if self.epsilon.is_empty() || self.epsilon.iter().any(|&e| !(e > 0.0 && e.is_finite())) {
            return Err(AlgorithmError::InvalidSetting(
                "every ABC scale ε must be positive and finite, and there must be at least one"
                    .into(),
            ));
        }
        let graph = model.graph();
        if model.simulators().is_empty() {
            return Err(AlgorithmError::InvalidArgument(
                "ABC needs a model with at least one observed simulator node".into(),
            ));
        }
        if let Some(&id) = model.latents().iter().find(|&&id| graph[id].is_simulator()) {
            return Err(AlgorithmError::InvalidArgument(format!(
                "{} is a simulator node that is not observed; ABC infers the parameters of observed simulators \
                 only",
                graph.describe(id)
            )));
        }
        Ok(())
    }
}

impl Smc {
    /// SMC-ABC: [`sample_with`](Self::sample_with) the approximate likelihood `abc` (see the
    /// [`abc`](crate::abc) module).
    ///
    /// # Errors
    ///
    /// As [`sample_with`](Self::sample_with); [`AlgorithmError::InvalidSetting`] for a scale ε that isn't positive
    /// and finite, or a number of scales that matches neither 1 nor a node's statistics;
    /// [`AlgorithmError::InvalidArgument`] if the model has no simulator node, a simulator node is not observed,
    /// the model has discrete latents (which SMC mutates over their Markov blankets, and a simulator has no terms
    /// there), a summary statistic fails or gives different numbers of values for the simulated and the observed
    /// data; and [`GraphError::Simulator`] if a simulator's function fails.
    pub fn sample_abc(
        &self,
        model: &Model,
        state: &State,
        nodes: &[NodeId],
        abc: &Abc,
        seed: u64,
    ) -> Result<SmcResult> {
        self.sample_with(model, state, nodes, abc, seed)
    }
}
