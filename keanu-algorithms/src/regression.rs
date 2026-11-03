//! Linear and logistic regression, and fitting models to data (Java's `model` package).
//!
//! ```
//! use keanu_algorithms::regression::{RegressionModel, Regularization, coefficient_of_determination};
//! use keanu_tensor::DoubleTensor;
//!
//! // y = 2x + 1
//! let x = DoubleTensor::vector(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
//! let y = &(&x * 2.0) + 1.0;
//!
//! let model = RegressionModel::linear(&x, &y)?.fit()?;
//! assert!((model.weights().as_slice()[0] - 2.0).abs() < 1e-4);
//! assert!((model.intercept() - 1.0).abs() < 1e-4);
//!
//! let predicted = model.predict(&DoubleTensor::vector(vec![10.0]))?;
//! assert!((predicted.as_slice()[0] - 21.0).abs() < 1e-3);
//! assert!(coefficient_of_determination(&model.predict(&x)?, &y)? > 0.999);
//!
//! // ridge regression shrinks the weights towards the prior mean
//! let ridge = RegressionModel::linear(&x, &y)?
//!     .regularization(Regularization::Ridge)
//!     .prior_on_weights_and_intercept(0.0, 0.1)
//!     .fit()?;
//! assert!(ridge.weights().as_slice()[0] < 2.0);
//! # Ok::<(), keanu_algorithms::AlgorithmError>(())
//! ```

use crate::error::{AlgorithmError, Result};
use crate::mcmc::SamplingOptions;
use crate::mcmc::select::Sampler;
use crate::optimizer::{Algorithm, Fitness, GradientOptimizer};
use crate::samples::NetworkSamples;
use keanu_distributions::KeanuRng;
use keanu_graph::{DoubleVar, GraphBuilder, Model, NodeId, State, Value};
use keanu_tensor::{BoolTensor, DoubleTensor, Element, Tensor};
use std::sync::Arc;

/// A model that makes predictions from inputs (Java's `PredictiveModel`).
pub trait PredictiveModel {
    /// The input type.
    type Input;
    /// The prediction type.
    type Output;

    /// Predicts the output for an input.
    fn predict(&self, input: &Self::Input) -> Result<Self::Output>;
}

/// How to fit a model's latents to its observations (Java's `ModelFitter` implementations).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Fitter {
    /// Maximize the log likelihood of the observations (Java's `MaximumLikelihoodModelFitter`).
    MaxLikelihood(Algorithm),
    /// Maximize the joint log probability (Java's `MAPModelFitter`).
    MaxAPosteriori(Algorithm),
    /// Sample the posterior and set the latents to the most probable sample (Java's `SamplingModelFitter`).
    Sampling(Box<SamplingFitting>),
}

/// Settings for fitting by posterior sampling.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SamplingFitting {
    /// The sampler, e.g. Metropolis–Hastings or NUTS.
    pub sampler: Sampler,
    /// The number of steps to run the chain for.
    pub count: usize,
    /// Steps to drop and how often to record.
    pub options: SamplingOptions,
    /// The random number generator the chain uses.
    pub rng: KeanuRng,
}

impl SamplingFitting {
    /// Samples `count` steps with the given sampler, recording every step, from a generator seeded with `seed`.
    pub fn new(sampler: Sampler, count: usize, seed: u64) -> Self {
        Self {
            sampler,
            count,
            options: SamplingOptions::default(),
            rng: KeanuRng::seed_from_u64(seed),
        }
    }

    /// Sets how many steps to drop and how often to record.
    #[must_use]
    pub fn with_options(mut self, options: SamplingOptions) -> Self {
        self.options = options;
        self
    }
}

impl Fitter {
    /// Fits the latents of `model`, leaving `state` at the fitted values. Sampling fitters return the samples.
    pub fn fit(&self, model: &Model, state: &mut State) -> Result<Option<NetworkSamples>> {
        match self {
            Self::MaxLikelihood(algorithm) => {
                GradientOptimizer::new(model)
                    .algorithm(algorithm.clone())
                    .optimize(state, Fitness::LogLikelihood)?;
                Ok(None)
            }
            Self::MaxAPosteriori(algorithm) => {
                GradientOptimizer::new(model)
                    .algorithm(algorithm.clone())
                    .optimize(state, Fitness::LogProb)?;
                Ok(None)
            }
            Self::Sampling(sampling) => {
                let mut rng = sampling.rng.clone();
                let samples = sampling.sampler.sample(
                    model,
                    state,
                    model.latents(),
                    sampling.count,
                    sampling.options,
                    &mut rng,
                )?;
                let best = samples.most_probable().ok_or_else(|| {
                    AlgorithmError::InvalidSetting("the sampler recorded no samples".into())
                })?;
                let values: Vec<(NodeId, Value)> = model
                    .latents()
                    .iter()
                    .filter_map(|&id| best.get(id).map(|v| (id, v)))
                    .collect();
                model.set_and_cascade(state, &values)?;
                Ok(Some(samples))
            }
        }
    }
}

/// The prior on the regression weights and intercept, and which fitter is used by default (Java's
/// `RegressionRegularization`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Regularization {
    /// Gaussian priors, fitted by maximum likelihood (so the priors have no effect).
    #[default]
    None,
    /// Laplace priors (L1), fitted by maximum a posteriori.
    Lasso,
    /// Gaussian priors (L2), fitted by maximum a posteriori.
    Ridge,
}

impl Regularization {
    fn default_fitter(self) -> Fitter {
        match self {
            Self::None => Fitter::MaxLikelihood(Algorithm::default()),
            Self::Lasso | Self::Ridge => Fitter::MaxAPosteriori(Algorithm::default()),
        }
    }
}

/// The kind of output a regression predicts.
pub trait RegressionOutput: Element {
    /// Adds the observation of the outputs given the linear predictor, returning the observed node.
    #[doc(hidden)]
    fn observation<'b>(
        builder: &'b GraphBuilder,
        linear_predictor: DoubleVar<'b>,
        outputs: &Tensor<Self>,
    ) -> NodeId;

    /// The prediction from the linear predictor.
    #[doc(hidden)]
    fn prediction(linear_predictor: DoubleTensor) -> Result<Tensor<Self>>;
}

impl RegressionOutput for f64 {
    /// Linear regression: outputs ~ Gaussian(linear predictor, 1). The prediction is the mean.
    fn observation<'b>(
        builder: &'b GraphBuilder,
        linear_predictor: DoubleVar<'b>,
        outputs: &DoubleTensor,
    ) -> NodeId {
        let y = builder
            .gaussian(linear_predictor, OBSERVATION_SIGMA)
            .label("output");
        builder.observe(y, outputs.clone());
        y.id()
    }

    fn prediction(linear_predictor: DoubleTensor) -> Result<DoubleTensor> {
        Ok(linear_predictor)
    }
}

impl RegressionOutput for bool {
    /// Logistic regression: outputs ~ Bernoulli(sigmoid(linear predictor)). The prediction is p > 0.5.
    fn observation<'b>(
        builder: &'b GraphBuilder,
        linear_predictor: DoubleVar<'b>,
        outputs: &BoolTensor,
    ) -> NodeId {
        let y = builder
            .bernoulli(linear_predictor.sigmoid())
            .label("output");
        builder.observe(y, outputs.clone());
        y.id()
    }

    fn prediction(linear_predictor: DoubleTensor) -> Result<BoolTensor> {
        Ok(linear_predictor.sigmoid().gt(&DoubleTensor::scalar(0.5))?)
    }
}

const OBSERVATION_SIGMA: f64 = 1.0;
const DEFAULT_PRIOR_MEAN: f64 = 0.0;
const DEFAULT_PRIOR_SCALE: f64 = 1.0;

/// A prior: means and scales (sigmas for Gaussian priors, betas for Laplace priors), broadcast to the shape of
/// what they are a prior on.
#[derive(Debug, Clone, PartialEq)]
pub struct Prior {
    /// The prior means.
    pub mean: DoubleTensor,
    /// The prior scales.
    pub scale: DoubleTensor,
}

impl Prior {
    /// A prior with the same mean and scale for every element.
    pub fn new(mean: f64, scale: f64) -> Self {
        Self {
            mean: DoubleTensor::scalar(mean),
            scale: DoubleTensor::scalar(scale),
        }
    }
}

impl Default for Prior {
    fn default() -> Self {
        Self::new(DEFAULT_PRIOR_MEAN, DEFAULT_PRIOR_SCALE)
    }
}

/// Scalars become a 1 × 1 matrix and vectors a column (n samples of one feature).
fn as_matrix<T: Element>(data: &Tensor<T>) -> Result<Tensor<T>> {
    Ok(match data.rank() {
        0 => data.reshape(&[1, 1])?,
        1 => data.reshape(&[data.len() as isize, 1])?,
        _ => data.clone(),
    })
}

/// Configures a [`RegressionModel`] (Java's `RegressionModelBuilder`).
#[derive(Debug, Clone)]
pub struct RegressionBuilder<O: RegressionOutput> {
    inputs: DoubleTensor,
    outputs: Tensor<O>,
    regularization: Regularization,
    weights_prior: Prior,
    intercept_prior: Prior,
    fitter: Option<Fitter>,
}

impl<O: RegressionOutput> RegressionBuilder<O> {
    fn new(inputs: &DoubleTensor, outputs: &Tensor<O>) -> Result<Self> {
        let inputs = as_matrix(inputs)?;
        let outputs = as_matrix(outputs)?;
        if inputs.rank() != 2 || outputs.rank() != 2 {
            return Err(AlgorithmError::InvalidArgument(
                "regression inputs and outputs must be at most rank 2".into(),
            ));
        }
        if inputs.shape()[0] != outputs.shape()[0] || outputs.shape()[1] != 1 {
            return Err(AlgorithmError::InvalidArgument(format!(
                "outputs of shape {:?} don't match inputs of shape {:?}: expected one output per input row",
                outputs.shape(),
                inputs.shape()
            )));
        }
        Ok(Self {
            inputs,
            outputs,
            regularization: Regularization::None,
            weights_prior: Prior::default(),
            intercept_prior: Prior::default(),
            fitter: None,
        })
    }

    /// Sets the regularization: the kind of prior, and the default fitter.
    pub fn regularization(mut self, regularization: Regularization) -> Self {
        self.regularization = regularization;
        self
    }

    /// Sets the prior on the weights. The means and scales broadcast to the weights' shape `[features, 1]`.
    pub fn prior_on_weights(
        mut self,
        mean: impl Into<DoubleTensor>,
        scale: impl Into<DoubleTensor>,
    ) -> Self {
        self.weights_prior = Prior {
            mean: mean.into(),
            scale: scale.into(),
        };
        self
    }

    /// Sets the prior on the intercept.
    pub fn prior_on_intercept(mut self, mean: f64, scale: f64) -> Self {
        self.intercept_prior = Prior::new(mean, scale);
        self
    }

    /// Sets the same prior on the weights and the intercept.
    pub fn prior_on_weights_and_intercept(self, mean: f64, scale: f64) -> Self {
        self.prior_on_weights(mean, scale)
            .prior_on_intercept(mean, scale)
    }

    /// Fits by posterior sampling instead of the regularization's default fitter.
    pub fn sampling(mut self, sampling: SamplingFitting) -> Self {
        self.fitter = Some(Fitter::Sampling(Box::new(sampling)));
        self
    }

    /// Fits with a specific fitter instead of the regularization's default.
    pub fn fitter(mut self, fitter: Fitter) -> Self {
        self.fitter = Some(fitter);
        self
    }

    /// Builds the model without fitting it. The weights and intercept start at their prior means.
    pub fn build(self) -> Result<RegressionModel<O>> {
        let features = self.inputs.shape()[1];
        let weights_shape = [features, 1];
        let b = GraphBuilder::new();
        let x = b.constant(self.inputs.clone()).label("input");
        let (weights, intercept) = match self.regularization {
            Regularization::None | Regularization::Ridge => (
                b.gaussian_with_shape(
                    &weights_shape,
                    self.weights_prior.mean.clone(),
                    self.weights_prior.scale.clone(),
                ),
                b.gaussian(
                    self.intercept_prior.mean.clone(),
                    self.intercept_prior.scale.clone(),
                ),
            ),
            Regularization::Lasso => (
                b.laplace_with_shape(
                    &weights_shape,
                    self.weights_prior.mean.clone(),
                    self.weights_prior.scale.clone(),
                ),
                b.laplace(
                    self.intercept_prior.mean.clone(),
                    self.intercept_prior.scale.clone(),
                ),
            ),
        };
        let weights = weights.label("weights");
        let intercept = intercept.label("intercept");

        // start the optimization or chain from the prior means
        let start = self.weights_prior.mean.broadcast_to(&weights_shape)?;
        b.set_value(weights, start);
        b.set_value(
            intercept,
            DoubleTensor::scalar(self.intercept_prior.mean.as_slice()[0]),
        );

        let linear_predictor = if features == 1 {
            weights * x
        } else {
            x.matmul(weights)
        } + intercept;
        let output = O::observation(&b, linear_predictor, &self.outputs);

        let (weights, intercept) = (weights.id(), intercept.id());
        let (graph, mut state) = b.build()?;
        let all: Vec<NodeId> = graph.ids().collect();
        // every value is set or deterministic, so no sampling happens and the generator is never used
        state.eval(&graph, &all, &mut KeanuRng::seed_from_u64(0))?;
        let model = Model::new(Arc::new(graph), &state)?;

        Ok(RegressionModel {
            fitter: self
                .fitter
                .unwrap_or_else(|| self.regularization.default_fitter()),
            model,
            state,
            weights,
            intercept,
            output,
            samples: None,
            _output: std::marker::PhantomData,
        })
    }

    /// Builds the model and fits it to the training data.
    pub fn fit(self) -> Result<RegressionModel<O>> {
        let mut model = self.build()?;
        model.fit()?;
        Ok(model)
    }
}

/// A regression model: outputs depend on a linear combination of the inputs, `x · weights + intercept`
/// (Java's `RegressionModel`). Linear regression observes `Gaussian(x · w + b, 1)`; logistic regression observes
/// `Bernoulli(sigmoid(x · w + b))`.
///
/// Inputs are matrices of shape `[samples, features]`; a vector is taken to be samples of a single feature.
#[derive(Debug)]
pub struct RegressionModel<O: RegressionOutput> {
    fitter: Fitter,
    model: Model,
    state: State,
    weights: NodeId,
    intercept: NodeId,
    output: NodeId,
    samples: Option<NetworkSamples>,
    _output: std::marker::PhantomData<O>,
}

/// A linear regression model.
pub type LinearRegression = RegressionModel<f64>;
/// A logistic regression model.
pub type LogisticRegression = RegressionModel<bool>;

impl RegressionModel<f64> {
    /// Configures a linear regression on training inputs `[samples, features]` and outputs `[samples]` or
    /// `[samples, 1]`.
    pub fn linear(inputs: &DoubleTensor, outputs: &DoubleTensor) -> Result<RegressionBuilder<f64>> {
        RegressionBuilder::new(inputs, outputs)
    }
}

impl RegressionModel<bool> {
    /// Configures a logistic regression on training inputs `[samples, features]` and boolean outputs.
    pub fn logistic(
        inputs: &DoubleTensor,
        outputs: &BoolTensor,
    ) -> Result<RegressionBuilder<bool>> {
        RegressionBuilder::new(inputs, outputs)
    }
}

impl<O: RegressionOutput> RegressionModel<O> {
    /// Fits (or refits) the weights and intercept to the training data with the model's fitter.
    pub fn fit(&mut self) -> Result<()> {
        self.samples = self.fitter.fit(&self.model, &mut self.state)?;
        Ok(())
    }

    /// The weights, shape `[features, 1]`.
    pub fn weights(&self) -> &DoubleTensor {
        self.state
            .f64(self.weights)
            .expect("weights always have a value")
    }

    /// The intercept.
    pub fn intercept(&self) -> f64 {
        self.state
            .f64(self.intercept)
            .expect("the intercept always has a value")
            .as_slice()[0]
    }

    /// The linear predictor `x · weights + intercept` for inputs `[samples, features]` (or a vector of samples of
    /// one feature).
    pub fn linear_predictor(&self, inputs: &DoubleTensor) -> Result<DoubleTensor> {
        let x = as_matrix(inputs)?;
        let weights = self.weights();
        if x.rank() != 2 || x.shape()[1] != weights.shape()[0] {
            return Err(AlgorithmError::InvalidArgument(format!(
                "inputs of shape {:?} don't have the model's {} features",
                inputs.shape(),
                weights.shape()[0]
            )));
        }
        let product = if weights.len() == 1 {
            x.mul(weights)?
        } else {
            x.matmul(weights)?
        };
        Ok(&product + self.intercept())
    }

    /// The log probability (`Fitness::LogProb`) or log likelihood (`Fitness::LogLikelihood`) of the training
    /// data with the given weights `[features, 1]` and intercept, without changing the model.
    pub fn fitness_at(
        &self,
        weights: &DoubleTensor,
        intercept: f64,
        fitness: Fitness,
    ) -> Result<f64> {
        let mut state = self.state.clone();
        self.model.set_and_cascade(
            &mut state,
            &[
                (self.weights, Value::F64(weights.clone())),
                (self.intercept, Value::F64(DoubleTensor::scalar(intercept))),
            ],
        )?;
        Ok(match fitness {
            Fitness::LogProb => self.model.log_prob(&state)?,
            Fitness::LogLikelihood => self.model.log_likelihood(&state)?,
        })
    }

    /// The samples recorded when the model was fitted by sampling.
    pub fn samples(&self) -> Option<&NetworkSamples> {
        self.samples.as_ref()
    }

    /// The underlying probabilistic model.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// The state of the underlying model (the training data and fitted values).
    pub fn state(&self) -> &State {
        &self.state
    }

    /// The weights node.
    pub fn weights_node(&self) -> NodeId {
        self.weights
    }

    /// The intercept node.
    pub fn intercept_node(&self) -> NodeId {
        self.intercept
    }

    /// The observed output node.
    pub fn output_node(&self) -> NodeId {
        self.output
    }
}

impl<O: RegressionOutput> PredictiveModel for RegressionModel<O> {
    type Input = DoubleTensor;
    type Output = Tensor<O>;

    /// Predicts outputs `[samples, 1]`: the mean for linear regression, and whether p > 0.5 for logistic regression.
    fn predict(&self, input: &DoubleTensor) -> Result<Tensor<O>> {
        O::prediction(self.linear_predictor(input)?)
    }
}

impl<O: RegressionOutput> RegressionModel<O> {
    /// Predicts outputs `[samples, 1]`; see [`PredictiveModel::predict`].
    pub fn predict(&self, input: &DoubleTensor) -> Result<Tensor<O>> {
        PredictiveModel::predict(self, input)
    }
}

/// The coefficient of determination R² of predictions: 1 - (residual sum of squares) / (total sum of squares).
/// 1 is a perfect fit; it is negative when predicting the mean would be better.
pub fn coefficient_of_determination(
    predicted: &DoubleTensor,
    actual: &DoubleTensor,
) -> Result<f64> {
    let (predicted, actual) = (as_matrix(predicted)?, as_matrix(actual)?);
    let residual = actual.sub(&predicted)?.map(|r| r * r).sum();
    let mean = actual.mean();
    let total = actual.map(|a| (a - mean) * (a - mean)).sum();
    Ok(1.0 - residual / total)
}

/// The fraction of predictions equal to the actual values.
pub fn accuracy(predicted: &BoolTensor, actual: &BoolTensor) -> Result<f64> {
    let (predicted, actual) = (as_matrix(predicted)?, as_matrix(actual)?);
    let correct = predicted.zip_with(&actual, |p, a| f64::from(u8::from(p == a)))?;
    Ok(correct.sum() / actual.len() as f64)
}
