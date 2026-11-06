//! Structural time series: a series modelled as a sum of interpretable components (a level or trend, seasonal
//! effects, an autoregression, a regression on covariates) plus observation noise, as in TensorFlow Probability's
//! `tfp.sts`.
//!
//! Every component but the regression is a small linear-Gaussian state-space model; [`StructuralModel`] stacks
//! their states into one state vector (block-diagonal transition and noise covariance, the observation matrix the
//! components' rows side by side) and builds a single [linear Gaussian state space
//! node](crate::GraphBuilder::linear_gaussian_state_space) for the series, so the components' states are integrated
//! out exactly by the Kalman filter and a sampler draws only the scales and coefficients. A regression is an
//! observation offset: the series minus X β follows the state-space model, and the likelihood is then a
//! [factor](crate::GraphBuilder::factor) of the state-space model's [log density](crate::GraphBuilder::log_density)
//! of that residual.
//!
//! Components, with their state x and noise w ~ N(0, σ²):
//!
//! | Component | State | Transition | Parameters |
//! |---|---|---|---|
//! | [`Component::local_level`] | level μ | μ_{t+1} = μ_t + w | `level_scale` |
//! | [`Component::local_linear_trend`] | level μ, slope δ | μ_{t+1} = μ_t + δ_t + w₁, δ_{t+1} = δ_t + w₂ | `level_scale`, `slope_scale` |
//! | [`Component::seasonal`] | the last S − 1 seasonal effects | γ_{t+1} = −(γ_t + … + γ_{t−S+2}) + w | `drift_scale` |
//! | [`Component::trigonometric_seasonal`] | a rotating pair per harmonic | rotation by 2πj / P, plus w on both | `drift_scale` |
//! | [`Component::autoregressive`] | the last p values | a_{t+1} = Σᵢ ρᵢ a_{t+1−i} + w | `coefficients`, `scale` |
//! | [`Component::regression`] | none (an offset X β) | — | `weights` |
//!
//! Each component contributes its first state (the level, the current seasonal effect, the current AR value; the
//! sum of the cosine states of the trigonometric form) to the observation, and the series is
//! y_t = Σ components + N(0, `observation_noise_scale`²).
//!
//! Scales and coefficients are graph variables: pass your own (a latent with your prior, or a constant to fix one)
//! with [`Component::with_scale`], [`Component::with_slope_scale`], [`Component::with_coefficients`] and
//! [`StructuralModel::with_observation_noise_scale`], or let the builder create latents with these default priors,
//! where s is the observed series' standard deviation (1 if the series is constant) and y₀ its first value:
//!
//! - every component scale (level, slope, seasonal drift, AR innovation): LogNormal(ln(0.05 s), 2);
//! - the observation noise scale: LogNormal(ln(0.1 s), 2);
//! - AR coefficients: N(0, 1) each (stationarity is not imposed);
//! - regression weights: N(0, 10 s / max_t |x_tj|) for covariate j.
//!
//! The created latents start at their prior medians (coefficients and weights at 0) and are labelled
//! `"<component name>.<parameter>"` and `"observation_noise_scale"`. The initial state is fixed (not a latent):
//! the level of the first local level or local linear trend component starts at N(y₀, s²) and every other state
//! (the level of any later such component included, so that the levels sum to y₀) at N(0, s²), independently.
//! With a regression the level still starts at y₀: that is the first residual y₀ − x₀ β at the default weights'
//! prior mean β = 0 (given weights are not read when the model is built).
//!
//! The model is time-invariant (as is the state-space node), so a seasonal component's season is one step:
//! TFP's `num_steps_per_season` > 1 is the trigonometric form with period `seasons × steps per season`.
//!
//! [`StructuralTimeSeries`] describes the built model: the nodes to record when sampling and where each
//! component's states sit. `keanu_algorithms::sts::StructuralPosterior` decomposes the series into its components
//! and forecasts it over posterior draws.
//!
//! ```
//! use keanu_graph::GraphBuilder;
//! use keanu_graph::sts::{Component, StructuralModel};
//!
//! let b = GraphBuilder::new();
//! let y: Vec<f64> = (0..24).map(|t| 0.1 * t as f64 + [1.0, -1.0, 0.5, -0.5][t % 4]).collect();
//! let sts = StructuralModel::new(&b, &y)
//!     .with(Component::local_linear_trend())
//!     .with(Component::seasonal(4))
//!     .build()?;
//! assert_eq!(sts.state_dimension(), 2 + 3);
//! assert_eq!(sts.parameters().len(), 4); // two trend scales, the seasonal drift and the noise
//! b.build()?;
//! # Ok::<(), keanu_graph::GraphError>(())
//! ```

use crate::builder::GraphBuilder;
use crate::error::{GraphError, Result};
use crate::graph::NodeId;
use crate::op::DistributionKind;
use crate::vars::DoubleVar;
use keanu_tensor::DoubleTensor;
use std::f64::consts::PI;
use std::ops::Range;

fn invalid(message: impl Into<String>) -> GraphError {
    GraphError::InvalidNode {
        op: "sts",
        message: message.into(),
    }
}

/// What a component is, with its sizes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ComponentKind {
    /// A random-walk level.
    LocalLevel,
    /// A random-walk level whose drift (slope) is itself a random walk.
    LocalLinearTrend,
    /// Seasonal effects that sum to zero (plus drift) over `seasons` consecutive steps.
    Seasonal {
        /// The number of seasons in a cycle.
        seasons: usize,
    },
    /// A seasonal pattern of the given period (in steps, not necessarily whole) from `harmonics` sinusoids.
    TrigonometricSeasonal {
        /// The period in steps.
        period: f64,
        /// The number of harmonics, 1 to ⌊period / 2⌋.
        harmonics: usize,
    },
    /// An autoregression of the given order.
    Autoregressive {
        /// The number of lags.
        order: usize,
    },
    /// A linear regression on `covariates` known covariates.
    Regression {
        /// The number of covariates (columns of the design matrix).
        covariates: usize,
    },
}

impl ComponentKind {
    fn default_name(&self) -> &'static str {
        match self {
            Self::LocalLevel => "level",
            Self::LocalLinearTrend => "trend",
            Self::Seasonal { .. } | Self::TrigonometricSeasonal { .. } => "seasonal",
            Self::Autoregressive { .. } => "autoregressive",
            Self::Regression { .. } => "regression",
        }
    }
}

/// One component of a [`StructuralModel`], with its options. See the [module documentation](self) for each kind's
/// state, transition and default priors.
#[derive(Debug, Clone)]
pub struct Component<'b> {
    name: String,
    kind: ComponentKind,
    scale: Option<DoubleVar<'b>>,
    slope_scale: Option<DoubleVar<'b>>,
    coefficients: Option<DoubleVar<'b>>,
    design: Option<DoubleTensor>,
}

impl<'b> Component<'b> {
    fn new(kind: ComponentKind) -> Self {
        Self {
            name: kind.default_name().to_string(),
            kind,
            scale: None,
            slope_scale: None,
            coefficients: None,
            design: None,
        }
    }

    /// A local level: a random walk μ_{t+1} = μ_t + N(0, `level_scale`²). Named `"level"`.
    pub fn local_level() -> Self {
        Self::new(ComponentKind::LocalLevel)
    }

    /// A local linear trend: a level μ_{t+1} = μ_t + δ_t + N(0, `level_scale`²) whose slope is a random walk
    /// δ_{t+1} = δ_t + N(0, `slope_scale`²). Named `"trend"`.
    pub fn local_linear_trend() -> Self {
        Self::new(ComponentKind::LocalLinearTrend)
    }

    /// Seasonal effects in the dummy-variable ("sum to zero") form, as TFP's `Seasonal` with one step per season:
    /// each step has one of `seasons` effects, the next effect is minus the sum of the previous S − 1 plus
    /// N(0, `drift_scale`²), so with no drift any S consecutive effects sum to zero. Its state holds the last S − 1
    /// effects. `seasons` must be at least 2. Named `"seasonal"`.
    pub fn seasonal(seasons: usize) -> Self {
        Self::new(ComponentKind::Seasonal { seasons })
    }

    /// A seasonal pattern of period `period` steps (any real number of at least 2, such as 52.18 weeks a year) as
    /// a sum of `harmonics` sinusoids of frequencies 2πj / period, each a rotating pair of states with
    /// N(0, `drift_scale`²) noise on both (Harvey's trigonometric seasonal); a harmonic at the Nyquist frequency
    /// (j = period / 2) has the cosine state alone. `harmonics` must be 1 to ⌊period / 2⌋; ⌊period / 2⌋ harmonics
    /// of a whole period are as flexible as [`Self::seasonal`]. Named `"seasonal"`.
    pub fn trigonometric_seasonal(period: f64, harmonics: usize) -> Self {
        Self::new(ComponentKind::TrigonometricSeasonal { period, harmonics })
    }

    /// An autoregression of order p: a_{t+1} = Σᵢ ρᵢ a_{t+1−i} + N(0, `scale`²), with `coefficients` ρ of shape
    /// `[p]`. `order` must be at least 1. Named `"autoregressive"`.
    pub fn autoregressive(order: usize) -> Self {
        Self::new(ComponentKind::Autoregressive { order })
    }

    /// A static linear regression X β on known covariates: `design` is X, [T, k], one row per observed step. It is
    /// an offset of the observations (it has no state), and forecasts take the covariates' future rows.
    /// Named `"regression"`.
    pub fn regression(design: DoubleTensor) -> Self {
        let covariates = design.shape().get(1).copied().unwrap_or(0);
        let mut c = Self::new(ComponentKind::Regression { covariates });
        c.design = Some(design);
        c
    }

    /// Names the component (for the decomposition and the latents' labels); names must be distinct.
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// The component's noise scale, a scalar: the level scale of a level or trend, the drift scale of a seasonal
    /// component, the innovation scale of an autoregression. A regression has none.
    pub fn with_scale(mut self, scale: DoubleVar<'b>) -> Self {
        self.scale = Some(scale);
        self
    }

    /// The slope scale of a local linear trend, a scalar.
    pub fn with_slope_scale(mut self, scale: DoubleVar<'b>) -> Self {
        self.slope_scale = Some(scale);
        self
    }

    /// The coefficients ρ of an autoregression (`[p]`) or the weights β of a regression (`[k]`).
    pub fn with_coefficients(mut self, coefficients: DoubleVar<'b>) -> Self {
        self.coefficients = Some(coefficients);
        self
    }

    /// The component's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The component's kind.
    pub fn kind(&self) -> ComponentKind {
        self.kind
    }
}

/// A builder of a structural time-series model of an observed series on a [`GraphBuilder`]. See the
/// [module documentation](self).
#[derive(Debug, Clone)]
pub struct StructuralModel<'b> {
    builder: &'b GraphBuilder,
    observed: Vec<f64>,
    components: Vec<Component<'b>>,
    noise_scale: Option<DoubleVar<'b>>,
}

/// One component of a built [`StructuralTimeSeries`]: where its states sit in the joint state vector, how they
/// are observed, and its parameters' nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentLayout {
    name: String,
    kind: ComponentKind,
    states: Range<usize>,
    observation: Vec<f64>,
    parameters: Vec<(String, NodeId)>,
    design: Option<DoubleTensor>,
    weights: Option<NodeId>,
}

impl ComponentLayout {
    /// The component's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The component's kind.
    pub fn kind(&self) -> ComponentKind {
        self.kind
    }

    /// The component's states in the joint state vector (empty for a regression).
    pub fn states(&self) -> Range<usize> {
        self.states.clone()
    }

    /// The component's block of the observation matrix: its contribution to the series is this row times its
    /// states (empty for a regression).
    pub fn observation(&self) -> &[f64] {
        &self.observation
    }

    /// The component's parameters, by name (`"level_scale"`, `"slope_scale"`, `"drift_scale"`, `"scale"`,
    /// `"coefficients"`, `"weights"`), given or created.
    pub fn parameters(&self) -> &[(String, NodeId)] {
        &self.parameters
    }

    /// A regression's design matrix, [T, k].
    pub fn design(&self) -> Option<&DoubleTensor> {
        self.design.as_ref()
    }

    /// A regression's weights node, `[k]`.
    pub fn weights(&self) -> Option<NodeId> {
        self.weights
    }
}

/// A built structural time-series model: its components' layout and the nodes of its parameters, the state-space
/// model's six parameters and the likelihood. See the [module documentation](self).
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralTimeSeries {
    observed: Vec<f64>,
    components: Vec<ComponentLayout>,
    state_dimension: usize,
    state_space: [NodeId; 6],
    noise_scale: NodeId,
    likelihood: NodeId,
    regression: bool,
}

impl StructuralTimeSeries {
    /// The observed series.
    pub fn observed(&self) -> &[f64] {
        &self.observed
    }

    /// The number of observed steps T.
    pub fn steps(&self) -> usize {
        self.observed.len()
    }

    /// The components, in the order they were added.
    pub fn components(&self) -> &[ComponentLayout] {
        &self.components
    }

    /// The component with this name.
    pub fn component(&self, name: &str) -> Option<&ComponentLayout> {
        self.components.iter().find(|c| c.name == name)
    }

    /// The dimension d of the joint state.
    pub fn state_dimension(&self) -> usize {
        self.state_dimension
    }

    /// The nodes of the state-space model's parameters, in the order of
    /// [`GraphBuilder::linear_gaussian_state_space`]: the initial mean `[d]` and covariance [d, d], the transition
    /// matrix and covariance [d, d], the observation matrix [1, d] and the observation covariance [1, 1].
    pub fn state_space_parameters(&self) -> [NodeId; 6] {
        self.state_space
    }

    /// The observation noise scale's node.
    pub fn observation_noise_scale(&self) -> NodeId {
        self.noise_scale
    }

    /// The likelihood: the observed linear Gaussian state space node, or, with a regression, the factor of the
    /// state-space log density of the series minus the regression.
    pub fn likelihood(&self) -> NodeId {
        self.likelihood
    }

    /// Whether the model has a regression component (and so a factor, not an observed node, as its likelihood).
    pub fn has_regression(&self) -> bool {
        self.regression
    }

    /// Every parameter's node, labelled `"<component>.<parameter>"` and `"observation_noise_scale"`: the nodes to
    /// record when sampling (with any latents that parameters you supplied depend on).
    pub fn parameters(&self) -> Vec<(String, NodeId)> {
        let mut all: Vec<(String, NodeId)> = self
            .components
            .iter()
            .flat_map(|c| {
                c.parameters
                    .iter()
                    .map(move |(p, id)| (format!("{}.{p}", c.name), *id))
            })
            .collect();
        all.push(("observation_noise_scale".into(), self.noise_scale));
        all
    }
}

/// One component's contribution to the joint model, before assembly.
struct Block<'b> {
    dimension: usize,
    /// The constant part of the transition matrix, [dim, dim] row-major.
    transition: Vec<f64>,
    observation: Vec<f64>,
    initial_mean: Vec<f64>,
    /// The variances, each with the states (as a 0/1 mask over the block) it applies to.
    noise: Vec<(DoubleVar<'b>, Vec<f64>)>,
    /// An autoregression's coefficients, which fill the block's first row of the transition matrix.
    coefficients: Option<DoubleVar<'b>>,
}

impl<'b> StructuralModel<'b> {
    /// A model of the `observed` series (one value per step), to be built on `builder`.
    pub fn new(builder: &'b GraphBuilder, observed: &[f64]) -> Self {
        Self {
            builder,
            observed: observed.to_vec(),
            components: Vec::new(),
            noise_scale: None,
        }
    }

    /// Adds a component.
    pub fn with(mut self, component: Component<'b>) -> Self {
        self.components.push(component);
        self
    }

    /// The observation noise scale, a scalar (by default a LogNormal(ln(0.1 s), 2) latent).
    pub fn with_observation_noise_scale(mut self, scale: DoubleVar<'b>) -> Self {
        self.noise_scale = Some(scale);
        self
    }

    /// The first value y₀ and the scale s of the observed series that the default priors use.
    fn data_scale(&self) -> (f64, f64) {
        let n = self.observed.len() as f64;
        let mean = self.observed.iter().sum::<f64>() / n;
        let var = if self.observed.len() > 1 {
            self.observed
                .iter()
                .map(|y| (y - mean).powi(2))
                .sum::<f64>()
                / (n - 1.0)
        } else {
            0.0
        };
        let s = if var > 0.0 { var.sqrt() } else { 1.0 };
        (self.observed[0], s)
    }

    /// A given scale, or a LogNormal(ln(median), 2) latent labelled `label` starting at its median.
    fn scale(&self, given: Option<DoubleVar<'b>>, median: f64, label: &str) -> DoubleVar<'b> {
        given.unwrap_or_else(|| {
            let b = self.builder;
            let v = b.log_normal(median.ln(), 2.0).label(label);
            b.set_value(v, DoubleTensor::scalar(median));
            v
        })
    }

    /// Given coefficients, or N(0, scales) latents labelled `label` starting at 0.
    fn coefficients(
        &self,
        given: Option<DoubleVar<'b>>,
        scales: Vec<f64>,
        label: &str,
    ) -> DoubleVar<'b> {
        given.unwrap_or_else(|| {
            let b = self.builder;
            let n = scales.len();
            let v = b
                .gaussian_with_shape(&[n], 0.0, DoubleTensor::vector(scales))
                .label(label);
            b.set_value(v, DoubleTensor::vector(vec![0.0; n]));
            v
        })
    }

    /// Checks the series and the components' options before anything is added to the builder.
    fn check(&self) -> Result<()> {
        if self.observed.is_empty() {
            return Err(invalid("the observed series is empty"));
        }
        if let Some(t) = self.observed.iter().position(|y| !y.is_finite()) {
            return Err(invalid(format!(
                "the observed value at step {t} is not finite"
            )));
        }
        if !self
            .components
            .iter()
            .any(|c| !matches!(c.kind, ComponentKind::Regression { .. }))
        {
            return Err(invalid(
                "needs at least one component with a state (a level, trend, seasonal or autoregressive component)",
            ));
        }
        let scalar = |v: Option<DoubleVar<'b>>, what: &str| match v {
            Some(v) if !v.shape().is_empty() => Err(invalid(format!(
                "{what} must be a scalar, not of shape {:?}",
                v.shape()
            ))),
            _ => Ok(()),
        };
        scalar(self.noise_scale, "the observation noise scale")?;
        for (i, c) in self.components.iter().enumerate() {
            scalar(c.scale, &format!("the scale of {:?}", c.name))?;
            scalar(c.slope_scale, &format!("the slope scale of {:?}", c.name))?;
            let length = match c.kind {
                ComponentKind::Autoregressive { order } => order,
                ComponentKind::Regression { covariates } => covariates,
                _ => 0,
            };
            if let Some(v) = c.coefficients.filter(|v| v.shape() != [length]) {
                return Err(invalid(format!(
                    "the coefficients of {:?} must have shape [{length}], not {:?}",
                    c.name,
                    v.shape()
                )));
            }
            if self.components[..i].iter().any(|o| o.name == c.name) {
                return Err(invalid(format!(
                    "two components are named {:?}; name them apart with `with_name`",
                    c.name
                )));
            }
            let what = |option: &str| {
                invalid(format!(
                    "the {} component {:?} has no {option}",
                    c.kind.default_name(),
                    c.name
                ))
            };
            match c.kind {
                ComponentKind::Seasonal { seasons } if seasons < 2 => {
                    return Err(invalid(format!(
                        "a seasonal component needs at least 2 seasons, not {seasons}"
                    )));
                }
                ComponentKind::TrigonometricSeasonal { period, harmonics } => {
                    if !(period.is_finite() && period >= 2.0) {
                        return Err(invalid(format!(
                            "a trigonometric seasonal component needs a period of at least 2 steps, not {period}"
                        )));
                    }
                    if harmonics == 0 || harmonics as f64 > period / 2.0 {
                        return Err(invalid(format!(
                            "a trigonometric seasonal component of period {period} needs 1 to {} harmonics, \
                             not {harmonics}",
                            (period / 2.0).floor()
                        )));
                    }
                }
                ComponentKind::Autoregressive { order: 0 } => {
                    return Err(invalid(
                        "an autoregressive component needs an order of at least 1",
                    ));
                }
                ComponentKind::Regression { covariates } => {
                    let design = c.design.as_ref().ok_or_else(|| what("design"))?;
                    if design.shape() != [self.observed.len(), covariates] || covariates == 0 {
                        return Err(invalid(format!(
                            "the design matrix of {:?} must have shape [{}, k] with k ≥ 1 (one row per observed \
                             step), not {:?}",
                            c.name,
                            self.observed.len(),
                            design.shape()
                        )));
                    }
                    if c.scale.is_some() {
                        return Err(what("scale"));
                    }
                    let x = design.as_slice();
                    for j in 0..covariates {
                        if x.iter().any(|v| !v.is_finite()) || column_max(x, covariates, j) == 0.0 {
                            return Err(invalid(format!(
                                "column {j} of the design matrix of {:?} is all zero or not finite",
                                c.name
                            )));
                        }
                    }
                }
                _ => {}
            }
            if c.slope_scale.is_some() && c.kind != ComponentKind::LocalLinearTrend {
                return Err(what("slope scale"));
            }
            if c.coefficients.is_some()
                && !matches!(
                    c.kind,
                    ComponentKind::Autoregressive { .. } | ComponentKind::Regression { .. }
                )
            {
                return Err(what("coefficients"));
            }
        }
        Ok(())
    }

    /// Builds the model on the builder, adding the latents it creates, the state-space parameters and the
    /// likelihood (observing the series), and returns its description.
    ///
    /// # Errors
    ///
    /// [`GraphError::InvalidNode`] (with op `"sts"`) for an empty or non-finite series, no component with a state,
    /// duplicate component names, invalid sizes (fewer than 2 seasons, an order of 0, a design matrix without one
    /// row per step), a given scale that isn't a scalar or coefficients of the wrong shape, and an option a
    /// component doesn't have. Nothing is added to the builder then. A given variable from another builder is the
    /// builder's error, reported when it builds.
    pub fn build(self) -> Result<StructuralTimeSeries> {
        self.check()?;
        let b = self.builder;
        let (y0, s) = self.data_scale();
        // the first level-type component (local level or linear trend) starts at y₀, the series' level; any other
        // at 0, so that the levels' sum starts at y₀ too
        let mut level_mean = Some(y0);
        let steps = self.observed.len();
        let mut blocks = Vec::new();
        let mut layouts = Vec::new();
        let mut regressions: Vec<(DoubleTensor, DoubleVar<'b>)> = Vec::new();
        let mut offset = 0;
        for c in &self.components {
            let label = |p: &str| format!("{}.{p}", c.name);
            let mut parameters = Vec::new();
            let mut regression = None;
            let block = match c.kind {
                ComponentKind::LocalLevel => {
                    let scale = self.scale(c.scale, 0.05 * s, &label("level_scale"));
                    parameters.push(("level_scale".to_string(), scale.id()));
                    Some(Block {
                        dimension: 1,
                        transition: vec![1.0],
                        observation: vec![1.0],
                        initial_mean: vec![level_mean.take().unwrap_or(0.0)],
                        noise: vec![(scale * scale, vec![1.0])],
                        coefficients: None,
                    })
                }
                ComponentKind::LocalLinearTrend => {
                    let level = self.scale(c.scale, 0.05 * s, &label("level_scale"));
                    let slope = self.scale(c.slope_scale, 0.05 * s, &label("slope_scale"));
                    parameters.push(("level_scale".to_string(), level.id()));
                    parameters.push(("slope_scale".to_string(), slope.id()));
                    Some(Block {
                        dimension: 2,
                        transition: vec![1.0, 1.0, 0.0, 1.0],
                        observation: vec![1.0, 0.0],
                        initial_mean: vec![level_mean.take().unwrap_or(0.0), 0.0],
                        noise: vec![
                            (level * level, vec![1.0, 0.0]),
                            (slope * slope, vec![0.0, 1.0]),
                        ],
                        coefficients: None,
                    })
                }
                ComponentKind::Seasonal { seasons } => {
                    let scale = self.scale(c.scale, 0.05 * s, &label("drift_scale"));
                    parameters.push(("drift_scale".to_string(), scale.id()));
                    let n = seasons - 1;
                    let mut transition = vec![0.0; n * n];
                    transition[..n].fill(-1.0);
                    for i in 1..n {
                        transition[i * n + i - 1] = 1.0;
                    }
                    Some(Block {
                        dimension: n,
                        transition,
                        observation: first(n),
                        initial_mean: vec![0.0; n],
                        noise: vec![(scale * scale, first(n))],
                        coefficients: None,
                    })
                }
                ComponentKind::TrigonometricSeasonal { period, harmonics } => {
                    let scale = self.scale(c.scale, 0.05 * s, &label("drift_scale"));
                    parameters.push(("drift_scale".to_string(), scale.id()));
                    // each harmonic a rotating (cosine, sine) pair; at the Nyquist frequency (j = period / 2) the
                    // sine state is always 0, so that harmonic has the cosine state alone
                    let sizes: Vec<usize> = (1..=harmonics)
                        .map(|j| if 2.0 * j as f64 == period { 1 } else { 2 })
                        .collect();
                    let n: usize = sizes.iter().sum();
                    let mut transition = vec![0.0; n * n];
                    let mut observation = vec![0.0; n];
                    let mut at = 0;
                    for (j, &size) in sizes.iter().enumerate() {
                        let lambda = 2.0 * PI * (j + 1) as f64 / period;
                        observation[at] = 1.0;
                        if size == 1 {
                            transition[at * n + at] = -1.0;
                        } else {
                            let (sin, cos) = lambda.sin_cos();
                            transition[at * n + at] = cos;
                            transition[at * n + at + 1] = sin;
                            transition[(at + 1) * n + at] = -sin;
                            transition[(at + 1) * n + at + 1] = cos;
                        }
                        at += size;
                    }
                    Some(Block {
                        dimension: n,
                        transition,
                        observation,
                        initial_mean: vec![0.0; n],
                        noise: vec![(scale * scale, vec![1.0; n])],
                        coefficients: None,
                    })
                }
                ComponentKind::Autoregressive { order } => {
                    let rho =
                        self.coefficients(c.coefficients, vec![1.0; order], &label("coefficients"));
                    let scale = self.scale(c.scale, 0.05 * s, &label("scale"));
                    parameters.push(("coefficients".to_string(), rho.id()));
                    parameters.push(("scale".to_string(), scale.id()));
                    let n = order;
                    let mut transition = vec![0.0; n * n];
                    for i in 1..n {
                        transition[i * n + i - 1] = 1.0;
                    }
                    Some(Block {
                        dimension: n,
                        transition,
                        observation: first(n),
                        initial_mean: vec![0.0; n],
                        noise: vec![(scale * scale, first(n))],
                        coefficients: Some(rho),
                    })
                }
                ComponentKind::Regression { covariates } => {
                    let design = c.design.clone().expect("checked");
                    let scales = (0..covariates)
                        .map(|j| 10.0 * s / column_max(design.as_slice(), covariates, j))
                        .collect();
                    let weights = self.coefficients(c.coefficients, scales, &label("weights"));
                    parameters.push(("weights".to_string(), weights.id()));
                    regressions.push((design.clone(), weights));
                    regression = Some((design, weights.id()));
                    None
                }
            };
            let dimension = block.as_ref().map_or(0, |block| block.dimension);
            let (design, weights) = regression.unzip();
            layouts.push(ComponentLayout {
                name: c.name.clone(),
                kind: c.kind,
                states: offset..offset + dimension,
                observation: block
                    .as_ref()
                    .map_or_else(Vec::new, |block| block.observation.clone()),
                parameters,
                design,
                weights,
            });
            if let Some(block) = block {
                blocks.push((offset, block));
            }
            offset += dimension;
        }
        let d = offset;
        let noise_scale = self.scale(self.noise_scale, 0.1 * s, "observation_noise_scale");

        // the joint matrices: block diagonal transition and noise, observation rows side by side
        let mut transition = vec![0.0; d * d];
        let mut observation = vec![0.0; d];
        let mut initial_mean = vec![0.0; d];
        let mut variance: Option<DoubleVar<'b>> = None;
        let mut ar_parts: Vec<DoubleVar<'b>> = Vec::new();
        for (at, block) in &blocks {
            let n = block.dimension;
            for i in 0..n {
                transition[(at + i) * d + at..(at + i) * d + at + n]
                    .copy_from_slice(&block.transition[i * n..(i + 1) * n]);
            }
            observation[*at..at + n].copy_from_slice(&block.observation);
            initial_mean[*at..at + n].copy_from_slice(&block.initial_mean);
            for (v, mask) in &block.noise {
                let mut placed = vec![0.0; d];
                placed[*at..at + n].copy_from_slice(mask);
                let term = *v * b.constant(DoubleTensor::vector(placed));
                variance = Some(match variance {
                    Some(sum) => sum + term,
                    None => term,
                });
            }
            if let Some(rho) = block.coefficients {
                // e_at ρᵀ S, with S placing the p coefficients in the block's columns
                let mut row = vec![0.0; d];
                row[*at] = 1.0;
                let mut place = vec![0.0; n * d];
                for i in 0..n {
                    place[i * d + at + i] = 1.0;
                }
                let part = b
                    .constant(DoubleTensor::new(&[d, 1], row)?)
                    .matmul(rho.reshape(&[1, n as isize]))
                    .matmul(b.constant(DoubleTensor::new(&[n, d], place)?));
                ar_parts.push(part);
            }
        }
        let mut f = b.constant(DoubleTensor::new(&[d, d], transition)?);
        for part in ar_parts {
            f = f + part;
        }
        let q = variance
            .expect("every state component has a noise scale")
            .diag();
        let m0 = b.constant(DoubleTensor::vector(initial_mean));
        let p0 = b.constant(DoubleTensor::eye(d)?.map(|x| x * s * s));
        let h = b.constant(DoubleTensor::new(&[1, d], observation)?);
        let r = (noise_scale * noise_scale).reshape(&[1, 1]);
        let y = DoubleTensor::new(&[steps, 1], self.observed.clone())?;
        let state_space = [m0.id(), p0.id(), f.id(), q.id(), h.id(), r.id()];
        let likelihood = if regressions.is_empty() {
            let node = b.linear_gaussian_state_space(steps, m0, p0, f, q, h, r);
            b.observe(node, y);
            node
        } else {
            let mut residual = b.constant(y);
            for (design, weights) in &regressions {
                let k = design.shape()[1] as isize;
                residual = residual - b.constant(design.clone()).matmul(weights.reshape(&[k, 1]));
            }
            b.factor(b.log_density(
                DistributionKind::LinearGaussianStateSpace,
                residual,
                &state_space,
            ))
        };
        Ok(StructuralTimeSeries {
            observed: self.observed,
            components: layouts,
            state_dimension: d,
            state_space,
            noise_scale: noise_scale.id(),
            likelihood: likelihood.id(),
            regression: !regressions.is_empty(),
        })
    }
}

/// The largest absolute value in column `j` of a row-major matrix with `columns` columns.
fn column_max(x: &[f64], columns: usize, j: usize) -> f64 {
    x.iter()
        .skip(j)
        .step_by(columns)
        .map(|v| v.abs())
        .fold(0.0, f64::max)
}

/// The unit vector of length `n` selecting the first element.
fn first(n: usize) -> Vec<f64> {
    let mut v = vec![0.0; n];
    v[0] = 1.0;
    v
}
