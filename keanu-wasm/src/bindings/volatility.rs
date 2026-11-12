//! Browser bindings for [`crate::demos::volatility`].

use crate::demos::volatility::{self, VolatilityKind, VolatilityModel};
use crate::{js_error, json};
use wasm_bindgen::prelude::*;

/// Volatility and value at risk of a daily return series.
#[wasm_bindgen]
#[derive(Debug)]
pub struct VolatilityDemo {
    model: VolatilityModel,
}

#[wasm_bindgen]
impl VolatilityDemo {
    /// Starts a NUTS fit of `returns` (fractions, 0.01 is 1%) with `draws` posterior draws: GARCH(1,1) when
    /// `stochastic` is false (about a second), else stochastic volatility (tens of seconds).
    #[wasm_bindgen(constructor)]
    pub fn new(
        returns: Vec<f64>,
        stochastic: bool,
        draws: u32,
        seed: u32,
    ) -> Result<VolatilityDemo, JsError> {
        let kind = if stochastic {
            VolatilityKind::Stochastic
        } else {
            VolatilityKind::Garch
        };
        Ok(Self {
            model: VolatilityModel::new(kind, &returns, draws as usize, seed.into())
                .map_err(js_error)?,
        })
    }

    /// The bundled sample of daily returns.
    #[wasm_bindgen(js_name = sampleReturns)]
    pub fn sample_returns() -> Vec<f64> {
        volatility::sample_returns()
    }

    /// Runs up to `steps` more NUTS transitions; true once finished.
    pub fn step(&mut self, steps: u32) -> Result<bool, JsError> {
        self.model.chain().advance(steps as usize).map_err(js_error)
    }

    /// The fraction of transitions run.
    pub fn progress(&mut self) -> f64 {
        self.model.chain().progress()
    }

    /// The report for a position worth `position`, with a `horizon`-day forecast, as JSON.
    pub fn report(&self, position: f64, horizon: u32) -> Result<String, JsError> {
        let report = self
            .model
            .report(position, horizon as usize, 20)
            .map_err(js_error)?;
        json(&report)
    }
}
