//! Browser bindings for [`crate::demos::reliability`].

use crate::demos::reliability::{self, ReliabilityModel};
use crate::{js_error, json};
use wasm_bindgen::prelude::*;

/// When a fleet of pumps will fail, from a synthetic history of failures and pumps still running.
#[wasm_bindgen]
#[derive(Debug)]
pub struct ReliabilityDemo {
    model: ReliabilityModel,
}

#[wasm_bindgen]
impl ReliabilityDemo {
    /// Generates a history of `units` pumps from `seed` and starts fitting the lifetime model with `draws` posterior
    /// draws.
    #[wasm_bindgen(constructor)]
    pub fn new(units: u32, draws: u32, seed: u32) -> Result<ReliabilityDemo, JsError> {
        let history = reliability::fleet(units as usize, seed.into());
        Ok(Self {
            model: ReliabilityModel::new(history, draws as usize, seed.into()).map_err(js_error)?,
        })
    }

    /// Runs up to `steps` more transitions; true once finished.
    pub fn step(&mut self, steps: u32) -> Result<bool, JsError> {
        self.model.chain().advance(steps as usize).map_err(js_error)
    }

    /// The fraction of transitions run.
    pub fn progress(&mut self) -> f64 {
        self.model.chain().progress()
    }

    /// The fleet's reliability and the claims on a `warranty_months` warranty for `fleet_size` new pumps at
    /// `cost_per_claim` each against `budget`, as JSON. Simulates 20 claim counts per posterior draw.
    pub fn report(
        &self,
        warranty_months: f64,
        fleet_size: u32,
        cost_per_claim: f64,
        budget: f64,
    ) -> Result<String, JsError> {
        let report = self
            .model
            .report(
                warranty_months,
                fleet_size.into(),
                cost_per_claim,
                budget,
                20,
            )
            .map_err(js_error)?;
        json(&report)
    }
}
