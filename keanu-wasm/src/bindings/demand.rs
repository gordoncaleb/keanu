//! Browser bindings for [`crate::demos::demand`].

use crate::demos::demand::{self, DemandModel, Sales};
use crate::{js_error, json};
use wasm_bindgen::prelude::*;

/// Simulated futures per posterior draw.
const SCENARIOS: usize = 20;

/// Stock to cover demand for one product, from a synthetic year of daily sales.
#[wasm_bindgen]
#[derive(Debug)]
pub struct DemandDemo {
    model: DemandModel,
    sales: Sales,
}

#[wasm_bindgen]
impl DemandDemo {
    /// Generates `periods` four-week periods of daily sales from `seed` (13 is a year), and starts fitting the
    /// demand model with `draws` posterior draws.
    #[wasm_bindgen(constructor)]
    pub fn new(periods: u32, draws: u32, seed: u32) -> Result<DemandDemo, JsError> {
        let sales = demand::sales(periods as usize, seed.into()).map_err(js_error)?;
        Ok(Self {
            model: DemandModel::new(&sales.history, draws as usize, seed.into())
                .map_err(js_error)?,
            sales,
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

    /// The stock that covers the next `period` days' demand (1 to 56) with probability `service_level`, with the
    /// history fit and forecast, as JSON. The first call simulates 20 futures per posterior draw; later calls reuse
    /// them.
    pub fn report(&self, period: u32, service_level: f64) -> Result<String, JsError> {
        let report = self
            .model
            .report(&self.sales, period as usize, service_level, SCENARIOS)
            .map_err(js_error)?;
        json(&report)
    }
}
