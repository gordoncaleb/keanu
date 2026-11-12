//! Browser bindings for [`crate::demos::epidemic`].

use crate::demos::epidemic::{self, EpidemicModel, Intervention};
use crate::{js_error, json};
use wasm_bindgen::prelude::*;

/// Hospital demand in an outbreak, forecast from its first weeks of synthetic reported cases.
#[wasm_bindgen]
#[derive(Debug)]
pub struct EpidemicDemo {
    model: EpidemicModel,
}

#[wasm_bindgen]
impl EpidemicDemo {
    /// Generates `days` days of reported cases from `seed`, and starts fitting the outbreak model with `draws`
    /// posterior draws.
    #[wasm_bindgen(constructor)]
    pub fn new(days: u32, draws: u32, seed: u32) -> Result<EpidemicDemo, JsError> {
        let cases = epidemic::reported_cases(days as usize, seed.into());
        Ok(Self {
            model: EpidemicModel::new(&cases, draws as usize, seed.into()).map_err(js_error)?,
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

    /// The forecast with `beds` beds, with contacts cut by the fraction `cut` from day `from_day` on, as JSON.
    pub fn report(&self, beds: f64, cut: f64, from_day: u32) -> Result<String, JsError> {
        let intervention = Intervention {
            cut: cut.clamp(0.0, 1.0),
            from_day: from_day as usize,
        };
        json(&self.model.report(beds, intervention).map_err(js_error)?)
    }
}
