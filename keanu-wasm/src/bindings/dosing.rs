//! Browser bindings for [`crate::demos::dosing`].

use crate::demos::dosing::{self, DosingModel};
use crate::{js_error, json};
use wasm_bindgen::prelude::*;

/// Dosing a fictional drug for a synthetic patient, from a few blood samples after the first dose.
#[wasm_bindgen]
#[derive(Debug)]
pub struct DosingDemo {
    model: DosingModel,
}

#[wasm_bindgen]
impl DosingDemo {
    /// Simulates the patient's blood samples from `seed` and starts fitting the model with `draws` posterior draws.
    #[wasm_bindgen(constructor)]
    pub fn new(draws: u32, seed: u32) -> Result<DosingDemo, JsError> {
        let samples = dosing::patient_samples(seed.into());
        Ok(Self {
            model: DosingModel::new(samples, draws as usize, seed.into()).map_err(js_error)?,
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

    /// The fit, and the outcome of `dose` mg every `interval` hours at steady state, as JSON. With
    /// `population_only`, the decision ignores the patient's samples and uses the population alone.
    pub fn report(
        &self,
        dose: f64,
        interval: f64,
        population_only: bool,
    ) -> Result<String, JsError> {
        json(
            &self
                .model
                .report(dose, interval, population_only)
                .map_err(js_error)?,
        )
    }
}
