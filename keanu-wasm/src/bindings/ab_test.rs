//! Browser bindings for [`crate::demos::ab_test`].

use crate::demos::ab_test::{AbTest, Arm};
use crate::{js_error, json};
use wasm_bindgen::prelude::*;

/// An A/B test of two variants' conversions.
#[wasm_bindgen]
#[derive(Debug)]
pub struct AbTestDemo {
    test: AbTest,
}

#[wasm_bindgen]
impl AbTestDemo {
    /// Starts a fit with `draws` posterior draws (after as many adaptation steps).
    #[wasm_bindgen(constructor)]
    pub fn new(
        visitors_a: u32,
        conversions_a: u32,
        visitors_b: u32,
        conversions_b: u32,
        draws: u32,
        seed: u32,
    ) -> Result<AbTestDemo, JsError> {
        let arm = |visitors: u32, conversions: u32| Arm {
            visitors: visitors.into(),
            conversions: conversions.into(),
        };
        Ok(Self {
            test: AbTest::new(
                arm(visitors_a, conversions_a),
                arm(visitors_b, conversions_b),
                draws as usize,
                seed.into(),
            )
            .map_err(js_error)?,
        })
    }

    /// Runs up to `steps` more transitions; true once finished.
    pub fn step(&mut self, steps: u32) -> Result<bool, JsError> {
        self.test.chain().advance(steps as usize).map_err(js_error)
    }

    /// The fraction of transitions run.
    pub fn progress(&mut self) -> f64 {
        self.test.chain().progress()
    }

    /// The decision report as JSON.
    pub fn report(&self) -> Result<String, JsError> {
        json(&self.test.report())
    }
}
