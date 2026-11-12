//! Keanu's applied demo models, and WebAssembly bindings that run them in a browser.
//!
//! The [`demos`] are plain Rust: an A/B test decision, credit portfolio risk, and volatility with value at risk.
//! Each fit is a NUTS chain ([`run::ChainRun`]) that advances a few transitions per call, so a page can run it in a
//! Web Worker and report progress between calls; reports are JSON.
//!
//! From JavaScript (after `wasm-bindgen --target web`):
//!
//! ```js
//! const test = new AbTestDemo(1500, 75, 1500, 98, 1000, 1);
//! while (!test.step(50)) postMessage({ progress: test.progress() });
//! const report = JSON.parse(test.report());
//! ```

pub mod demos;
pub mod run;

use demos::ab_test::{AbTest, Arm};
use demos::credit::{self, CreditModel, Loan, Outcome};
use demos::volatility::{self, VolatilityKind, VolatilityModel};
use wasm_bindgen::prelude::*;

fn js_error(error: impl std::fmt::Display) -> JsError {
    JsError::new(&error.to_string())
}

fn json(value: &impl serde::Serialize) -> Result<String, JsError> {
    serde_json::to_string(value).map_err(js_error)
}

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

/// Credit risk of a loan book, from a synthetic repayment history.
#[wasm_bindgen]
#[derive(Debug)]
pub struct CreditDemo {
    model: CreditModel,
    history: Vec<Outcome>,
    book: Vec<Loan>,
}

#[wasm_bindgen]
impl CreditDemo {
    /// Generates a history of `history_loans` past loans and a book of `book_loans` open ones from `seed`, and
    /// starts fitting the default model with `draws` posterior draws.
    #[wasm_bindgen(constructor)]
    pub fn new(
        history_loans: u32,
        book_loans: u32,
        draws: u32,
        seed: u32,
    ) -> Result<CreditDemo, JsError> {
        let history = credit::history(history_loans as usize, seed.into());
        let book = credit::book(book_loans as usize, seed.into());
        Ok(Self {
            model: CreditModel::new(&history, draws as usize, seed.into()).map_err(js_error)?,
            history,
            book,
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

    /// The open loans as JSON.
    pub fn book(&self) -> Result<String, JsError> {
        json(&self.book)
    }

    /// The book's risk with every default log-odds shifted by `stress`, and the probability that more than
    /// `default_threshold` loans default, as JSON. Uses at most 400 posterior draws with 50 scenarios each.
    pub fn report(&self, stress: f64, default_threshold: u32) -> Result<String, JsError> {
        let report = self
            .model
            .portfolio(
                &self.history,
                &self.book,
                stress,
                default_threshold.into(),
                400,
                50,
            )
            .map_err(js_error)?;
        json(&report)
    }
}

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
