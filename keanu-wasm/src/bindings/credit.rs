//! Browser bindings for [`crate::demos::credit`].

use crate::demos::credit::{self, CreditModel, Loan, Outcome};
use crate::{js_error, json};
use wasm_bindgen::prelude::*;

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
