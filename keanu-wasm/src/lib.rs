//! Keanu's applied demo models, and WebAssembly bindings that run them in a browser.
//!
//! The [`demos`] are plain Rust, one module per demo, and [`bindings`] has a JavaScript class for each. A fit is
//! usually a NUTS chain ([`run::ChainRun`]) that advances a few transitions per call, so a page can run it in a Web
//! Worker and report progress between calls. Reports are JSON.
//!
//! Every demo class has the same shape, which the site's worker relies on: a constructor that starts the fit,
//! `step(n)` that runs up to n more transitions and returns true once finished, `progress()` from 0 to 1, and
//! `report(...)` returning JSON. From JavaScript (after `wasm-bindgen --target web`):
//!
//! ```js
//! const test = new AbTestDemo(1500, 75, 1500, 98, 1000, 1);
//! while (!test.step(50)) postMessage({ progress: test.progress() });
//! const report = JSON.parse(test.report());
//! ```

pub mod bindings;
pub mod demos;
pub mod run;

use wasm_bindgen::JsError;

pub(crate) fn js_error(error: impl std::fmt::Display) -> JsError {
    JsError::new(&error.to_string())
}

pub(crate) fn json(value: &impl serde::Serialize) -> Result<String, JsError> {
    serde_json::to_string(value).map_err(js_error)
}
