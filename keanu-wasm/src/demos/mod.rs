//! Applied demo models, as plain Rust: each builds its model, fits it with a [`ChainRun`](crate::run::ChainRun)
//! (or an approximation), and reports the quantities a decision maker needs.

pub mod ab_test;
pub mod credit;
pub mod demand;
pub mod dosing;
pub mod epidemic;
pub mod reliability;
pub mod volatility;
