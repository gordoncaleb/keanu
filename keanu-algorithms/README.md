# keanu-algorithms

Part of [Keanu](https://crates.io/crates/keanu), a probabilistic programming library. MAP and maximum likelihood optimizers (Adam, conjugate gradient, BOBYQA), MCMC samplers (Metropolis–Hastings, NUTS), simulated annealing, a particle filter, regression models and sample statistics.

Most users should depend on the [`keanu`](https://crates.io/crates/keanu) facade crate, which re-exports this one.

Licensed under MIT, except `src/optimizer/bobyqa/prima/`, a translation of PRIMA's BOBYQA solver, which is under PRIMA's BSD 3-Clause licence: see `THIRD-PARTY-NOTICES.md`.
