# Keanu

Keanu is a probabilistic programming library for Rust. You describe a model as a graph of random variables and the
deterministic operations that combine them, observe some of them, and then find the most probable values of the
rest (MAP optimisation) or sample from their posterior (MCMC: NUTS, Metropolis–Hastings).

Every variable holds an n-dimensional tensor, and gradients come from automatic differentiation, so models with
vectors and matrices of variables work with gradient-based inference.

## Quick start

Two thermometers read a room's temperature, each with its own noise. What is the temperature?

```rust
use keanu::prelude::*;

fn main() -> keanu::Result<()> {
    // describe the model
    let b = GraphBuilder::new();
    let temperature = b.uniform(20.0, 30.0).label("temperature");
    let first = b.gaussian(temperature, 2.5);
    let second = b.gaussian(temperature, 5.0);
    b.observe(first, DoubleTensor::scalar(25.0));
    b.observe(second, DoubleTensor::scalar(30.0));
    let temperature = temperature.id();

    // build it, starting from a value inside the prior's support
    let (graph, mut state) = b.build()?;
    state.set(&graph, temperature, DoubleTensor::scalar(25.0))?;
    let model = Model::new(graph, &state)?;

    // the most probable temperature
    let map = GradientOptimizer::new(&model).max_a_posteriori(&mut state)?;
    let most_probable = map.get(temperature).unwrap().to_scalar().unwrap();
    assert!((most_probable - 26.0).abs() < 1e-3);

    // samples from the posterior
    let mut rng = KeanuRng::seed_from_u64(1);
    let samples = Nuts::default().sample(&model, &mut state, &[temperature], 2000, &mut rng)?;
    let mean = samples.f64(temperature).unwrap().mean()?.to_scalar().unwrap();
    assert!((mean - 26.0).abs() < 0.5);
    Ok(())
}
```

Add it to a project with `keanu = { git = "https://github.com/gordoncaleb/keanu" }` (it isn't on crates.io yet).

## Crates

| Crate | Contents |
|---|---|
| [`keanu`](keanu) | The facade most programs depend on, with a `prelude` |
| [`keanu-tensor`](keanu-tensor) | N-dimensional `f64` / `i64` / `bool` tensors: numpy-style broadcasting and slicing, reductions, linear algebra |
| [`keanu-distributions`](keanu-distributions) | Distributions: log probabilities, gradients and samplers, and `KeanuRng` |
| [`keanu-graph`](keanu-graph) | Building graphs, their `State`, value propagation, autodiff, log probabilities, `Model`, sequences (plates) |
| [`keanu-algorithms`](keanu-algorithms) | Optimizers (Adam, conjugate gradient, BOBYQA), MCMC (NUTS, Metropolis–Hastings, simulated annealing), forward sampling, the particle filter, regression models, sample statistics |
| [`keanu-io`](keanu-io) | Saving and loading models (JSON, protobuf) and Graphviz DOT export |
| [`keanu-parity`](keanu-parity) | Test-only: the golden vectors recorded from the Java implementation |

A `Graph` is immutable and `Send + Sync`, and all values live in a separate, cloneable `State`, so many chains can
share one graph. There is no global state and no native code.

## Features

Features of the `keanu` crate:

- `json` (default): load and save models as JSON.
- `protobuf`: load and save models as binary protobuf.
- `dot`: Graphviz DOT export.
- `parallel`: run independent MCMC chains in parallel with rayon.

## Documentation

- The guide, an mdBook in [`book/`](book): `mdbook serve book`. Its code samples are compiled and run as tests.
- API docs: `cargo doc --open`.
- [`docs/`](docs): design decisions, benchmarks against the Java implementation, and the API review.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): the checks to run before committing.

The minimum supported Rust version is **1.85**.

## History

Keanu began as a Java library by Improbable. This repository is its Rust rewrite, which keeps the Java version's
model semantics and was checked against results recorded from it ([`RUST_MIGRATION_PLAN.md`](RUST_MIGRATION_PLAN.md)
describes the process). The last version containing the Java implementation is the git tag **`java-final`**
(`git checkout java-final`).

## Licence

MIT (see [`LICENSE`](LICENSE)), except the BOBYQA solver in `keanu-algorithms`, which is a translation of
[PRIMA](https://github.com/libprima/prima)'s and stays under PRIMA's BSD 3-Clause licence (see
[`keanu-algorithms/THIRD-PARTY-NOTICES.md`](keanu-algorithms/THIRD-PARTY-NOTICES.md)).
