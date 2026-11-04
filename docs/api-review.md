# API review (Phase 7, step b)

A review of the public API of the published crates: `keanu`, `keanu-tensor`, `keanu-distributions`,
`keanu-graph`, `keanu-algorithms` and `keanu-io`. The aim was a surface that can stay semver-stable from the
first release, with idiomatic Rust names and no Java idioms.

## Changes

### Room to grow without breaking changes

- **Enums that will gain variants are `#[non_exhaustive]`.** Users matching on them need a `_` arm:
  - errors (already): `TensorError`, `DistributionError`, `GraphError`, `AlgorithmError`, `IoError`, `keanu::Error`
  - values and types: `Value`, `DType`
  - op families: `UnaryOp`, `BinaryOp`, `CompareOp`, `LogicOp`, `Predicate`, `ReduceOp`, `ScanOp`, `ArgOp`, `Op`,
    `DistributionKind`
  - algorithm choices: `Algorithm`, `Sampler`, `Fitter`, `Regularization`, `UpdateFormula`, `ConvergenceChecker`,
    `AnnealingSchedule`, `VariableSelector`, `RejectionStrategy`, the PRIMA `ExitStatus`
- **Closed sets stay exhaustive**, so they can be matched without a wildcard: `Fitness` (MAP or maximum likelihood)
  and `Norm` (L2 or max).
- **Settings structs are `#[non_exhaustive]` and use setters.** New settings can then be added in a minor release.
  Every field stays public and readable, and each one has a `#[must_use]` `with_*` setter:
  - The structs: `Nuts`, `QuadraticPotential`, `MetropolisHastings`, `SimulatedAnnealing`, `ForwardSampler`,
    `SamplingOptions`, `Adam`, `ConjugateGradient`, `Bobyqa`, the PRIMA `Settings`, `ParticleFilter`,
    `SamplingFitting`.
  - Usage: `Nuts::default().with_adapt_count(500).with_max_tree_height(8)`.
  - `Option` fields take the inner value (`with_initial_step_size(0.1)`).
  - `MetropolisHastings` is the exception: its builder methods are `proposal(impl ProposalDistribution)`,
    `shared_proposal(Arc<dyn …>)`, `selector` and `rejection` (one set of setters, as used in the examples).
  - `SamplingFitting::options` was renamed `with_options` for consistency.
- **Result structs are `#[non_exhaustive]`**, because users read them but don't construct them: `OptimizedResult`,
  `MapEstimate`, `StepResult`, `NutsStatistics`, `Summary`, the PRIMA `Outcome`, `Loaded`, `NodeLogProbGradient`,
  `DistributionOp`.
- **Distribution parameter and gradient structs stay exhaustive** (`Gaussian { mu, sigma }`,
  `GaussianGradient { x, mu, sigma }`, …). Their fields are the mathematical definition of the distribution, and
  users construct them directly.

### A smaller surface

- **`keanu-io` no longer exposes the prost-generated protobuf types** (`proto`, `ProtoModel`) or the
  `load_model`/`save_model` functions that took and returned them. The public API is the format-level functions
  (`load_protobuf`, `save_protobuf`, `load_json`, `save_json`) plus `to_dot`. A prost upgrade or a schema change
  therefore can't break users.
- **The loader and saver modules are compiled only with the `protobuf` or `json` feature.** Building with
  `default-features = false, features = ["dot"]` no longer compiles dead code.
- **`unreachable_pub` reports nothing:** every `pub` item is reachable from a crate root.

### Errors

- **Every crate has one error type** that implements `std::error::Error`, `Display`, `Debug`, `Clone` and
  `PartialEq`, and is `#[non_exhaustive]`. The facade's `keanu::Error` converts from each of them.
- **Wrapped errors are transparent**, as `thiserror`'s `#[error(transparent)]` is. `Display` shows the wrapped
  error's message and `source()` is the wrapped error's source, so error reports don't repeat a message.
  `GraphError`, `IoError` and `keanu::Error` were missing `source()`; the others returned the wrapped error and also
  printed its message.
- **`GraphError::NoPartial { of, wrt }` and `GraphError::MissingValue(..)` hold `NodeId`s** instead of bare `usize`
  indices.
- **The panic audit** covered every `unwrap`, `expect`, `panic!` and `unreachable!` in non-test code:
  - `keanu_io` saving a node type with no recorded schema name now returns `IoError::NotSaveable` instead of
    panicking.
  - Simulated annealing reading a latent without a value now returns `GraphError::MissingValue`.
  - The remaining sites are internal invariants, each with a message that states the invariant. Examples: a tensor
    built from a length computed from its own shape; offsets whose first element is 0; a parent that is always
    present while sorting; a regression model's own state, which is only exposed immutably.
  - Matches on the newly `#[non_exhaustive]` enums in `keanu-io`'s saver return `NotSaveable` for variants added
    later, rather than panicking.
- **A property-based fuzz test, `keanu-graph/tests/no_panic.rs`, checks that random graphs never panic.**
  - It builds random graphs from 23 kinds of step with random (often invalid) shapes, axes, slices, permutations
    and parameters. Construction, `build`, evaluation, the joint log prob and reverse-mode autodiff must return
    `Result`s and never panic.
  - A second property builds deep graphs whose steps are always valid, and checks that both autodiff modes run on
    them.
  - It found two panics, both now fixed:
    - **Linear algebra on 0 × 0 matrices** (`chunks(0)`). 0 × 0 now follows numpy: the inverse and Cholesky factor
      are empty, and the determinant is 1. Regression test: `keanu-tensor/tests/empty_matrices.rs`.
    - **Autodiff through index-mapping ops when the "of" node is empty.** A shape with a 0 dimension made the batch
      size 0, and the code divided by it. Regression test: `keanu-graph/tests/empty_tensors.rs`.
  - After the fixes, 20 consecutive runs (about 28,000 random graphs) passed.

### Naming

- **Names were checked against Rust conventions:** no `get_` prefixes, `with_*` for setters, `to_*`/`as_*`/`into_*`
  for conversions, and `is_*` for predicates.
- **Java's misspelling `adaption_window` is now `adaptation_window`**, on `QuadraticPotential` and its setter.
- **`where_` keeps its trailing underscore**, because `where` is a keyword. It's documented and consistent with
  `GraphBuilder`'s other combinators.
- **New method `Value::to_double_tensor()`** converts any value to f64 (integers as f64, booleans as 0.0 / 1.0).
  Three test helpers had been duplicating it.

### Traits and thread safety

- **`keanu/tests/api.rs` pins the contract:**
  - The core types are `Send + Sync`: tensors, `Graph`, `State`, `Model`, `Value`, `KeanuRng`, `NetworkSamples`,
    the settings structs, `OptimizedResult`, distributions and `keanu::Error`.
  - Settings, values and distributions are `Debug + Clone`.
  - The settings use the `with_*` style.
  - Errors are transparent, and invalid input returns an error rather than panicking.
- **`missing_debug_implementations`** is a workspace lint, and **`#![deny(missing_docs)]`** is set on every
  published crate.

## Dependencies and packaging

- **`faer` is used with `default-features = false`** and only the features Keanu needs (`std`, `rayon`,
  `linalg`). That removes sparse linear algebra, `rand` integration and `.npy` file IO, cutting the normal
  dependency tree from 129 to 103 crates.
- **Inter-crate dependencies go through `[workspace.dependencies]`**, each with a `version`, so they can be
  published.
- **Every published crate** has a description, README, `LICENSE`, keywords, categories, a docs.rs link and docs.rs
  metadata (`all-features`, `--cfg docsrs` with `doc_cfg`), plus an `include` list. Packages contain only sources,
  the README, the licence, and where relevant the PRIMA notices, the `.proto` schema or the example data.

## Minimum supported Rust version

**1.85**, the minimum for edition 2024. It's declared once as `workspace.package.rust-version` and inherited by
every crate. To reach it, these changes were made, none of them changing behaviour:

| Needed | Change |
|---|---|
| 1.94 | `std::f64::consts::EULER_GAMMA` and `GOLDEN_RATIO` replaced by local constants of the same value |
| 1.88 | four let-chains (`if let … && …`) written as nested `if`s |
| 1.87 | `is_multiple_of` replaced by `%` (clippy's suggestion respects the declared `rust-version`) |

Clippy uses `rust-version` to avoid suggesting newer APIs. The libraries build on 1.85 with every feature
combination (`cargo hack --rust-version`).

The test suite runs on the latest stable, because dev-dependencies such as `proptest` need 1.88 or later.

## Deferred

- **`cargo semver-checks`** needs a published baseline, so it can't run until the first release. From then on, it
  runs before every release against the previous one; see `CONTRIBUTING.md`.
- **Our own crates aren't licence-checked by `cargo deny`** while they're `publish = false`, which cargo-deny
  treats as private. The `publish` flag stays `false` until the first release is approved, as a guard against
  accidental publishing.
