# Keanu → Rust migration plan

Status: proposal. Written once the Java code base had been reduced to a pure JVM implementation (ND4J, OpenBLAS and
the Python bindings removed) and the known correctness bugs fixed (see `release_notes.md`, "Unreleased").

## End state

**This repository contains only Rust.** All Java, Kotlin and Gradle code is deleted (Phase 8), and Keanu is
published to crates.io as a library that other Rust projects use with:

```toml
[dependencies]
keanu = "1"
```

The Java implementation is a temporary reference used to prove parity. It is not a product the Rust version has
to interoperate with or stay compatible with.

## Goals

- A pure Rust implementation of Keanu with the same modelling capability and numerically equivalent results.
- **Library first.** Keanu should be a well-behaved dependency:
  - no global mutable state (Java's static vertex ID counter goes away)
  - graphs that are `Send + Sync`, with per-chain state, so callers can run models on their own threads
  - a semver-stable public API, documented on docs.rs
  - a small default dependency tree, with optional features for heavier parts (serialisation, parallelism)
  - a declared minimum supported Rust version (MSRV)
- No native (C/Fortran) dependencies, the same constraint the Java version now has.
- Migrate incrementally, with the Java implementation as the reference until the Rust version reaches parity,
  then delete Java.

Non-goals: Java or Python bindings, a source-compatible Java API, GPU support, or bit-for-bit reproduction of Java
random number streams. Samplers can only be compared statistically.

## What is being migrated

About 41k lines of main code in `keanu-project` (Java plus a little Kotlin), and a similar amount of tests.

| Area | Main LOC | Package | Notes |
|---|---|---|---|
| Vertices / graph / autodiff | ~15.9k | `vertices` | ~150 vertex types, forward and reverse mode autodiff, `LogProbGraph` |
| Tensors | ~9.4k | `tensor` | JVM double/int/long/boolean/generic tensors, broadcasting, slicing, linear algebra (`KeanuLapack`) |
| Algorithms | ~6.1k | `algorithms` | MH, NUTS, simulated annealing, forward sampling, particle filter, Adam / conjugate gradient / BOBYQA optimizers, statistics |
| Utilities / IO | ~3.0k | `util` | JSON/protobuf/MIR/DOT/CSV save and load, status bar |
| Distributions | ~2.4k | `distributions` | ~30 continuous and discrete distributions: logProb, dLogProb, sampling |
| Compiled backend | ~1.5k | `backend` | Generates Java source and compiles it at runtime with jOOR |
| Network / model / templating | ~2.9k | `network`, `model`, `templating` | `BayesianNetwork`, `KeanuProbabilisticModel`, lambda models, `Sequence` (plates) |
| Kotlin operator overloads | ~0.3k | `kotlin` | Become `std::ops` impls in Rust |

External libraries that need Rust equivalents:

| Java dependency | Used for | Rust replacement |
|---|---|---|
| commons-math3 special functions | logGamma, digamma, trigamma, beta, erf | `statrs` / `puruspe`; trigamma and tetragamma ported from Java (`SpecialFunctions`) |
| commons-math3 optimizers | Conjugate gradient, BOBYQA | `argmin` for CG / L-BFGS; BOBYQA has no mature pure-Rust crate, so port Apache's implementation (~2.5k lines) or replace it with `cobyla` |
| commons-math3 distributions (tests only) | Reference densities | `statrs` in tests |
| Guava, commons-lang3, Lombok | Collections, preconditions, boilerplate | std library |
| protobuf-java | Model serialisation (`mir.proto`, `BayesNet.proto`) | `prost` (same `.proto` files) |
| opencsv, gson | CSV / JSON IO | `csv`, `serde_json` |
| jOOR runtime compilation | Compiled computable graph | Interpreted op list (see Phase 5) |

## Strategy

**Bottom-up, with parity tests at each layer.** The dependency order is strict:

`tensor → distributions → graph + autodiff → model → algorithms → IO`

Each layer is ported, then held to the Java implementation by golden test vectors before the next layer starts.
This keeps every comparison deterministic until the algorithm layer, where only statistical comparison is possible.

**Use the serialised model format as the boundary between the two implementations.** Keanu already saves models as
protobuf (`BayesNet.proto`) and JSON. The Rust version should load exactly those files. Any model built in Java can
then be run in Rust and the results compared, with no FFI bridge.

**No Java interop.** Because the end state is Rust only, there is no JNI or FFM bridge. Java and Rust only
meet through files: the golden vectors and saved models. That keeps the Java side untouched and makes it simple to
delete.

**The golden vectors outlive Java.** While Java exists, it regenerates and verifies them. In Phase 8 they are frozen
and become ordinary checked-in test fixtures of the Rust test suite, so parity stays enforced after the Java code is
gone.

## Rust design decisions

1. **Tensors.**
   - **Storage:** use one `Tensor<T>` type, with row-major contiguous storage, over `f64`, `i32`, `i64` and `bool`. Build it on `ndarray`, or write it in-house as a thin wrapper.
   - **Semantics:** specify broadcasting, slicing and reductions to match the (now numpy-consistent) Java behaviour. Pin that behaviour with the golden vectors.
   - **Linear algebra:** use `faer` for matmul, LU and Cholesky (pure Rust, fast). The pure-Java `KeanuLapack` is the reference for semantics such as triangle handling and pivot conventions.

2. **Graph representation.**
   - **Today:** Java vertices are objects holding mutable values and parent/child pointers.
   - **In Rust:** use an arena:
     - `Graph { nodes: Vec<Node> }`
     - `Node { op: Op, parents: SmallVec<[NodeId; 3]>, shape }`
     - `Op` is an enum, one variant per vertex type.
   - **Values:** keep them out of the graph, in a separate `State { values: Vec<Value> }`.
   - **Why:**
     - It removes the shared-mutable-state class of bugs found in review: NUTS potential reuse, and the particle filter reading another particle's values.
     - Cloning a state per chain or per particle becomes cheap.
     - Running chains in parallel becomes safe (`Send + Sync` graph, owned states).

3. **Typed handles for the user API.**
   - **Handles:** `DoubleVar`, `IntVar` and `BoolVar` wrap a `NodeId` plus a graph reference.
   - **Operators:** implement them via `std::ops`, replacing both the Java fluent API and the Kotlin overloads.
   - **Generics:** Java's `<T, TENSOR, VERTEX>` generics collapse into these three handle types.

4. **Autodiff.**
   - **Reverse mode:** keep it as the primary path, as a tape built by walking the arena in topological order.
   - **Rules:** each `Op` implements `forward_partial` and `reverse_partial` with the same partial layout as Java. Forward partials are `[wrt..., of...]`; reverse partials are `[of..., wrt...]`.
   - **Coverage:** every op the Java version can differentiate must be differentiable in Rust. That now includes `cumSum`, `cumProd`, `tensorMultiply`, Cholesky, `standardize` and the rest.
   - **Lambda vertices:** they become closures with optional user-supplied derivative closures, as in Java.

5. **Distributions.**
   - **Shape:** a `Distribution` trait with `log_prob`, `d_log_prob` and `sample`, plus a parameter struct per distribution.
   - **Graph form:** the `LogProbGraph` (log prob as a graph) becomes a method that emits nodes into the arena.

6. **Randomness.**
   - **Generator:** use `rand` with `rand_pcg` or `ChaCha`, with explicit seeding.
   - **Samplers:** keep Keanu's own samplers (e.g. the Gamma sampler) rather than `rand_distr`, so the distributions are under test.

7. **Errors.**
   - **Recoverable failures:** use `Result` for shape mismatches and non-positive-definite matrices.
   - **Bugs only:** panic only on invariant violations.
   - **Supersedes:** this replaces Java's mix of `IllegalArgumentException`, `IllegalStateException` and NaN returns.

8. **Crate layout (Cargo workspace).**

   | Crate | Published | Contents |
   |---|---|---|
   | `keanu` | yes | Facade that re-exports the public API. This is what users depend on |
   | `keanu-tensor` | yes | Tensors and linear algebra. Usable on its own |
   | `keanu-distributions` | yes | Distributions and special functions |
   | `keanu-graph` | yes | Graph, ops and autodiff |
   | `keanu-algorithms` | yes | Optimizers, samplers, statistics |
   | `keanu-io` | yes | Save / load. Behind features so users who don't need it don't pay for `prost` / `serde` |
   | `keanu-parity` | no (`publish = false`) | Test-only crate that checks the golden vectors |

   - **Features on the `keanu` facade:**
     - `serde`: JSON save / load
     - `protobuf`: `.pb` save / load
     - `parallel`: `rayon`, for running chains in parallel
     - `dot`: graph export
   - **Defaults:** `serde` is on by default and the rest are opt-in.
   - **Releases:** the internal crates are versioned in lockstep with `keanu`, and users should depend only on
     `keanu`. Publishing the lower-level crates lets advanced users take, say, just `keanu-tensor`.

## Phases

Each phase ends with its exit criteria passing in CI.

### Phase 0: Prepare the Java side (Java)

**Status: done.** The exporter (`keanu-golden`) and the vectors are in [`golden-vectors/`](golden-vectors/README.md),
and `verifyGoldenVectors` runs as part of `check`. The scope decisions are recorded in the vectors README. Every
Java defect the exporter found was fixed in Java with a regression test, and there are no known issues outstanding.

- Add a golden vector exporter: a Gradle task that runs a fixed list of cases and writes inputs and outputs to JSON.
  It should cover:
  - tensor ops, broadcasting, slicing and linear algebra
  - `logProb` / `dLogProb` per distribution over a grid of parameters
  - vertex forward values and gradients for small graphs
  - saved models (protobuf and JSON) alongside their expected `logProb` / gradient
- Freeze the tensor and vertex semantics, i.e. no behaviour changes without regenerating the vectors.
- Make an explicit decision on deprecated or rarely used features to skip: `ProcessModelVertex`, the CSV POJO
  reader, `PrintVertex`, `AssertVertex`, DOT export.

**Exit:** golden vectors are checked in and generated by CI.

### Phase 1: `keanu-tensor`

**Status: done.** `rust/keanu-tensor` is an in-house tensor (decision record: `rust/docs/decisions/0001-tensor-library.md`),
with `faer` for matmul and Cholesky. It passes every case in `golden-vectors/tensors.json` except two i32-overflow
cases, which don't apply now that integers are 64-bit. It also has property tests for broadcasting, slicing,
permute, reshape and reductions. Differences from Java that are intentional:
- `tril(k)` follows numpy's sign of k (Java's `triLower(k)` was reversed).
- `cholesky_inverse` returns the full symmetric inverse.
- i32 and i64 are merged into i64.

- Port `Tensor`, the double/int/long/bool implementations, `TensorShape`, `Slicer`, broadcasting, reductions,
  `KeanuLapack` semantics (on `faer`), and `FixedPointPow`.
- Add property tests (`proptest`) for the broadcasting and slicing invariants.

**Exit:** 100% of the tensor golden vectors match (exact for integer ops, ≤1e-12 relative for floating point).

### Phase 2: `keanu-distributions`

**Status: done.** `rust/keanu-distributions` ports every distribution: 16 continuous, 7 discrete, plus HalfGaussian,
HalfCauchy and KDE. Each has an element-wise `log_prob`, a typed gradient struct, and Keanu's own samplers on
`KeanuRng` (ChaCha8 with polar Box–Muller Gaussians, passed explicitly with no global state).

It passes all 61 cases in `golden-vectors/distributions.json` with none skipped. It also has 28 statistical sampler
tests and finite-difference checks of every gradient.

Intentional differences from Java:
- `Categorical` is over category indices (`IntTensor`) rather than arbitrary objects.
- `Dirichlet` works on a single probability vector, as the Java implementation effectively did.
- `Uniform` and `UniformInt` have no gradient method, where Java threw.
- `Multinomial` validation is not ported (it was off by default in Java).

- Port the special functions, every distribution's `log_prob`, `d_log_prob` and `sample`, and the KDE.

**Exit:**
- Golden `logProb` / `dLogProb` vectors match.
- Sampler moment tests pass: mean, variance and Kolmogorov–Smirnov per distribution, including the k < 1 Gamma cases.
- Finite-difference checks pass for every `d_log_prob`.

### Phase 3: `keanu-graph`

**Status: done** (design: `rust/docs/decisions/0002-graph-design.md`). `rust/keanu-graph` has:

- the immutable arena `Graph` (`Send + Sync`) and a separate, cloneable `State`;
- `GraphBuilder`, whose typed `DoubleVar`/`IntVar`/`BoolVar` handles support `std::ops`, and which defers
  construction errors to `build()`;
- Java's eval / lazy eval / cascade / observe semantics;
- every deterministic op;
- proxies, lambdas, asserts, multiplexer and CPT;
- every distribution as a node, sampled through eval.

Step 2 adds forward and reverse mode autodiff (`forward_partials`, `reverse_partials`, `reverse_partials_from`) for
every differentiable op. Partials are shaped `[of shape..., wrt shape...]`, and traversal follows Java's
`Differentiator`.

All 121 cases in `golden-vectors/vertices.json` match Java's values, along with all 264 recorded forward and reverse
partials, with none skipped. Finite-difference checks in both modes cover every differentiable op, plus the traversal
rules.

Step 3 adds log probabilities and their gradients:
- `node_log_prob`, `log_prob`, `joint_log_prob`, `log_likelihood` and `log_prior`;
- `node_log_prob_gradient`;
- `log_prob_gradient(of, wrt)`, a port of Java's `LogProbGradientCalculator`;
- `joint_log_prob_gradient` and `log_likelihood_gradient`.

All six models in `golden-vectors/models.json` are rebuilt with the builder (the sequence model through proxies) and
match Java's log prob, log likelihood and both gradient maps, with none skipped. Finite-difference tests cover:
- the gradient with respect to a deterministic parameter expression, for every continuous distribution and for the
  discrete distributions' continuous parameters;
- the gradient with respect to the node's own value, for every continuous distribution;
- a multi-node model with shared parents, broadcasting parameters and an observed node.

Differences from Java that are intentional:

- `where` broadcasts all three operands, whereas Java required then/else to have equal shapes.
- `GetBooleanIndexVertex` isn't ported, because its value-dependent shape can't be a static node shape.
- Java's n-ary `AndMultiple`/`OrMultiple` are chains of `&` / `|`.
- `PrintVertex` and generic-object vertices aren't ported (see the design record).
- `multiplexer` is not differentiable, as in Java (only `where` and the f64 CPT route gradients).
- Lambdas are made differentiable by supplying their Jacobian (`GraphBuilder::lambda_with_jacobian`), rather than
  Java's forward and reverse partial callbacks.
- `UniformVertex`'s gradient with respect to its value is shaped like the value; Java shaped it like `xMax`. The two
  only differ for scalar bounds with a non-scalar value.
- Log probs and gradients read values from `State` and don't evaluate missing ones; evaluate first
  (`State::lazy_eval`). A missing value is a `MissingValue` error.
- Discrete nodes never contribute a gradient with respect to their own value, where Java's `UniformIntVertex` threw.

- Port the arena graph, all ops (vertex types), value propagation (cascade / lazy eval) and observation.
- Port forward and reverse autodiff and `LogProbGraph`.
- Port the generic and boolean ops, and `If` / multiplexer / where.

**Exit:**
- The golden vertex values and gradients match.
- A finite-difference gradient test passes for every differentiable op. Port `AutoDiffRegressionTest` and the per-vertex tests.

### Phase 4: Model layer and IO (`keanu-io`)

**Status: done.**
- **Model layer (`keanu-graph`: `Model`, `LambdaSection`).** It lives in `keanu-graph` rather than a separate
  crate because it's a thin layer over `Graph`, `State` and log probs, and the algorithms crate needs all three.
  `Model` is built from an `Arc<Graph>` and a state; it holds the latent and observed nodes and works on any
  `State`. It implements Java's semantics:
  - joint log prob and log likelihood, with impossible terms and violated observed constraints giving −∞
  - gradients with respect to the continuous latents
  - `set_and_cascade`
  - `log_prob_after(proposal)`, which recomputes only the affected lambda sections, as Java's `logProbAfter`
- **Sequences (`keanu_graph::sequence`).** `SequenceBuilder` (`count`, `from_iter`, `named`,
  `with_initial_state`, `with_transition`), with Java's label scheme and proxy connection rules. The sequence
  identifier counts per `GraphBuilder` (1, 2, ...) instead of being a global, so labels are reproducible.
- **`keanu-io`.** It reads and writes protobuf and JSON in the shared schema (vendored `.proto` files). The
  generated code is committed, so no `protoc` or build script is needed, and a test regenerates it with the pure
  Rust `protox` to catch staleness. Vertex types and parameters use the Java class and parameter names, so files
  round-trip both ways. DOT export is included. Features: `protobuf`, `json` and `dot`, all on by default.
- **Tests.**
  - Every golden model loads from both its `.pb` and its `.json` file. Deterministic values are recomputed in
    Rust, and log prob, log likelihood, both gradients and every labelled node's value match Java to 1e-10. That
    covers the new `catalog` model, which has one instance of each of 139 vertex types.
  - Rust save then Rust load round-trips every model, plus a Rust-built model using Rust-only ops.
  - Rust save then Java load is checked by `RustWrittenModelsTest` against committed Rust-written fixtures.
- **Intentional differences.**
  - Rust ops with no single Java vertex are saved as small Java subgraphs:
    - `neg` as `x * -1`
    - `reciprocal` as `1 / x`
    - `sqrt` as `x ^ 0.5`
    - `transpose` as a permute
    - the full symmetric `cholesky_inverse`
    - half distributions with an explicit zero location
  - On load, Java's `CholeskyInverseVertex`, `AllFalseVertex` and `AnyFalseVertex` become small Rust subgraphs.
  - Not supported: `PrintVertex` and `GetBooleanIndexVertex`; lambda, CPT and categorical nodes (Java can't save
    them either); max, min and standard deviation over some axes, and equality masks (no Java vertex); MIR
    (deferred).
  - Non-finite doubles in JSON are written as `"NaN"` / `"Infinity"` / `"-Infinity"`, as Java does.
- **Java bugs fixed** (see `release_notes.md`):
  - array parameters couldn't be saved
  - nullable arg-reduction and ellipsis parameters weren't nullable
  - `KDEVertex` failed at its own scalar shape
  - `UniformIntVertex` broke gradient models

- Port `BayesianNetwork`, the probabilistic model (with and without gradient), and lambda sections.
- Port loading and saving of `BayesNet.proto` / MIR with `prost`, plus JSON. Load must accept files written by Java.
- Port `Sequence` (plates / templating).

**Exit:** every golden saved model loads in Rust and gives the same `logProb` and gradient as Java.

### Phase 5: `keanu-algorithms`
- Port the optimizers:
  - Adam
  - conjugate gradient (via `argmin`, or ported)
  - BOBYQA (ported)
  - the convergence checkers
- Port the samplers:
  - Metropolis–Hastings: proposal distributions, variable selectors, rollback strategies
  - NUTS: tree, leapfrog, adaptive step size and potential
  - forward sampling
  - simulated annealing
  - the particle filter
- Port the statistics: autocorrelation, effective sample size, summary.
- Replace the jOOR compiled backend with a flattened, topologically ordered op list executed by an interpreter.
  It should be no slower than Java's runtime-compiled code, and it removes runtime code generation entirely.

**Exit:**
- On the golden models, optimizers converge to the same MAP within tolerance.
- Samplers pass the same statistical tests as the Java suite, plus a cross-implementation comparison: matching
  posterior moments within Monte Carlo error over N seeds.
- No `unsafe` outside `faer`.

### Phase 6: Examples, docs and benchmarks
- Port `keanu-examples` (coal mining disasters, custom distribution, Bayesian Methods for Hackers) as Rust examples.
- Port the JMH benchmarks to `criterion`, and publish a Java vs Rust comparison.
- Convert the docs snippets to Rust doctests.

**Exit:** the examples run in CI, and the docs build from doctests.

### Phase 7: Library hardening and first crates.io release
- Review the public API:
  - minimal surface, with crate internals marked `pub(crate)`
  - consistent naming
  - `#[non_exhaustive]` on public enums and error types
  - every public item documented, with `#![deny(missing_docs)]`
- Errors: one error type per crate, implementing `std::error::Error`. No panics on bad user input.
- Check the feature flags build in every combination (`cargo hack --feature-powerset`). Set and test the MSRV.
- Run `cargo semver-checks` in CI, and `cargo deny` for licences and advisories.
- Publish `0.x` releases to crates.io and get early users. Release `1.0` once the API has settled.

**Exit:**
- `keanu` is on crates.io with docs on docs.rs.
- A fresh Rust project can depend on it, build and run the ported examples with no extra setup.

### Phase 8: Delete Java
Precondition: the Rust version passes every golden vector, excluding the known issues, and has had at least one
release used in practice.

1. **Freeze the vectors.** Make a final run of `generateGoldenVectors`, then move `golden-vectors/` into the Rust
   workspace as `keanu-parity` fixtures. From here on they are never regenerated. A deliberate behaviour change in
   Rust updates the affected fixture by hand, with a note in the changelog.
2. **Tag the last Java state** (e.g. `java-final`) so the reference implementation can always be checked out and
   rerun, and say in the README how to get it.
3. **Delete** `keanu-project`, `keanu-golden`, `keanu-examples` (already ported), `benchmarks` (already ported),
   the Java/Kotlin docs snippets, `build.gradle`, `settings.gradle`, `gradle.properties`, `gradle/`, `gradlew*`,
   the Gradle and Java CI configuration, and `release_notes.md`. The latter is replaced by `CHANGELOG.md`.
4. **Replace CI** with Cargo:
   - `cargo fmt --check`
   - `cargo clippy -D warnings`
   - `cargo test --workspace --all-features`
   - feature powerset
   - MSRV
   - `semver-checks`
   - `deny`
   - doc build
5. **Move the Rust workspace to the repository root** and rewrite `README.md` and the docs site for Rust.
6. **Remove the protobuf schema's Java options** (`java_package` and the like), but keep the `.proto` files so saved
   models stay loadable.

**Exit:**
- `git ls-files` contains no `.java`, `.kt` or Gradle files.
- CI is Cargo only.
- `cargo test` checks the frozen golden vectors.

## Rough sizing

Assuming one engineer familiar with both languages. Treat these as relative weights, not commitments.

| Phase | Relative effort |
|---|---|
| 0 Golden vectors | S |
| 1 Tensors | L |
| 2 Distributions | M |
| 3 Graph and autodiff | XL (largest: ~150 op types and both autodiff modes) |
| 4 Model and IO | M |
| 5 Algorithms | L (BOBYQA port and NUTS are the riskiest parts) |
| 6 Examples / docs / benchmarks | S |
| 7 Library hardening and release | M |
| 8 Delete Java | S |

Phases 1 and 2 can overlap once the tensor API is stable. Within Phase 3, op families can be split across people
(arithmetic, trig/special, matrix, shape/slicing, boolean/generic), because the golden vectors make each family
independently verifiable.

## Testing strategy

1. **Golden vectors from Java** for everything deterministic. This is the main parity mechanism.
2. **Ported unit tests.** The Java suite has ~2,200 tests. Port the semantic ones and drop those that only exist
   because of Java (Lombok, mocks of final classes, serialisation plumbing).
3. **Finite-difference gradient checks** for every differentiable op and distribution, in a reusable helper like
   `TensorTestOperations`.
4. **Statistical tests** for samplers and MCMC:
   - moments and KS against known distributions
   - stationary-distribution checks like `PriorProposalHastingsCorrectionTest`
   - posterior comparisons against Java on the golden models
5. **Property tests** for shapes, broadcasting and slicing.
6. **Benchmarks** in CI, to catch performance regressions against the Java baseline.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Graph semantics depend on Java object identity and mutation (e.g. `setAndCascade`, lazy eval, proxies in `Sequence`) | Specify the cascade / lazy-eval rules from the Java code before Phase 3, and cover them with golden vectors |
| Random streams can't be compared exactly | Use only statistical tests for samplers; keep deterministic layers on exact golden vectors |
| BOBYQA has no pure-Rust implementation | Port Apache's; or ship CG/L-BFGS/Nelder–Mead first and add BOBYQA later |
| The partial-derivative tensor layout is subtle (`[wrt..., of...]` vs `[of..., wrt...]`) | Keep the Java layout and port the autodiff tests unchanged |
| Behaviour changes made during this cleanup (Exponential rate, NaN equality, slicing) | Golden vectors are generated after the fixes, so Rust targets the corrected behaviour |
| Scope creep into a redesign | Defer API redesign until parity; make the first Rust release semantically equivalent |
| Losing the reference once Java is deleted | Frozen golden vectors become Rust test fixtures, and the `java-final` tag keeps the Java implementation runnable |
| Defects in Java being copied into Rust | Known issues are listed in the vectors manifest and either fixed in Java before freezing or marked as intended Rust behaviour |
| A heavy or unstable dependency tree for downstream users | Optional features for IO and parallelism, `cargo deny`, and pure-Rust dependencies only (`faer`, `prost`, `serde`) |

## Suggested next steps

1. Finish Phase 0:
   - sign off the scope decisions in `golden-vectors/README.md`
   - fix or accept the known issues
   - add coverage for generic/boolean ops and `Sequence`
2. Create the Cargo workspace (initially in a `rust/` directory) with `keanu-parity` reading `golden-vectors/`, and
   run it in CI alongside the Gradle build.
3. Spike `keanu-tensor` on `ndarray` + `faer` against the tensor vectors (2–3 weeks). Use the spike to choose
   between `ndarray` and an in-house tensor before committing to Phase 1.
