# Keanu → Rust migration plan

Status: proposal. Written once the Java code base had been reduced to a pure JVM implementation (ND4J, OpenBLAS and
the Python bindings removed) and the known correctness bugs fixed (see `release_notes.md`, "Unreleased").

## Goals

- A pure Rust implementation of Keanu with the same modelling capability and numerically equivalent results.
- No native (C/Fortran) dependencies in the core, the same constraint the Java version now has.
- Migrate incrementally, with the Java implementation as the reference until the Rust version reaches parity.

Non-goals for the first release: a source-compatible Java API, a Python API, GPU support, or bit-for-bit
reproduction of Java random number streams. Samplers can only be compared statistically.

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

**No JNI bridge by default.** Calling Rust from Java through JNI or the Panama FFM API would reintroduce a native
dependency into the Java build. If a staged rollout to Java users is needed later, it can be added as an optional
phase (Phase 7) with the Rust crate shipped as a separate artifact.

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

8. **Crate layout (workspace).**
   - `keanu-tensor`
   - `keanu-distributions`
   - `keanu-graph` (ops and autodiff)
   - `keanu-algorithms`
   - `keanu-io`
   - `keanu`, the facade crate
   - `keanu-parity`, the test-only crate that consumes the golden vectors

## Phases

Each phase ends with its exit criteria passing in CI.

### Phase 0: Prepare the Java side (Java)
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
- Port `Tensor`, the double/int/long/bool implementations, `TensorShape`, `Slicer`, broadcasting, reductions,
  `KeanuLapack` semantics (on `faer`), and `FixedPointPow`.
- Add property tests (`proptest`) for the broadcasting and slicing invariants.

**Exit:** 100% of the tensor golden vectors match (exact for integer ops, ≤1e-12 relative for floating point).

### Phase 2: `keanu-distributions`
- Port the special functions, every distribution's `log_prob`, `d_log_prob` and `sample`, and the KDE.

**Exit:**
- Golden `logProb` / `dLogProb` vectors match.
- Sampler moment tests pass: mean, variance and Kolmogorov–Smirnov per distribution, including the k < 1 Gamma cases.
- Finite-difference checks pass for every `d_log_prob`.

### Phase 3: `keanu-graph`
- Port the arena graph, all ops (vertex types), value propagation (cascade / lazy eval) and observation.
- Port forward and reverse autodiff and `LogProbGraph`.
- Port the generic and boolean ops, and `If` / multiplexer / where.

**Exit:**
- The golden vertex values and gradients match.
- A finite-difference gradient test passes for every differentiable op. Port `AutoDiffRegressionTest` and the per-vertex tests.

### Phase 4: Model layer and IO (`keanu-io`)
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

### Phase 7 (optional): Java interop
- Only if existing Java users need a transition path. Options:
  - a thin JNI or Panama FFM wrapper
  - a separate process talking protobuf over stdio or gRPC
- Shipped as a separate artifact so the pure-Java Keanu stays dependency-free.

### Phase 8: Switch over
- Mark the Java implementation as maintenance-only, then archive it once the Rust version has had a release with
  parity on every golden test.

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

## Suggested first steps

1. Merge the Java cleanup, then tag it as the reference version for the migration.
2. Implement the golden vector exporter (Phase 0) for tensors and distributions only.
3. Spike `keanu-tensor` on `ndarray` + `faer` against those vectors (2–3 weeks). Use the spike to choose between
   `ndarray` and an in-house tensor before committing to Phase 1.
