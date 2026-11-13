# Benchmarks: Java vs Rust

Every Java JMH benchmark (in `benchmarks/` at the `java-final` tag) is mirrored by a criterion benchmark in
`keanu/benches` with the same shapes, operations, models and settings. `ModelBenchmark` (log prob, its gradient, and
Metropolis–Hastings on the golden models) was added on both sides for this comparison. Both sides load the same
golden model files (`keanu-parity/fixtures/golden-vectors/models/*.pb`). The Java numbers were measured before the
Java implementation was removed.

## Setup

| | |
|---|---|
| Machine | Apple M1 (8 cores), 16 GB, macOS 26.3 |
| Java | OpenJDK 25.0.2. JMH with `-wi 2 -w 1s -i 3 -r 1s -f 1 -bm avgt` (reduced from the project's 3 × 1 s warm-up, 5 × 1 s measurement and 3 forks, to keep the run under an hour) |
| Rust | rustc 1.99.1, `cargo bench --all-features` (release). Criterion with 1 s warm-up, 2–3 s measurement and 10–20 samples. Mean shown |
| Date | 2026-11-03. Rust re-measured after the Phase 7 performance work; Java numbers unchanged (same settings, same machine) |

Run them with:

```sh
# Java (check out the java-final tag first), from the repository root
./gradlew :benchmarks:shadowJar
java -jar benchmarks/build/libs/benchmarks.jar -wi 2 -w 1s -i 3 -r 1s -f 1 -bm avgt -tu us
# Rust
cargo bench -p keanu --all-features
```

The Rust benchmarks copy two behaviours of the Java ones so the comparison is fair:

- **`CompiledBackEnds`** runs 10,000 evaluations per call.
- **`OptimizerBenchmark`, `NUTSBenchmark` and `ModelBenchmark`'s Metropolis–Hastings** keep the model's state
  between calls. A Java vertex keeps its value, so after the first call each optimizer run starts at the optimum
  the previous run found, and each chain continues where the last one ended. The Rust optimizer benchmark also
  reports the cold start from A = B = 0, which is what a user's first run costs.

Some benchmarks aren't compared:

- **10240 × 10240** (`ElementwiseBinaryDoubleOperations`) was left out of both runs: one matrix multiply takes
  minutes.
- **The compiled backend** only exists in Java (it isn't ported).
- **`UnaryDoubleOperations` CHOLESKY**: Rust adds the identity to the matrix so that it is always positive
  definite. The cost is the same.

## Summary

Geometric mean of Java time ÷ Rust time per group (above 1 means Rust is faster). The "before" column is the first
Rust measurement, before the Phase 7 performance work.

| Group | Before | Now |
|---|---:|---:|
| Broadcasting element-wise ops | 1.9× | 1.8× |
| Element-wise ops and matrix multiply | 2.6× | 3.2× |
| Unary and matrix ops (100 per call) | 1.4× | 1.8× |
| Graph evaluation (10,000 per call) | 0.9× | 1.3× |
| Optimizers (warm start, as Java) | 0.8× | 2.3× |
| NUTS, 1000 samples | 0.5× | 1.2× |
| Golden models (log prob, gradient, MH) | 0.7× | 1.5× |

Every model-level benchmark is now at least as fast as Java (log probs, gradients, NUTS, graph evaluation and all
three optimizers), except Metropolis–Hastings on the thermometers model, which is level within measurement noise
(1.38 ms against 1.35 ms).

What changed (results are unchanged: every golden test still passes, and NUTS takes exactly the same trajectory):

- **No thread pool for small matrices.** `faer`'s `*` operator multiplied with its global (rayon) parallelism, so
  even a 4 × 2 product handed work to other threads. Small products now use a direct loop, medium ones `faer`
  sequentially, and only large ones run in parallel. Cholesky up to 64 × 64 uses a direct loop. This alone took
  the linear regression gradient from 22 µs to 4.6 µs.
- **One reverse sweep per gradient.** The log prob gradient ran a separate reverse-mode pass from every parameter
  of every probabilistic node. It now seeds all of them into a single sweep (reverse mode is linear, so the sum
  is the same) and skips constant parameters, which can't lead to a latent.
- **Small tensors don't allocate.** Tensors of up to four elements (every scalar) store their data inline, and
  the kernels build small results directly in that storage. Large results are still built in a `Vec` and handed
  over without copying.
- **A scalar fast path in reverse mode.** Element-wise ops on scalars compute their derivatives on `f64`s, using
  exactly the tensor path's formulas.
- **Fewer temporaries.** Distributions read their parameters in place rather than copying them to a common
  shape, and Gaussian computes both of its gradients in one pass. `Model` caches its node lists. Gradients use a
  dense table instead of a hash map, and the small buffers used by autodiff and cascades are on the stack.
- **NUTS shares trajectory points.** Tree ends and candidates share one immutable leapfrog state through `Rc`
  instead of cloning four vectors per step. Allocations per leapfrog step fell from 50 to 14.
- **Conjugate gradient line search.** Bracketing with parabolic extrapolation followed by Brent's method replaces
  pure golden-section steps refined to 1e-10. A warm start needs 48 evaluations instead of 215. The optimum is
  the same, within the golden tolerances.
- **BOBYQA temporaries** in the PRIMA translation are on the stack for small problems (recorded in the crate's
  third-party notices).

Still slower than Java:

- **Tiny tensors with broadcasting** (2 × 2 × 2 with 2 × 2, a scalar with 2 × 10) take 75–95 ns against Java's
  50–78 ns. This is the fixed cost of a call: shape checks and the result struct, which is larger now that small
  data is inline.
- **Tiny matrices:** a 1 × 1 matrix multiply (39 ns against 29 ns), and 2 × 2 determinants and 2 × 2 or
  5 × 5 Cholesky (0.7–0.9×). From 10 × 10 Rust is level or faster, apart from 10 × 10 inverses (0.9×).

## Results

### Broadcasting element-wise ops (`BroadcastBinaryDoubleOperations`)

| Case | Java | Rust | Java ÷ Rust |
|---|---:|---:|---:|
| DIVIDE ,2x100 | 165 ns | 97 ns | 1.7× |
| DIVIDE 1x1,2x10 | 60 ns | 73 ns | 0.8× |
| DIVIDE 1x10,2x10 | 100 ns | 75 ns | 1.3× |
| DIVIDE 20x1x20,20x20x20 | 72.75 µs | 3.97 µs | 18.3× |
| DIVIDE 20x200,20x20x200 | 295.61 µs | 15.75 µs | 18.8× |
| DIVIDE 2x10,1x1 | 49 ns | 74 ns | 0.7× |
| DIVIDE 2x10,1x10 | 77 ns | 75 ns | 1.0× |
| DIVIDE 2x100, | 137 ns | 97 ns | 1.4× |
| DIVIDE 2x2,2x2x2 | 78 ns | 94 ns | 0.8× |
| DIVIDE 2x2x2,2x2 | 50 ns | 92 ns | 0.5× |
| TIMES ,2x100 | 144 ns | 89 ns | 1.6× |
| TIMES 1x1,2x10 | 65 ns | 74 ns | 0.9× |
| TIMES 1x10,2x10 | 95 ns | 76 ns | 1.2× |
| TIMES 20x1x20,20x20x20 | 72.94 µs | 4.08 µs | 17.9× |
| TIMES 20x200,20x20x200 | 250.44 µs | 15.13 µs | 16.5× |
| TIMES 2x10,1x1 | 63 ns | 75 ns | 0.8× |
| TIMES 2x10,1x10 | 91 ns | 77 ns | 1.2× |
| TIMES 2x100, | 144 ns | 88 ns | 1.6× |
| TIMES 2x2,2x2x2 | 77 ns | 92 ns | 0.8× |
| TIMES 2x2x2,2x2 | 50 ns | 91 ns | 0.5× |

### Element-wise ops and matrix multiply on n × n (`ElementwiseBinaryDoubleOperations`)

| Case | Java | Rust | Java ÷ Rust |
|---|---:|---:|---:|
| MATRIX_MULTIPLY 1 | 29 ns | 39 ns | 0.7× |
| MATRIX_MULTIPLY 10 | 822 ns | 445 ns | 1.8× |
| MATRIX_MULTIPLY 100 | 474.73 µs | 50.58 µs | 9.4× |
| MATRIX_MULTIPLY 1000 | 477.04 ms | 12.15 ms | 39.2× |
| TIMES 1 | 20 ns | 13 ns | 1.5× |
| TIMES 10 | 99 ns | 60 ns | 1.7× |
| TIMES 100 | 7.65 µs | 2.86 µs | 2.7× |
| TIMES 1000 | 1.21 ms | 348.84 µs | 3.5× |

### 100 unary / matrix ops on n × n (`UnaryDoubleOperations`)

| Case | Java | Rust | Java ÷ Rust |
|---|---:|---:|---:|
| CHOLESKY 2 | 2.38 µs | 2.67 µs | 0.9× |
| CHOLESKY 5 | 6.80 µs | 7.94 µs | 0.9× |
| CHOLESKY 10 | 20.78 µs | 16.32 µs | 1.3× |
| CHOLESKY 50 | 936.70 µs | 670.50 µs | 1.4× |
| COS 2 | 6.69 µs | 3.36 µs | 2.0× |
| COS 5 | 21.39 µs | 9.61 µs | 2.2× |
| COS 10 | 96.75 µs | 37.04 µs | 2.6× |
| COS 50 | 3.43 ms | 972.69 µs | 3.5× |
| DETERMINANT 2 | 3.06 µs | 4.09 µs | 0.7× |
| DETERMINANT 5 | 8.57 µs | 8.25 µs | 1.0× |
| DETERMINANT 10 | 33.57 µs | 30.69 µs | 1.1× |
| DETERMINANT 50 | 2.07 ms | 902.00 µs | 2.3× |
| INVERSE 2 | 7.09 µs | 6.64 µs | 1.1× |
| INVERSE 5 | 34.82 µs | 31.39 µs | 1.1× |
| INVERSE 10 | 178.24 µs | 187.94 µs | 0.9× |
| INVERSE 50 | 12.08 ms | 11.22 ms | 1.1× |
| LOG 2 | 6.26 µs | 3.60 µs | 1.7× |
| LOG 5 | 18.99 µs | 10.19 µs | 1.9× |
| LOG 10 | 116.53 µs | 33.42 µs | 3.5× |
| LOG 50 | 1.78 ms | 687.14 µs | 2.6× |
| TRANSPOSE 2 | 8.34 µs | 8.06 µs | 1.0× |
| TRANSPOSE 5 | 32.96 µs | 11.17 µs | 3.0× |
| TRANSPOSE 10 | 118.37 µs | 19.43 µs | 6.1× |
| TRANSPOSE 50 | 2.84 ms | 218.87 µs | 13.0× |

### 10,000 graph evaluations (`CompiledBackEnds`, interpreted graph)

| Case | Java | Rust | Java ÷ Rust |
|---|---:|---:|---:|
| KEANU_COMPILED 0 | 4.65 ms | — | — |
| KEANU_COMPILED 100 | 65.40 ms | — | — |
| KEANU_GRAPH 0 | 1.58 ms | 1.05 ms | 1.5× |
| KEANU_GRAPH 100 | 193.37 ms | 186.88 ms | 1.0× |

### MAP on A ~ N(10, 0.1), B ~ N(5, 1), C ~ N(A·B, 0.1) (`OptimizerBenchmark`)

| Case | Java | Rust | Java ÷ Rust |
|---|---:|---:|---:|
| ADAM_OPTIMIZER | 99.27 µs | 18.80 µs | 5.3× |
| ADAM_OPTIMIZER (cold start) | — | 31.02 ms | — |
| BOBYQA_OPTIMIZER | 96.04 µs | 86.13 µs | 1.1× |
| BOBYQA_OPTIMIZER (cold start) | — | 188.12 µs | — |
| CONJUGATE_GRADIENT | 58.43 µs | 30.12 µs | 1.9× |
| CONJUGATE_GRADIENT (cold start) | — | 190.46 µs | — |

#### L-BFGS evaluation counts

L-BFGS (the default gradient optimizer since 2026-11-05) was added to `OptimizerBenchmark` as `LBFGS`. It was
measured on a loaded machine (Adam's warm start took 104 µs against the 18.8 µs above), so its times aren't
comparable with the table and aren't shown. What does compare is the number of fitness and gradient evaluations,
which `cargo bench -p keanu --all-features --bench models` prints. L-BFGS evaluates the fitness and its gradient
at every trial point, so each trial point counts twice.

| Model (MAP from the stored values) | Conjugate gradient | L-BFGS |
|---|---:|---:|
| `OptimizerBenchmark`, from A = B = 0 | 272 | 36 |
| `OptimizerBenchmark`, from the previous optimum (as Java) | 4 | 12 |
| thermometers | 20 | 8 |
| linearRegression | 320 | 36 |
| gammaPoisson | 20 | 10 |

L-BFGS needs fewer evaluations everywhere except the warm start, where it is already at the optimum and its
convergence tests (Stan's, such as a fitness change below 1e-12) are stricter than conjugate gradient's (1e-8).
The golden optimizer test (`keanu-algorithms/tests/golden_optimizers.rs`) checks that L-BFGS uses no more
evaluations than conjugate gradient on every conjugate gradient case, and prints the counts. L-BFGS needs between a half and a ninth of conjugate gradient's evaluations there; for example,
randomWalkSequence MAP takes 20 against 62 and betaBinomial MAP 6 against 34. hierarchicalGaussian's MAP is left
out: its posterior is unbounded, so neither optimizer has an optimum to reach.

### 1000 NUTS samples, adapting for 1000 (`NUTSBenchmark`)

| Case | Java | Rust | Java ÷ Rust |
|---|---:|---:|---:|
| takeSamples | 249.12 ms | 204.83 ms | 1.2× |

On 2026-11-06 (same machine, `cargo bench -p keanu --all-features --bench models -- NUTSBenchmark`), adding the
dense metric left the default diagonal metric unchanged: 264–267 ms against 259–279 ms just before it, in
alternating runs (the absolute times are higher than the table's because the machine was busier).

### Golden models: log prob, its gradient, and 1000 Metropolis–Hastings steps (`ModelBenchmark`)

| Case | Java | Rust | Java ÷ Rust |
|---|---:|---:|---:|
| logProb gammaPoisson | 819 ns | 270 ns | 3.0× |
| logProb hierarchicalGaussian | 1.11 µs | 544 ns | 2.0× |
| logProb linearRegression | 807 ns | 335 ns | 2.4× |
| logProb thermometers | 266 ns | 255 ns | 1.0× |
| logProbGradient gammaPoisson | 875 ns | 730 ns | 1.2× |
| logProbGradient hierarchicalGaussian | 2.04 µs | 1.69 µs | 1.2× |
| logProbGradient linearRegression | 2.69 µs | 2.29 µs | 1.2× |
| logProbGradient thermometers | 833 ns | 802 ns | 1.0× |
| metropolisHastings1000 gammaPoisson | 3.33 ms | 1.62 ms | 2.1× |
| metropolisHastings1000 hierarchicalGaussian | 2.17 ms | 1.57 ms | 1.4× |
| metropolisHastings1000 linearRegression | 2.36 ms | 1.57 ms | 1.5× |
| metropolisHastings1000 thermometers | 1.35 ms | 1.38 ms | 1.0× |
## Phase 4 inference benchmark

Run `cargo run --release --example inference_benchmark`. These measurements use an arm64 macOS workspace
with rustc 1.99.1. Sampling reports the median of five seeded runs with 500 warmup sweeps and 3,000
retained draws per chain. ESS per second uses the first coordinate and includes warmup time. For
differential evolution it uses one walker's ESS and includes the cost of updating all eight walkers.

| Target | Sampler | ESS/s | Evaluations |
| --- | --- | ---: | ---: |
| Correlated Gaussian | NUTS | 11,651 | 20,404 |
| Correlated Gaussian | Static HMC | 58,690 | 17,501 |
| Correlated Gaussian | Slice | 30,369 | 35,298 |
| Correlated Gaussian | Differential evolution | 16,112 | 28,008 |
| Discontinuous density | Slice | 141,177 | 27,716 |
| Discontinuous density | Differential evolution | 18,933 | 28,008 |

NUTS counts the leapfrog evaluations in its trees and excludes the separate step-size search. Static HMC
counts the initial gradient and its leapfrog evaluations. The gradient-free methods count joint density
evaluations. The discontinuous target is uniform on each half of its support, with the positive half
three times as dense as the negative half.

Optimizer timings are the medians of 21 runs from identical starts in constrained coordinates. All three
solvers reached the same log density to six decimal places on each target.

| Target | Optimizer | Time (µs) | Evaluations | Hessian-vector products |
| --- | --- | ---: | ---: | ---: |
| Correlated Gaussian | L-BFGS | 12.0 | 10 | 0 |
| Correlated Gaussian | Trust-region Newton | 28.8 | 6 | 3 |
| Correlated Gaussian | BOBYQA | 113.1 | 36 | 0 |
| Rosenbrock valley | L-BFGS | 138.6 | 84 | 0 |
| Rosenbrock valley | Trust-region Newton | 637.6 | 60 | 44 |
| Rosenbrock valley | BOBYQA | 519.2 | 152 | 0 |

L-BFGS reached the shared solution fastest on these two targets. They do not establish a case for adding a
full BFGS implementation. Evaluating variational update rules would require a separate benchmark on an
inference workload that needs them.

## Keanu, Stan and PyMC

Ten of the examples written for Keanu, Stan (CmdStan 2.40) and PyMC 6.4, sampled with each library's NUTS on the
same data: 4 chains in parallel, 1000 warm-up iterations and 1000 draws each. The method, the models and how to run
it are in [`benchmarks/ppl`](../benchmarks/ppl/README.md). Apple M1 (8 cores), 2026-11-13; the machine was shared
with other work, so each library's time is the fastest of three interleaved runs. ESS is the minimum bulk or tail
ESS over every parameter, by the same ArviZ code for all three. PyMC's Lotka–Volterra (`pm.ode`) was not run.

| Model | Keanu before (s) | Keanu (s) | Stan (s) | PyMC (s) | Keanu ESS/s | Stan ESS/s | PyMC ESS/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| eight schools (non-centred) | 0.39 | 0.07 | 0.11 | 0.92 | 18,705 | 16,700 | 2,297 |
| radon (centred) | 1.65 | 0.83 | 0.82 | 0.93 | 1,878 | 1,743 | 914 |
| robust regression | 1.13 | 0.23 | 0.11 | 0.59 | 11,384 | 21,406 | 3,888 |
| stochastic volatility | 58.59 | 3.92 | 2.94 | 9.08 | 282 | 380 | 70 |
| GP regression | 4.25 | 1.48 | 1.04 | 1.24 | 1,174 | 1,640 | 1,317 |
| Gaussian mixture | 2.05 | 1.50 | 1.23 | 0.99 | 1,598 | 1,659 | 2,501 |
| item response | 7.52 | 2.96 | 2.39 | 3.19 | 464 | 527 | 238 |
| hidden Markov model | 2.32 | 1.70 | 3.84 | 6.86 | 1,437 | 664 | 432 |
| disease mapping (BYM2) | 7.88 | 2.10 | 2.17 | 7.84 | 173 | 158 | 39 |
| Lotka–Volterra (ODE) | 83.28 | 7.65 | 12.59 | — | 126 | 63 | — |

"Keanu before" is the first measurement, before the performance work below. Every model's posterior means agree with
Stan's to within 0.1 posterior standard deviations (0.13 for PyMC's item response). Keanu and Stan take the same
number of leapfrog steps to within a few percent (both run Stan's NUTS and adaptation), so the differences are the
cost of a gradient.

What changed (the draws are bit-for-bit the same as before every change):

- **Compiled log densities for NUTS** (`CompiledModel`): the nodes that depend on the latents get registers reused
  from one evaluation to the next, element-wise ops, constant-index gathers and element moves run as vectorised
  kernels over them, Gaussian terms are computed from the registers, and everything else runs through its general
  rules. Leapfrog steps no longer set the state, and the gradient is pulled back through the transforms without
  allocating.
- **Compiled loop and ODE bodies**: a scan of scalars runs on a register tape (stochastic volatility gradient 277 →
  24 µs) and small ODE right-hand sides on an element tape with batched forward tangents (Lotka–Volterra 753 →
  81 µs).
- **Gradient-only jets** skip second derivatives (Student-t: no trigamma), and lnΓ and ψ remember their last
  arguments, which a broadcast shape parameter repeats (Student-t regression gradient 58 → 15 µs).
- **Merged dimensions** in broadcasting, strided reads and reductions, a fused multivariate Gaussian (one Cholesky
  factorisation per gradient), a direct scatter for `gather`'s reverse rule, and a table of ln k! for counts.
- **NUTS on flat vectors**, with U-turn checks that never form summed momenta and in-place momentum updates.

Still behind Stan: the robust regression (2.1× per gradient: the Student-t term still runs through the general
code), stochastic volatility (1.4×: the scan's values cross between the compiled program and the general scan),
the GP (1.35×: copies at the boundary of the dense linear algebra) and item response (1.25×). With chains in
parallel Keanu also loses more than Stan to the allocator, as its remaining small allocations contend between
threads (Stan runs chains as separate processes).
