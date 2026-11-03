# Benchmarks: Java vs Rust

Every Java JMH benchmark in `benchmarks/` is mirrored by a criterion benchmark in `rust/keanu/benches` with the same
shapes, operations, models and settings. `ModelBenchmark` (log prob, its gradient, and Metropolis–Hastings on the
golden models) was added on both sides for this comparison. Both sides load the same `golden-vectors/models/*.pb`
files.

## Setup

| | |
|---|---|
| Machine | Apple M1 (8 cores), 16 GB, macOS 26.3 |
| Java | OpenJDK 25.0.2. JMH with `-wi 2 -w 1s -i 3 -r 1s -f 1 -bm avgt` (reduced from the project's 3 × 1 s warm-up, 5 × 1 s measurement and 3 forks, to keep the run under an hour) |
| Rust | rustc 1.99.1, `cargo bench --all-features` (release). Criterion with 1 s warm-up, 2–3 s measurement and 10–20 samples. Mean shown |
| Date | 2026-11-03 (commit after `8d7f19413`) |

Run them with:

```sh
# Java, from the repository root
./gradlew :benchmarks:shadowJar
java -jar benchmarks/build/libs/benchmarks.jar -wi 2 -w 1s -i 3 -r 1s -f 1 -bm avgt -tu us
# Rust
cd rust && cargo bench -p keanu --all-features
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

Geometric mean of Java time ÷ Rust time per group (above 1 means Rust is faster):

| Group | Java ÷ Rust |
|---|---:|
| Broadcasting element-wise ops | 1.9× |
| Element-wise ops and matrix multiply | 2.6× |
| Unary and matrix ops (100 per call) | 1.4× |
| Graph evaluation (10,000 per call) | 0.9× |
| Optimizers (warm start, as Java) | 0.8× |
| NUTS, 1000 samples | 0.5× |
| Golden models (log prob, gradient, MH) | 0.7× |

- **Tensors are faster in Rust**, by up to 37× for large matrix multiplies (`faer`) and 16–19× for large
  broadcasts. Tiny tensors (a handful of elements) are about level, sometimes slightly slower.
- **Small matrix decompositions are slower in Rust.** Cholesky, determinant and inverse of 2–10 dimensional
  matrices are 1.2–5× slower: `faer` has a fixed per-call overhead. They're on par or faster at 50 × 50.
- **Graph evaluation, log probs and Metropolis–Hastings are about level.**
- **Log prob gradients are 2–8× slower in Rust**, and this is the reason NUTS is 2× slower (NUTS spends most of
  its time on gradients). Reverse mode allocates a partial tensor per node per call and keeps partials in hash
  maps. The worst case, linear regression with a matrix multiply, is 8× slower. This is the main performance item
  for Phase 7 (library hardening).
- **Optimizers:**
  - Warm-start Adam is 2.4× faster in Rust; warm-start conjugate gradient is 3.7× slower (it's a different
    implementation, with its own line search). Warm-start BOBYQA (translated from PRIMA rather than Apache Commons
    Math) is about level, 0.9×; it was measured later than the other rows, on the same machine.
  - From a cold start, Adam takes about 71 ms because its default step size (0.001) needs thousands of steps to
    move A from 0 to its optimum near 6, and conjugate gradient takes about 1.1 ms. The Java benchmark doesn't
    measure a cold start.

## Results

### Broadcasting element-wise ops (`BroadcastBinaryDoubleOperations`)

| Case | Java | Rust | Java ÷ Rust |
|---|---:|---:|---:|
| DIVIDE ,2x100 | 165 ns | 98 ns | 1.7× |
| DIVIDE 1x1,2x10 | 60 ns | 64 ns | 0.9× |
| DIVIDE 1x10,2x10 | 100 ns | 67 ns | 1.5× |
| DIVIDE 20x1x20,20x20x20 | 72.75 µs | 4.00 µs | 18.2× |
| DIVIDE 20x200,20x20x200 | 295.61 µs | 15.78 µs | 18.7× |
| DIVIDE 2x10,1x1 | 49 ns | 65 ns | 0.8× |
| DIVIDE 2x10,1x10 | 77 ns | 67 ns | 1.1× |
| DIVIDE 2x100, | 137 ns | 93 ns | 1.5× |
| DIVIDE 2x2,2x2x2 | 78 ns | 86 ns | 0.9× |
| DIVIDE 2x2x2,2x2 | 50 ns | 88 ns | 0.6× |
| TIMES ,2x100 | 144 ns | 90 ns | 1.6× |
| TIMES 1x1,2x10 | 65 ns | 66 ns | 1.0× |
| TIMES 1x10,2x10 | 95 ns | 68 ns | 1.4× |
| TIMES 20x1x20,20x20x20 | 72.94 µs | 4.23 µs | 17.2× |
| TIMES 20x200,20x20x200 | 250.44 µs | 15.43 µs | 16.2× |
| TIMES 2x10,1x1 | 63 ns | 66 ns | 1.0× |
| TIMES 2x10,1x10 | 91 ns | 68 ns | 1.3× |
| TIMES 2x100, | 144 ns | 90 ns | 1.6× |
| TIMES 2x2,2x2x2 | 77 ns | 88 ns | 0.9× |
| TIMES 2x2x2,2x2 | 50 ns | 89 ns | 0.6× |

### Element-wise ops and matrix multiply on n × n (`ElementwiseBinaryDoubleOperations`)

| Case | Java | Rust | Java ÷ Rust |
|---|---:|---:|---:|
| MATRIX_MULTIPLY 1 | 29 ns | 102 ns | 0.3× |
| MATRIX_MULTIPLY 10 | 822 ns | 369 ns | 2.2× |
| MATRIX_MULTIPLY 100 | 474.73 µs | 88.61 µs | 5.4× |
| MATRIX_MULTIPLY 1000 | 477.04 ms | 13.00 ms | 36.7× |
| TIMES 1 | 20 ns | 24 ns | 0.8× |
| TIMES 10 | 99 ns | 53 ns | 1.9× |
| TIMES 100 | 7.65 µs | 2.70 µs | 2.8× |
| TIMES 1000 | 1.21 ms | 356.91 µs | 3.4× |

### 100 unary / matrix ops on n × n (`UnaryDoubleOperations`)

| Case | Java | Rust | Java ÷ Rust |
|---|---:|---:|---:|
| CHOLESKY 2 | 2.38 µs | 12.99 µs | 0.2× |
| CHOLESKY 5 | 6.80 µs | 23.05 µs | 0.3× |
| CHOLESKY 10 | 20.78 µs | 47.67 µs | 0.4× |
| CHOLESKY 50 | 936.70 µs | 805.63 µs | 1.2× |
| COS 2 | 6.69 µs | 3.60 µs | 1.9× |
| COS 5 | 21.39 µs | 8.50 µs | 2.5× |
| COS 10 | 96.75 µs | 35.42 µs | 2.7× |
| COS 50 | 3.43 ms | 988.70 µs | 3.5× |
| DETERMINANT 2 | 3.06 µs | 8.09 µs | 0.4× |
| DETERMINANT 5 | 8.57 µs | 13.10 µs | 0.7× |
| DETERMINANT 10 | 33.57 µs | 39.92 µs | 0.8× |
| DETERMINANT 50 | 2.07 ms | 1.16 ms | 1.8× |
| INVERSE 2 | 7.09 µs | 10.88 µs | 0.7× |
| INVERSE 5 | 34.82 µs | 35.46 µs | 1.0× |
| INVERSE 10 | 178.24 µs | 194.54 µs | 0.9× |
| INVERSE 50 | 12.08 ms | 11.33 ms | 1.1× |
| LOG 2 | 6.26 µs | 3.74 µs | 1.7× |
| LOG 5 | 18.99 µs | 9.39 µs | 2.0× |
| LOG 10 | 116.53 µs | 31.75 µs | 3.7× |
| LOG 50 | 1.78 ms | 686.04 µs | 2.6× |
| TRANSPOSE 2 | 8.34 µs | 9.23 µs | 0.9× |
| TRANSPOSE 5 | 32.96 µs | 10.86 µs | 3.0× |
| TRANSPOSE 10 | 118.37 µs | 18.83 µs | 6.3× |
| TRANSPOSE 50 | 2.84 ms | 178.45 µs | 15.9× |

### 10,000 graph evaluations (`CompiledBackEnds`, interpreted graph)

| Case | Java | Rust | Java ÷ Rust |
|---|---:|---:|---:|
| KEANU_COMPILED 0 | 4.65 ms | — | — |
| KEANU_COMPILED 100 | 65.40 ms | — | — |
| KEANU_GRAPH 0 | 1.58 ms | 1.59 ms | 1.0× |
| KEANU_GRAPH 100 | 193.37 ms | 238.59 ms | 0.8× |

### MAP on A ~ N(10, 0.1), B ~ N(5, 1), C ~ N(A·B, 0.1) (`OptimizerBenchmark`)

| Case | Java | Rust | Java ÷ Rust |
|---|---:|---:|---:|
| ADAM_OPTIMIZER | 99.27 µs | 41.76 µs | 2.4× |
| ADAM_OPTIMIZER (cold start) | — | 71.20 ms | — |
| BOBYQA_OPTIMIZER | 96.04 µs | 110.36 µs | 0.9× |
| BOBYQA_OPTIMIZER (cold start) | — | 238.81 µs | — |
| CONJUGATE_GRADIENT | 58.43 µs | 214.57 µs | 0.3× |
| CONJUGATE_GRADIENT (cold start) | — | 1.13 ms | — |

### 1000 NUTS samples, adapting for 1000 (`NUTSBenchmark`)

| Case | Java | Rust | Java ÷ Rust |
|---|---:|---:|---:|
| takeSamples | 249.12 ms | 496.13 ms | 0.5× |

### Golden models: log prob, its gradient, and 1000 Metropolis–Hastings steps (`ModelBenchmark`)

| Case | Java | Rust | Java ÷ Rust |
|---|---:|---:|---:|
| logProb gammaPoisson | 819 ns | 427 ns | 1.9× |
| logProb hierarchicalGaussian | 1.11 µs | 1.12 µs | 1.0× |
| logProb linearRegression | 807 ns | 893 ns | 0.9× |
| logProb thermometers | 266 ns | 587 ns | 0.5× |
| logProbGradient gammaPoisson | 875 ns | 1.47 µs | 0.6× |
| logProbGradient hierarchicalGaussian | 2.04 µs | 4.38 µs | 0.5× |
| logProbGradient linearRegression | 2.69 µs | 22.28 µs | 0.1× |
| logProbGradient thermometers | 833 ns | 1.87 µs | 0.4× |
| metropolisHastings1000 gammaPoisson | 3.33 ms | 2.07 ms | 1.6× |
| metropolisHastings1000 hierarchicalGaussian | 2.17 ms | 2.52 ms | 0.9× |
| metropolisHastings1000 linearRegression | 2.36 ms | 2.47 ms | 1.0× |
| metropolisHastings1000 thermometers | 1.35 ms | 2.18 ms | 0.6× |