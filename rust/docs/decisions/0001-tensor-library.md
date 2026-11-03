# Decision 0001: tensor library

**Decision (2026-11-03): an in-house tensor (row-major `Vec` + shape) with `faer` for linear algebra.** The spike crates were removed after the decision; they are in the git history (commit 17061d1bc).

The question is what `keanu-tensor` should be built on:

- **A:** the `ndarray` crate (`spikes/tensor-ndarray`)
- **B:** an in-house tensor, a contiguous row-major `Vec<f64>` plus a shape (`spikes/tensor-inhouse`)

Both use `faer` for linear algebra.

## What was built

Both spikes implement the same representative slice of Keanu's tensor semantics:

- construction
- broadcasting binary ops
- unary maps
- full and per-dimension sums
- cumulative sums
- NaN-aware argMax
- reshape with a `-1` wildcard
- permute
- numpy slicing, including negative steps
- batched matrix multiply with transposes
- batched Cholesky

Both are checked against the same 120 cases from `golden-vectors/tensors.json` by a shared test
(`spikes/parity_tests.rs`).

## Correctness

Both pass all 120 cases: arithmetic bit-for-bit, and transcendental functions within 1e-14.

Semantic gaps found in `ndarray`, each needing extra code:

- **Slicing:** negative-step slices walk a forward range from its end, unlike numpy, so they have to be translated.
- **Reshape:** there is no `-1` wildcard.
- **Broadcasting:** arithmetic operators panic on incompatible shapes instead of returning an error, so every
  operation needs a pre-check.
- **Linear algebra:** there is none without a native LAPACK backend, so `faer` is needed anyway. That means two
  matrix stacks (`matrixmultiply` and `faer`) and copies between layouts.

## Performance

Measured with criterion, release build, Apple Silicon. "B" is after adding a contiguous inner-loop fast path,
which was about 40 lines.

| Benchmark | A ndarray | B in-house | Faster |
|---|---|---|---|
| broadcast add 1000×1000 + 1000 | 240 µs | 178 µs | B 1.3× (before the fast path, B was 2.2 ms) |
| element-wise times 1000×1000 | 354 µs | 354 µs | = |
| exp 1000×1000 | 6.1 ms | 6.1 ms | = |
| sum over axis 0, 1000×1000 | 347 µs | 162 µs | B 2.1× (before the fast path, B was 2.1 ms) |
| cumsum over axis 1, 1000×1000 | 3.7 ms | 3.7 ms | = |
| slice `::-1, ::2` of 1000×1000 | 220 µs | 1.09 ms | A 5× (B's copy routine doesn't have the fast path yet) |
| permute 100³ | 937 µs | 2.33 ms | A 2.5× (same) |
| matmul 256×256 | 763 µs | 444 µs | B 1.7× (`faer` vs `matrixmultiply`) |
| batched matmul 1000×4×4 · 4×4 | 257 µs | 106 µs | B 2.4× |
| Cholesky 200×200 | 294 µs | 199 µs | B 1.5× (A has to copy into `faer`) |
| 1000 chained scalar ops | 187 µs | 40 µs | B 4.7× |
| 1000 chained 3-element ops | 175 µs | 89 µs | B 2.0× |

## Other factors

| | A ndarray | B in-house |
|---|---|---|
| Spike code | 213 lines, plus about 180 borrowed from B (slicing, reshape, broadcast checks) | 459 lines |
| Dependencies (normal) | 105, mostly from `faer` | 99 |
| Clean release build | 64 s | 61 s |
| Zero-copy views | Yes (slices and permutes can be views) | No (every op copies), but views could be added later behind the same API |
| Generic over f64 / i64 / bool | Yes (`ArrayD<T>`) | Easy (`Tensor<T>`) |
| Control of semantics | Partial: work around slicing, reshape and panics | Total |
| Maintenance | ndarray is mature and widely used; we track its releases | We own every kernel |
