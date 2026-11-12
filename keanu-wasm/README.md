# keanu-wasm

Applied demo models built on Keanu, and WebAssembly bindings that run them in a browser. It is the engine for an
interactive demo site; it is not published.

| Demo | Question | Model | Fit in the browser |
| --- | --- | --- | --- |
| `AbTestDemo` | Should we ship variant B? | Beta-binomial conversion rates; P(B better), lift, expected loss | about 0.1 s |
| `CreditDemo` | How many loans will default, and what could they cost? | Hierarchical logistic regression by grade and debt-to-income; exact Poisson-binomial tail, simulated losses, 99% VaR and ES, stress slider without refitting | about 6 s, then about 0.25 s per scenario |
| `VolatilityDemo` (GARCH) | How much could this position lose tomorrow? | GARCH(1,1); volatility path, forecast, 95%/99% VaR, 97.5% ES | about 0.15 s |
| `VolatilityDemo` (stochastic) | The same, with volatility that is itself random | Stochastic volatility, one latent per day | about 12 s |

Times are for a single NUTS chain in Node 26 (V8, as in Chrome) on an Apple-silicon laptop, with the defaults in
the table of the bench script. Native code is about 1.6 times faster. Credit history and books are synthetic, drawn
from a seed with known default rates; the volatility sample is the simulated series of Keanu's
`stochastic_volatility` example.

## Use from JavaScript

Run each demo in a Web Worker: `step(n)` runs up to n NUTS transitions and returns true when the fit is finished,
so the worker can post `progress()` between calls. Reports are JSON strings.

```js
import init, { CreditDemo } from "./pkg/keanu_wasm.js";
await init();
const demo = new CreditDemo(3000, 250, 500, 7);   // history loans, book loans, draws, seed
while (!demo.step(25)) postMessage({ progress: demo.progress() });
const report = JSON.parse(demo.report(0.5, 15)); // stress on the log-odds scale, default threshold
```

## Build

```bash
CARGO_PROFILE_RELEASE_LTO=true CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
  cargo build --release --target wasm32-unknown-unknown -p keanu-wasm
wasm-bindgen --target web --out-dir pkg target/wasm32-unknown-unknown/release/keanu_wasm.wasm
```

`wasm-bindgen` must be version 0.2.130, matching the crate. The module is about 4.4 MB, 1.15 MB gzipped, before
`wasm-opt`. The demos' own tests run natively with `cargo test -p keanu-wasm --release`.
