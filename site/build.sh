#!/usr/bin/env bash
# Builds the demos' WebAssembly into site/pkg. Needs the wasm32-unknown-unknown target and wasm-bindgen-cli 0.2.130.
set -euo pipefail
cd "$(dirname "$0")/.."
CARGO_PROFILE_RELEASE_LTO=true CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
  cargo build --release --target wasm32-unknown-unknown -p keanu-wasm
wasm-bindgen --target web --out-dir site/pkg target/wasm32-unknown-unknown/release/keanu_wasm.wasm
echo "Built site/pkg. Serve the site directory, for example: python3 -m http.server -d site 8000"
