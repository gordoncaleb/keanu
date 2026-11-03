# Contributing

Run every check below from the repository root before committing. There is no CI yet.

## Toolchains and tools

- **Development:** the latest stable Rust (`rustup update stable`).
- **Minimum supported Rust version (MSRV): 1.85**, declared as `rust-version` in `Cargo.toml`. Install it with
  `rustup toolchain install 1.85.0`.
- **Tools:**
  ```sh
  cargo install --locked cargo-hack cargo-deny cargo-msrv mdbook
  # after the first release, also: cargo-semver-checks
  ```

## Checks

```sh
# formatting
cargo fmt --all --check

# lints, warnings as errors
cargo clippy --workspace --all-targets --all-features -- -D warnings

# tests (unit, integration, golden vectors, examples, doctests) with all features, then with defaults
cargo test --workspace --all-features
cargo test --workspace

# every feature combination of every crate: builds, and lints including tests and benches
cargo hack --workspace --feature-powerset --no-dev-deps check
cargo hack --workspace --feature-powerset clippy --all-targets -- -D warnings

# the MSRV: every feature combination builds on the declared rust-version
# (the test suite itself needs the latest stable, because of dev-dependencies)
cargo hack --workspace --exclude keanu-parity --feature-powerset --no-dev-deps --rust-version check
# to re-derive the real minimum after dependency upgrades
(cd keanu && cargo msrv find --min 1.85 -- cargo check --all-features)

# licences, security advisories, banned/duplicate crates, sources (configuration in deny.toml)
cargo deny check

# API docs, warnings (e.g. broken intra-doc links) as errors, and as docs.rs builds them
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
RUSTDOCFLAGS="-D warnings --cfg docsrs" cargo +nightly doc --workspace --no-deps --all-features

# the guide (its code samples are compiled and run by `cargo test -p keanu --test book`)
mdbook build book

# benchmarks (optional; compare against docs/benchmarks.md)
cargo bench -p keanu
```

## Golden vectors (Java parity)

The tests compare against vectors recorded from the Java implementation, frozen in
`keanu-parity/fixtures/golden-vectors` (see its `README.md`). They are never regenerated. If a deliberate behaviour
change alters a recorded result, fix it with a regression test, update only the affected case in the fixture by
hand, and record the change and the affected case names in `CHANGELOG.md`. The Java implementation that produced
them is at the `java-final` tag.

## Packaging

The crates are `publish = false` until the first release is approved, as a guard against accidental publishing.
**Do not run `cargo publish`.**

To check what each crate would package, run `cargo package -p keanu-tensor --allow-dirty --list` (and likewise for
the others). Packaging and verifying the crates that depend on other workspace crates needs them to be
publishable. To check the whole set, flip `publish` to `true` in a scratch copy of the workspace (never in the
real one) and run there:

```sh
cargo package --workspace --exclude keanu-parity --allow-dirty
```

## After the first release

- **Before each release,** run `cargo semver-checks --workspace --baseline-version <previous>`, and bump versions
  according to its report.
- **Bump versions together.** The internal crates are versioned in lockstep with `keanu`: update the version in
  each crate and in `[workspace.dependencies]`.
