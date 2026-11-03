# Installation

Add Keanu to your `Cargo.toml`:

```toml
[dependencies]
keanu = "0.1"
```

Keanu is pure Rust (no native libraries), so there's nothing else to install.

## Features

| Feature | Default | Enables |
|---|---|---|
| `json` | yes | Saving and loading models as JSON |
| `protobuf` | | Saving and loading models as binary protobuf |
| `dot` | | Exporting a graph as Graphviz DOT |
| `parallel` | | Running independent MCMC chains in parallel with rayon |

For example, to load protobuf models and run chains in parallel:

```toml
[dependencies]
keanu = { version = "0.1", features = ["protobuf", "parallel"] }
```

Most programs only need the prelude:

```rust,ignore
use keanu::prelude::*;
```
