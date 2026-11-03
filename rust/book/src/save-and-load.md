# Saving and loading

A model (its graph, plus its state's values and observations) can be saved as JSON (feature `json`, on by default)
or binary protobuf (feature `protobuf`), with some metadata.

```rust,ignore
{{#include ../../keanu/tests/book.rs:save_load}}
```

`load_json` / `load_protobuf` return the graph, the saved state, the metadata, and a map from saved ids to nodes.

The format is the one Java Keanu uses, so models round-trip between the two. Nodes that can't be saved return an
error: lambdas, custom distributions, CPTs, categorical nodes (Java can't save these either), and printing nodes.

With the `dot` feature, `keanu::io::to_dot(&graph, Some(&state), &metadata)` writes a Graphviz description of a graph.
