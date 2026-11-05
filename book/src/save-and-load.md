# Saving and loading

A model (its graph, plus its state's values and observations) can be saved as JSON (feature `json`, on by default)
or binary protobuf (feature `protobuf`), with some metadata.

```rust,ignore
{{#include ../../keanu/tests/book.rs:save_load}}
```

`load_json` / `load_protobuf` return the graph, the saved state and the metadata. Saved nodes keep their order, so
a node has the same `NodeId` after loading as it had when it was saved, and keep their labels.

A small file can describe enormous nodes, or a scan of so many steps that evaluating it would practically never
finish, so loading checks every node against `LoadOptions`: by default at most 2^31 elements per node, 2^24 steps
per scan and 2^32 units of work per scan (steps times the elements of the body's nodes). These protect against
untrusted files; `load_json_with` / `load_protobuf_with` take larger limits for trusted models.

## The format

Files use Keanu's own protobuf schema, `keanu-io/proto/keanu/graph.proto`; the JSON form is its standard protobuf
JSON mapping. Each node is saved with its parents and a typed operation: a unary or binary operation, a reduction,
a shape operation and so on, each with its own parameters, or a distribution identified by its kind with its
parameters as the node's parents. A distribution built around another (truncated, censored, mixture) saves its base
as a nested distribution type, recursively, so a mixture of truncated distributions round-trips. Every operation and
distribution can be saved except lambda nodes and custom distributions (also as the base of another), which return
`IoError::NotSaveable`.

Every file records the format version it was written with. Adding operations or distributions keeps the version,
so older files keep loading; a file written with a newer version than this one reads returns
`IoError::UnsupportedVersion`.

With the `dot` feature, `keanu::io::to_dot(&graph, Some(&state), &metadata)` writes a Graphviz description of a graph.
