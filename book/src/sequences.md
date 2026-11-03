# Sequences (plates)

A sequence repeats a group of nodes, for example one group per time step, per agent or per data point. Nodes in one
group can depend on nodes in the previous group through **proxies**: placeholders that are connected to the previous
item's node with the same label.

```rust,ignore
{{#include ../../keanu/tests/book.rs:sequence}}
```

- `with_initial_state(label, node)` gives the first item's proxies something to connect to.
- `count(n, factory)` builds n items; `from_iter(data, factory)` builds one per data element.
- `double_proxy_for(label)` (and the `int_` and `bool_` forms) creates a proxy for the previous item's node.
- `with_transition` connects proxies to differently labelled nodes.

Each item's nodes are labelled in their own namespace (`Sequence_Item_<index>.<sequence id>.<label>`, prefixed with
the sequence's name if it has one), so they can be found after building or loading.
