# Getting started

## Your model as a graph

A model is a graph of **nodes**. A node is a random variable (a *probabilistic* node, with a distribution) or a
deterministic function of other nodes. An edge means a node depends on another.

Take two random variables A and B, and C = A + B:

```text
(A) ─┐
     + ─▶ (C)
(B) ─┘
```

You describe the graph with a `GraphBuilder`. Building it gives you the immutable `Graph` and a `State` holding the
nodes' values:

```rust,ignore
{{#include ../../keanu/tests/book.rs:describe}}
```

The handles (`a`, `b_var`, `c` before `.id()`) borrow the builder, so you keep the `NodeId`s to refer to nodes
after building.

## Propagating changes forward

Setting a node's value and *cascading* recomputes everything downstream of it:

```rust,ignore
{{#include ../../keanu/tests/book.rs:cascade}}
```

If you change several nodes, set them all first and cascade once, so C is only computed once:

```rust,ignore
{{#include ../../keanu/tests/book.rs:lazy}}
```

## Observing a value

Observing a node fixes its value: it's data, not something to infer. Probabilistic nodes (and boolean deterministic
nodes, as constraints) can be observed:

```rust,ignore
{{#include ../../keanu/tests/book.rs:observe}}
```

## Evaluating

Nodes start without values unless you set or observe them. `lazy_eval` computes what's needed for the nodes you
ask for, sampling probabilistic nodes from their priors (`eval` recomputes everything):

```rust,ignore
{{#include ../../keanu/tests/book.rs:eval}}
```
