# Decision 0002: graph, state and autodiff design (`keanu-graph`)

Status: accepted (2026-11-03). Covers Phase 3.

## Problem

The 188 Java vertex classes are objects that hold mutable values and pointers to their parents and children. A model
*is* its vertices, so:

- a chain, a particle or an optimizer run can't have its own state without copying the network;
- running anything in parallel is unsafe (the review found real bugs from shared state: NUTS potential reuse, and
  the particle filter reading another particle's values);
- ids come from a global counter.

The Rust version must keep Java's semantics while fixing that shape.

## Decisions

### 1. An immutable arena graph, with state kept separately

- **`Graph`** is `nodes: Vec<Node>`, where `Node { op: Op, parents: SmallVec<[NodeId; 4]>, dtype, shape, label }`.
  `NodeId` is an index into the arena.
- **Ids are assigned in creation order**, and a node's parents must already exist. The id order is therefore a
  topological order. Every traversal (eval, cascade, autodiff) uses it, which replaces Java's priority queues keyed
  by vertex id.
- **The graph is immutable once built and is `Send + Sync`.**
- **`State` holds the values:** `values: Vec<Option<Value>>` and `observed: Vec<bool>`. It's cheap to `Clone`, so
  each chain or particle gets its own state over one shared graph.
- **No global state.** Ids are per graph, and there is no default RNG.

### 2. Building graphs: a builder with typed handles

```rust
let b = GraphBuilder::new();
let mu = b.gaussian(0.0, 10.0).label("mu");
let sigma = b.half_gaussian(2.0);
let x = b.gaussian_with(&[3], mu, sigma);
let y = (x * 2.0 + 1.0).exp().sum();
let (graph, state) = b.build()?;     // the first construction error, if any, surfaces here
```

- `GraphBuilder` uses interior mutability (`RefCell`), so handles can add nodes. It is not `Send`; that's fine,
  because only the built `Graph` is shared.
- **Handles** are `Copy`, carry their dtype statically, and implement `std::ops` (`+ - * /` with tensors, handles
  and scalars; `Neg`) plus methods for everything else. The handles are `DoubleVar<'b>`, `IntVar<'b>` and
  `BoolVar<'b>`, each wrapping a `NodeId` and `&'b GraphBuilder`.
- **Shapes** are computed and validated when a node is created, as in Java.
- **Errors are deferred to `build()`.** Operators can't return `Result`, so the builder records the first error
  (with the op and label) and `build()` returns it. `builder.error()` exposes it earlier.

### 3. Values

`Value` is `F64(DoubleTensor) | I64(IntTensor) | Bool(BoolTensor)`. Java's generic-object vertices aren't ported.
`Categorical` works over category indices, and `CPT`, multiplexer and if/where work over `Value`.

### 4. Ops

`Op` is an enum, grouped into families rather than one variant per Java class:

- `Constant`
- `Unary(UnaryOp)` and `Binary(BinaryOp)`, element-wise
- `Compare`, `Mask` and `Logic` (bool ops)
- `Reduce { kind, axes }` and `Scan { kind, axis }`
- `ArgReduce`
- `Reshape`, `Permute`, `Broadcast`, `Slice`, `Concat`, `Take`, `Diag`, `DiagPart`, `Triangle`, `FillTriangular`
- `MatMul`, `TensorMultiply`, `Inverse`, `Determinant`, `Cholesky`, `CholeskyInverse`
- `Cast`
- `Where`, `Multiplexer`, `Cpt`
- `Proxy`
- `Lambda`: `Arc<dyn Fn(&[&Value]) -> Result<Value> + Send + Sync>`, with optional forward and reverse derivative
  closures
- `Assert`
- `Distribution(DistributionOp)`: the probabilistic nodes. Their parents are the distribution parameters.

Each op provides:

- a shape rule;
- `eval(parents) -> Result<Value>`;
- whether it is differentiable;
- forward and reverse autodiff rules.

The vertex-level semantics are ported from the Java vertex classes: the non-differentiable set (floor, ceil, round,
sign, comparisons, casts), the max/min reduction gradients that split ties, and broadcasting.

### 5. Value propagation (Java's `VertexValuePropagation`)

- **`State::eval(graph, nodes, rng)`** computes the nodes and all their ancestors. A probabilistic node without a
  value is sampled from its prior given its parents.
- **`State::lazy_eval(graph, nodes, rng)`** is the same, but only computes missing values.
- **`State::cascade(graph, from)`** recomputes the non-probabilistic descendants of `from` in id order and stops at
  probabilistic nodes.
- **`State::set(node, value)`** is ignored for observed nodes, as Java's `setValue` is.
- **`State::observe(node, value)`** is allowed for probabilistic nodes and for boolean non-probabilistic nodes (as
  Java's `isObservable`). Anything else is an error.
- **Observed non-probabilistic nodes keep their value** during propagation.

### 6. Autodiff

- **Forward and reverse mode both work over the arena:** forward goes in increasing id order from the "with respect
  to" node, and reverse in decreasing id order from the "of" node.
- **The partial layouts are kept from Java**, to reduce risk when porting the rules: forward partials are
  `[wrt..., of...]` internally, and the public results are `[of..., wrt...]`.
- **Traversal stops at probabilistic nodes and non-differentiable ops.** Asking for a partial that doesn't exist
  returns an error, matching the Java fix.

### 7. Log probability and its gradient

- **`log_prob(graph, state, nodes)`** is the sum of each probabilistic node's distribution log prob at its value.
- **`log_prob_gradient(graph, state, of, wrt)`** ports `LogProbGradientCalculator`. For each probabilistic node in
  `of`, it takes the distribution's `d_log_prob` with respect to its value (if unobserved) and its parameters, then
  reverse-propagates through deterministic parents to the f64 latents in `wrt`.

Both live in `keanu-graph`. Models and networks (Phase 4) build on them.

### 8. Labels

Labels are optional strings with namespaces, as in Java's `VertexLabel` (needed by `Sequence`). They must be
unique within a graph; `build()` rejects duplicates.

### 9. Proxies

A proxy node's parent is connected after creation (Java's `Sequence`). The parent must already exist (a smaller
id), which keeps id order topological; connecting to a later node is an error. A proxy evaluates to its parent's
value and is differentiable as the identity.

### 10. Not ported

- `PrintVertex`: replaced later by tracing hooks.
- `LogProbGraph` and the compiled backend: Rust evaluates graphs directly (decided in the vectors README).

## Phase 3 steps and tests

1. **Core:** graph, builder, handles, `Value`, `State`, propagation, constants and every deterministic op's value.
   `tests/golden.rs` checks every value in `golden-vectors/vertices.json`, with the coverage guard.
2. **Autodiff:** forward and reverse rules for every differentiable op. The golden test also checks
   `forwardPartials` and `reversePartials`. There is a finite-difference test for every differentiable op.
3. **Probabilistic nodes:** every distribution as a node; sampling through eval; `log_prob` and
   `log_prob_gradient`. Tests rebuild the golden models in Rust and compare against `models.json`'s expected log
   prob, log likelihood and gradients. Loading the saved files comes in Phase 4.
