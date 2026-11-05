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

## Hidden Markov models and state-space models

Two kinds of time series have distributions of their own, whose log density is the exact marginal likelihood of the
observations with the hidden states summed or integrated out. Gradient based algorithms (NUTS, the gradient
optimizer) then work on the continuous parameters directly, with no per-step latent nodes.

**Hidden Markov models.** `hidden_markov_model(steps, emission, parameters, initial, transition)` observes a hidden
chain of K states: the first state is drawn from the `initial` probabilities ([K]), each later one from the row of
the `transition` matrix ([K, K]) for the state before it, and each step's observation from the `emission`
distribution with that state's parameters. The emission is any element-wise distribution kind (as for a mixture),
continuous or discrete, and its parameters carry a trailing state axis.

```rust,ignore
{{#include ../../keanu/tests/book.rs:hmm}}
```

- `initial` and `transition` are probabilities, not log probabilities: non-negative and summing to 1 along their last
  axis (within 1e-5). Give each row of the transition matrix a Dirichlet prior and stack the rows: they stay on the
  simplex under the gradient based algorithms. Probabilities off the simplex give a log probability of −∞ (so
  Metropolis–Hastings rejects a proposal of them), and an error when sampled.
- The value has shape [steps]; `hidden_markov_model_with_shape` takes [T, ...] for several observations per step, all
  emitted by that step's state. An emission parameter may have a leading time axis for parameters that vary over
  time.
- The log density is computed by the log-space forward algorithm in O(T K²) and its gradient by the backward
  algorithm, both exact. Drawing a value simulates the chain and then each observation. For the chain alone,
  `keanu_distributions::HiddenMarkovChain` gives the log likelihood of any per-state emission log probabilities, the
  smoothed state marginals (its gradient with respect to them) and simulated state paths.

**Linear-Gaussian state-space models.** `linear_gaussian_state_space(steps, initial_mean, initial_covariance,
transition_matrix, transition_covariance, observation_matrix, observation_covariance)` is the model of the Kalman
filter: a hidden state x of dimension d with x₀ ~ N(initial mean, initial covariance) and
x_{t+1} = F x_t + N(0, Q), observed as y_t = H x_t + N(0, R) in dimension m. The value is the observations, [steps, m].

```rust,ignore
{{#include ../../keanu/tests/book.rs:kalman}}
```

- Shapes: the initial mean [d]; the initial covariance, F and Q [d, d]; H [m, d]; R [m, m]. Mismatches are build
  errors. The covariances must be symmetric positive semidefinite; an innovation covariance H P Hᵀ + R that is not
  positive definite makes the observations impossible.
- The log likelihood uses Cholesky solves of the innovation covariance and the Joseph-form covariance update, which
  stays symmetric positive semidefinite over long series and near-deterministic models. Its gradient with respect to
  the observations and every parameter is exact (the filter differentiated in reverse mode). Covariance gradients are
  symmetric, as for the multivariate Gaussian.
- `keanu_distributions::LinearGaussianStateSpace::filter` returns the filtered means and covariances at each step.

## Scan: a loop in one node

A sequence adds one group of nodes per item, so building a 10,000-step model makes tens of thousands of nodes. When
each step is a *deterministic* function of the previous step's state, the per-step inputs and some constants, a
**scan** expresses the loop as a single node holding the loop body as a nested graph:

```rust,ignore
{{#include ../../keanu/tests/book.rs:scan}}
```

`b.scan(init, &xs, &constants, body)` calls `body` once, on a fresh builder, with placeholders for the carry, for
step t of each per-step input (sliced along its first axis) and for each constant, and the body returns the next
carry. The scan's value stacks the carry after each step, shape `[T, carry shape...]`. `b.scan_steps(T, init,
&constants, body)` is the same loop without per-step inputs (T may be 0, giving an empty value).

- The body is deterministic. Random variables come from outside: draw a vector latent for the per-step noise and pass
  it as a per-step input, as with `eps` above. A body containing a distribution is rejected when building.
- The body only sees its placeholders. Values from the enclosing graph enter as constants; using an outer handle in
  the body doesn't compile.
- There is one carry tensor. Carry several values by packing them into one tensor with `stack` (and `slice` to read
  them back) — for example `[level, trend]` for a local linear trend.
- Scans are differentiable to first and second order (NUTS, optimizers, ADVI, Laplace) through the initial carry, the
  per-step inputs and the constants. Reverse mode recomputes each step's body from the stacked output instead of
  storing it, so memory stays proportional to T. Scans nest, and save, load and export to DOT (the body is drawn as
  a cluster).

Use a **scan** for long deterministic recursions (autoregressive means, state-space filters written by hand, ODE
solvers stepped explicitly): graph building is constant in T and the model stays small. Use an **unrolled sequence**
when a step's random variables have distributions that depend on the previous step and can't be rewritten as
independent noise passed in (a Poisson count whose rate is the previous count, a discrete state chain), when you want
each step's nodes labelled in the graph, and for the [particle filter](particle-filter.md), which adds
observations and samples latents one step at a time and so needs the steps as separate nodes.
