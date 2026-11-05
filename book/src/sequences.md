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
  simplex under the gradient based algorithms.
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
