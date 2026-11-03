# Introduction

Keanu is a probabilistic programming library. You describe a model as a **Bayesian network**: random variables
(probabilistic nodes), the deterministic operations that combine them, and observations of some of them. Keanu then
answers questions about the variables you didn't observe (the **latents**):

- **What are their most probable values?** Gradient-based optimization (maximum a posteriori or maximum likelihood).
- **What are their posterior distributions?** Markov chain Monte Carlo: Metropolis–Hastings, the No-U-Turn sampler
  (NUTS) and simulated annealing, plus forward sampling from the prior and a particle filter for finding probable
  starting states.

Every node holds an n-dimensional tensor, so a single node can stand for a whole vector or matrix of variables, and
gradients are computed by automatic differentiation.

This is the Rust implementation of [Keanu](https://github.com/improbable-research/keanu), originally a Java library.
The model semantics are the same; [Coming from Java Keanu](from-java.md) lists the differences.

## How the pieces fit

```text
GraphBuilder ──build()──▶ Graph (immutable, Send + Sync) ─┐
                          State (values and observations) ─┼─▶ Model ─▶ optimizers, samplers
```

A `Graph` describes the model and never changes. All values live in a `State`, which is cheap to clone, so several
chains, particles or optimizer runs can share one graph, each with its own state.
