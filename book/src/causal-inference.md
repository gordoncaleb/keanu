# Interventions

A fitted model answers two different questions about a treatment X and an outcome Y:

- **Conditioning**, p(Y | X = x): what is Y like among the cases where X *happens to be* x? Observing X tells us
  about its causes too, so this mixes the effect of X with the effects of whatever made X take that value.
- **Intervening**, p(Y | do(X = x)): what would Y be if X were *set* to x, whatever its causes? X's causes keep
  their own distributions, and only X's effect on Y remains.

They differ whenever a confounder drives both X and Y. If sicker patients are more likely to be treated, the
treated do worse than the untreated even when the treatment helps; comparing them answers the conditioning
question, not the causal one.

## The do-operator

`intervene(&graph, &state, &[(x, value)])` rewrites a graph so that each listed node is a constant holding the value
given, with its label. It returns the new graph and state and an `Intervened` record:

- an intervened latent loses its distribution, and so its term in the joint log probability;
- an intervened observed node's observation is replaced by the value;
- an intervened deterministic node stops computing from its parents;
- a node that influenced the model only through intervened nodes (an unobserved node all of whose children are
  intervened or dropped, such as the coefficients of the treatment's logistic regression) is dropped;
- everything else is copied with its label, value and observation, and keeps its dependence on its parents, so a
  latent downstream of an intervened node depends on its new value. Deterministic nodes downstream of an
  intervention are recomputed.

`intervened.node(id)` gives the new node of each original node (`None` for the dropped ones), and
`intervened.dropped()` lists the dropped nodes. Several nodes can be intervened on at once, with values of their
type and shape.

## Interventional predictions

The posterior of the original model is what we know about the parameters; the intervened graph says what happens
under the intervention. `PredictiveSampler::interventional` combines them: it carries each posterior draw over to
the intervened graph (`NetworkSamples::map_nodes` with `Intervened::carried`, leaving out the intervened and dropped
nodes), sets the intervened values and draws the targets, giving draws from p(y | do(x), data). The targets are the
original graph's observed nodes, and the draws are recorded under their new ids:

```rust,ignore
{{#include ../../keanu/tests/book.rs:intervene}}
```

The average treatment effect is the difference between two interventions' predictive means, E[Y | do(X = 1)] −
E[Y | do(X = 0)], computed draw by draw for its posterior. The [causal effect example](examples.md#a-causal-effect)
fits a confounded study and shows conditioning overstating the effect almost threefold while the interventions
recover it.

## Limits

Interventions answer questions about populations: each predictive draw keeps the posterior's parameters and draws
everything downstream of the intervention afresh. A counterfactual about one unit ("would this patient have
recovered without treatment?") also needs that unit's own noise, inferred from what was observed of it (abduction);
Keanu gives only what the posterior predictive on an intervened graph gives. Latents that stay in the intervened
graph keep their posterior draws, which for a unit-level latent upstream of the intervention is the abducted value.
