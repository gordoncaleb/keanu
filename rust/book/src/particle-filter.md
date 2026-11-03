# Particle filter

A particle filter finds probable states of a model by adding observations one at a time. At each step it samples
the latents those observations depend on, for many candidate states (*particles*), and keeps the more probable ones.
The best particles make good starting states for MCMC.

```rust,ignore
{{#include ../../keanu/tests/book.rs:particle_filter}}
```

`ParticleFilter`'s settings are `particles` (how many), `resampling_cycles`, and `resampling_proportion`
(the fraction of the worst particles replaced each cycle).
