# Example: thermometers

A room's temperature is somewhere between 20 and 30 degrees. Two thermometers read it: one reads 25 with a standard
deviation of 2.5, and the other, worse one reads 30 with a standard deviation of 5. What is the temperature?

```rust,ignore
{{#include ../../keanu/tests/book.rs:thermometers}}
```

The temperature's starting value has to be inside the uniform prior's support (outside it, the probability is zero).

The most probable temperature weights each reading by its precision: (25/2.5² + 30/5²) / (1/2.5² + 1/5²) = 26.
See [Most probable values](inference-map.md) to compute it, and [Posterior sampling](inference-sampling.md) for the
whole posterior.
