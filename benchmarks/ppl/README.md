# Keanu, Stan and PyMC on the same models

Ten models from Keanu's examples, written for Keanu, Stan (CmdStan) and PyMC, sampled with each library's NUTS on
the same data, and compared on time and effective sample size (ESS) per second.

| Model | Keanu example | What it exercises |
|---|---|---|
| `eight_schools` | `eight_schools.rs` (non-centred) | a tiny hierarchical model: per-step overhead |
| `radon` | `radon.rs` (centred) | 85 group intercepts gathered over 919 observations |
| `robust_regression` | `robust_regression.rs` | a Student-t likelihood with learnt degrees of freedom |
| `stochastic_volatility` | `stochastic_volatility.rs` | a 200-step AR(1) latent series (a loop: Keanu's `scan`) |
| `gp_regression` | `gp_regression.rs` | an exact GP marginal likelihood (40 × 40 Cholesky) |
| `gaussian_mixture` | `gaussian_mixture.rs` | a two-component mixture likelihood on 300 points |
| `item_response` | `item_response.rs` | a 2PL item response model with 700 Bernoulli-logit responses |
| `hmm` | `hmm.rs` | a two-state hidden Markov model over 400 steps (forward algorithm) |
| `disease_mapping` | `disease_mapping.rs` | a BYM2 spatial model (ICAR) with a Poisson likelihood |
| `lotka_volterra` | `lotka_volterra.rs` | an ODE (RK45 with forward sensitivities) |

## Method

- **Same model, same data.** The Keanu runner (`keanu/benches/ppl_comparison.rs`) builds each model as its example
  does and writes the data to `out/<model>/data.json`, which the Stan programs (`stan/`) and the PyMC models
  (`pymc_models.py`) read. Priors and parameterisations match; where a library has a built-in for something Keanu
  composes (Stan's `hmm_marginal`, `ode_rk45_tol`, `sum_to_zero_vector`; PyMC's `NormalMixture`, `ICAR`, `pm.ode`)
  it is used, as a user of that library would.
- **Same sampler settings.** 4 chains in parallel, 1000 warm-up iterations and 1000 draws each, diagonal mass
  matrix, and the example's target acceptance rate (0.95 for eight schools and stochastic volatility, 0.9 for
  Lotka–Volterra, 0.8 otherwise). Each library uses its own default initialisation except for Lotka–Volterra, where
  Stan's and PyMC's random initial values strand chains far from the posterior, so every library starts there from
  the example's centres.
- **Time** is the wall time of sampling, after compilation: Keanu's from the start of the chains to the end, Stan's
  from launching CmdStan to its exit (the model is compiled beforehand), PyMC's `sampling_time` (its compile time is
  excluded).
- **ESS** is the minimum over every parameter of ArviZ's rank-normalised bulk and tail ESS, computed by the same code
  for all three libraries' draws.
- **Leapfrog steps** are counted over warm-up and sampling; time per gradient is wall time × chains ÷ steps.
- **Posterior agreement**: the largest difference between a library's posterior means and Stan's, in Stan's
  posterior standard deviations.
- **A shared machine.** `python run.py all -n 3` interleaves the three libraries model by model and keeps each
  library's fastest of three runs. Every run of a library draws the same samples (the seeds are fixed), so only the
  time varies; the fastest run is the one least disturbed by other work on the machine.

## Running it

```sh
# once: a virtual environment with PyMC, CmdStanPy and ArviZ, and CmdStan itself
python3 -m venv benchmarks/ppl/.venv
benchmarks/ppl/.venv/bin/pip install pymc cmdstanpy arviz
benchmarks/ppl/.venv/bin/python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

# Keanu (writes the data the others read), then Stan and PyMC, then the report
cargo bench -p keanu --features parallel --bench ppl_comparison
cd benchmarks/ppl
.venv/bin/python run.py stan
.venv/bin/python run.py pymc
.venv/bin/python run.py report

# or all three, interleaved, best of three
.venv/bin/python run.py all -n 3
.venv/bin/python run.py report
```

The Keanu runner has two more modes for profiling: `KEANU_BENCH_GRADIENT=<models>` times one gradient and one
single-chain leapfrog step per model, and `KEANU_BENCH_CHECK=1` checks that the compiled log probability and
gradient are bit-for-bit the general path's at a few points per model.

## Results

See [the results in `docs/benchmarks.md`](../../docs/benchmarks.md#keanu-stan-and-pymc).
