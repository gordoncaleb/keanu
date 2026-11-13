"""Runs the Stan and PyMC versions of the benchmark models, and compares them with Keanu's run.

    python run.py stan  [model ...]   # sample with CmdStan, write out/<model>/stan.{csv,json}
    python run.py pymc  [model ...]   # sample with PyMC,    write out/<model>/pymc.{csv,json}
    python run.py report              # print the comparison table and write out/report.{md,json}

The Keanu side (out/<model>/data.json and keanu.{csv,json}) is written first by

    cargo bench -p keanu --features parallel --bench ppl_comparison -- benchmarks/ppl/out

Every library samples 4 chains in parallel with 1000 warm-up iterations and 1000 draws, at the example's target
acceptance rate, and every library's draws go through the same ESS code (ArviZ's rank-normalised bulk and tail ESS).
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CHAINS, WARMUP, DRAWS, SEED = 4, 1000, 1000, 1
MODELS = [
    "eight_schools",
    "radon",
    "robust_regression",
    "stochastic_volatility",
    "gp_regression",
    "gaussian_mixture",
    "item_response",
    "hmm",
    "disease_mapping",
    "lotka_volterra",
]
# Keanu starts every chain at the example's starting point. Stan and PyMC use their own default initialisation
# except where that fails: from Stan's random inits some Lotka–Volterra chains get stuck far from the posterior.
INITS = {"lotka_volterra": {"rates": [0.8, 0.6, 0.6, 0.5]}}
TARGET_ACCEPT = {"eight_schools": 0.95, "stochastic_volatility": 0.95, "lotka_volterra": 0.9}


def keanu_columns(model):
    with open(OUT / model / "keanu.csv") as f:
        return f.readline().strip().split(",")[2:]


def write_draws(model, lib, columns, arrays, stats):
    """arrays: name -> [chain, draw] array, for each of `columns`."""
    rows = []
    for c in range(CHAINS):
        for d in range(DRAWS):
            rows.append([c, d] + [arrays[col][c, d] for col in columns])
    np.savetxt(
        OUT / model / f"{lib}.csv",
        np.array(rows),
        delimiter=",",
        header=",".join(["chain", "draw"] + columns),
        comments="",
        fmt=["%d", "%d"] + ["%.10e"] * len(columns),
    )
    with open(OUT / model / f"{lib}.json", "w") as f:
        json.dump(stats, f, indent=2)


def run_stan(models):
    import cmdstanpy

    for model in models:
        data = OUT / model / "data.json"
        compile_start = time.perf_counter()
        sm = cmdstanpy.CmdStanModel(stan_file=HERE / "stan" / f"{model}.stan")
        compile_seconds = time.perf_counter() - compile_start
        start = time.perf_counter()
        fit = sm.sample(
            data=str(data),
            chains=CHAINS,
            parallel_chains=CHAINS,
            iter_warmup=WARMUP,
            iter_sampling=DRAWS,
            adapt_delta=TARGET_ACCEPT.get(model, 0.8),
            seed=SEED,
            inits=INITS.get(model),
            save_warmup=True,
            show_progress=False,
            show_console=False,
        )
        seconds = time.perf_counter() - start
        draws = fit.draws(inc_warmup=True)  # [iteration, chain, column]
        names = list(fit.column_names)
        leapfrog = draws[:, :, names.index("n_leapfrog__")]
        divergent = draws[WARMUP:, :, names.index("divergent__")]
        arrays = {}
        for col in keanu_columns(model):
            arrays[col] = draws[WARMUP:, :, names.index(stan_name(col))].T
        stats = {
            "seconds": seconds,
            "compile_seconds": compile_seconds,
            "leapfrog": int(leapfrog.sum()),
            "warmup_leapfrog": int(leapfrog[:WARMUP].sum()),
            "divergences": int(divergent.sum()),
        }
        write_draws(model, "stan", keanu_columns(model), arrays, stats)
        print(f"{model:>22}: stan {seconds:7.2f} s, {stats['leapfrog']:>9} leapfrog, {stats['divergences']} div")


def stan_name(col):
    """eta.0 -> eta[1], transition.0.1 -> transition[1,2]"""
    parts = col.split(".")
    if len(parts) == 1:
        return col
    return f"{parts[0]}[{','.join(str(int(p) + 1) for p in parts[1:])}]"


def run_pymc(models):
    import pymc as pm

    import pymc_models

    for model in models:
        with open(OUT / model / "data.json") as f:
            data = json.load(f)
        m, target_accept = pymc_models.MODELS[model](data)
        start = time.perf_counter()
        with m:
            idata = pm.sample(
                draws=DRAWS,
                tune=WARMUP,
                chains=CHAINS,
                cores=CHAINS,
                target_accept=target_accept,
                random_seed=SEED,
                initvals={k: np.array(v) for k, v in INITS.get(model, {}).items()} or None,
                discard_tuned_samples=False,
                progressbar=False,
                compute_convergence_checks=False,
            )
        wall = time.perf_counter() - start
        seconds = float(idata.posterior.attrs["sampling_time"])
        steps_draw = idata.sample_stats["n_steps"].values
        steps_warm = idata.warmup_sample_stats["n_steps"].values
        arrays = {}
        for col in keanu_columns(model):
            parts = col.split(".")
            values = idata.posterior[parts[0]].values  # [chain, draw, ...]
            index = tuple(int(p) for p in parts[1:])
            arrays[col] = values[(slice(None), slice(None)) + index]
        stats = {
            "seconds": seconds,
            "wall_seconds": wall,
            "leapfrog": int(steps_draw.sum() + steps_warm.sum()),
            "warmup_leapfrog": int(steps_warm.sum()),
            "divergences": int(idata.sample_stats["diverging"].values.sum()),
        }
        write_draws(model, "pymc", keanu_columns(model), arrays, stats)
        print(f"{model:>22}: pymc {seconds:7.2f} s ({wall:.2f} s wall), {stats['leapfrog']:>9} leapfrog")


def load(model, lib):
    csv, js = OUT / model / f"{lib}.csv", OUT / model / f"{lib}.json"
    if not csv.exists():
        return None
    table = np.loadtxt(csv, delimiter=",", skiprows=1, ndmin=2)
    columns = open(csv).readline().strip().split(",")[2:]
    draws = table[:, 2:].reshape(CHAINS, DRAWS, len(columns))
    stats = json.load(open(js))
    return columns, draws, stats


def report():
    from arviz_stats.base import array_stats

    libs = ["keanu", "stan", "pymc"]
    rows, results = [], {}
    for model in MODELS:
        results[model] = {}
        means = {}
        for lib in libs:
            loaded = load(model, lib)
            if loaded is None:
                continue
            columns, draws, stats = loaded
            ess_bulk = array_stats.ess(draws, chain_axis=0, draw_axis=1, method="bulk")
            ess_tail = array_stats.ess(draws, chain_axis=0, draw_axis=1, method="tail", prob=(0.05, 0.95))
            ess = float(np.nanmin(np.minimum(ess_bulk, ess_tail)))
            per_grad = stats["seconds"] * CHAINS / stats["leapfrog"] * 1e6
            means[lib] = draws.mean(axis=(0, 1))
            results[model][lib] = {
                "seconds": stats["seconds"],
                "min_ess": ess,
                "ess_per_second": ess / stats["seconds"],
                "leapfrog": stats["leapfrog"],
                "us_per_gradient": per_grad,
                "divergences": stats["divergences"],
                "worst": columns[int(np.nanargmin(np.minimum(ess_bulk, ess_tail)))],
            }
        # posterior agreement: the largest gap between Keanu's posterior means and Stan's, in Stan posterior sds
        if "keanu" in means and "stan" in means:
            _, sdraws, _ = load(model, "stan")
            sd = sdraws.std(axis=(0, 1))
            for lib in ("keanu", "pymc"):
                if lib in means:
                    gap = np.abs(means[lib] - means["stan"]) / np.maximum(sd, 1e-12)
                    results[model][lib]["max_mean_gap_sd"] = float(gap.max())

    header = (
        f"| Model | Library | Time (s) | Min ESS | ESS/s | Leapfrog steps | µs/gradient | Div. | Mean gap (sd) |\n"
        f"|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for model, r in results.items():
        for lib in libs:
            if lib not in r:
                continue
            x = r[lib]
            gap = f"{x['max_mean_gap_sd']:.2f}" if "max_mean_gap_sd" in x else "—"
            lines.append(
                f"| {model} | {lib} | {x['seconds']:.2f} | {x['min_ess']:.0f} | {x['ess_per_second']:.0f} | "
                f"{x['leapfrog']} | {x['us_per_gradient']:.1f} | {x['divergences']} | {gap} |"
            )
    lines.append("")
    lines.append("| Model | Keanu ESS/s ÷ Stan | Keanu ESS/s ÷ PyMC | Keanu µs/grad ÷ Stan | Keanu µs/grad ÷ PyMC |")
    lines.append("|---|---:|---:|---:|---:|")
    for model, r in results.items():
        if "keanu" not in r:
            continue
        k = r["keanu"]
        cells = []
        for key in ("ess_per_second", "us_per_gradient"):
            for lib in ("stan", "pymc"):
                cells.append(f"{k[key] / r[lib][key]:.2f}" if lib in r else "—")
        lines.append(f"| {model} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    text = "\n".join(lines)
    print(text)
    (OUT / "report.md").write_text(text + "\n")
    (OUT / "report.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    cmd, *models = sys.argv[1:]
    models = models or MODELS
    {"stan": lambda: run_stan(models), "pymc": lambda: run_pymc(models), "report": report}[cmd]()
