import json, sys
import numpy as np
import arviz_stats
from arviz_stats.base.array import array_stats as A

rng = np.random.default_rng(20261105)


def ar1(chains, draws, rho, offsets):
    x = np.zeros((chains, draws))
    for c in range(chains):
        v = rng.normal()
        for t in range(draws):
            v = rho * v + rng.normal()
            x[c, t] = v + offsets[c]
    return x


cases = {
    "ar1_offset_odd_draws": ar1(4, 101, 0.5, [0.0, 0.0, 0.3, -0.2]),
    "student_t_with_ties": np.round(rng.standard_t(3, size=(3, 60)), 1),
    "scale_mismatch": rng.normal(size=(4, 80)) * np.array([[1.0], [1.0], [1.0], [3.0]]),
    "small_two_chains": rng.normal(size=(2, 7)),
    "single_chain": ar1(1, 40, 0.8, [1.0]),
    "strongly_correlated": ar1(2, 300, 0.95, [0.0, 0.0]),
}


def enc(v):
    # JSON has no NaN: write it as null
    return None if v != v else v


out = {
    "generator": "arviz-stats %s, numpy %s, scipy; see the comment in tests/diagnostics.rs"
    % (arviz_stats.__version__, np.__version__),
    "cases": [],
}
for name, x in cases.items():
    out["cases"].append(
        {
            "name": name,
            "chains": x.tolist(),
            "r_hat": enc(float(A._rhat_rank(x))),
            "ess_bulk": enc(float(A._ess_bulk(x))),
            "ess_tail": enc(float(A._ess_tail(x, (0.05, 0.95)))),
            "mcse_mean": enc(float(A._mcse_mean(x))),
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)),
            "q5": float(np.quantile(x, 0.05)),
            "q50": float(np.quantile(x, 0.5)),
            "q95": float(np.quantile(x, 0.95)),
        }
    )
json.dump(out, sys.stdout, indent=1)
print()
