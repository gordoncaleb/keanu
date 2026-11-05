"""Reference values for tests/psis.rs and tests/loo.rs, from arviz-stats.

The inputs are deterministic formulas that the Rust tests rebuild exactly: u(i, a, p) = ((a·i + 13) mod p + ½) / p.
Run with `python arviz_loo.py > arviz_loo.json`.
"""
import json
import numpy as np
import scipy
import arviz_stats
from scipy.special import logsumexp
from scipy.optimize import minimize, Bounds, LinearConstraint
from arviz_stats.base import array_stats


def u(i, a, p):
    return ((i * a + 13) % p + 0.5) / p


# PSIS: arviz-stats' psislw takes a log likelihood and smooths the log ratios -ll; S = 1600 so that its tail length,
# floor(3·√S), equals loo's ceil(3·√S) = 120
S = 1600
i = np.arange(S)
psis_cases = {
    "gpd_shape_0_6": np.log(((1 - u(i, 7919, 1601)) ** (-0.6) - 1) / 0.6),
    "bounded_trig": 2.0 * np.sin(i * 0.37) + 0.5 * np.cos(i * 1.3),
}
checked = [0, 1, 2, 17, 500, 999, 1599]
psis_out = []
for name, lr in psis_cases.items():
    lw, k = array_stats.psislw(-lr, r_eff=1.0)
    idx = sorted(set(checked + [int(np.argmax(lr)), int(np.argmin(lr))]))
    psis_out.append({
        "name": name,
        "pareto_k": float(k),
        "normalized_log_weights": {str(j): float(lw[j]) for j in idx},
    })

# LOO: two models of six observations, 1000 draws; r_eff = 1
S = 1000
y = np.array([-0.4, 0.1, 0.3, 0.8, 1.2, 4.0])
s = np.arange(S)
m = 0.2 + 0.8 * (u(s, 7919, 1009) - 0.5)
sd = 0.8 + 0.4 * u(s, 104729, 1013)
models = {
    "narrow": -0.5 * np.log(2 * np.pi) - np.log(sd)[:, None]
    - 0.5 * ((y[None, :] - m[:, None]) / sd[:, None]) ** 2,
    "wide": -0.5 * np.log(2 * np.pi) - np.log(2.0) - 0.5 * ((y[None, :] - m[:, None]) / 2.0) ** 2,
}
loo_out = []
elpds = []
for name, ll in models.items():
    elpd_i, k_i, p_i = [], [], []
    for j in range(ll.shape[1]):
        lw, k = array_stats.psislw(ll[:, j], r_eff=1.0)
        e = logsumexp(lw + ll[:, j])
        lpd = logsumexp(ll[:, j]) - np.log(S)
        elpd_i.append(float(e))
        k_i.append(float(k))
        p_i.append(float(lpd - e))
    elpd_i = np.array(elpd_i)
    elpds.append(elpd_i)
    n = len(elpd_i)
    loo_out.append({
        "name": name,
        "elpd": float(elpd_i.sum()),
        "se": float(np.sqrt(n * np.var(elpd_i, ddof=1))),
        "p_loo": float(np.sum(p_i)),
        "elpd_i": elpd_i.tolist(),
        "pareto_k": k_i,
    })

# stacking weights exactly as arviz-stats' compare computes them
exp_ic = np.exp(np.stack(elpds, axis=1))
result = minimize(
    lambda w: -np.sum(np.log(exp_ic @ w)),
    np.full(2, 0.5),
    jac=lambda w: -np.sum(exp_ic / (exp_ic @ w)[:, None], axis=0),
    bounds=Bounds(np.zeros(2), np.ones(2)),
    constraints=LinearConstraint(np.ones(2), 1, 1),
    tol=1e-12,
)

print(json.dumps({
    "generator": "arviz-stats %s, numpy %s, scipy %s; see arviz_loo.py"
    % (arviz_stats.__version__, np.__version__, scipy.__version__),
    "psis": psis_out,
    "loo": loo_out,
    "stacking_weights": result.x.tolist(),
}, indent=1))
