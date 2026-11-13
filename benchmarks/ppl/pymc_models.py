"""The benchmark models in PyMC, one function per model, each taking the data dict written by the Keanu runner."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan


def eight_schools(d):
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfCauchy("tau", 5)
        eta = pm.Normal("eta", 0, 1, shape=d["J"])
        pm.Normal("y", mu + tau * eta, np.array(d["sigma"]), observed=np.array(d["y"]))
    return m, 0.95


def radon(d):
    county = np.array(d["county"])
    with pm.Model() as m:
        mu_a = pm.Normal("mu_a", 0, 10)
        sigma_a = pm.Exponential("sigma_a", 1)
        b = pm.Normal("b", 0, 10)
        sigma_y = pm.Exponential("sigma_y", 1)
        a = pm.Normal("a", mu_a, sigma_a, shape=d["J"])
        pm.Normal("log_radon", a[county] + b * np.array(d["floor"]), sigma_y, observed=np.array(d["log_radon"]))
    return m, 0.8


def robust_regression(d):
    x = np.array(d["x"])
    with pm.Model() as m:
        intercept = pm.Normal("intercept", 0, 1)
        slope = pm.Normal("slope", 0, 2)
        scale = pm.LogNormal("scale", np.log(0.5), 0.7)
        nu = pm.LogNormal("nu", np.log(5.0), 0.9)
        pm.StudentT("y", nu=nu, mu=intercept + slope * x, sigma=scale, observed=np.array(d["y"]))
    return m, 0.8


def stochastic_volatility(d):
    T = d["T"]
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 10)
        phi_raw = pm.Beta("phi_raw", 20, 1.5)
        phi = pm.Deterministic("phi", 2 * phi_raw - 1)
        sigma = pm.HalfCauchy("sigma", 5)
        eps_0 = pm.Normal("eps_0", 0, 1)
        eps = pm.Normal("eps", 0, 1, shape=T)
        h0 = mu + sigma / pt.sqrt(1 - phi**2) * eps_0
        h, _ = scan(
            lambda shock, prev, level, phi: level + phi * prev + shock,
            sequences=[sigma * eps],
            outputs_info=[h0],
            non_sequences=[mu * (1 - phi), phi],
        )
        h = pm.Deterministic("h", h)
        pm.Normal("r", 0, pt.exp(h / 2), observed=np.array(d["r"]))
    return m, 0.95


def gp_regression(d):
    x = np.array(d["x"])[:, None]
    with pm.Model() as m:
        amplitude = pm.HalfNormal("amplitude", 2)
        length_scale = pm.HalfNormal("length_scale", 2)
        noise = pm.HalfNormal("noise", 1)
        cov = amplitude**2 * pm.gp.cov.ExpQuad(1, ls=length_scale)
        gp = pm.gp.Marginal(cov_func=cov)
        gp.marginal_likelihood("y", X=x, y=np.array(d["y"]), sigma=pt.sqrt(noise**2 + 1e-6))
    return m, 0.8


def gaussian_mixture(d):
    with pm.Model() as m:
        first = pm.Normal("first", 0, 10)
        gap = pm.HalfNormal("gap", 10)
        means = pm.Deterministic("means", pt.stack([first, first + gap]))
        sigmas = pm.HalfNormal("sigmas", 5, shape=2)
        weights = pm.Dirichlet("weights", a=np.array([2.0, 2.0]))
        pm.NormalMixture("y", w=weights, mu=means, sigma=sigmas, observed=np.array(d["y"]))
    return m, 0.8


def item_response(d):
    student, item = np.array(d["student"]), np.array(d["item"])
    with pm.Model() as m:
        ability = pm.Normal("ability", 0, 1, shape=d["S"])
        difficulty_mean = pm.Normal("difficulty_mean", 0, 1)
        difficulty_scale = pm.LogNormal("difficulty_scale", 0, 0.5)
        difficulty_offset = pm.Normal("difficulty_offset", 0, 1, shape=d["I"])
        difficulty = pm.Deterministic("difficulty", difficulty_mean + difficulty_scale * difficulty_offset)
        log_discrimination_mean = pm.Normal("log_discrimination_mean", 0, 0.4)
        log_discrimination_scale = pm.LogNormal("log_discrimination_scale", np.log(0.3), 0.5)
        discrimination_offset = pm.Normal("discrimination_offset", 0, 1, shape=d["I"])
        discrimination = pm.Deterministic(
            "discrimination", pt.exp(log_discrimination_mean + log_discrimination_scale * discrimination_offset)
        )
        logit = discrimination[item] * (ability[student] - difficulty[item])
        pm.Bernoulli("correct", logit_p=logit, observed=np.array(d["correct"]))
    return m, 0.8


def hmm(d):
    y = np.array(d["y"])
    with pm.Model() as m:
        rows = pm.Dirichlet("rows", a=np.ones((2, 2)), shape=(2, 2))
        transition = pm.Deterministic("transition", rows)
        first = pm.Normal("first", 0, 5)
        gap = pm.HalfNormal("gap", 5)
        means = pm.Deterministic("means", pt.stack([first, first + gap]))
        sigma = pm.HalfNormal("sigma", 2)
        # the forward algorithm in log space
        log_emission = pm.logp(pm.Normal.dist(means[None, :], sigma), y[:, None])  # [T, 2]
        log_transition = pt.log(transition)
        alpha0 = np.log([0.5, 0.5]) + log_emission[0]
        alphas, _ = scan(
            lambda e, alpha, lt: pt.logsumexp(alpha[:, None] + lt, axis=0) + e,
            sequences=[log_emission[1:]],
            outputs_info=[alpha0],
            non_sequences=[log_transition],
        )
        pm.Potential("likelihood", pt.logsumexp(alphas[-1]))
    return m, 0.8


def disease_mapping(d):
    N = d["N"]
    W = np.zeros((N, N))
    for i, j in zip(d["node1"], d["node2"]):
        W[i, j] = W[j, i] = 1
    with pm.Model() as m:
        beta0 = pm.Normal("beta0", 0, 1)
        beta1 = pm.Normal("beta1", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        rho = pm.Beta("rho", 1, 1)
        phi = pm.ICAR("phi", W=W)
        theta = pm.Normal("theta", 0, 1, shape=N)
        effect = pm.Deterministic(
            "effect", sigma * (pt.sqrt(rho / d["scaling_factor"]) * phi + pt.sqrt(1 - rho) * theta)
        )
        log_rate = np.log(np.array(d["E"])) + beta0 + beta1 * np.array(d["x"]) + effect
        pm.Poisson("y", mu=pt.exp(log_rate), observed=np.array(d["y"]))
    return m, 0.8


def lotka_volterra(d):
    def rhs(z, t, rates):
        return [z[0] * (rates[0] - rates[1] * z[1]), z[1] * (rates[3] * z[0] - rates[2])]

    ode = pm.ode.DifferentialEquation(func=rhs, times=np.array(d["ts"]), n_states=2, n_theta=4, t0=0)
    with pm.Model() as m:
        rates = pm.LogNormal("rates", np.log([0.8, 0.6, 0.6, 0.5]), 0.35, shape=4)
        z = ode(y0=np.array(d["y0"]), theta=rates)
        pm.LogNormal("y", pt.log(z), d["noise"], observed=np.array(d["y"]))
    return m, 0.9


MODELS = {
    name: fn
    for name, fn in globals().items()
    if callable(fn) and not name.startswith("_") and fn.__module__ == __name__
}
