#!/usr/bin/env python3
"""High-precision reference values of the Wiener first-passage time log density (tests/wiener.rs).

Regenerate with mpmath (pip install mpmath) from the repository root:

    python3 keanu-distributions/tests/fixtures/wiener.py > keanu-distributions/tests/fixtures/wiener.json

The standard density f(u | w) (boundaries 0 and 1, no drift, start w, at the lower boundary) is the method-of-images
sum (2 pi u^3)^(-1/2) sum_k (w + 2k) exp(-(w + 2k)^2 / (2u)) over enough images for every u, at a working precision
raised with u to absorb its cancellation (f falls like exp(-pi^2 u / 2) while its terms do not), so it is
independent of the choice between the small- and large-time series that the Rust code makes. With sv = 0 the log
density is Navarro and Fuss's -2 ln a - a v w - v^2 t / 2 + ln f(t / a^2 | w) (w -> 1 - w and v -> -v at the upper
boundary). With sv > 0 it is not the closed form of Blurton et al. (2017) that the Rust code uses but its definition,
the sv = 0 density integrated over the drift's normal distribution by mpmath's quadrature. Each value is computed
with at least 40 significant digits and rounded once to the nearest f64.
"""

import json

import mpmath as mp


def log_f(u, w):
    """ln f(u | w) by the method of images."""
    extra = int(u * mp.pi ** 2 / 2 / mp.log(10)) + 10
    with mp.workdps(50 + extra):
        u, w = mp.mpf(u), mp.mpf(w)
        n = int(mp.ceil(10 * mp.sqrt(u) + 10))
        total = mp.fsum((w + 2 * k) * mp.exp(-((w + 2 * k) ** 2) / (2 * u)) for k in range(-n, n + 1))
        return mp.log(total) - mp.mpf(1.5) * mp.log(u) - mp.log(2 * mp.pi) / 2


def log_density_fixed(t, a, w, v, upper):
    if upper:
        w, v = 1 - w, -v
    return -2 * mp.log(a) - a * v * w - v * v * t / 2 + log_f(t / (a * a), w)


def log_density(x, a, t0, w, v, sv, upper):
    mp.mp.dps = 40
    x, a, t0, w, v, sv = (mp.mpf(z) for z in (x, a, t0, w, v, sv))
    t = x - t0
    if sv == 0:
        return log_density_fixed(t, a, w, v, upper)
    # the density averaged over the drift: f(u | w) does not depend on it
    lf = log_density_fixed(t, a, w, 0, upper)
    sign = -1 if upper else 1
    ww = 1 - w if upper else w

    def integrand(nu):
        drift = sign * nu
        return mp.exp(-a * drift * ww - drift * drift * t / 2) * mp.npdf(nu, v, sv)

    integral = mp.quad(integrand, [-mp.inf, v - 8 * sv, v, v + 8 * sv, mp.inf])
    return lf + mp.log(integral)


cases = []
T0 = 0.2
for t in [0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
    for a in [0.5, 1.0, 2.0, 3.0]:
        for w in [0.05, 0.3, 0.5, 0.8, 0.97]:
            for v in [-3.0, -0.5, 0.0, 1.2, 4.0]:
                for upper in [False, True]:
                    x = T0 + t
                    lp = log_density(x, a, T0, w, v, 0.0, upper)
                    cases.append({"x": x, "a": a, "t0": T0, "w": w, "v": v, "sv": 0.0, "boundary": int(upper),
                                  "log_p": float(lp)})
for t in [0.03, 0.4, 1.5, 6.0]:
    for a in [0.8, 2.0]:
        for w in [0.2, 0.6]:
            for v in [-1.5, 0.7, 3.0]:
                for sv in [0.4, 1.5]:
                    for upper in [False, True]:
                        x = T0 + t
                        lp = log_density(x, a, T0, w, v, sv, upper)
                        cases.append({"x": x, "a": a, "t0": T0, "w": w, "v": v, "sv": sv, "boundary": int(upper),
                                      "log_p": float(lp)})

print(json.dumps(cases, indent=None, separators=(",", ":")))
