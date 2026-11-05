#!/usr/bin/env python3
"""High-precision reference values for keanu-tensor's special functions (tests/special.rs).

Regenerate with mpmath (pip install mpmath) from the repository root:

    python3 keanu-tensor/tests/fixtures/special.py > keanu-tensor/tests/fixtures/special.json

Every value is computed with 50 significant digits and rounded once to the nearest f64. Each case carries the
relative tolerance the test applies: 1e-12 on a function's normal range, 1e-9 in its tails and for the
shape-parameter derivatives of the incomplete gamma and beta functions (taken by mpmath's numerical
differentiation at high precision, or by quadrature for large shapes), and 1e-9 (1e-8 for the derivatives) for the
incomplete beta function with shapes of 1e6 and more. Values that are zero or outside the f64 range are left out;
the test checks those by hand.
"""

import json

import mpmath as mp

mp.mp.dps = 50

NORMAL = 1e-12
TAIL = 1e-9
LARGE_SHAPE = 1e-9
LARGE_SHAPE_DERIVATIVE = 1e-8
cases = []


def add(name, args, value, tol=NORMAL):
    if not mp.isfinite(value):
        return
    v = float(value)
    if v == 0.0 or abs(v) < 1e-300 or abs(v) > 1e300:
        return
    cases.append({"f": name, "x": [float(a) for a in args], "v": v, "tol": tol})


def mpf(x):
    return mp.mpf(float(x))


# error functions
for x in [1e-300, 1e-20, 1e-8, 1e-3, 0.1, 0.25, 0.4769362762044699, 0.5, 0.9, 1.0, 1.49, 1.5, 1.51, 2.0, 2.5,
          3.0, 4.0, 5.0, 5.9]:
    for s in (1, -1):
        add("erf", [s * x], mp.erf(mpf(s * x)))
for x in [-5.0, -1.0, -0.3, 1e-20, 1e-5, 0.3, 0.5, 1.0, 1.4, 1.5, 1.6, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 15.0, 20.0,
          26.0, 26.5]:
    add("erfc", [x], mp.erfc(mpf(x)), TAIL if x > 5 else NORMAL)
for x in [-3.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0, 10.0, 100.0, 1e5, 1e10, 1e100]:
    add("erfcx", [x], mp.exp(mpf(x) ** 2) * mp.erfc(mpf(x)))
for y in [1e-300, 1e-15, 1e-5, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999999, 1 - 1e-10, 1 - 2.0 ** -52]:
    for s in (1, -1):
        add("erf_inv", [s * y], mp.erfinv(mpf(s * y)))
for q in [1e-300, 1e-100, 1e-20, 1e-5, 0.01, 0.3, 0.5, 0.9, 1.1, 1.5, 1.7, 1.99, 2 - 1e-10]:
    # erfc(w) = q  <=>  erf(w) = 1 - q
    add("erfc_inv", [q], mp.erfinv(1 - mpf(q)))

# normal distribution functions
for x in [-37.5, -30.0, -20.0, -10.0, -5.0, -2.0, -1.0, -1e-3, 1e-3, 0.5, 1.0, 3.0, 8.0]:
    add("ndtr", [x], mp.ncdf(mpf(x)), TAIL if x < -5 else NORMAL)
for x in [-1e6, -1e3, -100.0, -40.0, -38.0, -20.0, -10.0, -5.0, -3.0, -2.0, -1.0, -0.1, 0.1, 1.0, 2.0, 3.0,
          5.0, 8.0, 10.0, 20.0, 37.0]:
    add("log_ndtr", [x], mp.log(mp.ncdf(mpf(x))), TAIL if abs(x) > 5 else NORMAL)
for p in [1e-300, 1e-100, 1e-20, 1e-10, 1e-3, 0.02425, 0.1, 0.3, 0.4999, 0.5001, 0.7, 0.9, 0.97575, 0.999,
          1 - 1e-10, 1 - 2.0 ** -53]:
    add("ndtri", [p], -mp.sqrt(2) * mp.erfinv(1 - 2 * mpf(p)), TAIL if p < 1e-10 or p > 1 - 1e-10 else NORMAL)

# elementary log-space helpers
for x in [-800.0, -40.0, -10.0, -1.0, -1e-10, 0.0, 1e-10, 1.0, 10.0, 40.0, 800.0]:
    add("softplus", [x], mp.log1p(mp.exp(mpf(x))))
for p in [1e-300, 1e-10, 0.01, 0.25, 0.4, 0.5 - 1e-10, 0.5 + 1e-10, 0.6, 0.75, 0.99, 1 - 1e-10]:
    add("logit", [p], mp.log(mpf(p) / (1 - mpf(p))))
for x in [-800.0, -40.0, -10.0, -1.0, -0.6931471805599453, -0.5, -1e-5, -1e-20]:
    add("log1mexp", [x], mp.log(1 - mp.exp(mpf(x))))

# log beta
for a, b in [(0.5, 0.5), (1.0, 2.0), (2.5, 3.5), (0.01, 7.0), (1e-10, 3.0), (9.0, 11.0), (3.0, 15.0), (0.5, 1e10),
             (2.0, 1e6), (25.0, 40.0), (1e3, 1e3), (1e6, 2e6), (7.5, 1e3), (1e-5, 1e-5)]:
    add("log_beta", [a, b], mp.log(mp.beta(mpf(a), mpf(b))))


# regularised incomplete gamma and its derivative in a
def gp(a, x):
    return mp.gammainc(a, 0, x, regularized=True)


def gq(a, x):
    return mp.gammainc(a, x, mp.inf, regularized=True)


for a in [1e-10, 0.01, 0.1, 0.5, 1.0, 2.5, 10.0, 50.0, 200.0, 1000.0, 1e4]:
    for x in sorted({a * 0.01, a * 0.5, a * 0.9, a, a + 1, a * 1.1 + 0.5, 2 * a + 1, 5 * a + 10, 0.3, 1.0, 3.0}):
        A, X = mpf(a), mpf(x)
        p, q = gp(A, X), gq(A, X)
        add("gamma_p", [a, x], p, TAIL if p < 1e-6 else NORMAL)
        add("gamma_q", [a, x], q, TAIL if q < 1e-6 else NORMAL)
        if a >= 0.01:
            # differentiate whichever of P and Q is smaller, so the difference quotient keeps its precision
            if p < q:
                add("gamma_p_da", [a, x], mp.diff(lambda s: gp(s, X), A), TAIL)
            else:
                add("gamma_p_da", [a, x], -mp.diff(lambda s: gq(s, X), A), TAIL)


# regularised incomplete beta and its derivatives in a and b
def ib(a, b, x):
    return mp.betainc(a, b, 0, x, regularized=True)


def ib_upper(a, b, x):
    # 1 - I_x(a, b) = I_(1-x)(b, a); mpmath subtracts for the interval [x, 1], losing the small result
    return mp.betainc(b, a, 0, 1 - x, regularized=True)


for a in [0.1, 0.5, 1.0, 2.5, 10.0, 100.0, 1000.0]:
    for b in [0.1, 0.5, 1.0, 2.5, 10.0, 100.0, 1000.0]:
        for x in [1e-3, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999]:
            A, B, X = mpf(a), mpf(b), mpf(x)
            i = ib(A, B, X)
            add("inc_beta", [a, b, x], i, TAIL if min(i, 1 - i) < 1e-6 else NORMAL)
            # differentiate whichever of I and 1 - I is smaller, as for the incomplete gamma function
            if i < 0.5:
                add("inc_beta_da", [a, b, x], mp.diff(lambda s: ib(s, B, X), A), TAIL)
                add("inc_beta_db", [a, b, x], mp.diff(lambda s: ib(A, s, X), B), TAIL)
            else:
                add("inc_beta_da", [a, b, x], -mp.diff(lambda s: ib_upper(s, B, X), A), TAIL)
                add("inc_beta_db", [a, b, x], -mp.diff(lambda s: ib_upper(A, s, X), B), TAIL)


# Large shapes, where the series and continued fractions would need ~10 sqrt(shape) terms near the peak. mpmath's
# hypergeometric series don't converge there either, so these come from quadrature of the density (and of its
# derivative in the shape) over the smaller tail: in steps of `step` (the standard deviation, or the decay length
# of the density at x if shorter) from x out to 60 steps beyond x or the mean.
def tail_quad(log_density, d_log_density, x, mean, step, lower):
    """(the smaller tail's integral, and the derivative of the lower tail's)."""
    f = lambda t: mp.exp(log_density(t))
    g = lambda t: d_log_density(t) * f(t)
    if lower:
        lo = min(mean, x) - 60 * step
        pts = [lo + k * step for k in range(int((x - lo) / step) + 1)] + [x]
        return mp.quad(f, pts), mp.quad(g, pts)
    hi = max(mean, x) + 60 * step
    pts = [x] + [x + k * step for k in range(1, int((hi - x) / step) + 1)] + [hi]
    return mp.quad(f, pts), -mp.quad(g, pts)


def gamma_quad(a, x):
    """(P, Q, dP/da) for a large shape by quadrature."""
    A, X = mpf(a), mpf(x)
    lg, psi = mp.loggamma(A), mp.digamma(A)
    lower = X <= A
    slope = abs((A - 1) / X - 1)
    step = min(mp.sqrt(A), 1 / slope) if slope else mp.sqrt(A)
    small, dp = tail_quad(lambda t: (A - 1) * mp.log(t) - t - lg, lambda t: mp.log(t) - psi, X, A, step, lower)
    return (small, 1 - small, dp) if lower else (1 - small, small, dp)


def add_gamma_logs(a, x, p, q, dp):
    add("log_gamma_p", [a, x], mp.log(p))
    add("log_gamma_q", [a, x], mp.log(q))
    add("log_gamma_p_da", [a, x], dp / p, TAIL)
    add("log_gamma_q_da", [a, x], -dp / q, TAIL)


for a in [1e6, 1e8, 1e11, 1e12]:
    for c in [-8.0, -3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0, 8.0]:
        x = a + c * a ** 0.5
        p, q, dp = gamma_quad(a, x)
        add("gamma_p", [a, x], p)
        add("gamma_q", [a, x], q)
        add("gamma_p_da", [a, x], dp, TAIL)
        add_gamma_logs(a, x, p, q, dp)

for a, b in [(1e6, 1e6), (1e8, 3e8), (1e10, 1e10), (3e10, 1e10)]:
    A, B = mpf(a), mpf(b)
    lb, psi_a, psi_b, psi_ab = mp.log(mp.beta(A, B)), mp.digamma(A), mp.digamma(B), mp.digamma(A + B)
    mean = A / (A + B)
    sd = mp.sqrt(A * B / ((A + B) ** 2 * (A + B + 1)))
    for c in [-8.0, -1.0, 0.0, 0.5, 3.0]:
        x = float(mean + c * sd)
        X = mpf(x)
        lower = X <= mean
        log_density = lambda t: (A - 1) * mp.log(t) + (B - 1) * mp.log(1 - t) - lb
        small, da = tail_quad(log_density, lambda t: mp.log(t) - psi_a + psi_ab, X, mean, sd, lower)
        _, db = tail_quad(log_density, lambda t: mp.log(1 - t) - psi_b + psi_ab, X, mean, sd, lower)
        i = small if lower else 1 - small
        # the continued fraction takes ~sqrt(shape) steps here, and its rounding errors accumulate
        add("inc_beta", [a, b, x], i, LARGE_SHAPE)
        add("log_inc_beta", [a, b, x], mp.log(i), LARGE_SHAPE)
        add("inc_beta_da", [a, b, x], da, LARGE_SHAPE_DERIVATIVE)
        add("inc_beta_db", [a, b, x], db, LARGE_SHAPE_DERIVATIVE)
        add("log_inc_beta_da", [a, b, x], da / i, LARGE_SHAPE_DERIVATIVE)
        add("log_inc_beta_db", [a, b, x], db / i, LARGE_SHAPE_DERIVATIVE)

# logs in the far tails, where the functions themselves underflow (or nearly), and the derivatives of the logs
for a, x in [(201.0, 1.0), (2.0, 800.0), (0.5, 1e4), (10.0, 1e3), (1e3, 10.0), (1e3, 3e3), (50.0, 1e-5),
             (0.1, 1e-300), (30.0, 500.0), (2.5, 0.3), (150.0, 100.0), (150.0, 220.0)]:
    A, X = mpf(a), mpf(x)
    p, q = gp(A, X), gq(A, X)
    add("log_gamma_p", [a, x], mp.log(p))
    add("log_gamma_q", [a, x], mp.log(q))
    add("log_gamma_p_da", [a, x], mp.diff(lambda s: mp.log(gp(s, X)), A), TAIL)
    add("log_gamma_q_da", [a, x], mp.diff(lambda s: mp.log(gq(s, X)), A), TAIL)
for a, x in [(1e6, 3e5), (1e6, 2e6), (1e12, 5e11), (1e12, 2e12)]:
    add_gamma_logs(a, x, *gamma_quad(a, x))

for a, b, x in [(1001.0, 3.0, 0.4), (2.5, 0.5, 1e-140), (901.0, 100.0, 0.001), (0.5, 0.5, 1e-300), (50.0, 50.0, 0.01),
                (1e3, 1e3, 0.1), (10.0, 2.0, 1e-40), (3.0, 1001.0, 0.9), (2.0, 7.0, 0.3), (0.3, 4.0, 0.95)]:
    A, B, X = mpf(a), mpf(b), mpf(x)
    add("log_inc_beta", [a, b, x], mp.log(ib(A, B, X)))
    add("log_inc_beta_da", [a, b, x], mp.diff(lambda s: mp.log(ib(s, B, X)), A), TAIL)
    add("log_inc_beta_db", [a, b, x], mp.diff(lambda s: mp.log(ib(A, s, X)), B), TAIL)

# log-scaled modified Bessel functions of the first kind
for k in [1e-10, 1e-3, 0.1, 1.0, 5.0, 15.0, 29.9, 30.1, 50.0, 100.0, 1e3, 1e4, 1e5, 1e8]:
    add("log_bessel_i0", [k], mp.log(mp.besseli(0, mpf(k))))
    add("log_bessel_i1", [k], mp.log(mp.besseli(1, mpf(k))))

print('{"cases": [\n' + ",\n".join(json.dumps(c) for c in cases) + "\n]}")
