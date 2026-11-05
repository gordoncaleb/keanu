#!/usr/bin/env python3
"""High-precision reference values for keanu-tensor's special functions (tests/special.rs).

Regenerate with mpmath (pip install mpmath) from the repository root:

    python3 keanu-tensor/tests/fixtures/special.py > keanu-tensor/tests/fixtures/special.json

Every value is computed with 50 significant digits and rounded once to the nearest f64. Each case carries the
relative tolerance the test applies: 1e-12 on a function's normal range, 1e-9 in its tails and for the
shape-parameter derivatives of the incomplete gamma and beta functions (taken by mpmath's numerical
differentiation at high precision). Values that are zero or outside the f64 range are left out; the test checks
those by hand.
"""

import json

import mpmath as mp

mp.mp.dps = 50

NORMAL = 1e-12
TAIL = 1e-9
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

# log-scaled modified Bessel functions of the first kind
for k in [1e-10, 1e-3, 0.1, 1.0, 5.0, 15.0, 29.9, 30.1, 50.0, 100.0, 1e3, 1e4, 1e5, 1e8]:
    add("log_bessel_i0", [k], mp.log(mp.besseli(0, mpf(k))))
    add("log_bessel_i1", [k], mp.log(mp.besseli(1, mpf(k))))

print('{"cases": [\n' + ",\n".join(json.dumps(c) for c in cases) + "\n]}")
