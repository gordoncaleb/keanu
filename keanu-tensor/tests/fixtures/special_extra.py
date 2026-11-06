#!/usr/bin/env python3
"""High-precision reference values for keanu-tensor's Bessel, Lambert W, hypergeometric and Owen's T functions
(tests/special_extra.rs).

Regenerate with mpmath (pip install mpmath) from the repository root:

    python3 keanu-tensor/tests/fixtures/special_extra.py > keanu-tensor/tests/fixtures/special_extra.json

Every value is computed with 50 significant digits and rounded once to the nearest f64. Each case carries a
tolerance `tol` (1e-14 in general; for K_nu and I_nu themselves and their scaled forms 3e-14, or 6e-16 times |ln|
of the value (or for nu >= 30 of sqrt(nu^2 + x^2), the size of the terms of Debye's exponent) if that is larger,
since an exponentially large or small value carries the rounding error of its exponent;
2e-14 for Owen's T; for 2F1 by region and derivative order, see `hyp_tol`) and a floor `floor`: the test checks |actual - expected| <= tol * max(|expected|, floor). Logs of
functions that over- or underflow (ln K_nu, ln I_nu) use floor 1, so that the check is on the relative error of the
function itself; everything else uses floor 0 (a plain relative error). Owen's T is computed by mpmath's
quadrature of its defining integral (in y = h x, with breakpoints), independently of the reductions the Rust code
uses. The derivatives of 2F1 in its parameters are mpmath's numerical derivatives at 50 digits. Values that are
zero or outside the f64 range are left out.
"""

import json

import mpmath as mp

mp.mp.dps = 50

NORMAL = 1e-14
VALUE = 3e-14
cases = []


def add(name, args, value, tol=NORMAL, floor=0.0):
    if not mp.isfinite(value):
        return
    v = float(value)
    if v == 0.0 or abs(v) < 1e-300 or abs(v) > 1e300:
        return
    cases.append({"f": name, "x": [float(a) for a in args], "v": v, "tol": tol, "floor": floor})


def mpf(x):
    return mp.mpf(float(x))


def exponential(value, magnitude=0.0):
    # a function computed as the exponential of its log l has a relative error of a few units in the last place of
    # |l|, or of the largest term of l (|nu| sqrt(1 + (x/nu)^2) in Debye's expansions)
    return max(VALUE, 6e-16 * max(abs(float(mp.log(value))), magnitude)) if value > 0 else VALUE


# modified Bessel functions
ORDERS = [0.0, 1e-9, 0.1, 0.25, 0.5, 0.75, 1.0, 1.4, 1.5, 2.0, 2.5, 3.3, 5.0, 7.5, 10.0, 17.5, 29.9, 30.0, 31.0,
          50.0, 100.0, 500.0, 2000.0]
ARGS = [1e-300, 1e-20, 1e-6, 1e-3, 0.1, 0.5, 1.0, 1.9, 2.0, 2.1, 3.0, 5.0, 10.0, 20.0, 29.0, 31.0, 50.0, 100.0,
        400.0, 900.0, 1000.0, 3000.0, 1e4, 1e6, 1e10]
for nu in ORDERS:
    for x in ARGS:
        n, z = mpf(nu), mpf(x)
        try:
            k = mp.besselk(n, z)
            k1 = mp.besselk(n + 1, z)
        except (ValueError, mp.libmp.NoConvergence):
            # mpmath's hypergeometric sums don't converge for some huge arguments: leave those out
            continue
        add("log_bessel_k", [nu, x], mp.log(k), NORMAL, 1.0)
        add("bessel_k_ratio", [nu, x], k1 / k)
        add("log_bessel_k_dx", [nu, x], n / z - k1 / k)
        debye = (nu * nu + x * x) ** 0.5 if nu >= 30 else 0.0
        add("bessel_k", [nu, x], k, exponential(k, debye))
        add("bessel_k_scaled", [nu, x], mp.exp(z) * k, exponential(mp.exp(z) * k))
        try:
            i = mp.besseli(n, z)
            i1 = mp.besseli(n + 1, z)
        except (ValueError, mp.libmp.NoConvergence):
            i = None
        if i is not None:
            add("log_bessel_i", [nu, x], mp.log(i), NORMAL, 1.0)
            add("bessel_i_ratio", [nu, x], i1 / i)
            add("log_bessel_i_dx", [nu, x], i1 / i + n / z)
            add("bessel_i", [nu, x], i, exponential(i, debye))
            add("bessel_i_scaled", [nu, x], mp.exp(-z) * i, exponential(mp.exp(-z) * i))

# Lambert W
for x in [-0.36787944117144, -0.3678, -0.367, -0.36, -0.35, -0.3, -0.25, -0.2, -0.1, -0.01, -1e-5, -1e-12, -1e-300,
          1e-300, 1e-12, 1e-5, 0.01, 0.1, 0.5, 1.0, 2.0, 2.718281828459045, 3.0, 10.0, 100.0, 1e5, 1e10, 1e100,
          1e300]:
    add("lambert_w0", [x], mp.lambertw(mpf(x), 0).real)
    if x < 0:
        add("lambert_wm1", [x], mp.lambertw(mpf(x), -1).real)


# Owen's T
def owens_t(h, a):
    h, a = abs(mpf(h)), mpf(a)
    if h == 0:
        return mp.atan(a) / (2 * mp.pi)
    if h < 1e-3:
        f = lambda x: mp.exp(-h * h * (1 + x * x) / 2) / (1 + x * x)
        points = [0] + [p for p in [1, 10, 100, 1000] if p < a] + [a]
        return mp.quad(f, points) / (2 * mp.pi)
    upper = min(a * h, mpf(80))
    f = lambda y: mp.exp(-y * y / 2) / (1 + (y / h) ** 2)
    points = [0] + [p for p in [0.25, 0.5, 1, 2, 4, 8, 16, 32] if p < upper] + [upper]
    return mp.exp(-h * h / 2) / (2 * mp.pi * h) * mp.quad(f, points)


OWENS_T = 2e-14
for h in [0.0, 1e-10, 1e-3, 0.1, 0.5, 0.67, 0.7, 1.0, 1.01, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 25.0, 35.0]:
    for a in [1e-10, 1e-3, 0.1, 0.5, 0.9, 0.999, 1.0, 1.001, 1.5, 3.0, 10.0, 100.0, 1e5]:
        t = owens_t(h, a)
        add("owens_t", [h, a], t, OWENS_T)
        add("owens_t", [-h, a], t, OWENS_T)
        add("owens_t", [h, -a], -t, OWENS_T)

# Gauss hypergeometric function
PARAMS = [
    (1.0, 1.0, 2.0), (0.5, 0.5, 1.5), (2.0, 3.0, 4.5), (0.3, 1.7, 2.9), (-0.5, 1.5, 2.5), (-3.0, 2.0, 1.5),
    (2.5, -1.3, 3.1), (5.0, 7.0, 3.0), (12.0, 0.5, 13.0), (1.0, 2.0, 3.0), (0.1, 0.2, 0.3), (10.0, 10.0, 21.0),
    (1.5, 2.5, 4.0), (25.0, 1.5, 26.0), (-2.5, -1.5, 0.7),
]
ZS = [-100.0, -20.0, -3.0, -1.0, -0.75, -0.5, -0.1, 0.0, 1e-8, 0.1, 0.3, 0.5, 0.6, 0.75, 0.9, 0.99, 0.999, 1 - 1e-6,
      1 - 1e-10]
def hyp_tol(z, order):
    # values, first and second derivatives; the continuation loses accuracy in the second derivatives as z nears 1
    if order == 0:
        return NORMAL if -1 <= z < 0.999 else 5e-14
    if order == 1:
        return 1e-12 if z <= 0.999 else 1e-11
    if z <= 0.999:
        return 1e-11
    return 1e-9 if z <= 1 - 1e-6 else 1e-5


for a, b, c in PARAMS:
    for z in ZS:
        A, B, C, Z = mpf(a), mpf(b), mpf(c), mpf(z)
        f = lambda a_, b_, c_, z_: mp.hyp2f1(a_, b_, c_, z_)
        add("hyp2f1", [a, b, c, z], f(A, B, C, Z), hyp_tol(z, 0))
        add("hyp2f1_dz", [a, b, c, z], mp.diff(lambda t: f(A, B, C, t), Z), hyp_tol(z, 0))
        for i, name in enumerate(["hyp2f1_da", "hyp2f1_db", "hyp2f1_dc"]):
            def along(t, i=i):
                p = [A, B, C]
                p[i] = t
                return f(p[0], p[1], p[2], Z)
            add(name, [a, b, c, z], mp.diff(along, [A, B, C][i]), hyp_tol(z, 1))
        add("hyp2f1_dz2", [a, b, c, z], mp.diff(lambda t: f(A, B, C, t), Z, 2), hyp_tol(z, 2))
        add("hyp2f1_da2", [a, b, c, z], mp.diff(lambda t: f(t, B, C, Z), A, 2), hyp_tol(z, 2))
        add("hyp2f1_dadz", [a, b, c, z], mp.diff(lambda s, t: f(s, B, C, t), (A, Z), (1, 1)), hyp_tol(z, 2))
        add("hyp2f1_dbdc", [a, b, c, z], mp.diff(lambda s, t: f(A, s, t, Z), (B, C), (1, 1)), hyp_tol(z, 2))

print(json.dumps({"cases": cases}, indent=0))
