#!/usr/bin/env python3
"""Taylor coefficients of the coefficient functions C_k(eta) of Temme's uniform asymptotic expansion of the
regularised incomplete gamma function (DLMF 8.12.3-8.12.9), for keanu-tensor's `special.rs`.

    Q(a, x) = erfc(eta sqrt(a/2)) / 2 + e^(-a eta^2 / 2) / sqrt(2 pi a) sum_k C_k(eta) a^-k,

with lambda = x / a and eta = sign(lambda - 1) sqrt(2 (lambda - 1 - ln lambda)). The C_k satisfy
C_0 = 1/(lambda - 1) - 1/eta and C_k = C'_(k-1)(eta) / eta + c_k / (lambda - 1), where 1/Gamma*(a) ~ sum c_k a^-k.
Their removable singularities at eta = 0 make the closed forms cancel badly near the transition, so the code
evaluates their Taylor series instead, computed here exactly in rational arithmetic (standard library only) and
rounded once to f64:

    python3 keanu-tensor/tests/fixtures/temme.py [K] [N]

prints a Rust array of K (default 12) rows of N + 1 (default 30) coefficients, rows C_0 ... C_(K-1), each in
increasing powers of eta.
"""

import sys
from fractions import Fraction as F
from math import comb

K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
D = N + 2 * K + 4  # working degree: each step of the recurrence loses two


def mul(a, b, n=D):
    r = [F(0)] * n
    for i, x in enumerate(a[:n]):
        if x:
            for j, y in enumerate(b[: n - i]):
                r[i + j] += x * y
    return r


def inv(a, n=D):
    r = [F(0)] * n
    r[0] = 1 / a[0]
    for k in range(1, n):
        r[k] = -sum(a[j] * r[k - j] for j in range(1, min(k, len(a) - 1) + 1)) / a[0]
    return r


def sqrt1(a, n=D):
    # the square root of a series with constant term 1
    r = [F(0)] * n
    r[0] = F(1)
    for k in range(1, n):
        r[k] = (a[k] - sum(r[j] * r[k - j] for j in range(1, k))) / 2
    return r


def compose(f, g, n=D):
    # f(g(x)) for g(0) = 0
    r = [F(0)] * n
    p = [F(1)] + [F(0)] * (n - 1)
    for c in f[:n]:
        if c:
            r = [x + c * y for x, y in zip(r, p)]
        p = mul(p, g, n)
    return r


# eta = mu h(mu) with mu = lambda - 1 and h(mu)^2 = 2 (mu - ln(1 + mu)) / mu^2 = 2 sum_(j>=2) (-1)^j mu^(j-2) / j
h = sqrt1([F(2 * (-1) ** j, j) for j in range(2, D + 2)])
# revert to mu(eta) by iterating mu = eta / h(mu), which fixes one more coefficient each time
mu = [F(0), F(1)] + [F(0)] * (D - 2)
for _ in range(D + 1):
    new = [F(0)] + inv(compose(h, mu))[: D - 1]
    if new == mu:
        break
    mu = new
mu_over_eta = mu[1:] + [F(0)]
eta_over_mu = inv(mu_over_eta)

# 1/Gamma*(a) = exp(-sum_j B_2j / (2j (2j - 1)) a^(1 - 2j)), as a series in t = 1/a
bern = [F(0)] * (2 * K + 6)
bern[0] = F(1)
for m in range(1, len(bern)):
    bern[m] = -sum(comb(m + 1, k) * bern[k] for k in range(m)) / (m + 1)
s = [F(0)] * (K + 2)
for j in range(1, K + 2):
    if 2 * j - 1 < K + 2:
        s[2 * j - 1] = -bern[2 * j] / (2 * j * (2 * j - 1))
c = [F(0)] * (K + 2)
c[0] = F(1)
for k in range(1, K + 2):
    c[k] = sum(j * s[j] * c[k - j] for j in range(1, k + 1)) / k

# C_0 = (eta/mu - 1) / eta, then the recurrence; each pole at eta = 0 must cancel
t = eta_over_mu[:]
t[0] -= 1
C = [t[1:] + [F(0)]]
for k in range(1, K):
    prev = C[-1]
    total = [j * prev[j] for j in range(1, len(prev))] + [F(0)]
    total = [x + c[k] * y for x, y in zip(total, eta_over_mu)]
    assert total[0] == 0, (k, total[0])
    C.append(total[1:] + [F(0)])

rows = []
for Ck in C:
    rows.append("    [\n" + "".join(f"        {float(x)!r},\n" for x in Ck[: N + 1]) + "    ],\n")
print(f"const TEMME: [[f64; {N + 1}]; {K}] = [\n" + "".join(rows) + "];")
