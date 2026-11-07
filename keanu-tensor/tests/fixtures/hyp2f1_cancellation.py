"""Reference values of the Gauss hypergeometric function 2F1(a, b; c; z) for parameters whose series terms change
sign (negative a or b, c below a or b), where the sum cancels, written to hyp2f1_cancellation.json with mpmath at 60
digits. Run with `python3 hyp2f1_cancellation.py > hyp2f1_cancellation.json`."""

import json
import random

import mpmath as mp

mp.mp.dps = 60
random.seed(2026)

# reported by review: relative errors of 1e-10 to 3e-3 before the double-double evaluation
cases = [
    (25.0, 25.0, 0.3, -1.0), (-20.0, -20.0, 0.5, -1.0), (-23.8, 23.32, 5.42, 0.786), (15.89, -24.73, 7.55, 0.921),
    (-10.0, 5.0, 2.5, 0.99), (-3.0, 2.0, 4.0, -1e6),
]
zs = [lambda: round(random.uniform(-1, 0.999), 3), lambda: round(random.uniform(-100, -1), 2),
      lambda: random.choice([0.999, 1 - 1e-6, 1 - 1e-10])]
for _ in range(600):
    a, b = round(random.uniform(-25, 25), 2), round(random.uniform(-25, 25), 2)
    c = round(random.uniform(0.3, 26), 2)
    cases.append((a, b, c, random.choice(zs)()))
for _ in range(200):
    # polynomials
    a, b = float(random.randint(-25, 0)), round(random.uniform(-25, 25), 2)
    c = round(random.uniform(0.3, 26), 2)
    cases.append((a, b, c, random.choice(zs)()))

out = []
for a, b, c, z in cases:
    v = mp.hyp2f1(mp.mpf(a), mp.mpf(b), mp.mpf(c), mp.mpf(z))
    out.append({"x": [a, b, c, z], "v": float(v)})
print(json.dumps({"cases": out}, indent=0))
