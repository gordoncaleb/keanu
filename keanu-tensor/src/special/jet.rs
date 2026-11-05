//! Second-order forward-mode dual numbers ("jets") for differentiating the series, continued fractions and
//! asymptotic expansions of the incomplete gamma and beta functions with respect to their shapes.
//!
//! A [`Jet<K>`] carries a value with its gradient and Hessian with respect to `K` variables, and every operation
//! applies the chain rule to second order. `Jet<0>` is a plain number (its arrays are empty), so one generic
//! implementation of an algorithm gives the value alone (K = 0), or the value with its derivatives (K = 1 for the
//! incomplete gamma function's shape, K = 2 for the incomplete beta function's two shapes), from the same
//! arithmetic.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A value with its gradient `g` and (symmetric) Hessian `h` with respect to `K` variables.
#[derive(Debug, Clone, Copy)]
pub(super) struct Jet<const K: usize> {
    pub(super) v: f64,
    pub(super) g: [f64; K],
    pub(super) h: [[f64; K]; K],
}

impl<const K: usize> Jet<K> {
    /// A constant: zero derivatives.
    pub(super) const fn constant(v: f64) -> Self {
        Self {
            v,
            g: [0.0; K],
            h: [[0.0; K]; K],
        }
    }

    /// NaN with NaN derivatives.
    pub(super) const NAN: Self = Self {
        v: f64::NAN,
        g: [f64::NAN; K],
        h: [[f64::NAN; K]; K],
    };

    /// A value with NaN derivatives (for a value at the end of its range, such as ln 0, where they don't exist).
    pub(super) const fn without_derivatives(v: f64) -> Self {
        Self {
            v,
            g: [f64::NAN; K],
            h: [[f64::NAN; K]; K],
        }
    }

    /// The `i`-th variable, with value `v` (a constant if `i >= K`).
    pub(super) fn var(v: f64, i: usize) -> Self {
        let mut jet = Self::constant(v);
        if i < K {
            jet.g[i] = 1.0;
        }
        jet
    }

    /// f(self), given f, f' and f'' at `self.v`.
    pub(super) fn chain(self, f0: f64, f1: f64, f2: f64) -> Self {
        let mut r = Self::constant(f0);
        for i in 0..K {
            r.g[i] = f1 * self.g[i];
            for j in 0..K {
                r.h[i][j] = f1 * self.h[i][j] + f2 * self.g[i] * self.g[j];
            }
        }
        r
    }

    /// f(x, y), given f's value, gradient (fx, fy) and Hessian (fxx, fxy, fyy) at (x.v, y.v).
    pub(super) fn bivariate(
        value: f64,
        (x, y): (Self, Self),
        (fx, fy): (f64, f64),
        (fxx, fxy, fyy): (f64, f64, f64),
    ) -> Self {
        let mut r = Self::constant(value);
        for i in 0..K {
            r.g[i] = fx * x.g[i] + fy * y.g[i];
            for j in 0..K {
                r.h[i][j] = fx * x.h[i][j]
                    + fy * y.h[i][j]
                    + fxx * x.g[i] * x.g[j]
                    + fxy * (x.g[i] * y.g[j] + y.g[i] * x.g[j])
                    + fyy * y.g[i] * y.g[j];
            }
        }
        r
    }

    pub(super) fn ln(self) -> Self {
        let r = 1.0 / self.v;
        self.chain(self.v.ln(), r, -r * r)
    }

    /// ln(1 + self).
    pub(super) fn ln_1p(self) -> Self {
        let r = 1.0 / (1.0 + self.v);
        self.chain(self.v.ln_1p(), r, -r * r)
    }

    /// e^self, given its value (computed more accurately elsewhere than as `self.v.exp()`, say as a product).
    pub(super) fn exp_with(self, value: f64) -> Self {
        self.chain(value, value, value)
    }

    pub(super) fn exp(self) -> Self {
        self.exp_with(self.v.exp())
    }

    pub(super) fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        self.chain(s, 0.5 / s, -0.25 / (s * self.v))
    }

    pub(super) fn recip(self) -> Self {
        let r = 1.0 / self.v;
        self.chain(r, -r * r, 2.0 * r * r * r)
    }

    /// Whether this term is negligible against `sum`: each of its components is at most `tolerance` times the
    /// sum's.
    pub(super) fn negligible(&self, sum: &Self, tolerance: f64) -> bool {
        let small = |t: f64, s: f64| t.abs() <= tolerance * s.abs();
        small(self.v, sum.v)
            && (0..K).all(|i| {
                small(self.g[i], sum.g[i]) && (0..K).all(|j| small(self.h[i][j], sum.h[i][j]))
            })
    }
}

impl<const K: usize> Add for Jet<K> {
    type Output = Self;
    fn add(mut self, o: Self) -> Self {
        self.v += o.v;
        for i in 0..K {
            self.g[i] += o.g[i];
            for j in 0..K {
                self.h[i][j] += o.h[i][j];
            }
        }
        self
    }
}

impl<const K: usize> Add<f64> for Jet<K> {
    type Output = Self;
    fn add(mut self, o: f64) -> Self {
        self.v += o;
        self
    }
}

impl<const K: usize> Add<Jet<K>> for f64 {
    type Output = Jet<K>;
    fn add(self, o: Jet<K>) -> Jet<K> {
        o + self
    }
}

impl<const K: usize> Neg for Jet<K> {
    type Output = Self;
    fn neg(self) -> Self {
        self * -1.0
    }
}

impl<const K: usize> Sub for Jet<K> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        self + -o
    }
}

impl<const K: usize> Sub<f64> for Jet<K> {
    type Output = Self;
    fn sub(mut self, o: f64) -> Self {
        self.v -= o;
        self
    }
}

impl<const K: usize> Sub<Jet<K>> for f64 {
    type Output = Jet<K>;
    fn sub(self, o: Jet<K>) -> Jet<K> {
        let mut r = -o;
        r.v = self - o.v;
        r
    }
}

impl<const K: usize> Mul for Jet<K> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        let mut r = Self::constant(self.v * o.v);
        for i in 0..K {
            r.g[i] = self.v * o.g[i] + o.v * self.g[i];
            for j in 0..K {
                r.h[i][j] = self.v * o.h[i][j]
                    + o.v * self.h[i][j]
                    + self.g[i] * o.g[j]
                    + o.g[i] * self.g[j];
            }
        }
        r
    }
}

impl<const K: usize> Mul<f64> for Jet<K> {
    type Output = Self;
    fn mul(mut self, o: f64) -> Self {
        self.v *= o;
        for i in 0..K {
            self.g[i] *= o;
            for j in 0..K {
                self.h[i][j] *= o;
            }
        }
        self
    }
}

impl<const K: usize> Mul<Jet<K>> for f64 {
    type Output = Jet<K>;
    fn mul(self, o: Jet<K>) -> Jet<K> {
        o * self
    }
}

impl<const K: usize> Div for Jet<K> {
    type Output = Self;
    /// q = a / b from a = q b: q' = (a' − q b') / b and q'' = (a'' − q b'' − q' b'ᵀ − b' q'ᵀ) / b.
    fn div(self, o: Self) -> Self {
        let q = self.v / o.v;
        let mut r = Self::constant(q);
        for i in 0..K {
            r.g[i] = (self.g[i] - q * o.g[i]) / o.v;
        }
        for i in 0..K {
            for j in 0..K {
                r.h[i][j] =
                    (self.h[i][j] - q * o.h[i][j] - r.g[i] * o.g[j] - o.g[i] * r.g[j]) / o.v;
            }
        }
        r
    }
}

impl<const K: usize> Div<f64> for Jet<K> {
    type Output = Self;
    fn div(mut self, o: f64) -> Self {
        self.v /= o;
        for i in 0..K {
            self.g[i] /= o;
            for j in 0..K {
                self.h[i][j] /= o;
            }
        }
        self
    }
}

impl<const K: usize> Div<Jet<K>> for f64 {
    type Output = Jet<K>;
    fn div(self, o: Jet<K>) -> Jet<K> {
        Jet::constant(self) / o
    }
}

#[cfg(test)]
mod tests {
    use super::Jet;

    #[test]
    fn jets_differentiate_compositions_to_second_order() {
        // f(a, b) = ln(a b + 1) / sqrt(a) − e^(a − b) at (2, 3)
        let (a, b) = (Jet::<2>::var(2.0, 0), Jet::<2>::var(3.0, 1));
        let f = |a: Jet<2>, b: Jet<2>| (a * b + 1.0).ln() / a.sqrt() - (a - b).exp();
        let r = f(a, b);
        let value = |x: f64, y: f64| (x * y + 1.0).ln() / x.sqrt() - (x - y).exp();
        assert!((r.v - value(2.0, 3.0)).abs() < 1e-15);
        let h = 1e-4;
        let fd = |i: usize, j: usize| {
            let e = |k: usize, s: f64| if k == 0 { (s, 0.0) } else { (0.0, s) };
            let (di, dj) = (e(i, h), e(j, h));
            (value(2.0 + di.0 + dj.0, 3.0 + di.1 + dj.1)
                - value(2.0 + di.0 - dj.0, 3.0 + di.1 - dj.1)
                - value(2.0 - di.0 + dj.0, 3.0 - di.1 + dj.1)
                + value(2.0 - di.0 - dj.0, 3.0 - di.1 - dj.1))
                / (4.0 * h * h)
        };
        for i in 0..2 {
            let e = |s: f64| {
                if i == 0 {
                    (2.0 + s, 3.0)
                } else {
                    (2.0, 3.0 + s)
                }
            };
            let ((x1, y1), (x0, y0)) = (e(h), e(-h));
            let gradient = (value(x1, y1) - value(x0, y0)) / (2.0 * h);
            assert!((r.g[i] - gradient).abs() < 1e-8, "{i}");
            for j in 0..2 {
                assert!((r.h[i][j] - fd(i, j)).abs() < 1e-6, "{i} {j}");
                assert_eq!(r.h[i][j], r.h[j][i]);
            }
        }
        // division and reciprocals agree
        let q = 1.0 / (a * b);
        let p = a.recip() * b.recip();
        for i in 0..2 {
            assert!((q.g[i] - p.g[i]).abs() < 1e-15);
            for j in 0..2 {
                assert!((q.h[i][j] - p.h[i][j]).abs() < 1e-15);
            }
        }
    }
}
