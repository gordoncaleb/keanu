//! Second-order forward-mode dual numbers ("jets"), in which the log densities and log CDFs of the families in
//! [`crate::continuous`] that are defined by one scalar formula are written once and differentiated exactly.
//!
//! A [`Jet<K>`] carries a value with its gradient and Hessian with respect to `K` variables, and every operation
//! applies the chain rule to second order. `Jet<0>` is a plain number (its arrays are empty and the special
//! functions skip their derivatives), so one generic formula gives the log density alone (K = 0) or with its
//! gradient and Hessian with respect to the value and every parameter (K = 1 + the number of parameters).

use keanu_tensor::special::{
    bessel_i_ratio, digamma, erf, log_bessel_i0, log_gamma_p, log_gamma_p_da, log_gamma_p_da2,
    log_gamma_q, log_gamma_q_da, log_gamma_q_da2, log_inc_beta, log_inc_beta_da, log_inc_beta_da2,
    log_inc_beta_dadb, log_inc_beta_db, log_inc_beta_db2, log_ndtr, log1mexp, ndtr, owens_t,
    trigamma,
};
use std::f64::consts::{FRAC_1_SQRT_2, PI};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// ln √(2π).
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

/// One element's log density (or log CDF or log survival function) with its gradient and Hessian with respect to
/// the value (index 0) and each parameter (index `j + 1` for the j-th, in the distribution's parameter order).
///
/// Returned by the `log_prob_derivatives` and `log_cdf_derivatives` functions of the families written as one
/// scalar formula (for example [`SkewNormal::log_prob_derivatives`](crate::SkewNormal::log_prob_derivatives)),
/// whose derivatives are computed exactly by forward-mode automatic differentiation to second order. Where the
/// value is −∞ (outside the support) the derivatives are zero, and unspecified as for every gradient here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElementDerivatives<const N: usize> {
    /// The log density (or log CDF or log survival function).
    pub value: f64,
    /// Its derivatives with respect to the value and each parameter.
    pub gradient: [f64; N],
    /// Its second derivatives (symmetric).
    pub hessian: [[f64; N]; N],
}

/// A value with its gradient `g` and (symmetric) Hessian `h` with respect to `K` variables. `H` is either `K` (the
/// Hessian is carried) or 0 (first order only: the gradient path, which then skips every second derivative).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Jet<const K: usize, const H: usize = K> {
    pub(crate) v: f64,
    pub(crate) g: [f64; K],
    pub(crate) h: [[f64; H]; H],
}

/// `f` at `x` and `parameters`, with x the 0-th variable and the j-th parameter the (j + 1)-th.
pub(crate) fn derivatives<const N: usize, const P: usize>(
    x: f64,
    parameters: [f64; P],
    f: impl Fn(Jet<N>, [Jet<N>; P]) -> Jet<N>,
) -> ElementDerivatives<N> {
    debug_assert_eq!(N, P + 1);
    let jet = f(
        Jet::var(x, 0),
        std::array::from_fn(|j| Jet::var(parameters[j], j + 1)),
    );
    // the jet's Hessian is symmetric up to rounding: make it exactly so
    let hessian =
        std::array::from_fn(|i| std::array::from_fn(|j| 0.5 * (jet.h[i][j] + jet.h[j][i])));
    ElementDerivatives {
        value: jet.v,
        gradient: jet.g,
        hessian,
    }
}

/// `f` and its gradient at `x` and `parameters`, with x the 0-th variable and the j-th parameter the (j + 1)-th:
/// [`derivatives`] without the Hessian, which is not computed.
pub(crate) fn gradient<const N: usize, const P: usize>(
    x: f64,
    parameters: [f64; P],
    f: impl Fn(Jet<N, 0>, [Jet<N, 0>; P]) -> Jet<N, 0>,
) -> (f64, [f64; N]) {
    debug_assert_eq!(N, P + 1);
    let jet = f(
        Jet::var(x, 0),
        std::array::from_fn(|j| Jet::var(parameters[j], j + 1)),
    );
    (jet.v, jet.g)
}

/// `f` at `x` and `parameters`, without derivatives.
pub(crate) fn value<const P: usize>(
    x: f64,
    parameters: [f64; P],
    f: impl Fn(Jet<0>, [Jet<0>; P]) -> Jet<0>,
) -> f64 {
    f(
        Jet::constant(x),
        std::array::from_fn(|j| Jet::constant(parameters[j])),
    )
    .v
}

impl<const K: usize, const H: usize> Jet<K, H> {
    /// Whether derivatives are carried (false for a plain number, `Jet<0>`).
    const DIFFERENTIATED: bool = K > 0;
    /// Whether second derivatives are carried.
    const SECOND: bool = H > 0;

    /// A constant: zero derivatives.
    pub(crate) const fn constant(v: f64) -> Self {
        Self {
            v,
            g: [0.0; K],
            h: [[0.0; H]; H],
        }
    }

    /// The `i`-th variable, with value `v` (a constant if `i >= K`).
    pub(crate) fn var(v: f64, i: usize) -> Self {
        let mut jet = Self::constant(v);
        if i < K {
            jet.g[i] = 1.0;
        }
        jet
    }

    /// f(self), given f, f' and f'' at `self.v`.
    pub(crate) fn chain(self, f0: f64, f1: f64, f2: f64) -> Self {
        let mut r = Self::constant(f0);
        for i in 0..K {
            r.g[i] = f1 * self.g[i];
        }
        for i in 0..H {
            for j in 0..H {
                r.h[i][j] = f1 * self.h[i][j] + f2 * self.g[i] * self.g[j];
            }
        }
        r
    }

    /// f(args), given f's value, gradient and Hessian with respect to its `M` arguments.
    pub(crate) fn compose<const M: usize>(
        value: f64,
        args: &[Self; M],
        gradient: [f64; M],
        hessian: [[f64; M]; M],
    ) -> Self {
        let mut r = Self::constant(value);
        for i in 0..K {
            r.g[i] = (0..M).map(|m| gradient[m] * args[m].g[i]).sum();
        }
        for i in 0..H {
            for j in 0..H {
                let mut total = 0.0;
                for m in 0..M {
                    total += gradient[m] * args[m].h[i][j];
                    for n in 0..M {
                        total += hessian[m][n] * args[m].g[i] * args[n].g[j];
                    }
                }
                r.h[i][j] = total;
            }
        }
        r
    }

    pub(crate) fn ln(self) -> Self {
        let r = 1.0 / self.v;
        self.chain(self.v.ln(), r, -r * r)
    }

    /// ln(1 + self).
    pub(crate) fn ln_1p(self) -> Self {
        let r = 1.0 / (1.0 + self.v);
        self.chain(self.v.ln_1p(), r, -r * r)
    }

    pub(crate) fn exp(self) -> Self {
        let e = self.v.exp();
        self.chain(e, e, e)
    }

    /// e^self − 1, accurate near 0.
    pub(crate) fn exp_m1(self) -> Self {
        let e = self.v.exp();
        self.chain(self.v.exp_m1(), e, e)
    }

    pub(crate) fn sin(self) -> Self {
        let (s, c) = self.v.sin_cos();
        self.chain(s, c, -s)
    }

    pub(crate) fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        self.chain(s, 0.5 / s, -0.25 / (s * self.v))
    }

    pub(crate) fn square(self) -> Self {
        self * self
    }

    /// ln Γ(self).
    pub(crate) fn ln_gamma(self) -> Self {
        let [value, first, second] = memo_ln_gamma(self.v, Self::DIFFERENTIATED, Self::SECOND);
        if !Self::DIFFERENTIATED {
            return Self::constant(value);
        }
        self.chain(value, first, second)
    }

    /// softplus(self) = ln(1 + e^self).
    pub(crate) fn softplus(self) -> Self {
        let s = sigmoid(self.v);
        self.chain(keanu_tensor::special::softplus(self.v), s, s * (1.0 - s))
    }

    /// ln(1 − e^self), for self <= 0: with q = 1/(e^−self − 1), the derivatives are −q and −q(1 + q).
    pub(crate) fn log1mexp(self) -> Self {
        let value = log1mexp(self.v);
        if !Self::DIFFERENTIATED {
            return Self::constant(value);
        }
        let q = 1.0 / (-self.v).exp_m1();
        self.chain(value, -q, -q * (1.0 + q))
    }

    /// ln(1 − exp(−e^{−self})), the log survival function of the standard Gumbel at self, in logs for large self,
    /// where e^{−self} underflows: there it is −self + ln(1 − t/2 + t²/6) with t = e^{−self}.
    pub(crate) fn log_gumbel_ccdf(self) -> Self {
        let t = (-self).exp();
        if self.v > 11.5 {
            -self + (t * (t * (1.0 / 6.0) - 0.5)).ln_1p()
        } else {
            (-t).log1mexp()
        }
    }

    /// ln(e^self + e^other), stably.
    pub(crate) fn log_add_exp(self, other: Self) -> Self {
        let (high, low) = if self.v >= other.v {
            (self, other)
        } else {
            (other, self)
        };
        if low.v == f64::NEG_INFINITY {
            return high;
        }
        high + (low - high).exp().ln_1p()
    }

    /// Φ(self), the standard normal CDF.
    pub(crate) fn ndtr(self) -> Self {
        let pdf = (-0.5 * self.v * self.v - LN_SQRT_2PI).exp();
        self.chain(ndtr(self.v), pdf, -self.v * pdf)
    }

    /// ln Φ(self): with the ratio r = φ/Φ, the derivatives are r and −r(self + r). The ratio is formed from erfcx in
    /// the lower tail, where φ and Φ both underflow.
    pub(crate) fn log_ndtr(self) -> Self {
        let y = self.v;
        let value = log_ndtr(y);
        if !Self::DIFFERENTIATED {
            return Self::constant(value);
        }
        let r = if y < -1.0 {
            // Φ(y) = ½ erfcx(−y/√2) e^{−y²/2}, so φ/Φ = √(2/π) / erfcx(−y/√2)
            (2.0 / PI).sqrt() / keanu_tensor::special::erfcx(-y * FRAC_1_SQRT_2)
        } else {
            (-0.5 * y * y - LN_SQRT_2PI - value).exp()
        };
        self.chain(value, r, -r * (y + r))
    }

    /// ln I₀(self) for self >= 0: with A = I₁/I₀, the derivatives are A and 1 − A/self − A² (A/self → ½ at 0).
    pub(crate) fn log_bessel_i0(self) -> Self {
        let z = self.v;
        let value = log_bessel_i0(z);
        if !Self::DIFFERENTIATED {
            return Self::constant(value);
        }
        let a = bessel_i_ratio(0.0, z);
        // A = z/2 − z³/16 + O(z⁵)
        let a_over_z = if z < 1e-4 { 0.5 - z * z / 16.0 } else { a / z };
        self.chain(value, a, 1.0 - a_over_z - a * a)
    }

    /// ln I_ν(self) for a constant order ν >= 0 and self > 0: with A = I_{ν+1}/I_ν, the derivatives are A + ν/self
    /// and A' − ν/self², where A' = 1 − A² − (2ν + 1)A/self.
    pub(crate) fn log_bessel_i(self, nu: f64) -> Self {
        let z = self.v;
        let value = keanu_tensor::special::log_bessel_i(nu, z);
        if !Self::DIFFERENTIATED {
            return Self::constant(value);
        }
        let a = bessel_i_ratio(nu, z);
        let da = crate::real::bessel_i_ratio_dx(nu, z, a);
        self.chain(value, a + nu / z, da - nu / (z * z))
    }

    /// ln(1 + self) / self, continued smoothly through 0 (where it is 1), with its derivatives from its power
    /// series Σ (−u)ᵏ/(k + 1) near 0, where the closed forms cancel.
    pub(crate) fn log1p_over(self) -> Self {
        let u = self.v;
        if u.abs() < 0.25 {
            // L = Σ cₖ uᵏ, L' = Σ k cₖ u^{k−1} and L'' = Σ k(k − 1) cₖ u^{k−2}, with cₖ = (−1)ᵏ/(k + 1); 50
            // terms reach rounding for |u| < 1/4
            let (mut l0, mut l1, mut l2) = (0.0, 0.0, 0.0);
            // u^k, u^(k−1) and u^(k−2)
            let (mut p0, mut p1, mut p2) = (1.0, 0.0, 0.0);
            for k in 0..50 {
                let kf = k as f64;
                let c = if k % 2 == 0 { 1.0 } else { -1.0 } / (kf + 1.0);
                l0 += c * p0;
                l1 += c * kf * p1;
                l2 += c * kf * (kf - 1.0) * p2;
                p2 = if k == 0 { 1.0 } else { p1 };
                p1 = p0;
                p0 *= u;
            }
            return self.chain(l0, l1, l2);
        }
        let l = u.ln_1p();
        let d1 = 1.0 / (1.0 + u);
        let d2 = -d1 * d1;
        self.chain(
            l / u,
            d1 / u - l / (u * u),
            d2 / u - 2.0 * d1 / (u * u) + 2.0 * l / (u * u * u),
        )
    }

    /// Owen's T(h, a) at (self, a), with its derivatives T_h = −φ(h)(Φ(ah) − ½), T_a = e^{−h²(1+a²)/2}/(2π(1+a²)),
    /// T_hh = hφ(h)(Φ(ah) − ½) − aφ(h)φ(ah), T_ha = −h(1 + a²) T_a and T_aa = −a(h² + 2/(1 + a²)) T_a.
    pub(crate) fn owens_t(self, a: Self) -> Self {
        let h = self.v;
        let av = a.v;
        let value = owens_t(h, av);
        if !Self::DIFFERENTIATED {
            return Self::constant(value);
        }
        let phi = |t: f64| (-0.5 * t * t - LN_SQRT_2PI).exp();
        let centred = 0.5 * erf(av * h * FRAC_1_SQRT_2);
        let s = 1.0 + av * av;
        let t_a = (-0.5 * h * h * s).exp() / (2.0 * PI * s);
        let t_h = -phi(h) * centred;
        let t_hh = h * phi(h) * centred - av * phi(h) * phi(av * h);
        let t_ha = -h * s * t_a;
        let t_aa = -av * (h * h + 2.0 / s) * t_a;
        Self::compose(value, &[self, a], [t_h, t_a], [[t_hh, t_ha], [t_ha, t_aa]])
    }

    /// ln I_x(a, b), the log regularised incomplete beta function, at x = self (whose complement 1 − x is given
    /// separately, computed directly). With f the Beta(a, b) density and λ the result, λ_x = f/I,
    /// λ_xx = λ_x((a − 1)/x − (b − 1)/(1 − x) − λ_x), λ_xa = λ_x(ln x − ψ(a) + ψ(a + b) − λ_a) (likewise for b), and
    /// the shape derivatives are the incomplete beta function's own.
    pub(crate) fn log_inc_beta(self, complement: f64, a: Self, b: Self) -> Self {
        let (x, av, bv) = (self.v, a.v, b.v);
        let value = log_inc_beta(av, bv, x);
        if !Self::DIFFERENTIATED {
            return Self::constant(value);
        }
        let (la, lb) = (log_inc_beta_da(av, bv, x), log_inc_beta_db(av, bv, x));
        let (laa, lab, lbb) = if Self::SECOND {
            (
                log_inc_beta_da2(av, bv, x),
                log_inc_beta_dadb(av, bv, x),
                log_inc_beta_db2(av, bv, x),
            )
        } else {
            (0.0, 0.0, 0.0)
        };
        let log_density = (av - 1.0) * x.ln() + (bv - 1.0) * complement.ln()
            - keanu_tensor::special::log_beta(av, bv);
        let lx = (log_density - value).exp();
        let both = digamma(av + bv);
        let lxx = lx * ((av - 1.0) / x - (bv - 1.0) / complement - lx);
        let lxa = lx * (x.ln() - digamma(av) + both - la);
        let lxb = lx * (complement.ln() - digamma(bv) + both - lb);
        Self::compose(
            value,
            &[self, a, b],
            [lx, la, lb],
            [[lxx, lxa, lxb], [lxa, laa, lab], [lxb, lab, lbb]],
        )
    }

    /// ln P(k, y) (or ln Q(k, y) if `upper`), the log regularised incomplete gamma function, at y = self. With
    /// p(y) = y^{k−1} e^{−y}/Γ(k), λ_y = ±p/P, λ_yy = λ_y((k − 1)/y − 1 − λ_y), λ_yk = λ_y(ln y − ψ(k) − λ_k), and
    /// the shape derivatives are the incomplete gamma function's own.
    pub(crate) fn log_gamma_p(self, k: Self, upper: bool) -> Self {
        let (y, kv) = (self.v, k.v);
        let value = if upper {
            log_gamma_q(kv, y)
        } else {
            log_gamma_p(kv, y)
        };
        if !Self::DIFFERENTIATED {
            return Self::constant(value);
        }
        let (lk, lkk) = match (upper, Self::SECOND) {
            (true, true) => (log_gamma_q_da(kv, y), log_gamma_q_da2(kv, y)),
            (false, true) => (log_gamma_p_da(kv, y), log_gamma_p_da2(kv, y)),
            (true, false) => (log_gamma_q_da(kv, y), 0.0),
            (false, false) => (log_gamma_p_da(kv, y), 0.0),
        };
        let log_density = (kv - 1.0) * y.ln() - y - keanu_tensor::special::ln_gamma(kv);
        let sign = if upper { -1.0 } else { 1.0 };
        let ly = sign * (log_density - value).exp();
        let lyy = ly * ((kv - 1.0) / y - 1.0 - ly);
        let lyk = ly * (y.ln() - digamma(kv) - lk);
        Self::compose(value, &[self, k], [ly, lk], [[lyy, lyk], [lyk, lkk]])
    }
}

/// A remembered argument of [`memo_ln_gamma`] and what was computed there.
#[derive(Clone, Copy)]
struct LnGammaEntry {
    x: f64,
    /// lnΓ(x), ψ(x) and ψ'(x).
    values: [f64; 3],
    /// Whether ψ(x) and ψ'(x) were computed.
    computed: [bool; 2],
}

thread_local! {
    /// The last two arguments of [`memo_ln_gamma`], most recent first.
    static LN_GAMMA_MEMO: std::cell::Cell<[Option<LnGammaEntry>; 2]> =
        const { std::cell::Cell::new([None; 2]) };
}

/// lnΓ(x), with ψ(x) if `first` and ψ'(x) if `second` (else 0). A family whose shape parameter is broadcast over
/// many values (the degrees of freedom of a Student-t likelihood, say) asks for the same few arguments for every
/// element, so the results at the last two arguments are remembered.
fn memo_ln_gamma(x: f64, first: bool, second: bool) -> [f64; 3] {
    LN_GAMMA_MEMO.with(|memo| {
        let mut entries = memo.get();
        for slot in 0..2 {
            if let Some(entry) = entries[slot]
                && entry.x == x
                && (entry.computed[0] || !first)
                && (entry.computed[1] || !second)
            {
                if slot == 1 {
                    entries.swap(0, 1);
                    memo.set(entries);
                }
                let [value, d1, d2] = entry.values;
                return [
                    value,
                    if first { d1 } else { 0.0 },
                    if second { d2 } else { 0.0 },
                ];
            }
        }
        let values = [
            keanu_tensor::special::ln_gamma(x),
            if first { digamma(x) } else { 0.0 },
            if second { trigamma(x) } else { 0.0 },
        ];
        entries[1] = entries[0];
        entries[0] = Some(LnGammaEntry {
            x,
            values,
            computed: [first, second],
        });
        memo.set(entries);
        values
    })
}

/// 1 / (1 + e^−m), without overflow.
fn sigmoid(m: f64) -> f64 {
    if m >= 0.0 {
        1.0 / (1.0 + (-m).exp())
    } else {
        let e = m.exp();
        e / (1.0 + e)
    }
}

impl<const K: usize, const H: usize> Add for Jet<K, H> {
    type Output = Self;
    fn add(mut self, o: Self) -> Self {
        self.v += o.v;
        for i in 0..K {
            self.g[i] += o.g[i];
        }
        for i in 0..H {
            for j in 0..H {
                self.h[i][j] += o.h[i][j];
            }
        }
        self
    }
}

impl<const K: usize, const H: usize> Add<f64> for Jet<K, H> {
    type Output = Self;
    fn add(mut self, o: f64) -> Self {
        self.v += o;
        self
    }
}

impl<const K: usize, const H: usize> Add<Jet<K, H>> for f64 {
    type Output = Jet<K, H>;
    fn add(self, o: Jet<K, H>) -> Jet<K, H> {
        o + self
    }
}

impl<const K: usize, const H: usize> Neg for Jet<K, H> {
    type Output = Self;
    fn neg(self) -> Self {
        self * -1.0
    }
}

impl<const K: usize, const H: usize> Sub for Jet<K, H> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        self + -o
    }
}

impl<const K: usize, const H: usize> Sub<f64> for Jet<K, H> {
    type Output = Self;
    fn sub(mut self, o: f64) -> Self {
        self.v -= o;
        self
    }
}

impl<const K: usize, const H: usize> Sub<Jet<K, H>> for f64 {
    type Output = Jet<K, H>;
    fn sub(self, o: Jet<K, H>) -> Jet<K, H> {
        let mut r = -o;
        r.v = self - o.v;
        r
    }
}

impl<const K: usize, const H: usize> Mul for Jet<K, H> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        let mut r = Self::constant(self.v * o.v);
        for i in 0..K {
            r.g[i] = self.v * o.g[i] + o.v * self.g[i];
        }
        for i in 0..H {
            for j in 0..H {
                r.h[i][j] = self.v * o.h[i][j]
                    + o.v * self.h[i][j]
                    + self.g[i] * o.g[j]
                    + o.g[i] * self.g[j];
            }
        }
        r
    }
}

impl<const K: usize, const H: usize> Mul<f64> for Jet<K, H> {
    type Output = Self;
    fn mul(mut self, o: f64) -> Self {
        self.v *= o;
        for i in 0..K {
            self.g[i] *= o;
        }
        for i in 0..H {
            for j in 0..H {
                self.h[i][j] *= o;
            }
        }
        self
    }
}

impl<const K: usize, const H: usize> Mul<Jet<K, H>> for f64 {
    type Output = Jet<K, H>;
    fn mul(self, o: Jet<K, H>) -> Jet<K, H> {
        o * self
    }
}

impl<const K: usize, const H: usize> Div for Jet<K, H> {
    type Output = Self;
    /// q = a / b from a = q b: q' = (a' − q b') / b and q'' = (a'' − q b'' − q' b'ᵀ − b' q'ᵀ) / b.
    fn div(self, o: Self) -> Self {
        let q = self.v / o.v;
        let mut r = Self::constant(q);
        for i in 0..K {
            r.g[i] = (self.g[i] - q * o.g[i]) / o.v;
        }
        for i in 0..H {
            for j in 0..H {
                r.h[i][j] =
                    (self.h[i][j] - q * o.h[i][j] - r.g[i] * o.g[j] - o.g[i] * r.g[j]) / o.v;
            }
        }
        r
    }
}

impl<const K: usize, const H: usize> Div<f64> for Jet<K, H> {
    type Output = Self;
    fn div(self, o: f64) -> Self {
        self * (1.0 / o)
    }
}

impl<const K: usize, const H: usize> Div<Jet<K, H>> for f64 {
    type Output = Jet<K, H>;
    fn div(self, o: Jet<K, H>) -> Jet<K, H> {
        o.chain(
            self / o.v,
            -self / (o.v * o.v),
            2.0 * self / (o.v * o.v * o.v),
        )
    }
}
