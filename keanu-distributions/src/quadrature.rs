//! Adaptive Gauss–Kronrod quadrature of positive integrands and their jets.

use crate::jet::Jet;

// The 15-point Kronrod rule on [-1, 1], with its embedded seven-point Gauss rule.
const NODES: [(f64, f64, f64); 8] = [
    (0.991_455_371_120_812_6, 0.022_935_322_010_529_22, 0.0),
    (
        0.949_107_912_342_758_5,
        0.063_092_092_629_978_55,
        0.129_484_966_168_869_7,
    ),
    (0.864_864_423_359_769_1, 0.104_790_010_322_250_2, 0.0),
    (
        0.741_531_185_599_394_5,
        0.140_653_259_715_525_9,
        0.279_705_391_489_276_7,
    ),
    (0.586_087_235_467_691_1, 0.169_004_726_639_267_9, 0.0),
    (
        0.405_845_151_377_397_2,
        0.190_350_578_064_785_4,
        0.381_830_050_505_118_9,
    ),
    (0.207_784_955_007_898_5, 0.204_432_940_075_298_9, 0.0),
    (0.0, 0.209_482_141_084_727_8, 0.417_959_183_673_469_4),
];

fn panel<const K: usize>(f: &impl Fn(f64) -> Jet<K>, lo: f64, hi: f64) -> (Jet<K>, f64) {
    let half = (hi - lo) / 2.0;
    let center = lo + half;
    let mut kronrod = Jet::constant(f64::NEG_INFINITY);
    let mut gauss = Jet::<0>::constant(f64::NEG_INFINITY);
    for (node, wk, wg) in NODES {
        let delta = half * node;
        let mut term = f(center + delta);
        if delta != 0.0 {
            term = term.log_add_exp(f(center - delta));
        }
        kronrod = kronrod.log_add_exp(term + wk.ln());
        if wg > 0.0 {
            gauss = gauss.log_add_exp(Jet::constant(term.v + wg.ln()));
        }
    }
    (kronrod + half.ln(), (kronrod.v - gauss.v).abs())
}

pub(crate) fn integrate_log<const K: usize>(f: &impl Fn(f64) -> Jet<K>) -> Jet<K> {
    fn refine<const K: usize>(
        f: &impl Fn(f64) -> Jet<K>,
        lo: f64,
        hi: f64,
        (value, error): (Jet<K>, f64),
        best: &mut f64,
        depth: usize,
    ) -> Jet<K> {
        if error < 2e-11 + 4.0 * f64::EPSILON * value.v.abs()
            || value.v < *best - 42.0
            || !value.v.is_finite()
            || depth == 16
        {
            return value;
        }
        let mid = (lo + hi) / 2.0;
        let left = panel(f, lo, mid);
        let right = panel(f, mid, hi);
        *best = best.max(left.0.v).max(right.0.v);
        refine(f, lo, mid, left, best, depth + 1).log_add_exp(refine(
            f,
            mid,
            hi,
            right,
            best,
            depth + 1,
        ))
    }

    let root = panel(f, 0.0, 1.0);
    let mut best = root.0.v;
    refine(f, 0.0, 1.0, root, &mut best, 0)
}

/// Integrates a signed function on [0, 1]. The caller scales the integrand to order one.
pub(crate) fn integrate_signed<const K: usize>(f: &impl Fn(f64) -> Jet<K>) -> Jet<K> {
    fn refine<const K: usize>(
        f: &impl Fn(f64) -> Jet<K>,
        lo: f64,
        hi: f64,
        depth: usize,
    ) -> Jet<K> {
        let half = (hi - lo) / 2.0;
        let center = lo + half;
        let mut kronrod = Jet::constant(0.0);
        let mut gauss = 0.0;
        for (node, wk, wg) in NODES {
            let delta = half * node;
            let pair = if delta == 0.0 {
                f(center)
            } else {
                f(center + delta) + f(center - delta)
            };
            kronrod = kronrod + wk * pair;
            gauss += wg * pair.v;
        }
        let result = half * kronrod;
        if !result.v.is_finite()
            || half * (kronrod.v - gauss).abs() < 2e-13 * (1.0 + result.v.abs())
            || depth == 16
        {
            result
        } else {
            refine(f, lo, center, depth + 1) + refine(f, center, hi, depth + 1)
        }
    }
    refine(f, 0.0, 1.0, 0)
}
