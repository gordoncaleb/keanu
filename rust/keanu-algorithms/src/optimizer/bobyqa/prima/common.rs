// This file is a Rust translation of part of PRIMA's BOBYQA (https://github.com/libprima/prima, revision
// 43863c69ccd6408dedcd782d0dfdcbaad404dcde) and is distributed under PRIMA's BSD 3-Clause licence:
//
// Copyright (c) 2020--2026, Zaikun ZHANG ( https://www.zhangzk.net )
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
//    following disclaimer in the documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
//    products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// See THIRD-PARTY-NOTICES.md in this crate for the changes made in the translation.

//! Constants, exit conditions and the shared routines of PRIMA's `common` directory that BOBYQA uses
//! (`checkexit`, `evaluate`, `xinbd`, `redrho`, `redrat`, `interval_max`, `planerot` and the `powalg` and
//! `shiftbase` routines for the quadratic model and the inverse of the KKT matrix).

use super::linalg::{Mat, dot, fmax, fmin, maxloc, norm, sign, sum_sq};

pub(crate) const EPS: f64 = f64::EPSILON;
pub(crate) const REALMAX: f64 = f64::MAX;
pub(crate) const REALMIN: f64 = f64::MIN_POSITIVE;
/// Any bound with an absolute value at least this is treated as no bound.
pub(crate) const BOUNDMAX: f64 = 0.25 * REALMAX;
/// Function values are capped at this (and NaN is replaced by it).
pub(crate) const FUNCMAX: f64 = 1.0e30;
pub(crate) const RHOBEG_DFT: f64 = 1.0;
pub(crate) const RHOEND_DFT: f64 = 1.0e-6;
pub(crate) const MAXFUN_DIM_DFT: usize = 500;

/// Why the solver stopped (PRIMA's `infos_mod`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitStatus {
    /// The trust region radius reached its final value: the normal, converged exit.
    SmallTrustRegionRadius,
    /// The target function value was reached.
    TargetReached,
    /// The maximum number of function evaluations was reached.
    MaxEvaluationsReached,
    /// The maximum number of trust region iterations was reached.
    MaxIterationsReached,
    /// A point with NaN or infinite coordinates was produced.
    NanInfX,
    /// The objective returned NaN or +infinity.
    NanInfF,
    /// The quadratic model became NaN or infinite.
    NanInfModel,
    /// Rounding errors became damaging.
    DamagingRounding,
    /// A lower bound is not (sufficiently) below its upper bound.
    NoSpaceBetweenBounds,
}

/// PRIMA's `checkexit` for unconstrained problems: None means continue.
pub(crate) fn checkexit(
    maxfun: usize,
    nf: usize,
    f: f64,
    ftarget: f64,
    x: &[f64],
) -> Option<ExitStatus> {
    let mut info = None;
    if x.iter().any(|v| !v.is_finite()) {
        info = Some(ExitStatus::NanInfX);
    }
    if f.is_nan() || f == f64::INFINITY {
        info = Some(ExitStatus::NanInfF);
    }
    if f <= ftarget {
        info = Some(ExitStatus::TargetReached);
    }
    if nf >= maxfun {
        info = Some(ExitStatus::MaxEvaluationsReached);
    }
    info
}

/// Replaces NaN by 0 and caps the magnitude at REALMAX.
pub(crate) fn moderatex(x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&v| {
            if v.is_nan() {
                0.0
            } else {
                v.clamp(-REALMAX, REALMAX)
            }
        })
        .collect()
}

fn moderatef(f: f64) -> f64 {
    let f = if f.is_nan() { FUNCMAX } else { f };
    f.clamp(-REALMAX, FUNCMAX)
}

/// Evaluates the objective with PRIMA's safeguards: NaN coordinates give a NaN value without calling the
/// objective, and the returned value is moderated (NaN becomes FUNCMAX, values are capped at FUNCMAX).
pub(crate) fn evaluate(calfun: &mut dyn FnMut(&[f64]) -> f64, x: &[f64]) -> f64 {
    if x.iter().any(|v| v.is_nan()) {
        x.iter().sum()
    } else {
        moderatef(calfun(&moderatex(x)))
    }
}

/// XBASE + STEP projected into the bounds, snapping to a bound exactly when STEP reaches SL or SU.
pub(crate) fn xinbd(
    xbase: &[f64],
    step: &[f64],
    xl: &[f64],
    xu: &[f64],
    sl: &[f64],
    su: &[f64],
) -> Vec<f64> {
    (0..xbase.len())
        .map(|i| {
            let s = fmax(sl[i], fmin(su[i], step[i]));
            let mut x = fmax(xl[i], fmin(xu[i], xbase[i] + s));
            if s <= sl[i] {
                x = xl[i];
            }
            if s >= su[i] {
                x = xu[i];
            }
            x
        })
        .collect()
}

/// The next value of the lower bound of the trust region radius.
pub(crate) fn redrho(rho_in: f64, rhoend: f64) -> f64 {
    let rho_ratio = rho_in / rhoend;
    if rho_ratio > 250.0 {
        0.1 * rho_in
    } else if rho_ratio <= 16.0 {
        rhoend
    } else {
        rho_ratio.sqrt() * rhoend
    }
}

/// The reduction ratio ARED / PRED with PRIMA's safeguards.
pub(crate) fn redrat(ared: f64, pred: f64, rshrink: f64) -> f64 {
    if ared.is_nan() {
        -REALMAX
    } else if pred.is_nan() || pred <= 0.0 {
        if ared > 0.0 { 0.5 * rshrink } else { -REALMAX }
    } else if pred == f64::INFINITY && ared == f64::INFINITY {
        1.0
    } else if pred == f64::INFINITY && ared == f64::NEG_INFINITY {
        -REALMAX
    } else {
        ared / pred
    }
}

/// PRIMA's `linspace`.
fn linspace(xstart: f64, xstop: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let nm = n as i64 - 1;
    let mut x: Vec<f64> = if n == 1 || xstart == xstop {
        vec![xstop; n]
    } else if xstart.abs() == xstop.abs() {
        let xunit = xstop / nm as f64;
        let mut v: Vec<f64> = (0..n as i64)
            .map(|k| xunit * (-nm + 2 * k) as f64)
            .collect();
        if nm % 2 == 0 {
            v[(nm / 2) as usize] = 0.0;
        }
        v
    } else {
        let xunit = (xstop - xstart) / nm as f64;
        (0..n).map(|k| xstart + xunit * k as f64).collect()
    };
    x[0] = xstart;
    x[n - 1] = xstop;
    x
}

/// Approximately maximises `fun` on [lb, ub] by evaluating it on a grid and refining the best point with a
/// parabola through it and its neighbours.
pub(crate) fn interval_max(fun: impl Fn(f64) -> f64, lb: f64, ub: f64, grid_size: usize) -> f64 {
    if ub <= lb {
        return lb;
    }
    let xgrid = linspace(lb, ub, grid_size);
    let fgrid: Vec<f64> = xgrid.iter().map(|&x| fun(x)).collect();
    if fgrid.iter().all(|f| f.is_nan()) {
        return lb;
    }
    let kopt = maxloc(&fgrid);
    let fopt = fgrid[kopt];
    if kopt == 0 {
        lb
    } else if kopt == grid_size - 1 {
        ub
    } else {
        let fprev = fgrid[kopt - 1];
        let fnext = fgrid[kopt + 1];
        let mut step = 0.0;
        if (fprev - fnext).abs() > 0.0 {
            step = 0.5 * ((fnext - fprev) / (fopt + fopt - fprev - fnext));
        }
        if step.is_finite() && step.abs() > 0.0 {
            lb + (ub - lb) * (kopt as f64 + step) / (grid_size - 1) as f64
        } else {
            xgrid[kopt]
        }
    }
}

/// The Givens rotation G = [[c, s], [-s, c]] with G * x = [r, 0].
pub(crate) fn planerot(x: [f64; 2]) -> [[f64; 2]; 2] {
    let (c, s);
    if x[0].is_nan() || x[1].is_nan() {
        c = 1.0;
        s = 0.0;
    } else if x[0].is_infinite() && x[1].is_infinite() {
        c = sign(1.0 / 2f64.sqrt(), x[0]);
        s = sign(1.0 / 2f64.sqrt(), x[1]);
    } else if x[0].abs() <= 0.0 && x[1].abs() <= 0.0 {
        c = 1.0;
        s = 0.0;
    } else if x[1].abs() <= EPS * x[0].abs() {
        c = sign(1.0, x[0]);
        s = 0.0;
    } else if x[0].abs() <= EPS * x[1].abs() {
        c = 0.0;
        s = sign(1.0, x[1]);
    } else if x
        .iter()
        .all(|v| v.abs() > REALMIN.sqrt() && v.abs() < (REALMAX / 2.1).sqrt())
    {
        let r = norm(&x);
        c = x[0] / r;
        s = x[1] / r;
    } else if x[0].abs() > x[1].abs() {
        let t = x[1] / x[0];
        let u = sign(1f64.max(t.abs()).max((1.0 + t * t).sqrt()), x[0]);
        c = 1.0 / u;
        s = t / u;
    } else {
        let t = x[0] / x[1];
        let u = sign(1f64.max(t.abs()).max((1.0 + t * t).sqrt()), x[1]);
        c = t / u;
        s = 1.0 / u;
    }
    [[c, s], [-s, c]]
}

/// Applies the rotation to columns `a` and `b` of `m`: [m(:, a), m(:, b)] = [m(:, a), m(:, b)] * Gᵀ.
pub(crate) fn rotate_cols(m: &mut Mat, a: usize, b: usize, g: [[f64; 2]; 2]) {
    for i in 0..m.rows {
        let (u, v) = (m[(i, a)], m[(i, b)]);
        m[(i, a)] = g[0][0] * u + g[0][1] * v;
        m[(i, b)] = g[1][0] * u + g[1][1] * v;
    }
}

/// XPT * (PQ .* (XPTᵀ * X)) + HQ * X: the product of the model's Hessian with X.
pub(crate) fn hess_mul(x: &[f64], xpt: &Mat, pq: &[f64], hq: Option<&Mat>) -> Vec<f64> {
    let w: Vec<f64> = xpt.tmul_vec(x).iter().zip(pq).map(|(a, b)| a * b).collect();
    let mut y = xpt.mul_vec(&w);
    if let Some(hq) = hq {
        for (j, &xj) in x.iter().enumerate() {
            for (i, yi) in y.iter_mut().enumerate() {
                *yi += hq[(i, j)] * xj;
            }
        }
    }
    y
}

/// Q(XOPT + D) - Q(XOPT) for the quadratic model with gradient GQ at XOPT and Hessian HQ + XPT diag(PQ) XPTᵀ.
pub(crate) fn quadinc(d: &[f64], xpt: &Mat, gq: &[f64], pq: &[f64], hq: &Mat) -> f64 {
    let dxpt = xpt.tmul_vec(d);
    let hd = hq.mul_vec(d);
    let first: f64 = d
        .iter()
        .zip(gq.iter().zip(&hd))
        .map(|(di, (g, h))| di * (g + 0.5 * h))
        .sum();
    first + 0.5 * dxpt.iter().zip(pq).map(|(x, p)| x * p * x).sum::<f64>()
}

fn wcheck(kref: usize, d: &[f64], xpt: &Mat) -> Vec<f64> {
    let xref = xpt.col(kref);
    let dx = xpt.tmul_vec(d);
    let rx = xpt.tmul_vec(xref);
    dx.iter().zip(&rx).map(|(w, r)| w * (0.5 * w + r)).collect()
}

/// Columns [0, npt) of the product H * [WCHECK; D] without the +1 at KREF.
fn vlag_raw(bmat: &Mat, d: &[f64], zmat: &Mat, w: &[f64]) -> Vec<f64> {
    let npt = zmat.rows;
    let n = bmat.rows;
    let zw = zmat.tmul_vec(w);
    let mut vlag = zmat.mul_vec(&zw);
    let bd = bmat.tmul_vec_cols(npt, d);
    for k in 0..npt {
        vlag[k] += bd[k];
    }
    let wd: Vec<f64> = w.iter().chain(d.iter()).copied().collect();
    vlag.extend(bmat.mul_vec(&wd));
    debug_assert_eq!(vlag.len(), npt + n);
    vlag
}

/// The values of the Lagrange functions at XPT(:, KREF) + D (PRIMA's `calvlag`).
pub(crate) fn calvlag(kref: usize, bmat: &Mat, d: &[f64], xpt: &Mat, zmat: &Mat) -> Vec<f64> {
    let w = wcheck(kref, d, xpt);
    let mut vlag = vlag_raw(bmat, d, zmat, &w);
    vlag[kref] += 1.0;
    vlag
}

/// The BETA of the denominator of the updating formula (PRIMA's `calbeta`).
pub(crate) fn calbeta(kref: usize, bmat: &Mat, d: &[f64], xpt: &Mat, zmat: &Mat) -> f64 {
    let npt = zmat.rows;
    let w = wcheck(kref, d, xpt);
    let vlag = vlag_raw(bmat, d, zmat, &w);
    let xref = xpt.col(kref);
    let dxref = dot(d, xref);
    let dsq = dot(d, d);
    let xrefsq = dot(xref, xref);
    let dvlag = dot(d, &vlag[npt..]);
    let wvlag = dot(&w, &vlag[..npt]);
    dxref * dxref + dsq * (xrefsq + dxref + dxref + 0.5 * dsq) - dvlag - wvlag
}

/// The denominators of the updating formula for replacing each interpolation point (PRIMA's `calden`).
pub(crate) fn calden(kref: usize, bmat: &Mat, d: &[f64], xpt: &Mat, zmat: &Mat) -> Vec<f64> {
    let npt = zmat.rows;
    let vlag = calvlag(kref, bmat, d, xpt, zmat);
    let beta = calbeta(kref, bmat, d, xpt, zmat);
    (0..npt)
        .map(|k| sum_sq(&zmat.row(k)) * beta + vlag[k] * vlag[k])
        .collect()
}

/// The 0-based pairs (i, j) of variables used by the interpolation points beyond the first 2n + 1
/// (PRIMA's `setij`).
pub(crate) fn setij(n: usize, npt: usize) -> Vec<(usize, usize)> {
    if npt < 2 * n + 2 {
        return Vec::new();
    }
    (n..=npt - n - 2)
        .map(|k| {
            let ell = k / n;
            let i1 = k - n * ell + 1;
            let j1 = (i1 + ell - 1) % n + 1;
            (i1 - 1, j1 - 1)
        })
        .collect()
}

/// Moves the base point XBASE to XBASE + XOPT, updating XPT, BMAT and the implicit/explicit Hessian
/// (PRIMA's `shiftbase` for the least Frobenius norm interpolation).
pub(crate) fn shiftbase(
    kopt: usize,
    xbase: &mut [f64],
    xpt: &mut Mat,
    zmat: &Mat,
    bmat: &mut Mat,
    pq: &[f64],
    hq: &mut Mat,
) {
    let n = xpt.rows;
    let npt = xpt.cols;
    let xopt = xpt.col(kopt).to_vec();
    let xoptsq = dot(&xopt, &xopt);
    let sxpt: Vec<f64> = xpt
        .tmul_vec(&xopt)
        .iter()
        .map(|v| v - 0.5 * xoptsq)
        .collect();
    let qxoptq = 0.25 * xoptsq;
    let mut ymat = Mat::zeros(n, npt);
    for k in 0..npt {
        for i in 0..n {
            ymat[(i, k)] = sxpt[k] * (xpt[(i, k)] - 0.5 * xopt[i]) + qxoptq * xopt[i];
        }
    }
    // BYMAT = BMAT(:, 1:NPT) * YMATᵀ (n × n)
    let mut bymat = Mat::zeros(n, n);
    for a in 0..n {
        for b in 0..n {
            bymat[(a, b)] = (0..npt).map(|k| bmat[(a, k)] * ymat[(b, k)]).sum();
        }
    }
    for a in 0..n {
        for b in 0..n {
            bmat[(a, npt + b)] += bymat[(a, b)] + bymat[(b, a)];
        }
    }
    // YZMAT = YMAT * ZMAT (n × (npt - n - 1))
    let nz = zmat.cols;
    let mut yzmat = Mat::zeros(n, nz);
    for a in 0..n {
        for c in 0..nz {
            yzmat[(a, c)] = (0..npt).map(|k| ymat[(a, k)] * zmat[(k, c)]).sum();
        }
    }
    for a in 0..n {
        for b in 0..n {
            bmat[(a, npt + b)] += (0..nz).map(|c| yzmat[(a, c)] * yzmat[(b, c)]).sum::<f64>();
        }
    }
    for a in 0..n {
        for k in 0..npt {
            bmat[(a, k)] += (0..nz).map(|c| yzmat[(a, c)] * zmat[(k, c)]).sum::<f64>();
        }
    }
    let sum_pq: f64 = pq.iter().sum();
    let v: Vec<f64> = xpt
        .mul_vec(pq)
        .iter()
        .zip(&xopt)
        .map(|(a, x)| a - 0.5 * sum_pq * x)
        .collect();
    for a in 0..n {
        for b in 0..n {
            hq[(a, b)] += v[a] * xopt[b] + v[b] * xopt[a];
        }
    }
    for i in 0..n {
        xbase[i] += xopt[i];
    }
    for k in 0..npt {
        for i in 0..n {
            xpt[(i, k)] -= xopt[i];
        }
    }
    for i in 0..n {
        xpt[(i, kopt)] = 0.0;
    }
}
