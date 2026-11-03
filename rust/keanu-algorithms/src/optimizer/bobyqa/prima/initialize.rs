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

//! Initialisation of the interpolation set, the quadratic model and the inverse of the KKT matrix
//! (PRIMA's `initialize.f90` for BOBYQA).

use super::common::{ExitStatus, REALMAX, checkexit, evaluate, setij, xinbd};
use super::linalg::{Mat, fmax, fmin};

/// The interpolation set and the function values on it.
pub(crate) struct InitialPoints {
    pub ij: Vec<(usize, usize)>,
    pub kopt: usize,
    pub nf: usize,
    pub fval: Vec<f64>,
    pub sl: Vec<f64>,
    pub su: Vec<f64>,
    pub xbase: Vec<f64>,
    pub xpt: Mat,
    pub info: Option<ExitStatus>,
    /// The best point evaluated, in case the solver exits during initialisation.
    pub best_x: Vec<f64>,
}

/// PRIMA's `initxf`: sets up and evaluates the initial interpolation points.
#[allow(clippy::too_many_arguments)]
pub(crate) fn initxf(
    calfun: &mut dyn FnMut(&[f64]) -> f64,
    maxfun: usize,
    npt: usize,
    ftarget: f64,
    rhobeg: f64,
    xl: &[f64],
    xu: &[f64],
    x0_in: &[f64],
) -> InitialPoints {
    let n = x0_in.len();
    let mut x0 = x0_in.to_vec();
    let mut sl: Vec<f64> = (0..n).map(|i| xl[i] - x0[i]).collect();
    let mut su: Vec<f64> = (0..n).map(|i| xu[i] - x0[i]).collect();
    for i in 0..n {
        if sl[i] < 0.0 {
            sl[i] = fmin(sl[i], -rhobeg);
        } else {
            x0[i] = xl[i];
            sl[i] = 0.0;
            su[i] = xu[i] - xl[i];
        }
    }
    for i in 0..n {
        if su[i] > 0.0 {
            su[i] = fmax(su[i], rhobeg);
        } else {
            x0[i] = xu[i];
            sl[i] = xl[i] - xu[i];
            su[i] = 0.0;
        }
    }
    let xbase = x0;
    let mut evaluated = vec![false; npt];
    let mut fval = vec![REALMAX; npt];
    let mut xpt = Mat::zeros(n, npt);
    for k in 0..n {
        xpt[(k, k + 1)] = rhobeg;
        if su[k] <= 0.0 {
            xpt[(k, k + 1)] = -rhobeg;
        }
    }
    for k in 0..(npt - n - 1).min(n) {
        xpt[(k, k + n + 1)] = -rhobeg;
        if sl[k] >= 0.0 {
            xpt[(k, k + n + 1)] = fmin(2.0 * rhobeg, su[k]);
        }
        if su[k] <= 0.0 {
            xpt[(k, k + n + 1)] = fmax(-2.0 * rhobeg, sl[k]);
        }
    }

    let mut info = None;
    for k in 0..npt.min(2 * n + 1) {
        let x = xinbd(&xbase, xpt.col(k), xl, xu, &sl, &su);
        let f = evaluate(calfun, &x);
        evaluated[k] = true;
        fval[k] = f;
        if let Some(exit) = checkexit(maxfun, k + 1, f, ftarget, &x) {
            info = Some(exit);
            break;
        }
    }

    // When both points along a coordinate direction straddle XBASE, put the better one first.
    for k1 in 2..=(npt - n).min(n + 1) {
        let k = k1 - 1;
        if xpt[(k1 - 2, k)] * xpt[(k1 - 2, k + n)] < 0.0 && fval[k + n] < fval[k] {
            fval.swap(k, k + n);
            xpt.swap_cols(k, k + n);
        }
    }

    let ij = setij(n, npt);
    for (m, &(i, j)) in ij.iter().enumerate() {
        let col: Vec<f64> = (0..n).map(|r| xpt[(r, i + 1)] + xpt[(r, j + 1)]).collect();
        xpt.set_col(2 * n + 1 + m, &col);
    }

    if info.is_none() {
        for k in 2 * n + 1..npt {
            let x = xinbd(&xbase, xpt.col(k), xl, xu, &sl, &su);
            let f = evaluate(calfun, &x);
            evaluated[k] = true;
            fval[k] = f;
            if let Some(exit) = checkexit(maxfun, k + 1, f, ftarget, &x) {
                info = Some(exit);
                break;
            }
        }
    }

    let nf = evaluated.iter().filter(|&&e| e).count();
    let mut kopt = 0;
    for k in 0..npt {
        if evaluated[k] && (!evaluated[kopt] || fval[k] < fval[kopt]) {
            kopt = k;
        }
    }
    let best_x = xinbd(&xbase, xpt.col(kopt), xl, xu, &sl, &su);
    InitialPoints {
        ij,
        kopt,
        nf,
        fval,
        sl,
        su,
        xbase,
        xpt,
        info,
        best_x,
    }
}

/// PRIMA's `initq`: the initial quadratic model (gradient at XOPT, explicit Hessian, zero implicit Hessian).
pub(crate) fn initq(ij: &[(usize, usize)], fval: &[f64], xpt: &Mat) -> (Vec<f64>, Mat, Vec<f64>) {
    let n = xpt.rows;
    let npt = xpt.cols;
    let fbase = fval[0];
    let mut gopt: Vec<f64> = (0..n)
        .map(|i| (fval[i + 1] - fbase) / xpt[(i, i + 1)])
        .collect();
    let ndiag = n.min(npt - n - 1);
    let xa: Vec<f64> = (0..ndiag).map(|i| xpt[(i, i + 1)]).collect();
    let xb: Vec<f64> = (0..ndiag).map(|i| xpt[(i, n + 1 + i)]).collect();
    for i in 0..ndiag {
        gopt[i] = (gopt[i] * xb[i] - ((fval[n + 1 + i] - fbase) / xb[i]) * xa[i]) / (xb[i] - xa[i]);
    }
    let mut hq = Mat::zeros(n, n);
    for k in 0..ndiag {
        hq[(k, k)] = 2.0 * ((fval[k + 1] - fbase) / xa[k] - (fval[n + k + 1] - fbase) / xb[k])
            / (xa[k] - xb[k]);
    }
    for (k, &(i, j)) in ij.iter().enumerate() {
        let col = k + 2 * n + 1;
        let xi = xpt[(i, col)];
        let xj = xpt[(j, col)];
        hq[(i, j)] = (fbase - fval[i + 1] - fval[j + 1] + fval[col]) / (xi * xj);
        hq[(j, i)] = hq[(i, j)];
    }
    let kopt = super::linalg::minloc(fval);
    if kopt != 0 {
        let hx = hq.mul_vec(xpt.col(kopt));
        for i in 0..n {
            gopt[i] += hx[i];
        }
    }
    (gopt, hq, vec![0.0; npt])
}

/// PRIMA's `inith`: the initial BMAT and ZMAT of the inverse of the KKT matrix.
pub(crate) fn inith(ij: &[(usize, usize)], xpt: &Mat) -> (Mat, Mat) {
    let n = xpt.rows;
    let npt = xpt.cols;
    let rhobeg = xpt.col(1).iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let rhosq = rhobeg * rhobeg;
    let ndiag = n.min(npt - n - 1);
    let xa: Vec<f64> = (0..ndiag).map(|i| xpt[(i, i + 1)]).collect();
    let xb: Vec<f64> = (0..ndiag).map(|i| xpt[(i, n + 1 + i)]).collect();

    let mut bmat = Mat::zeros(n, npt + n);
    for i in 0..ndiag {
        bmat[(i, 0)] = -(xa[i] + xb[i]) / (xa[i] * xb[i]);
    }
    for k in 0..ndiag {
        bmat[(k, k + n + 1)] = -0.5 / xpt[(k, k + 1)];
        bmat[(k, k + 1)] = -bmat[(k, 0)] - bmat[(k, k + n + 1)];
    }
    for k in ndiag..n {
        bmat[(k, 0)] = -1.0 / xpt[(k, k + 1)];
        bmat[(k, k + 1)] = -bmat[(k, 0)];
        bmat[(k, npt + k)] = -0.5 * rhosq;
    }

    let mut zmat = Mat::zeros(npt, npt - n - 1);
    for k in 0..ndiag {
        zmat[(0, k)] = 2f64.sqrt() / (xa[k] * xb[k]);
    }
    for k in 0..ndiag {
        zmat[(k + 1, k)] = -zmat[(0, k)] - 0.5f64.sqrt() / rhosq;
        zmat[(k + n + 1, k)] = 0.5f64.sqrt() / rhosq;
    }
    for k in ndiag..npt - n - 1 {
        zmat[(0, k)] = 1.0 / rhosq;
        zmat[(k + n + 1, k)] = 1.0 / rhosq;
        let (i, j) = ij[k - n];
        zmat[(i + 1, k)] = -1.0 / rhosq;
        zmat[(j + 1, k)] = -1.0 / rhosq;
    }
    (bmat, zmat)
}
