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

//! Updating the interpolation set, the inverse of the KKT matrix and the quadratic model
//! (PRIMA's `update.f90` for BOBYQA).

use super::common::{calbeta, calvlag, hess_mul, planerot, rotate_cols};
use super::linalg::{Mat, Vector, dot, fmax, fmin, sum_abs};

/// HQ += ALPHA * X * Xᵀ, computed on the lower triangle and mirrored so HQ stays exactly symmetric
/// (PRIMA's `r1update`).
pub(crate) fn r1update(hq: &mut Mat, alpha: f64, x: &[f64]) {
    let n = x.len();
    for j in 0..n {
        for i in j..n {
            hq[(i, j)] += alpha * x[i] * x[j];
        }
    }
    hq.symmetrize_block(0);
}

/// HQ += ALPHA * (X * Yᵀ + Y * Xᵀ) on the lower triangle, mirrored (PRIMA's `r2update`).
pub(crate) fn r2update(hq: &mut Mat, alpha: f64, x: &[f64], y: &[f64]) {
    let n = x.len();
    for j in 0..n {
        for i in j..n {
            hq[(i, j)] += alpha * x[i] * y[j] + alpha * y[i] * x[j];
        }
    }
    hq.symmetrize_block(0);
}

/// The KNEW-th column of H (the inverse of the KKT matrix): [ZMAT * ZMAT(KNEW, :)ᵀ; BMAT(:, KNEW)].
fn hcol(knew: usize, bmat: &Mat, zmat: &Mat) -> Vector {
    let mut h = zmat.mul_vec(&zmat.row(knew));
    h.extend_from_slice(bmat.col(knew));
    h
}

/// Applies the rank-two update to BMAT with the given VLAG and HCOL, then symmetrises its last n columns.
pub(crate) fn update_bmat(
    bmat: &mut Mat,
    npt: usize,
    alpha: f64,
    beta: f64,
    tau: f64,
    denom: f64,
    vlag: &[f64],
    hcol: &[f64],
) {
    let n = bmat.rows;
    let v1: Vec<f64> = (0..n)
        .map(|i| (alpha * vlag[npt + i] - tau * hcol[npt + i]) / denom)
        .collect();
    let v2: Vec<f64> = (0..n)
        .map(|i| (-beta * hcol[npt + i] - tau * vlag[npt + i]) / denom)
        .collect();
    for j in 0..npt + n {
        for i in 0..n {
            bmat[(i, j)] += v1[i] * vlag[j] + v2[i] * hcol[j];
        }
    }
    bmat.symmetrize_block(npt);
}

/// Zeroes ZMAT(KNEW, 2:) by Givens rotations into the first column (PRIMA's loop in `updateh`).
pub(crate) fn rotate_zmat_row(zmat: &mut Mat, knew: usize) {
    let threshold = 1.0e-20 * zmat.max_abs();
    for j in 1..zmat.cols {
        if zmat[(knew, j)].abs() > threshold {
            let g = planerot([zmat[(knew, 0)], zmat[(knew, j)]]);
            rotate_cols(zmat, 0, j, g);
        }
        zmat[(knew, j)] = 0.0;
    }
}

/// PRIMA's `updateh`: updates BMAT and ZMAT when XPT(:, KNEW) is replaced by XOPT + D. Returns false (leaving
/// them unchanged) if the update would be damaged by rounding.
pub(crate) fn updateh(
    knew: usize,
    kopt: usize,
    d: &[f64],
    xpt: &Mat,
    bmat: &mut Mat,
    zmat: &mut Mat,
) -> bool {
    let npt = xpt.cols;
    let h = hcol(knew, bmat, zmat);
    let beta = calbeta(kopt, bmat, d, xpt, zmat);
    let mut vlag = calvlag(kopt, bmat, d, xpt, zmat);
    let alpha = h[knew];
    let tau = vlag[knew];
    let denom = alpha * beta + tau * tau;
    vlag[knew] -= 1.0;
    if !((sum_abs(&h) + sum_abs(&vlag) + beta.abs()).is_finite() && denom > 0.0) {
        return false;
    }
    update_bmat(bmat, npt, alpha, beta, tau, denom, &vlag, &h);
    rotate_zmat_row(zmat, knew);
    let sqrtdn = denom.sqrt();
    let zk1 = zmat[(knew, 0)];
    for k in 0..npt {
        zmat[(k, 0)] = (tau / sqrtdn) * zmat[(k, 0)] - (zk1 / sqrtdn) * vlag[k];
    }
    true
}

/// PRIMA's `updatexf`: replaces XPT(:, KNEW) and FVAL(KNEW), updating KOPT if the new point is better.
pub(crate) fn updatexf(
    knew: usize,
    ximproved: bool,
    f: f64,
    xnew: &[f64],
    kopt: &mut usize,
    fval: &mut [f64],
    xpt: &mut Mat,
) {
    xpt.set_col(knew, xnew);
    fval[knew] = f;
    if ximproved {
        *kopt = knew;
    }
}

/// PRIMA's `updateq`: updates the quadratic model after XPT(:, KNEW) changed from XDROP to XOSAV + D.
#[allow(clippy::too_many_arguments)]
pub(crate) fn updateq(
    knew: usize,
    ximproved: bool,
    bmat: &Mat,
    d: &[f64],
    moderr: f64,
    xdrop: &[f64],
    xosav: &[f64],
    xpt: &Mat,
    zmat: &Mat,
    gopt: &mut [f64],
    hq: &mut Mat,
    pq: &mut [f64],
) {
    let n = gopt.len();
    r1update(hq, pq[knew], xdrop);
    pq[knew] = 0.0;
    let pqinc: Vec<f64> = zmat
        .mul_vec(&zmat.row(knew))
        .iter()
        .map(|v| moderr * v)
        .collect();
    for (p, inc) in pq.iter_mut().zip(&pqinc) {
        *p += inc;
    }
    let hx = hess_mul(xosav, xpt, &pqinc, None);
    for i in 0..n {
        gopt[i] += moderr * bmat[(i, knew)] + hx[i];
    }
    if ximproved {
        let hd = hess_mul(d, xpt, pq, Some(hq));
        for i in 0..n {
            gopt[i] += hd[i];
        }
    }
}

/// PRIMA's `tryqalt`: replaces the model by the alternative (least Frobenius norm) model when the latter has had
/// a much smaller projected gradient for three consecutive iterations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tryqalt(
    bmat: &Mat,
    fval: &[f64],
    ratio: f64,
    sl: &[f64],
    su: &[f64],
    xopt: &[f64],
    xpt: &Mat,
    zmat: &Mat,
    itest: &mut usize,
    gopt: &mut Vec<f64>,
    hq: &mut Mat,
    pq: &mut Vec<f64>,
) {
    let n = gopt.len();
    let npt = pq.len();
    let project = |g: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| {
                let mut v = g[i];
                if xopt[i] >= su[i] {
                    v = fmax(0.0, v);
                }
                if xopt[i] <= sl[i] {
                    v = fmin(0.0, v);
                }
                v
            })
            .collect()
    };
    let pgopt = project(gopt);
    let zf = zmat.tmul_vec(fval);
    let pqalt = zmat.mul_vec(&zf);
    let bf = bmat.mul_vec_cols(npt, fval);
    let hx = hess_mul(xopt, xpt, &pqalt, None);
    let galt: Vec<f64> = (0..n).map(|i| bf[i] + hx[i]).collect();
    let pgalt = project(&galt);
    if ratio > 0.1 || dot(&pgopt, &pgopt) < 10.0 * dot(&pgalt, &pgalt) {
        *itest = 0;
    } else {
        *itest += 1;
    }
    if *itest >= 3 {
        *gopt = galt;
        *pq = pqalt.into_vec();
        hq.fill(0.0);
        *itest = 0;
    }
}
