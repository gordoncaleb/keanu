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

//! Recovery from damaging rounding errors by rebuilding the interpolation set around XOPT (PRIMA's `rescue.f90`).

use super::common::{Calfun, ExitStatus, checkexit, evaluate, hess_mul, setij, xinbd};
use super::linalg::{Mat, dot, fmax, fmin, maxloc, sum_abs, sum_sq};
use super::update::{r1update, r2update, rotate_zmat_row, update_bmat};

/// Everything RESCUE may change.
pub(crate) struct RescueState<'a> {
    pub kopt: &'a mut usize,
    pub nf: &'a mut usize,
    pub fval: &'a mut [f64],
    pub gopt: &'a mut [f64],
    pub hq: &'a mut Mat,
    pub pq: &'a mut [f64],
    pub sl: &'a mut [f64],
    pub su: &'a mut [f64],
    pub xbase: &'a mut [f64],
    pub xpt: &'a mut Mat,
    pub bmat: &'a mut Mat,
    pub zmat: &'a mut Mat,
}

/// Decodes PTSID into the 0-based coordinates (IP, IQ) of a provisional point, using None for "no coordinate".
fn ptsid_coords(ptsid: f64, n: usize) -> (Option<usize>, Option<usize>) {
    let ip = ptsid.floor() as i64;
    let iq = ((n as f64 + 1.0) * ptsid - ((n as i64 + 1) * ip) as f64).floor() as i64;
    let to = |v: i64| if v > 0 { Some((v - 1) as usize) } else { None };
    (to(ip), to(iq))
}

/// PRIMA's `rescue`. Returns an exit status if the solver must stop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rescue(
    calfun: &mut Calfun<'_>,
    maxfun: usize,
    delta: f64,
    ftarget: f64,
    xl: &[f64],
    xu: &[f64],
    st: RescueState<'_>,
) -> Option<ExitStatus> {
    let n = st.xpt.rows;
    let npt = st.xpt.cols;
    if *st.nf >= maxfun {
        st.bmat.fill(0.0);
        st.zmat.fill(0.0);
        return Some(ExitStatus::MaxEvaluationsReached);
    }

    let kopt0 = *st.kopt;
    let xopt = st.xpt.col(kopt0).to_vec();
    for i in 0..n {
        st.sl[i] = fmin(st.sl[i] - xopt[i], 0.0);
        st.su[i] = fmax(st.su[i] - xopt[i], 0.0);
        st.xbase[i] = fmin(fmax(xl[i], st.xbase[i] + xopt[i]), xu[i]);
    }
    for k in 0..npt {
        for i in 0..n {
            st.xpt[(i, k)] -= xopt[i];
        }
    }
    for i in 0..n {
        st.xpt[(i, kopt0)] = 0.0;
    }
    let sum_pq: f64 = st.pq.iter().sum();
    let v: Vec<f64> = st
        .xpt
        .mul_vec(st.pq)
        .iter()
        .zip(&xopt)
        .map(|(a, x)| a + 0.5 * sum_pq * x)
        .collect();
    r2update(st.hq, 1.0, &xopt, &v);

    // the provisional points, along the coordinate directions and their pairwise sums
    let mut ptsaux = [vec![0.0; n], vec![0.0; n]];
    for i in 0..n {
        ptsaux[0][i] = fmin(delta, st.su[i]);
        ptsaux[1][i] = fmax(-delta, st.sl[i]);
        if ptsaux[0][i] + ptsaux[1][i] < 0.0 {
            let t = ptsaux[0][i];
            ptsaux[0][i] = ptsaux[1][i];
            ptsaux[1][i] = t;
        }
        if ptsaux[1][i].abs() < 0.5 * ptsaux[0][i].abs() {
            ptsaux[1][i] = 0.5 * ptsaux[0][i];
        }
    }

    let sfrac = 0.5 / (n as f64 + 1.0);
    let mut ptsid = vec![0.0; npt];
    ptsid[0] = sfrac;
    st.bmat.fill(0.0);
    st.zmat.fill(0.0);
    for k in 0..n {
        ptsid[k + 1] = (k + 1) as f64 + sfrac;
        if k + 1 <= npt - n - 1 {
            ptsid[k + n + 1] = (k + 1) as f64 / (n as f64 + 1.0) + sfrac;
            let temp = 1.0 / (ptsaux[0][k] - ptsaux[1][k]);
            st.bmat[(k, k + 1)] = -temp + 1.0 / ptsaux[0][k];
            st.bmat[(k, k + n + 1)] = temp + 1.0 / ptsaux[1][k];
            st.bmat[(k, 0)] = -st.bmat[(k, k + 1)] - st.bmat[(k, k + n + 1)];
            st.zmat[(0, k)] = 2f64.sqrt() / (ptsaux[0][k] * ptsaux[1][k]).abs();
            st.zmat[(k + 1, k)] = st.zmat[(0, k)] * ptsaux[1][k] * temp;
            st.zmat[(k + n + 1, k)] = -st.zmat[(0, k)] * ptsaux[0][k] * temp;
        } else {
            st.bmat[(k, 0)] = -1.0 / ptsaux[0][k];
            st.bmat[(k, k + 1)] = 1.0 / ptsaux[0][k];
            st.bmat[(k, k + npt)] = -0.5 * ptsaux[0][k] * ptsaux[0][k];
        }
    }
    let ij = setij(n, npt);
    for k in 2 * n + 1..npt {
        let (ip, iq) = ij[k - 2 * n - 1];
        ptsid[k] = (ip + 1) as f64 + (iq + 1) as f64 / (n as f64 + 1.0) + sfrac;
        let temp = 1.0 / (ptsaux[0][ip] * ptsaux[0][iq]);
        let col = k - n - 1;
        st.zmat[(0, col)] = temp;
        st.zmat[(k, col)] = temp;
        st.zmat[(ip + 1, col)] = -temp;
        st.zmat[(iq + 1, col)] = -temp;
    }
    if kopt0 != 0 {
        st.bmat.swap_cols(0, kopt0);
        st.zmat.swap_rows(0, kopt0);
    }
    ptsid[0] = ptsid[kopt0];
    ptsid[kopt0] = 0.0;

    // replace the original points by the provisional ones one at a time while the denominators stay healthy
    let mut score: Vec<f64> = (0..npt).map(|k| sum_sq(st.xpt.col(k)).sqrt()).collect();
    score[kopt0] = 0.0;
    let scoreinc = score.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut nprov = npt - 1;
    let maxiter = (npt * npt).min(i32::MAX as usize);
    let mut wmv = vec![0.0; npt + n];
    for _ in 0..maxiter {
        if score.iter().all(|&s| s <= 0.0) || nprov <= 1 {
            break;
        }
        // KORIG: the original point with the smallest positive score
        let mut korig = 0;
        let mut best = f64::INFINITY;
        for (k, &s) in score.iter().enumerate() {
            if s > 0.0 && s < best {
                best = s;
                korig = k;
            }
        }
        for k in 0..npt {
            let w = if k == kopt0 {
                0.0
            } else if ptsid[k] <= 0.0 {
                dot(st.xpt.col(korig), st.xpt.col(k))
            } else {
                match ptsid_coords(ptsid[k], n) {
                    (Some(ip), Some(iq)) => {
                        st.xpt[(ip, korig)] * ptsaux[0][ip] + st.xpt[(iq, korig)] * ptsaux[0][iq]
                    }
                    (Some(ip), None) => st.xpt[(ip, korig)] * ptsaux[0][ip],
                    (None, Some(iq)) => st.xpt[(iq, korig)] * ptsaux[1][iq],
                    (None, None) => 0.0,
                }
            };
            wmv[k] = 0.5 * w * w;
        }
        for i in 0..n {
            wmv[npt + i] = st.xpt[(i, korig)];
        }
        let w1 = &wmv[..npt];
        let w2 = &wmv[npt..];
        let zw = st.zmat.tmul_vec(w1);
        let mut vlag = st.zmat.mul_vec(&zw);
        let bw2 = st.bmat.tmul_vec_cols(npt, w2);
        for k in 0..npt {
            vlag[k] += bw2[k];
        }
        vlag.extend(st.bmat.mul_vec(&wmv));
        // W'HW = ||Zᵀ W1||² + W2ᵀ (B1 W1 + B W) (W2 as in Powell's BOBYQA; see THIRD-PARTY-NOTICES.md)
        let b1w1 = st.bmat.mul_vec_cols(npt, w1);
        let bw = st.bmat.mul_vec(&wmv);
        let bsum: f64 = w2
            .iter()
            .zip(b1w1.iter().zip(&bw))
            .map(|(a, (b, c))| a * (b + c))
            .sum();
        let beta = 0.5 * sum_sq(st.xpt.col(korig)).powi(2) - sum_sq(&zw) - bsum;
        vlag[kopt0] += 1.0;

        let mut den = vec![0.0; npt];
        for k in 0..npt {
            if ptsid[k] > 0.0 {
                let hdiag = sum_sq(&st.zmat.row(k));
                den[k] = hdiag * beta + vlag[k] * vlag[k];
            }
        }
        let vmax = vlag[..npt]
            .iter()
            .map(|v| v * v)
            .fold(f64::NEG_INFINITY, f64::max);
        if !(sum_abs(&vlag).is_finite() && den.iter().any(|&d| d > 5.0e-2 * vmax)) {
            score[korig] = -score[korig] - scoreinc;
            continue;
        }
        let kprov = maxloc(&den);
        if kprov != korig {
            st.bmat.swap_cols(kprov, korig);
            st.zmat.swap_rows(kprov, korig);
            vlag.swap(kprov, korig);
        }
        ptsid[kprov] = ptsid[korig];
        ptsid[korig] = 0.0;
        score[korig] = 0.0;
        for s in score.iter_mut() {
            *s = s.abs();
        }
        updateh_rsc(korig, beta, &vlag, st.bmat, st.zmat);
        nprov -= 1;
    }

    // evaluate the remaining provisional points and update the model for each
    let kbase = *st.kopt;
    let fbase = st.fval[*st.kopt];
    let mut info = None;
    if nprov > 0 {
        for kpt in 0..npt {
            if ptsid[kpt] <= 0.0 {
                continue;
            }
            r1update(st.hq, st.pq[kpt], st.xpt.col(kpt));
            st.pq[kpt] = 0.0;
            let (ip, iq) = ptsid_coords(ptsid[kpt], n);
            let mut xnew = vec![0.0; n];
            let (mut xp, mut xq) = (0.0, 0.0);
            match (ip, iq) {
                (Some(p), Some(q)) => {
                    xp = ptsaux[0][p];
                    xnew[p] = xp;
                    xq = ptsaux[0][q];
                    xnew[q] = xq;
                }
                (Some(p), None) => {
                    xp = ptsaux[0][p];
                    xnew[p] = xp;
                }
                (None, Some(q)) => {
                    xq = ptsaux[1][q];
                    xnew[q] = xq;
                }
                (None, None) => {}
            }
            let change: f64 = xnew
                .iter()
                .zip(st.xpt.col(kpt))
                .map(|(a, b)| (a - b).abs())
                .sum();
            if change <= 1.0e-2 * delta || !sum_abs(&xnew).is_finite() {
                continue;
            }
            st.xpt.set_col(kpt, &xnew);
            let x = xinbd(st.xbase, st.xpt.col(kpt), xl, xu, st.sl, st.su);
            let f = evaluate(calfun, &x);
            *st.nf += 1;
            st.fval[kpt] = f;
            if f < st.fval[*st.kopt] {
                *st.kopt = kpt;
            }
            if let Some(exit) = checkexit(maxfun, *st.nf, f, ftarget, &x, calfun.terminate) {
                info = Some(exit);
                break;
            }

            let mut vquad = fbase;
            let mut xxpt = vec![0.0; npt];
            match (ip, iq) {
                (Some(p), Some(q)) => {
                    vquad += xp * (st.gopt[p] + 0.5 * xp * st.hq[(p, p)]);
                    vquad += xq * (st.gopt[q] + 0.5 * xq * st.hq[(q, q)]);
                    vquad += xp * xq * st.hq[(p, q)];
                    for k in 0..npt {
                        xxpt[k] = xp * st.xpt[(p, k)] + xq * st.xpt[(q, k)];
                    }
                }
                (Some(p), None) => {
                    vquad += xp * (st.gopt[p] + 0.5 * xp * st.hq[(p, p)]);
                    for k in 0..npt {
                        xxpt[k] = xp * st.xpt[(p, k)];
                    }
                }
                (None, Some(q)) => {
                    vquad += xq * (st.gopt[q] + 0.5 * xq * st.hq[(q, q)]);
                    for k in 0..npt {
                        xxpt[k] = xq * st.xpt[(q, k)];
                    }
                }
                (None, None) => {}
            }
            vquad += 0.5
                * xxpt
                    .iter()
                    .zip(st.pq.iter())
                    .map(|(x, p)| x * p * x)
                    .sum::<f64>();
            let moderr = f - vquad;
            for i in 0..n {
                st.gopt[i] += moderr * st.bmat[(i, kpt)];
            }
            let pqinc: Vec<f64> = st
                .zmat
                .mul_vec(&st.zmat.row(kpt))
                .iter()
                .map(|v| moderr * v)
                .collect();
            for k in 0..npt {
                if ptsid[k] <= 0.0 {
                    st.pq[k] += pqinc[k];
                }
            }
            for k in 0..npt {
                if ptsid[k] <= 0.0 {
                    continue;
                }
                match ptsid_coords(ptsid[k], n) {
                    (Some(p), Some(q)) => {
                        st.hq[(p, p)] += pqinc[k] * ptsaux[0][p] * ptsaux[0][p];
                        st.hq[(q, q)] += pqinc[k] * ptsaux[0][q] * ptsaux[0][q];
                        st.hq[(p, q)] += pqinc[k] * ptsaux[0][p] * ptsaux[0][q];
                        st.hq[(q, p)] = st.hq[(p, q)];
                    }
                    (Some(p), None) => st.hq[(p, p)] += pqinc[k] * ptsaux[0][p] * ptsaux[0][p],
                    (None, Some(q)) => st.hq[(q, q)] += pqinc[k] * ptsaux[1][q] * ptsaux[1][q],
                    (None, None) => {}
                }
            }
            ptsid[kpt] = 0.0;
        }
    }

    if *st.kopt != kbase {
        let xk = st.xpt.col(*st.kopt).to_vec();
        let hx = hess_mul(&xk, st.xpt, st.pq, Some(st.hq));
        for i in 0..n {
            st.gopt[i] += hx[i];
        }
    }
    info
}

/// PRIMA's `updateh_rsc`: the update of BMAT and ZMAT used by RESCUE, with VLAG and BETA supplied.
fn updateh_rsc(knew: usize, beta: f64, vlag_in: &[f64], bmat: &mut Mat, zmat: &mut Mat) -> bool {
    let npt = zmat.rows;
    let n = bmat.rows;
    let mut vlag = vlag_in.to_vec();
    let tau = vlag[knew];
    let denom = sum_sq(&zmat.row(knew)) * beta + tau * tau;
    if !((sum_abs(&vlag) + beta.abs()).is_finite() && denom > 0.0) {
        return false;
    }
    vlag[knew] -= 1.0;
    rotate_zmat_row(zmat, knew);
    let zk1 = zmat[(knew, 0)];
    let mut hcol: Vec<f64> = (0..npt).map(|k| zk1 * zmat[(k, 0)]).collect();
    hcol.extend_from_slice(bmat.col(knew));
    let sqrtdn = denom.sqrt();
    let zknew1 = zk1 / sqrtdn;
    for k in 0..npt {
        zmat[(k, 0)] = (tau / sqrtdn) * zmat[(k, 0)] - zknew1 * vlag[k];
    }
    zmat[(knew, 0)] = zknew1;
    let alpha = hcol[knew];
    debug_assert_eq!(hcol.len(), npt + n);
    update_bmat(bmat, npt, alpha, beta, tau, denom, &vlag, &hcol);
    true
}
