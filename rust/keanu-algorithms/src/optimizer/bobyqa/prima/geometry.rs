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

//! Choosing the interpolation point to drop and the geometry-improving step (PRIMA's `geometry.f90` for BOBYQA).

use super::common::{calden, hess_mul};
use super::linalg::{Mat, Vector, dot, fmax, fmin, maxloc, norm, sign, sum_abs};

fn distsq_from(xpt: &Mat, centre: &[f64]) -> Vec<f64> {
    (0..xpt.cols)
        .map(|k| {
            xpt.col(k)
                .iter()
                .zip(centre)
                .map(|(a, b)| (a - b) * (a - b))
                .sum()
        })
        .collect()
}

/// PRIMA's `setdrop_tr`: the interpolation point to replace after a trust region step, or None.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setdrop_tr(
    kopt: usize,
    ximproved: bool,
    bmat: &Mat,
    d: &[f64],
    rho: f64,
    xpt: &Mat,
    zmat: &Mat,
) -> Option<usize> {
    let n = xpt.rows;
    let distsq = if ximproved {
        let centre: Vector = (0..n).map(|i| xpt[(i, kopt)] + d[i]).collect();
        distsq_from(xpt, &centre)
    } else {
        distsq_from(xpt, xpt.col(kopt))
    };
    let den = calden(kopt, bmat, d, xpt, zmat);
    let mut score: Vec<f64> = distsq
        .iter()
        .zip(&den)
        .map(|(&ds, &dn)| fmax(1.0, ds / (rho * rho)).powi(4) * dn)
        .collect();
    if !ximproved {
        score[kopt] = -1.0;
    }
    for s in score.iter_mut() {
        if s.is_nan() {
            *s = -1.0;
        }
    }
    let mut knew = None;
    if score.iter().any(|&s| s > 1.0) || (ximproved && score.iter().any(|&s| s > 0.0)) {
        knew = Some(maxloc(&score));
    }
    if ximproved && knew.is_none() {
        knew = Some(maxloc(&distsq));
    }
    knew
}

/// PRIMA's `geostep`: a step from XOPT that improves the geometry of the interpolation set by making the
/// KNEW-th Lagrange function large, within DELBAR and the bounds.
#[allow(clippy::too_many_arguments)]
pub(crate) fn geostep(
    knew: usize,
    kopt: usize,
    bmat: &Mat,
    delbar: f64,
    sl: &[f64],
    su: &[f64],
    xpt: &Mat,
    zmat: &Mat,
) -> Vec<f64> {
    let n = xpt.rows;
    let npt = xpt.cols;
    let fallback = |xpt: &Mat, xopt: &[f64]| -> Vec<f64> {
        let d: Vector = (0..n).map(|i| xpt[(i, knew)] - xopt[i]).collect();
        let scale = fmin(0.5, delbar / norm(&d));
        d.iter().map(|v| scale * v).collect()
    };

    let zrow = zmat.row(knew);
    let pqlag = zmat.mul_vec(&zrow);
    let alpha = pqlag[knew];
    let xopt = xpt.col(kopt).to_vec();
    let hx = hess_mul(&xopt, xpt, &pqlag, None);
    let mut glag: Vec<f64> = (0..n).map(|i| bmat[(i, knew)] + hx[i]).collect();
    if !sum_abs(&glag).is_finite() {
        return fallback(xpt, &xopt);
    }

    let gx = dot(&glag, &xopt);
    let mut dderiv: Vec<f64> = xpt.tmul_vec(&glag).iter().map(|v| v - gx).collect();
    let distsq = distsq_from(xpt, &xopt);

    // stplen(:, k) = [slbd, subd, stpm]; isbd(:, k) = [ilbd, iubd, 0] with signed 1-based bound indices
    let mut stplen = vec![[0.0f64; 3]; npt];
    let mut isbd = vec![[0i64; 3]; npt];
    for k in 0..npt {
        if k == kopt || dderiv[k].is_nan() {
            dderiv[k] = 0.0;
            continue;
        }
        let mut subd = delbar / distsq[k].sqrt();
        let mut slbd = -subd;
        let mut ilbd = 0i64;
        let mut iubd = 0i64;
        let sumin = fmin(1.0, subd);
        let xdiff: Vector = (0..n).map(|i| xpt[(i, k)] - xopt[i]).collect();
        let mut lfrac: Vec<f64> = xdiff.iter().map(|&x| sign(subd, -x)).collect();
        let mut ufrac: Vec<f64> = xdiff.iter().map(|&x| sign(subd, x)).collect();
        for i in 0..n {
            if sl[i] - xopt[i] > -xdiff[i].abs() * subd {
                lfrac[i] = (sl[i] - xopt[i]) / xdiff[i];
            }
            if su[i] - xopt[i] < xdiff[i].abs() * subd {
                ufrac[i] = (su[i] - xopt[i]) / xdiff[i];
            }
        }
        let mut slbd_test = vec![slbd; n];
        let mut subd_test = vec![subd; n];
        for i in 0..n {
            if xdiff[i] > 0.0 {
                slbd_test[i] = lfrac[i];
                subd_test[i] = ufrac[i];
            }
            if xdiff[i] < 0.0 {
                slbd_test[i] = ufrac[i];
                subd_test[i] = lfrac[i];
            }
        }
        if slbd_test.iter().any(|&v| v > slbd) {
            let i = maxloc(&slbd_test);
            slbd = slbd_test[i];
            ilbd = -((i + 1) as i64) * sign(1.0, xdiff[i]) as i64;
        }
        if subd_test.iter().any(|&v| v < subd) {
            let i = super::linalg::minloc(&subd_test);
            subd = fmax(sumin, subd_test[i]);
            iubd = ((i + 1) as i64) * sign(1.0, xdiff[i]) as i64;
        }
        let mut stpm = 0.5;
        if k == knew {
            stpm = slbd;
            if (1.0 - dderiv[k]).abs() > 0.0 {
                stpm = -0.5 * dderiv[k] / (1.0 - dderiv[k]);
            }
        }
        stpm = fmax(slbd, fmin(subd, stpm));
        stplen[k] = [slbd, subd, stpm];
        isbd[k] = [ilbd, iubd, 0];
    }

    // predicted |denominator| for each candidate step, choosing the best
    let mut predsq = vec![[0.0f64; 3]; npt];
    for k in 0..npt {
        for r in 0..3 {
            let t = stplen[k][r];
            let mut vlag = if k == knew {
                t * (t * (1.0 - dderiv[k]) + dderiv[k])
            } else {
                t * (1.0 - t) * dderiv[k]
            };
            if vlag.is_nan() {
                vlag = 0.0;
            }
            let betabd = 0.5 * (t * (1.0 - t) * distsq[k]).powi(2);
            let mut p = vlag * vlag * (vlag * vlag + alpha * betabd);
            if p.is_nan() {
                p = 0.0;
            }
            predsq[k][r] = p;
        }
    }
    let ksqs: Vec<usize> = (0..3)
        .map(|r| maxloc(&predsq.iter().map(|p| p[r]).collect::<Vec<_>>()))
        .collect();
    let isq = maxloc(&[predsq[ksqs[0]][0], predsq[ksqs[1]][1], predsq[ksqs[2]][2]]);
    let ksq = ksqs[isq];
    let stpsiz = stplen[ksq][isq];
    let ibd = isbd[ksq][isq];
    let mut xline: Vec<f64> = (0..n)
        .map(|i| {
            fmax(
                sl[i],
                fmin(su[i], xopt[i] + stpsiz * (xpt[(i, ksq)] - xopt[i])),
            )
        })
        .collect();
    if ibd < 0 {
        let i = (-ibd - 1) as usize;
        xline[i] = sl[i];
    }
    if ibd > 0 {
        let i = (ibd - 1) as usize;
        xline[i] = su[i];
    }
    let mut d: Vec<f64> = (0..n).map(|i| xline[i] - xopt[i]).collect();
    let den_line = calden(kopt, bmat, &d, xpt, zmat);
    if delbar > 1.0e-2 {
        return d;
    }

    // try a Cauchy step (both uphill and downhill)
    let bigstp = delbar + delbar;
    let mut xcauchy = xopt.clone();
    let mut vlagsq_cauchy = 0.0;
    for uphill in 0..2 {
        if uphill == 1 {
            glag.iter_mut().for_each(|g| *g = -*g);
        }
        let mut s = vec![0.0; n];
        let mut mask_free: Vec<bool> = (0..n)
            .map(|i| fmin(xopt[i] - sl[i], glag[i]) > 0.0 || fmax(xopt[i] - su[i], glag[i]) < 0.0)
            .collect();
        for i in 0..n {
            if mask_free[i] {
                s[i] = bigstp;
            }
        }
        let mut ggfree: f64 = (0..n)
            .filter(|&i| mask_free[i])
            .map(|i| glag[i] * glag[i])
            .sum();
        if ggfree <= 0.0 || ggfree.is_nan() {
            continue;
        }
        let mut sfixsq = 0.0;
        let mut grdstp = 0.0;
        for _ in 0..n {
            let resis = delbar * delbar - sfixsq;
            if resis <= 0.0 {
                break;
            }
            let ssqsav = sfixsq;
            grdstp = (resis / ggfree).sqrt();
            let mut fixed = vec![false; n];
            for i in 0..n {
                let xtemp = xopt[i] - grdstp * glag[i];
                let fixl = s[i] >= bigstp && xtemp <= sl[i];
                let fixu = s[i] >= bigstp && xtemp >= su[i];
                mask_free[i] = s[i] >= bigstp && !(fixl || fixu);
                if fixl {
                    s[i] = sl[i] - xopt[i];
                }
                if fixu {
                    s[i] = su[i] - xopt[i];
                }
                fixed[i] = fixl || fixu;
            }
            sfixsq += (0..n)
                .filter(|&i| fixed[i])
                .map(|i| s[i] * s[i])
                .sum::<f64>();
            ggfree = (0..n)
                .filter(|&i| mask_free[i])
                .map(|i| glag[i] * glag[i])
                .sum();
            if !(sfixsq > ssqsav && ggfree > 0.0) {
                break;
            }
        }
        let mut x = vec![0.0; n];
        for i in 0..n {
            x[i] = if glag[i] > 0.0 { sl[i] } else { su[i] };
            if s[i].abs() <= 0.0 {
                x[i] = xopt[i];
            }
            if s[i] >= bigstp {
                x[i] = fmax(sl[i], fmin(su[i], xopt[i] - grdstp * glag[i]));
                s[i] = -grdstp * glag[i];
            }
        }
        let gs = dot(&glag, &s);
        let sxpt = xpt.tmul_vec(&s);
        let mut curv: f64 = sxpt.iter().zip(&pqlag).map(|(v, p)| v * p * v).sum();
        if uphill == 1 {
            curv = -curv;
        }
        let vlagsq;
        if curv > -gs && curv < -(1.0 + 2f64.sqrt()) * gs {
            let scaling = -gs / curv;
            for i in 0..n {
                x[i] = fmax(sl[i], fmin(su[i], xopt[i] + scaling * s[i]));
            }
            vlagsq = (0.5 * gs * scaling).powi(2);
        } else {
            vlagsq = (gs + 0.5 * curv).powi(2);
        }
        if vlagsq > vlagsq_cauchy {
            xcauchy = x;
            vlagsq_cauchy = vlagsq;
        }
    }
    let s: Vector = (0..n).map(|i| xcauchy[i] - xopt[i]).collect();
    let den_cauchy = calden(kopt, bmat, &s, xpt, zmat);
    if den_cauchy[knew] > fmax(den_line[knew], 0.0) || den_line[knew].is_nan() {
        d = s.into_vec();
    }
    if sum_abs(&d) <= 0.0 || !sum_abs(&d).is_finite() {
        d = fallback(xpt, &xopt);
    }
    d
}
