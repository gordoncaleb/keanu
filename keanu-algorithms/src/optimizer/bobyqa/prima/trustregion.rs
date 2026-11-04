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

//! The bound-constrained trust region subproblem and the trust region radius update
//! (PRIMA's `trustregion.f90` for BOBYQA).

use super::common::{EPS, REALMAX, REALMIN, hess_mul, interval_max};
use super::linalg::{Mat, Vector, fmax, fmin, minloc, sign};
use std::borrow::Cow;

fn free_dot(xbdi: &[i32], a: &[f64], b: &[f64]) -> f64 {
    xbdi.iter()
        .zip(a.iter().zip(b))
        .filter(|(x, _)| **x == 0)
        .map(|(_, (u, v))| u * v)
        .sum()
}

fn free_sumsq(xbdi: &[i32], a: &[f64]) -> f64 {
    free_dot(xbdi, a, a)
}

/// PRIMA's `trsbox`: approximately minimises the quadratic model within the trust region and the bounds by a
/// truncated conjugate gradient method followed by two-dimensional searches. Returns (D, CRVMIN).
#[allow(clippy::too_many_arguments)]
pub(crate) fn trsbox(
    delta: f64,
    gopt_in: &[f64],
    hq_in: &Mat,
    pq_in: &[f64],
    sl: &[f64],
    su: &[f64],
    tol: f64,
    xopt: &[f64],
    xpt: &Mat,
) -> (Vec<f64>, f64) {
    let n = gopt_in.len();
    let gmax = gopt_in.iter().fold(0.0f64, |m, g| m.max(g.abs()));
    let (gopt, pq, hq, scaled, modscal) = if gmax > 1.0e12 {
        let modscal = fmax(2.0 * REALMIN, 1.0 / gmax);
        let mut hq = hq_in.clone();
        for j in 0..n {
            for i in 0..n {
                hq[(i, j)] *= modscal;
            }
        }
        (
            Cow::Owned(gopt_in.iter().map(|g| g * modscal).collect::<Vec<_>>()),
            Cow::Owned(pq_in.iter().map(|p| p * modscal).collect::<Vec<_>>()),
            Cow::Owned(hq),
            true,
            modscal,
        )
    } else {
        // no scaling needed: use the model as it is, without copying it
        (
            Cow::Borrowed(gopt_in),
            Cow::Borrowed(pq_in),
            Cow::Borrowed(hq_in),
            false,
            1.0,
        )
    };

    let mut xbdi = vec![0i32; n];
    for i in 0..n {
        if xopt[i] >= su[i] && gopt[i] <= 0.0 {
            xbdi[i] = 1;
        }
        if xopt[i] <= sl[i] && gopt[i] >= 0.0 {
            xbdi[i] = -1;
        }
    }
    let mut nact = xbdi.iter().filter(|&&x| x != 0).count();
    let mut d = vec![0.0; n];
    let mut crvmin = -REALMAX;
    let mut gnew = gopt.to_vec();
    let mut gredsq = free_sumsq(&xbdi, &gnew);
    let mut delsq = delta * delta;
    let mut qred = 0.0;
    let mut beta = 0.0;
    let mut itercg = 0usize;
    let mut twod_search = false;
    let mut s = vec![0.0; n];
    let mut ggsav = 0.0;

    let maxiter = 10_000usize.min((n - nact) * (n - nact));
    for _ in 0..maxiter {
        let resid = delsq - free_sumsq(&xbdi, &d);
        if resid <= 0.0 {
            twod_search = true;
            break;
        }
        for i in 0..n {
            s[i] = if itercg == 0 {
                -gnew[i]
            } else {
                beta * s[i] - gnew[i]
            };
            if xbdi[i] != 0 {
                s[i] = 0.0;
            }
        }
        let stepsq: f64 = s.iter().map(|v| v * v).sum();
        let ds = free_dot(&xbdi, &d, &s);
        if !(stepsq > EPS * delsq && gredsq * delsq > (tol * qred).powi(2) && !ds.is_nan()) {
            break;
        }
        let sqrtd = (stepsq * resid + ds * ds)
            .sqrt()
            .max((stepsq * resid).sqrt())
            .max(ds.abs());
        let bstep = if ds >= 0.0 {
            resid / (sqrtd + ds)
        } else {
            (sqrtd - ds) / stepsq
        };
        if bstep <= 0.0 || !bstep.is_finite() {
            break;
        }
        let hs = hess_mul(&s, xpt, &pq, Some(&hq));
        let shs = free_dot(&xbdi, &s, &hs);
        let mut stplen = bstep;
        if shs > 0.0 {
            stplen = fmin(bstep, gredsq / shs);
        }

        let mut sbound = vec![stplen; n];
        for i in 0..n {
            let xnew = xopt[i] + d[i];
            let xtest = xnew + stplen * s[i];
            if s[i] > 0.0 && xtest > su[i] {
                sbound[i] = (su[i] - xnew) / s[i];
            }
            if s[i] < 0.0 && xtest < sl[i] {
                sbound[i] = (sl[i] - xnew) / s[i];
            }
            if sbound[i].is_nan() {
                sbound[i] = stplen;
            }
        }
        let mut iact: Option<usize> = None;
        if sbound.iter().any(|&b| b < stplen) {
            let i = minloc(&sbound);
            iact = Some(i);
            stplen = sbound[i];
        }

        let mut sdec = 0.0;
        if stplen > 0.0 {
            itercg += 1;
            let rayleighq = shs / stepsq;
            if iact.is_none() && rayleighq > 0.0 {
                crvmin = if crvmin <= -REALMAX {
                    rayleighq
                } else {
                    fmin(crvmin, rayleighq)
                };
            }
            ggsav = gredsq;
            for i in 0..n {
                gnew[i] += stplen * hs[i];
            }
            gredsq = free_sumsq(&xbdi, &gnew);
            let dold = d.clone();
            for i in 0..n {
                d[i] += stplen * s[i];
            }
            if !d.iter().map(|v| v.abs()).sum::<f64>().is_finite() {
                d = dold;
                break;
            }
            sdec = fmax(stplen * (ggsav - 0.5 * stplen * shs), 0.0);
            qred += sdec;
        }

        if let Some(i) = iact {
            nact += 1;
            xbdi[i] = sign(1.0, s[i]) as i32;
            if nact >= n {
                break;
            }
            delsq -= d[i] * d[i];
            if delsq <= 0.0 {
                twod_search = true;
                break;
            }
            beta = 0.0;
            itercg = 0;
            gredsq = free_sumsq(&xbdi, &gnew);
        } else if stplen < bstep {
            if itercg >= n - nact || sdec <= tol * qred || sdec.is_nan() || qred.is_nan() {
                break;
            }
            beta = gredsq / ggsav;
        } else {
            twod_search = true;
            break;
        }
    }

    let maxiter = if twod_search {
        crvmin = 0.0;
        10 * (n - nact)
    } else {
        0
    };

    let mut nactsav = nact as i64 - 1;
    let mut dredsq = 0.0;
    let mut hdred: Vector = smallvec::smallvec![0.0; n];
    for iter in 1..=maxiter {
        let xnew: Vector = (0..n).map(|i| xopt[i] + d[i]).collect();
        for i in 0..n {
            if xbdi[i] == 0 && xnew[i] >= su[i] {
                xbdi[i] = 1;
            }
            if xbdi[i] == 0 && xnew[i] <= sl[i] {
                xbdi[i] = -1;
            }
        }
        nact = xbdi.iter().filter(|&&x| x != 0).count();
        if nact + 1 >= n {
            break;
        }
        gredsq = free_sumsq(&xbdi, &gnew);
        let dredg = free_dot(&xbdi, &d, &gnew);
        if iter == 1 || nact as i64 > nactsav {
            dredsq = free_sumsq(&xbdi, &d);
            let dred: Vector = (0..n)
                .map(|i| if xbdi[i] == 0 { d[i] } else { 0.0 })
                .collect();
            hdred = hess_mul(&dred, xpt, &pq, Some(&hq));
            nactsav = nact as i64;
        }
        let temp = gredsq * dredsq - dredg * dredg;
        if !(temp > tol * tol * fmax(gredsq * dredsq, qred * qred)) {
            break;
        }
        let temp = temp.sqrt();
        for i in 0..n {
            s[i] = if xbdi[i] != 0 {
                0.0
            } else {
                (dredg * d[i] - dredsq * gnew[i]) / temp
            };
        }
        let sredg = -temp;

        let mut tanbd = vec![1.0; n];
        for i in 0..n {
            let ssq = d[i] * d[i] + s[i] * s[i];
            let mut sqdscr = -REALMAX;
            if xbdi[i] == 0 && xopt[i] - sl[i] < ssq.sqrt() {
                sqdscr = fmax(0.0, ssq - (xopt[i] - sl[i]).powi(2)).sqrt();
            }
            if sqdscr - s[i] > 0.0 {
                tanbd[i] = fmin(tanbd[i], (xnew[i] - sl[i]) / (sqdscr - s[i]));
            }
            let mut sqdscr = -REALMAX;
            if xbdi[i] == 0 && su[i] - xopt[i] < ssq.sqrt() {
                sqdscr = fmax(0.0, ssq - (su[i] - xopt[i]).powi(2)).sqrt();
            }
            if sqdscr + s[i] > 0.0 {
                tanbd[i] = fmin(tanbd[i], (su[i] - xnew[i]) / (sqdscr + s[i]));
            }
            if tanbd[i].is_nan() {
                tanbd[i] = 0.0;
            }
        }
        let mut iact: Option<usize> = None;
        let mut hangt_bd = 1.0;
        if tanbd.iter().any(|&t| t < 1.0) {
            let i = minloc(&tanbd);
            iact = Some(i);
            hangt_bd = tanbd[i];
        }
        if hangt_bd <= 0.0 {
            break;
        }

        let hs = hess_mul(&s, xpt, &pq, Some(&hq));
        let shs = free_dot(&xbdi, &s, &hs);
        let dhs = free_dot(&xbdi, &d, &hs);
        let dhd = free_dot(&xbdi, &d, &hdred);
        let args = [shs, dhd, dhs, dredg, sredg];
        if args.iter().any(|a| a.is_nan()) {
            break;
        }
        let grid_size = 2 * (17.0 * hangt_bd + 4.1).round() as usize;
        let hangt = interval_max(|h| interval_fun_trsbox(h, &args), 0.0, hangt_bd, grid_size);
        let sdec = interval_fun_trsbox(hangt, &args);
        if !(sdec > 0.0) {
            break;
        }

        let cth = fmin(
            (1.0 - hangt * hangt) / (1.0 + hangt * hangt),
            1.0 - hangt * hangt,
        );
        let sth = fmin((hangt + hangt) / (1.0 + hangt * hangt), hangt + hangt);
        for i in 0..n {
            gnew[i] += (cth - 1.0) * hdred[i] + sth * hs[i];
        }
        let dold = d.clone();
        for i in 0..n {
            if xbdi[i] == 0 {
                d[i] = cth * d[i] + sth * s[i];
            }
        }
        if !d.iter().map(|v| v.abs()).sum::<f64>().is_finite() {
            d = dold;
            break;
        }
        for i in 0..n {
            hdred[i] = cth * hdred[i] + sth * hs[i];
        }
        qred += sdec;
        match iact {
            Some(i) if hangt >= hangt_bd => {
                xbdi[i] = sign(1.0, xopt[i] + d[i] - 0.5 * (sl[i] + su[i])) as i32;
            }
            _ => {
                if !(sdec > tol * qred) {
                    break;
                }
            }
        }
    }

    let mut xnew: Vec<f64> = (0..n)
        .map(|i| fmax(sl[i], fmin(su[i], xopt[i] + d[i])))
        .collect();
    for i in 0..n {
        if xbdi[i] == -1 {
            xnew[i] = sl[i];
        }
        if xbdi[i] == 1 {
            xnew[i] = su[i];
        }
    }
    let d: Vector = (0..n).map(|i| xnew[i] - xopt[i]).collect();
    if crvmin <= -REALMAX || crvmin.is_nan() {
        crvmin = 0.0;
    }
    if scaled && crvmin > 0.0 {
        crvmin /= modscal;
    }
    (d.into_vec(), crvmin)
}

fn interval_fun_trsbox(hangt: f64, args: &[f64; 5]) -> f64 {
    let mut f = 0.0;
    if hangt.abs() > 0.0 {
        let sth = (hangt + hangt) / (1.0 + hangt * hangt);
        f = args[0] + hangt * (hangt * args[1] - args[2] - args[2]);
        f = sth * (hangt * args[3] - args[4] - 0.5 * sth * f);
    }
    f
}

/// PRIMA's `trrad`: the new trust region radius from the reduction ratio.
pub(crate) fn trrad(
    delta_in: f64,
    dnorm: f64,
    eta1: f64,
    eta2: f64,
    gamma1: f64,
    gamma2: f64,
    ratio: f64,
) -> f64 {
    if ratio <= eta1 {
        fmin(gamma1 * delta_in, dnorm)
    } else if ratio <= eta2 {
        fmax(gamma1 * delta_in, dnorm)
    } else {
        fmax(gamma1 * delta_in, gamma2 * dnorm)
    }
}
