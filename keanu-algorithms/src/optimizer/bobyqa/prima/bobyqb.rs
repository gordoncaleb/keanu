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

//! The main BOBYQA iteration (PRIMA's `bobyqb.f90`).

use super::common::{
    Calfun, ExitStatus, Lagrange, REALMAX, checkexit, evaluate, hess_mul, quadinc, redrat, redrho,
    shiftbase, xinbd,
};
use super::geometry::{geostep, setdrop_tr};
use super::initialize::{inith, initq, initxf};
use super::linalg::{Mat, Vector, fmax, fmin, maxloc, norm, sum_abs};
use super::rescue::{RescueState, rescue};
use super::trustregion::{trrad, trsbox};
use super::update::{tryqalt, updateh, updateq, updatexf};
use super::{Options, Outcome};

fn model_finite(gopt: &[f64], hq: &Mat, pq: &[f64]) -> bool {
    gopt.iter().all(|v| v.is_finite()) && hq.all_finite() && pq.iter().all(|v| v.is_finite())
}

/// PRIMA's `bobyqb`, with preprocessed arguments.
pub(crate) fn bobyqb(
    calfun: &mut Calfun<'_>,
    opts: &Options,
    xl: &[f64],
    xu: &[f64],
    x0: &[f64],
) -> Outcome {
    let n = x0.len();
    let npt = opts.npt;
    let (maxfun, ftarget, rhobeg, rhoend) = (opts.maxfun, opts.ftarget, opts.rhobeg, opts.rhoend);
    let (eta1, eta2, gamma1, gamma2) = (opts.eta1, opts.eta2, opts.gamma1, opts.gamma2);
    const TRTOL: f64 = 1.0e-2;

    let init = initxf(calfun, maxfun, npt, ftarget, rhobeg, xl, xu, x0);
    let ij = init.ij;
    let mut kopt = init.kopt;
    let mut nf = init.nf;
    let mut fval = init.fval;
    let mut sl = init.sl;
    let mut su = init.su;
    let mut xbase = init.xbase;
    let mut xpt = init.xpt;
    let mut x = init.best_x;
    let mut f = fval[kopt];

    let (mut bmat, mut zmat) = (Mat::zeros(n, npt + n), Mat::zeros(npt, npt - n - 1));
    let (mut gopt, mut hq, mut pq) = (vec![0.0; n], Mat::zeros(n, n), vec![0.0; npt]);
    let mut subinfo = init.info;
    if subinfo.is_none() {
        (bmat, zmat) = inith(&ij, &xpt);
        (gopt, hq, pq) = initq(&ij, &fval, &xpt);
        if !model_finite(&gopt, &hq, &pq) {
            subinfo = Some(ExitStatus::NanInfModel);
        }
    }
    if let Some(info) = subinfo {
        return Outcome {
            x,
            f,
            evaluations: nf,
            status: info,
        };
    }

    let mut rho = rhobeg;
    let mut delta = rho;
    let mut ebound = 0.0;
    let mut rescued = false;
    let mut shortd;
    let mut ratio = -1.0;
    let mut dnorm_rec = [REALMAX; 2];
    let mut moderr_rec = [REALMAX; 2];
    let mut itest = 0usize;
    let gamma3 = fmax(1.0, fmin(0.75 * gamma2, 1.5));
    let info;
    let mut d;
    let mut dnorm;

    // MAXTR is effectively unbounded, as in PRIMA, so the MaxIterationsReached exit cannot occur
    'iterations: loop {
        let (step, crvmin) = trsbox(delta, &gopt, &hq, &pq, &sl, &su, TRTOL, xpt.col(kopt), &xpt);
        d = step;
        dnorm = fmin(delta, norm(&d));
        shortd = dnorm <= 0.5 * rho;
        let qred = -quadinc(&d, &xpt, &gopt, &pq, &hq);
        let trfail = !(qred > 1.0e-6 * rho * rho);
        let mut knew_tr: Option<usize> = None;

        if shortd || trfail {
            delta *= 0.1;
            if delta <= gamma3 * rho {
                delta = rho;
            }
            ebound = errbd(
                crvmin,
                &d,
                &gopt,
                &hq,
                &moderr_rec,
                &pq,
                rho,
                &sl,
                &su,
                xpt.col(kopt),
                &xpt,
            );
        } else {
            let step: Vector = (0..n).map(|i| xpt[(i, kopt)] + d[i]).collect();
            x = xinbd(&xbase, &step, xl, xu, &sl, &su);
            f = evaluate(calfun, &x);
            nf += 1;
            rescued = false;
            if let Some(exit) = checkexit(maxfun, nf, f, ftarget, &x, calfun.terminate) {
                info = exit;
                break 'iterations;
            }
            dnorm_rec = [dnorm_rec[1], dnorm];
            let mut moderr = f - fval[kopt] + qred;
            moderr_rec = [moderr_rec[1], moderr];
            ratio = redrat(fval[kopt] - f, qred, eta1);
            delta = trrad(delta, dnorm, eta1, eta2, gamma1, gamma2, ratio);
            if delta <= gamma3 * rho {
                delta = rho;
            }
            let mut ximproved = f < fval[kopt];

            // the Lagrange values and denominators for D, shared by the rescue test, SETDROP_TR and UPDATEH (and
            // recomputed if RESCUE changes the interpolation set)
            let mut lagrange = Lagrange::new(kopt, &bmat, &d, &xpt, &zmat);
            let mut den = lagrange.den(&zmat);
            let vlag = &lagrange.vlag;
            let vmax = vlag[..npt]
                .iter()
                .map(|v| v * v)
                .fold(f64::NEG_INFINITY, f64::max);
            let to_rescue =
                ximproved && !(sum_abs(vlag).is_finite() && den.iter().any(|&v| v > vmax));
            if to_rescue {
                if rescued {
                    info = ExitStatus::DamagingRounding;
                    break 'iterations;
                }
                let exit = rescue(
                    calfun,
                    maxfun,
                    delta,
                    ftarget,
                    xl,
                    xu,
                    RescueState {
                        kopt: &mut kopt,
                        nf: &mut nf,
                        fval: &mut fval,
                        gopt: &mut gopt,
                        hq: &mut hq,
                        pq: &mut pq,
                        sl: &mut sl,
                        su: &mut su,
                        xbase: &mut xbase,
                        xpt: &mut xpt,
                        bmat: &mut bmat,
                        zmat: &mut zmat,
                    },
                );
                if let Some(exit) = exit {
                    info = exit;
                    break 'iterations;
                }
                rescued = true;
                dnorm_rec = [REALMAX; 2];
                moderr_rec = [REALMAX; 2];
                d = (0..n)
                    .map(|i| fmax(sl[i], fmin(su[i], d[i])) - xpt[(i, kopt)])
                    .collect();
                moderr = f - fval[kopt] - quadinc(&d, &xpt, &gopt, &pq, &hq);
                ximproved = f < fval[kopt];
                lagrange = Lagrange::new(kopt, &bmat, &d, &xpt, &zmat);
                den = lagrange.den(&zmat);
            }

            knew_tr = setdrop_tr(kopt, ximproved, &den, &d, rho, &xpt);
            if let Some(knew) = knew_tr {
                let xdrop = xpt.col(knew).to_vec();
                let xosav = xpt.col(kopt).to_vec();
                updateh(knew, lagrange, &mut bmat, &mut zmat);
                let xnew: Vector = (0..n)
                    .map(|i| fmax(sl[i], fmin(su[i], xosav[i] + d[i])))
                    .collect();
                updatexf(knew, ximproved, f, &xnew, &mut kopt, &mut fval, &mut xpt);
                updateq(
                    knew, ximproved, &bmat, &d, moderr, &xdrop, &xosav, &xpt, &zmat, &mut gopt,
                    &mut hq, &mut pq,
                );
                let fshift: Vector = fval.iter().map(|v| v - fval[kopt]).collect();
                let xopt = xpt.col(kopt).to_vec();
                tryqalt(
                    &bmat, &fshift, ratio, &sl, &su, &xopt, &xpt, &zmat, &mut itest, &mut gopt,
                    &mut hq, &mut pq,
                );
                if !model_finite(&gopt, &hq, &pq) {
                    info = ExitStatus::NanInfModel;
                    break 'iterations;
                }
            }
        }

        let accurate_mod =
            moderr_rec.iter().all(|m| m.abs() <= ebound) && dnorm_rec.iter().all(|&dn| dn <= rho);
        let distsq: Vector = (0..npt)
            .map(|k| (0..n).map(|i| (xpt[(i, k)] - xpt[(i, kopt)]).powi(2)).sum())
            .collect();
        let close_itpset = distsq
            .iter()
            .all(|&ds| ds <= fmax(delta * delta, (10.0 * rho).powi(2)));
        let adequate_geo = (shortd && accurate_mod) || close_itpset;
        let small_trrad = fmax(delta, dnorm) <= rho;
        let bad_trstep = shortd || trfail || ratio <= eta1 || knew_tr.is_none();
        let improve_geo = bad_trstep && !adequate_geo;
        let bad_trstep = shortd || trfail || ratio <= 0.0 || knew_tr.is_none();
        let reduce_rho = bad_trstep && adequate_geo && small_trrad;

        if improve_geo {
            let knew_geo = maxloc(&distsq);
            let dmax = distsq.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let delbar = fmax(fmin(0.1 * dmax.sqrt(), delta), rho);
            d = geostep(knew_geo, kopt, &bmat, delbar, &sl, &su, &xpt, &zmat);
            let lagrange = Lagrange::new(kopt, &bmat, &d, &xpt, &zmat);
            let den = lagrange.den(&zmat);
            let vlag = &lagrange.vlag;
            let to_rescue = !(sum_abs(vlag).is_finite()
                && den[knew_geo] > 0.5 * vlag[knew_geo] * vlag[knew_geo]);
            if to_rescue {
                if rescued {
                    info = ExitStatus::DamagingRounding;
                    break 'iterations;
                }
                let exit = rescue(
                    calfun,
                    maxfun,
                    delta,
                    ftarget,
                    xl,
                    xu,
                    RescueState {
                        kopt: &mut kopt,
                        nf: &mut nf,
                        fval: &mut fval,
                        gopt: &mut gopt,
                        hq: &mut hq,
                        pq: &mut pq,
                        sl: &mut sl,
                        su: &mut su,
                        xbase: &mut xbase,
                        xpt: &mut xpt,
                        bmat: &mut bmat,
                        zmat: &mut zmat,
                    },
                );
                if let Some(exit) = exit {
                    info = exit;
                    break 'iterations;
                }
                rescued = true;
                dnorm_rec = [REALMAX; 2];
                moderr_rec = [REALMAX; 2];
            } else {
                let step: Vector = (0..n).map(|i| xpt[(i, kopt)] + d[i]).collect();
                x = xinbd(&xbase, &step, xl, xu, &sl, &su);
                f = evaluate(calfun, &x);
                nf += 1;
                rescued = false;
                if let Some(exit) = checkexit(maxfun, nf, f, ftarget, &x, calfun.terminate) {
                    info = exit;
                    break 'iterations;
                }
                dnorm = fmin(delbar, norm(&d));
                dnorm_rec = [dnorm_rec[1], dnorm];
                let moderr = f - fval[kopt] - quadinc(&d, &xpt, &gopt, &pq, &hq);
                moderr_rec = [moderr_rec[1], moderr];
                let ximproved = f < fval[kopt];
                let xdrop = xpt.col(knew_geo).to_vec();
                let xosav = xpt.col(kopt).to_vec();
                updateh(knew_geo, lagrange, &mut bmat, &mut zmat);
                let xnew: Vector = (0..n)
                    .map(|i| fmax(sl[i], fmin(su[i], xosav[i] + d[i])))
                    .collect();
                updatexf(
                    knew_geo, ximproved, f, &xnew, &mut kopt, &mut fval, &mut xpt,
                );
                updateq(
                    knew_geo, ximproved, &bmat, &d, moderr, &xdrop, &xosav, &xpt, &zmat, &mut gopt,
                    &mut hq, &mut pq,
                );
                if !model_finite(&gopt, &hq, &pq) {
                    info = ExitStatus::NanInfModel;
                    break 'iterations;
                }
            }
        }

        if reduce_rho {
            if rho <= rhoend {
                info = ExitStatus::SmallTrustRegionRadius;
                break 'iterations;
            }
            delta = fmax(0.5 * rho, redrho(rho, rhoend));
            rho = redrho(rho, rhoend);
            dnorm_rec = [REALMAX; 2];
            moderr_rec = [REALMAX; 2];
        }

        let xopt_sq: f64 = xpt.col(kopt).iter().map(|v| v * v).sum();
        if xopt_sq >= 1.0e3 * delta * delta {
            for i in 0..n {
                sl[i] = fmin(sl[i] - xpt[(i, kopt)], 0.0);
                su[i] = fmax(su[i] - xpt[(i, kopt)], 0.0);
            }
            shiftbase(kopt, &mut xbase, &mut xpt, &zmat, &mut bmat, &pq, &mut hq);
            for i in 0..n {
                xbase[i] = fmax(xl[i], fmin(xu[i], xbase[i]));
            }
        }
    }

    if info == ExitStatus::SmallTrustRegionRadius && shortd && dnorm > 0.1 * rhoend && nf < maxfun {
        let step: Vector = (0..n).map(|i| xpt[(i, kopt)] + d[i]).collect();
        x = xinbd(&xbase, &step, xl, xu, &sl, &su);
        f = evaluate(calfun, &x);
        nf += 1;
    }
    if fval[kopt] < f || f.is_nan() {
        x = xinbd(&xbase, xpt.col(kopt), xl, xu, &sl, &su);
        f = fval[kopt];
    }
    Outcome {
        x,
        f,
        evaluations: nf,
        status: info,
    }
}

/// PRIMA's `errbd`: a bound on the error of the model used to decide whether the geometry is adequate.
#[allow(clippy::too_many_arguments)]
fn errbd(
    crvmin: f64,
    d: &[f64],
    gopt: &[f64],
    hq: &Mat,
    moderr_rec: &[f64; 2],
    pq: &[f64],
    rho: f64,
    sl: &[f64],
    su: &[f64],
    xopt: &[f64],
    xpt: &Mat,
) -> f64 {
    let n = d.len();
    let hd = hess_mul(d, xpt, pq, Some(hq));
    let first = moderr_rec
        .iter()
        .map(|m| m.abs())
        .fold(f64::NEG_INFINITY, f64::max);
    let mut ebound = f64::INFINITY;
    for i in 0..n {
        let xnew = xopt[i] + d[i];
        let gnew = gopt[i] + hd[i];
        let mut bfirst = first;
        if xnew <= sl[i] {
            bfirst = gnew * rho;
        }
        if xnew >= su[i] {
            bfirst = -gnew * rho;
        }
        let xpt2pq: f64 = (0..xpt.cols)
            .map(|k| xpt[(i, k)] * xpt[(i, k)] * pq[k])
            .sum();
        let bsecond = 0.5 * (hq[(i, i)] + xpt2pq) * rho * rho;
        ebound = fmin(ebound, fmax(bfirst, bfirst + bsecond));
    }
    if crvmin > 0.0 {
        ebound = fmin(ebound, 0.125 * crvmin * rho * rho);
    }
    ebound
}
