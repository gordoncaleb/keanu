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

//! A Rust translation of PRIMA's BOBYQA: M. J. D. Powell's method for bound-constrained minimisation without
//! derivatives, by trust region steps on quadratic models built from interpolation.
//!
//! The entry point is [`minimize`], which mirrors PRIMA's `bobyqa` subroutine, including its preprocessing of
//! invalid arguments.

// This module is a line-by-line translation of PRIMA's Fortran so that it can be audited against the original:
// - negated comparisons such as `!(a > b)` are deliberate: they are PRIMA's `.NOT. (A > B)`, which is true when either
//   side is NaN, and rewriting them as `a <= b` would change the behaviour;
// - index loops and long argument lists mirror the Fortran.
#![allow(
    clippy::neg_cmp_op_on_partial_ord,
    clippy::needless_range_loop,
    clippy::int_plus_one,
    clippy::needless_late_init,
    clippy::too_many_arguments
)]

mod bobyqb;
mod common;
mod geometry;
mod initialize;
mod linalg;
mod rescue;
mod trustregion;
mod update;

pub use common::ExitStatus;
use common::{BOUNDMAX, Calfun, EPS, MAXFUN_DIM_DFT, RHOBEG_DFT, RHOEND_DFT, moderatex};
use std::ops::ControlFlow;

/// Settings for [`minimize`]. `None` means PRIMA's default.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct Settings {
    /// Initial trust region radius (default 1, reduced to a quarter of the smallest bound range if needed).
    pub rhobeg: Option<f64>,
    /// Final trust region radius: the accuracy required in the variables (default 1e-6 scaled with RHOBEG).
    pub rhoend: Option<f64>,
    /// Maximum number of function evaluations (default 500 n).
    pub maxfun: Option<usize>,
    /// Number of interpolation points, between n + 2 and (n + 1)(n + 2) / 2 (default 2n + 1).
    pub npt: Option<usize>,
    /// Stop once a function value at or below this is found (default -infinity, i.e. never).
    pub ftarget: Option<f64>,
    /// Keep the starting point even if it is close to a bound (default: true unless a valid RHOBEG is given).
    pub honour_x0: Option<bool>,
}

/// Setters for each setting, for use as `Settings::default().with_...(...)`.
impl Settings {
    /// Sets [`rhobeg`](Self::rhobeg) to `Some(rhobeg)`.
    #[must_use]
    pub fn with_rhobeg(mut self, rhobeg: f64) -> Self {
        self.rhobeg = Some(rhobeg);
        self
    }

    /// Sets [`rhoend`](Self::rhoend) to `Some(rhoend)`.
    #[must_use]
    pub fn with_rhoend(mut self, rhoend: f64) -> Self {
        self.rhoend = Some(rhoend);
        self
    }

    /// Sets [`maxfun`](Self::maxfun) to `Some(maxfun)`.
    #[must_use]
    pub fn with_maxfun(mut self, maxfun: usize) -> Self {
        self.maxfun = Some(maxfun);
        self
    }

    /// Sets [`npt`](Self::npt) to `Some(npt)`.
    #[must_use]
    pub fn with_npt(mut self, npt: usize) -> Self {
        self.npt = Some(npt);
        self
    }

    /// Sets [`ftarget`](Self::ftarget) to `Some(ftarget)`.
    #[must_use]
    pub fn with_ftarget(mut self, ftarget: f64) -> Self {
        self.ftarget = Some(ftarget);
        self
    }

    /// Sets [`honour_x0`](Self::honour_x0) to `Some(honour_x0)`.
    #[must_use]
    pub fn with_honour_x0(mut self, honour_x0: bool) -> Self {
        self.honour_x0 = Some(honour_x0);
        self
    }
}

/// The result of [`minimize`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Outcome {
    /// The best point found.
    pub x: Vec<f64>,
    /// The (moderated) function value at `x`.
    pub f: f64,
    /// The number of function evaluations used.
    pub evaluations: usize,
    /// Why the solver stopped.
    pub status: ExitStatus,
}

/// The preprocessed arguments.
#[derive(Debug, Clone)]
pub(crate) struct Options {
    pub maxfun: usize,
    pub npt: usize,
    pub ftarget: f64,
    pub rhobeg: f64,
    pub rhoend: f64,
    pub eta1: f64,
    pub eta2: f64,
    pub gamma1: f64,
    pub gamma2: f64,
}

/// Minimises `calfun` subject to `xl <= x <= xu` starting from `x0`. Bounds may be infinite (or empty slices
/// for none). NaN values of the objective are treated as a very large value.
pub fn minimize(
    mut calfun: impl FnMut(&[f64]) -> f64,
    x0: &[f64],
    xl: &[f64],
    xu: &[f64],
    settings: &Settings,
) -> Outcome {
    minimize_with_termination(|x| ControlFlow::Continue(calfun(x)), x0, xl, xu, settings)
}

/// Like [`minimize`], but the objective can stop the solver: returning `ControlFlow::Break(f)` records the value
/// `f` and ends the solve with [`ExitStatus::CallbackTerminate`] (PRIMA's callback `terminate` flag), returning
/// the best point evaluated so far. Use it to stop at once when the objective fails.
pub fn minimize_with_termination(
    mut calfun: impl FnMut(&[f64]) -> ControlFlow<f64, f64>,
    x0: &[f64],
    xl: &[f64],
    xu: &[f64],
    settings: &Settings,
) -> Outcome {
    let n = x0.len();
    // as PRIMA: only a NaN or very low lower bound (or NaN or very high upper bound) means "no bound"; a lower
    // bound of +infinity (or an upper bound of -infinity) is kept, so it leaves no space between the bounds
    let bound = |b: &[f64], i: usize, default: f64, unbounded: fn(f64) -> bool| -> f64 {
        let v = if b.is_empty() { default } else { b[i] };
        if v.is_nan() || unbounded(v) {
            default
        } else {
            v
        }
    };
    let xl: Vec<f64> = (0..n)
        .map(|i| bound(xl, i, -BOUNDMAX, |v| v < -BOUNDMAX))
        .collect();
    let xu: Vec<f64> = (0..n)
        .map(|i| bound(xu, i, BOUNDMAX, |v| v > BOUNDMAX))
        .collect();
    if (0..n).any(|i| xu[i] - xl[i] < 2.0 * EPS) {
        return Outcome {
            x: x0.to_vec(),
            f: f64::NAN,
            evaluations: 0,
            status: ExitStatus::NoSpaceBetweenBounds,
        };
    }
    let mut x: Vec<f64> = moderatex(x0)
        .iter()
        .enumerate()
        .map(|(i, &v)| v.min(xu[i]).max(xl[i]))
        .collect();

    // defaults (PRIMA's bobyqa.f90)
    let mut rhobeg = match (settings.rhobeg, settings.rhoend) {
        (Some(r), _) => r,
        (None, Some(e)) if e.is_finite() && e > 0.0 => (10.0 * e).max(RHOBEG_DFT),
        _ => RHOBEG_DFT,
    };
    let mut rhoend = match settings.rhoend {
        Some(e) => e,
        None if rhobeg > 0.0 => EPS.max(((RHOEND_DFT / RHOBEG_DFT) * rhobeg).min(RHOEND_DFT)),
        None => RHOEND_DFT,
    };
    let mut ftarget = settings.ftarget.unwrap_or(-f64::MAX);
    let mut maxfun = settings.maxfun.unwrap_or(MAXFUN_DIM_DFT * n);
    let mut npt = settings.npt.unwrap_or(if maxfun >= n + 3 {
        (maxfun - 1).min(2 * n + 1)
    } else {
        2 * n + 1
    });
    let honour_x0 = settings.honour_x0.unwrap_or(match settings.rhobeg {
        Some(r) => !(r.is_finite() && r > 0.0),
        None => true,
    });
    let (eta1, eta2, gamma1, gamma2) = (0.1, 0.7, 0.5, 2.0);

    // preprocessing (PRIMA's preproc.f90, the parts that apply to BOBYQA)
    let min_maxfun = n + 3;
    if maxfun < min_maxfun {
        maxfun = if maxfun > 0 {
            min_maxfun
        } else {
            min_maxfun.max(10_000)
        };
    }
    if ftarget.is_nan() {
        ftarget = -f64::MAX;
    }
    // (n + 2)(n + 1) is even, so this is PRIMA's 2·npt > (n + 2)(n + 1) without overflowing for a huge npt
    if npt < n + 2 || npt >= maxfun || npt > (n + 2) * (n + 1) / 2 {
        npt = (maxfun - 1).min(2 * n + 1);
    }
    let min_range = (0..n).map(|i| xu[i] - xl[i]).fold(f64::INFINITY, f64::min);
    let rhobeg_default = EPS.max(RHOBEG_DFT.min(min_range / 4.0));
    let rhoend_default = EPS.max(((RHOEND_DFT / RHOBEG_DFT) * rhobeg_default).min(RHOEND_DFT));
    let rhobeg_in = rhobeg;
    if rhobeg > min_range / 2.0 {
        rhobeg = min_range / 4.0;
    }
    if !(rhobeg.is_finite() && rhobeg > 0.0) {
        rhobeg = rhobeg_default;
    }
    if !(rhoend.is_finite() && rhoend >= 0.0 && rhoend <= rhobeg) {
        rhoend = EPS.max(((RHOEND_DFT / RHOBEG_DFT) * rhobeg).min(rhoend_default));
    }
    if !honour_x0 {
        for i in 0..n {
            if x[i] <= xl[i] + 0.5 * rhobeg {
                x[i] = xl[i];
            } else if x[i] < xl[i] + rhobeg {
                x[i] = xl[i] + rhobeg;
            }
            if x[i] >= xu[i] - 0.5 * rhobeg {
                x[i] = xu[i];
            } else if x[i] > xu[i] - rhobeg {
                x[i] = xu[i] - rhobeg;
            }
        }
    }
    let lbx: Vec<bool> = (0..n)
        .map(|i| xl[i].is_finite() && x[i] - xl[i] <= EPS * 1f64.max(xl[i].abs()))
        .collect();
    let ubx: Vec<bool> = (0..n)
        .map(|i| xu[i].is_finite() && x[i] - xu[i] >= -EPS * 1f64.max(xu[i].abs()))
        .collect();
    for i in 0..n {
        if lbx[i] {
            x[i] = xl[i];
        }
        if ubx[i] {
            x[i] = xu[i];
        }
    }
    let mut smallest = rhobeg;
    for i in 0..n {
        if !lbx[i] {
            smallest = smallest.min(x[i] - xl[i]);
        }
        if !ubx[i] {
            smallest = smallest.min(xu[i] - x[i]);
        }
    }
    rhobeg = EPS.max(smallest);
    if rhobeg_in - rhobeg > EPS * 1f64.max(rhobeg_in) {
        rhoend = EPS.max(((rhoend / rhobeg_in) * rhobeg).min(rhoend));
    }
    rhobeg = rhobeg.max(EPS);
    rhoend = rhoend.max(EPS).min(rhobeg);

    let opts = Options {
        maxfun,
        npt,
        ftarget,
        rhobeg,
        rhoend,
        eta1,
        eta2,
        gamma1,
        gamma2,
    };
    bobyqb::bobyqb(&mut Calfun::new(&mut calfun), &opts, &xl, &xu, &x)
}

#[cfg(test)]
mod tests {
    use super::common::{Calfun, calvlag, quadinc};
    use super::initialize::{inith, initq, initxf};
    use super::linalg::Mat;
    use super::rescue::{RescueState, rescue};
    use std::ops::ControlFlow;

    /// Lagrange functions are 1 at their own point and 0 at the others: calvlag at XPT(:, k) - XOPT is e_k.
    fn assert_lagrange_property(kopt: usize, xpt: &Mat, bmat: &Mat, zmat: &Mat) {
        let (n, npt) = (xpt.rows, xpt.cols);
        for k in 0..npt {
            let d: Vec<f64> = (0..n).map(|i| xpt[(i, k)] - xpt[(i, kopt)]).collect();
            let vlag = calvlag(kopt, bmat, &d, xpt, zmat);
            for j in 0..npt {
                let expected = if j == k { 1.0 } else { 0.0 };
                assert!(
                    (vlag[j] - expected).abs() < 1e-8,
                    "l_{j}(x_{k}) = {} (expected {expected})",
                    vlag[j]
                );
            }
        }
    }

    /// The quadratic model interpolates the function values: Q(x_k) - Q(x_opt) = f_k - f_opt.
    fn assert_model_interpolates(
        kopt: usize,
        xpt: &Mat,
        fval: &[f64],
        gopt: &[f64],
        hq: &Mat,
        pq: &[f64],
    ) {
        let n = xpt.rows;
        for k in 0..xpt.cols {
            let d: Vec<f64> = (0..n).map(|i| xpt[(i, k)] - xpt[(i, kopt)]).collect();
            let q = quadinc(&d, xpt, gopt, pq, hq);
            let expected = fval[k] - fval[kopt];
            assert!(
                (q - expected).abs() < 1e-8 * expected.abs().max(1.0),
                "model mismatch at point {k}: {q} vs {expected}"
            );
        }
    }

    #[test]
    fn initialisation_and_rescue_keep_the_interpolation_conditions() {
        // an arbitrary quadratic, so the model is exact on any interpolation set
        let mut quadratic = |x: &[f64]| {
            ControlFlow::Continue(
                3.0 * x[0] * x[0] + x[0] * x[1] + 2.0 * x[1] * x[1] - x[2]
                    + 0.5 * x[2] * x[2]
                    + x[0] * x[2],
            )
        };
        let mut f = Calfun::new(&mut quadratic);
        let (n, npt) = (3usize, 8usize); // npt > 2n + 1 exercises the cross terms
        let (xl, xu) = (vec![-5.0; n], vec![5.0; n]);
        let init = initxf(
            &mut f,
            1000,
            npt,
            f64::NEG_INFINITY,
            0.5,
            &xl,
            &xu,
            &[0.3, -0.2, 1.0],
        );
        let (mut bmat, mut zmat) = inith(&init.ij, &init.xpt);
        let (mut gopt, mut hq, mut pq) = initq(&init.ij, &init.fval, &init.xpt);
        assert_lagrange_property(init.kopt, &init.xpt, &bmat, &zmat);
        assert_model_interpolates(init.kopt, &init.xpt, &init.fval, &gopt, &hq, &pq);

        let (mut kopt, mut nf, mut fval) = (init.kopt, init.nf, init.fval.clone());
        let (mut sl, mut su, mut xbase, mut xpt) = (
            init.sl.clone(),
            init.su.clone(),
            init.xbase.clone(),
            init.xpt.clone(),
        );
        let exit = rescue(
            &mut f,
            1000,
            0.4,
            f64::NEG_INFINITY,
            &xl,
            &xu,
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
        assert!(exit.is_none());
        assert!(nf > init.nf, "rescue evaluates new points");
        assert_lagrange_property(kopt, &xpt, &bmat, &zmat);
        assert_model_interpolates(kopt, &xpt, &fval, &gopt, &hq, &pq);
    }
}
