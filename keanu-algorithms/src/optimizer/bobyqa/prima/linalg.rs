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

//! Small dense helpers used by the BOBYQA translation: a column-major matrix (so Fortran's `A(i, j)` maps to
//! `a[(i, j)]` with 0-based indices) and vector operations.

use std::ops::{Index, IndexMut};

/// A temporary vector, inline for the small problems Keanu typically solves (Keanu change: avoids allocations).
pub(crate) type Vector = smallvec::SmallVec<[f64; 16]>;

/// A dense column-major matrix.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Mat {
    pub rows: usize,
    pub cols: usize,
    data: Vec<f64>,
}

impl Mat {
    pub(crate) fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub(crate) fn col(&self, j: usize) -> &[f64] {
        &self.data[j * self.rows..(j + 1) * self.rows]
    }

    pub(crate) fn col_mut(&mut self, j: usize) -> &mut [f64] {
        &mut self.data[j * self.rows..(j + 1) * self.rows]
    }

    pub(crate) fn set_col(&mut self, j: usize, v: &[f64]) {
        self.col_mut(j).copy_from_slice(v);
    }

    pub(crate) fn row(&self, i: usize) -> Vector {
        (0..self.cols).map(|j| self[(i, j)]).collect()
    }

    pub(crate) fn swap_cols(&mut self, a: usize, b: usize) {
        if a != b {
            for i in 0..self.rows {
                self.data.swap(a * self.rows + i, b * self.rows + i);
            }
        }
    }

    pub(crate) fn swap_rows(&mut self, a: usize, b: usize) {
        if a != b {
            for j in 0..self.cols {
                self.data.swap(j * self.rows + a, j * self.rows + b);
            }
        }
    }

    pub(crate) fn fill(&mut self, value: f64) {
        self.data.iter_mut().for_each(|x| *x = value);
    }

    pub(crate) fn all_finite(&self) -> bool {
        self.data.iter().all(|x| x.is_finite())
    }

    pub(crate) fn max_abs(&self) -> f64 {
        self.data
            .iter()
            .fold(0.0, |m, x| if x.abs() > m { x.abs() } else { m })
    }

    /// self * v (length cols) -> length rows
    pub(crate) fn mul_vec(&self, v: &[f64]) -> Vector {
        let mut out: Vector = smallvec::smallvec![0.0; self.rows];
        for j in 0..self.cols {
            let vj = v[j];
            for (o, &a) in out.iter_mut().zip(self.col(j)) {
                *o += a * vj;
            }
        }
        out
    }

    /// Columns [0, ncols) of self times v (length ncols).
    pub(crate) fn mul_vec_cols(&self, ncols: usize, v: &[f64]) -> Vector {
        let mut out: Vector = smallvec::smallvec![0.0; self.rows];
        for (j, &vj) in v.iter().enumerate().take(ncols) {
            for (o, &a) in out.iter_mut().zip(self.col(j)) {
                *o += a * vj;
            }
        }
        out
    }

    /// selfᵀ * v (length rows) -> length cols
    pub(crate) fn tmul_vec(&self, v: &[f64]) -> Vector {
        (0..self.cols).map(|j| dot(self.col(j), v)).collect()
    }

    /// Columns [0, ncols) of selfᵀ times v (length rows) -> length ncols
    pub(crate) fn tmul_vec_cols(&self, ncols: usize, v: &[f64]) -> Vector {
        (0..ncols).map(|j| dot(self.col(j), v)).collect()
    }

    /// Copies the lower triangle of the square block starting at column `start` into its upper triangle
    /// (PRIMA's `symmetrize`).
    pub(crate) fn symmetrize_block(&mut self, start: usize) {
        let n = self.rows;
        for j in 0..n {
            for i in 0..j {
                let v = self[(j, start + i)];
                self[(i, start + j)] = v;
            }
        }
    }
}

impl Index<(usize, usize)> for Mat {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[j * self.rows + i]
    }
}

impl IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[j * self.rows + i]
    }
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

pub(crate) fn sum_sq(a: &[f64]) -> f64 {
    dot(a, a)
}

pub(crate) fn sum_abs(a: &[f64]) -> f64 {
    a.iter().map(|x| x.abs()).sum()
}

/// Fortran `MAXLOC`: index of the first largest element, ignoring NaN (0 if every element is NaN).
pub(crate) fn maxloc(a: &[f64]) -> usize {
    let mut best: Option<usize> = None;
    for (i, &v) in a.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        match best {
            Some(b) if !(v > a[b]) => {}
            _ => best = Some(i),
        }
    }
    best.unwrap_or(0)
}

/// Fortran `MINLOC`: index of the first smallest element, ignoring NaN (0 if every element is NaN).
pub(crate) fn minloc(a: &[f64]) -> usize {
    let mut best: Option<usize> = None;
    for (i, &v) in a.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        match best {
            Some(b) if !(v < a[b]) => {}
            _ => best = Some(i),
        }
    }
    best.unwrap_or(0)
}

/// Fortran `MAX(A, B)` (IEEE maxNum: a NaN argument is ignored, as gfortran does).
pub(crate) fn fmax(a: f64, b: f64) -> f64 {
    a.max(b)
}

/// Fortran `MIN(A, B)` (IEEE minNum).
pub(crate) fn fmin(a: f64, b: f64) -> f64 {
    a.min(b)
}

/// Fortran `SIGN(A, B)`: |A| with the sign of B (including the sign of zero).
pub(crate) fn sign(a: f64, b: f64) -> f64 {
    a.abs().copysign(b)
}
