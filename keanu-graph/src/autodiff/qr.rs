//! Differential rules for a thin QR factorization with a fixed diagonal sign convention.

use super::tr;
use crate::error::Result;
use keanu_tensor::{DoubleTensor, TensorError};

fn split(factors: &DoubleTensor) -> Result<(DoubleTensor, DoubleTensor)> {
    let [.., packed_rows, cols] = factors.shape() else {
        unreachable!("QR factors have rank at least two");
    };
    let rows = packed_rows - cols;
    Ok((
        factors.slice(&format!("...,:{rows},:"))?,
        factors.slice(&format!("...,{rows}:,:"))?,
    ))
}

fn full_rank(r: &DoubleTensor, rows: usize) -> Result<()> {
    let n = r.shape()[r.rank() - 1];
    for matrix in r.as_slice().chunks_exact(n.saturating_mul(n).max(1)) {
        let mut diagonal = (0..n).map(|i| matrix[i * n + i].abs());
        let scale = diagonal.clone().fold(0.0_f64, f64::max);
        if diagonal.any(|v| v <= scale * (rows as f64 * f64::EPSILON)) {
            return Err(TensorError::SingularMatrix.into());
        }
    }
    Ok(())
}

/// With B = dA R⁻¹ and C = Qᵀ B, dQ = B − Q U and dR = U R, where U is C minus
/// the skew matrix whose strict lower triangle is C's strict lower triangle.
pub(super) fn jvp(factors: &DoubleTensor, da: &DoubleTensor) -> Result<DoubleTensor> {
    let (q, r) = split(factors)?;
    full_rank(&r, q.shape()[q.rank() - 2])?;
    let b = tr(&r.solve_triangular(&tr(da)?, false, true)?)?;
    let c = q.matmul_transposed(&b, true, false)?;
    let lower = c.tril(-1)?;
    let upper = c.triu(0)?.add(&tr(&lower)?)?;
    let dq = b.sub(&q.matmul(&upper)?)?;
    let dr = upper.matmul(&r)?.triu(0)?;
    Ok(DoubleTensor::concat(-2, &[&dq, &dr])?)
}

fn copy_upper(x: &DoubleTensor) -> Result<DoubleTensor> {
    Ok(x.triu(0)?.add(&tr(&x.triu(1)?)?)?)
}

fn solve_adjoint(
    q: &DoubleTensor,
    r: &DoubleTensor,
    gq: &DoubleTensor,
    gr: &DoubleTensor,
) -> Result<(DoubleTensor, DoubleTensor)> {
    let m = copy_upper(
        &gr.matmul_transposed(r, false, true)?
            .sub(&q.matmul_transposed(gq, true, false)?)?,
    )?;
    // Āᵀ = R⁻¹ (Q̄ + Q M)ᵀ.
    let gradient_t = r.solve_triangular(&tr(&gq.add(&q.matmul(&m)?)?)?, false, false)?;
    Ok((m, gradient_t))
}

pub(super) fn vjp(factors: &DoubleTensor, g: &DoubleTensor) -> Result<DoubleTensor> {
    let (q, r) = split(factors)?;
    full_rank(&r, q.shape()[q.rank() - 2])?;
    let (gq, gr) = split(g)?;
    let (_, gradient_t) = solve_adjoint(&q, &r, &gq, &gr)?;
    tr(&gradient_t)
}

/// Differentiate the adjoint solve with Q̄ and R̄ fixed. The reverse sweep handles their tangents.
pub(super) fn vjp_tangent(
    factors: &DoubleTensor,
    g: &DoubleTensor,
    tangent: &DoubleTensor,
) -> Result<DoubleTensor> {
    let (q, r) = split(factors)?;
    full_rank(&r, q.shape()[q.rank() - 2])?;
    let (gq, gr) = split(g)?;
    let (dq, dr) = split(tangent)?;
    let (m, gradient_t) = solve_adjoint(&q, &r, &gq, &gr)?;
    let dm = copy_upper(
        &gr.matmul_transposed(&dr, false, true)?
            .sub(&dq.matmul_transposed(&gq, true, false)?)?,
    )?;
    let dw = dq.matmul(&m)?.add(&q.matmul(&dm)?)?;
    let rhs = tr(&dw)?.sub(&dr.matmul(&gradient_t)?)?;
    tr(&r.solve_triangular(&rhs, false, false)?)
}
