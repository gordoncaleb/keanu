//! Second-order differentiation: Hessian-vector products by forward-over-reverse (see the
//! [parent module](super)). Only the Hessian-vector product path uses it: the forward sweep of tangents, the
//! reverse sweep carrying each cotangent's derivative along them, and each op's second-order rule
//! ([`vjp_tangent`]) with the element-wise second derivatives it needs.

use super::*;

// ---------------------------------------------------------------------------------------------------------------
// second order: Hessian-vector products by forward-over-reverse
//
// For a scalar f of some latents, H·v is the derivative of the gradient along v. A forward sweep pushes the tangents
// of `b` directions through the deterministic nodes ([`forward_tangents`]). The reverse sweep then carries, for each
// node, the ordinary cotangent ḡ = ∂f/∂node (batch 1) and its derivative along each direction, ġ (batch b). An op
// with parents p and output y maps them to its parents as
//
//   p̄ = vjp(p, ḡ)      and      ṗ̄ = vjp(p, ġ) + Σⱼ ∂/∂pⱼ vjp(p, ḡ) · ṗⱼ,
//
// where the last term, [`vjp_tangent`], is the derivative of the op's vector-Jacobian product along the forward
// tangents of its parents: zero for linear ops, and the op's second derivatives contracted with ḡ otherwise.

/// Forward tangents along `b` directions: the seeds' batched tangents `[b, node shape...]`, pushed in id order
/// through the deterministic, differentiable nodes that depend on them. Probabilistic nodes are leaves (only seeded
/// ones have a tangent). None for nodes that don't depend on a seed.
pub(crate) fn forward_tangents(
    graph: &Graph,
    state: &State,
    seeds: impl IntoIterator<Item = (NodeId, DoubleTensor)>,
    b: usize,
) -> Result<Vec<Option<DoubleTensor>>> {
    let mut tangents: Vec<Option<DoubleTensor>> = vec![None; graph.len()];
    let mut first = graph.len();
    for (id, tangent) in seeds {
        graph.check_ids([&id])?;
        first = first.min(id.0);
        tangents[id.0] = Some(tangent.into_reshape(&batched(b, &graph[id].shape))?);
    }
    for i in first.saturating_add(1)..graph.len() {
        let id = NodeId(i);
        let node = &graph[id];
        if tangents[i].is_some() || node.is_probabilistic() || !is_differentiable(graph, id) {
            continue;
        }
        let parent_tangents: SmallVec<[Option<&DoubleTensor>; 4]> = node
            .parents
            .iter()
            .map(|p| tangents[p.0].as_ref())
            .collect();
        if parent_tangents.iter().all(Option::is_none) {
            continue;
        }
        let parent_values = node
            .parents
            .iter()
            .map(|&p| value(state, p))
            .collect::<Result<SmallVec<[&Value; 4]>>>()?;
        let out = value(state, id)?;
        let tangent = jvp(graph, id, &parent_values, out, &parent_tangents, b)?;
        drop(parent_tangents);
        tangents[i] = tangent;
    }
    Ok(tangents)
}

/// One seed of a second-order sweep: a node, its cotangent (shaped like the node, or `[1, node shape...]`) and the
/// derivatives of that cotangent along the directions (`[b, node shape...]`), either of which may be absent.
pub(crate) type SecondOrderSeed = (NodeId, Option<DoubleTensor>, Option<DoubleTensor>);

/// The second-order reverse sweep: from the seeds, visits differentiable ancestors in decreasing id order (stopping
/// at probabilistic nodes and at the nodes marked in `is_wrt`, as [`reverse_sweep`]), carrying each node's
/// cotangent `[1, shape...]` and its derivatives along the `b` directions `[b, shape...]`, given the forward
/// `tangents` of every node from [`forward_tangents`]. Emits both at the `is_wrt` nodes.
///
/// Fails with [`GraphError::NoSecondOrderRule`] at a node whose second derivatives are needed (it has a cotangent
/// and a parent with a tangent) and which has no second-order rule: a lambda.
pub(crate) fn second_order_sweep(
    graph: &Graph,
    state: &State,
    tangents: &[Option<DoubleTensor>],
    seeds: impl IntoIterator<Item = SecondOrderSeed>,
    b: usize,
    is_wrt: &[bool],
    mut emit: impl FnMut(NodeId, Option<DoubleTensor>, Option<DoubleTensor>) -> Result<()>,
) -> Result<()> {
    let n = graph.len();
    let mut cotangents: Vec<Option<DoubleTensor>> = vec![None; n];
    let mut derivatives: Vec<Option<DoubleTensor>> = vec![None; n];
    for (id, cotangent, derivative) in seeds {
        graph.check_ids([&id])?;
        let shape = &graph[id].shape;
        if let Some(c) = cotangent {
            accumulate(&mut cotangents[id.0], c.into_reshape(&batched(1, shape))?)?;
        }
        if let Some(d) = derivative {
            accumulate(&mut derivatives[id.0], d.into_reshape(&batched(b, shape))?)?;
        }
    }
    let tangent_of = |id: NodeId| tangents.get(id.0).and_then(Option::as_ref);
    for i in (0..n).rev() {
        let (cotangent, derivative) = (cotangents[i].take(), derivatives[i].take());
        if cotangent.is_none() && derivative.is_none() {
            continue;
        }
        let id = NodeId(i);
        if is_wrt.get(i).copied().unwrap_or(false) {
            emit(id, cotangent, derivative)?;
            continue;
        }
        let node = &graph[id];
        if node.is_probabilistic() || !is_differentiable(graph, id) {
            continue;
        }
        let parent_values = node
            .parents
            .iter()
            .map(|&p| value(state, p))
            .collect::<Result<SmallVec<[&Value; 4]>>>()?;
        let out = value(state, id)?;
        let parent_tangents: SmallVec<[Option<&DoubleTensor>; 4]> =
            node.parents.iter().map(|&p| tangent_of(p)).collect();
        let curvature = match &cotangent {
            Some(g) if parent_tangents.iter().any(Option::is_some) => Some(vjp_tangent(
                graph,
                id,
                &parent_values,
                out,
                g,
                &parent_tangents,
                tangent_of(id),
                b,
            )?),
            _ => None,
        };
        let first = cotangent
            .map(|g| vjp(graph, id, &parent_values, out, g, 1))
            .transpose()?;
        let second = derivative
            .map(|d| vjp(graph, id, &parent_values, out, d, b))
            .transpose()?;
        let (mut first, mut second, mut curvature) = (
            first.map(Vec::into_iter),
            second.map(Vec::into_iter),
            curvature.map(Vec::into_iter),
        );
        for &parent in &node.parents {
            let f = first.as_mut().and_then(Iterator::next).flatten();
            let s = second.as_mut().and_then(Iterator::next).flatten();
            let c = curvature.as_mut().and_then(Iterator::next).flatten();
            if !is_differentiable(graph, parent) {
                continue;
            }
            if let Some(f) = f {
                accumulate(&mut cotangents[parent.0], f)?;
            }
            if let Some(d) = add_opt(s, c)? {
                accumulate(&mut derivatives[parent.0], d)?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// second-order rules: the derivative of each op's vjp along the forward tangents (arms in the order of `vjp`)

/// The derivative of [`vjp`] for the cotangent `g` (`[1, out...]`) along the forward tangents `t` of the parents
/// (`[b, parent...]`, None for a parent that doesn't depend on the directions), holding `g` fixed: for each parent i,
/// `Σⱼ ∂/∂pⱼ vjpᵢ(p, g) · ṗⱼ`, shaped `[b, parent...]`, or None where it is zero. `yt` is the output's forward
/// tangent, which some rules use in place of recomputing it.
#[allow(clippy::too_many_arguments)]
fn vjp_tangent(
    graph: &Graph,
    id: NodeId,
    parents: &[&Value],
    out: &Value,
    g: &DoubleTensor,
    t: &[Option<&DoubleTensor>],
    yt: Option<&DoubleTensor>,
    b: usize,
) -> Result<Vec<Option<DoubleTensor>>> {
    let node = &graph[id];
    let out_shape = node.shape.to_vec();
    let y = f64v(out)?;
    let op = &node.op;
    let none = || vec![None; parents.len()];
    let out_tangent = || {
        yt.ok_or_else(|| GraphError::InvalidValue(format!("node #{} has no forward tangent", id.0)))
    };
    // the cotangent of a reduction to `out_shape`, laid out to broadcast against its input: [1, kept...]
    let g_kept = |shape: &[usize], axes: &[usize]| g.reshape(&batched(1, &kept_shape(shape, axes)));
    Ok(match op {
        // linear in the differentiable parents, or piecewise linear (selections, masks): the vjp doesn't change
        // along the tangents
        Op::Proxy
        | Op::Reshape(_)
        | Op::Permute(_)
        | Op::Transpose
        | Op::Broadcast(_)
        | Op::Slice(_)
        | Op::Concat(_)
        | Op::Stack(_)
        | Op::Take(_)
        | Op::Diag
        | Op::DiagPart
        | Op::Tril(_)
        | Op::Triu(_)
        | Op::FillTriangular { .. }
        | Op::TrianglePart { .. }
        | Op::Gather { .. }
        | Op::ScatterAdd { .. }
        | Op::Where
        | Op::Cpt { .. }
        | Op::Multiplexer
        | Op::SetWithMask
        | Op::ReplaceNan(_)
        | Op::Factor => none(),
        Op::Unary(u) => {
            // x̄ = ḡ f'(x): its derivative is ḡ f''(x) ẋ
            let Some(t0) = t[0] else { return Ok(none()) };
            let x = f64v(parents[0])?;
            let d2 = x.zip_with(y, |x, y| unary_second_derivative(*u, x, y))?;
            vec![Some(reduce_to(scale(t0, &d2)?.mul(g)?, x.shape())?)]
        }
        Op::Binary(bop) => {
            if matches!(
                bop,
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Max | BinaryOp::Min
            ) {
                return Ok(none());
            }
            // l̄ = ḡ f_l and r̄ = ḡ f_r: their derivatives are ḡ (f_ll l̇ + f_lr ṙ) and ḡ (f_rl l̇ + f_rr ṙ)
            let (l, r) = (f64v(parents[0])?, f64v(parents[1])?);
            let [ll, lr, rl, rr] = binary_second_partials(*bop, l, r, y)?;
            let along = |dl: &DoubleTensor, dr: &DoubleTensor, shape: &[usize]| {
                let from_left = t[0].map(|tl| scale(tl, dl)).transpose()?;
                let from_right = t[1].map(|tr| scale(tr, dr)).transpose()?;
                add_opt(from_left, from_right)?
                    .map(|d| reduce_to(fit(d, b, &out_shape)?.mul(g)?, shape))
                    .transpose()
            };
            vec![along(&ll, &lr, l.shape())?, along(&rl, &rr, r.shape())?]
        }
        Op::Reduce { kind, axes } => {
            let Some(t0) = t[0] else { return Ok(none()) };
            let x = f64v(parents[0])?;
            match kind {
                ReduceOp::Sum | ReduceOp::Mean | ReduceOp::Max | ReduceOp::Min => none(),
                ReduceOp::Product => {
                    // x̄ᵢ = ḡ × (the product of the group's other elements), differentiated along ẋ
                    let axes = product_axes(axes, x)?;
                    let others = product_of_others_tangent(x, t0, &axes)?;
                    vec![Some(g_kept(x.shape(), &axes)?.mul(&others)?)]
                }
                ReduceOp::StdDev => vec![Some(std_dev_vjp_tangent(x, t0, g)?)],
                ReduceOp::LogSumExp => {
                    // x̄ = ḡ softmax(x): its derivative is ḡ s (ẋ - Σ_group s ẋ)
                    let (axes, _) = reduced_axes(axes, x.rank(), x.shape())?;
                    let s = x.softmax_axes(&signed(&axes, 0))?;
                    let shifted: Vec<usize> = axes.iter().map(|a| a + 1).collect();
                    let mean = sum_keep_axes(&t0.mul(&s)?, &shifted)?;
                    let ds = t0.sub(&mean)?.mul(&s)?;
                    vec![Some(g_kept(x.shape(), &axes)?.mul(&ds)?)]
                }
            }
        }
        Op::Softmax { axis } => {
            // x̄ = y (ḡ - Σ ḡ y): its derivative is ẏ (ḡ - Σ ḡ y) - y Σ ḡ ẏ
            if t[0].is_none() {
                return Ok(none());
            }
            let yt = out_tangent()?;
            let axis = resolve_axis(*axis, y.rank())? + 1;
            let centred = g.sub(&sum_keep(&g.mul(y)?, axis)?)?;
            let correction = sum_keep(&yt.mul(g)?, axis)?.mul(y)?;
            vec![Some(yt.mul(&centred)?.sub(&correction)?)]
        }
        Op::LogSoftmax { axis } => {
            // x̄ = ḡ - softmax Σ ḡ, and softmax' = softmax ẏ: its derivative is -(softmax ẏ) Σ ḡ
            if t[0].is_none() {
                return Ok(none());
            }
            let yt = out_tangent()?;
            let axis = resolve_axis(*axis, y.rank())? + 1;
            vec![Some(sum_keep(g, axis)?.mul(&yt.mul(&y.exp())?)?.neg())]
        }
        Op::Standardize => {
            let Some(t0) = t[0] else { return Ok(none()) };
            vec![Some(standardize_vjp_tangent(f64v(parents[0])?, t0, g)?)]
        }
        Op::Cumulative { kind, axis } => {
            let Some(t0) = t[0] else { return Ok(none()) };
            let x = f64v(parents[0])?;
            let axis = resolve_axis(*axis, x.rank())?;
            match kind {
                CumulativeOp::CumSum => none(),
                CumulativeOp::CumProd => vec![Some(cumprod_reverse_tangent(
                    g,
                    x,
                    y,
                    t0,
                    out_tangent()?,
                    axis,
                )?)],
            }
        }
        Op::TensorMultiply {
            left_axes,
            right_axes,
        } => {
            // the vjp is bilinear in (G, the other operand): its derivative is the vjp with the other operand
            // replaced by its tangent, one direction at a time
            let (l, r) = (f64v(parents[0])?, f64v(parents[1])?);
            let c = Contraction::new(left_axes, right_axes, l.rank(), r.rank())?;
            let (kl, kr) = (c.kept_left.len(), c.kept_right.len());
            let g_kept_left: Vec<usize> = (1..1 + kl).collect();
            let g_kept_right: Vec<usize> = (1 + kl..1 + kl + kr).collect();
            let dl = t[1]
                .map(|tr| {
                    per_direction(tr, |rk| {
                        Ok(g.tensor_multiply(
                            rk,
                            &signed(&g_kept_right, 0),
                            &signed(&c.kept_right, 0),
                        )?
                        .permute(&Contraction::to_operand(&c.kept_left, &c.left, &c.right))?)
                    })
                })
                .transpose()?;
            let dr = t[0]
                .map(|tl| {
                    per_direction(tl, |lk| {
                        Ok(g.tensor_multiply(
                            lk,
                            &signed(&g_kept_left, 0),
                            &signed(&c.kept_left, 0),
                        )?
                        .permute(&Contraction::to_operand(&c.kept_right, &c.right, &c.left))?)
                    })
                })
                .transpose()?;
            vec![dl, dr]
        }
        Op::MatMul {
            transpose_left,
            transpose_right,
        } => {
            // as tensor_multiply: d op(A) = G op(Ḃ)ᵀ and d op(B) = op(Ȧ)ᵀ G
            let (a, bm) = (f64v(parents[0])?, f64v(parents[1])?);
            let (tl, tr_) = (*transpose_left, *transpose_right);
            let rank = out_shape.len();
            let da = t[1]
                .map(|tb| -> Result<DoubleTensor> {
                    let d_op_a = g.matmul_transposed(&*lift(tb, rank)?, false, !tr_)?;
                    reduce_to(if tl { tr(&d_op_a)? } else { d_op_a }, a.shape())
                })
                .transpose()?;
            let db = t[0]
                .map(|ta| -> Result<DoubleTensor> {
                    let d_op_b = lift(ta, rank)?.matmul_transposed(g, !tl, false)?;
                    reduce_to(if tr_ { tr(&d_op_b)? } else { d_op_b }, bm.shape())
                })
                .transpose()?;
            vec![da, db]
        }
        Op::Inverse => {
            // Ā = -Xᵀ G Xᵀ: its derivative is -(Ẋᵀ G Xᵀ + Xᵀ G Ẋᵀ)
            if t[0].is_none() {
                return Ok(none());
            }
            let (xt, xdt) = (tr(y)?, tr(out_tangent()?)?);
            let d = xdt
                .matmul(g)?
                .matmul(&xt)?
                .add(&xt.matmul(g)?.matmul(&xdt)?)?;
            vec![Some(d.neg())]
        }
        Op::Determinant => {
            // Ā = g det A⁻ᵀ: its derivative is g (ḋet A⁻ᵀ - det (A⁻¹ Ȧ A⁻¹)ᵀ)
            let Some(t0) = t[0] else { return Ok(none()) };
            let a = f64v(parents[0])?;
            let inv = a.inverse()?;
            let m = inv.matmul(&*lift(t0, a.rank())?)?.matmul(&inv)?;
            let det = y.reshape(&matrix_broadcast(1, &out_shape)[1..])?;
            let d = out_tangent()?
                .reshape(&matrix_broadcast(b, &out_shape))?
                .mul(&tr(&inv)?)?
                .sub(&tr(&m)?.mul(&det)?)?;
            vec![Some(d.mul(&g.reshape(&matrix_broadcast(1, &out_shape))?)?)]
        }
        Op::Cholesky => {
            // S = L⁻ᵀ phi(Lᵀ L̄) L⁻¹, read back to the lower triangle; with D = d(L⁻¹) = -L⁻¹ L̇ L⁻¹,
            // Ṡ = Dᵀ phi(Lᵀ L̄) L⁻¹ + L⁻ᵀ phi(L̇ᵀ L̄) L⁻¹ + L⁻ᵀ phi(Lᵀ L̄) D
            if t[0].is_none() {
                return Ok(none());
            }
            let (l, l_dot) = (y, out_tangent()?);
            let l_inv = l.inverse()?;
            let l_bar = g.tril(0)?;
            let inner = phi(&l.matmul_transposed(&l_bar, true, false)?)?;
            let d = l_inv.matmul(l_dot)?.matmul(&l_inv)?.neg();
            let first = tr(&d)?.matmul(&inner)?.matmul(&l_inv)?;
            let second = l_inv
                .matmul_transposed(
                    &phi(&l_dot.matmul_transposed(&l_bar, true, false)?)?,
                    true,
                    false,
                )?
                .matmul(&l_inv)?;
            let third = l_inv.matmul_transposed(&inner, true, false)?.matmul(&d)?;
            vec![Some(lower_from_symmetric(
                &first.add(&second)?.add(&third)?,
            )?)]
        }
        Op::CholeskyInverse => {
            // M = -X G X and L̄ = tril((M + Mᵀ) L): with Ṁ = -(Ẋ G X + X G Ẋ), its derivative is
            // tril((Ṁ + Ṁᵀ) L + (M + Mᵀ) L̇)
            let Some(t0) = t[0] else { return Ok(none()) };
            let x_dot = out_tangent()?;
            let l = f64v(parents[0])?.tril(0)?;
            let l_dot = lift(t0, out_shape.len())?.tril(0)?;
            let m = y.matmul(g)?.matmul(y)?.neg();
            let m_dot = x_dot
                .matmul(g)?
                .matmul(y)?
                .add(&y.matmul(g)?.matmul(x_dot)?)?
                .neg();
            let d = m_dot
                .add(&tr(&m_dot)?)?
                .matmul(&l)?
                .add(&m.add(&tr(&m)?)?.matmul(&l_dot)?)?;
            vec![Some(d.tril(0)?)]
        }
        Op::SolveTriangular { .. } | Op::Solve => {
            // B̄ = A⁻ᵀ X̄ and Ā = -B̄ Xᵀ, with A = op(tri(T)) for a triangular solve: d B̄ = -A⁻ᵀ Ȧᵀ B̄ and
            // d Ā = -(d B̄ Xᵀ + B̄ Ẋᵀ)
            let (a, rhs) = (f64v(parents[0])?, f64v(parents[1])?);
            let vector = rhs.rank() == 1;
            let x = as_columns(y, vector)?;
            let x_dot = as_columns(out_tangent()?, vector)?;
            let rank = x.rank();
            let rhs_bar = apply_solve(op, a, &*as_columns(g, vector)?, true)?;
            let rhs_bar_dot = t[0]
                .map(|ta| -> Result<DoubleTensor> {
                    let da = lift(ta, rank)?;
                    let da = match *op {
                        Op::SolveTriangular { lower, transpose } => {
                            let triangle = solve_triangle(&da, lower)?;
                            if transpose { tr(&triangle)? } else { triangle }
                        }
                        _ => da.into_owned(),
                    };
                    Ok(
                        apply_solve(op, a, &da.matmul_transposed(&rhs_bar, true, false)?, true)?
                            .neg(),
                    )
                })
                .transpose()?;
            let mut op_a_bar_dot = rhs_bar.matmul_transposed(&x_dot, false, true)?;
            if let Some(d) = &rhs_bar_dot {
                op_a_bar_dot = op_a_bar_dot.add(&d.matmul_transposed(&x, false, true)?)?;
            }
            let op_a_bar_dot = op_a_bar_dot.neg();
            let a_bar_dot = match *op {
                Op::SolveTriangular { lower, transpose } => solve_triangle(
                    &if transpose {
                        tr(&op_a_bar_dot)?
                    } else {
                        op_a_bar_dot
                    },
                    lower,
                )?,
                _ => op_a_bar_dot,
            };
            let rhs_bar_dot = rhs_bar_dot
                .map(|d| -> Result<DoubleTensor> {
                    let d = if vector {
                        d.into_reshape(&batched(b, &out_shape))?
                    } else {
                        d
                    };
                    reduce_to(d, rhs.shape())
                })
                .transpose()?;
            vec![Some(reduce_to(a_bar_dot, a.shape())?), rhs_bar_dot]
        }
        Op::LogAbsDeterminant => {
            // Ā = ḡ A⁻ᵀ: its derivative is -ḡ (A⁻¹ Ȧ A⁻¹)ᵀ
            let Some(t0) = t[0] else { return Ok(none()) };
            let a = f64v(parents[0])?;
            let inv = a.inverse()?;
            let m = inv.matmul(&*lift(t0, a.rank())?)?.matmul(&inv)?;
            vec![Some(
                tr(&m)?
                    .mul(&g.reshape(&matrix_broadcast(1, &out_shape))?)?
                    .neg(),
            )]
        }
        Op::Eigvalsh => {
            // S̄ = Q diag(λ̄) Qᵀ with Q̇ = Q (F ∘ (Qᵀ Ṡ Q)): its derivative is Q̇ diag(λ̄) Qᵀ + Q diag(λ̄) Q̇ᵀ
            let Some(t0) = t[0] else { return Ok(none()) };
            let a = f64v(parents[0])?;
            let (lambda, q) = a.eigh()?;
            let ds = symmetric_from_lower(&*lift(t0, a.rank())?)?;
            let rotated = q.matmul_transposed(&ds, true, false)?.matmul(&q)?;
            let q_dot = q.matmul(&rotated.mul(&eigen_gaps(&lambda)?)?)?;
            let mut shape = batched(1, &out_shape);
            shape.insert(shape.len() - 1, 1);
            let w = q_dot
                .mul(&g.reshape(&shape)?)?
                .matmul_transposed(&q, false, true)?;
            vec![Some(lower_from_symmetric(&w.add(&tr(&w)?)?)?)]
        }
        Op::EighVectors => {
            // S̄ = Q (F ∘ (Qᵀ Q̄)) Qᵀ: its derivative is Q̇ (F ∘ C) Qᵀ + Q (Ḟ ∘ C + F ∘ Ċ) Qᵀ + Q (F ∘ C) Q̇ᵀ with
            // C = Qᵀ Q̄, Ċ = Q̇ᵀ Q̄ and Ḟᵢⱼ = -Fᵢⱼ² (λ̇ⱼ - λ̇ᵢ), λ̇ᵢ = (Qᵀ Ṡ Q)ᵢᵢ
            let Some(t0) = t[0] else { return Ok(none()) };
            let a = f64v(parents[0])?;
            let (lambda, _) = a.eigh()?;
            let (q, q_dot) = (y, out_tangent()?);
            let f = eigen_gaps(&lambda)?;
            let ds = symmetric_from_lower(&*lift(t0, a.rank())?)?;
            let rotated = q.matmul_transposed(&ds, true, false)?.matmul(q)?;
            let lambda_dot = rotated.diag_part()?;
            let mut row: Dims = lambda_dot.shape().iter().map(|&d| d as isize).collect();
            let mut column = row.clone();
            row.insert(row.len() - 1, 1);
            column.push(1);
            let gaps = lambda_dot
                .reshape(&row)?
                .sub(&lambda_dot.reshape(&column)?)?;
            let f_dot = f.mul(&f)?.mul(&gaps)?.neg();
            let c = q.matmul_transposed(g, true, false)?;
            let c_dot = q_dot.matmul_transposed(g, true, false)?;
            let inner = c.mul(&f)?;
            let inner_dot = f_dot.mul(&c)?.add(&c_dot.mul(&f)?)?;
            let s_bar_dot = q_dot
                .matmul(&inner)?
                .matmul_transposed(q, false, true)?
                .add(&q.matmul(&inner_dot)?.matmul_transposed(q, false, true)?)?
                .add(&q.matmul(&inner)?.matmul_transposed(q_dot, false, true)?)?;
            vec![Some(lower_from_symmetric(&s_bar_dot)?)]
        }
        Op::MatrixExp => {
            // Ā = L(Aᵀ, G): its derivative along Ȧ is the second Fréchet derivative L²(Aᵀ; G, Ȧᵀ)
            let Some(t0) = t[0] else { return Ok(none()) };
            let at = tr(f64v(parents[0])?)?;
            vec![Some(at.matrix_exp_frechet_second(g, &tr(t0)?)?)]
        }
        Op::Scan(body) => body.vjp_tangent(parents, y, g, t, yt, b)?,
        Op::LogDensity(kind) => crate::op::log_density::vjp_tangent(
            kind, parents, g, t, b, &out_shape,
        )
        .map_err(|e| match e {
            GraphError::NoSecondOrderRule(m) => {
                GraphError::NoSecondOrderRule(format!("{m} of node {}", graph.describe(id)))
            }
            other => other,
        })?,
        Op::Lambda(lambda) => {
            return Err(GraphError::NoSecondOrderRule(format!(
                "the lambda {:?} (node {})",
                lambda.name,
                graph.describe(id)
            )));
        }
        _ => none(),
    })
}

/// The shape of a reduction's input with the reduced axes kept as length 1.
fn kept_shape(shape: &[usize], axes: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .enumerate()
        .map(|(i, &d)| if axes.contains(&i) { 1 } else { d })
        .collect()
}

/// Sums a tensor over several axes, keeping them with length 1.
fn sum_keep_axes(t: &DoubleTensor, axes: &[usize]) -> Result<DoubleTensor> {
    if axes.is_empty() {
        return Ok(t.clone());
    }
    let kept: Dims = kept_shape(t.shape(), axes)
        .into_iter()
        .map(|d| d as isize)
        .collect();
    let signed: Vec<isize> = axes.iter().map(|&a| a as isize).collect();
    Ok(t.sum_axes(&signed)?.into_reshape(&kept)?)
}

/// The shape `[b, shape..., 1, 1]`, for a per-matrix quantity broadcast against matrices.
fn matrix_broadcast(b: usize, shape: &[usize]) -> Dims {
    let mut dims = batched(b, shape);
    dims.extend([1, 1]);
    dims
}

/// Applies `f` to each direction of a batched tensor `[b, shape...]` (given as `[shape...]`), each giving
/// `[1, ...]`, and joins the results along the batch axis.
fn per_direction(
    t: &DoubleTensor,
    mut f: impl FnMut(&DoubleTensor) -> Result<DoubleTensor>,
) -> Result<DoubleTensor> {
    let b = t.shape()[0];
    let inner = &t.shape()[1..];
    let n: usize = inner.iter().product();
    if b == 0 {
        let probe = f(&DoubleTensor::zeros(inner))?;
        return Ok(DoubleTensor::zeros(&batched_shape(0, &probe.shape()[1..])));
    }
    let parts = (0..b)
        .map(|k| {
            f(&DoubleTensor::new(
                inner,
                t.as_slice()[k * n..(k + 1) * n].to_vec(),
            )?)
        })
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<&DoubleTensor> = parts.iter().collect();
    Ok(DoubleTensor::concat(0, &refs)?)
}

/// For a product over `axes` (distinct and in range) of x: the derivative along each direction of `xt`
/// (`[b, x...]`) of each element's product of the other elements of its group, from exclusive prefix and suffix
/// products on dual numbers, so exact where elements are zero.
fn product_of_others_tangent(
    x: &DoubleTensor,
    xt: &DoubleTensor,
    axes: &[usize],
) -> Result<DoubleTensor> {
    let rank = x.rank();
    let permutation: Vec<usize> = (0..rank)
        .filter(|a| !axes.contains(a))
        .chain(axes.iter().copied())
        .collect();
    let batched_permutation: Vec<usize> = std::iter::once(0)
        .chain(permutation.iter().map(|p| p + 1))
        .collect();
    let group: usize = axes.iter().map(|&a| x.shape()[a]).product();
    let px = x.permute(&permutation)?;
    let pt = xt.permute(&batched_permutation)?;
    let (xs, ts) = (px.as_slice(), pt.as_slice());
    let n = xs.len();
    let mut out = vec![0.0; ts.len()];
    let mut prefix = vec![(0.0, 0.0); group];
    if group > 0 && n > 0 {
        for (tk, ok) in ts.chunks(n).zip(out.chunks_mut(n)) {
            for ((x, t), o) in xs
                .chunks(group)
                .zip(tk.chunks(group))
                .zip(ok.chunks_mut(group))
            {
                // exclusive prefix products (P, Ṗ) with Pᵢ₊₁ = Pᵢ xᵢ, then the suffix (S, Ṡ) on the way back
                let (mut p, mut dp) = (1.0, 0.0);
                for i in 0..group {
                    prefix[i] = (p, dp);
                    (p, dp) = (p * x[i], dp * x[i] + p * t[i]);
                }
                let (mut s, mut ds) = (1.0, 0.0);
                for i in (0..group).rev() {
                    let (p, dp) = prefix[i];
                    o[i] = dp * s + p * ds;
                    (s, ds) = (s * x[i], ds * x[i] + s * t[i]);
                }
            }
        }
    }
    let mut inverse = vec![0; rank + 1];
    for (i, &p) in batched_permutation.iter().enumerate() {
        inverse[p] = i;
    }
    Ok(DoubleTensor::new(pt.shape(), out)?.permute(&inverse)?)
}

/// The derivative of the standard deviation's vjp, x̄ⱼ = ḡ cⱼ / ((n - 1) s) with c = x - mean(x), along each
/// direction of `xt`: ḡ (ċⱼ / ((n - 1) s) - cⱼ ṡ / ((n - 1) s²)), with ṡ = c·ẋ / ((n - 1) s).
fn std_dev_vjp_tangent(
    x: &DoubleTensor,
    xt: &DoubleTensor,
    g: &DoubleTensor,
) -> Result<DoubleTensor> {
    let n = x.len();
    let (s, mean) = (x.std_dev(), x.mean());
    let g = g.as_slice()[0];
    let c: Vec<f64> = x.as_slice().iter().map(|v| v - mean).collect();
    let scale = (n as f64 - 1.0) * s;
    let mut out = vec![0.0; xt.len()];
    for (t, o) in xt.as_slice().chunks(n.max(1)).zip(out.chunks_mut(n.max(1))) {
        let t_mean = t.iter().sum::<f64>() / n as f64;
        let s_dot = c.iter().zip(t).map(|(c, t)| c * t).sum::<f64>() / scale;
        for ((o, &t), &c) in o.iter_mut().zip(t).zip(&c) {
            *o = g * ((t - t_mean) / scale - c * s_dot / (scale * s));
        }
    }
    Ok(DoubleTensor::new(xt.shape(), out)?)
}

/// The derivative of standardize's vjp, x̄ⱼ = (ḡⱼ - mean ḡ) / s - cⱼ (c·ḡ) / ((n - 1) s³), along each direction of
/// `xt`, with ċ = ẋ - mean(ẋ) and ṡ = c·ẋ / ((n - 1) s).
fn standardize_vjp_tangent(
    x: &DoubleTensor,
    xt: &DoubleTensor,
    g: &DoubleTensor,
) -> Result<DoubleTensor> {
    let n = x.len();
    let (s, mean) = (x.std_dev(), x.mean());
    let m = n as f64 - 1.0;
    let g = g.as_slice();
    let g_mean = g.iter().sum::<f64>() / n as f64;
    let c: Vec<f64> = x.as_slice().iter().map(|v| v - mean).collect();
    let cg: f64 = c.iter().zip(g).map(|(c, g)| c * g).sum();
    let s3 = s * s * s;
    let mut out = vec![0.0; xt.len()];
    for (t, o) in xt.as_slice().chunks(n.max(1)).zip(out.chunks_mut(n.max(1))) {
        let t_mean = t.iter().sum::<f64>() / n as f64;
        let s_dot = c.iter().zip(t).map(|(c, t)| c * t).sum::<f64>() / (m * s);
        let c_dot_g: f64 = t.iter().zip(g).map(|(t, g)| (t - t_mean) * g).sum();
        for j in 0..n {
            let c_dot = t[j] - t_mean;
            o[j] = -(g[j] - g_mean) * s_dot / (s * s)
                - c_dot * cg / (m * s3)
                - c[j] * c_dot_g / (m * s3)
                + 3.0 * c[j] * cg * s_dot / (m * s3 * s);
        }
    }
    Ok(DoubleTensor::new(xt.shape(), out)?)
}

/// The derivative of cumprod's vjp (see [`cumprod_reverse`]: x̄ᵢ = yᵢ₋₁ aᵢ with aᵢ = ḡᵢ + xᵢ₊₁ aᵢ₊₁) along each
/// direction of `xt`, given the output's tangent `yt`: ẏᵢ₋₁ aᵢ + yᵢ₋₁ ȧᵢ, with ȧᵢ = ẋᵢ₊₁ aᵢ₊₁ + xᵢ₊₁ ȧᵢ₊₁.
fn cumprod_reverse_tangent(
    g: &DoubleTensor,
    x: &DoubleTensor,
    y: &DoubleTensor,
    xt: &DoubleTensor,
    yt: &DoubleTensor,
    axis: usize,
) -> Result<DoubleTensor> {
    let (outer, n, stride) = lanes(x.shape(), axis);
    let per = x.len();
    let (xs, ys, gs, xts, yts) = (
        x.as_slice(),
        y.as_slice(),
        g.as_slice(),
        xt.as_slice(),
        yt.as_slice(),
    );
    let mut data = vec![0.0; xts.len()];
    for k in 0..xt.shape()[0] {
        let base_k = k * per;
        for o in 0..outer {
            for i in 0..stride {
                let base = o * n * stride + i;
                let (mut a, mut da) = (0.0, 0.0);
                for m in (0..n).rev() {
                    let xi = base + m * stride;
                    if m + 1 < n {
                        let next = xi + stride;
                        da = xts[base_k + next] * a + xs[next] * da;
                        a *= xs[next];
                    }
                    a += gs[xi];
                    data[base_k + xi] = if m == 0 {
                        da
                    } else {
                        yts[base_k + xi - stride] * a + ys[xi - stride] * da
                    };
                }
            }
        }
    }
    Ok(DoubleTensor::new(xt.shape(), data)?)
}

/// d² y / d x² for a differentiable unary op at x with y = op(x) (zero for the piecewise linear ones).
fn unary_second_derivative(u: UnaryOp, x: f64, y: f64) -> f64 {
    use std::f64::consts::{FRAC_2_SQRT_PI, LN_2, LN_10};
    match u {
        UnaryOp::Abs | UnaryOp::Neg => 0.0,
        UnaryOp::Reciprocal => 2.0 * y * y * y,
        UnaryOp::Exp => y,
        UnaryOp::Exp2 => y * LN_2 * LN_2,
        UnaryOp::ExpM1 => y + 1.0,
        UnaryOp::Ln => -1.0 / (x * x),
        UnaryOp::Ln1p => -1.0 / ((1.0 + x) * (1.0 + x)),
        UnaryOp::Log2 => -1.0 / (x * x * LN_2),
        UnaryOp::Log10 => -1.0 / (x * x * LN_10),
        UnaryOp::Sqrt => -0.25 / (y * y * y),
        UnaryOp::Sin => -x.sin(),
        UnaryOp::Cos => -x.cos(),
        UnaryOp::Tan => {
            let c = x.cos();
            2.0 * x.sin() / (c * c * c)
        }
        UnaryOp::Asin => x * (1.0 - x * x).powf(-1.5),
        UnaryOp::Acos => -x * (1.0 - x * x).powf(-1.5),
        UnaryOp::Atan => -2.0 * x / ((1.0 + x * x) * (1.0 + x * x)),
        UnaryOp::Sinh => x.sinh(),
        UnaryOp::Cosh => x.cosh(),
        UnaryOp::Tanh => -2.0 * y * (1.0 - y * y),
        UnaryOp::Asinh => -x * (x * x + 1.0).powf(-1.5),
        UnaryOp::Acosh => -x * (x * x - 1.0).powf(-1.5),
        UnaryOp::Atanh => 2.0 * x / ((1.0 - x * x) * (1.0 - x * x)),
        UnaryOp::Sigmoid => y * (1.0 - y) * (1.0 - 2.0 * y),
        UnaryOp::LnGamma => special::trigamma(x),
        UnaryOp::Digamma => special::tetragamma(x),
        UnaryOp::Trigamma => special::pentagamma(x),
        UnaryOp::Erf => -2.0 * x * FRAC_2_SQRT_PI * (-x * x).exp(),
        UnaryOp::Erfc => 2.0 * x * FRAC_2_SQRT_PI * (-x * x).exp(),
        // with d = f'(x): f'' = 2 y d² for erf⁻¹ and y d² for Φ⁻¹
        UnaryOp::ErfInv => {
            let d = special_unary_derivative(u, x, y);
            2.0 * y * d * d
        }
        UnaryOp::Ndtri => {
            let d = special_unary_derivative(u, x, y);
            y * d * d
        }
        // with r = φ(x) / Φ(x): r' = -r (x + r)
        UnaryOp::LogNdtr => {
            let r = special_unary_derivative(u, x, y);
            -r * (x + r)
        }
        UnaryOp::Softplus => {
            let s = 1.0 / (1.0 + (-x).exp());
            s * (1.0 - s)
        }
        UnaryOp::Logit => (2.0 * x - 1.0) / (x * x * (1.0 - x) * (1.0 - x)),
        UnaryOp::Log1mExp => {
            let e = (-x).exp_m1();
            -(-x).exp() / (e * e)
        }
        UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round | UnaryOp::Sign => 0.0,
    }
}

/// The second partial derivatives `[f_ll, f_lr, f_rl, f_rr]` of an element-wise binary op at its operands, each
/// with the output's shape: `f_lr` is ∂(∂y/∂l)/∂r and `f_rl` is ∂(∂y/∂r)/∂l, which differ only where the first
/// derivatives have special values (`safe_ln_times`).
fn binary_second_partials(
    op: BinaryOp,
    l: &DoubleTensor,
    r: &DoubleTensor,
    y: &DoubleTensor,
) -> Result<[DoubleTensor; 4]> {
    let shape = y.shape();
    let (l, r) = (l.broadcast_to(shape)?, r.broadcast_to(shape)?);
    let n = y.len();
    let mut parts: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::with_capacity(n));
    for ((&l, &r), &y) in l.as_slice().iter().zip(r.as_slice()).zip(y.as_slice()) {
        for (part, value) in parts
            .iter_mut()
            .zip(binary_second_partials_scalar(op, l, r, y))
        {
            part.push(value);
        }
    }
    let [ll, lr, rl, rr] = parts;
    Ok([
        DoubleTensor::new(shape, ll)?,
        DoubleTensor::new(shape, lr)?,
        DoubleTensor::new(shape, rl)?,
        DoubleTensor::new(shape, rr)?,
    ])
}

/// `[f_ll, f_lr, f_rl, f_rr]` at (l, r) with y = op(l, r), consistent with [`binary_derivatives`] (zero for the
/// piecewise linear ops).
fn binary_second_partials_scalar(op: BinaryOp, l: f64, r: f64, y: f64) -> [f64; 4] {
    match op {
        BinaryOp::Mul => [0.0, 1.0, 1.0, 0.0],
        BinaryOp::Div => {
            let lr = -1.0 / (r * r);
            [0.0, lr, lr, 2.0 * l / (r * r * r)]
        }
        BinaryOp::Pow => {
            let lr = l.powf(r - 1.0) * (1.0 + r * l.ln());
            [r * (r - 1.0) * l.powf(r - 2.0), lr, lr, l.ln() * l.ln() * y]
        }
        // left is x and right is y: atan2(y, x), with s = x² + y²
        BinaryOp::Atan2 => {
            let s2 = (l * l + r * r) * (l * l + r * r);
            let lr = (r * r - l * l) / s2;
            [2.0 * l * r / s2, lr, lr, -2.0 * l * r / s2]
        }
        // with the weights wₗ = e^(l - y) and wᵣ = e^(r - y): f_ll = f_rr = wₗ wᵣ = -f_lr
        BinaryOp::LogAddExp => {
            let w = (l - y).exp() * (r - y).exp();
            [w, -w, -w, w]
        }
        BinaryOp::LogAddExp2 => {
            let w = std::f64::consts::LN_2 * (l - y).exp2() * (r - y).exp2();
            [w, -w, -w, w]
        }
        // y ln x: d/dx = y / x and d/dy = ln x (NaN where y == 0, as the first derivative)
        BinaryOp::SafeLnTimes => {
            let rl = if r == 0.0 { f64::NAN } else { 1.0 / l };
            [-r / (l * l), 1.0 / l, rl, 0.0]
        }
        BinaryOp::LogBeta => {
            let both = special::trigamma(l + r);
            [
                special::trigamma(l) - both,
                -both,
                -both,
                special::trigamma(r) - both,
            ]
        }
        _ => [0.0; 4],
    }
}
