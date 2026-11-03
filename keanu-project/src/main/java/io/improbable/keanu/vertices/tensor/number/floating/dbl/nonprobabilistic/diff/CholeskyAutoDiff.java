package io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

/**
 * Auto diff for the Cholesky decomposition and the Cholesky inverse.
 * <p>
 * Both operations only read the lower triangle of their input (as LAPACK's potrf/potri do), so the
 * input A is effectively sym(A) = tril(A) + strictlyLower(A)^T and the derivatives below account for that.
 * All operations work on the trailing two dimensions so leading wrt/of and batch dimensions broadcast.
 * <p>
 * Credit: Iain Murray, "Differentiation of the Cholesky decomposition", 2016.
 */
public final class CholeskyAutoDiff {

    private CholeskyAutoDiff() {
    }

    /**
     * L = chol(A). dL = L * phi(L^-1 * dA * L^-T)
     */
    public static DoubleTensor forwardCholesky(DoubleTensor dInput, DoubleTensor L) {
        final DoubleTensor Linv = L.matrixInverse();
        final DoubleTensor dA = symmetricFromLower(dInput);
        final DoubleTensor inner = Linv.matrixMultiply(dA).matrixMultiply(Linv, false, true);
        return L.matrixMultiply(phi(inner));
    }

    /**
     * Given the gradient with respect to L, S = L^-T * phi(L^T * Lbar) * L^-1 and the gradient with respect to
     * the lower triangle that was read is tril(S + S^T) - diag(S).
     */
    public static DoubleTensor reverseCholesky(DoubleTensor dOutput, DoubleTensor L) {
        final DoubleTensor Linv = L.matrixInverse();
        // only the lower triangle of L depends on A, the upper triangle is always zero
        final DoubleTensor Lbar = dOutput.triLower(0);
        final DoubleTensor S = Linv.matrixMultiply(phi(L.matrixMultiply(Lbar, true, false)), true, false).matrixMultiply(Linv);
        return lowerGradientOfSymmetric(S);
    }

    /**
     * Y = tril(X) + strictlyUpper(input) where X = (L L^T)^-1 and L = tril(input).
     * dX = -X (dL L^T + L dL^T) X
     */
    public static DoubleTensor forwardCholeskyInverse(DoubleTensor dInput, DoubleTensor input) {
        final DoubleTensor L = input.triLower(0);
        final DoubleTensor X = symmetricFromLower(input.choleskyInverse());
        final DoubleTensor dL = dInput.triLower(0);

        final DoubleTensor dLLt = dL.matrixMultiply(L, false, true);
        final DoubleTensor dA = dLLt.plus(dLLt.transpose());
        final DoubleTensor dX = X.matrixMultiply(dA).matrixMultiply(X).unaryMinusInPlace();

        return dX.triLower(0).plusInPlace(strictlyUpper(dInput));
    }

    /**
     * With Xbar = tril(Ybar) and M = -X Xbar X the gradient with respect to L is (M + M^T) L, which is read
     * from the lower triangle of the input. The strictly upper triangle is passed through unchanged.
     */
    public static DoubleTensor reverseCholeskyInverse(DoubleTensor dOutput, DoubleTensor input) {
        final DoubleTensor L = input.triLower(0);
        final DoubleTensor X = symmetricFromLower(input.choleskyInverse());
        final DoubleTensor Xbar = dOutput.triLower(0);

        final DoubleTensor M = X.matrixMultiply(Xbar).matrixMultiply(X).unaryMinusInPlace();
        final DoubleTensor Lbar = M.plus(M.transpose()).matrixMultiply(L);

        return Lbar.triLower(0).plusInPlace(strictlyUpper(dOutput));
    }

    /**
     * phi(X) is the lower triangle of X with the diagonal halved
     */
    private static DoubleTensor phi(DoubleTensor X) {
        final DoubleTensor lower = X.triLower(0);
        final DoubleTensor strictlyLower = X.triLower(1);
        return lower.plus(strictlyLower).timesInPlace(0.5);
    }

    private static DoubleTensor symmetricFromLower(DoubleTensor X) {
        return X.triLower(0).plusInPlace(X.triLower(1).transpose());
    }

    private static DoubleTensor lowerGradientOfSymmetric(DoubleTensor S) {
        final DoubleTensor sum = S.plus(S.transpose());
        // diagonal of (S + S^T) is 2 * diag(S), so remove one diag(S)
        final DoubleTensor diagOfS = S.triLower(0).minusInPlace(S.triLower(1));
        return sum.triLower(0).minusInPlace(diagOfS);
    }

    private static DoubleTensor strictlyUpper(DoubleTensor X) {
        return X.minus(X.triLower(0));
    }
}
