package io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.function.Function;

/**
 * Forward and reverse mode auto diff for a unary op y = f(x) from its dense Jacobian J, where
 * J[i, j] = dy_i / dx_j over the flattened output (length m) and input (length n).
 * <p>
 * This costs O(m * n) memory so it is intended for reductions and operations along a dimension
 * (e.g. cumulative sums/products, standard deviation, standardize) where a closed form for the partial is awkward.
 */
public final class JacobianAutoDiff {

    private JacobianAutoDiff() {
    }

    /**
     * @param dInput   the forward partial of the input with respect to some wrt, shaped [wrt..., input...]
     * @param jacobian dy/dx shaped [m, n]
     * @param outShape the shape of y
     * @return the forward partial of y with respect to the same wrt, shaped [wrt..., output...]
     */
    public static ForwardModePartialDerivative forward(ForwardModePartialDerivative dInput, DoubleTensor jacobian, long[] outShape) {
        if (!dInput.isPresent()) {
            return dInput;
        }

        final long[] wrtShape = dInput.getWrtShape();
        final long wrtLength = TensorShape.getLength(wrtShape);
        final long inLength = jacobian.getShape()[1];

        final DoubleTensor partial = dInput.get()
            .reshape(wrtLength, inLength)
            .matrixMultiply(jacobian, false, true)
            .reshape(TensorShape.concat(wrtShape, outShape));

        return new ForwardModePartialDerivative(wrtShape, partial);
    }

    /**
     * @param dOutput  the reverse partial of some of with respect to y, shaped [of..., output...]
     * @param jacobian dy/dx shaped [m, n]
     * @param inShape  the shape of x
     * @return the reverse partial of the same of with respect to x, shaped [of..., input...]
     */
    public static ReverseModePartialDerivative reverse(ReverseModePartialDerivative dOutput, DoubleTensor jacobian, long[] inShape) {
        final long[] ofShape = dOutput.getOfShape();
        final long ofLength = TensorShape.getLength(ofShape);
        final long outLength = jacobian.getShape()[0];

        final DoubleTensor partial = dOutput.get()
            .reshape(ofLength, outLength)
            .matrixMultiply(jacobian)
            .reshape(TensorShape.concat(ofShape, inShape));

        return new ReverseModePartialDerivative(ofShape, partial);
    }

    /**
     * Exact Jacobian of an op that is affine in each individual input element when the others are held fixed
     * (e.g. cumulative sum or cumulative product). Column j is f(x with x_j = 1) - f(x with x_j = 0), which is
     * exact for such ops and, unlike dividing by x_j, is well defined when elements of x are zero.
     *
     * @param op    the operation
     * @param input x
     * @return J shaped [m, n]
     */
    public static DoubleTensor jacobianOfElementwiseAffineOp(Function<DoubleTensor, DoubleTensor> op, DoubleTensor input) {
        final double[] x = input.asFlatDoubleArray();
        final long[] shape = input.getShape();
        final int n = x.length;

        double[][] columns = new double[n][];
        for (int j = 0; j < n; j++) {
            final double[] withOne = x.clone();
            final double[] withZero = x.clone();
            withOne[j] = 1.0;
            withZero[j] = 0.0;
            columns[j] = op.apply(DoubleTensor.create(withOne, shape))
                .minus(op.apply(DoubleTensor.create(withZero, shape)))
                .asFlatDoubleArray();
        }

        final int m = n == 0 ? 0 : columns[0].length;
        final double[] jacobian = new double[m * n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                jacobian[i * n + j] = columns[j][i];
            }
        }

        return DoubleTensor.create(jacobian, m, n);
    }
}
