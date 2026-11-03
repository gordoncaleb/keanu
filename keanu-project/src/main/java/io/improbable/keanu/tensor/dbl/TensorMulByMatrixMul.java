package io.improbable.keanu.tensor.dbl;

import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import io.improbable.keanu.tensor.NumberTensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TensorMulByMatrixMul {

    public static <T extends Number, TENSOR extends NumberTensor<T, TENSOR>> TENSOR tensorMmul(TENSOR left, TENSOR right, int[] dims0, int[] dims1) {

        long[] leftShape = left.getShape();
        long[] rightShape = right.getShape();

        // normalised copies so negative dimensions are supported and the caller's arrays are not mutated
        final int[] dimsLeft = normaliseDimensions(dims0, leftShape, "left");
        final int[] dimsRight = normaliseDimensions(dims1, rightShape, "right");

        validateTensorMmul(leftShape, rightShape, dimsLeft, dimsRight);

        List<Integer> leftDimsKept = getKeptDimensions(leftShape.length, dimsLeft);
        List<Integer> rightDimsKept = getKeptDimensions(rightShape.length, dimsRight);

        int[] leftDimsPermuted = Ints.concat(Ints.toArray(leftDimsKept), dimsLeft);
        int[] rightDimsPermuted = Ints.concat(dimsRight, Ints.toArray(rightDimsKept));

        long dimsLength = calculateDimensionsLength(leftShape, dimsLeft);
        long[] leftTensorAsMatrixShape = {-1, dimsLength};
        long[] rightTensorAsMatrixShape = {dimsLength, -1};

        TENSOR leftTensorAsMatrix = left.permute(leftDimsPermuted).reshape(leftTensorAsMatrixShape);
        TENSOR rightTensorAsMatrix = right.permute(rightDimsPermuted).reshape(rightTensorAsMatrixShape);
        TENSOR resultAsMatrix = leftTensorAsMatrix.matrixMultiply(rightTensorAsMatrix);

        long[] leftKeptShape = getKeptShape(leftShape, leftDimsKept);
        long[] rightKeptShape = getKeptShape(rightShape, rightDimsKept);

        long[] resultShape = Longs.concat(leftKeptShape, rightKeptShape);
        return resultAsMatrix.reshape(resultShape);
    }

    public static long[] getResultShape(long[] leftShape, long[] rightShape, int[] dims0, int[] dims1) {

        final int[] dimsLeft = normaliseDimensions(dims0, leftShape, "left");
        final int[] dimsRight = normaliseDimensions(dims1, rightShape, "right");

        List<Integer> leftDimsKept = getKeptDimensions(leftShape.length, dimsLeft);
        List<Integer> rightDimsKept = getKeptDimensions(rightShape.length, dimsRight);

        long[] leftKeptShape = getKeptShape(leftShape, leftDimsKept);
        long[] rightKeptShape = getKeptShape(rightShape, rightDimsKept);

        return Longs.concat(leftKeptShape, rightKeptShape);
    }

    private static int[] normaliseDimensions(int[] dims, long[] shape, String side) {
        final int[] normalised = new int[dims.length];
        for (int i = 0; i < dims.length; i++) {
            final int dim = dims[i] < 0 ? dims[i] + shape.length : dims[i];
            if (dim < 0 || dim >= shape.length) {
                throw new IllegalArgumentException(
                    "Invalid " + side + " dimension " + dims[i] + " for shape " + Arrays.toString(shape)
                );
            }
            normalised[i] = dim;
        }
        return normalised;
    }

    /**
     * Validates that the multiply dimensions match for the left shape and the right shape.
     *
     * @param leftShape  the shape of the left tensor being mmul
     * @param rightShape the shape of the right tensor being mmul
     * @param dimsLeft   normalised (non-negative) dimensions along the left to multiply
     * @param dimsRight  normalised (non-negative) dimensions along the right to multiply
     */
    private static void validateTensorMmul(long[] leftShape, long[] rightShape, int[] dimsLeft, int[] dimsRight) {

        if (dimsLeft.length != dimsRight.length) {
            throw new IllegalArgumentException("The same number of dimensions must be given for the left and right.");
        }

        for (int i = 0; i < dimsLeft.length; i++) {
            if (leftShape[dimsLeft[i]] != rightShape[dimsRight[i]]) {
                throw new IllegalArgumentException("Size of the given axes at each dimension must be the same size.");
            }
        }
    }

    private static List<Integer> getKeptDimensions(int shapeLength, int[] dims) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < shapeLength; i++) {
            if (!Ints.contains(dims, i)) {
                result.add(i);
            }
        }
        return result;
    }

    private static long[] getKeptShape(long[] shape, List<Integer> keptDimensions) {
        long[] keptShape = Longs.toArray(keptDimensions);
        for (int i = 0; i < keptShape.length; i++) {
            keptShape[i] = shape[Ints.checkedCast(keptShape[i])];
        }
        return keptShape;
    }

    private static long calculateDimensionsLength(long[] shape, int[] dims) {
        long length = 1;
        int aLength = Math.min(shape.length, dims.length);
        for (int i = 0; i < aLength; i++) {
            length *= shape[dims[i]];
        }
        return length;
    }

}
