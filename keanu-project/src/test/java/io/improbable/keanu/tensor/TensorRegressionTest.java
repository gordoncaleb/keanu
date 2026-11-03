package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.KeanuLapack;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.lng.LongTensor;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Regression tests for tensor bugs found in code review.
 */
public class TensorRegressionTest {

    @Test
    public void atan2WithScalarMatchesAtan2WithTensor() {
        DoubleTensor x = DoubleTensor.create(1.0, -1.0, 2.0);

        DoubleTensor withScalar = x.atan2(0.5);
        DoubleTensor withTensor = x.atan2(DoubleTensor.create(0.5, 0.5, 0.5));

        // atan2(y, x) element-wise
        assertArrayEquals(new double[]{Math.atan2(0.5, 1.0), Math.atan2(0.5, -1.0), Math.atan2(0.5, 2.0)}, withScalar.asFlatDoubleArray(), 1e-12);
        assertArrayEquals(withTensor.asFlatDoubleArray(), withScalar.asFlatDoubleArray(), 1e-12);
        assertArrayEquals(new double[]{0, 0}, DoubleTensor.create(1.0, 1.0).atan2(0.0).asFlatDoubleArray(), 0.0);
    }

    @Test
    public void maxAndMinHandleInfinity() {
        assertEquals(Double.NEGATIVE_INFINITY, DoubleTensor.create(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY).max().scalar(), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, DoubleTensor.create(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY).min().scalar(), 0.0);
    }

    @Test
    public void integerPowWithNegativeExponentIsConsistentAcrossCodePaths() {
        IntegerTensor base = IntegerTensor.create(2, 3, 1, -1);

        assertArrayEquals(new int[]{0, 0, 1, -1}, base.pow(IntegerTensor.create(-1, -1, -1, -1)).asFlatIntegerArray());
        assertArrayEquals(new int[]{0, 0, 1, -1}, base.pow(-1).asFlatIntegerArray());
        assertArrayEquals(new int[]{8, 27, 1, -1}, base.pow(IntegerTensor.create(3, 3, 3, 3)).asFlatIntegerArray());
        assertArrayEquals(new int[]{8, 27, 1, -1}, base.pow(3).asFlatIntegerArray());
        assertArrayEquals(new int[]{1, 1, 1, 1}, base.pow(0).asFlatIntegerArray());
    }

    @Test
    public void longPowWithNegativeExponentIsConsistentAcrossCodePaths() {
        LongTensor base = LongTensor.create(2L, 1L, -1L);
        assertArrayEquals(new long[]{0, 1, 1}, base.pow(LongTensor.create(-2L, -2L, -2L)).asFlatLongArray());
        assertArrayEquals(new long[]{0, 1, 1}, base.pow(-2L).asFlatLongArray());
    }

    @Test
    public void integerPowOverflowWrapsTheSameOnEveryCodePath() {
        IntegerTensor base = IntegerTensor.create(3);
        assertEquals(base.pow(IntegerTensor.create(40)).scalar(), base.pow(40).scalar());
    }

    @Test(expected = ArithmeticException.class)
    public void zeroToANegativePowerThrows() {
        IntegerTensor.create(0, 1).pow(-1);
    }

    @Test
    public void emptySliceWithStepGreaterThanOne() {
        DoubleTensor a = DoubleTensor.arange(0, 5);
        assertEquals(0, a.slice("3:3:2").getLength());
        assertEquals(0, a.slice("2:1:3").getLength());
        assertArrayEquals(new double[]{0, 2, 4}, a.slice("::2").asFlatDoubleArray(), 0.0);
        assertArrayEquals(new double[]{1, 3}, a.slice("1::2").asFlatDoubleArray(), 0.0);
    }

    @Test
    public void reverseSliceMatchesNumpy() {
        DoubleTensor a = DoubleTensor.arange(0, 5);
        assertArrayEquals(new double[]{4, 3, 2, 1, 0}, a.slice("::-1").asFlatDoubleArray(), 0.0);
        assertArrayEquals(new double[]{4, 2, 0}, a.slice("::-2").asFlatDoubleArray(), 0.0);
        assertArrayEquals(new double[]{3, 2}, a.slice("3:1:-1").asFlatDoubleArray(), 0.0);
        assertArrayEquals(new double[]{3, 2, 1, 0}, a.slice("-2::-1").asFlatDoubleArray(), 0.0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void reshapeWithNegativeOneRequiresEvenDivision() {
        DoubleTensor.arange(0, 6).reshape(4, -1);
    }

    @Test
    public void reshapeWithNegativeOneInfersDimension() {
        assertArrayEquals(new long[]{2, 3}, DoubleTensor.arange(0, 6).reshape(2, -1).getShape());
    }

    @Test
    public void diagPartOfBatchedNonSquareMatrices() {
        // 2 batches of 2x3 matrices
        DoubleTensor a = DoubleTensor.arange(0, 12).reshape(2, 2, 3);

        DoubleTensor diag = a.diagPart();

        assertArrayEquals(new long[]{2, 2}, diag.getShape());
        assertArrayEquals(new double[]{0, 4, 6, 10}, diag.asFlatDoubleArray(), 0.0);

        // 2 batches of 3x2 matrices
        DoubleTensor b = DoubleTensor.arange(0, 12).reshape(2, 3, 2);
        assertArrayEquals(new double[]{0, 3, 6, 9}, b.diagPart().asFlatDoubleArray(), 0.0);
    }

    @Test
    public void matrixDeterminantOfBatchWithOneSingularMatrix() {
        DoubleTensor batch = DoubleTensor.create(new double[]{
            1, 2,
            2, 4,

            1, 2,
            3, 4
        }, 2, 2, 2);

        DoubleTensor det = batch.matrixDeterminant();

        assertArrayEquals(new long[]{2}, det.getShape());
        assertArrayEquals(new double[]{0.0, -2.0}, det.asFlatDoubleArray(), 1e-12);
    }

    @Test
    public void tensorMultiplySupportsNegativeDimensionsAndDoesNotMutateArguments() {
        DoubleTensor a = DoubleTensor.arange(0, 6).reshape(2, 3);
        DoubleTensor b = DoubleTensor.arange(0, 12).reshape(3, 4);

        int[] dimsLeft = new int[]{-1};
        int[] dimsRight = new int[]{0};

        DoubleTensor result = a.tensorMultiply(b, dimsLeft, dimsRight);

        assertEquals(a.matrixMultiply(b), result);
        assertArrayEquals(new int[]{-1}, dimsLeft);
        assertArrayEquals(new int[]{0}, dimsRight);
    }

    @Test
    public void logAddExpOfMatchingInfinities() {
        DoubleTensor negInf = DoubleTensor.create(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        DoubleTensor other = DoubleTensor.create(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, 0.0);

        assertArrayEquals(new double[]{Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, 0.0}, negInf.logAddExp(other).asFlatDoubleArray(), 1e-12);
        assertArrayEquals(new double[]{Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, 0.0}, negInf.logAddExp2(other).asFlatDoubleArray(), 1e-12);
    }

    @Test
    public void argMaxAndArgMinSupportNegativeAxes() {
        DoubleTensor a = DoubleTensor.create(new double[]{1, 5, 2, 7, 0, 3}, 2, 3);

        assertEquals(a.argMax(1), a.argMax(-1));
        assertEquals(a.argMin(0), a.argMin(-2));
        assertArrayEquals(new int[]{1, 0}, a.argMax(-1).asFlatIntegerArray());

        IntegerTensor i = IntegerTensor.create(new int[]{1, 5, 2, 7, 0, 3}, 2, 3);
        assertArrayEquals(new int[]{1, 0}, i.argMax(-1).asFlatIntegerArray());
    }

    @Test
    public void argMaxReturnsFirstNaNLikeNumpy() {
        assertEquals(0, DoubleTensor.create(Double.NaN, 1, 2).argMax().scalar().intValue());
        assertEquals(1, DoubleTensor.create(1, Double.NaN, 2).argMax().scalar().intValue());
        assertEquals(1, DoubleTensor.create(1, Double.NaN, 2).argMin().scalar().intValue());
        assertEquals(2, DoubleTensor.create(1, Double.NaN, 2).nanArgMax().scalar().intValue());
    }

    @Test
    public void elementwiseEqualityUsesIEEESemanticsOnEveryCodePath() {
        DoubleTensor withNaN = DoubleTensor.create(Double.NaN, 1.0);

        assertArrayEquals(new Boolean[]{false, true}, withNaN.elementwiseEquals(DoubleTensor.create(Double.NaN, 1.0)).asFlatArray());
        assertArrayEquals(new Boolean[]{false, false}, withNaN.elementwiseEquals(Double.NaN).asFlatArray());
        assertFalse(DoubleTensor.scalar(Double.NaN).elementwiseEquals(Double.NaN).scalar());

        assertArrayEquals(new Boolean[]{true, true}, DoubleTensor.create(0.0, 1.0).elementwiseEquals(DoubleTensor.create(-0.0, 1.0)).asFlatArray());
        assertTrue(DoubleTensor.scalar(0.0).elementwiseEquals(-0.0).scalar());
    }

    @Test
    public void pureJavaLinearAlgebraMatchesCommonsMath() {
        Random random = new Random(1);
        int n = 5;
        double[] values = new double[n * n];
        for (int i = 0; i < values.length; i++) {
            values[i] = random.nextGaussian();
        }

        DoubleTensor a = DoubleTensor.create(values, n, n);
        // symmetric positive definite
        DoubleTensor spd = a.matrixMultiply(a.transpose()).plus(DoubleTensor.eye(n).times(n));

        RealMatrix aMatrix = toMatrix(a);
        RealMatrix spdMatrix = toMatrix(spd);

        assertArrayEquals(flatten(aMatrix.multiply(aMatrix.transpose())), a.matrixMultiply(a.transpose()).asFlatDoubleArray(), 1e-10);
        assertArrayEquals(flatten(aMatrix.transpose().multiply(aMatrix)), a.matrixMultiply(a, true, false).asFlatDoubleArray(), 1e-10);
        assertArrayEquals(flatten(new LUDecomposition(aMatrix).getSolver().getInverse()), a.matrixInverse().asFlatDoubleArray(), 1e-10);
        assertEquals(new LUDecomposition(aMatrix).getDeterminant(), a.matrixDeterminant().scalar(), 1e-10);
        assertArrayEquals(flatten(new CholeskyDecomposition(spdMatrix).getL()), spd.choleskyDecomposition().asFlatDoubleArray(), 1e-10);
        assertArrayEquals(
            flatten(new LUDecomposition(spdMatrix).getSolver().getInverse()),
            symmetricFromLower(spd.choleskyDecomposition().choleskyInverse()).asFlatDoubleArray(),
            1e-10
        );
    }

    @Test
    public void choleskyReportsNonPositiveDefiniteMatrices() {
        double[] notPositiveDefinite = new double[]{1, 2, 2, 1};
        assertEquals(2, KeanuLapack.dpotrf(KeanuLapack.Triangular.LOWER, 2, notPositiveDefinite, 0));
    }

    private static DoubleTensor symmetricFromLower(DoubleTensor lower) {
        return lower.plus(lower.triLower(1).transpose());
    }

    private static RealMatrix toMatrix(DoubleTensor t) {
        int rows = (int) t.getShape()[0];
        int cols = (int) t.getShape()[1];
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = t.getValue(i, j);
            }
        }
        return new Array2DRowRealMatrix(data);
    }

    private static double[] flatten(RealMatrix m) {
        double[] flat = new double[m.getRowDimension() * m.getColumnDimension()];
        for (int i = 0; i < m.getRowDimension(); i++) {
            for (int j = 0; j < m.getColumnDimension(); j++) {
                flat[i * m.getColumnDimension() + j] = m.getEntry(i, j);
            }
        }
        return flat;
    }

    @Test
    public void logGammaOfInfinityIsInfinity() {
        assertEquals(Double.POSITIVE_INFINITY, DoubleTensor.scalar(Double.POSITIVE_INFINITY).logGamma().scalar(), 0.0);
        assertEquals(Math.log(6.0), DoubleTensor.scalar(4.0).logGamma().scalar(), 1e-12);
    }
}
