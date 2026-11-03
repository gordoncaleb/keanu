package io.improbable.keanu.vertices.tensor.number;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.math.SpecialFunctions;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import org.apache.commons.math3.special.Gamma;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static org.junit.Assert.assertEquals;

/**
 * Finite difference checks for operations that previously threw UnsupportedOperationException when differentiated.
 */
public class AutoDiffRegressionTest {

    private static final double INCREMENT = 1e-6;
    private static final double DELTA = 1e-8;

    private static UniformVertex input(DoubleTensor value) {
        UniformVertex v = new UniformVertex(value.getShape(), -100.0, 100.0);
        v.setValue(value);
        return v;
    }

    private static void check(UniformVertex in, DoubleVertex out) {
        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(in), out, INCREMENT, DELTA);
    }

    @Test
    public void cumSum() {
        UniformVertex a = input(DoubleTensor.create(new double[]{1, -2, 3, 0.5, 2, -1}, 2, 3));
        check(a, a.cumSum(1).times(ConstantVertex.of(DoubleTensor.arange(1, 7).reshape(2, 3))));
        check(a, a.cumSum(0).times(ConstantVertex.of(DoubleTensor.arange(1, 7).reshape(2, 3))));
    }

    @Test
    public void cumProdIncludingZeros() {
        UniformVertex a = input(DoubleTensor.create(new double[]{1.5, 0.0, 3, 0.5, 2, -1}, 2, 3));
        check(a, a.cumProd(1).times(ConstantVertex.of(DoubleTensor.arange(1, 7).reshape(2, 3))));
        check(a, a.cumProd(0));
    }

    @Test
    public void standardDeviation() {
        UniformVertex a = input(DoubleTensor.create(1, 4, 2, 8, 5));
        check(a, a.standardDeviation());
    }

    @Test
    public void standardize() {
        UniformVertex a = input(DoubleTensor.create(new double[]{1, 4, 2, 8, 5, 3}, 2, 3));
        check(a, a.standardize().times(ConstantVertex.of(DoubleTensor.arange(1, 7).reshape(2, 3))));
    }

    @Test
    public void trigamma() {
        UniformVertex a = input(DoubleTensor.create(0.3, 1.0, 2.5, 12.0));
        check(a, a.trigamma());
    }

    @Test
    public void tetragammaMatchesNumericalDerivativeOfTrigamma() {
        for (double x : new double[]{-2.5, -0.3, 0.1, 0.5, 1, 3.7, 15, 120}) {
            double h = 1e-5 * Math.max(1, Math.abs(x));
            double numeric = (Gamma.trigamma(x + h) - Gamma.trigamma(x - h)) / (2 * h);
            assertEquals("tetragamma(" + x + ")", numeric, SpecialFunctions.tetragamma(x), 1e-5 * Math.max(1, Math.abs(numeric)));
        }
    }

    @Test
    public void logAddExpAndLogAddExp2WithBroadcast() {
        UniformVertex a = input(DoubleTensor.create(new double[]{1, -2, 3, 0.5, 2, -1}, 2, 3));
        UniformVertex b = input(DoubleTensor.create(0.3, -0.7, 1.2));

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(a, b), a.logAddExp(b), INCREMENT, DELTA);
        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(a, b), a.logAddExp2(b), INCREMENT, DELTA);
    }

    private static DoubleTensor spd(int n, long seed) {
        java.util.Random random = new java.util.Random(seed);
        double[] values = new double[n * n];
        for (int i = 0; i < values.length; i++) {
            values[i] = random.nextGaussian();
        }
        DoubleTensor a = DoubleTensor.create(values, n, n);
        return a.matrixMultiply(a.transpose()).plus(DoubleTensor.eye(n).times(n));
    }

    @Test
    public void choleskyDecompositionOfSymmetricInput() {
        UniformVertex a = input(DoubleTensor.create(new double[]{1, 0.2, -0.3, 0.4, 1, 0.1, -0.2, 0.3, 1}, 3, 3));
        DoubleVertex spd = a.matrixMultiply(a.transpose()).plus(ConstantVertex.of(DoubleTensor.eye(3).times(3.0)));
        check(a, spd.choleskyDecomposition().times(ConstantVertex.of(DoubleTensor.arange(1, 10).reshape(3, 3))));
    }

    @Test
    public void choleskyDecompositionOnlyDependsOnTheLowerTriangle() {
        UniformVertex a = input(spd(4, 1));
        check(a, a.choleskyDecomposition().times(ConstantVertex.of(DoubleTensor.arange(1, 17).reshape(4, 4))));
    }

    @Test
    public void choleskyDecompositionBatched() {
        DoubleTensor batch = DoubleTensor.concat(0, spd(3, 1).reshape(1, 3, 3), spd(3, 2).reshape(1, 3, 3));
        UniformVertex a = input(batch);
        check(a, a.choleskyDecomposition());
    }

    @Test
    public void choleskyInverse() {
        UniformVertex a = input(spd(3, 3).choleskyDecomposition());
        check(a, a.choleskyInverse().times(ConstantVertex.of(DoubleTensor.arange(1, 10).reshape(3, 3))));
    }

    @Test
    public void choleskyInverseOfCholeskyDecomposition() {
        UniformVertex a = input(spd(3, 4));
        check(a, a.choleskyDecomposition().choleskyInverse());
    }

    @Test
    public void tensorMultiply() {
        UniformVertex a = input(DoubleTensor.arange(1, 25).reshape(2, 3, 4).div(10));
        UniformVertex b = input(DoubleTensor.arange(1, 13).reshape(4, 3).div(7));

        // contract a's dims 1 and 2 with b's dims 1 and 0 (deliberately out of order)
        finiteDifferenceMatchesForwardAndReverseModeGradient(
            ImmutableList.of(a, b), a.tensorMultiply(b, new int[]{1, 2}, new int[]{1, 0}), INCREMENT, DELTA
        );

        // single contraction with a negative dimension
        finiteDifferenceMatchesForwardAndReverseModeGradient(
            ImmutableList.of(a, b), a.tensorMultiply(b, new int[]{-1}, new int[]{0}), INCREMENT, DELTA
        );

        // contraction over a middle dimension of a
        UniformVertex c = input(DoubleTensor.arange(1, 7).reshape(3, 2).div(5));
        finiteDifferenceMatchesForwardAndReverseModeGradient(
            ImmutableList.of(a, c), a.tensorMultiply(c, new int[]{1}, new int[]{0}), INCREMENT, DELTA
        );
    }

    /**
     * floor, ceil, round and sign are deliberately not differentiable: a log prob through them is piecewise
     * constant, so DifferentiableChecker must steer such models away from gradient based algorithms. Asking for a
     * gradient through them is an error with a clear message rather than a NullPointerException (or, previously for
     * round, a silently wrong identity gradient).
     */
    @Test
    public void piecewiseConstantOpsAreNotDifferentiable() {
        UniformVertex a = input(DoubleTensor.create(-1.7, 0.3, 2.2));

        for (DoubleVertex op : new DoubleVertex[]{a.floor(), a.ceil(), a.round(), a.sign()}) {
            org.junit.Assert.assertFalse(op.isDifferentiable());
            assertThrows(() -> io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiator.reverseModeAutoDiff(op, a).withRespectTo(a));
            assertThrows(() -> io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiator.forwardModeAutoDiff(a, op).of(op));
        }
    }

    private static void assertThrows(Runnable runnable) {
        try {
            runnable.run();
        } catch (IllegalArgumentException expected) {
            return;
        }
        org.junit.Assert.fail("Expected IllegalArgumentException");
    }

    @Test
    public void maxAndMinReductionsPassTheGradientToTheSelectedElement() {
        UniformVertex a = input(DoubleTensor.create(new double[]{-1.7, 4.2, 0.3, 1.6, -2.2, 3.9}, 2, 3));
        check(a, a.max());
        check(a, a.min());

        // a full reduction is a scalar vertex, not shaped like its input
        org.junit.Assert.assertArrayEquals(new long[0], a.max().getShape());
        org.junit.Assert.assertArrayEquals(new long[0], a.min().getShape());
    }

    @Test
    public void maxReductionSplitsTheGradientBetweenTies() {
        UniformVertex a = input(DoubleTensor.create(1.0, 3.0, 3.0, 2.0));
        DoubleVertex max = a.max();
        DoubleTensor reverse = io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiator
            .reverseModeAutoDiff(max, a).withRespectTo(a);
        DoubleTensor forward = io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiator
            .forwardModeAutoDiff(a, max).of(max);
        org.junit.Assert.assertArrayEquals(new double[]{0, 0.5, 0.5, 0}, reverse.asFlatDoubleArray(), 0.0);
        org.junit.Assert.assertArrayEquals(new double[]{0, 0.5, 0.5, 0}, forward.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void binaryMaxAndMinBroadcastLikeTheTensorOperations() {
        UniformVertex a = input(DoubleTensor.create(new double[]{-1.7, 4.2, 0.3, 1.6, -2.2, 3.9}, 2, 3));
        UniformVertex b = input(DoubleTensor.create(0.5, 1.0, 2.5));
        UniformVertex scalar = input(DoubleTensor.scalar(1.1));

        assertEquals(a.getValue().max(b.getValue()), a.max(b).eval());
        assertEquals(a.getValue().min(b.getValue()), a.min(b).eval());
        assertEquals(scalar.getValue().max(a.getValue()), scalar.max(a).eval());

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(a, b), a.max(b), INCREMENT, DELTA);
        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(a, b), a.min(b), INCREMENT, DELTA);
        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(scalar, a), scalar.min(a), INCREMENT, DELTA);
    }

    @Test
    public void productGradientIsCorrectWhereElementsAreZero() {
        UniformVertex a = input(DoubleTensor.create(new double[]{1.5, 0.0, -2.0, 3.0, 0.5, 2.0}, 2, 3));
        check(a, a.product());
        check(a, a.product(1));

        // d/dx_1 of x0 * x1 * x2 at x1 = 0 is x0 * x2, not 0 / 0
        UniformVertex b = input(DoubleTensor.create(1.5, 0.0, -2.0));
        DoubleTensor gradient = io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiator
            .reverseModeAutoDiff(b.product(), b).withRespectTo(b);
        org.junit.Assert.assertArrayEquals(new double[]{0.0, -3.0, 0.0}, gradient.asFlatDoubleArray(), 1e-12);
    }

    @Test
    public void aVertexUsedAsBothOperandsGetsBothContributionsInReverseMode() {
        UniformVertex x = input(DoubleTensor.create(0.75, 1.0, 1.25));
        UniformVertex y = input(DoubleTensor.scalar(0.7));
        DoubleVertex product = x.times(y);
        DoubleVertex squared = product.times(product).sum();

        // d/dx sum((xy)^2) = 2 x y^2, which used to come out halved because one operand's partial was overwritten
        DoubleTensor dx = io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiator
            .reverseModeAutoDiff(squared, x).withRespectTo(x);
        org.junit.Assert.assertArrayEquals(new double[]{2 * 0.75 * 0.49, 2 * 1.0 * 0.49, 2 * 1.25 * 0.49}, dx.asFlatDoubleArray(), 1e-12);
        check(x, squared);
        check(x, x.times(x).plus(x.div(x.plus(2.0))).sum());
    }

    @Test
    public void forwardModeThroughAFullSumHasTheRightShape() {
        UniformVertex x = input(DoubleTensor.create(0.75, 1.0, 1.25));
        UniformVertex y = input(DoubleTensor.scalar(0.7));
        DoubleVertex out = x.times(y).sum();
        DoubleVertex outMean = x.times(y).mean();

        DoubleTensor dy = io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiator
            .forwardModeAutoDiff(y, out).of(out);
        org.junit.Assert.assertArrayEquals(new long[0], dy.getShape());
        org.junit.Assert.assertEquals(3.0, dy.scalar(), 1e-12);
        check(y, out);
        check(y, outMean);
        check(x, x.reshape(3, 1).times(y).sum());
    }

    @Test
    public void whereGivesAZeroPartialForTheUnselectedBranch() {
        UniformVertex a = input(DoubleTensor.scalar(5.0));
        UniformVertex b = input(DoubleTensor.create(0.7, 0.9, 1.1, 1.3).reshape(2, 2));
        DoubleVertex out = a.max(b);

        DoubleTensor forward = io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiator
            .forwardModeAutoDiff(b, out).of(out);
        org.junit.Assert.assertEquals(0.0, forward.abs().sum().scalar(), 0.0);
        check(b, out);
    }
}
