package io.improbable.keanu.tensor.dbl;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.FloatingPointScalarOperations;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.bool.JVMBooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.jvm.JVMFloatingPointTensor;
import io.improbable.keanu.tensor.jvm.JVMNumberTensor;
import io.improbable.keanu.tensor.jvm.JVMTensor;
import io.improbable.keanu.tensor.jvm.ResultWrapper;
import io.improbable.keanu.tensor.lng.JVMLongTensorFactory;
import io.improbable.keanu.tensor.lng.LongTensor;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import java.util.Arrays;

import static io.improbable.keanu.tensor.TensorShape.getBroadcastedFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;
import static io.improbable.keanu.tensor.dbl.KeanuLapack.dgemm;
import static io.improbable.keanu.tensor.dbl.KeanuLapack.dgetrf;
import static io.improbable.keanu.tensor.dbl.KeanuLapack.dgetri;
import static io.improbable.keanu.tensor.dbl.KeanuLapack.dpotrf;
import static io.improbable.keanu.tensor.dbl.KeanuLapack.dpotri;

public class JVMDoubleTensor extends JVMFloatingPointTensor<Double, DoubleTensor, DoubleBuffer.PrimitiveDoubleWrapper> implements DoubleTensor {

    static final DoubleBuffer.DoubleArrayWrapperFactory factory = new DoubleBuffer.DoubleArrayWrapperFactory();

    JVMDoubleTensor(DoubleBuffer.PrimitiveDoubleWrapper buffer, long[] shape, long[] stride) {
        super(buffer, shape, stride);
    }

    JVMDoubleTensor(DoubleBuffer.PrimitiveDoubleWrapper buffer, long[] shape) {
        super(buffer, shape, getRowFirstStride(shape));
    }

    JVMDoubleTensor(ResultWrapper<Double, DoubleBuffer.PrimitiveDoubleWrapper> resultWrapper) {
        this(resultWrapper.outputBuffer, resultWrapper.outputShape, resultWrapper.outputStride);
    }

    JVMDoubleTensor(double[] data, long[] shape, long[] stride) {
        this(factory.create(data), shape, stride);
    }

    JVMDoubleTensor(double[] data, long[] shape) {
        this(factory.create(data), shape);
    }

    JVMDoubleTensor(double value) {
        super(new DoubleBuffer.DoubleWrapper(value), new long[0], new long[0]);
    }

    @Override
    protected DoubleTensor create(DoubleBuffer.PrimitiveDoubleWrapper buffer, long[] shape, long[] stride) {
        return new JVMDoubleTensor(buffer, shape, stride);
    }

    @Override
    protected DoubleTensor set(DoubleBuffer.PrimitiveDoubleWrapper buffer, long[] shape, long[] stride) {
        this.buffer = buffer;
        this.shape = shape;
        this.stride = stride;
        return this;
    }

    @Override
    protected DoubleBuffer.DoubleArrayWrapperFactory getFactory() {
        return factory;
    }

    @Override
    protected FloatingPointScalarOperations<Double> getOperations() {
        return DoubleScalarOperations.INSTANCE;
    }

    @Override
    protected JVMTensor<Double, DoubleTensor, DoubleBuffer.PrimitiveDoubleWrapper> getAsJVMTensor(DoubleTensor that) {
        return asJVM(that);
    }

    static JVMNumberTensor<Double, DoubleTensor, DoubleBuffer.PrimitiveDoubleWrapper> asJVM(DoubleTensor that) {
        if (that instanceof JVMDoubleTensor) {
            return ((JVMDoubleTensor) that);
        } else {
            return new JVMDoubleTensor(factory.create(that.asFlatDoubleArray()), that.getShape(), that.getStride());
        }
    }

    @Override
    public BooleanTensor toBoolean() {
        return new JVMBooleanTensor(buffer.equal(1.0), getShape(), getStride());
    }

    @Override
    public DoubleTensor toDouble() {
        return this;
    }

    @Override
    public IntegerTensor toInteger() {
        return IntegerTensor.create(buffer.asIntegerArray(), getShape());
    }

    @Override
    public LongTensor toLong() {
        return JVMLongTensorFactory.INSTANCE.create(buffer.asLongArray(), getShape());
    }

    @Override
    public double[] asFlatDoubleArray() {
        return buffer.copy().asDoubleArray();
    }

    @Override
    public int[] asFlatIntegerArray() {
        return buffer.asIntegerArray();
    }

    @Override
    public long[] asFlatLongArray() {
        return buffer.asLongArray();
    }

    @Override
    public Double[] asFlatArray() {
        return buffer.asArray();
    }

    @Override
    public DoubleTensor choleskyDecomposition() {

        Preconditions.checkArgument(shape.length >= 2, "Matrix decomposition must be performed on matrix");
        final int M = Ints.checkedCast(shape[shape.length - 2]);
        final int N = Ints.checkedCast(shape[shape.length - 1]);
        Preconditions.checkArgument(M == N, "Cholesky decomposition input must be square");

        final long[] batchShape = ArrayUtils.subarray(shape, 0, shape.length - 2);
        final int batchLength = TensorShape.getLengthAsInt(batchShape);
        final int batchSize = M * N;

        final double[] newBuffer = buffer.copy().asDoubleArray();
        for (int batch = 0; batch < batchLength; batch++) {

            final int position = batch * batchSize;

            final int factorizationResult = dpotrf(KeanuLapack.Triangular.LOWER, N, newBuffer, position);

            if (factorizationResult != 0) {
                throw new IllegalStateException(
                    "Cholesky decomposition failed for input " + this.toString() +
                        (batchLength > 1 ? "\nat batch " + batch : "")
                );
            }

            zeroOutUpperTriangle(N, newBuffer, position);

        }

        return new JVMDoubleTensor(newBuffer, getShape(), getStride());
    }

    private void zeroOutUpperTriangle(int N, double[] buffer, int position) {
        if (N > 1) {
            for (int i = 0; i < N; i++) {
                for (int j = i + 1; j < N; j++) {
                    buffer[position + i * N + j] = 0;
                }
            }
        }
    }

    @Override
    public DoubleTensor choleskyInverse() {
        Preconditions.checkArgument(shape.length >= 2, "Matrix decomposition must be performed on matrix");
        final int M = Ints.checkedCast(shape[shape.length - 2]);
        final int N = Ints.checkedCast(shape[shape.length - 1]);
        Preconditions.checkArgument(M == N, "Cholesky inverse input must be square");

        final long[] batchShape = ArrayUtils.subarray(shape, 0, shape.length - 2);
        final int batchLength = TensorShape.getLengthAsInt(batchShape);
        final int batchSize = M * N;

        final double[] newBuffer = buffer.copy().asDoubleArray();
        for (int batch = 0; batch < batchLength; batch++) {

            int inverseResult = dpotri(KeanuLapack.Triangular.LOWER, N, newBuffer, batch * batchSize);

            if (inverseResult != 0) {
                throw new IllegalStateException("Cholesky inverse failed");
            }

        }

        return new JVMDoubleTensor(newBuffer, getShape(), getStride());
    }

    @Override
    public DoubleTensor matrixDeterminant() {
        Preconditions.checkArgument(shape.length >= 2, "Matrix decomposition must be performed on matrix");
        final int M = Ints.checkedCast(shape[shape.length - 2]);
        final int N = Ints.checkedCast(shape[shape.length - 1]);

        final long[] batchShape = ArrayUtils.subarray(shape, 0, shape.length - 2);
        final int batchLength = TensorShape.getLengthAsInt(batchShape);
        final int batchSize = M * N;

        final double[] newBuffer = buffer.copy().asDoubleArray();
        final double[] results = new double[batchLength];
        for (int batch = 0; batch < batchLength; batch++) {

            final int[] ipiv = new int[Math.min(M, N)];
            final int position = batch * batchSize;

            final int factorizationResult = dgetrf(M, N, newBuffer, position, ipiv);

            if (factorizationResult < 0) {
                throw new IllegalStateException("Matrix factorization failed");
            } else if (factorizationResult > 0) {
                results[batch] = 0.0;
                continue;
            }

            //credit: https://stackoverflow.com/questions/47315471/compute-determinant-from-lu-decomposition-in-lapack
            double detp = 1.;
            for (int j = 0; j < ipiv.length; j++) {
                if (j + 1 != ipiv[j]) {
                    detp = -detp;
                }
            }

            double detU = 1.0;
            for (int i = 0; i < M; i++) {
                detU *= newBuffer[position + i * N + i];
            }

            results[batch] = detU * detp;
        }

        return new JVMDoubleTensor(results, batchShape, TensorShape.getRowFirstStride(batchShape));
    }

    @Override
    public DoubleTensor matrixInverse() {
        final int m = Ints.checkedCast(shape[shape.length - 2]);
        final int n = Ints.checkedCast(shape[shape.length - 1]);

        final long[] batchShape = ArrayUtils.subarray(shape, 0, shape.length - 2);
        final int batchLength = TensorShape.getLengthAsInt(batchShape);
        final int batchSize = m * n;

        final double[] newBuffer = buffer.copy().asDoubleArray();
        for (int batch = 0; batch < batchLength; batch++) {

            final int[] ipiv = new int[Math.min(m, n)];
            final int position = batch * batchSize;

            final int factorizationResult = dgetrf(m, n, newBuffer, position, ipiv);

            if (factorizationResult < 0) {
                throw new IllegalStateException("Matrix factorization failed");
            } else if (factorizationResult > 0) {
                throw new SingularMatrixException();
            }

            int inverseResult = dgetri(m, newBuffer, position, ipiv);

            if (inverseResult != 0) {
                throw new IllegalStateException("Matrix inverse failed");
            }

        }

        return new JVMDoubleTensor(newBuffer, getShape(), getStride());
    }

    private void throwMatrixMultiplyException(long[] leftShape, long[] rightShape) {
        throw new IllegalArgumentException(
            "Cannot matrix multiply with shapes " + Arrays.toString(leftShape) + " and " + Arrays.toString(rightShape)
        );
    }

    @Override
    public DoubleTensor matrixMultiply(DoubleTensor that, boolean transposeLeft, boolean transposeRight) {

        final long[] rightShape = that.getShape();
        final long[] leftShape = this.getShape();

        if (leftShape.length < 2 || rightShape.length < 2) {
            throwMatrixMultiplyException(leftShape, rightShape);
        }

        //C = alpha*A*B + beta*C
        //(M,N) = (M,k)(k,N) + (M,N)
        final int K = Ints.checkedCast(transposeLeft ? leftShape[leftShape.length - 2] : leftShape[leftShape.length - 1]);
        final int N = Ints.checkedCast(transposeRight ? rightShape[rightShape.length - 2] : rightShape[rightShape.length - 1]);
        final int M = Ints.checkedCast(transposeLeft ? leftShape[leftShape.length - 1] : leftShape[leftShape.length - 2]);
        final int lda = transposeLeft ? M : K;
        final int ldb = transposeRight ? K : N;

        final int KFromRight = Ints.checkedCast(transposeRight ? rightShape[rightShape.length - 1] : rightShape[rightShape.length - 2]);
        if (K != KFromRight) {
            throwMatrixMultiplyException(leftShape, rightShape);
        }

        if (leftShape.length > 2 || rightShape.length > 2) {
            return batchMatrixMultiply(that, K, M, N, lda, ldb, transposeLeft, transposeRight);
        }

        final long[] resultShape = new long[]{M, N};

        final double[] C = new double[Ints.checkedCast(M * N)];
        final double[] A = buffer.asDoubleArray();
        final double[] B = getAsJVMTensor(that).getBuffer().asDoubleArray();

        dgemm(transposeLeft, transposeRight, M, N, K, A, 0, lda, B, 0, ldb, C, 0, N);

        return new JVMDoubleTensor(C, resultShape);
    }

    private DoubleTensor batchMatrixMultiply(DoubleTensor that, int K, int M, int N, int lda, int ldb, boolean transposeLeft, boolean transposeRight) {

        final long[] rightShape = that.getShape();
        final long[] rightStride = that.getStride();
        final long[] leftShape = this.getShape();
        final long[] leftStride = this.getStride();

        final long[] batchShapeLeft = ArrayUtils.subarray(leftShape, 0, leftShape.length - 2);
        final long[] batchStrideLeft = ArrayUtils.subarray(leftStride, 0, leftStride.length - 2);
        final long[] batchShapeRight = ArrayUtils.subarray(rightShape, 0, rightShape.length - 2);
        final long[] batchStrideRight = ArrayUtils.subarray(rightStride, 0, rightStride.length - 2);

        final long[] batchShape = TensorShape.getBroadcastResultShape(batchShapeLeft, batchShapeRight);
        final long batchLength = TensorShape.getLength(batchShape);

        //C = alpha*A*B + beta*C
        //(M,N) = (M,k)(k,N) + (M,N)
        final long[] resultShape = TensorShape.concat(batchShape, new long[]{M, N});
        final long[] resultStride = TensorShape.getRowFirstStride(resultShape);
        final long[] batchResultStride = ArrayUtils.subarray(resultStride, 0, resultStride.length - 2);

        final int resultBatchSize = M * N;
        final int batchSizeA = M * K;
        final int batchSizeB = N * K;

        final double[] outputBuffer = new double[Ints.checkedCast(resultBatchSize * batchLength)];
        final double[] leftInput = buffer.asDoubleArray();
        final double[] rightInput = getAsJVMTensor(that).getBuffer().asDoubleArray();

        for (int i = 0; i < batchLength; i++) {

            final int resultPosition = i * resultBatchSize;
            final int k = Ints.checkedCast(getBroadcastedFlatIndex(resultPosition, batchResultStride, batchShapeLeft, batchStrideLeft));
            final int j = Ints.checkedCast(getBroadcastedFlatIndex(resultPosition, batchResultStride, batchShapeRight, batchStrideRight));

            dgemm(transposeLeft, transposeRight, M, N, K, leftInput, k, lda, rightInput, j, ldb, outputBuffer, resultPosition, N);
        }

        return new JVMDoubleTensor(outputBuffer, resultShape, resultStride);
    }

    @Override
    public DoubleTensor setWithMaskInPlace(DoubleTensor mask, Double value) {
        return broadcastableBinaryOpWithAutoBroadcastInPlace((l, r) -> r == 1.0 ? value : l, getAsJVMTensor(mask));
    }

    @Override
    public DoubleTensor mean() {
        return new JVMDoubleTensor(buffer.sum() / buffer.getLength());
    }

    @Override
    public DoubleTensor mean(int... overDimensions) {
        overDimensions = TensorShape.getAbsoluteDimensions(shape.length, overDimensions);
        final long length = TensorShape.getLength(shape, overDimensions);
        return sum(overDimensions).divInPlace((double) length);
    }

    @Override
    public DoubleTensor standardDeviation() {

        SummaryStatistics stats = new SummaryStatistics();
        for (int i = 0; i < buffer.getLength(); i++) {
            stats.addValue(buffer.get(i));
        }

        return new JVMDoubleTensor(stats.getStandardDeviation());
    }

    @Override
    public DoubleTensor replaceNaNInPlace(Double value) {
        buffer.apply(v -> Double.isNaN(v) ? value : v);
        return this;
    }

}
