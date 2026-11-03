package io.improbable.keanu.golden;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.lng.LongTensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static io.improbable.keanu.golden.Cases.invoke;
import static io.improbable.keanu.golden.Cases.params;
import static io.improbable.keanu.golden.Cases.result;
import static io.improbable.keanu.golden.Cases.testCase;

/**
 * Golden cases for tensor operations. Each case records the receiver ("args"[0]), any further arguments, and the
 * result of calling the named method. Inputs are rebuilt for every case so in-place bugs cannot leak between cases.
 */
final class TensorCases {

    private TensorCases() {
    }

    static final String[] UNARY_DOUBLE_OPS = {
        "abs", "unaryMinus", "reciprocal", "sign",
        "exp", "exp2", "expM1", "log", "log1p", "log2", "log10", "sqrt",
        "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "sigmoid", "floor", "ceil", "round",
        "logGamma", "digamma", "trigamma"
    };

    static final String[] BINARY_DOUBLE_OPS = {
        "plus", "minus", "times", "div", "pow", "atan2", "max", "min", "reverseMinus", "reverseDiv",
        "logAddExp", "logAddExp2", "safeLogTimes",
        "greaterThanMask", "greaterThanOrEqualToMask", "lessThanMask", "lessThanOrEqualToMask",
        "greaterThan", "greaterThanOrEqual", "lessThan", "lessThanOrEqual", "elementwiseEquals", "notEqualTo"
    };

    static final String[] BINARY_FIXED_POINT_OPS = {
        "plus", "minus", "times", "div", "mod", "pow", "max", "min",
        "greaterThan", "lessThan", "elementwiseEquals"
    };

    private static DoubleTensor values() {
        return DoubleTensor.create(-2.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 2.0, 3.7);
    }

    private static DoubleTensor specials() {
        return DoubleTensor.create(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
    }

    private static DoubleTensor cube() {
        // shape [2, 3, 4] with distinct values, some negative
        return DoubleTensor.arange(0, 24).minus(9.5).times(0.75).reshape(2, 3, 4);
    }

    private static DoubleTensor matrix(long seed, long rows, long cols) {
        java.util.Random random = new java.util.Random(seed);
        double[] data = new double[(int) (rows * cols)];
        for (int i = 0; i < data.length; i++) {
            // fixed decimal values keep the inputs readable and exactly representable in the JSON
            data[i] = Math.round(random.nextGaussian() * 1000) / 1000.0;
        }
        return DoubleTensor.create(data, rows, cols);
    }

    private static DoubleTensor spd(long seed, int n) {
        DoubleTensor a = matrix(seed, n, n);
        return a.matrixMultiply(a.transpose()).plus(DoubleTensor.eye(n).times(n));
    }

    static List<Map<String, Object>> generate() {
        List<Map<String, Object>> cases = new ArrayList<>();

        // element-wise unary double ops, including signed zero and the non-finite values
        for (String op : UNARY_DOUBLE_OPS) {
            add(cases, "unary/" + op, op, TensorCases::values);
            add(cases, "unary/" + op + "/specials", op, TensorCases::specials);
            add(cases, "unary/" + op + "/scalar", op, () -> DoubleTensor.scalar(0.3));
        }

        // element-wise binary double ops with broadcasting (and a shape mismatch error)
        long[][][] shapePairs = {
            {{3}, {3}},
            {{2, 3}, {3}},
            {{2, 1}, {1, 3}},
            {{}, {2, 2}},
            {{2, 2}, {}},
            {{2, 1, 3}, {4, 1}},
            {{2, 3}, {4}}
        };
        for (String op : BINARY_DOUBLE_OPS) {
            for (long[][] pair : shapePairs) {
                final long[] leftShape = pair[0];
                final long[] rightShape = pair[1];
                add(cases, "binary/" + op + "/" + shapeName(leftShape) + "x" + shapeName(rightShape), op,
                    () -> filled(leftShape, 1.25, -0.75), () -> filled(rightShape, -0.5, 0.6));
            }
            add(cases, "binary/" + op + "/specials", op, TensorCases::specials,
                () -> DoubleTensor.create(Double.NaN, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY));
            add(cases, "binary/" + op + "/signedZeros", op,
                () -> DoubleTensor.create(0.0, -0.0, 0.0, -0.0), () -> DoubleTensor.create(0.0, 0.0, -0.0, -0.0));
        }

        // reductions and scans
        String[] fullReductions = {"sum", "product", "max", "min", "mean", "standardDeviation", "argMax", "argMin", "nanArgMax", "nanArgMin"};
        for (String op : fullReductions) {
            add(cases, "reduce/" + op, op, TensorCases::cube);
            add(cases, "reduce/" + op + "/withNaN", op, () -> DoubleTensor.create(1.0, Double.NaN, 3.0, -2.0));
        }
        for (int[] dims : new int[][]{{0}, {1}, {-1}, {0, 2}, {}}) {
            for (String op : new String[]{"sum", "product", "mean"}) {
                add(cases, "reduce/" + op + "/dims" + java.util.Arrays.toString(dims), op, TensorCases::cube, () -> dims);
            }
        }
        for (int axis : new int[]{0, 1, 2, -1, -3, 3}) {
            for (String op : new String[]{"argMax", "argMin", "cumSum", "cumProd"}) {
                add(cases, "reduce/" + op + "/axis" + axis, op, TensorCases::cube, () -> axis);
            }
        }
        add(cases, "reduce/standardize", "standardize", TensorCases::cube);

        // shape manipulation
        for (long[] shape : new long[][]{{4, 6}, {24}, {2, -1}, {-1, 4, 3}, {5, -1}, {7}, {-1, -1}}) {
            add(cases, "shape/reshape/" + shapeName(shape), "reshape", TensorCases::cube, () -> shape);
        }
        for (int[] permutation : new int[][]{{0, 1, 2}, {2, 0, 1}, {1, 0, 2}, {2, 1, 0}}) {
            add(cases, "shape/permute" + java.util.Arrays.toString(permutation), "permute", TensorCases::cube, () -> permutation);
        }
        add(cases, "shape/transpose", "transpose", TensorCases::cube);
        for (long[] shape : new long[][]{{2, 2, 3, 4}, {3, 4}, {2, 1, 4}, {3}}) {
            add(cases, "shape/broadcast/" + shapeName(shape), "broadcast", () -> DoubleTensor.arange(0, 4).times(1.5), () -> shape);
        }
        add(cases, "shape/diag", "diag", () -> DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6}, 2, 3));
        for (long[] shape : new long[][]{{3, 3}, {2, 3}, {3, 2}, {2, 2, 3}, {2, 3, 2}}) {
            add(cases, "shape/diagPart/" + shapeName(shape), "diagPart",
                () -> DoubleTensor.arange(0, (double) io.improbable.keanu.tensor.TensorShape.getLength(shape)).reshape(shape));
        }
        for (int k : new int[]{-1, 0, 1, 2}) {
            add(cases, "shape/triLower/" + k, "triLower", () -> DoubleTensor.arange(1, 25).reshape(2, 3, 4), () -> k);
            add(cases, "shape/triUpper/" + k, "triUpper", () -> DoubleTensor.arange(1, 25).reshape(2, 3, 4), () -> k);
        }
        add(cases, "shape/fillTriangular/lower", "fillTriangular", () -> DoubleTensor.arange(1, 7), () -> false, () -> true);
        add(cases, "shape/fillTriangular/both", "fillTriangular", () -> DoubleTensor.arange(1, 7), () -> true, () -> true);

        // slicing with numpy semantics
        String[] slices = {
            "0", "1", "-1", ":", "::2", "1::2", "::-1", "::-2", "-2::-1", "3:1:-1", "3:3:2", "2:1:3", "1:", ":-1", "-3:",
            "0, :", ":, 1", "..., 0", "1, ..., ::-1", ":, ::2, 1:3", "5:", "-10:"
        };
        for (String slice : slices) {
            add(cases, "slice/" + slice, "slice", TensorCases::cube, () -> slice);
        }
        add(cases, "slice/vector/::-1", "slice", () -> DoubleTensor.arange(0, 5), () -> "::-1");

        // linear algebra
        add(cases, "linalg/matrixMultiply/2x3.3x4", "matrixMultiply", () -> matrix(1, 2, 3), () -> matrix(2, 3, 4));
        add(cases, "linalg/matrixMultiply/transposeLeft", "matrixMultiply", () -> matrix(1, 3, 2), () -> matrix(2, 3, 4), () -> true, () -> false);
        add(cases, "linalg/matrixMultiply/transposeRight", "matrixMultiply", () -> matrix(1, 2, 3), () -> matrix(2, 4, 3), () -> false, () -> true);
        add(cases, "linalg/matrixMultiply/transposeBoth", "matrixMultiply", () -> matrix(1, 3, 2), () -> matrix(2, 4, 3), () -> true, () -> true);
        add(cases, "linalg/matrixMultiply/batch", "matrixMultiply", () -> matrix(1, 6, 3).reshape(2, 3, 3), () -> matrix(2, 3, 2));
        add(cases, "linalg/matrixMultiply/batchBroadcast", "matrixMultiply", () -> matrix(1, 6, 2).reshape(3, 1, 2, 2), () -> matrix(2, 4, 2).reshape(2, 2, 2));
        add(cases, "linalg/matrixMultiply/mismatch", "matrixMultiply", () -> matrix(1, 2, 3), () -> matrix(2, 2, 3));
        add(cases, "linalg/matrixMultiply/vector", "matrixMultiply", () -> DoubleTensor.create(1, 2, 3), () -> matrix(2, 3, 2));
        add(cases, "linalg/matrixInverse", "matrixInverse", () -> matrix(3, 4, 4));
        add(cases, "linalg/matrixInverse/batch", "matrixInverse", () -> matrix(3, 8, 4).reshape(2, 4, 4));
        add(cases, "linalg/matrixInverse/singular", "matrixInverse", () -> DoubleTensor.create(new double[]{1, 2, 2, 4}, 2, 2));
        add(cases, "linalg/matrixDeterminant", "matrixDeterminant", () -> matrix(4, 4, 4));
        add(cases, "linalg/matrixDeterminant/batchWithSingular", "matrixDeterminant",
            () -> DoubleTensor.create(new double[]{1, 2, 2, 4, 1, 2, 3, 4}, 2, 2, 2));
        add(cases, "linalg/choleskyDecomposition", "choleskyDecomposition", () -> spd(5, 4));
        add(cases, "linalg/choleskyDecomposition/batch", "choleskyDecomposition",
            () -> DoubleTensor.concat(0, spd(5, 3).reshape(1, 3, 3), spd(6, 3).reshape(1, 3, 3)));
        add(cases, "linalg/choleskyDecomposition/notPositiveDefinite", "choleskyDecomposition",
            () -> DoubleTensor.create(new double[]{1, 2, 2, 1}, 2, 2));
        add(cases, "linalg/choleskyInverse", "choleskyInverse", () -> spd(7, 4).choleskyDecomposition());
        add(cases, "linalg/tensorMultiply/lastWithFirst", "tensorMultiply", TensorCases::cube, () -> matrix(8, 4, 2), () -> new int[]{2}, () -> new int[]{0});
        add(cases, "linalg/tensorMultiply/negativeDims", "tensorMultiply", TensorCases::cube, () -> matrix(8, 4, 2), () -> new int[]{-1}, () -> new int[]{0});
        add(cases, "linalg/tensorMultiply/twoDimsOutOfOrder", "tensorMultiply", TensorCases::cube, () -> matrix(9, 4, 3), () -> new int[]{1, 2}, () -> new int[]{1, 0});
        add(cases, "linalg/tensorMultiply/middleDim", "tensorMultiply", TensorCases::cube, () -> matrix(10, 3, 2), () -> new int[]{1}, () -> new int[]{0});
        add(cases, "linalg/tensorMultiply/mismatch", "tensorMultiply", TensorCases::cube, () -> matrix(10, 3, 2), () -> new int[]{2}, () -> new int[]{0});

        // integer and long ops
        Supplier<IntegerTensor> ints = () -> IntegerTensor.create(-7, -3, -1, 0, 1, 2, 5, 9);
        Supplier<IntegerTensor> intDivisors = () -> IntegerTensor.create(2, -2, 3, 5, -1, 3, -4, 2);
        for (String op : BINARY_FIXED_POINT_OPS) {
            add(cases, "i32/" + op, op, ints::get, intDivisors::get);
            add(cases, "i64/" + op, op, () -> toLong(ints.get()), () -> toLong(intDivisors.get()));
            add(cases, "i32/" + op + "/broadcast", op, () -> IntegerTensor.create(new int[]{1, 2, 3, 4, 5, 6}, 2, 3), () -> IntegerTensor.create(2, 3, 4));
        }
        add(cases, "i32/div/byZero", "div", () -> IntegerTensor.create(1, 2), () -> IntegerTensor.create(0, 1));
        add(cases, "i32/pow/negativeExponent", "pow", () -> IntegerTensor.create(2, 1, -1, -1, 3), () -> IntegerTensor.create(-1, -3, -3, -2, -2));
        add(cases, "i32/pow/zeroToNegative", "pow", () -> IntegerTensor.create(0, 2), () -> IntegerTensor.create(-1, 2));
        add(cases, "i32/pow/overflow", "pow", () -> IntegerTensor.create(3, 2), () -> IntegerTensor.create(40, 31));
        add(cases, "i64/pow/overflow", "pow", () -> LongTensor.create(3L, 2L), () -> LongTensor.create(40L, 63L));
        for (String op : new String[]{"abs", "unaryMinus", "sum", "product", "max", "min", "argMax", "argMin"}) {
            add(cases, "i32/" + op, op, ints::get);
            add(cases, "i64/" + op, op, () -> toLong(ints.get()));
        }
        add(cases, "i32/cumSum", "cumSum", () -> IntegerTensor.create(new int[]{1, 2, 3, 4, 5, 6}, 2, 3), () -> 1);
        add(cases, "i32/cumProd", "cumProd", () -> IntegerTensor.create(new int[]{1, 2, 3, 4, 5, 6}, 2, 3), () -> 0);
        add(cases, "i32/sum/overflow", "sum", () -> IntegerTensor.create(Integer.MAX_VALUE, 1));
        add(cases, "i32/toDouble", "toDouble", ints::get);
        add(cases, "f64/toInteger", "toInteger", TensorCases::values);

        // boolean ops
        Supplier<BooleanTensor> p = () -> BooleanTensor.create(true, true, false, false);
        Supplier<BooleanTensor> q = () -> BooleanTensor.create(true, false, true, false);
        for (String op : new String[]{"and", "or", "xor"}) {
            add(cases, "bool/" + op, op, p::get, q::get);
        }
        for (String op : new String[]{"not", "allTrue", "allFalse", "anyTrue", "anyFalse", "toDoubleMask", "toIntegerMask"}) {
            add(cases, "bool/" + op, op, p::get);
        }

        return cases;
    }

    private static LongTensor toLong(IntegerTensor t) {
        int[] values = t.asFlatIntegerArray();
        long[] longs = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            longs[i] = values[i];
        }
        return LongTensor.create(longs, t.getShape());
    }

    private static DoubleTensor filled(long[] shape, double start, double step) {
        int length = (int) io.improbable.keanu.tensor.TensorShape.getLength(shape);
        double[] data = new double[length];
        for (int i = 0; i < length; i++) {
            data[i] = start + step * i;
        }
        return DoubleTensor.create(data, shape);
    }

    private static String shapeName(long[] shape) {
        return java.util.Arrays.toString(shape).replace(" ", "");
    }

    @SafeVarargs
    private static void add(List<Map<String, Object>> cases, String name, String op, Supplier<? extends Object>... argSuppliers) {
        List<Object> args = new ArrayList<>();
        for (Supplier<?> s : argSuppliers) {
            args.add(s.get());
        }

        Object result = result(() -> {
            Object[] callArgs = new Object[argSuppliers.length - 1];
            for (int i = 1; i < argSuppliers.length; i++) {
                callArgs[i - 1] = argSuppliers[i].get();
            }
            return invoke(argSuppliers[0].get(), op, callArgs);
        });

        cases.add(testCase(name, op, args, params(), result));
    }

    static boolean isTensor(Object o) {
        return o instanceof Tensor;
    }
}
