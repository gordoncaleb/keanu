package io.improbable.keanu.golden;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiator;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static io.improbable.keanu.golden.Cases.invoke;
import static io.improbable.keanu.golden.Cases.result;
import static io.improbable.keanu.golden.Cases.testCase;

/**
 * Golden cases for vertex graphs: the output value and the partial derivative of the output with respect to each
 * input, from both forward and reverse mode auto diff. Partials are shaped [output shape..., input shape...].
 */
final class VertexCases {

    private VertexCases() {
    }

    private static final List<String> SKIPPED = new ArrayList<>();

    static List<String> skipped() {
        return SKIPPED;
    }

    private static DoubleTensor inRange(double low, double high, long... shape) {
        int length = (int) io.improbable.keanu.tensor.TensorShape.getLength(shape);
        double[] data = new double[length];
        for (int i = 0; i < length; i++) {
            data[i] = low + (high - low) * (i + 1) / (length + 1);
        }
        return DoubleTensor.create(data, shape);
    }

    private static DoubleTensor spd(int n) {
        DoubleTensor a = DoubleTensor.arange(1, n * n + 1).reshape(n, n).sin();
        return a.matrixMultiply(a.transpose()).plus(DoubleTensor.eye(n).times(n));
    }

    static List<Map<String, Object>> generate() {
        List<Map<String, Object>> cases = new ArrayList<>();

        // unary element-wise ops on a [2, 3] input inside each op's domain
        Map<String, double[]> domains = new LinkedHashMap<>();
        for (String op : TensorCases.UNARY_DOUBLE_OPS) {
            domains.put(op, new double[]{0.2, 2.5});
        }
        domains.put("unaryMinus", new double[]{-2, 2});
        domains.put("abs", new double[]{-2, 2});
        domains.put("asin", new double[]{-0.8, 0.8});
        domains.put("acos", new double[]{-0.8, 0.8});
        domains.put("atanh", new double[]{-0.8, 0.8});
        domains.put("acosh", new double[]{1.2, 3});
        domains.put("tan", new double[]{-1.2, 1.2});
        for (Map.Entry<String, double[]> entry : domains.entrySet()) {
            String op = entry.getKey();
            double[] range = entry.getValue();
            add(cases, "unary/" + op, op, inputs(inRange(range[0], range[1], 2, 3)), in -> (DoubleVertex) invoke(in.get(0), op));
        }

        // binary ops with broadcasting
        for (String op : new String[]{"plus", "minus", "times", "div", "pow", "atan2", "logAddExp", "logAddExp2", "max", "min"}) {
            add(cases, "binary/" + op + "/[2,3]x[2,3]", op, inputs(inRange(0.3, 2, 2, 3), inRange(0.4, 1.7, 2, 3).reverseMinus(2.1)),
                in -> (DoubleVertex) invoke(in.get(0), op, in.get(1)));
            add(cases, "binary/" + op + "/[2,3]x[3]", op, inputs(inRange(0.3, 2, 2, 3), inRange(0.5, 1.5, 3)),
                in -> (DoubleVertex) invoke(in.get(0), op, in.get(1)));
            add(cases, "binary/" + op + "/[]x[2,2]", op, inputs(DoubleTensor.scalar(1.3), inRange(0.5, 1.5, 2, 2)),
                in -> (DoubleVertex) invoke(in.get(0), op, in.get(1)));
        }

        // reductions, scans and shape ops
        DoubleTensor cube = inRange(-1.5, 2.5, 2, 3, 4);
        add(cases, "reduce/sum", "sum", inputs(cube), in -> in.get(0).sum());
        add(cases, "reduce/sum/dims[0,2]", "sum", inputs(cube), in -> in.get(0).sum(0, 2));
        add(cases, "reduce/sum/dims[-1]", "sum", inputs(cube), in -> in.get(0).sum(-1));
        add(cases, "reduce/product", "product", inputs(inRange(0.5, 1.5, 2, 3)), in -> in.get(0).product());
        add(cases, "reduce/mean", "mean", inputs(cube), in -> in.get(0).mean());
        add(cases, "reduce/mean/dims[1]", "mean", inputs(cube), in -> in.get(0).mean(1));
        add(cases, "reduce/max", "max", inputs(cube), in -> in.get(0).max());
        add(cases, "reduce/min", "min", inputs(cube), in -> in.get(0).min());
        add(cases, "reduce/standardDeviation", "standardDeviation", inputs(cube), in -> in.get(0).standardDeviation());
        add(cases, "reduce/standardize", "standardize", inputs(inRange(-1, 3, 2, 3)), in -> in.get(0).standardize());
        add(cases, "reduce/cumSum/1", "cumSum", inputs(cube), in -> in.get(0).cumSum(1));
        add(cases, "reduce/cumProd/-1", "cumProd", inputs(inRange(-1.5, 2.5, 2, 3)), in -> in.get(0).cumProd(-1));
        add(cases, "reduce/cumProd/withZero", "cumProd", inputs(DoubleTensor.create(1.5, 0.0, -2.0, 3.0)), in -> in.get(0).cumProd(0));

        add(cases, "shape/reshape", "reshape", inputs(cube), in -> in.get(0).reshape(4, 6));
        add(cases, "shape/permute", "permute", inputs(cube), in -> in.get(0).permute(2, 0, 1));
        add(cases, "shape/transpose", "transpose", inputs(inRange(-1, 1, 2, 3)), in -> in.get(0).transpose());
        add(cases, "shape/slice", "slice", inputs(cube), in -> in.get(0).slice("1, ::-1, 1:3"));
        add(cases, "shape/diag", "diag", inputs(inRange(-1, 1, 3)), in -> in.get(0).diag());
        add(cases, "shape/diagPart", "diagPart", inputs(inRange(-1, 1, 2, 3, 3)), in -> in.get(0).diagPart());
        add(cases, "shape/triLower", "triLower", inputs(inRange(-1, 1, 3, 3)), in -> in.get(0).triLower(0));
        add(cases, "shape/triUpper", "triUpper", inputs(inRange(-1, 1, 3, 3)), in -> in.get(0).triUpper(1));
        add(cases, "shape/broadcast", "broadcast", inputs(inRange(-1, 1, 3)), in -> in.get(0).broadcast(2, 3));
        add(cases, "shape/fillTriangular", "fillTriangular", inputs(inRange(-1, 1, 6)), in -> in.get(0).fillTriangular(true, true));

        // linear algebra
        add(cases, "linalg/matrixMultiply", "matrixMultiply", inputs(inRange(-1, 1, 2, 3), inRange(-1, 2, 3, 4)),
            in -> in.get(0).matrixMultiply(in.get(1)));
        add(cases, "linalg/matrixMultiply/batch", "matrixMultiply", inputs(inRange(-1, 1, 2, 2, 3), inRange(-1, 2, 3, 2)),
            in -> in.get(0).matrixMultiply(in.get(1)));
        add(cases, "linalg/tensorMultiply", "tensorMultiply", inputs(cube, inRange(-1, 2, 4, 3)),
            in -> in.get(0).tensorMultiply(in.get(1), new int[]{1, 2}, new int[]{1, 0}));
        add(cases, "linalg/matrixInverse", "matrixInverse", inputs(spd(3)), in -> in.get(0).matrixInverse());
        add(cases, "linalg/matrixDeterminant", "matrixDeterminant", inputs(spd(3)), in -> in.get(0).matrixDeterminant());
        add(cases, "linalg/choleskyDecomposition", "choleskyDecomposition", inputs(spd(3)), in -> in.get(0).choleskyDecomposition());
        add(cases, "linalg/choleskyInverse", "choleskyInverse", inputs(spd(3).choleskyDecomposition()), in -> in.get(0).choleskyInverse());

        // composite graphs
        add(cases, "graph/sharedSubexpression", "graph", inputs(inRange(0.5, 1.5, 3), DoubleTensor.scalar(0.7)),
            in -> {
                DoubleVertex shared = in.get(0).times(in.get(1)).sin();
                return shared.times(shared).plus(shared.exp()).sum();
            });
        add(cases, "graph/logSumExpLike", "graph", inputs(inRange(-2, 2, 4)),
            in -> in.get(0).exp().sum().log().minus(in.get(0).max()));
        add(cases, "graph/quadraticForm", "graph", inputs(inRange(-1, 1, 3), spd(3)),
            in -> {
                DoubleVertex x = in.get(0).reshape(3, 1);
                return x.transpose().matrixMultiply(in.get(1).matrixInverse()).matrixMultiply(x).sum();
            });
        add(cases, "graph/withConstant", "graph", inputs(inRange(0.1, 0.9, 2, 2)),
            in -> in.get(0).times(ConstantVertex.of(DoubleTensor.create(new double[]{1, -2, 3, -4}, 2, 2))).sigmoid().sum());

        return cases;
    }

    private static List<DoubleTensor> inputs(DoubleTensor... values) {
        List<DoubleTensor> list = new ArrayList<>();
        for (DoubleTensor v : values) {
            list.add(v);
        }
        return list;
    }

    private static void add(List<Map<String, Object>> cases, String name, String op, List<DoubleTensor> inputValues,
                            Function<List<DoubleVertex>, DoubleVertex> graph) {
        List<DoubleVertex> inputs = new ArrayList<>();
        for (DoubleTensor value : inputValues) {
            UniformVertex in = new UniformVertex(value.getShape(), -1000.0, 1000.0);
            in.setValue(value);
            inputs.add(in);
        }

        final DoubleVertex output;
        try {
            output = graph.apply(inputs);
        } catch (IllegalStateException e) {
            if (e.getMessage() != null && e.getMessage().startsWith("No method")) {
                SKIPPED.add(name);
                return;
            }
            throw e;
        } catch (RuntimeException e) {
            // the graph can't be built, e.g. a shape the vertex rejects; record that as the expected result
            Throwable cause = e instanceof Cases.ReflectiveCallException ? e.getCause() : e;
            Map<String, Object> error = new LinkedHashMap<>();
            error.put("error", cause.getClass().getSimpleName());
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("value", error);
            cases.add(testCase(name, op, new ArrayList<>(inputValues), null, result));
            return;
        }

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("value", result(output::eval));

        List<Object> forward = new ArrayList<>();
        List<Object> reverse = new ArrayList<>();
        for (DoubleVertex input : inputs) {
            forward.add(result(() -> Differentiator.forwardModeAutoDiff(input, output).of(output)));
            reverse.add(result(() -> Differentiator.reverseModeAutoDiff(output, input).withRespectTo(input)));
        }
        result.put("forwardPartials", forward);
        result.put("reversePartials", reverse);

        cases.add(testCase(name, op, new ArrayList<>(inputValues), null, result));
    }
}
