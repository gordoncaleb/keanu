package io.improbable.keanu.golden;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.model.ModelScoring;
import io.improbable.keanu.model.regression.RegressionModel;
import io.improbable.keanu.model.regression.RegressionModelBuilder;
import io.improbable.keanu.model.regression.RegressionRegularization;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.VertexId;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static io.improbable.keanu.golden.Cases.list;
import static io.improbable.keanu.golden.Cases.result;
import static io.improbable.keanu.golden.Cases.testCase;

/**
 * Golden cases for the regression models (Java's model package): fitted weights and intercepts, predictions on held
 * out inputs, and scores, for linear and logistic regression with each regularization. The data is generated from
 * fixed formulas (no random numbers) so the cases are reproducible.
 */
final class RegressionCases {

    private RegressionCases() {
    }

    /**
     * A deterministic pseudo-uniform number in [0, 1).
     */
    private static double hash(int i, int salt) {
        double v = Math.sin(i * 12.9898 + salt * 78.233) * 43758.5453;
        return v - Math.floor(v);
    }

    private static DoubleTensor features(int rows, int features, int salt) {
        double[] data = new double[rows * features];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < features; j++) {
                data[i * features + j] = 10.0 * hash(i, salt + 31 * j) - 5.0 + j;
            }
        }
        return DoubleTensor.create(data, rows, features);
    }

    private static DoubleTensor linearPredictor(DoubleTensor x, double[] weights, double intercept) {
        return x.matrixMultiply(DoubleTensor.create(weights, weights.length, 1)).plus(intercept);
    }

    private static DoubleTensor linearOutput(DoubleTensor x, double[] weights, double intercept, int salt) {
        DoubleTensor mean = linearPredictor(x, weights, intercept);
        double[] noise = new double[(int) mean.getLength()];
        for (int i = 0; i < noise.length; i++) {
            noise[i] = 0.8 * (hash(i, salt + 7) - 0.5);
        }
        return mean.plus(DoubleTensor.create(noise, mean.getShape()));
    }

    private static BooleanTensor logisticOutput(DoubleTensor x, double[] weights, double intercept, int salt) {
        double[] p = linearPredictor(x, weights, intercept).sigmoid().asFlatDoubleArray();
        boolean[] y = new boolean[p.length];
        for (int i = 0; i < p.length; i++) {
            y[i] = hash(i, salt + 11) < p[i];
        }
        return BooleanTensor.create(y, new long[]{p.length, 1});
    }

    private interface Configure {
        RegressionModelBuilder apply(RegressionModelBuilder builder);
    }

    static List<Map<String, Object>> generate() {
        List<Map<String, Object>> cases = new ArrayList<>();

        double[][] weightSets = {{2.5}, {1.5, -2.0, 0.5}};
        RegressionRegularization[] regularizations = RegressionRegularization.values();

        for (double[] weights : weightSets) {
            int features = weights.length;
            DoubleTensor xTrain = features(60, features, 1);
            DoubleTensor xTest = features(10, features, 2);
            DoubleTensor yLinear = linearOutput(xTrain, weights, 4.0, 3);
            DoubleTensor yLinearTest = linearOutput(xTest, weights, 4.0, 4);
            BooleanTensor yLogistic = logisticOutput(xTrain, weights, 0.5, 5);
            BooleanTensor yLogisticTest = logisticOutput(xTest, weights, 0.5, 6);

            for (RegressionRegularization regularization : regularizations) {
                String base = "/" + features + "features/" + regularization.name().toLowerCase();

                cases.add(linear("linear" + base + "/defaultPriors", xTrain, yLinear, xTest, yLinearTest, regularization, null,
                    b -> b));
                cases.add(linear("linear" + base + "/priors(0,0.5)", xTrain, yLinear, xTest, yLinearTest, regularization, new double[]{0, 0.5},
                    b -> b.withPriorOnWeightsAndIntercept(0, 0.5)));
                cases.add(logistic("logistic" + base + "/defaultPriors", xTrain, yLogistic, xTest, yLogisticTest, regularization, null,
                    b -> b));
                cases.add(logistic("logistic" + base + "/priors(0,0.5)", xTrain, yLogistic, xTest, yLogisticTest, regularization, new double[]{0, 0.5},
                    b -> b.withPriorOnWeightsAndIntercept(0, 0.5)));
            }
        }

        // rank 1 training data is samples of a single feature
        DoubleTensor xVector = DoubleTensor.create(0, 1, 2, 3, 4, 5, 6, 7);
        DoubleTensor yVector = xVector.times(2.0).plus(1.0);
        cases.add(linear("linear/rank1/none/defaultPriors", xVector, yVector, DoubleTensor.create(10.0, 20.0), DoubleTensor.create(21.0, 41.0),
            RegressionRegularization.NONE, null, b -> b));

        // scores
        DoubleTensor predicted = DoubleTensor.create(1.0, 2.5, 2.9, 4.2);
        DoubleTensor actual = DoubleTensor.create(1.1, 2.0, 3.0, 4.0);
        cases.add(testCase("score/coefficientOfDetermination", "coefficientOfDetermination", list(predicted, actual), null,
            result(() -> ModelScoring.coefficientOfDetermination(predicted, actual))));
        BooleanTensor predictedClasses = BooleanTensor.create(true, true, false, false, true);
        BooleanTensor actualClasses = BooleanTensor.create(true, false, false, true, true);
        cases.add(testCase("score/accuracy", "accuracy", list(predictedClasses, actualClasses), null,
            result(() -> ModelScoring.accuracy(predictedClasses, actualClasses))));

        return cases;
    }

    private static Map<String, Object> settings(RegressionRegularization regularization, double[] priors) {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("regularization", regularization.name());
        if (priors != null) {
            p.put("priorMean", priors[0]);
            p.put("priorScale", priors[1]);
        }
        return p;
    }

    private static Map<String, Object> linear(String name, DoubleTensor x, DoubleTensor y, DoubleTensor xTest, DoubleTensor yTest,
                                              RegressionRegularization regularization, double[] priors, Configure configure) {
        VertexId.resetIdGenerator();
        // Java's fitters start from a draw from the prior, using the global default random
        KeanuRandom.setDefaultRandomSeed(1);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("fit", result(() -> {
            RegressionModel<DoubleTensor> model = (RegressionModel<DoubleTensor>) configure.apply(
                RegressionModel.withTrainingData(x, y).withRegularization(regularization)).build();
            Map<String, Object> fit = new LinkedHashMap<>();
            fit.put("weights", Cases.encode(model.getWeightVertex().getValue()));
            fit.put("intercept", Cases.encode(model.getInterceptVertex().getValue()));
            DoubleTensor prediction = model.predict(xTest);
            fit.put("prediction", Cases.encode(prediction));
            fit.put("coefficientOfDetermination", ModelScoring.coefficientOfDetermination(prediction, yTest.reshape(prediction.getShape())));
            return fit;
        }));
        return testCase(name, "linearRegression", list(x, y, xTest, yTest), settings(regularization, priors), result);
    }

    private static Map<String, Object> logistic(String name, DoubleTensor x, BooleanTensor y, DoubleTensor xTest, BooleanTensor yTest,
                                                RegressionRegularization regularization, double[] priors, Configure configure) {
        VertexId.resetIdGenerator();
        // Java's fitters start from a draw from the prior, using the global default random
        KeanuRandom.setDefaultRandomSeed(1);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("fit", result(() -> {
            RegressionModel<BooleanTensor> model = (RegressionModel<BooleanTensor>) configure.apply(
                RegressionModel.withTrainingData(x, y).withRegularization(regularization)).build();
            Map<String, Object> fit = new LinkedHashMap<>();
            fit.put("weights", Cases.encode(model.getWeightVertex().getValue()));
            fit.put("intercept", Cases.encode(model.getInterceptVertex().getValue()));
            BooleanTensor prediction = model.predict(xTest);
            fit.put("prediction", Cases.encode(prediction));
            fit.put("accuracy", ModelScoring.accuracy(prediction, yTest));
            return fit;
        }));
        return testCase(name, "logisticRegression", list(x, y, xTest, yTest), settings(regularization, priors), result);
    }
}
