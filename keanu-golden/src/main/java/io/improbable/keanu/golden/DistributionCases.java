package io.improbable.keanu.golden;

import io.improbable.keanu.distributions.continuous.Beta;
import io.improbable.keanu.distributions.continuous.Cauchy;
import io.improbable.keanu.distributions.continuous.ChiSquared;
import io.improbable.keanu.distributions.continuous.Dirichlet;
import io.improbable.keanu.distributions.continuous.Exponential;
import io.improbable.keanu.distributions.continuous.Gamma;
import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.distributions.continuous.InverseGamma;
import io.improbable.keanu.distributions.continuous.Laplace;
import io.improbable.keanu.distributions.continuous.LogNormal;
import io.improbable.keanu.distributions.continuous.Logistic;
import io.improbable.keanu.distributions.continuous.MultivariateGaussian;
import io.improbable.keanu.distributions.continuous.Pareto;
import io.improbable.keanu.distributions.continuous.StudentT;
import io.improbable.keanu.distributions.continuous.Triangular;
import io.improbable.keanu.distributions.continuous.Uniform;
import io.improbable.keanu.distributions.discrete.Bernoulli;
import io.improbable.keanu.distributions.discrete.Binomial;
import io.improbable.keanu.distributions.discrete.Geometric;
import io.improbable.keanu.distributions.discrete.Multinomial;
import io.improbable.keanu.distributions.discrete.Poisson;
import io.improbable.keanu.distributions.discrete.UniformInt;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.distributions.hyperparam.ParameterName;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.HalfCauchyVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.HalfGaussianVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.function.Function;

import static io.improbable.keanu.golden.Cases.list;
import static io.improbable.keanu.golden.Cases.result;
import static io.improbable.keanu.golden.Cases.testCase;

/**
 * Golden cases for distributions: the element-wise logProb and its derivatives with respect to x and each
 * parameter, evaluated over a grid of x values (inside and outside the support) for several parameter settings.
 * Gradients are element-wise and shaped like the broadcast of x and the parameters.
 */
final class DistributionCases {

    private DistributionCases() {
    }

    private static final DoubleTensor REALS = DoubleTensor.create(-3.0, -1.0, -0.25, 0.0, 0.5, 1.0, 2.5);
    private static final DoubleTensor POSITIVE = DoubleTensor.create(-1.0, 0.0, 0.1, 0.5, 1.0, 2.0, 4.5);
    private static final DoubleTensor UNIT = DoubleTensor.create(-0.5, 0.0, 0.05, 0.3, 0.5, 0.9, 1.0, 1.5);
    private static final IntegerTensor COUNTS = IntegerTensor.create(-1, 0, 1, 2, 3, 5, 8);

    private static Map<String, Object> diffs(Diffs diffs) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (Field field : Diffs.class.getFields()) {
            if (Modifier.isStatic(field.getModifiers()) && field.getType() == ParameterName.class) {
                try {
                    ParameterName name = (ParameterName) field.get(null);
                    result.put(field.getName(), diffs.get(name).getValue());
                } catch (NoSuchElementException ignored) {
                    // this distribution doesn't have that parameter
                } catch (IllegalAccessException e) {
                    throw new IllegalStateException(e);
                }
            }
        }
        return result;
    }

    private static Map<String, Object> named(Object... keyValues) {
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < keyValues.length; i += 2) {
            if (keyValues[i + 1] != null) {
                m.put((String) keyValues[i], keyValues[i + 1]);
            }
        }
        return m;
    }

    private static void add(List<Map<String, Object>> cases, String distribution, String setting,
                            Map<String, Object> parameters, Tensor<?, ?> x,
                            Function<Tensor<?, ?>, Object> logProb, Function<Tensor<?, ?>, Object> dLogProb) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("logProb", result(() -> logProb.apply(x)));
        result.put("dLogProb", result(() -> dLogProb.apply(x)));
        cases.add(testCase(distribution + "/" + setting, distribution, list(x), parameters, result));
    }

    private static DoubleTensor d(double... values) {
        return DoubleTensor.create(values);
    }

    private static DoubleTensor s(double value) {
        return DoubleTensor.scalar(value);
    }

    static List<Map<String, Object>> generate() {
        List<Map<String, Object>> cases = new ArrayList<>();

        double[][] gaussianParams = {{0, 1}, {1.5, 0.3}, {-2, 4}};
        for (double[] p : gaussianParams) {
            Gaussian dist = Gaussian.withParameters(s(p[0]), s(p[1]));
            add(cases, "Gaussian", "mu=" + p[0] + ",sigma=" + p[1], named("mu", s(p[0]), "sigma", s(p[1])), REALS,
                x -> dist.logProb((DoubleTensor) x),
                x -> {
                    DoubleTensor[] r = dist.dLogProb((DoubleTensor) x, true, true, true);
                    return named("X", r[0], "MU", r[1], "SIGMA", r[2]);
                });
        }
        DoubleTensor muVec = d(-1, 0, 2);
        DoubleTensor sigmaVec = d(0.5, 1, 2);
        Gaussian gaussianVector = Gaussian.withParameters(muVec, sigmaVec);
        add(cases, "Gaussian", "vectorParams", named("mu", muVec, "sigma", sigmaVec), d(0.1, -0.4, 3),
            x -> gaussianVector.logProb((DoubleTensor) x),
            x -> {
                DoubleTensor[] r = gaussianVector.dLogProb((DoubleTensor) x, true, true, true);
                return named("X", r[0], "MU", r[1], "SIGMA", r[2]);
            });

        double[][] betaParams = {{2, 5, 0, 1}, {0.5, 0.5, 0, 1}, {2, 3, -1, 2}};
        for (double[] p : betaParams) {
            Beta dist = Beta.withParameters(s(p[0]), s(p[1]), s(p[2]), s(p[3]));
            add(cases, "Beta", "alpha=" + p[0] + ",beta=" + p[1] + ",min=" + p[2] + ",max=" + p[3],
                named("alpha", s(p[0]), "beta", s(p[1]), "xMin", s(p[2]), "xMax", s(p[3])), UNIT,
                x -> dist.logProb((DoubleTensor) x), x -> diffs(dist.dLogProb((DoubleTensor) x)));
        }

        for (double[] p : new double[][]{{0, 1}, {2, 0.5}}) {
            Cauchy dist = Cauchy.withParameters(s(p[0]), s(p[1]));
            add(cases, "Cauchy", "location=" + p[0] + ",scale=" + p[1], named("location", s(p[0]), "scale", s(p[1])), REALS,
                x -> dist.logProb((DoubleTensor) x), x -> diffs(dist.dLogProb((DoubleTensor) x)));
        }

        for (int k : new int[]{1, 2, 5}) {
            ChiSquared dist = ChiSquared.withParameters(IntegerTensor.scalar(k));
            add(cases, "ChiSquared", "k=" + k, named("k", IntegerTensor.scalar(k)), POSITIVE,
                x -> dist.logProb((DoubleTensor) x),
                x -> named("X", dist.dLogProb((DoubleTensor) x, true)[0]));
        }

        for (double[] concentration : new double[][]{{1, 1, 1}, {0.5, 2, 3}}) {
            DoubleTensor c = d(concentration);
            Dirichlet dist = Dirichlet.withParameters(c);
            // Dirichlet takes a single simplex vector (batched x is not supported)
            for (DoubleTensor x : new DoubleTensor[]{d(0.2, 0.3, 0.5), d(0.1, 0.1, 0.8)}) {
                add(cases, "Dirichlet", "concentration=" + java.util.Arrays.toString(concentration) + ",x=" + java.util.Arrays.toString(x.asFlatDoubleArray()),
                    named("concentration", c), x,
                    v -> dist.logProb((DoubleTensor) v), v -> diffs(dist.dLogProb((DoubleTensor) v)));
            }
        }

        for (double rate : new double[]{0.5, 1, 3}) {
            Exponential dist = Exponential.withParameters(s(rate));
            add(cases, "Exponential", "rate=" + rate, named("rate", s(rate)), POSITIVE,
                x -> dist.logProb((DoubleTensor) x), x -> diffs(dist.dLogProb((DoubleTensor) x)));
        }

        for (double[] p : new double[][]{{1, 0.5}, {2, 1}, {0.5, 3}}) {
            Gamma dist = Gamma.withParameters(s(p[0]), s(p[1]));
            add(cases, "Gamma", "theta=" + p[0] + ",k=" + p[1], named("theta", s(p[0]), "k", s(p[1])), POSITIVE,
                x -> dist.logProb((DoubleTensor) x), x -> diffs(dist.dLogProb((DoubleTensor) x)));
        }

        for (double[] p : new double[][]{{1, 1}, {3, 0.5}}) {
            InverseGamma dist = InverseGamma.withParameters(s(p[0]), s(p[1]));
            add(cases, "InverseGamma", "alpha=" + p[0] + ",beta=" + p[1], named("alpha", s(p[0]), "beta", s(p[1])), POSITIVE,
                x -> dist.logProb((DoubleTensor) x), x -> diffs(dist.dLogProb((DoubleTensor) x)));
        }

        for (double[] p : new double[][]{{0, 1}, {1, 0.5}}) {
            Laplace dist = Laplace.withParameters(s(p[0]), s(p[1]));
            add(cases, "Laplace", "mu=" + p[0] + ",beta=" + p[1], named("mu", s(p[0]), "beta", s(p[1])), REALS,
                x -> dist.logProb((DoubleTensor) x), x -> diffs(dist.dLogProb((DoubleTensor) x)));

            Logistic logistic = Logistic.withParameters(s(p[0]), s(p[1]));
            add(cases, "Logistic", "mu=" + p[0] + ",s=" + p[1], named("mu", s(p[0]), "s", s(p[1])), REALS,
                x -> logistic.logProb((DoubleTensor) x), x -> diffs(logistic.dLogProb((DoubleTensor) x)));

            LogNormal logNormal = LogNormal.withParameters(s(p[0]), s(p[1]));
            add(cases, "LogNormal", "mu=" + p[0] + ",sigma=" + p[1], named("mu", s(p[0]), "sigma", s(p[1])), POSITIVE,
                x -> logNormal.logProb((DoubleTensor) x), x -> diffs(logNormal.dLogProb((DoubleTensor) x)));
        }

        DoubleTensor mvMu = d(0.5, -1);
        DoubleTensor mvCov = DoubleTensor.create(new double[]{2, 0.3, 0.3, 1}, 2, 2);
        MultivariateGaussian mvg = MultivariateGaussian.withParameters(mvMu, mvCov);
        for (DoubleTensor x : new DoubleTensor[]{d(0, 0), d(1.5, -2), DoubleTensor.create(new double[]{0, 0, 1, 1, -1, 2}, 3, 2)}) {
            add(cases, "MultivariateGaussian", "x" + java.util.Arrays.toString(x.getShape()), named("mu", mvMu, "covariance", mvCov), x,
                v -> mvg.logProb((DoubleTensor) v),
                v -> {
                    DoubleTensor[] r = mvg.dLogProb((DoubleTensor) v, true, true, true);
                    return named("X", r[0], "MU", r[1], "COVARIANCE", r[2]);
                });
        }

        for (double[] p : new double[][]{{1, 2}, {0.5, 1}}) {
            Pareto dist = Pareto.withParameters(s(p[0]), s(p[1]));
            add(cases, "Pareto", "location=" + p[0] + ",scale=" + p[1], named("location", s(p[0]), "scale", s(p[1])), POSITIVE,
                x -> dist.logProb((DoubleTensor) x), x -> diffs(dist.dLogProb((DoubleTensor) x)));
        }

        for (int v : new int[]{1, 3, 10}) {
            StudentT dist = StudentT.withParameters(IntegerTensor.scalar(v));
            add(cases, "StudentT", "v=" + v, named("v", IntegerTensor.scalar(v)), REALS,
                x -> dist.logProb((DoubleTensor) x), x -> diffs(dist.dLogProb((DoubleTensor) x)));
        }

        for (double[] p : new double[][]{{0, 1, 0.5}, {-1, 2, 0}, {-1, 2, 2}, {0, 1, 0.25}}) {
            Triangular dist = Triangular.withParameters(s(p[0]), s(p[1]), s(p[2]));
            add(cases, "Triangular", "min=" + p[0] + ",max=" + p[1] + ",c=" + p[2],
                named("xMin", s(p[0]), "xMax", s(p[1]), "c", s(p[2])),
                d(-1.5, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 2, 2.5),
                x -> dist.logProb((DoubleTensor) x), x -> diffs(dist.dLogProb((DoubleTensor) x)));
        }

        Uniform uniform = Uniform.withParameters(s(-1), s(2));
        add(cases, "Uniform", "min=-1,max=2", named("xMin", s(-1), "xMax", s(2)), REALS,
            x -> uniform.logProb((DoubleTensor) x), x -> diffs(uniform.dLogProb((DoubleTensor) x)));

        for (double p : new double[]{0.2, 0.5, 1.0}) {
            Bernoulli dist = Bernoulli.withParameters(s(p));
            add(cases, "Bernoulli", "p=" + p, named("probTrue", s(p)), BooleanTensor.create(true, false, true),
                x -> dist.logProb((BooleanTensor) x), x -> named("PROB_TRUE", dist.dLogProb((BooleanTensor) x)));
        }

        for (double[] p : new double[][]{{0.3, 5}, {0.5, 3}, {0.0, 4}}) {
            Binomial dist = Binomial.withParameters(s(p[0]), IntegerTensor.scalar((int) p[1]));
            add(cases, "Binomial", "p=" + p[0] + ",n=" + (int) p[1], named("p", s(p[0]), "n", IntegerTensor.scalar((int) p[1])), COUNTS,
                x -> dist.logProb((IntegerTensor) x), x -> named("P", dist.dLogProb((IntegerTensor) x, true)[0]));
        }

        for (double p : new double[]{0.25, 0.9}) {
            Geometric dist = Geometric.withParameters(s(p));
            add(cases, "Geometric", "p=" + p, named("p", s(p)), COUNTS,
                x -> dist.logProb((IntegerTensor) x), x -> named("P", dist.dLogProb((IntegerTensor) x, true)[0]));
        }

        for (DoubleTensor p : new DoubleTensor[]{d(0.2, 0.5, 0.3), d(2, 1, 3)}) {
            Multinomial dist = Multinomial.withParameters(IntegerTensor.scalar(5), p);
            add(cases, "Multinomial", "p=" + java.util.Arrays.toString(p.asFlatDoubleArray()), named("n", IntegerTensor.scalar(5), "p", p),
                IntegerTensor.create(1, 3, 1),
                x -> dist.logProb((IntegerTensor) x), x -> named("P", dist.dLogProb((IntegerTensor) x)));
        }

        for (double mu : new double[]{0.5, 3, 20}) {
            Poisson dist = Poisson.withParameters(s(mu));
            add(cases, "Poisson", "mu=" + mu, named("mu", s(mu)), COUNTS,
                x -> dist.logProb((IntegerTensor) x), x -> named("MU", dist.dLogProb((IntegerTensor) x, true)[0]));
        }

        add(cases, "UniformInt", "min=-2,max=3", named("xMin", IntegerTensor.scalar(-2), "xMax", IntegerTensor.scalar(3)), COUNTS,
            x -> UniformInt.withParameters(IntegerTensor.scalar(-2), IntegerTensor.scalar(3)).logProb((IntegerTensor) x),
            x -> {
                throw new UnsupportedOperationException("discrete");
            });

        // distributions that only exist as vertices: logProb is the total over x, gradients come from the vertex
        for (double scale : new double[]{0.5, 2}) {
            addVertexCase(cases, "HalfGaussian", "sigma=" + scale, scale, POSITIVE,
                param -> new HalfGaussianVertex(POSITIVE.getShape(), param));
            addVertexCase(cases, "HalfCauchy", "scale=" + scale, scale, POSITIVE,
                param -> new HalfCauchyVertex(POSITIVE.getShape(), param));
        }

        return cases;
    }

    private static void addVertexCase(List<Map<String, Object>> cases, String distribution, String setting,
                                      double parameter, DoubleTensor x,
                                      Function<UniformVertex, io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.ProbabilisticDouble> vertexFactory) {
        UniformVertex param = new UniformVertex(0.0, 100.0);
        param.setValue(parameter);
        io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.ProbabilisticDouble vertex = vertexFactory.apply(param);

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("totalLogProb", result(() -> vertex.logProb(x)));
        result.put("dTotalLogProb", result(() -> {
            Map<Vertex, DoubleTensor> d = vertex.dLogProb(x, param, (Vertex) vertex);
            return named("X", d.get(vertex), "PARAM", d.get(param));
        }));
        cases.add(testCase(distribution + "/" + setting, distribution, list(x), named("param", s(parameter)), result));
    }
}
