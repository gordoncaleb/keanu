package io.improbable.keanu.golden;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.network.NetworkLoader;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.util.io.JsonLoader;
import io.improbable.keanu.util.io.JsonSaver;
import io.improbable.keanu.util.io.ProtobufLoader;
import io.improbable.keanu.util.io.ProtobufSaver;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic.BinomialVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic.PoissonVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.BetaVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.ExponentialVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GammaVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.HalfGaussianVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Golden cases for whole models. Each model is saved in Keanu's protobuf and JSON formats, together with the
 * expected log prob, log likelihood and gradients (keyed by vertex label) at the saved latent values.
 * <p>
 * Every model is reloaded from both formats in Java and must reproduce the same log prob, so a broken
 * save/load round trip fails the export rather than producing bad vectors.
 */
final class ModelCases {

    private ModelCases() {
    }

    /**
     * The models used for inference reference results (every golden model except the vertex catalog).
     */
    static Map<String, Supplier<BayesianNetwork>> inferenceModels() {
        Map<String, Supplier<BayesianNetwork>> models = new LinkedHashMap<>();
        models.put("thermometers", ModelCases::thermometers);
        models.put("hierarchicalGaussian", ModelCases::hierarchicalGaussian);
        models.put("linearRegression", ModelCases::linearRegression);
        models.put("betaBinomial", ModelCases::betaBinomial);
        models.put("gammaPoisson", ModelCases::gammaPoisson);
        models.put("randomWalkSequence", ModelCases::randomWalkSequence);
        return models;
    }

    /**
     * Builds a golden model with fresh, deterministic vertex ids and sequence labels.
     */
    static BayesianNetwork fresh(Supplier<BayesianNetwork> builder) {
        VertexId.resetIdGenerator();
        io.improbable.keanu.templating.SequenceBuilder.resetSequenceIdentifiers();
        return builder.get();
    }

    static String labelOf(Vertex vertex) {
        return name(vertex);
    }

    static List<Map<String, Object>> generate(Path modelsDir) throws IOException {
        Files.createDirectories(modelsDir);

        Map<String, Supplier<BayesianNetwork>> models = new LinkedHashMap<>(inferenceModels());
        models.put("catalog", CatalogModel::build);

        List<Map<String, Object>> cases = new ArrayList<>();
        for (Map.Entry<String, Supplier<BayesianNetwork>> entry : models.entrySet()) {
            cases.add(export(entry.getKey(), entry.getValue(), modelsDir));
        }
        return cases;
    }

    private static Map<String, Object> export(String name, Supplier<BayesianNetwork> builder, Path modelsDir) throws IOException {
        // fresh, deterministic vertex ids and sequence labels for every model
        VertexId.resetIdGenerator();
        io.improbable.keanu.templating.SequenceBuilder.resetSequenceIdentifiers();
        BayesianNetwork network = builder.get();

        Map<String, String> metadata = Collections.singletonMap("name", name);

        ByteArrayOutputStream protobuf = new ByteArrayOutputStream();
        new ProtobufSaver(network).save(protobuf, true, metadata);
        Files.write(modelsDir.resolve(name + ".pb"), protobuf.toByteArray());

        ByteArrayOutputStream json = new ByteArrayOutputStream();
        new JsonSaver(network).save(json, true, metadata);
        Files.write(modelsDir.resolve(name + ".json"), json.toByteArray());

        Map<String, Object> expected = evaluate(network);

        // round trip in Java: both formats must give the same answers
        checkRoundTrip(name, "protobuf", new ProtobufLoader(), protobuf.toByteArray(), expected);
        checkRoundTrip(name, "json", new JsonLoader(), json.toByteArray(), expected);

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("name", name);
        result.put("protobuf", "models/" + name + ".pb");
        result.put("json", "models/" + name + ".json");
        result.put("expected", expected);
        return result;
    }

    private static void checkRoundTrip(String name, String format, NetworkLoader loader, byte[] bytes, Map<String, Object> expected) throws IOException {
        VertexId.resetIdGenerator();
        BayesianNetwork reloaded = loader.loadNetwork(new ByteArrayInputStream(bytes));
        Map<String, Object> actual = evaluate(reloaded);
        // gradient contributions are summed in hash set order, which depends on vertex ids, so allow for the last bits
        if (!approximatelyEqual(actual, expected)) {
            throw new IllegalStateException("Model " + name + " does not round trip through " + format
                + "\nexpected " + Json.write(expected) + "\nactual " + Json.write(actual));
        }
    }

    private static Map<String, Object> evaluate(BayesianNetwork network) {
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(network);
        KeanuProbabilisticModelWithGradient gradientModel = new KeanuProbabilisticModelWithGradient(network);

        List<Vertex> latents = new ArrayList<>(network.getLatentVertices());
        latents.sort(Comparator.comparing(ModelCases::name));

        Map<String, Object> latentValues = new LinkedHashMap<>();
        for (Vertex latent : latents) {
            latentValues.put(name(latent), Cases.encode(latent.getValue()));
        }

        Map<VariableReference, DoubleTensor> logProbGradients = gradientModel.logProbGradients();
        Map<VariableReference, DoubleTensor> logLikelihoodGradients = gradientModel.logLikelihoodGradients();

        Map<String, Object> dLogProb = new LinkedHashMap<>();
        Map<String, Object> dLogLikelihood = new LinkedHashMap<>();
        for (Vertex latent : latents) {
            String label = name(latent);
            if (logProbGradients.containsKey(latent.getId())) {
                dLogProb.put(label, Cases.encode(logProbGradients.get(latent.getId())));
            }
            if (logLikelihoodGradients.containsKey(latent.getId())) {
                dLogLikelihood.put(label, Cases.encode(logLikelihoodGradients.get(latent.getId())));
            }
        }

        // the value of every labeled vertex, so deterministic nodes are checked too
        List<Vertex> labeled = new ArrayList<>();
        for (Vertex vertex : network.getAllVertices()) {
            if (vertex.getLabel() != null) {
                labeled.add(vertex);
            }
        }
        labeled.sort(Comparator.comparing(ModelCases::name));
        Map<String, Object> nodeValues = new LinkedHashMap<>();
        for (Vertex vertex : labeled) {
            nodeValues.put(name(vertex), Cases.encode(vertex.getValue()));
        }

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("latentValues", latentValues);
        result.put("nodeValues", nodeValues);
        result.put("logProb", model.logProb());
        result.put("logLikelihood", model.logLikelihood());
        result.put("dLogProb", dLogProb);
        result.put("dLogLikelihood", dLogLikelihood);
        return result;
    }

    /**
     * The vertex label, which for vertices created by a Sequence includes the sequence and item prefix.
     */
    private static String name(Vertex vertex) {
        if (vertex.getLabel() == null) {
            throw new IllegalStateException("Every latent in a golden model needs a label: " + vertex);
        }
        return vertex.getLabel().toString();
    }

    /**
     * A Gaussian random walk built with a Sequence (plate): x[t] ~ N(x[t-1], 0.5), y[t] ~ N(x[t], 1) observed.
     */
    private static BayesianNetwork randomWalkSequence() {
        io.improbable.keanu.vertices.VertexLabel xLabel = new io.improbable.keanu.vertices.VertexLabel("x");
        io.improbable.keanu.vertices.VertexLabel yLabel = new io.improbable.keanu.vertices.VertexLabel("y");
        double[] observations = {0.3, 0.9, 0.4, 1.6};
        double[] latentValues = {0.2, 0.7, 0.6, 1.4};
        int[] step = {0};

        io.improbable.keanu.templating.Sequence sequence = new io.improbable.keanu.templating.SequenceBuilder<Integer>()
            .withInitialState(xLabel, ConstantVertex.of(0.0))
            .count(observations.length)
            .withFactory(item -> {
                DoubleVertex xPrevious = item.addDoubleProxyFor(xLabel);
                GaussianVertex x = new GaussianVertex(xPrevious, 0.5);
                x.setValue(latentValues[step[0]]);
                GaussianVertex y = new GaussianVertex(x, 1.0);
                y.observe(observations[step[0]]);
                item.add(xLabel, x);
                item.add(yLabel, y);
                step[0]++;
            })
            .build();

        return sequence.toBayesianNetwork();
    }

    private static BayesianNetwork thermometers() {
        UniformVertex temperature = new UniformVertex(20.0, 30.0);
        temperature.setLabel("temperature");
        temperature.setValue(25.3);

        GaussianVertex first = new GaussianVertex(temperature, 2.5);
        first.setLabel("thermometerOne");
        first.observe(25.0);

        GaussianVertex second = new GaussianVertex(temperature, 5.0);
        second.setLabel("thermometerTwo");
        second.observe(30.0);

        return new BayesianNetwork(temperature.getConnectedGraph());
    }

    private static BayesianNetwork hierarchicalGaussian() {
        GaussianVertex mu = new GaussianVertex(0, 10);
        mu.setLabel("mu");
        mu.setValue(1.2);

        HalfGaussianVertex sigma = new HalfGaussianVertex(2.0);
        sigma.setLabel("sigma");
        sigma.setValue(0.8);

        GaussianVertex groupMeans = new GaussianVertex(new long[]{3}, mu, sigma);
        groupMeans.setLabel("groupMeans");
        groupMeans.setValue(DoubleTensor.create(0.5, 1.5, 2.0));

        GaussianVertex observations = new GaussianVertex(new long[]{2, 3}, groupMeans, 1.0);
        observations.setLabel("observations");
        observations.observe(DoubleTensor.create(new double[]{0.4, 1.9, 2.2, 0.7, 1.1, 2.6}, 2, 3));

        return new BayesianNetwork(mu.getConnectedGraph());
    }

    private static BayesianNetwork linearRegression() {
        DoubleVertex x = ConstantVertex.of(DoubleTensor.create(new double[]{1, 0.5, 2, -1, 3, 0.2, -0.5, 1.5}, 4, 2));
        x.setLabel("x");

        GaussianVertex weights = new GaussianVertex(new long[]{2, 1}, 0, 1);
        weights.setLabel("weights");
        weights.setValue(DoubleTensor.create(new double[]{0.7, -1.1}, 2, 1));

        GaussianVertex intercept = new GaussianVertex(0, 5);
        intercept.setLabel("intercept");
        intercept.setValue(0.3);

        ExponentialVertex noise = new ExponentialVertex(2.0);
        noise.setLabel("noise");
        noise.setValue(0.6);

        DoubleVertex yHat = x.matrixMultiply(weights).plus(intercept);
        GaussianVertex y = new GaussianVertex(yHat, noise);
        y.setLabel("y");
        y.observe(DoubleTensor.create(new double[]{0.1, 3.2, 1.5, -1.0}, 4, 1));

        return new BayesianNetwork(y.getConnectedGraph());
    }

    private static BayesianNetwork betaBinomial() {
        BetaVertex p = new BetaVertex(2.0, 3.0);
        p.setLabel("p");
        p.setValue(0.35);

        BinomialVertex successes = new BinomialVertex(new long[]{4}, p, 10);
        successes.setLabel("successes");
        successes.observe(IntegerTensor.create(3, 5, 2, 4));

        return new BayesianNetwork(p.getConnectedGraph());
    }

    private static BayesianNetwork gammaPoisson() {
        GammaVertex rate = new GammaVertex(2.0, 1.5);
        rate.setLabel("rate");
        rate.setValue(2.4);

        PoissonVertex counts = new PoissonVertex(new long[]{5}, rate);
        counts.setLabel("counts");
        counts.observe(IntegerTensor.create(1, 4, 2, 0, 3));

        IntegerVertex total = counts.sum();
        total.setLabel("total");

        return new BayesianNetwork(rate.getConnectedGraph());
    }

    private static boolean approximatelyEqual(Object a, Object b) {
        if (a instanceof Map && b instanceof Map) {
            Map<?, ?> ma = (Map<?, ?>) a;
            Map<?, ?> mb = (Map<?, ?>) b;
            return ma.keySet().equals(mb.keySet()) && ma.keySet().stream().allMatch(k -> approximatelyEqual(ma.get(k), mb.get(k)));
        }
        if (a instanceof double[] && b instanceof double[]) {
            double[] da = (double[]) a;
            double[] db = (double[]) b;
            if (da.length != db.length) {
                return false;
            }
            for (int i = 0; i < da.length; i++) {
                if (!approximatelyEqual((Object) da[i], (Object) db[i])) {
                    return false;
                }
            }
            return true;
        }
        if (a instanceof Double && b instanceof Double) {
            double x = (Double) a;
            double y = (Double) b;
            return Double.compare(x, y) == 0 || Math.abs(x - y) <= 1e-12 * Math.max(1.0, Math.abs(y));
        }
        if (a instanceof long[] && b instanceof long[]) {
            return java.util.Arrays.equals((long[]) a, (long[]) b);
        }
        if (a instanceof int[] && b instanceof int[]) {
            return java.util.Arrays.equals((int[]) a, (int[]) b);
        }
        if (a instanceof boolean[] && b instanceof boolean[]) {
            return java.util.Arrays.equals((boolean[]) a, (boolean[]) b);
        }
        return java.util.Objects.equals(a, b);
    }
}
