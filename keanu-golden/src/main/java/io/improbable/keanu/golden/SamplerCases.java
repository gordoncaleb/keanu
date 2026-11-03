package io.improbable.keanu.golden;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.mcmc.nuts.NUTS;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Reference posterior summaries from fixed-seed Java sampler runs: for each latent of each golden model, the
 * element-wise posterior mean and standard deviation, and the chain's effective sample size per element (Geyer's
 * initial monotone sequence estimator). A Rust chain should agree within the combined Monte Carlo error
 * sd * sqrt(1 / ess_java + 1 / ess_rust), not exactly: the random streams differ.
 */
final class SamplerCases {

    private SamplerCases() {
    }

    static final int NUTS_ADAPT = 500;
    static final int NUTS_DROP = 500;
    static final int NUTS_SAMPLES = 3000;
    static final int MH_DROP = 2000;
    static final int MH_SAMPLES = 20000;

    static List<Map<String, Object>> generate() {
        List<Map<String, Object>> cases = new ArrayList<>();
        for (Map.Entry<String, Supplier<BayesianNetwork>> model : ModelCases.inferenceModels().entrySet()) {
            cases.add(run(model.getKey(), model.getValue(), "NUTS"));
            cases.add(run(model.getKey(), model.getValue(), "MetropolisHastings"));
        }
        return cases;
    }

    private static Map<String, Object> run(String modelName, Supplier<BayesianNetwork> builder, String samplerName) {
        BayesianNetwork network = ModelCases.fresh(builder);
        List<Vertex> latents = new ArrayList<>(network.getLatentVertices());
        latents.sort(Comparator.comparing(ModelCases::labelOf));

        Map<String, Object> settings = new LinkedHashMap<>();
        settings.put("name", samplerName);
        PosteriorSamplingAlgorithm algorithm;
        KeanuProbabilisticModel model;
        int total;
        int drop;
        if (samplerName.equals("NUTS")) {
            algorithm = NUTS.builder().random(new KeanuRandom(1)).adaptCount(NUTS_ADAPT).build();
            model = new KeanuProbabilisticModelWithGradient(network);
            total = NUTS_SAMPLES;
            drop = NUTS_DROP;
            settings.put("adaptCount", NUTS_ADAPT);
        } else {
            // defaults: prior proposals, one latent per step, rollback to cached values
            algorithm = MetropolisHastings.builder().random(new KeanuRandom(2)).build();
            model = new KeanuProbabilisticModel(network);
            total = MH_SAMPLES;
            drop = MH_DROP;
        }
        settings.put("samples", total);
        settings.put("drop", drop);

        Map<String, Object> result = new LinkedHashMap<>();
        try {
            NetworkSamples samples = algorithm.generatePosteriorSamples(model, latents).dropCount(drop).generate(total);
            Map<String, Object> posterior = new LinkedHashMap<>();
            for (Vertex latent : latents) {
                List<DoubleTensor> chain = samples.getDoubleTensorSamples(latent).asList();
                posterior.put(ModelCases.labelOf(latent), summarize(chain));
            }
            result.put("posterior", posterior);
        } catch (RuntimeException e) {
            result.put("error", e.getClass().getSimpleName());
        }

        Map<String, Object> c = new LinkedHashMap<>();
        c.put("name", modelName + "/" + samplerName);
        c.put("model", modelName);
        c.put("sampler", settings);
        c.put("result", result);
        return c;
    }

    private static Map<String, Object> summarize(List<DoubleTensor> chain) {
        long[] shape = chain.get(0).getShape();
        int length = (int) chain.get(0).getLength();
        double[] mean = new double[length];
        double[] sd = new double[length];
        double[] ess = new double[length];
        for (int e = 0; e < length; e++) {
            double[] series = new double[chain.size()];
            for (int i = 0; i < series.length; i++) {
                series[i] = chain.get(i).asFlatDoubleArray()[e];
            }
            double m = 0;
            for (double x : series) {
                m += x;
            }
            m /= series.length;
            double ss = 0;
            for (double x : series) {
                ss += (x - m) * (x - m);
            }
            mean[e] = m;
            sd[e] = Math.sqrt(ss / (series.length - 1));
            ess[e] = effectiveSampleSize(series, m, ss / series.length);
        }
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("mean", Cases.encode(DoubleTensor.create(mean, shape)));
        summary.put("sd", Cases.encode(DoubleTensor.create(sd, shape)));
        summary.put("ess", Cases.encode(DoubleTensor.create(ess, shape)));
        return summary;
    }

    /**
     * Geyer's initial monotone sequence estimator, with autocorrelations computed directly (1/n normalisation).
     */
    static double effectiveSampleSize(double[] x, double mean, double variance) {
        int n = x.length;
        if (variance == 0) {
            return n;
        }
        double sum = 0;
        double previous = Double.POSITIVE_INFINITY;
        for (int k = 0; k + 1 < n; k += 2) {
            double pair = autocorrelation(x, mean, variance, k) + autocorrelation(x, mean, variance, k + 1);
            if (pair <= 0) {
                break;
            }
            pair = Math.min(pair, previous);
            sum += pair;
            previous = pair;
        }
        double tau = -1 + 2 * sum;
        return tau <= 0 ? n : Math.min(n / tau, n);
    }

    private static double autocorrelation(double[] x, double mean, double variance, int lag) {
        double s = 0;
        for (int i = 0; i + lag < x.length; i++) {
            s += (x[i] - mean) * (x[i + lag] - mean);
        }
        return s / x.length / variance;
    }
}
