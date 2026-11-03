package io.improbable.keanu.golden;

import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.Adam;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.ConjugateGradient;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizationAlgorithm;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.BOBYQA;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.vertices.Vertex;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Reference results for the gradient optimizers: MAP and maximum likelihood on each golden model with Adam and
 * conjugate gradient. The optimum (keyed by latent label) and its fitness are recorded; optimizers converge to the
 * same optimum by different paths, so they should be compared with an optimization tolerance, not exactly.
 */
final class AlgorithmCases {

    private AlgorithmCases() {
    }

    static final int ADAM_MAX_EVALUATIONS = 200_000;

    static List<Map<String, Object>> generate() {
        List<Map<String, Object>> cases = new ArrayList<>();
        for (Map.Entry<String, Supplier<BayesianNetwork>> model : ModelCases.inferenceModels().entrySet()) {
            for (String algorithm : new String[]{"Adam", "ConjugateGradient"}) {
                for (String fitness : new String[]{"MAP", "MLE"}) {
                    cases.add(optimize(model.getKey(), model.getValue(), algorithm, fitness));
                }
            }
            for (String fitness : new String[]{"MAP", "MLE"}) {
                cases.add(bobyqa(model.getKey(), model.getValue(), "BOBYQA", fitness, null, null));
                cases.add(bobyqa(model.getKey(), model.getValue(), "BOBYQA/radius0.5/range5", fitness, 0.5, 5.0));
            }
        }
        return cases;
    }

    /**
     * Java's BOBYQA (Apache Commons Math). Keanu's defaults are an initial trust region radius of 10, a stopping
     * radius of 1e-8 and no bounds; BOBYQA needs at least two latent dimensions.
     */
    private static Map<String, Object> bobyqa(String modelName, Supplier<BayesianNetwork> builder, String name, String fitness,
                                              Double initialRadius, Double boundsRange) {
        BayesianNetwork network = ModelCases.fresh(builder);

        Map<String, Object> settings = new LinkedHashMap<>();
        settings.put("name", "BOBYQA");
        BOBYQA.BOBYQABuilder algorithm = BOBYQA.builder();
        if (initialRadius != null) {
            algorithm.initialTrustRegionRadius(initialRadius);
            settings.put("initialTrustRegionRadius", initialRadius);
        }
        if (boundsRange != null) {
            algorithm.boundsRange(boundsRange);
            settings.put("boundsRange", boundsRange);
        }

        Map<String, Object> result = new LinkedHashMap<>();
        try {
            NonGradientOptimizer optimizer = NonGradientOptimizer.builder()
                .probabilisticModel(new KeanuProbabilisticModel(network))
                .algorithm(algorithm.build())
                .build();
            OptimizedResult optimized = fitness.equals("MAP") ? optimizer.maxAPosteriori() : optimizer.maxLikelihood();

            List<Vertex> latents = new ArrayList<>(network.getLatentVertices());
            latents.sort(Comparator.comparing(ModelCases::labelOf));
            Map<String, Object> values = new LinkedHashMap<>();
            for (Vertex latent : latents) {
                values.put(ModelCases.labelOf(latent), Cases.encode(optimized.getValueFor(latent.getId())));
            }
            result.put("values", values);
            result.put("fitness", optimized.getFitness());
        } catch (RuntimeException e) {
            result.put("error", e.getClass().getSimpleName());
        }

        Map<String, Object> c = new LinkedHashMap<>();
        c.put("name", modelName + "/" + name + "/" + fitness);
        c.put("model", modelName);
        c.put("algorithm", settings);
        c.put("fitness", fitness);
        c.put("result", result);
        return c;
    }

    private static Map<String, Object> optimize(String modelName, Supplier<BayesianNetwork> builder, String algorithmName, String fitness) {
        BayesianNetwork network = ModelCases.fresh(builder);

        Map<String, Object> settings = new LinkedHashMap<>();
        settings.put("name", algorithmName);
        GradientOptimizationAlgorithm algorithm;
        Adam adam = null;
        if (algorithmName.equals("Adam")) {
            // defaults: alpha 0.001, beta1 0.9, beta2 0.999, epsilon 1e-8, relative L2 convergence 1e-6
            adam = Adam.builder().maxEvaluations(ADAM_MAX_EVALUATIONS).build();
            settings.put("maxEvaluations", ADAM_MAX_EVALUATIONS);
            algorithm = adam;
        } else {
            // defaults: Polak-Ribiere, relative and absolute thresholds 1e-8
            algorithm = ConjugateGradient.builder().build();
        }

        Map<String, Object> result = new LinkedHashMap<>();
        try {
            GradientOptimizer optimizer = GradientOptimizer.builder()
                .probabilisticModel(new KeanuProbabilisticModelWithGradient(network))
                .algorithm(algorithm)
                .build();
            OptimizedResult optimized = fitness.equals("MAP") ? optimizer.maxAPosteriori() : optimizer.maxLikelihood();

            List<Vertex> latents = new ArrayList<>(network.getLatentVertices());
            latents.sort(Comparator.comparing(ModelCases::labelOf));
            Map<String, Object> values = new LinkedHashMap<>();
            for (Vertex latent : latents) {
                values.put(ModelCases.labelOf(latent), Cases.encode(optimized.getValueFor(latent.getId())));
            }
            result.put("values", values);
            result.put("fitness", optimized.getFitness());
            if (adam != null) {
                result.put("converged", adam.getStatistics().didConverge());
            }
        } catch (RuntimeException e) {
            result.put("error", e.getClass().getSimpleName());
        }

        Map<String, Object> c = new LinkedHashMap<>();
        c.put("name", modelName + "/" + algorithmName + "/" + fitness);
        c.put("model", modelName);
        c.put("algorithm", settings);
        c.put("fitness", fitness);
        c.put("result", result);
        return c;
    }
}
