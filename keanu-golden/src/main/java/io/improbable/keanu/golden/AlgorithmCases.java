package io.improbable.keanu.golden;

import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.Adam;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.ConjugateGradient;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizationAlgorithm;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
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
        }
        return cases;
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
