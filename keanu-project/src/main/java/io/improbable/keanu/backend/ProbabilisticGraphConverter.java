package io.improbable.keanu.backend;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphSupplier;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import lombok.experimental.UtilityClass;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * This class is a utility class for creating joint logProb graphs. It takes a BayesianNetwork and returns variable
 * references to outputs of the log prior probability and log likelihood.
 */
@UtilityClass
public class ProbabilisticGraphConverter {

    public static Optional<VariableReference> convertLogProbObservation(BayesianNetwork network,
                                                                        ComputableGraphBuilder<?> graphBuilder) {
        return addLogProbCalculation(
            graphBuilder,
            network.getObservedVertices()
        );
    }

    public static VariableReference convertLogProbPrior(BayesianNetwork network,
                                                        ComputableGraphBuilder<?> graphBuilder) {
        return addLogProbCalculation(
            graphBuilder,
            network.getLatentVertices()
        ).orElseThrow(() -> new IllegalArgumentException("Network must contain latent variables"));
    }

    private static Optional<VariableReference> addLogProbCalculation(ComputableGraphBuilder<?> graphBuilder,
                                                                     List<Vertex> probabilisticVertices) {
        List<VariableReference> logProbOps = probabilisticVertices.stream()
            .map(visiting -> {
                if (visiting instanceof LogProbGraphSupplier) {
                    LogProbGraph logProbGraph = ((LogProbGraphSupplier) visiting).logProbGraph();
                    return addLogProbGraph(logProbGraph, graphBuilder);
                } else {
                    throw new IllegalArgumentException("Vertex type " + visiting.getClass() + " logProb as a graph not supported");
                }
            }).collect(Collectors.toList());

        Optional<VariableReference> logProbSummation = logProbOps.stream()
            .reduce(graphBuilder::add);

        return logProbSummation;
    }

    /**
     * @param logProbGraph the graph to add that represents the logProb calculation
     * @param graphBuilder the builder that contains state for the probabilistic graph building so far.
     */
    private static VariableReference addLogProbGraph(LogProbGraph logProbGraph,
                                                     ComputableGraphBuilder<?> graphBuilder) {

        graphBuilder.connect(logProbGraph.getInputs());

        // Most distributions return an element-wise log prob so it needs summing to give the vertex's total log prob.
        // This must happen before the per-vertex outputs are added together, otherwise they would broadcast.
        final DoubleVertex logProbOutput = logProbGraph.getLogProbOutput();
        final DoubleVertex totalLogProb = logProbOutput.getShape().length == 0 ? logProbOutput : logProbOutput.sum();

        graphBuilder.convert(totalLogProb.getConnectedGraph());
        return totalLogProb.getReference();
    }

}
