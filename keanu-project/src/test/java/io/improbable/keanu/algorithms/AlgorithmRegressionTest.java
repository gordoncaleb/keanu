package io.improbable.keanu.algorithms;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.mcmc.nuts.NUTS;
import io.improbable.keanu.algorithms.variational.optimizer.ConvergenceChecker;
import io.improbable.keanu.algorithms.variational.optimizer.RelativeConvergenceChecker;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiator;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialsWithRespectTo;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Regression tests for inference algorithm bugs found in code review.
 */
public class AlgorithmRegressionTest {

    @Test
    public void relativeConvergenceCheckerConvergesWhenPositionIsZero() {
        RelativeConvergenceChecker checker = new RelativeConvergenceChecker(ConvergenceChecker.Norm.L2, 1e-6, 1e-6);

        assertTrue(checker.hasConverged(
            new DoubleTensor[]{DoubleTensor.scalar(0.0)},
            new DoubleTensor[]{DoubleTensor.scalar(0.0)}
        ));

        assertTrue(checker.hasConverged(
            new DoubleTensor[]{DoubleTensor.scalar(0.5)},
            new DoubleTensor[]{DoubleTensor.scalar(0.5)}
        ));
    }

    @Test
    public void adamConvergesWhenTheOptimumIsZero() {
        GaussianVertex a = new GaussianVertex(0, 1);
        a.setValue(0.5);

        Keanu.Optimizer.Gradient.builderFor(a.getConnectedGraph())
            .algorithm(io.improbable.keanu.algorithms.variational.optimizer.gradient.Adam.builder()
                .maxEvaluations(5000)
                .build())
            .build()
            .maxAPosteriori();

        assertEquals(0.0, a.getValue().scalar(), 1e-3);
    }

    @Test
    public void steepNegativeGradientIsNotTreatedAsFlat() {
        GaussianVertex a = new GaussianVertex(0, 1);
        a.setValue(100.0);
        GaussianVertex b = new GaussianVertex(0, 1);
        b.setValue(0.0);

        Keanu.Optimizer.Gradient.of(new BayesianNetwork(Arrays.asList(a, b))).maxAPosteriori();

        assertEquals(0.0, a.getValue().scalar(), 1e-2);
    }

    @Test
    public void downSampleReportsTheNumberOfSamplesKept() {
        VariableReference reference = new GaussianVertex(0, 1).getReference();
        Map<VariableReference, List<DoubleTensor>> samplesByVariable = new HashMap<>();
        List<DoubleTensor> samples = new ArrayList<>();
        List<Double> logProbs = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            samples.add(DoubleTensor.scalar(i));
            logProbs.add((double) -i);
        }
        samplesByVariable.put(reference, samples);

        NetworkSamples downSampled = new NetworkSamples(samplesByVariable, logProbs, 5).downSample(2);

        assertEquals(3, downSampled.size());
        assertEquals(3, downSampled.toNetworkStates().size());
    }

    @Test
    public void nutsCanBeReusedForASecondSamplingRun() {
        GaussianVertex a = new GaussianVertex(0, 1);
        a.setValue(0.1);
        KeanuProbabilisticModelWithGradient model = new KeanuProbabilisticModelWithGradient(new BayesianNetwork(a.getConnectedGraph()));

        NUTS nuts = NUTS.builder()
            .random(new KeanuRandom(1))
            .adaptCount(100)
            .build();

        nuts.getPosteriorSamples(model, model.getLatentVariables(), 200);
        NetworkSamples second = nuts.getPosteriorSamples(model, model.getLatentVariables(), 200);

        List<DoubleTensor> values = second.getDoubleTensorSamples(a).asList();
        long distinct = values.stream().map(DoubleTensor::scalar).distinct().count();
        assertTrue("the second chain should move, but only had " + distinct + " distinct values", distinct > 50);
        values.forEach(v -> assertTrue(Double.isFinite(v.scalar())));
    }

    @Test
    public void forwardModeAutoDiffReturnsPartialsForEveryRequestedVertex() {
        GaussianVertex a = new GaussianVertex(0, 1);
        a.setValue(1.0);
        DoubleVertex b = a.times(2.0);
        DoubleVertex c = b.times(3.0);

        PartialsWithRespectTo partials = Differentiator.forwardModeAutoDiff(a, b, c);

        assertNotNull(partials.of(b));
        assertNotNull(partials.of(c));
        assertEquals(2.0, partials.of(b).scalar(), 1e-12);
        assertEquals(6.0, partials.of(c).scalar(), 1e-12);
    }

    @Test
    public void adamMaxLikelihoodWithALatentTheLikelihoodDoesNotDependOn() {
        GaussianVertex unrelated = new GaussianVertex(0, 1);
        unrelated.setValue(0.5);
        GaussianVertex mu = new GaussianVertex(0, 10);
        mu.setValue(1.0);
        GaussianVertex observed = new GaussianVertex(mu, 1);
        observed.observe(3.0);
        // connect the unrelated latent to the same graph without affecting the likelihood
        DoubleVertex joined = unrelated.plus(mu);

        io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer.builder()
            .probabilisticModel(new KeanuProbabilisticModelWithGradient(new BayesianNetwork(joined.getConnectedGraph())))
            .algorithm(io.improbable.keanu.algorithms.variational.optimizer.gradient.Adam.builder().maxEvaluations(100000).build())
            .build()
            .maxLikelihood();

        assertEquals(3.0, mu.getValue().scalar(), 1e-2);
        assertEquals(0.5, unrelated.getValue().scalar(), 0.0);
    }
}
