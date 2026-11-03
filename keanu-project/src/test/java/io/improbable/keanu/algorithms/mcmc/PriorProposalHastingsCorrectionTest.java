package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.mcmc.proposal.PriorProposalDistribution;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.assertEquals;

public class PriorProposalHastingsCorrectionTest {

    /**
     * a ~ N(0, 1), b ~ N(a, 1) has Var(b) = 2 and Cov(a, b) = 1. Proposing both at once from the prior samples b'
     * given the old a, so q(x'|x) must be evaluated with the pre-proposal parent values for the chain to target
     * the right distribution.
     */
    @Test
    public void jointPriorProposalOfDependentVariablesTargetsTheCorrectDistribution() {
        double[] moments = sampleMoments(new PriorProposalDistribution(), 40000);

        assertEquals("Var(a)", 1.0, moments[0], 0.1);
        assertEquals("Var(b)", 2.0, moments[1], 0.15);
        assertEquals("Cov(a, b)", 1.0, moments[2], 0.1);
    }

    static double[] sampleMoments(ProposalDistribution proposalDistribution, int sampleCount) {
        GaussianVertex a = new GaussianVertex(0, 1);
        GaussianVertex b = new GaussianVertex(a, 1);
        a.setValue(0.3);
        b.setValue(-0.2);

        KeanuProbabilisticModel model = new KeanuProbabilisticModel(new BayesianNetwork(a.getConnectedGraph()));

        MetropolisHastingsStep step = new MetropolisHastingsStep(
            model,
            proposalDistribution,
            new RollBackToCachedValuesOnRejection(),
            new KeanuRandom(1)
        );

        Set<Variable> both = new HashSet<>(model.getLatentVariables());
        double logProb = model.logProb();

        double sumA = 0, sumB = 0, sumAA = 0, sumBB = 0, sumAB = 0;
        for (int i = 0; i < sampleCount; i++) {
            logProb = step.step(both, logProb).getLogProbabilityAfterStep();
            double av = a.getValue().scalar();
            double bv = b.getValue().scalar();
            sumA += av;
            sumB += bv;
            sumAA += av * av;
            sumBB += bv * bv;
            sumAB += av * bv;
        }

        double n = sampleCount;
        return new double[]{
            sumAA / n - Math.pow(sumA / n, 2),
            sumBB / n - Math.pow(sumB / n, 2),
            sumAB / n - (sumA / n) * (sumB / n)
        };
    }
}
