package io.improbable.keanu.algorithms.particlefiltering;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;

public class ParticleWeightsTest {

    @Test
    public void weightsDoNotUnderflowForVeryNegativeLogProbs() {
        List<Particle> particles = Arrays.asList(withLogProb(-2000), withLogProb(-2001), withLogProb(-3000));

        double[] weights = ParticleFilter.normalisedWeights(particles);

        assertArrayEquals(new double[]{1.0, Math.exp(-1), 0.0}, weights, 1e-12);
    }

    @Test
    public void weightsAreUniformWhenEveryParticleIsImpossible() {
        List<Particle> particles = Arrays.asList(withLogProb(Double.NEGATIVE_INFINITY), withLogProb(Double.NEGATIVE_INFINITY));
        assertArrayEquals(new double[]{1.0, 1.0}, ParticleFilter.normalisedWeights(particles), 0.0);
    }

    private static Particle withLogProb(double logProb) {
        return new Particle() {
            @Override
            public double logProb() {
                return logProb;
            }
        };
    }
}
