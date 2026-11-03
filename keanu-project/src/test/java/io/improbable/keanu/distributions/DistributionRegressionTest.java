package io.improbable.keanu.distributions;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.continuous.Gamma;
import io.improbable.keanu.distributions.continuous.Triangular;
import io.improbable.keanu.distributions.discrete.Binomial;
import io.improbable.keanu.distributions.discrete.Multinomial;
import io.improbable.keanu.distributions.discrete.Poisson;
import io.improbable.keanu.distributions.discrete.UniformInt;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.junit.Test;

import java.util.function.Function;

import static io.improbable.keanu.distributions.hyperparam.Diffs.A;
import static io.improbable.keanu.distributions.hyperparam.Diffs.B;
import static io.improbable.keanu.distributions.hyperparam.Diffs.C;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Regression tests for distribution bugs found in code review.
 */
public class DistributionRegressionTest {

    @Test
    public void gammaSamplesHaveCorrectMomentsWhenShapeIsLessThanOne() {
        KeanuRandom random = new KeanuRandom(1);
        int sampleCount = 200000;

        for (double k : new double[]{0.2, 0.5, 0.9}) {
            double theta = 2.0;
            DoubleTensor samples = Gamma.withParameters(DoubleTensor.scalar(theta), DoubleTensor.scalar(k))
                .sample(new long[]{sampleCount}, random);

            double mean = samples.mean().scalar();
            double variance = samples.minus(mean).pow(2).mean().scalar();

            assertEquals("mean for k=" + k, k * theta, mean, 0.02 * k * theta + 0.01);
            assertEquals("variance for k=" + k, k * theta * theta, variance, 0.05 * k * theta * theta + 0.01);
        }
    }

    @Test
    public void uniformIntSamplesNegativeRangesUniformly() {
        KeanuRandom random = new KeanuRandom(1);
        int sampleCount = 60000;
        int[] samples = UniformInt.withParameters(IntegerTensor.scalar(-3), IntegerTensor.scalar(2))
            .sample(new long[]{sampleCount}, random)
            .asFlatIntegerArray();

        int[] counts = new int[5];
        for (int s : samples) {
            assertTrue("sample " + s + " is outside [-3, 2)", s >= -3 && s < 2);
            counts[s + 3]++;
        }

        for (int count : counts) {
            assertEquals(sampleCount / 5.0, count, sampleCount * 0.01);
        }
    }

    @Test
    public void multinomialGradientMatchesFiniteDifference() {
        IntegerTensor n = IntegerTensor.scalar(5);
        IntegerTensor x = IntegerTensor.create(1, 3, 1);

        for (DoubleTensor p : new DoubleTensor[]{DoubleTensor.create(0.2, 0.5, 0.3), DoubleTensor.create(2.0, 1.0, 3.0)}) {
            DoubleTensor gradient = Multinomial.withParameters(n, p).dLogProb(x);

            double[] pValues = p.asFlatDoubleArray();
            for (int j = 0; j < pValues.length; j++) {
                double h = 1e-6;
                double[] up = pValues.clone();
                double[] down = pValues.clone();
                up[j] += h;
                down[j] -= h;
                double numeric = (logProb(n, up, x) - logProb(n, down, x)) / (2 * h);
                assertEquals(numeric, gradient.getValue(j), 1e-5);
            }
        }
    }

    private static double logProb(IntegerTensor n, double[] p, IntegerTensor x) {
        return Multinomial.withParameters(n, DoubleTensor.create(p)).logProb(x).sum().scalar();
    }

    @Test
    public void poissonSampleBroadcastsMuAlongLeadingDimension() {
        KeanuRandom random = new KeanuRandom(1);
        DoubleTensor mu = DoubleTensor.create(new double[]{1, 1000}, 2, 1);

        IntegerTensor samples = Poisson.withParameters(mu).sample(new long[]{2, 3}, random);

        int[] flat = samples.asFlatIntegerArray();
        for (int i = 0; i < 3; i++) {
            assertTrue("row 0 should come from mu=1 but was " + flat[i], flat[i] < 20);
            assertTrue("row 1 should come from mu=1000 but was " + flat[3 + i], flat[3 + i] > 500);
        }
    }

    @Test
    public void binomialAndPoissonLogProbIsNegativeInfinityOutsideSupport() {
        DoubleTensor binomial = Binomial.withParameters(DoubleTensor.scalar(0.5), IntegerTensor.scalar(3))
            .logProb(IntegerTensor.create(-1, 4, 2));
        assertEquals(Double.NEGATIVE_INFINITY, binomial.getValue(0), 0.0);
        assertEquals(Double.NEGATIVE_INFINITY, binomial.getValue(1), 0.0);
        assertEquals(Math.log(3.0 / 8.0), binomial.getValue(2), 1e-12);

        DoubleTensor poisson = Poisson.withParameters(DoubleTensor.scalar(2.0)).logProb(IntegerTensor.create(-1, 0));
        assertEquals(Double.NEGATIVE_INFINITY, poisson.getValue(0), 0.0);
        assertEquals(-2.0, poisson.getValue(1), 1e-12);
    }

    @Test
    public void triangularLogProbAtModeAndEdges() {
        Triangular t = Triangular.withParameters(DoubleTensor.scalar(0.0), DoubleTensor.scalar(1.0), DoubleTensor.scalar(0.5));
        assertEquals(Math.log(2.0), t.logProb(DoubleTensor.scalar(0.5)).scalar(), 1e-12);
        assertEquals(Math.log(1.0), t.logProb(DoubleTensor.scalar(0.25)).scalar(), 1e-12);
        assertEquals(Double.NEGATIVE_INFINITY, t.logProb(DoubleTensor.scalar(1.5)).scalar(), 0.0);

        // mode at the lower bound: pdf = 2(1 - x)
        Triangular left = Triangular.withParameters(DoubleTensor.scalar(0.0), DoubleTensor.scalar(1.0), DoubleTensor.scalar(0.0));
        assertArrayEquals(
            new double[]{Math.log(2.0), Math.log(1.0), Double.NEGATIVE_INFINITY},
            left.logProb(DoubleTensor.create(0.0, 0.5, 1.0)).asFlatDoubleArray(),
            1e-12
        );

        // mode at the upper bound: pdf = 2x
        Triangular right = Triangular.withParameters(DoubleTensor.scalar(0.0), DoubleTensor.scalar(1.0), DoubleTensor.scalar(1.0));
        assertArrayEquals(
            new double[]{Double.NEGATIVE_INFINITY, Math.log(1.0), Math.log(2.0)},
            right.logProb(DoubleTensor.create(0.0, 0.5, 1.0)).asFlatDoubleArray(),
            1e-12
        );
    }

    @Test
    public void triangularGradientMatchesFiniteDifference() {
        double a = 1.0;
        double b = 4.0;
        double c = 2.5;

        for (double x : new double[]{1.7, 3.2}) {
            Diffs diffs = Triangular.withParameters(DoubleTensor.scalar(a), DoubleTensor.scalar(b), DoubleTensor.scalar(c))
                .dLogProb(DoubleTensor.scalar(x));

            assertEquals(numeric(v -> triangularLogProb(a, b, c, v), x), diffs.get(X).getValue().scalar(), 1e-5);
            assertEquals(numeric(v -> triangularLogProb(v, b, c, x), a), diffs.get(A).getValue().scalar(), 1e-5);
            assertEquals(numeric(v -> triangularLogProb(a, v, c, x), b), diffs.get(B).getValue().scalar(), 1e-5);
            assertEquals(numeric(v -> triangularLogProb(a, b, v, x), c), diffs.get(C).getValue().scalar(), 1e-5);
        }
    }

    private static double triangularLogProb(double a, double b, double c, double x) {
        return Triangular.withParameters(DoubleTensor.scalar(a), DoubleTensor.scalar(b), DoubleTensor.scalar(c))
            .logProb(DoubleTensor.scalar(x)).scalar();
    }

    private static double numeric(Function<Double, Double> f, double at) {
        double h = 1e-6;
        return (f.apply(at + h) - f.apply(at - h)) / (2 * h);
    }

    @Test
    public void continuousLogProbIsNegativeInfinityOutsideSupportRatherThanNaN() {
        DoubleTensor x = DoubleTensor.create(-1.0, -0.5);
        DoubleTensor[] logProbs = new DoubleTensor[]{
            Gamma.withParameters(DoubleTensor.scalar(2.0), DoubleTensor.scalar(1.5)).logProb(x),
            io.improbable.keanu.distributions.continuous.ChiSquared.withParameters(IntegerTensor.scalar(3)).logProb(x),
            io.improbable.keanu.distributions.continuous.InverseGamma.withParameters(DoubleTensor.scalar(2.0), DoubleTensor.scalar(1.0)).logProb(x),
            io.improbable.keanu.distributions.continuous.LogNormal.withParameters(DoubleTensor.scalar(0.0), DoubleTensor.scalar(1.0)).logProb(x),
            io.improbable.keanu.distributions.continuous.Beta.withParameters(DoubleTensor.scalar(2.0), DoubleTensor.scalar(3.0), DoubleTensor.scalar(0.0), DoubleTensor.scalar(1.0)).logProb(x)
        };
        for (DoubleTensor logProb : logProbs) {
            assertArrayEquals(new double[]{Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY}, logProb.asFlatDoubleArray(), 0.0);
        }

        assertEquals(Double.NEGATIVE_INFINITY, io.improbable.keanu.distributions.continuous.LogNormal.withParameters(DoubleTensor.scalar(0.0), DoubleTensor.scalar(1.0)).logProb(DoubleTensor.scalar(0.0)).scalar(), 0.0);
        assertEquals(Double.NEGATIVE_INFINITY, io.improbable.keanu.distributions.continuous.InverseGamma.withParameters(DoubleTensor.scalar(2.0), DoubleTensor.scalar(1.0)).logProb(DoubleTensor.scalar(0.0)).scalar(), 0.0);
    }

    @Test
    public void gammaWithShapeOneIsExponentialIncludingAtZero() {
        double theta = 2.0;
        DoubleTensor logProb = Gamma.withParameters(DoubleTensor.scalar(theta), DoubleTensor.scalar(1.0)).logProb(DoubleTensor.create(0.0, 1.5));
        assertArrayEquals(new double[]{-Math.log(theta), -1.5 / theta - Math.log(theta)}, logProb.asFlatDoubleArray(), 1e-12);
    }

    @Test
    public void betaLogProbHonoursItsBounds() {
        // Beta(2, 3) on [-1, 2] at x is Beta(2, 3) on [0, 1] at (x + 1) / 3, divided by the range 3
        io.improbable.keanu.distributions.continuous.Beta bounded = io.improbable.keanu.distributions.continuous.Beta.withParameters(
            DoubleTensor.scalar(2.0), DoubleTensor.scalar(3.0), DoubleTensor.scalar(-1.0), DoubleTensor.scalar(2.0));
        io.improbable.keanu.distributions.continuous.Beta unit = io.improbable.keanu.distributions.continuous.Beta.withParameters(
            DoubleTensor.scalar(2.0), DoubleTensor.scalar(3.0), DoubleTensor.scalar(0.0), DoubleTensor.scalar(1.0));

        for (double x : new double[]{-0.5, 0.0, 1.5}) {
            double expected = unit.logProb(DoubleTensor.scalar((x + 1) / 3)).scalar() - Math.log(3);
            assertEquals(expected, bounded.logProb(DoubleTensor.scalar(x)).scalar(), 1e-12);

            double h = 1e-6;
            double numeric = (bounded.logProb(DoubleTensor.scalar(x + h)).scalar() - bounded.logProb(DoubleTensor.scalar(x - h)).scalar()) / (2 * h);
            assertEquals(numeric, bounded.dLogProb(DoubleTensor.scalar(x)).get(X).getValue().scalar(), 1e-5);
        }
        assertEquals(Double.NEGATIVE_INFINITY, bounded.logProb(DoubleTensor.scalar(2.5)).scalar(), 0.0);

        // alpha == 1 is finite at the lower bound
        io.improbable.keanu.distributions.continuous.Beta alphaOne = io.improbable.keanu.distributions.continuous.Beta.withParameters(
            DoubleTensor.scalar(1.0), DoubleTensor.scalar(2.0), DoubleTensor.scalar(0.0), DoubleTensor.scalar(1.0));
        assertEquals(Math.log(2.0), alphaOne.logProb(DoubleTensor.scalar(0.0)).scalar(), 1e-12);
    }
}
