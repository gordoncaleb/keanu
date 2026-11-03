package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

import static io.improbable.keanu.distributions.hyperparam.Diffs.LAMBDA;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;

public class Exponential implements ContinuousDistribution {

    private final DoubleTensor lambda;

    public static Exponential withParameters(DoubleTensor lambda) {
        return new Exponential(lambda);
    }

    private Exponential(DoubleTensor lambda) {
        this.lambda = lambda;
    }

    /**
     * Samples using the inverse CDF: x = -ln(U) / rate
     */
    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        return random.nextDouble(shape).logInPlace().divInPlace(lambda).unaryMinusInPlace();
    }

    /**
     * logPdf = ln(rate) - rate * x for x &gt;= 0, -infinity otherwise
     */
    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor logProb = lambda.log().minusInPlace(x.times(lambda));
        return logProb.setWithMask(x.lessThanMask(DoubleTensor.scalar(0.0)), Double.NEGATIVE_INFINITY);
    }

    public static DoubleVertex logProbOutput(DoublePlaceholderVertex x, DoublePlaceholderVertex lambda) {
        final DoubleVertex logProb = lambda.log().minus(x.times(lambda));
        return logProb.setWithMask(x.lessThanMask(0.), Double.NEGATIVE_INFINITY);
    }

    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor dLogPdx = DoubleTensor.zeros(x.getShape()).minusInPlace(lambda);
        final DoubleTensor dLogPdlambda = lambda.reciprocal().minusInPlace(x);
        return new Diffs()
            .put(LAMBDA, dLogPdlambda)
            .put(X, dLogPdx);
    }
}
