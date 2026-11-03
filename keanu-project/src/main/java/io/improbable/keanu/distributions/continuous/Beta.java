package io.improbable.keanu.distributions.continuous;

import com.google.common.base.Preconditions;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

import static io.improbable.keanu.distributions.hyperparam.Diffs.A;
import static io.improbable.keanu.distributions.hyperparam.Diffs.B;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;

public class Beta implements ContinuousDistribution {

    private final DoubleTensor alpha;
    private final DoubleTensor beta;
    private final DoubleTensor xMin;
    private final DoubleTensor xMax;

    public static Beta withParameters(DoubleTensor alpha, DoubleTensor beta, DoubleTensor xMin, DoubleTensor xMax) {
        return new Beta(alpha, beta, xMin, xMax);
    }

    private Beta(DoubleTensor alpha, DoubleTensor beta, DoubleTensor xMin, DoubleTensor xMax) {
        this.alpha = alpha;
        this.beta = beta;
        this.xMin = xMin;
        this.xMax = xMax;
    }

    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        Preconditions.checkArgument(alpha.greaterThan(0.).allTrue().scalar() && beta.greaterThan(0.).allTrue().scalar(),
            "alpha and beta must be positive. alpha: " + alpha + " beta: " + beta);

        final DoubleTensor one = DoubleTensor.scalar(1.0);
        final DoubleTensor y1 = random.nextGamma(shape, one, alpha);
        final DoubleTensor y2 = random.nextGamma(shape, one, beta);

        final DoubleTensor range = xMax.minus(xMin);
        final DoubleTensor y1PlusY2 = y1.plus(y2);

        final DoubleTensor lessThan = xMax.minus(y2.div(y1PlusY2).timesInPlace(range));
        final DoubleTensor greaterThan = xMin.plus(y1.div(y1PlusY2).timesInPlace(range));

        final DoubleTensor lessMask = alpha.lessThanMask(beta);
        final DoubleTensor greaterMask = alpha.greaterThanOrEqualToMask(beta);

        return lessMask.timesInPlace(lessThan).plusInPlace(greaterMask.timesInPlace(greaterThan));
    }

    /**
     * Beta on [xMin, xMax]: with y = (x - xMin) / (xMax - xMin),
     * logPdf(x) = (alpha - 1) ln(y) + (beta - 1) ln(1 - y) - lnB(alpha, beta) - ln(xMax - xMin)
     * and -infinity outside [xMin, xMax].
     */
    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor range = xMax.minus(xMin);
        final DoubleTensor y = x.minus(xMin).divInPlace(range);
        final DoubleTensor oneMinusY = y.unaryMinus().plusInPlace(1.);

        final DoubleTensor lnGammaAlpha = alpha.logGamma();
        final DoubleTensor lnGammaBeta = beta.logGamma();
        final DoubleTensor alphaPlusBetaLnGamma = (alpha.plus(beta)).logGammaInPlace();
        final DoubleTensor betaFunction = lnGammaAlpha.plusInPlace(lnGammaBeta).minusInPlace(alphaPlusBetaLnGamma);

        // 0 * ln(0) = 0 so alpha == 1 or beta == 1 are finite at the bounds
        final DoubleTensor alphaMinusOneTimesLnY = y.safeLogTimes(alpha.minus(1.));
        final DoubleTensor betaMinusOneTimesLnOneMinusY = oneMinusY.safeLogTimes(beta.minus(1.));

        final DoubleTensor logProb = alphaMinusOneTimesLnY
            .plusInPlace(betaMinusOneTimesLnOneMinusY)
            .minusInPlace(betaFunction)
            .minusInPlace(range.log());

        final DoubleTensor outside = x.lessThanMask(xMin).plusInPlace(x.greaterThanMask(xMax));
        return logProb.setWithMaskInPlace(outside.greaterThanMask(DoubleTensor.scalar(0.0)), Double.NEGATIVE_INFINITY);
    }

    public static DoubleVertex logProbOutput(DoublePlaceholderVertex x, DoublePlaceholderVertex alpha, DoublePlaceholderVertex beta) {
        final DoubleVertex lnGammaAlpha = alpha.logGamma();
        final DoubleVertex lnGammaBeta = beta.logGamma();
        final DoubleVertex alphaPlusBetaLnGamma = (alpha.plus(beta)).logGamma();
        // the graph form is only used by BetaVertex, which is always on [0, 1]
        final DoubleVertex alphaMinusOneTimesLnX = x.safeLogTimes(alpha.minus(1.));
        final DoubleVertex betaMinusOneTimesOneMinusXLn = x.unaryMinus().plus(1.).safeLogTimes(beta.minus(1.));

        final DoubleVertex betaFunction = lnGammaAlpha.plus(lnGammaBeta).minus(alphaPlusBetaLnGamma);

        return alphaMinusOneTimesLnX.plus(betaMinusOneTimesOneMinusXLn).minus(betaFunction)
            .setWithMask(x.lessThanMask(0.), Double.NEGATIVE_INFINITY)
            .setWithMask(x.greaterThanMask(1.), Double.NEGATIVE_INFINITY);
    }

    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor range = xMax.minus(xMin);
        final DoubleTensor y = x.minus(xMin).divInPlace(range);
        final DoubleTensor oneMinusY = y.unaryMinus().plusInPlace(1.0);
        final DoubleTensor digammaAlphaPlusBeta = alpha.plus(beta).digammaInPlace();
        final DoubleTensor alphaMinusOneDivY = y.reciprocal().timesInPlace(alpha.minus(1));

        final DoubleTensor dLogPdy = alphaMinusOneDivY.minusInPlace(oneMinusY.reciprocal().timesInPlace(beta.minus(1)));
        final DoubleTensor dLogPdx = dLogPdy.divInPlace(range);
        final DoubleTensor dLogPda = y.log().plusInPlace(digammaAlphaPlusBeta.minus(alpha.digamma()));
        final DoubleTensor dLogPdb = oneMinusY.logInPlace().plusInPlace(digammaAlphaPlusBeta.minusInPlace(beta.digamma()));

        return new Diffs()
            .put(A, dLogPda)
            .put(B, dLogPdb)
            .put(X, dLogPdx);
    }

}
