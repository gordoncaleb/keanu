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
import static io.improbable.keanu.distributions.hyperparam.Diffs.C;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;

public class Triangular implements ContinuousDistribution {

    private final DoubleTensor xMin;
    private final DoubleTensor xMax;
    private final DoubleTensor c;

    public static Triangular withParameters(DoubleTensor xMin, DoubleTensor xMax, DoubleTensor c) {
        return new Triangular(xMin, xMax, c);
    }

    private Triangular(DoubleTensor xMin, DoubleTensor xMax, DoubleTensor c) {
        this.xMin = xMin;
        this.xMax = xMax;
        this.c = c;
    }

    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        Preconditions.checkArgument(c.greaterThanOrEqual(xMin).allTrue().scalar() && c.lessThanOrEqual(xMax).allTrue().scalar(),
            "center must be between xMin and xMax. c: " + c + " xMin: " + xMin + " xMax: " + xMax);

        final DoubleTensor p = random.nextDouble(shape);
        final DoubleTensor q = p.unaryMinus().plusInPlace(1.0);
        final DoubleTensor range = xMax.minus(xMin);

        final DoubleTensor conditional = (c.minus(xMin)).divInPlace(xMax.minus(xMin));

        final DoubleTensor lessThan = xMin.plus((range.times(c.minus(xMin).timesInPlace(p))).sqrtInPlace());
        final DoubleTensor greaterThan = xMax.minus((range.timesInPlace(xMax.minus(c).timesInPlace(q))).sqrtInPlace());

        final DoubleTensor lessThanMask = p.lessThanOrEqualToMask(conditional);
        final DoubleTensor greaterThanMask = p.greaterThanMask(conditional);

        return (lessThan.timesInPlace(lessThanMask).plusInPlace(greaterThan.timesInPlace(greaterThanMask)));
    }

    /**
     * pdf(x) = 2(x - a) / ((b - a)(c - a))  for a &lt;= x &lt; c
     *        = 2 / (b - a)                  for x == c
     *        = 2(b - x) / ((b - a)(b - c))  for c &lt; x &lt;= b
     *        = 0                            otherwise
     * <p>
     * where a = xMin, b = xMax and c is the mode. c may equal a or b.
     */
    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final Regions r = new Regions(x);

        final DoubleTensor range = xMax.minus(xMin);
        final DoubleTensor left = x.minus(xMin).timesInPlace(2.).divInPlace(range.times(r.safeCMinusA));
        final DoubleTensor right = xMax.minus(x).timesInPlace(2.).divInPlace(range.times(r.safeBMinusC));
        final DoubleTensor mode = range.reciprocal().timesInPlace(2.);

        return left.timesInPlace(r.left)
            .plusInPlace(right.timesInPlace(r.right))
            .plusInPlace(mode.timesInPlace(r.mode))
            .logInPlace();
    }

    public static DoubleVertex logProbOutput(DoublePlaceholderVertex x, DoublePlaceholderVertex xMin, DoublePlaceholderVertex xMax, DoublePlaceholderVertex c) {
        final DoubleVertex range = xMax.minus(xMin);
        final DoubleVertex cMinusA = c.minus(xMin);
        final DoubleVertex bMinusC = xMax.minus(c);
        final DoubleVertex safeCMinusA = cMinusA.setWithMask(cMinusA.lessThanOrEqualToMask(0.), 1.0);
        final DoubleVertex safeBMinusC = bMinusC.setWithMask(bMinusC.lessThanOrEqualToMask(0.), 1.0);

        final DoubleVertex leftMask = x.greaterThanOrEqualToMask(xMin).times(x.lessThanMask(c));
        final DoubleVertex rightMask = x.greaterThanMask(c).times(x.lessThanOrEqualToMask(xMax));
        final DoubleVertex modeMask = x.greaterThanOrEqualToMask(c).times(x.lessThanOrEqualToMask(c));

        final DoubleVertex left = x.minus(xMin).times(2.).div(range.times(safeCMinusA));
        final DoubleVertex right = xMax.minus(x).times(2.).div(range.times(safeBMinusC));
        final DoubleVertex mode = range.reverseDiv(2.);

        return left.times(leftMask).plus(right.times(rightMask)).plus(mode.times(modeMask)).log();
    }

    /**
     * Element-wise derivatives of the log pdf. They are zero outside of the support.
     */
    public Diffs dLogProb(DoubleTensor x) {
        final Regions r = new Regions(x);

        final DoubleTensor range = xMax.minus(xMin);
        final DoubleTensor invRange = range.reciprocal();
        final DoubleTensor invCMinusA = r.safeCMinusA.reciprocal();
        final DoubleTensor invBMinusC = r.safeBMinusC.reciprocal();

        // left: log(2) + log(x - a) - log(b - a) - log(c - a)
        final DoubleTensor xMinusA = x.minus(xMin);
        final DoubleTensor invXMinusA = xMinusA.setWithMask(xMinusA.lessThanOrEqualToMask(DoubleTensor.scalar(0.)), 1.0).reciprocalInPlace();
        final DoubleTensor leftDx = invXMinusA;
        final DoubleTensor leftDa = invXMinusA.unaryMinus().plusInPlace(invRange).plusInPlace(invCMinusA);
        final DoubleTensor leftDb = invRange.unaryMinus();
        final DoubleTensor leftDc = invCMinusA.unaryMinus();

        // right: log(2) + log(b - x) - log(b - a) - log(b - c)
        final DoubleTensor bMinusX = xMax.minus(x);
        final DoubleTensor invBMinusX = bMinusX.setWithMask(bMinusX.lessThanOrEqualToMask(DoubleTensor.scalar(0.)), 1.0).reciprocalInPlace();
        final DoubleTensor rightDx = invBMinusX.unaryMinus();
        final DoubleTensor rightDa = invRange;
        final DoubleTensor rightDb = invBMinusX.minus(invRange).minusInPlace(invBMinusC);
        final DoubleTensor rightDc = invBMinusC;

        // mode: log(2) - log(b - a)
        final DoubleTensor modeDa = invRange;
        final DoubleTensor modeDb = invRange.unaryMinus();

        final DoubleTensor dx = leftDx.times(r.left).plusInPlace(rightDx.times(r.right));
        final DoubleTensor da = leftDa.times(r.left).plusInPlace(rightDa.times(r.right)).plusInPlace(modeDa.times(r.mode));
        final DoubleTensor db = leftDb.times(r.left).plusInPlace(rightDb.times(r.right)).plusInPlace(modeDb.times(r.mode));
        final DoubleTensor dc = leftDc.times(r.left).plusInPlace(rightDc.times(r.right));

        return new Diffs()
            .put(A, da)
            .put(B, db)
            .put(C, dc)
            .put(X, dx);
    }

    /**
     * Masks (1.0 where true, 0.0 otherwise) for which piece of the pdf applies to each x, and denominators that are
     * safe to divide by even where c == a or c == b (the piece that would divide by zero is always masked out there).
     */
    private class Regions {
        final DoubleTensor left;
        final DoubleTensor right;
        final DoubleTensor mode;
        final DoubleTensor safeCMinusA;
        final DoubleTensor safeBMinusC;

        Regions(DoubleTensor x) {
            final DoubleTensor zero = DoubleTensor.scalar(0.);
            left = x.greaterThanOrEqualToMask(xMin).timesInPlace(x.lessThanMask(c));
            right = x.greaterThanMask(c).timesInPlace(x.lessThanOrEqualToMask(xMax));
            mode = x.greaterThanOrEqualToMask(c).timesInPlace(x.lessThanOrEqualToMask(c));

            final DoubleTensor cMinusA = c.minus(xMin);
            final DoubleTensor bMinusC = xMax.minus(c);
            safeCMinusA = cMinusA.setWithMask(cMinusA.lessThanOrEqualToMask(zero), 1.0);
            safeBMinusC = bMinusC.setWithMask(bMinusC.lessThanOrEqualToMask(zero), 1.0);
        }
    }

}
