package io.improbable.keanu.golden;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.If;
import io.improbable.keanu.vertices.tensor.bool.BooleanVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.BooleanProxyVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.multiple.BooleanConcatenationVertex;
import io.improbable.keanu.vertices.tensor.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.tensor.generic.nonprobabilistic.MultiplexerVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.IntegerProxyVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.operators.multiple.IntegerConcatenationVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic.BinomialVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic.GeometricVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic.MultinomialVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic.PoissonVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic.UniformIntVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.BetaVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.CauchyVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.ChiSquaredVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.DirichletVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.ExponentialVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GammaVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.HalfCauchyVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.HalfGaussianVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.InverseGammaVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.KDEVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.LaplaceVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.LogNormalVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.LogisticVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.MultivariateGaussianVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.ParetoVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.StudentTVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.TriangularVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.utility.AssertVertex;

import java.util.ArrayList;
import java.util.List;

/**
 * A model containing (at least) one labeled instance of every saveable vertex type, so the saved files record the
 * exact type name and parameter names of each, and the expected values of every node.
 */
final class CatalogModel {

    private CatalogModel() {
    }

    private static final List<Vertex<?, ?>> NODES = new ArrayList<>();

    private static <V extends Vertex<?, ?>> V add(String label, V vertex) {
        vertex.setLabel(label);
        NODES.add(vertex);
        return vertex;
    }

    static BayesianNetwork build() {
        NODES.clear();
        DoubleVertex a = add("a", ConstantVertex.of(DoubleTensor.create(new double[]{0.2, 0.5, 0.7, 0.3, 0.9, 0.4}, 2, 3)));
        DoubleVertex b = add("b", ConstantVertex.of(DoubleTensor.create(new double[]{0.6, 0.1, 0.8, 0.25, 0.35, 0.45}, 2, 3)));
        DoubleVertex row = add("row", ConstantVertex.of(DoubleTensor.create(0.3, 0.6, 0.9)));
        DoubleVertex square = add("square", ConstantVertex.of(DoubleTensor.create(new double[]{4, 1, 0.5, 1, 3, 0.2, 0.5, 0.2, 2}, 3, 3)));
        DoubleVertex big = add("big", ConstantVertex.of(DoubleTensor.create(1.5, -2.5, 3.5)));
        DoubleVertex withNaN = add("withNaN", ConstantVertex.of(DoubleTensor.create(1.0, Double.NaN, 3.0)));
        IntegerVertex i = add("i", ConstantVertex.of(IntegerTensor.create(new int[]{-7, -3, 0, 5, 9, 2}, 2, 3)));
        IntegerVertex j = add("j", ConstantVertex.of(IntegerTensor.create(2, -2, 3)));
        BooleanVertex t = add("t", ConstantVertex.of(BooleanTensor.create(true, false, true)));
        BooleanVertex f = add("f", ConstantVertex.of(BooleanTensor.create(false, false, true)));

        // element-wise binary
        add("plus", a.plus(b));
        add("minus", a.minus(b));
        add("times", a.times(b));
        add("div", a.div(b));
        add("pow", a.pow(b));
        add("maxBinary", a.max(b));
        add("minBinary", a.min(b));
        add("atan2", a.atan2(b));
        add("logAddExp", a.logAddExp(b));
        add("logAddExp2", a.logAddExp2(b));
        add("safeLogTimes", a.safeLogTimes(b));
        add("gtMask", a.greaterThanMask(b));
        add("geMask", a.greaterThanOrEqualToMask(b));
        add("ltMask", a.lessThanMask(b));
        add("leMask", a.lessThanOrEqualToMask(b));
        add("broadcastPlus", a.plus(row));

        // element-wise unary
        add("abs", big.abs());
        add("unaryMinus", big.unaryMinus());
        add("sign", big.sign());
        add("exp", a.exp());
        add("exp2", a.exp2());
        add("expM1", a.expM1());
        add("log", a.log());
        add("log1p", a.log1p());
        add("log2", a.log2());
        add("log10", a.log10());
        add("sqrt", a.sqrt());
        add("reciprocal", a.reciprocal());
        add("sin", a.sin());
        add("cos", a.cos());
        add("tan", a.tan());
        add("asin", a.asin());
        add("acos", a.acos());
        add("atan", a.atan());
        add("sinh", a.sinh());
        add("cosh", a.cosh());
        add("tanh", a.tanh());
        add("asinh", a.asinh());
        add("acosh", a.plus(1.5).acosh());
        add("atanh", a.atanh());
        add("sigmoid", a.sigmoid());
        add("floor", big.floor());
        add("ceil", big.ceil());
        add("round", big.round());
        add("logGamma", a.logGamma());
        add("digamma", a.digamma());
        add("trigamma", a.trigamma());
        add("replaceNaN", withNaN.replaceNaN(0.5));
        add("setWithMask", a.setWithMask(a.greaterThanMask(b), 9.0));

        // reductions and scans
        add("sumAll", a.sum());
        add("sumDims", a.sum(1));
        add("product", a.product());
        add("productDims", a.product(0));
        add("mean", a.mean());
        add("meanDims", a.mean(1));
        add("maxUnary", a.max());
        add("minUnary", a.min());
        add("standardDeviation", a.standardDeviation());
        add("standardize", a.standardize());
        add("cumSum", a.cumSum(1));
        add("cumProd", a.cumProd(0));
        add("argMax", a.argMax(1));
        add("argMin", a.argMin());
        add("nanArgMax", withNaN.nanArgMax());
        add("nanArgMin", withNaN.nanArgMin());

        // shapes
        add("reshape", a.reshape(3, 2));
        add("permute", a.permute(1, 0));
        add("broadcast", row.broadcast(2, 3));
        add("sliceIndex", a.slice(1, 0));
        add("stridedSlice", a.slice("::-1, 1:"));
        add("take", a.take(1, 2));
        add("diag", row.diag());
        add("diagPart", square.diagPart());
        add("triLower", square.triLower(0));
        add("triUpper", square.triUpper(1));
        add("fillTriangular", ConstantVertex.of(DoubleTensor.create(1, 2, 3, 4, 5, 6)).fillTriangular(true, true));
        add("trianglePart", square.trianglePart(true));
        add("concat", new ConcatenationVertex(0, a, b));
        add("intConcat", new IntegerConcatenationVertex(0, i, i));
        add("boolConcat", new BooleanConcatenationVertex(0, t, f));

        // linear algebra
        add("matrixMultiply", a.matrixMultiply(square));
        add("tensorMultiply", a.tensorMultiply(square, new int[]{1}, new int[]{0}));
        add("matrixInverse", square.matrixInverse());
        add("matrixDeterminant", square.matrixDeterminant());
        add("cholesky", square.choleskyDecomposition());
        add("choleskyInverse", square.choleskyDecomposition().choleskyInverse());

        // comparisons, logic, casts
        add("greaterThan", a.greaterThan(b));
        add("greaterThanOrEqual", a.greaterThanOrEqual(b));
        add("lessThan", a.lessThan(b));
        add("lessThanOrEqual", a.lessThanOrEqual(b));
        add("equals", a.elementwiseEquals(b));
        add("notEquals", a.notEqualTo(b));
        add("equalsWithinEpsilon", a.equalsWithinEpsilon(b, 0.3));
        add("and", t.and(f));
        add("or", t.or(f));
        add("xor", t.xor(f));
        add("not", t.not());
        add("allTrue", t.allTrue());
        add("allFalse", t.allFalse());
        add("anyTrue", t.anyTrue());
        add("anyFalse", t.anyFalse());
        add("isNaN", withNaN.isNaN());
        add("notNaN", withNaN.notNaN());
        add("isFinite", withNaN.isFinite());
        add("isInfinite", withNaN.isInfinite());
        add("isPositiveInfinity", withNaN.isPositiveInfinity());
        add("isNegativeInfinity", withNaN.isNegativeInfinity());
        add("boolToDoubleMask", t.toDoubleMask());
        add("boolToIntegerMask", t.toIntegerMask());
        add("doubleToInteger", big.toInteger());
        add("integerToDouble", i.toDouble());
        add("doubleToBoolean", big.toBoolean());

        // integer ops
        add("intPlus", i.plus(j));
        add("intMinus", i.minus(j));
        add("intTimes", i.times(j));
        add("intDiv", i.div(j));
        add("intMod", i.mod(j));
        add("intPow", i.pow(j));
        add("intAbs", i.abs());
        add("intSum", i.sum());
        add("intMax", i.max(j));

        // selection
        add("where", If.isTrue(a.greaterThan(b)).then(a).orElse(b));
        add("multiplexer", new MultiplexerVertex<>(ConstantVertex.of(1), a, b));
        add("assert", new AssertVertex(t.or(t.not()), "always true"));

        // proxies
        IntegerProxyVertex intProxy = add("intProxy", new IntegerProxyVertex(new long[]{3}, "intProxy"));
        intProxy.setParent(j);
        BooleanProxyVertex boolProxy = add("boolProxy", new BooleanProxyVertex(new long[]{3}, "boolProxy"));
        boolProxy.setParent(t);

        // probabilistic
        DoubleVertex mu = add("mu", new GaussianVertex(0.0, 1.0));
        mu.setValue(0.3);
        latent("gaussian", new GaussianVertex(new long[]{3}, mu, 2.0), DoubleTensor.create(0.1, -0.4, 0.8));
        latent("halfGaussian", new HalfGaussianVertex(new long[]{2}, 1.5), DoubleTensor.create(0.5, 2.0));
        latent("cauchy", new CauchyVertex(mu, 1.0), DoubleTensor.scalar(0.7));
        latent("halfCauchy", new HalfCauchyVertex(2.0), DoubleTensor.scalar(1.2));
        latent("beta", new BetaVertex(2.0, 3.0), DoubleTensor.scalar(0.35));
        latent("chiSquared", new ChiSquaredVertex(3), DoubleTensor.scalar(2.5));
        latent("dirichlet", new DirichletVertex(1.0, 2.0, 3.0), DoubleTensor.create(0.2, 0.3, 0.5));
        latent("exponential", new ExponentialVertex(1.5), DoubleTensor.scalar(0.4));
        latent("gamma", new GammaVertex(2.0, 1.5), DoubleTensor.scalar(2.4));
        latent("inverseGamma", new InverseGammaVertex(3.0, 0.5), DoubleTensor.scalar(0.3));
        latent("laplace", new LaplaceVertex(mu, 0.5), DoubleTensor.scalar(-0.2));
        latent("logNormal", new LogNormalVertex(0.0, 1.0), DoubleTensor.scalar(1.3));
        latent("logistic", new LogisticVertex(0.0, 1.0), DoubleTensor.scalar(0.6));
        latent("multivariateGaussian", new MultivariateGaussianVertex(
            ConstantVertex.of(DoubleTensor.create(0.5, -1.0)),
            ConstantVertex.of(DoubleTensor.create(new double[]{2, 0.3, 0.3, 1}, 2, 2))
        ), DoubleTensor.create(1.0, -0.5));
        latent("pareto", new ParetoVertex(1.0, 2.0), DoubleTensor.scalar(1.7));
        latent("studentT", new StudentTVertex(4), DoubleTensor.scalar(-0.8));
        latent("triangular", new TriangularVertex(0.0, 1.0, 0.25), DoubleTensor.scalar(0.6));
        latent("uniform", new UniformVertex(-1.0, 2.0), DoubleTensor.scalar(0.5));
        latent("kde", new KDEVertex(DoubleTensor.create(0.1, 0.4, 0.5, 1.2), 0.3), DoubleTensor.scalar(0.45));

        observed("bernoulli", new BernoulliVertex(new long[]{3}, ConstantVertex.of(0.3)), BooleanTensor.create(true, false, true));
        observed("binomial", new BinomialVertex(new long[]{2}, 0.4, 5), IntegerTensor.create(2, 4));
        observed("geometric", new GeometricVertex(new long[]{2}, 0.25), IntegerTensor.create(1, 3));
        observed("multinomial", new MultinomialVertex(5, ConstantVertex.of(DoubleTensor.create(0.2, 0.5, 0.3))), IntegerTensor.create(1, 3, 1));
        observed("poisson", new PoissonVertex(new long[]{3}, mu.exp()), IntegerTensor.create(0, 2, 1));
        observed("uniformInt", new UniformIntVertex(new long[]{2}, -2, 3), IntegerTensor.create(-1, 2));
        DoubleVertex gaussianObserved = new GaussianVertex(new long[]{2, 3}, a.plus(mu), 0.5);
        observed("gaussianObserved", gaussianObserved, DoubleTensor.create(new double[]{0.4, 0.9, 1.1, 0.6, 1.3, 0.5}, 2, 3));

        return new BayesianNetwork(a.getConnectedGraph().isEmpty() ? NODES : allConnected());
    }

    private static List<Vertex> allConnected() {
        java.util.Set<Vertex> all = new java.util.LinkedHashSet<>();
        for (Vertex<?, ?> v : NODES) {
            all.addAll(v.getConnectedGraph());
        }
        return new ArrayList<>(all);
    }

    @SuppressWarnings("unchecked")
    private static <T> void latent(String label, Vertex<T, ?> vertex, T value) {
        add(label, vertex);
        vertex.setValue(value);
    }

    @SuppressWarnings("unchecked")
    private static <T> void observed(String label, Vertex<T, ?> vertex, T value) {
        add(label, vertex);
        vertex.observe(value);
    }
}
