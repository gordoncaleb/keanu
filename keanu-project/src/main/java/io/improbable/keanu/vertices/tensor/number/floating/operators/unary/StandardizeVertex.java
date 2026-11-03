package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.UnaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.JacobianAutoDiff;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ReverseModePartialDerivative;

import java.util.Collections;
import java.util.Map;

public class StandardizeVertex<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    public StandardizeVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex) {
        super(inputVertex);
    }

    @Override
    protected TENSOR op(TENSOR value) {
        return value.standardize();
    }

    private DoubleTensor jacobian() {
        // z_i = (x_i - m) / s
        // dz_i/dx_j = (delta_ij - 1/n) / s - (x_i - m) * (x_j - m) / ((n - 1) * s^3)
        final DoubleTensor x = inputVertex.getValue().toDouble();
        final int n = (int) x.getLength();
        final double s = x.standardDeviation().scalar();
        final DoubleTensor centered = x.minus(x.mean()).reshape(n, 1);

        final DoubleTensor identityPart = DoubleTensor.eye(n).minusInPlace(1.0 / n).divInPlace(s);
        final DoubleTensor outerPart = centered.matrixMultiply(centered, false, true).divInPlace((n - 1) * s * s * s);
        return identityPart.minusInPlace(outerPart);
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        return JacobianAutoDiff.forward(derivativeOfParentsWithRespectToInput.get(inputVertex), jacobian(), getShape());
    }

    @Override
    public Map<Vertex, ReverseModePartialDerivative> reverseModeAutoDifferentiation(ReverseModePartialDerivative derivativeOfOutputWithRespectToSelf) {
        return Collections.singletonMap(inputVertex, JacobianAutoDiff.reverse(derivativeOfOutputWithRespectToSelf, jacobian(), inputVertex.getShape()));
    }
}