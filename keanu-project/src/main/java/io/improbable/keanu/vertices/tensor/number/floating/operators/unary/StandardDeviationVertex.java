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

public class StandardDeviationVertex<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    /**
     * @param inputVertex the vertex
     */
    public StandardDeviationVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex) {
        super(new long[0], inputVertex, inputVertex.ofType());
    }

    @Override
    protected TENSOR op(TENSOR value) {
        return value.standardDeviation();
    }

    private DoubleTensor jacobian() {
        // s = sqrt(sum((x - m)^2) / (n - 1)) so ds/dx_j = (x_j - m) / ((n - 1) * s)
        final DoubleTensor x = inputVertex.getValue().toDouble();
        final long n = x.getLength();
        final DoubleTensor s = x.standardDeviation();
        final DoubleTensor dsdx = x.minus(x.mean()).divInPlace(s.times((double) (n - 1)));
        return dsdx.reshape(1, n);
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