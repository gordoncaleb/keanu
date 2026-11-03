package io.improbable.keanu.vertices.tensor.number.operators.unary;

import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
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

public class MaxUnaryVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    public MaxUnaryVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex) {
        // a full reduction always produces a scalar
        super(Tensor.SCALAR_SHAPE, inputVertex, inputVertex.ofType());
    }

    @Override
    protected TENSOR op(TENSOR value) {
        return value.max();
    }

    /**
     * The gradient flows to the maximum element. If several elements are equal to the maximum it is split equally
     * between them.
     */
    private DoubleTensor jacobian() {
        final DoubleTensor input = inputVertex.getValue().toDouble();
        final DoubleTensor selected = input.elementwiseEquals(input.max()).toDoubleMask();
        final double count = selected.sum().scalar();
        return selected.divInPlace(count).reshape(1, input.getLength());
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        final ForwardModePartialDerivative dInput = derivativeOfParentsWithRespectToInput.getOrDefault(inputVertex, ForwardModePartialDerivative.EMPTY);
        return JacobianAutoDiff.forward(dInput, jacobian(), getShape());
    }

    @Override
    public Map<Vertex, ReverseModePartialDerivative> reverseModeAutoDifferentiation(ReverseModePartialDerivative derivativeOfOutputWithRespectToSelf) {
        return Collections.singletonMap(inputVertex, JacobianAutoDiff.reverse(derivativeOfOutputWithRespectToSelf, jacobian(), inputVertex.getShape()));
    }
}
