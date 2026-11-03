package io.improbable.keanu.vertices.tensor.number.operators.unary;

import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
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

public class CumSumVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private final static String REQUESTED_DIMENSION = "requestedDimension";
    private final int requestedDimension;

    public CumSumVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                        @LoadVertexParam(REQUESTED_DIMENSION) int requestedDimension) {
        super(inputVertex, inputVertex.ofType());
        this.requestedDimension = requestedDimension;
    }

    @Override
    protected TENSOR op(TENSOR value) {
        return value.cumSum(requestedDimension);
    }

    @SaveVertexParam(REQUESTED_DIMENSION)
    public int getRequestedDimension() {
        return this.requestedDimension;
    }

    private DoubleTensor jacobian() {
        // cumulative sum is linear, so its Jacobian is exact from the affine-per-element construction
        return JacobianAutoDiff.jacobianOfElementwiseAffineOp(x -> x.cumSum(requestedDimension), inputVertex.getValue().toDouble());
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
