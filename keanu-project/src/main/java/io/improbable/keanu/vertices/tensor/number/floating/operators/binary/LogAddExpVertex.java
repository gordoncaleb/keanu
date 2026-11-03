package io.improbable.keanu.vertices.tensor.number.floating.operators.binary;

import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.BinaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.AutoDiffBroadcast;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ReverseModePartialDerivative;

import java.util.HashMap;
import java.util.Map;

public class LogAddExpVertex<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends BinaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    public LogAddExpVertex(@LoadVertexParam(LEFT_NAME) TensorVertex<T, TENSOR, VERTEX> left,
                           @LoadVertexParam(RIGHT_NAME) TensorVertex<T, TENSOR, VERTEX> right) {
        super(left, right, left.ofType());
    }

    @Override
    protected TENSOR op(TENSOR left, TENSOR right) {
        return left.logAddExp(right);
    }

    /**
     * out = log(e^l + e^r) so d out/dl = e^(l - out) and d out/dr = e^(r - out)
     */
    private DoubleTensor[] weights() {
        final DoubleTensor l = left.getValue().toDouble();
        final DoubleTensor r = right.getValue().toDouble();
        final DoubleTensor out = getValue().toDouble();
        return new DoubleTensor[]{l.minus(out).expInPlace(), r.minus(out).expInPlace()};
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        final ForwardModePartialDerivative dLeft = derivativeOfParentsWithRespectToInput.getOrDefault(left, ForwardModePartialDerivative.EMPTY);
        final ForwardModePartialDerivative dRight = derivativeOfParentsWithRespectToInput.getOrDefault(right, ForwardModePartialDerivative.EMPTY);
        final DoubleTensor[] w = weights();

        return dLeft.multiply(w[0]).add(dRight.multiply(w[1]), this.getShape());
    }

    @Override
    public Map<Vertex, ReverseModePartialDerivative> reverseModeAutoDifferentiation(ReverseModePartialDerivative derivativeOfOutputWithRespectToSelf) {
        final DoubleTensor[] w = weights();
        final Map<Vertex, ReverseModePartialDerivative> partials = new HashMap<>();

        partials.merge(left, AutoDiffBroadcast.correctForBroadcastPartialReverse(
            derivativeOfOutputWithRespectToSelf.multiply(w[0]), this.getShape(), left.getShape()
        ), ReverseModePartialDerivative::add);
        partials.merge(right, AutoDiffBroadcast.correctForBroadcastPartialReverse(
            derivativeOfOutputWithRespectToSelf.multiply(w[1]), this.getShape(), right.getShape()
        ), ReverseModePartialDerivative::add);

        return partials;
    }
}