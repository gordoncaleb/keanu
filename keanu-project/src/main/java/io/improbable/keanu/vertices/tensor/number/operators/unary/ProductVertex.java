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

import static io.improbable.keanu.tensor.TensorShape.getReductionResultShape;

public class ProductVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private final static String OVER_DIMENSIONS = "overDimensions";
    private final int[] overDimensions;

    public ProductVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                         @LoadVertexParam(value = OVER_DIMENSIONS, isNullable = true) int[] overDimensions) {
        super(getReductionResultShape(inputVertex.getShape(), overDimensions), inputVertex, inputVertex.ofType());
        this.overDimensions = overDimensions;
    }

    public ProductVertex(TensorVertex<T, TENSOR, VERTEX> inputVertex) {
        super(new long[0], inputVertex, inputVertex.ofType());
        this.overDimensions = null;
    }

    @Override
    protected TENSOR op(TENSOR value) {
        if (overDimensions == null) {
            return value.product();
        } else {
            return value.product(overDimensions);
        }
    }

    /**
     * A product is affine in each element when the others are held fixed, so its Jacobian is exact from
     * {@link JacobianAutoDiff#jacobianOfElementwiseAffineOp}. Unlike dividing the product by each element this is
     * correct where elements are zero.
     */
    private DoubleTensor jacobian() {
        return JacobianAutoDiff.jacobianOfElementwiseAffineOp(
            x -> overDimensions == null ? x.product() : x.product(overDimensions),
            inputVertex.getValue().toDouble()
        );
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        return JacobianAutoDiff.forward(derivativeOfParentsWithRespectToInput.get(inputVertex), jacobian(), getShape());
    }

    @Override
    public Map<Vertex, ReverseModePartialDerivative> reverseModeAutoDifferentiation(ReverseModePartialDerivative partial) {
        return Collections.singletonMap(inputVertex, JacobianAutoDiff.reverse(partial, jacobian(), inputVertex.getShape()));
    }

    @SaveVertexParam(value = OVER_DIMENSIONS, isNullable = true)
    public int[] getOverDimensions() {
        return overDimensions;
    }
}
