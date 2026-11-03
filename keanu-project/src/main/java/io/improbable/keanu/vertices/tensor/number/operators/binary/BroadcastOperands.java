package io.improbable.keanu.vertices.tensor.number.operators.binary;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;

import java.util.Arrays;

/**
 * Broadcasts the two operands of an element-wise binary vertex to their common shape, so that vertices built on
 * operations that require equal shapes (e.g. WhereVertex) broadcast in the same way as the tensor operations.
 */
final class BroadcastOperands {

    private BroadcastOperands() {
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static <V extends Vertex<?, ?>> V[] broadcast(V left, V right) {
        final long[] shape = TensorShape.getBroadcastResultShape(left.getShape(), right.getShape());
        final Vertex[] operands = new Vertex[]{to(left, shape), to(right, shape)};
        return (V[]) operands;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static Vertex to(Vertex vertex, long[] shape) {
        if (Arrays.equals(vertex.getShape(), shape)) {
            return vertex;
        }
        return ((TensorVertex) vertex).broadcast(shape);
    }
}
