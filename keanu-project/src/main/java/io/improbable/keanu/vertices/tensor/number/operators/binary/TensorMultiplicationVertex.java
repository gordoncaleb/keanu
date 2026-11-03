package io.improbable.keanu.vertices.tensor.number.operators.binary;

import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.TensorMulByMatrixMul;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.BinaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ReverseModePartialDerivative;

import java.util.HashMap;
import java.util.Map;

public class TensorMultiplicationVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends BinaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private static final String DIMS_LEFT = "dimsLeft";
    private static final String DIMS_RIGHT = "dimsRight";

    private final int[] dimsLeft;
    private final int[] dimsRight;

    /**
     * Tensor multiplies one vertex by another. C = AB.
     *
     * @param left      the left vertex for operand
     * @param right     the right vertex for operand
     * @param dimsLeft  The dimensions of the left for multiplying. The left shape at these dimensions must align with the
     *                  shape of the corresponding right vertex at its specified dimensions.
     * @param dimsRight The dimensions of the right for multiplying. The right shape at these dimensions must align with the
     *                  shape of the corresponding left vertex at its specified dimensions.
     */
    public TensorMultiplicationVertex(@LoadVertexParam(LEFT_NAME) TensorVertex<T, TENSOR, VERTEX> left,
                                      @LoadVertexParam(RIGHT_NAME) TensorVertex<T, TENSOR, VERTEX> right,
                                      @LoadVertexParam(DIMS_LEFT) int[] dimsLeft,
                                      @LoadVertexParam(DIMS_RIGHT) int[] dimsRight) {
        super(TensorMulByMatrixMul.getResultShape(left.getShape(), right.getShape(), dimsLeft, dimsRight),
            left, right, left.ofType());
        this.dimsLeft = dimsLeft;
        this.dimsRight = dimsRight;
    }

    @Override
    protected TENSOR op(TENSOR l, TENSOR r) {
        return l.tensorMultiply(r, dimsLeft, dimsRight);
    }

    /**
     * C = A . B contracted over dimsLeft of A and dimsRight of B. C's dimensions are A's kept dimensions followed by
     * B's kept dimensions. dC = dA . B + A . dB
     */
    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        final ForwardModePartialDerivative dLeft = derivativeOfParentsWithRespectToInput.getOrDefault(left, ForwardModePartialDerivative.EMPTY);
        final ForwardModePartialDerivative dRight = derivativeOfParentsWithRespectToInput.getOrDefault(right, ForwardModePartialDerivative.EMPTY);

        final DoubleTensor A = left.getValue().toDouble();
        final DoubleTensor B = right.getValue().toDouble();
        final int[] dimsA = absolute(dimsLeft, A.getRank());
        final int[] dimsB = absolute(dimsRight, B.getRank());

        DoubleTensor result = null;
        long[] wrtShape = null;

        if (dLeft.isPresent()) {
            // [wrt..., A...] . B -> [wrt..., A kept..., B kept...]
            wrtShape = dLeft.getWrtShape();
            result = dLeft.get().tensorMultiply(B, offset(dimsA, wrtShape.length), dimsB);
        }

        if (dRight.isPresent()) {
            // A . [wrt..., B...] -> [A kept..., wrt..., B kept...] then move wrt to the front
            final long[] rightWrtShape = dRight.getWrtShape();
            final int aKept = A.getRank() - dimsA.length;
            final DoubleTensor fromRight = A.tensorMultiply(dRight.get(), dimsA, offset(dimsB, rightWrtShape.length));

            final int rank = fromRight.getRank();
            final int[] permutation = new int[rank];
            int p = 0;
            for (int i = 0; i < rightWrtShape.length; i++) {
                permutation[p++] = aKept + i;
            }
            for (int i = 0; i < aKept; i++) {
                permutation[p++] = i;
            }
            for (int i = aKept + rightWrtShape.length; i < rank; i++) {
                permutation[p++] = i;
            }

            final DoubleTensor permuted = fromRight.permute(permutation);
            wrtShape = rightWrtShape;
            result = result == null ? permuted : result.plus(permuted);
        }

        return result == null ? ForwardModePartialDerivative.EMPTY : new ForwardModePartialDerivative(wrtShape, result);
    }

    @Override
    public Map<Vertex, ReverseModePartialDerivative> reverseModeAutoDifferentiation(ReverseModePartialDerivative derivativeOfOutputWithRespectToSelf) {
        final DoubleTensor G = derivativeOfOutputWithRespectToSelf.get();
        final long[] ofShape = derivativeOfOutputWithRespectToSelf.getOfShape();
        final int ofRank = ofShape.length;

        final DoubleTensor A = left.getValue().toDouble();
        final DoubleTensor B = right.getValue().toDouble();
        final int[] dimsA = absolute(dimsLeft, A.getRank());
        final int[] dimsB = absolute(dimsRight, B.getRank());
        final int[] keptA = kept(dimsA, A.getRank());
        final int[] keptB = kept(dimsB, B.getRank());

        final Map<Vertex, ReverseModePartialDerivative> partials = new HashMap<>();

        // dA: contract G's B-kept dims with B's kept dims -> [of..., A kept..., B contracted (ascending B dim order)...]
        final int[] gBKeptDims = new int[keptB.length];
        for (int i = 0; i < keptB.length; i++) {
            gBKeptDims[i] = ofRank + keptA.length + i;
        }
        final DoubleTensor dAUnordered = G.tensorMultiply(B, gBKeptDims, keptB);
        final int[] bContractedAscending = sorted(dimsB);
        final int[] dAOrder = new int[keptA.length + bContractedAscending.length];
        for (int i = 0; i < keptA.length; i++) {
            dAOrder[i] = keptA[i];
        }
        for (int i = 0; i < bContractedAscending.length; i++) {
            dAOrder[keptA.length + i] = dimsA[indexOf(dimsB, bContractedAscending[i])];
        }
        partials.merge(left, new ReverseModePartialDerivative(ofShape, toOriginalOrder(dAUnordered, ofRank, dAOrder)), ReverseModePartialDerivative::add);

        // dB: contract A's kept dims with G's A-kept dims -> [A contracted (ascending A dim order)..., of..., B kept...]
        final int[] gAKeptDims = new int[keptA.length];
        for (int i = 0; i < keptA.length; i++) {
            gAKeptDims[i] = ofRank + i;
        }
        final DoubleTensor dBUnordered = A.tensorMultiply(G, keptA, gAKeptDims);
        final int[] aContractedAscending = sorted(dimsA);

        // move of dims to the front: [of..., A contracted..., B kept...]
        final int rank = dBUnordered.getRank();
        final int[] ofFirst = new int[rank];
        int p = 0;
        for (int i = 0; i < ofRank; i++) {
            ofFirst[p++] = aContractedAscending.length + i;
        }
        for (int i = 0; i < aContractedAscending.length; i++) {
            ofFirst[p++] = i;
        }
        for (int i = aContractedAscending.length + ofRank; i < rank; i++) {
            ofFirst[p++] = i;
        }
        final DoubleTensor dBOfFirst = dBUnordered.permute(ofFirst);

        final int[] dBOrder = new int[aContractedAscending.length + keptB.length];
        for (int i = 0; i < aContractedAscending.length; i++) {
            dBOrder[i] = dimsB[indexOf(dimsA, aContractedAscending[i])];
        }
        for (int i = 0; i < keptB.length; i++) {
            dBOrder[aContractedAscending.length + i] = keptB[i];
        }
        partials.merge(right, new ReverseModePartialDerivative(ofShape, toOriginalOrder(dBOfFirst, ofRank, dBOrder)), ReverseModePartialDerivative::add);

        return partials;
    }

    /**
     * @param tensor       shaped [of..., operand dims in the order given by currentOrder]
     * @param ofRank       number of leading of dimensions
     * @param currentOrder the operand dimension held at each trailing position
     * @return the tensor permuted to [of..., operand dims in their original order]
     */
    private static DoubleTensor toOriginalOrder(DoubleTensor tensor, int ofRank, int[] currentOrder) {
        final int[] permutation = new int[ofRank + currentOrder.length];
        for (int i = 0; i < ofRank; i++) {
            permutation[i] = i;
        }
        for (int operandDim = 0; operandDim < currentOrder.length; operandDim++) {
            permutation[ofRank + operandDim] = ofRank + indexOf(currentOrder, operandDim);
        }
        return tensor.permute(permutation);
    }

    private static int[] absolute(int[] dims, int rank) {
        final int[] result = new int[dims.length];
        for (int i = 0; i < dims.length; i++) {
            result[i] = dims[i] < 0 ? dims[i] + rank : dims[i];
        }
        return result;
    }

    private static int[] offset(int[] dims, int by) {
        final int[] result = new int[dims.length];
        for (int i = 0; i < dims.length; i++) {
            result[i] = dims[i] + by;
        }
        return result;
    }

    private static int[] kept(int[] contracted, int rank) {
        final int[] result = new int[rank - contracted.length];
        int k = 0;
        for (int i = 0; i < rank; i++) {
            if (indexOf(contracted, i) < 0) {
                result[k++] = i;
            }
        }
        return result;
    }

    private static int[] sorted(int[] dims) {
        final int[] copy = dims.clone();
        java.util.Arrays.sort(copy);
        return copy;
    }

    private static int indexOf(int[] array, int value) {
        for (int i = 0; i < array.length; i++) {
            if (array[i] == value) {
                return i;
            }
        }
        return -1;
    }

    @SaveVertexParam(DIMS_LEFT)
    public int[] getDimsLeft() {
        return dimsLeft;
    }

    @SaveVertexParam(DIMS_RIGHT)
    public int[] getDimsRight() {
        return dimsRight;
    }
}
