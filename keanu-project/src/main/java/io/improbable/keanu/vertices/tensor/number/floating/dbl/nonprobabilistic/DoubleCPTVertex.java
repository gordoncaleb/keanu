package io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.generic.nonprobabilistic.CPTCondition;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ReverseModePartialDerivative;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DoubleCPTVertex extends VertexImpl<DoubleTensor, DoubleVertex> implements DoubleVertex, Differentiable, NonProbabilistic<DoubleTensor>, NonSaveableVertex {

    private final List<Vertex<? extends Tensor<?, ?>, ?>> inputs;
    private final Map<CPTCondition, DoubleVertex> conditions;
    private final DoubleVertex defaultResult;

    public DoubleCPTVertex(List<Vertex<? extends Tensor<?, ?>, ?>> inputs,
                           Map<CPTCondition, DoubleVertex> conditions,
                           DoubleVertex defaultResult) {
        super(defaultResult.getShape());
        this.inputs = inputs;
        this.conditions = conditions;
        this.defaultResult = defaultResult;
        addParents(inputs);
        addParents(conditions.values());
        addParent(defaultResult);
    }

    @Override
    public DoubleTensor calculate() {
        final CPTCondition condition = CPTCondition.from(inputs, v -> v.getValue().scalar());
        DoubleVertex vertex = conditions.get(condition);
        return vertex == null ? defaultResult.getValue() : vertex.getValue();
    }

    @Override
    /**
     * The selected outcome's partial. If it doesn't depend on the input but another outcome does, the partial is
     * zero rather than absent, since the output still depends on the input through a differentiable vertex.
     */
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        final CPTCondition condition = CPTCondition.from(inputs, (vertex) -> vertex.getValue().scalar());
        DoubleVertex vertex = conditions.get(condition);
        DoubleVertex selected = vertex == null ? defaultResult : vertex;
        ForwardModePartialDerivative selectedPartial = derivativeOfParentsWithRespectToInput.get(selected);
        if (selectedPartial != null && selectedPartial.isPresent()) {
            return selectedPartial;
        }

        List<DoubleVertex> outcomes = new ArrayList<>(conditions.values());
        outcomes.add(defaultResult);
        for (DoubleVertex outcome : outcomes) {
            ForwardModePartialDerivative partial = derivativeOfParentsWithRespectToInput.get(outcome);
            if (partial != null && partial.isPresent()) {
                long[] wrtShape = partial.getWrtShape();
                return new ForwardModePartialDerivative(wrtShape, DoubleTensor.zeros(TensorShape.concat(wrtShape, getShape())));
            }
        }
        return ForwardModePartialDerivative.EMPTY;
    }

    /**
     * The selected outcome gets the partial and the others get a zero partial (see the forward mode).
     */
    public Map<Vertex, ReverseModePartialDerivative> reverseModeAutoDifferentiation(ReverseModePartialDerivative derivativeOfOutputWithRespectToSelf) {
        final CPTCondition condition = CPTCondition.from(inputs, (vertex) -> vertex.getValue().scalar());
        DoubleVertex conditionVertex = conditions.get(condition);
        DoubleVertex selected = conditionVertex == null ? defaultResult : conditionVertex;

        List<DoubleVertex> outcomes = new ArrayList<>(conditions.values());
        outcomes.add(defaultResult);

        Map<Vertex, ReverseModePartialDerivative> partials = new HashMap<>();
        for (DoubleVertex outcome : outcomes) {
            ReverseModePartialDerivative partial = outcome == selected
                ? derivativeOfOutputWithRespectToSelf
                : derivativeOfOutputWithRespectToSelf.multiply(0.0);
            partials.merge(outcome, partial, ReverseModePartialDerivative::add);
        }
        return partials;
    }
}
