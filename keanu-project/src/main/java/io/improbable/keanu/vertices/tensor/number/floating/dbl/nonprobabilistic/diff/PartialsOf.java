package io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Map;

@AllArgsConstructor
public class PartialsOf {

    @Getter
    private final Vertex<?, ?> of;

    private final Map<VertexId, ReverseModePartialDerivative> partials;

    public DoubleTensor withRespectTo(Vertex vertex) {
        return withRespectTo(vertex.getId());
    }

    /**
     * @param id the vertex to get the partial with respect to
     * @return the partial derivative of the "of" vertex with respect to the given vertex
     * @throws IllegalArgumentException if there is no partial, i.e. the "of" vertex does not depend on the given
     *                                  vertex through differentiable vertices (e.g. it depends on it via floor or round)
     */
    public DoubleTensor withRespectTo(VertexId id) {
        final ReverseModePartialDerivative partial = partials.get(id);
        if (partial == null || !partial.isPresent()) {
            throw new IllegalArgumentException(
                "No partial derivative of " + of.getId() + " with respect to " + id
                    + ". It does not depend on that vertex through differentiable vertices."
            );
        }
        return partial.get();
    }

    public Map<VertexId, ReverseModePartialDerivative> asMap() {
        return partials;
    }

}
