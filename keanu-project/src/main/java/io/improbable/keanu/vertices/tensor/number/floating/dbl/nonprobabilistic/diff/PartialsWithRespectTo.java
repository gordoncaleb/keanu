package io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Map;

@AllArgsConstructor
public class PartialsWithRespectTo {

    @Getter
    private final Vertex wrt;

    private final Map<VertexId, ForwardModePartialDerivative> partials;

    public DoubleTensor of(Vertex vertex) {
        return of(vertex.getId());
    }

    /**
     * @param id the vertex to get the partial of
     * @return the partial derivative of the given vertex with respect to the "wrt" vertex
     * @throws IllegalArgumentException if there is no partial, i.e. the given vertex does not depend on the "wrt"
     *                                  vertex through differentiable vertices (e.g. it depends on it via floor or round)
     */
    public DoubleTensor of(VertexId id) {
        final ForwardModePartialDerivative partial = partials.get(id);
        if (partial == null || !partial.isPresent()) {
            throw new IllegalArgumentException(
                "No partial derivative of " + id + " with respect to " + wrt.getId()
                    + ". It does not depend on that vertex through differentiable vertices."
            );
        }
        return partial.getOfWrt();
    }
}
