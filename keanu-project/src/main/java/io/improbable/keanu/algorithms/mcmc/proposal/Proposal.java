package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class Proposal {

    private final Map<Variable, Object> perVariableProposalTo;
    private final Map<Variable, Object> perVariableProposalFrom;
    private Double recordedLogProbAtToGivenFrom;

    public Proposal() {
        this.perVariableProposalTo = new HashMap<>();
        this.perVariableProposalFrom = new HashMap<>();
    }

    public <T> void setProposal(Variable<T, ?> variable, T to) {
        perVariableProposalFrom.put(variable, variable.getValue());
        perVariableProposalTo.put(variable, to);
    }

    public <T> T getProposalTo(Variable<T, ?> variable) {
        return (T) perVariableProposalTo.get(variable);
    }

    public Map<VariableReference, Object> getProposalTo() {
        Map<VariableReference, Object> asMap = new HashMap<>();

        for (Map.Entry<Variable, Object> entry : perVariableProposalTo.entrySet()) {
            asMap.put(entry.getKey().getReference(), entry.getValue());
        }

        return asMap;
    }

    public <T> T getProposalFrom(Variable<T, ?> variable) {
        return (T) perVariableProposalFrom.get(variable);
    }

    public Set<Variable> getVariablesWithProposal() {
        return perVariableProposalTo.keySet();
    }

    /**
     * Some proposal distributions can only evaluate q(x'|x) correctly at the time the proposal is created,
     * e.g. when q depends on the state of the rest of the network which changes once the proposal is applied.
     *
     * @return q(x'|x) if it was recorded when the proposal was created, otherwise null
     */
    public Double getRecordedLogProbAtToGivenFrom() {
        return recordedLogProbAtToGivenFrom;
    }

    public void setRecordedLogProbAtToGivenFrom(double logProb) {
        this.recordedLogProbAtToGivenFrom = logProb;
    }
}
