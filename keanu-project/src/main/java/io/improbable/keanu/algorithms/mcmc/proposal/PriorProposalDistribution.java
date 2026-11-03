package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;

import java.util.Collections;
import java.util.List;
import java.util.Set;

public class PriorProposalDistribution implements ProposalDistribution {
    private final ProposalNotifier proposalNotifier;

    public PriorProposalDistribution() {
        this(Collections.emptyList());
    }

    public PriorProposalDistribution(List<ProposalListener> listeners) {
        this.proposalNotifier = new ProposalNotifier(listeners);
    }

    @Override
    public Proposal getProposal(Set<? extends Variable> variables, KeanuRandom random) {
        Proposal proposal = new Proposal();
        for (Variable<?, ?> variable : variables) {
            setFor(variable, random, proposal);
        }

        // Each proposed value is sampled from its prior given the parents' current (pre-proposal) values.
        // q(x'|x) must be evaluated now: once the proposal is applied, any proposed variables that are parents of
        // other proposed variables will have changed, and the prior would be evaluated with the wrong parents.
        double logProbAtToGivenFrom = 0.0;
        for (Variable variable : proposal.getVariablesWithProposal()) {
            logProbAtToGivenFrom += ((Probabilistic) variable).logProb(proposal.getProposalTo(variable));
        }
        proposal.setRecordedLogProbAtToGivenFrom(logProbAtToGivenFrom);

        proposalNotifier.notifyProposalCreated(proposal);
        return proposal;
    }

    @Override
    public <T> double logProb(Probabilistic<T> variable, T ofValue, T givenValue) {
        return variable.logProb(ofValue);
    }

    @Override
    public double logProbAtToGivenFrom(Proposal proposal) {
        final Double recorded = proposal.getRecordedLogProbAtToGivenFrom();
        return recorded != null ? recorded : ProposalDistribution.super.logProbAtToGivenFrom(proposal);
    }

    private <T> void setFor(Variable<T, ?> variable, KeanuRandom random, Proposal proposal) {

        if (variable instanceof Vertex && variable instanceof Probabilistic) {
            Vertex<T, ?> vertex = (Vertex<T, ?>) variable;
            proposal.setProposal(variable, ((Probabilistic<T>) vertex).sample(random));
        } else {
            throw new IllegalArgumentException(this.getClass().getSimpleName() + " is to only be used with Keanu's Vertex");
        }
    }

    @Override
    public void onProposalRejected() {
        proposalNotifier.notifyProposalRejected();
    }

}
