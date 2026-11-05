//! Simulated annealing: finding the MAP, its variable selector and schedule checks.

mod mcmc_common;

use keanu_algorithms::AlgorithmError;
use keanu_algorithms::mcmc::annealing::{AnnealingSchedule, SimulatedAnnealing};
use keanu_algorithms::mcmc::metropolis::{MetropolisHastings, VariableSelector};
use keanu_algorithms::mcmc::proposal::{GaussianProposal, Proposal, ProposalDistribution};
use keanu_distributions::KeanuRng;
use keanu_graph::{GraphBuilder, Model, NodeId, State};
use keanu_tensor::DoubleTensor;
use mcmc_common::*;
use std::sync::{Arc, Mutex};

#[test]
fn simulated_annealing_finds_the_map() {
    let (model, mut state, mu, m, _) = gaussian_model();
    let annealing =
        SimulatedAnnealing::new(MetropolisHastings::default().proposal(GaussianProposal::new(0.2)));
    let estimate = annealing
        .max_a_posteriori(&model, &mut state, 5000, &mut rng(17))
        .unwrap();
    let value = estimate
        .values
        .iter()
        .find(|(id, _)| *id == mu)
        .unwrap()
        .1
        .as_f64()
        .unwrap()
        .as_slice()[0];
    assert!((value - m).abs() < 0.05, "MAP {value} vs {m}");
    assert!(estimate.log_prob >= model.log_prob(&state).unwrap());
    assert!(AnnealingSchedule::exponential(10, 2.0, 0.01).temperature(10) - 0.01 < 1e-12);
}

/// A proposal that records how many nodes each proposal was for.
#[derive(Debug, Default)]
struct CountingProposal {
    sizes: Mutex<Vec<usize>>,
}

impl ProposalDistribution for CountingProposal {
    fn propose(
        &self,
        model: &Model,
        state: &State,
        nodes: &[NodeId],
        rng: &mut KeanuRng,
    ) -> keanu_algorithms::Result<Proposal> {
        self.sizes.lock().unwrap().push(nodes.len());
        GaussianProposal::new(0.1).propose(model, state, nodes, rng)
    }
    fn log_prob_from_given_to(
        &self,
        _: &Model,
        _: &State,
        _: &Proposal,
    ) -> keanu_algorithms::Result<f64> {
        Ok(0.0)
    }
    fn log_prob_to_given_from(
        &self,
        _: &Model,
        _: &State,
        _: &Proposal,
    ) -> keanu_algorithms::Result<f64> {
        Ok(0.0)
    }
}

/// Regression: Java's simulated annealing ignored the configured variable selector.
#[test]
fn simulated_annealing_honours_the_variable_selector() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    let y = b.gaussian(0.0, 1.0);
    b.set_value(x, DoubleTensor::scalar(0.5));
    b.set_value(y, DoubleTensor::scalar(0.5));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let proposal = Arc::new(CountingProposal::default());
    let mh = MetropolisHastings::default()
        .shared_proposal(proposal.clone())
        .selector(VariableSelector::Full);
    SimulatedAnnealing::new(mh)
        .max_a_posteriori(&model, &mut state, 20, &mut rng(18))
        .unwrap();
    assert!(proposal.sizes.lock().unwrap().iter().all(|&n| n == 2));
}

/// Regression: a schedule over zero iterations gave NaN temperatures, and non-positive temperatures weren't
/// rejected.
#[test]
fn simulated_annealing_rejects_invalid_schedules() {
    let (model, mut state, ..) = gaussian_model();
    let annealing = SimulatedAnnealing::default();
    for schedule in [
        AnnealingSchedule::exponential(0, 2.0, 0.01),
        AnnealingSchedule::exponential(10, 0.0, 0.01),
        AnnealingSchedule::exponential(10, 2.0, -1.0),
        AnnealingSchedule::exponential(10, f64::NAN, 0.01),
        AnnealingSchedule::exponential(10, 2.0, f64::INFINITY),
    ] {
        assert!(
            matches!(
                annealing.max_a_posteriori_with(&model, &mut state, 5, schedule, &mut rng(1)),
                Err(AlgorithmError::InvalidSetting(_))
            ),
            "{schedule:?}"
        );
    }
    // no iterations with the default schedule just returns the start
    assert!(
        annealing
            .max_a_posteriori(&model, &mut state, 0, &mut rng(1))
            .is_ok()
    );
}
