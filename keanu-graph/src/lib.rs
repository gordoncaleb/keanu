//! Computation graphs for [Keanu](https://github.com/gordoncaleb/keanu): probabilistic and deterministic nodes,
//! value propagation and automatic differentiation.
//!
//! A [`Graph`] is an immutable arena of nodes, built with a [`GraphBuilder`] through typed handles
//! ([`DoubleVar`], [`IntVar`], [`BoolVar`]). Node ids are assigned in creation order and parents always precede
//! their children, so id order is a topological order. Values live in a separate [`State`], so one graph can be
//! shared (it is `Send + Sync`) by many independent states.
//!
//! ```
//! use keanu_distributions::KeanuRng;
//! use keanu_graph::GraphBuilder;
//! use keanu_tensor::DoubleTensor;
//!
//! let b = GraphBuilder::new();
//! let temperature = b.uniform(20.0, 30.0).label("temperature");
//! let reading = b.gaussian(temperature, 2.5);
//! b.observe(reading, DoubleTensor::scalar(25.0));
//! let doubled = temperature * 2.0;
//! let (temperature, doubled) = (temperature.id(), doubled.id());
//! let (graph, mut state) = b.build()?;
//!
//! let mut rng = KeanuRng::seed_from_u64(1);
//! state.lazy_eval(&graph, &[doubled], &mut rng)?;      // samples the temperature from its prior
//! let t = state.f64(temperature).unwrap().as_slice()[0];
//! assert_eq!(state.f64(doubled).unwrap().as_slice(), &[2.0 * t]);
//!
//! state.set_and_cascade(&graph, temperature, DoubleTensor::scalar(21.0))?;
//! assert_eq!(state.f64(doubled).unwrap().as_slice(), &[42.0]);
//! # Ok::<(), keanu_graph::GraphError>(())
//! ```

#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod autodiff;
mod builder;
mod distributions;
mod error;
pub mod gp;
mod graph;
mod impute;
mod intervene;
mod logprob;
mod marginalize;
mod model;
pub mod ode;
pub mod op;
pub mod parameterisation;
mod rewrite;
pub mod sequence;
pub mod spatial;
mod state;
pub mod sts;
pub mod transform;
mod value;
mod vars;

pub use autodiff::{
    Partials, forward_partials, is_differentiable, reverse_partials, reverse_partials_from,
};
pub use builder::GraphBuilder;
pub use error::{GraphError, Result};
pub use graph::{Graph, Node, NodeId};
pub use impute::{Imputed, ImputedNode, MaskedObservation, observe_masked};
pub use intervene::{Intervened, intervene};
pub use logprob::{
    LogProbGradient, NodeLogProbGradient, joint_log_prob, joint_log_prob_gradient, log_likelihood,
    log_likelihood_gradient, log_prior, log_prob, log_prob_gradient, node_log_prob,
    node_log_prob_at, node_log_prob_event_terms, node_log_prob_gradient, node_log_prob_terms,
    node_support,
};
pub use marginalize::{
    JointMarginalized, MAX_MARGINALIZED_VALUES, Marginalized, MarginalizedLatent, marginalize,
    marginalize_joint,
};
pub use model::{LambdaSection, Model, downstream_lambda_section, upstream_lambda_section};
pub use op::{CustomDistribution, CustomGradient, Support};
pub use parameterisation::{HessianMethod, Parameterisation, Space};
pub use spatial::Bym2;
pub use state::State;
pub use value::{DType, Value};
pub use vars::{BoolVar, DoubleVar, IntVar, IntoBoolVar, IntoDoubleVar, IntoIntVar, Var};

#[cfg(test)]
mod send_sync {
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn graph_and_state_are_send_and_sync() {
        assert_send_sync::<super::Graph>();
        assert_send_sync::<super::State>();
    }
}
