use keanu_algorithms::diagnostics::{
    self, ElementSummary, MIN_ESS, RHAT_THRESHOLD, ess_bulk, ess_tail, mcse_mean, rhat,
};
use keanu_algorithms::mcmc::chains::sample_chains;
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_algorithms::{AlgorithmError, ChainSamples, NetworkSamples};
use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, GraphBuilder, Model, NodeId, State};
use keanu_tensor::DoubleTensor;

fn ar1(rho: f64, draws: usize, rng: &mut KeanuRng) -> Vec<f64> {
    // start in the stationary distribution, so there is no burn-in
    let mut x = rng.next_gaussian() / (1.0 - rho * rho).sqrt();
    (0..draws)
        .map(|_| {
            x = rho * x + rng.next_gaussian();
            x
        })
        .collect()
}

fn iid(draws: usize, rng: &mut KeanuRng) -> Vec<f64> {
    ar1(0.0, draws, rng)
}

/// The classic split R-hat on the raw draws (no rank normalisation, no folding).
fn raw_split_rhat(chains: &[Vec<f64>]) -> f64 {
    let half = chains[0].len() / 2;
    let split: Vec<&[f64]> = chains
        .iter()
        .flat_map(|c| [&c[..half], &c[c.len() - half..]])
        .collect();
    let n = half as f64;
    let means: Vec<f64> = split.iter().map(|c| c.iter().sum::<f64>() / n).collect();
    let w = split
        .iter()
        .zip(&means)
        .map(|(c, m)| c.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0))
        .sum::<f64>()
        / split.len() as f64;
    let grand = means.iter().sum::<f64>() / means.len() as f64;
    let b = n * means.iter().map(|m| (m - grand).powi(2)).sum::<f64>() / (means.len() as f64 - 1.0);
    (((n - 1.0) / n * w + b / n) / w).sqrt()
}

#[test]
fn iid_chains_have_rhat_near_one_and_bulk_ess_near_the_draw_count() {
    let mut rng = KeanuRng::seed_from_u64(1);
    let chains: Vec<Vec<f64>> = (0..4).map(|_| iid(1000, &mut rng)).collect();
    let r = rhat(&chains).unwrap();
    assert!(r < RHAT_THRESHOLD, "{r}");
    // for independent draws the bulk ESS estimate is near the 4000 draws: τ̂ sums a few estimated
    // autocorrelations, each with a standard error of about 1/√4000, so it is within about ±0.1 of 1
    let ess = ess_bulk(&chains).unwrap();
    assert!((ess / 4000.0 - 1.0).abs() < 0.15, "{ess}");
    let tail = ess_tail(&chains).unwrap();
    assert!(tail > MIN_ESS, "{tail}");
    // the MCSE of the mean of independent N(0, 1) draws is about 1 / sqrt(4000)
    let mcse = mcse_mean(&chains).unwrap();
    assert!((mcse * 4000f64.sqrt() - 1.0).abs() < 0.1, "{mcse}");
}

#[test]
fn a_shifted_chain_gives_a_large_rhat() {
    let mut rng = KeanuRng::seed_from_u64(2);
    let mut chains: Vec<Vec<f64>> = (0..4).map(|_| iid(1000, &mut rng)).collect();
    chains[3].iter_mut().for_each(|x| *x += 3.0);
    let r = rhat(&chains).unwrap();
    assert!(r > 1.1, "{r}");
}

#[test]
fn folded_rhat_flags_a_chain_with_three_times_the_scale() {
    let mut rng = KeanuRng::seed_from_u64(3);
    let mut chains: Vec<Vec<f64>> = (0..4).map(|_| iid(1000, &mut rng)).collect();
    chains[3].iter_mut().for_each(|x| *x *= 3.0);
    // the means agree, so R-hat on the raw draws does not notice
    let raw = raw_split_rhat(&chains);
    assert!(raw < RHAT_THRESHOLD, "{raw}");
    // the folded (tail) R-hat does
    let r = rhat(&chains).unwrap();
    assert!(r > 1.1, "{r}");
}

#[test]
fn bulk_ess_of_ar1_matches_the_analytic_value() {
    let rho = 0.9;
    let mut rng = KeanuRng::seed_from_u64(4);
    let chains: Vec<Vec<f64>> = (0..4).map(|_| ar1(rho, 20_000, &mut rng)).collect();
    let n = 80_000.0;
    let expected = n * (1.0 - rho) / (1.0 + rho);
    let ess = ess_bulk(&chains).unwrap();
    assert!((ess / expected - 1.0).abs() < 0.15, "{ess} vs {expected}");
}

/// `tests/data/arviz_diagnostics.json` was generated offline by `tests/data/arviz_diagnostics.py` with arviz-stats
/// 1.3.3 (`_rhat_rank`, `_ess_bulk`, `_ess_tail` with probabilities (0.05, 0.95), `_mcse_mean`) and NumPy's linear
/// quantiles. A null R-hat is ArviZ's NaN (fewer than two chains).
#[test]
fn diagnostics_match_arviz() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/arviz_diagnostics.json");
    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 6);
    let close = |name: &str, what: &str, actual: f64, expected: f64| {
        assert!(
            (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0),
            "{name} {what}: {actual} vs ArviZ {expected}"
        );
    };
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let chains: Vec<Vec<f64>> = serde_json::from_value(case["chains"].clone()).unwrap();
        let value = |key: &str| case[key].as_f64().unwrap();
        let s = diagnostics::summarize(&chains).unwrap();
        match case["r_hat"].as_f64() {
            Some(r) => close(name, "r_hat", s.r_hat, r),
            None => assert!(s.r_hat.is_nan(), "{name}"),
        }
        close(name, "ess_bulk", s.ess_bulk, value("ess_bulk"));
        close(name, "ess_tail", s.ess_tail, value("ess_tail"));
        close(name, "mcse_mean", s.mcse_mean, value("mcse_mean"));
        close(name, "mean", s.mean, value("mean"));
        close(name, "sd", s.sd, value("sd"));
        close(name, "q5", s.q5, value("q5"));
        close(name, "q50", s.q50, value("q50"));
        close(name, "q95", s.q95, value("q95"));
    }
}

#[test]
fn constant_chains_have_nan_diagnostics() {
    let chains = vec![vec![2.5; 100]; 4];
    let s = diagnostics::summarize(&chains).unwrap();
    assert!(s.r_hat.is_nan(), "{s:?}");
    assert!(s.ess_bulk.is_nan(), "{s:?}");
    assert!(s.ess_tail.is_nan(), "{s:?}");
    assert!(s.mcse_mean.is_nan(), "{s:?}");
    assert_eq!(s.mean, 2.5);
    assert_eq!(s.sd, 0.0);
    assert_eq!((s.q5, s.q50, s.q95), (2.5, 2.5, 2.5));

    // each chain constant but at a different value: the chains disagree completely
    let stuck: Vec<Vec<f64>> = (0..4).map(|c| vec![c as f64; 100]).collect();
    assert_eq!(rhat(&stuck).unwrap(), f64::INFINITY);
    assert!(!diagnostics::summarize(&stuck).unwrap().is_converged());
}

#[test]
fn frozen_chains_are_not_converged() {
    // four samplers stuck at their (shared) initial value
    let frozen = vec![vec![0.3; 1000]; 4];
    let s = diagnostics::summarize(&frozen).unwrap();
    assert!(!s.is_converged(), "{s:?}");
    // more stuck draws must not look like more information
    let longer = vec![vec![0.3; 10_000]; 4];
    assert!(ess_bulk(&longer).unwrap().is_nan());

    // through ChainSamples::summary as well
    let (graph, mut state, x, n) = vector_graph();
    let chain = network_samples(&graph, &mut state, x, n, &[[1.0, -2.0]; 500]);
    let samples = ChainSamples::new(vec![chain; 4]).unwrap();
    let summary = samples.summary().unwrap();
    assert!(!summary.is_converged(), "{summary}");
}

#[test]
fn disagreeing_chains_of_tiny_values_have_a_large_rhat() {
    let mut rng = KeanuRng::seed_from_u64(6);
    let mut chains: Vec<Vec<f64>> = (0..4).map(|_| iid(1000, &mut rng)).collect();
    chains[3].iter_mut().for_each(|x| *x += 3.0);
    // the same chains at a scale of 1e-18, so every draw is within 1e-15 of every other: R-hat is scale
    // invariant (scaling keeps the ranks), so it must still flag them
    let tiny: Vec<Vec<f64>> = chains
        .iter()
        .map(|c| c.iter().map(|x| x * 1e-18).collect())
        .collect();
    let r = rhat(&tiny).unwrap();
    assert!(r > RHAT_THRESHOLD, "{r}");
    assert!((r - rhat(&chains).unwrap()).abs() < 1e-9, "{r}");
    let s = diagnostics::summarize(&tiny).unwrap();
    assert!(!s.is_converged(), "{s:?}");
    assert!(s.ess_bulk < 4000.0, "{s:?}");
}

#[test]
fn too_few_draws_or_chains_and_non_finite_draws_give_nan() {
    let short = vec![vec![1.0, 2.0, 3.0]; 4];
    let s = diagnostics::summarize(&short).unwrap();
    assert!(s.r_hat.is_nan() && s.ess_bulk.is_nan() && s.ess_tail.is_nan() && s.mcse_mean.is_nan());
    assert_eq!(s.mean, 2.0);

    let mut rng = KeanuRng::seed_from_u64(5);
    let one = vec![iid(100, &mut rng)];
    assert!(rhat(&one).unwrap().is_nan());
    assert!(ess_bulk(&one).unwrap() > 0.0);

    let mut with_nan: Vec<Vec<f64>> = (0..2).map(|_| iid(100, &mut rng)).collect();
    with_nan[1][7] = f64::NAN;
    let s = diagnostics::summarize(&with_nan).unwrap();
    assert!(s.r_hat.is_nan() && s.ess_bulk.is_nan() && s.ess_tail.is_nan() && s.mcse_mean.is_nan());
    with_nan[1][7] = f64::INFINITY;
    assert!(rhat(&with_nan).unwrap().is_nan());
}

#[test]
fn ragged_or_missing_chains_are_errors() {
    let ragged = vec![vec![0.0; 10], vec![0.0; 9]];
    assert!(matches!(
        rhat(&ragged),
        Err(AlgorithmError::InvalidArgument(_))
    ));
    assert!(matches!(
        diagnostics::summarize(&ragged),
        Err(AlgorithmError::InvalidArgument(_))
    ));
    let none: Vec<Vec<f64>> = Vec::new();
    assert!(matches!(
        ess_bulk(&none),
        Err(AlgorithmError::InvalidArgument(_))
    ));
}

/// A graph with a vector node and an int node, and `count` samples of chain `chain` with known values.
fn network_samples(
    graph: &Graph,
    state: &mut State,
    x: NodeId,
    n: NodeId,
    values: &[[f64; 2]],
) -> NetworkSamples {
    let mut samples = NetworkSamples::new(graph, &[x, n]).unwrap();
    for v in values {
        state
            .set(graph, x, DoubleTensor::vector(v.to_vec()))
            .unwrap();
        state
            .set(graph, n, keanu_tensor::IntTensor::scalar(1))
            .unwrap();
        samples.record(state, 0.0).unwrap();
    }
    samples
}

fn vector_graph() -> (Graph, State, NodeId, NodeId) {
    let b = GraphBuilder::new();
    let x = b.gaussian_with_shape(&[2], 0.0, 1.0).id();
    let n = b.poisson(1.0).id();
    let (graph, state) = b.build().unwrap();
    (graph, state, x, n)
}

#[test]
fn chain_samples_summarise_every_element_of_every_f64_node() {
    let (graph, mut state, x, n) = vector_graph();
    let mut rng = KeanuRng::seed_from_u64(6);
    let chains: Vec<NetworkSamples> = (0..4)
        .map(|_| {
            let values: Vec<[f64; 2]> = (0..500)
                .map(|_| [rng.next_gaussian(), 10.0 + rng.next_gaussian()])
                .collect();
            network_samples(&graph, &mut state, x, n, &values)
        })
        .collect();
    let samples = ChainSamples::new(chains).unwrap();
    assert_eq!((samples.num_chains(), samples.draws()), (4, 500));
    assert_eq!(samples.nodes(), &[x, n]);

    let second = samples.element(x, &[1]).unwrap();
    assert_eq!(second.len(), 4);
    assert_eq!(
        second[2],
        samples.chains()[2].f64(x).unwrap().element(&[1]).unwrap()
    );
    let direct = diagnostics::summarize(&second).unwrap();
    assert_eq!(samples.summarize(x, &[1]).unwrap(), direct);
    assert_eq!(samples.rhat(x, &[1]).unwrap(), direct.r_hat);
    assert_eq!(samples.ess_bulk(x, &[1]).unwrap(), direct.ess_bulk);
    assert_eq!(samples.ess_tail(x, &[1]).unwrap(), direct.ess_tail);
    assert_eq!(samples.mcse_mean(x, &[1]).unwrap(), direct.mcse_mean);

    // the int node is skipped; the vector node has a row per element
    let summary = samples.summary().unwrap();
    let rows = summary.rows();
    assert_eq!(rows.len(), 2);
    assert_eq!((rows[0].node, rows[0].index.as_slice()), (x, &[0][..]));
    assert_eq!((rows[1].node, rows[1].index.as_slice()), (x, &[1][..]));
    assert_eq!(rows[1].stats, direct);
    assert_eq!(summary.get(x, &[1]), Some(&direct));
    assert_eq!(summary.get(n, &[]), None);
    assert!((rows[1].stats.mean - 10.0).abs() < 0.1);
    assert!(summary.is_converged());
    let table = summary.to_string();
    for column in [
        "node",
        "mean",
        "sd",
        "5%",
        "50%",
        "95%",
        "mcse_mean",
        "ess_bulk",
        "ess_tail",
        "r_hat",
    ] {
        assert!(table.contains(column), "{table}");
    }
    assert_eq!(table.lines().count(), 3, "{table}");

    // an int node's element is its values as f64; an unsampled node or a bad index is an error
    let ints = samples.element(n, &[]).unwrap();
    assert_eq!(ints.len(), samples.num_chains());
    assert!(ints.iter().flatten().all(|&x| x == 1.0));
    assert!(samples.element(NodeId::from_index(99), &[]).is_err());
    assert!(samples.element(x, &[5]).is_err());
}

#[test]
fn chain_samples_reject_mismatched_chains() {
    let (graph, mut state, x, n) = vector_graph();
    let a = network_samples(&graph, &mut state, x, n, &[[0.0, 0.0]; 10]);
    let shorter = network_samples(&graph, &mut state, x, n, &[[0.0, 0.0]; 9]);
    assert!(matches!(
        ChainSamples::new(vec![a.clone(), shorter]),
        Err(AlgorithmError::InvalidArgument(_))
    ));
    let mut other_nodes = NetworkSamples::new(&graph, &[x]).unwrap();
    for _ in 0..10 {
        other_nodes.record(&state, 0.0).unwrap();
    }
    assert!(matches!(
        ChainSamples::new(vec![a.clone(), other_nodes]),
        Err(AlgorithmError::InvalidArgument(_))
    ));
    assert!(matches!(
        ChainSamples::new(Vec::new()),
        Err(AlgorithmError::InvalidArgument(_))
    ));
    let ok = ChainSamples::new(vec![a.clone(), a.clone()]).unwrap();
    assert_eq!(ok.into_chains(), vec![a.clone(), a]);
}

#[test]
fn nuts_chains_converge_on_a_simple_posterior() {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0);
    let x = b.gaussian(mu, 1.0);
    b.observe(x, DoubleTensor::scalar(1.0));
    let mu = mu.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let nuts = Nuts::default().with_adapt_count(200);
    let samples = sample_chains(4, 11, &state, |state, rng| {
        nuts.sample(&model, state, &[mu], 500, rng)
    })
    .unwrap();
    let summary = samples.summary().unwrap();
    let row: &ElementSummary = summary.get(mu, &[]).unwrap();
    assert!(row.r_hat < RHAT_THRESHOLD, "{summary}");
    assert!(
        row.ess_bulk > MIN_ESS && row.ess_tail > MIN_ESS,
        "{summary}"
    );
    assert!(row.is_converged());
    // the posterior is N(1/2, 1/2)
    assert!(
        (row.mean - 0.5).abs() < 5.0 * row.mcse_mean.max(0.01),
        "{summary}"
    );
    assert!(
        sample_chains(0, 11, &state, |state, rng| nuts.sample(
            &model,
            state,
            &[mu],
            10,
            rng
        ))
        .is_err()
    );
}
