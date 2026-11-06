//! CmdStan CSV export, read back by a parser that follows ArviZ's `from_cmdstan` rules: lines starting with `#` are
//! comments (the configuration before the header, the adaptation results after it), the first other line is the
//! header, columns whose names end in `__` are sampler statistics, and a variable's elements are named by its name
//! and 1-based indices joined with dots, which place each column in the variable's array.

use keanu_algorithms::export::CmdStanCsv;
use keanu_algorithms::loglik::PointwiseLogLik;
use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::chains::run_chains;
use keanu_algorithms::mcmc::metropolis::MetropolisHastings;
use keanu_algorithms::mcmc::nuts::{Metric, Nuts, NutsStatistics, QuadraticPotential};
use keanu_algorithms::{AlgorithmError, ChainSamples};
use keanu_distributions::KeanuRng;
use keanu_graph::{GraphBuilder, Model, NodeId, State};
use keanu_tensor::DoubleTensor;
use std::collections::BTreeMap;

/// A parsed CmdStan CSV file.
#[derive(Debug)]
struct CmdStanFile {
    /// `key = value` comment lines before the header (indentation dropped).
    config: BTreeMap<String, String>,
    /// Comment lines after the header.
    adaptation: Vec<String>,
    header: Vec<String>,
    /// One row per draw.
    rows: Vec<Vec<f64>>,
}

/// A variable read from the columns: its shape (the largest index along each axis) and its values per draw in
/// row-major order.
#[derive(Debug)]
struct Variable {
    shape: Vec<usize>,
    draws: Vec<Vec<f64>>,
}

impl CmdStanFile {
    fn parse(text: &str) -> Self {
        let mut config = BTreeMap::new();
        let mut adaptation: Vec<String> = Vec::new();
        let mut header: Option<Vec<String>> = None;
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for line in text.lines() {
            if let Some(comment) = line.strip_prefix('#') {
                if header.is_none() {
                    if let Some((key, value)) = comment.trim().split_once(" = ") {
                        config.insert(key.to_owned(), value.to_owned());
                    }
                } else {
                    adaptation.push(comment.trim().to_owned());
                }
            } else if header.is_none() {
                header = Some(line.split(',').map(str::to_owned).collect());
            } else {
                rows.push(
                    line.split(',')
                        .map(|v| v.parse::<f64>().unwrap_or_else(|_| panic!("{v}")))
                        .collect(),
                );
            }
        }
        let header = header.expect("no header");
        assert!(rows.iter().all(|r| r.len() == header.len()));
        Self {
            config,
            adaptation,
            header,
            rows,
        }
    }

    fn sampler_columns(&self) -> Vec<&str> {
        self.header
            .iter()
            .filter(|c| c.ends_with("__"))
            .map(String::as_str)
            .collect()
    }

    fn column(&self, name: &str) -> Vec<f64> {
        let i = self.header.iter().position(|c| c == name).unwrap();
        self.rows.iter().map(|r| r[i]).collect()
    }

    /// The variables, as `from_cmdstan` reshapes them: each column `name.i.j` goes to element (i − 1, j − 1).
    fn variables(&self) -> BTreeMap<String, Variable> {
        let mut located: BTreeMap<String, Vec<(Vec<usize>, usize)>> = BTreeMap::new();
        for (column, name) in self.header.iter().enumerate() {
            if name.ends_with("__") {
                continue;
            }
            let mut parts = name.split('.');
            let base = parts.next().unwrap().to_owned();
            let index: Vec<usize> = parts.map(|p| p.parse::<usize>().unwrap() - 1).collect();
            located.entry(base).or_default().push((index, column));
        }
        located
            .into_iter()
            .map(|(name, columns)| {
                let rank = columns[0].0.len();
                let shape: Vec<usize> = (0..rank)
                    .map(|axis| columns.iter().map(|(i, _)| i[axis] + 1).max().unwrap())
                    .collect();
                let size: usize = shape.iter().product();
                assert_eq!(columns.len(), size, "{name} has missing elements");
                let draws = self
                    .rows
                    .iter()
                    .map(|row| {
                        let mut values = vec![f64::NAN; size];
                        for (index, column) in &columns {
                            let offset = index
                                .iter()
                                .zip(&shape)
                                .fold(0, |offset, (i, dim)| offset * dim + i);
                            values[offset] = row[*column];
                        }
                        values
                    })
                    .collect();
                (name, Variable { shape, draws })
            })
            .collect()
    }
}

/// mu ~ N(0, 2), theta ~ N(mu, 1) of shape [3], L ~ N(0, 1) of shape [2, 3], y ~ N(theta, 1) observed.
fn model() -> (Model, State, Vec<NodeId>) {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 2.0).label("mu");
    let theta = b.gaussian_with_shape(&[3], mu, 1.0).label("theta");
    let l = b.gaussian_with_shape(&[2, 3], 0.0, 1.0).label("L");
    let y = b.gaussian(theta, 1.0).label("y");
    b.observe(y, DoubleTensor::vector(vec![0.5, -0.3, 1.2]));
    let latents = vec![mu.id(), theta.id(), l.id()];
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    (model, state, latents)
}

const ADAPT: usize = 150;
const OPTIONS_INTERVAL: usize = 2;

fn nuts_run() -> (
    Model,
    State,
    Vec<NodeId>,
    Nuts,
    ChainSamples,
    Vec<NutsStatistics>,
    SamplingOptions,
) {
    let (model, state, latents) = model();
    let nuts = Nuts::default()
        .with_adapt_count(ADAPT)
        .with_save_statistics(true);
    let options = SamplingOptions::default()
        .with_drop(ADAPT)
        .with_down_sample_interval(OPTIONS_INTERVAL);
    let runs = run_chains(2, 5, &state, |state, rng| {
        nuts.sample_with_statistics(&model, state, &latents, ADAPT + 200, options, rng)
    })
    .unwrap();
    let (samples, statistics): (Vec<_>, Vec<_>) = runs.into_iter().unzip();
    let chains = ChainSamples::new(samples).unwrap();
    (model, state, latents, nuts, chains, statistics, options)
}

/// AE4: the draws, the sampler columns and the pointwise log likelihood read back as they were.
#[test]
fn nuts_chains_read_back_as_arviz_reads_them() {
    let (model, state, latents, nuts, chains, statistics, options) = nuts_run();
    let loglik = PointwiseLogLik::from_chains(&model, &state, &chains).unwrap();
    let csv = CmdStanCsv::new(model.graph())
        .with_model_name("hierarchy")
        .with_nuts(&nuts, &statistics, options)
        .with_log_lik(&loglik);
    for (chain, stats) in statistics.iter().enumerate() {
        let mut out = Vec::new();
        csv.write_chain(&chains, chain, &mut out).unwrap();
        let file = CmdStanFile::parse(std::str::from_utf8(&out).unwrap());

        assert_eq!(
            file.sampler_columns(),
            [
                "lp__",
                "accept_stat__",
                "stepsize__",
                "treedepth__",
                "n_leapfrog__",
                "divergent__",
                "energy__"
            ]
        );
        // Stan's column-major order for the matrix
        let l_columns: Vec<&str> = file
            .header
            .iter()
            .filter(|c| c.starts_with("L."))
            .map(String::as_str)
            .collect();
        assert_eq!(
            l_columns,
            ["L.1.1", "L.2.1", "L.1.2", "L.2.2", "L.1.3", "L.2.3"]
        );
        assert_eq!(file.config["model"], "hierarchy");
        assert_eq!(file.config["method"], "sample");
        // CmdStan's num_samples counts the post-warm-up iterations before thinning: the rows are
        // ⌈num_samples / thin⌉, as cmdstanpy checks
        assert_eq!(
            file.config["num_samples"],
            (100 * OPTIONS_INTERVAL).to_string()
        );
        let num_samples: usize = file.config["num_samples"].parse().unwrap();
        assert_eq!(num_samples.div_ceil(OPTIONS_INTERVAL), file.rows.len());
        assert_eq!(file.config["num_warmup"], ADAPT.to_string());
        assert_eq!(file.config["thin"], OPTIONS_INTERVAL.to_string());
        assert_eq!(file.config["engine"], "nuts");
        assert_eq!(file.config["delta"], "0.8");
        assert_eq!(file.config["max_depth"], "10");
        assert_eq!(file.config["id"], (chain + 1).to_string());
        assert_eq!(file.rows.len(), 100);

        // the draws, exactly
        let draws = &chains.chains()[chain];
        let variables = file.variables();
        for (name, &id) in ["mu", "theta", "L"].iter().zip(&latents) {
            let variable = &variables[*name];
            let samples = draws.f64(id).unwrap();
            let expected_shape = samples.as_slice()[0].shape().to_vec();
            assert_eq!(variable.shape, expected_shape, "{name}");
            for (read, sample) in variable.draws.iter().zip(samples.as_slice()) {
                assert_eq!(read.as_slice(), sample.as_slice(), "{name}");
            }
        }
        assert_eq!(file.column("lp__"), draws.log_probs());

        // the pointwise log likelihood, as log_lik[i]
        let log_lik = &variables["log_lik"];
        assert_eq!(log_lik.shape, [3]);
        assert_eq!(log_lik.draws.as_slice(), loglik.chain(chain).unwrap());

        // the sampler statistics of the recorded steps: drop + i · interval
        let steps: Vec<usize> = (0..100).map(|i| ADAPT + i * OPTIONS_INTERVAL).collect();
        let at = |values: &[f64]| steps.iter().map(|&s| values[s]).collect::<Vec<_>>();
        assert_eq!(file.column("stepsize__"), at(&stats.step_size));
        assert_eq!(file.column("energy__"), at(&stats.energy));
        assert_eq!(file.column("accept_stat__"), at(&stats.accept_stat));
        let depth: Vec<f64> = stats.tree_depth.iter().map(|&d| d as f64).collect();
        assert_eq!(file.column("treedepth__"), at(&depth));
        let leapfrog: Vec<f64> = stats.tree_size.iter().map(|&n| n as f64).collect();
        assert_eq!(file.column("n_leapfrog__"), at(&leapfrog));
        let flags = stats.divergent_flags();
        let divergent: Vec<f64> = steps
            .iter()
            .map(|&s| f64::from(u8::from(flags[s])))
            .collect();
        assert_eq!(file.column("divergent__"), divergent);
        assert!(
            file.column("accept_stat__")
                .iter()
                .all(|a| (0.0..=1.0).contains(a))
        );
        for (depth, leapfrog) in file
            .column("treedepth__")
            .iter()
            .zip(file.column("n_leapfrog__"))
        {
            assert!(leapfrog <= 2f64.powf(*depth) && leapfrog >= 1.0);
        }

        // the adaptation results: the final step size and the ten variances of the inverse mass matrix
        assert_eq!(file.adaptation[0], "Adaptation terminated");
        let step: f64 = file.adaptation[1]
            .strip_prefix("Step size = ")
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(step, *stats.step_size.last().unwrap());
        assert_eq!(
            file.adaptation[2],
            "Diagonal elements of inverse mass matrix:"
        );
        let metric: Vec<f64> = file.adaptation[3]
            .split(", ")
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(metric, stats.inverse_metric);
        assert_eq!(metric.len(), 10);
        assert!(metric.iter().all(|v| *v > 0.0 && *v != 1.0));
    }
}

/// A dense metric is written as CmdStan writes one: `metric = dense_e`, and the matrix one row per comment line.
#[test]
fn a_dense_metric_is_written_row_by_row() {
    let (model, state, latents) = model();
    let nuts = Nuts::default()
        .with_adapt_count(ADAPT)
        .with_save_statistics(true)
        .with_potential(QuadraticPotential::default().with_metric(Metric::Dense));
    let options = SamplingOptions::default().with_drop(ADAPT);
    let runs = run_chains(1, 6, &state, |state, rng| {
        nuts.sample_with_statistics(&model, state, &latents, ADAPT + 20, options, rng)
    })
    .unwrap();
    let (samples, statistics): (Vec<_>, Vec<_>) = runs.into_iter().unzip();
    let chains = ChainSamples::new(samples).unwrap();
    let mut out = Vec::new();
    CmdStanCsv::new(model.graph())
        .with_nuts(&nuts, &statistics, options)
        .write_chain(&chains, 0, &mut out)
        .unwrap();
    let file = CmdStanFile::parse(std::str::from_utf8(&out).unwrap());
    assert_eq!(file.config["metric"], "dense_e");
    assert_eq!(file.adaptation[0], "Adaptation terminated");
    assert_eq!(file.adaptation[2], "Elements of inverse mass matrix:");
    // ten latent elements: ten rows of ten
    let rows: Vec<Vec<f64>> = file.adaptation[3..]
        .iter()
        .map(|row| row.split(", ").map(|v| v.parse().unwrap()).collect())
        .collect();
    assert_eq!(rows.len(), 10);
    let matrix = statistics[0].dense_inverse_metric.as_ref().unwrap();
    assert_eq!(rows.concat(), matrix.as_slice());
    // the matrix was adapted, and is symmetric
    assert!(
        rows.iter()
            .enumerate()
            .any(|(i, row)| row[(i + 1) % 10] != 0.0)
    );
    for (i, row) in rows.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            assert_eq!(*value, rows[j][i]);
        }
    }
}

/// Metropolis–Hastings draws, with an integer and a boolean latent and an unlabelled node: only `lp__` is a
/// sampler column, integers and booleans are written as integers, and unlabelled nodes are named by their index.
#[test]
fn metropolis_draws_have_only_lp() {
    let b = GraphBuilder::new();
    let rate = b.gamma(1.0, 2.0);
    let count = b.poisson(rate).label("count");
    let flag = b.bernoulli(0.3).label("flag");
    let y = b.gaussian(rate, 1.0);
    b.observe(y, DoubleTensor::scalar(1.5));
    let (rate, count, flag) = (rate.id(), count.id(), flag.id());
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let model = Model::new(graph, &state).unwrap();
    let samples = MetropolisHastings::default()
        .sample(
            &model,
            &mut state,
            &[rate, count, flag],
            50,
            &mut KeanuRng::seed_from_u64(2),
        )
        .unwrap();
    let chains = ChainSamples::new(vec![samples]).unwrap();
    let mut out = Vec::new();
    CmdStanCsv::new(model.graph())
        .write_chain(&chains, 0, &mut out)
        .unwrap();
    let text = String::from_utf8(out).unwrap();
    let file = CmdStanFile::parse(&text);
    assert_eq!(file.sampler_columns(), ["lp__"]);
    let rate_name = format!("node_{}", rate.index());
    assert_eq!(file.header, ["lp__", rate_name.as_str(), "count", "flag"]);
    assert!(file.adaptation.is_empty());
    assert_eq!(file.config["num_warmup"], "0");
    let draws = &chains.chains()[0];
    let counts: Vec<f64> = draws
        .i64(count)
        .unwrap()
        .as_slice()
        .iter()
        .map(|t| t.as_slice()[0] as f64)
        .collect();
    assert_eq!(file.column("count"), counts);
    let flags: Vec<f64> = draws
        .bool(flag)
        .unwrap()
        .as_slice()
        .iter()
        .map(|t| f64::from(u8::from(t.as_slice()[0])))
        .collect();
    assert_eq!(file.column("flag"), flags);
    // integers are written without a decimal point
    let data_line = text.lines().filter(|l| !l.starts_with('#')).nth(1).unwrap();
    let count_field = data_line.split(',').nth(2).unwrap();
    assert!(count_field.parse::<i64>().is_ok(), "{count_field}");
}

#[test]
fn files_are_written_per_chain() {
    let (model, _, _, nuts, chains, statistics, options) = nuts_run();
    let directory = std::env::temp_dir().join(format!("keanu-export-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let paths = CmdStanCsv::new(model.graph())
        .with_nuts(&nuts, &statistics, options)
        .write_files(&chains, &directory, "output")
        .unwrap();
    assert_eq!(
        paths,
        [
            directory.join("output_1.csv"),
            directory.join("output_2.csv")
        ]
    );
    let second = CmdStanFile::parse(&std::fs::read_to_string(&paths[1]).unwrap());
    assert_eq!(second.config["id"], "2");
    assert_eq!(second.rows.len(), 100);
    std::fs::remove_dir_all(&directory).unwrap();
    assert!(matches!(
        CmdStanCsv::new(model.graph()).write_files(&chains, directory.join("missing"), "x"),
        Err(AlgorithmError::Io(_))
    ));
}

#[test]
fn invalid_exports_are_rejected() {
    let (model, state, latents, nuts, chains, statistics, options) = nuts_run();
    let invalid = |result: keanu_algorithms::Result<()>| {
        assert!(
            matches!(result, Err(AlgorithmError::InvalidArgument(_))),
            "{result:?}"
        );
    };
    let mut out = Vec::new();
    // no third chain
    invalid(CmdStanCsv::new(model.graph()).write_chain(&chains, 2, &mut out));
    // statistics of only one chain
    invalid(
        CmdStanCsv::new(model.graph())
            .with_nuts(&nuts, &statistics[..1], options)
            .write_chain(&chains, 1, &mut out),
    );
    // options that don't match the draws: dropping more steps than were taken runs past the statistics
    invalid(
        CmdStanCsv::new(model.graph())
            .with_nuts(
                &nuts,
                &statistics,
                SamplingOptions::default().with_drop(300),
            )
            .write_chain(&chains, 0, &mut out),
    );
    // statistics not saved
    let quiet = Nuts::default().with_adapt_count(50);
    let (samples, stats) = quiet
        .sample_with_statistics(
            &model,
            &mut state.clone(),
            &latents,
            60,
            SamplingOptions::default().with_drop(50),
            &mut KeanuRng::seed_from_u64(3),
        )
        .unwrap();
    let one = ChainSamples::new(vec![samples]).unwrap();
    invalid(
        CmdStanCsv::new(model.graph())
            .with_nuts(
                &quiet,
                std::slice::from_ref(&stats),
                SamplingOptions::default().with_drop(50),
            )
            .write_chain(&one, 0, &mut out),
    );
    // a log likelihood of other draws
    let loglik = PointwiseLogLik::from_chains(&model, &state, &one).unwrap();
    invalid(
        CmdStanCsv::new(model.graph())
            .with_log_lik(&loglik)
            .write_chain(&chains, 0, &mut out),
    );
    // two nodes with the same name
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0).label("x.1");
    let c = b.gaussian(0.0, 1.0).label("x_1");
    let (a, c) = (a.id(), c.id());
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let samples = MetropolisHastings::default()
        .sample(
            &model,
            &mut state,
            &[a, c],
            5,
            &mut KeanuRng::seed_from_u64(4),
        )
        .unwrap();
    let chains = ChainSamples::new(vec![samples]).unwrap();
    invalid(CmdStanCsv::new(model.graph()).write_chain(&chains, 0, &mut out));
}
