//! Export of draws to the CmdStan CSV format, which ArviZ (`from_cmdstan`), R's `posterior` (through `cmdstanr`
//! or `rstan::read_stan_csv`) and other Stan tools read.
//!
//! Each chain is one file (or [`Write`]r), laid out as CmdStan writes its output (see the CmdStan guide, "Stan CSV
//! file format"):
//!
//! 1. comment lines, starting with `#`, describing the run in CmdStan's `key = value` form: the model name, the
//!    method (`sample`), the number of post-warm-up iterations before thinning (`num_samples`, the draws times
//!    the thinning interval, as CmdStan counts them), warm-up steps and the thinning interval, and for NUTS the target
//!    acceptance probability, maximum tree depth and the metric (`diag_e` or `dense_e`), and the chain's `id`
//!    (1-based);
//! 2. a header line naming the columns: the sampler's columns, whose names end in `__`, then each sampled node's
//!    elements, then the pointwise log likelihood if given;
//! 3. for NUTS, the adaptation results as comments, `# Adaptation terminated`, `# Step size = …` and
//!    `# Diagonal elements of inverse mass matrix:` followed by the values, or for a dense metric
//!    ([`Metric::Dense`]) `# Elements of inverse mass matrix:` followed by one comment line per row of the matrix;
//! 4. one line per draw.
//!
//! Column names follow Stan's conventions: a scalar node is its name; an element of a tensor is the name followed
//! by its 1-based indices joined with dots (`theta.1`, `L.2.1`), and the elements are written in column-major order
//! (the first index varying fastest), as Stan writes matrices. A node's name is its label, or `node_` and its index
//! for an unlabelled node; characters other than letters, digits and `_` become `_`. The pointwise log likelihood
//! ([`PointwiseLogLik`]) is written as `log_lik.1`, `log_lik.2`, … in its points' order, the name ArviZ and `loo`
//! look for.
//!
//! The sampler columns are `lp__` (the recorded log probability) for every sampler, and for NUTS with
//! [`Nuts::save_statistics`] also `accept_stat__`, `stepsize__`, `treedepth__`, `n_leapfrog__`, `divergent__` and
//! `energy__` from [`NutsStatistics`]. Keanu's `lp__` is the model's log density at the constrained values, without
//! the Jacobian of the transforms that Stan's includes; and the inverse mass matrix is in the unconstrained
//! coordinates NUTS moved in, in the order of the latents given to it.
//!
//! Numbers are written so that they read back exactly (the shortest representation that round-trips), with
//! `nan`, `inf` and `-inf` as Stan writes them, integers as integers and booleans as 0 and 1.
//!
//! ```
//! use keanu_algorithms::export::CmdStanCsv;
//! use keanu_algorithms::mcmc::SamplingOptions;
//! use keanu_algorithms::mcmc::chains::run_chains;
//! use keanu_algorithms::mcmc::nuts::Nuts;
//! use keanu_algorithms::ChainSamples;
//! use keanu_graph::{GraphBuilder, Model};
//! use keanu_tensor::DoubleTensor;
//!
//! let b = GraphBuilder::new();
//! let mu = b.gaussian(0.0, 5.0).label("mu");
//! let y = b.gaussian_with_shape(&[3], mu, 1.0).label("y");
//! b.observe(y, DoubleTensor::vector(vec![0.4, 1.3, 0.9]));
//! let mu = mu.id();
//! let (graph, state) = b.build()?;
//! let model = Model::new(graph, &state)?;
//! let nuts = Nuts::default().with_adapt_count(100).with_save_statistics(true);
//! let options = SamplingOptions::default().with_drop(100);
//! let runs = run_chains(2, 1, &state, |state, rng| {
//!     nuts.sample_with_statistics(&model, state, &[mu], 300, options, rng)
//! })?;
//! let (samples, statistics): (Vec<_>, Vec<_>) = runs.into_iter().unzip();
//! let chains = ChainSamples::new(samples)?;
//!
//! let csv = CmdStanCsv::new(model.graph()).with_model_name("normal_mean").with_nuts(&nuts, &statistics, options);
//! let mut out = Vec::new();
//! csv.write_chain(&chains, 0, &mut out)?;
//! let text = String::from_utf8(out)?;
//! let header = text.lines().find(|l| !l.starts_with('#')).unwrap();
//! assert_eq!(header, "lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,divergent__,energy__,mu");
//! assert_eq!(text.lines().filter(|l| !l.starts_with('#')).count(), 201);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{AlgorithmError, Result};
use crate::loglik::PointwiseLogLik;
use crate::mcmc::SamplingOptions;
use crate::mcmc::nuts::{Metric, Nuts, NutsStatistics};
use crate::samples::{ChainSamples, NetworkSample, NetworkSamples};
use crate::support::node_name;
use keanu_graph::{Graph, GraphError, NodeId, Value};
use std::collections::HashSet;
use std::fmt::Display;
use std::io::Write;
use std::path::{Path, PathBuf};

/// The sampler columns of NUTS, after `lp__`.
const NUTS_COLUMNS: [&str; 6] = [
    "accept_stat__",
    "stepsize__",
    "treedepth__",
    "n_leapfrog__",
    "divergent__",
    "energy__",
];

/// Streams a long-form CSV trace with one row per tensor element of each draw.
///
/// The columns are `chain,draw,node,label,index,dtype,value`. Chain, draw, node and tensor indices are
/// zero-based. The index is a bracketed list in row-major order, or `[]` for a scalar. Labels are preserved
/// verbatim with CSV escaping. Unlabeled nodes use `#` followed by the node ID. Values retain their type
/// (`f64`, `i64` or `bool`), including integers above the precision of f64.
///
/// Returns an error if a sampled node is missing from the graph or a draw has the wrong type or shape.
/// An error can leave a partial trace in the writer. Pass a buffered writer when exporting to a file.
pub fn write_labeled_trace(
    graph: &Graph,
    samples: &ChainSamples,
    out: &mut impl Write,
) -> Result<()> {
    let nodes = samples
        .nodes()
        .iter()
        .map(|&id| {
            let node = graph.node(id).ok_or(GraphError::UnknownNode(id))?;
            let label = node_name(graph, id).replace('"', "\"\"");
            Ok((id, node, label))
        })
        .collect::<Result<Vec<_>>>()?;
    let io = |e: std::io::Error| AlgorithmError::Io(e.to_string());
    writeln!(out, "chain,draw,node,label,index,dtype,value").map_err(io)?;
    for (chain, draws) in samples.chains().iter().enumerate() {
        for draw in draws.iter() {
            for (id, node, label) in &nodes {
                let value = draw.get(*id).ok_or(GraphError::MissingValue(*id))?;
                if value.shape() != node.shape() || value.dtype() != node.dtype() {
                    return Err(AlgorithmError::InvalidArgument(format!(
                        "chain {chain} draw {} disagrees with the graph's type or shape for node {}",
                        draw.index(),
                        id.index(),
                    )));
                }
                let prefix = format!("{chain},{},{},\"{label}\"", draw.index(), id.index());
                let shape = value.shape();
                match &value {
                    Value::F64(t) => trace_elements(
                        out,
                        &prefix,
                        shape,
                        "f64",
                        t.as_slice().iter().map(|&x| number(x)),
                    ),
                    Value::I64(t) => {
                        trace_elements(out, &prefix, shape, "i64", t.as_slice().iter())
                    }
                    Value::Bool(t) => {
                        trace_elements(out, &prefix, shape, "bool", t.as_slice().iter())
                    }
                    _ => {
                        return Err(AlgorithmError::InvalidArgument(
                            "unsupported trace value type".into(),
                        ));
                    }
                }
                .map_err(io)?;
            }
        }
    }
    out.flush().map_err(io)
}

fn trace_elements(
    out: &mut impl Write,
    prefix: &str,
    shape: &[usize],
    dtype: &str,
    values: impl Iterator<Item = impl Display>,
) -> std::io::Result<()> {
    let mut index = vec![0; shape.len()];
    for value in values {
        writeln!(out, "{prefix},\"{index:?}\",{dtype},{value}")?;
        for (i, &size) in index.iter_mut().zip(shape).rev() {
            *i += 1;
            if *i < size {
                break;
            }
            *i = 0;
        }
    }
    Ok(())
}

/// A CmdStan CSV writer for the draws of one model; see the [module documentation](self).
#[derive(Debug, Clone)]
pub struct CmdStanCsv<'a> {
    graph: &'a Graph,
    model_name: String,
    log_lik: Option<&'a PointwiseLogLik>,
    nuts: Option<NutsRun<'a>>,
}

/// The NUTS settings and per-chain statistics of a run.
#[derive(Debug, Clone, Copy)]
struct NutsRun<'a> {
    settings: &'a Nuts,
    statistics: &'a [NutsStatistics],
    options: SamplingOptions,
}

impl<'a> CmdStanCsv<'a> {
    /// A writer for draws of nodes of `graph`, with only the `lp__` sampler column and model name `keanu_model`.
    pub fn new(graph: &'a Graph) -> Self {
        Self {
            graph,
            model_name: "keanu_model".into(),
            log_lik: None,
            nuts: None,
        }
    }

    /// Sets the model name written in the `# model = …` comment.
    #[must_use]
    pub fn with_model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = name.into();
        self
    }

    /// Writes the pointwise log likelihood of the same draws as `log_lik.1`, `log_lik.2`, … columns.
    #[must_use]
    pub fn with_log_lik(mut self, log_lik: &'a PointwiseLogLik) -> Self {
        self.log_lik = Some(log_lik);
        self
    }

    /// Writes NUTS's sampler columns and settings: `settings` is the configuration the chains ran with (it must
    /// have [`Nuts::save_statistics`]), `statistics` each chain's [`NutsStatistics`] in chain order, and `options`
    /// the dropping and down-sampling the draws were recorded with, which say which step each draw is.
    #[must_use]
    pub fn with_nuts(
        mut self,
        settings: &'a Nuts,
        statistics: &'a [NutsStatistics],
        options: SamplingOptions,
    ) -> Self {
        self.nuts = Some(NutsRun {
            settings,
            statistics,
            options,
        });
        self
    }

    /// Writes chain `chain` (0-based) of `samples` to `out`.
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidArgument`] if there is no such chain, two sampled nodes would get the same name,
    /// the pointwise log likelihood has a different number of chains or draws, or the NUTS statistics are missing
    /// (not saved, or fewer chains) or too short for the draws; [`AlgorithmError::Io`] if writing fails.
    pub fn write_chain(
        &self,
        samples: &ChainSamples,
        chain: usize,
        out: &mut impl Write,
    ) -> Result<()> {
        let draws = samples.chains().get(chain).ok_or_else(|| {
            AlgorithmError::InvalidArgument(format!(
                "there is no chain {chain} of {}",
                samples.num_chains()
            ))
        })?;
        let text = self.render(samples, draws, chain)?;
        out.write_all(text.as_bytes())
            .and_then(|()| out.flush())
            .map_err(|e| AlgorithmError::Io(e.to_string()))
    }

    /// Writes every chain of `samples` to its own file in `directory`, named `{prefix}_1.csv`, `{prefix}_2.csv`, …
    /// as CmdStan names them, and returns their paths.
    ///
    /// # Errors
    ///
    /// As [`write_chain`](Self::write_chain), and [`AlgorithmError::Io`] if a file can't be created.
    pub fn write_files(
        &self,
        samples: &ChainSamples,
        directory: impl AsRef<Path>,
        prefix: &str,
    ) -> Result<Vec<PathBuf>> {
        (0..samples.num_chains())
            .map(|chain| {
                let path = directory
                    .as_ref()
                    .join(format!("{prefix}_{}.csv", chain + 1));
                let file = std::fs::File::create(&path)
                    .map_err(|e| AlgorithmError::Io(format!("{}: {e}", path.display())))?;
                let mut out = std::io::BufWriter::new(file);
                self.write_chain(samples, chain, &mut out)?;
                Ok(path)
            })
            .collect()
    }

    /// The whole text of one chain's file.
    fn render(
        &self,
        samples: &ChainSamples,
        draws: &NetworkSamples,
        chain: usize,
    ) -> Result<String> {
        let columns = self.columns(draws)?;
        let nuts = self.nuts_columns(chain, draws.len())?;
        let log_lik = match self.log_lik {
            Some(log_lik) => {
                if log_lik.num_chains() != samples.num_chains() || log_lik.draws() != draws.len() {
                    return Err(AlgorithmError::InvalidArgument(format!(
                        "the pointwise log likelihood has {} chains of {} draws, but the samples {} of {}",
                        log_lik.num_chains(),
                        log_lik.draws(),
                        samples.num_chains(),
                        draws.len()
                    )));
                }
                log_lik.chain(chain)
            }
            None => None,
        };

        let mut text = String::new();
        self.write_configuration(&mut text, draws.len(), chain);
        let mut header = vec!["lp__".to_owned()];
        if nuts.is_some() {
            header.extend(NUTS_COLUMNS.iter().map(|c| (*c).to_owned()));
        }
        for column in &columns {
            header.extend(column.names.iter().cloned());
        }
        if let Some(log_lik) = self.log_lik {
            header.extend((1..=log_lik.len()).map(|i| format!("log_lik.{i}")));
        }
        text.push_str(&header.join(","));
        text.push('\n');
        if let Some(run) = &self.nuts {
            let statistics = &run.statistics[chain];
            text.push_str("# Adaptation terminated\n");
            if let Some(step) = statistics.step_size.last() {
                text.push_str(&format!("# Step size = {}\n", number(*step)));
            }
            match &statistics.dense_inverse_metric {
                // as CmdStan writes a dense metric: one comment line per row
                Some(matrix) => {
                    text.push_str("# Elements of inverse mass matrix:\n");
                    let n = statistics.inverse_metric.len();
                    for row in matrix.as_slice().chunks(n.max(1)) {
                        let row: Vec<String> = row.iter().map(|v| number(*v)).collect();
                        text.push_str("# ");
                        text.push_str(&row.join(", "));
                        text.push('\n');
                    }
                }
                None => {
                    text.push_str("# Diagonal elements of inverse mass matrix:\n# ");
                    let metric: Vec<String> = statistics
                        .inverse_metric
                        .iter()
                        .map(|v| number(*v))
                        .collect();
                    text.push_str(&metric.join(", "));
                    text.push('\n');
                }
            }
        }

        let mut row: Vec<String> = Vec::with_capacity(header.len());
        for (index, sample) in draws.iter().enumerate() {
            row.clear();
            row.push(number(sample.log_prob()));
            if let Some(nuts) = &nuts {
                nuts.push_row(index, &mut row);
            }
            for column in &columns {
                column.push_values(&sample, &mut row)?;
            }
            if let Some(values) = log_lik {
                row.extend(values[index].iter().map(|v| number(*v)));
            }
            text.push_str(&row.join(","));
            text.push('\n');
        }
        Ok(text)
    }

    /// The `#` comment lines describing the run, in CmdStan's layout.
    fn write_configuration(&self, text: &mut String, draws: usize, chain: usize) {
        let mut line = |indent: usize, entry: String| {
            text.push('#');
            text.push_str(&" ".repeat(1 + 2 * indent));
            text.push_str(&entry);
            text.push('\n');
        };
        line(0, format!("model = {}", self.model_name));
        line(0, "method = sample".into());
        line(1, "sample".into());
        let (warmup, thin) = self.nuts.as_ref().map_or((0, 1), |run| {
            (run.options.drop, run.options.down_sample_interval)
        });
        // CmdStan's num_samples counts the iterations after warm-up before thinning, and readers such as cmdstanpy
        // expect ⌈num_samples / thin⌉ rows; every row stands for `thin` iterations
        line(2, format!("num_samples = {}", draws * thin));
        line(2, format!("num_warmup = {warmup}"));
        line(2, "save_warmup = 0".into());
        line(2, format!("thin = {thin}"));
        match &self.nuts {
            Some(run) => {
                line(2, "adapt".into());
                line(
                    3,
                    format!("engaged = {}", u8::from(run.settings.adapt_count > 0)),
                );
                line(
                    3,
                    format!("delta = {}", number(run.settings.target_acceptance_prob)),
                );
                line(2, "algorithm = hmc".into());
                line(3, "hmc".into());
                line(4, "engine = nuts".into());
                line(5, "nuts".into());
                line(6, format!("max_depth = {}", run.settings.max_tree_height));
                let metric = match run.settings.potential.metric {
                    Metric::Dense => "dense_e",
                    _ => "diag_e",
                };
                line(4, format!("metric = {metric}"));
            }
            None => line(2, "algorithm = keanu".into()),
        }
        line(0, format!("id = {}", chain + 1));
        line(0, "generator = keanu".into());
    }

    /// The columns of each sampled node.
    fn columns(&self, draws: &NetworkSamples) -> Result<Vec<NodeColumns>> {
        let first = draws
            .sample(0)
            .ok_or_else(|| AlgorithmError::InvalidArgument("there are no draws to write".into()))?;
        let mut seen = HashSet::new();
        draws
            .nodes()
            .iter()
            .map(|&id| {
                let name = stan_name(self.graph, id);
                if !seen.insert(name.clone()) || name == "log_lik" && self.log_lik.is_some() {
                    return Err(AlgorithmError::InvalidArgument(format!(
                        "two columns would be called {name}; give the nodes distinct labels"
                    )));
                }
                let shape = first
                    .get(id)
                    .map(|value| value.shape().to_vec())
                    .ok_or_else(|| {
                        AlgorithmError::InvalidArgument(format!("the draws lack {name}"))
                    })?;
                Ok(NodeColumns::new(id, name, &shape))
            })
            .collect()
    }

    /// The NUTS columns of one chain's draws, if writing them.
    fn nuts_columns(&self, chain: usize, draws: usize) -> Result<Option<NutsColumns<'a>>> {
        let Some(run) = self.nuts else {
            return Ok(None);
        };
        let statistics = run.statistics.get(chain).ok_or_else(|| {
            AlgorithmError::InvalidArgument(format!(
                "there are NUTS statistics for {} chains, not chain {chain}",
                run.statistics.len()
            ))
        })?;
        let interval = run.options.down_sample_interval.max(1);
        let steps: Vec<usize> = (0..draws)
            .map(|i| run.options.drop + i * interval)
            .collect();
        let needed = steps.last().map_or(0, |s| s + 1);
        let lengths = [
            statistics.step_size.len(),
            statistics.accept_stat.len(),
            statistics.tree_depth.len(),
            statistics.tree_size.len(),
            statistics.energy.len(),
        ];
        if lengths.iter().any(|&n| n < needed) {
            return Err(AlgorithmError::InvalidArgument(format!(
                "chain {chain}'s NUTS statistics cover fewer than the {needed} steps its draws need; sample with \
                 `Nuts::with_save_statistics(true)` and pass the sampling options the draws were recorded with"
            )));
        }
        Ok(Some(NutsColumns {
            statistics,
            divergent: statistics.divergent_flags(),
            steps,
        }))
    }
}

/// The NUTS statistics of one chain and the step of each draw.
struct NutsColumns<'a> {
    statistics: &'a NutsStatistics,
    divergent: Vec<bool>,
    steps: Vec<usize>,
}

impl NutsColumns<'_> {
    fn push_row(&self, draw: usize, row: &mut Vec<String>) {
        let s = self.steps[draw];
        let st = self.statistics;
        row.push(number(st.accept_stat[s]));
        row.push(number(st.step_size[s]));
        row.push(st.tree_depth[s].to_string());
        row.push(st.tree_size[s].to_string());
        row.push(u8::from(self.divergent.get(s).copied().unwrap_or(false)).to_string());
        row.push(number(st.energy[s]));
    }
}

/// The columns of one node: its element names in column-major order, and the row-major offset of each.
struct NodeColumns {
    id: NodeId,
    names: Vec<String>,
    offsets: Vec<usize>,
}

impl NodeColumns {
    fn new(id: NodeId, name: String, shape: &[usize]) -> Self {
        if shape.is_empty() {
            return Self {
                id,
                names: vec![name],
                offsets: vec![0],
            };
        }
        let count: usize = shape.iter().product();
        let mut names = Vec::with_capacity(count);
        let mut offsets = Vec::with_capacity(count);
        // row-major strides
        let mut strides = vec![1; shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1] * shape[axis + 1];
        }
        for flat in 0..count {
            // column-major: the first index varies fastest
            let mut rest = flat;
            let mut offset = 0;
            let mut index = Vec::with_capacity(shape.len());
            for (axis, &dim) in shape.iter().enumerate() {
                let i = rest % dim;
                rest /= dim;
                offset += i * strides[axis];
                index.push((i + 1).to_string());
            }
            names.push(format!("{name}.{}", index.join(".")));
            offsets.push(offset);
        }
        Self { id, names, offsets }
    }

    fn push_values(&self, sample: &NetworkSample<'_>, row: &mut Vec<String>) -> Result<()> {
        let missing = || {
            AlgorithmError::InvalidArgument(format!(
                "draw {} lacks {}",
                sample.index(),
                self.names[0]
            ))
        };
        match sample.get(self.id).ok_or_else(missing)? {
            Value::F64(t) => {
                let data = t.as_slice();
                row.extend(self.offsets.iter().map(|&o| number(data[o])));
            }
            Value::I64(t) => {
                let data = t.as_slice();
                row.extend(self.offsets.iter().map(|&o| data[o].to_string()));
            }
            Value::Bool(t) => {
                let data = t.as_slice();
                row.extend(self.offsets.iter().map(|&o| u8::from(data[o]).to_string()));
            }
            _ => {
                return Err(AlgorithmError::InvalidArgument(format!(
                    "{} has a value type CmdStan CSV can't hold",
                    self.names[0]
                )));
            }
        }
        Ok(())
    }
}

/// A node's column name: its label, or `node_` and its index, with characters other than ASCII letters, digits
/// and `_` replaced by `_`.
fn stan_name(graph: &Graph, id: NodeId) -> String {
    match graph.node(id).and_then(|node| node.label()) {
        Some(label) if !label.is_empty() => label
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect(),
        _ => format!("node_{}", id.index()),
    }
}

/// A number as Stan writes it: the shortest form that reads back exactly, in exponent form when very large or
/// small, and `nan`, `inf` or `-inf`.
fn number(x: f64) -> String {
    if x.is_nan() {
        "nan".into()
    } else if x.is_infinite() {
        if x > 0.0 { "inf".into() } else { "-inf".into() }
    } else if x == 0.0 || (1e-5..1e16).contains(&x.abs()) {
        format!("{x}")
    } else {
        format!("{x:e}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_round_trip() {
        for x in [
            0.0,
            -0.0,
            1.5,
            -2.25,
            1e-7,
            3.0e20,
            0.1,
            123_456.789,
            f64::MIN_POSITIVE,
        ] {
            let text = number(x);
            assert_eq!(text.parse::<f64>().unwrap(), x, "{text}");
        }
        assert_eq!(number(1e-7), "1e-7");
        assert_eq!(number(0.25), "0.25");
        assert_eq!(number(f64::NAN), "nan");
        assert_eq!(number(f64::NEG_INFINITY), "-inf");
    }

    #[test]
    fn matrices_are_column_major() {
        let columns = NodeColumns::new(NodeId::from_index(0), "L".into(), &[2, 3]);
        assert_eq!(
            columns.names,
            ["L.1.1", "L.2.1", "L.1.2", "L.2.2", "L.1.3", "L.2.3"]
        );
        // row-major offsets of those elements
        assert_eq!(columns.offsets, [0, 3, 1, 4, 2, 5]);
        let scalar = NodeColumns::new(NodeId::from_index(0), "mu".into(), &[]);
        assert_eq!(scalar.names, ["mu"]);
    }
}
