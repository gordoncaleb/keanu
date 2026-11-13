//! The Keanu side of the Keanu / Stan / PyMC comparison in `benchmarks/ppl`.
//!
//! Each model is one of the examples', on the example's data, sampled with NUTS: 4 chains in parallel, 1000
//! adaptation steps and 1000 draws each. For each model this writes, under the output directory:
//!
//! - `<model>/data.json`: the data, which the Stan and PyMC versions read;
//! - `<model>/keanu.csv`: the draws (`chain,draw,<param>,...`, array elements as `name.i`);
//! - `<model>/keanu.json`: the wall time of sampling, the leapfrog steps and the divergences.
//!
//! Run with `cargo bench -p keanu --features parallel --bench ppl_comparison -- <out dir> [model ...]`.

#![allow(dead_code)]

#[path = "../examples/disease_mapping.rs"]
mod disease_mapping;
#[path = "../examples/eight_schools.rs"]
mod eight_schools;
#[path = "../examples/gaussian_mixture.rs"]
mod gaussian_mixture;
#[path = "../examples/gp_regression.rs"]
mod gp_regression;
#[path = "../examples/hmm.rs"]
mod hmm;
#[path = "../examples/radon.rs"]
mod radon;
#[path = "../examples/stochastic_volatility.rs"]
mod stochastic_volatility;

use keanu::algorithms::mcmc::chains::run_chains;
use keanu::graph::gp::{GaussianProcess, Kernel};
use keanu::graph::ode::OdeOptions;
use keanu::graph::op::DistributionKind;
use keanu::prelude::*;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

const CHAINS: usize = 4;
const WARMUP: usize = 1000;
const DRAWS: usize = 1000;
const SEED: u64 = 1;

/// A built model ready to sample: what to record under which name, and the NUTS settings.
struct Bench {
    model: Model,
    state: State,
    /// The nodes recorded, by name.
    recorded: Vec<(&'static str, NodeId)>,
    target_accept: f64,
    data: Json,
}

/// A tiny JSON writer for the data files.
#[derive(Default)]
struct Json(Vec<(String, String)>);

impl Json {
    fn num(mut self, key: &str, value: f64) -> Self {
        self.0.push((key.into(), fmt_f64(value)));
        self
    }
    fn int(mut self, key: &str, value: i64) -> Self {
        self.0.push((key.into(), value.to_string()));
        self
    }
    fn vec(mut self, key: &str, values: &[f64]) -> Self {
        let items: Vec<String> = values.iter().map(|&v| fmt_f64(v)).collect();
        self.0.push((key.into(), format!("[{}]", items.join(","))));
        self
    }
    fn ints(mut self, key: &str, values: &[i64]) -> Self {
        let items: Vec<String> = values.iter().map(i64::to_string).collect();
        self.0.push((key.into(), format!("[{}]", items.join(","))));
        self
    }
    fn matrix(mut self, key: &str, rows: &[Vec<f64>]) -> Self {
        let rows: Vec<String> = rows
            .iter()
            .map(|r| {
                let items: Vec<String> = r.iter().map(|&v| fmt_f64(v)).collect();
                format!("[{}]", items.join(","))
            })
            .collect();
        self.0.push((key.into(), format!("[{}]", rows.join(","))));
        self
    }
    fn render(&self) -> String {
        let fields: Vec<String> = self
            .0
            .iter()
            .map(|(k, v)| format!("  \"{k}\": {v}"))
            .collect();
        format!("{{\n{}\n}}\n", fields.join(",\n"))
    }
}

fn fmt_f64(v: f64) -> String {
    format!("{v:e}")
}

fn eight_schools() -> Result<Bench> {
    use eight_schools::{EFFECTS, STANDARD_ERRORS};
    let n = EFFECTS.len();
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 5.0);
    let tau = b.half_cauchy(5.0);
    b.set_value(mu, DoubleTensor::scalar(0.0));
    b.set_value(tau, DoubleTensor::scalar(1.0));
    let eta = b.gaussian_with_shape(&[n], 0.0, 1.0);
    b.set_value(eta, DoubleTensor::vector(vec![0.0; n]));
    let theta = mu + tau * eta;
    let sigma = b.constant(DoubleTensor::vector(STANDARD_ERRORS.to_vec()));
    let y = b.gaussian_with_shape(&[n], theta, sigma);
    b.observe(y, DoubleTensor::vector(EFFECTS.to_vec()));
    let (mu, tau, eta) = (mu.id(), tau.id(), eta.id());
    let (graph, state) = b.build()?;
    Ok(Bench {
        model: Model::new(graph, &state)?,
        state,
        recorded: vec![("mu", mu), ("tau", tau), ("eta", eta)],
        target_accept: 0.95,
        data: Json::default()
            .int("J", n as i64)
            .vec("y", &EFFECTS)
            .vec("sigma", &STANDARD_ERRORS),
    })
}

fn radon() -> Result<Bench> {
    let houses = radon::data();
    let r = radon::model()?;
    Ok(Bench {
        model: r.model,
        state: r.state,
        recorded: vec![
            ("mu_a", r.mu_a),
            ("sigma_a", r.sigma_a),
            ("b", r.b),
            ("sigma_y", r.sigma_y),
            ("a", r.a),
        ],
        target_accept: 0.8,
        data: Json::default()
            .int("N", houses.log_radon.len() as i64)
            .int("J", radon::COUNTIES as i64)
            .ints("county", &houses.county)
            .vec("floor", &houses.floor)
            .vec("log_radon", &houses.log_radon),
    })
}

fn robust_regression() -> Result<Bench> {
    // the example's data
    const POINTS: usize = 100;
    let mut rng = KeanuRng::seed_from_u64(42);
    let x: Vec<f64> = (0..POINTS)
        .map(|i| 4.0 * i as f64 / (POINTS - 1) as f64 - 2.0)
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&x| {
            let variance = (0..3).map(|_| rng.next_gaussian().powi(2)).sum::<f64>() / 3.0;
            0.5 + 1.5 * x + 0.35 * rng.next_gaussian() / variance.sqrt()
        })
        .collect();
    let b = GraphBuilder::new();
    let intercept = b.gaussian(0.0, 1.0);
    let slope = b.gaussian(0.0, 2.0);
    let scale = b.log_normal(0.5_f64.ln(), 0.7);
    let nu = b.log_normal(5.0_f64.ln(), 0.9);
    for (node, value) in [(intercept, 0.0), (slope, 1.0), (scale, 0.5), (nu, 5.0)] {
        b.set_value(node, DoubleTensor::scalar(value));
    }
    let mean = intercept + slope * b.constant(DoubleTensor::vector(x.clone()));
    b.observe(
        b.location_scale_student_t(nu, mean, scale),
        DoubleTensor::vector(y.clone()),
    );
    let ids = [intercept.id(), slope.id(), scale.id(), nu.id()];
    let (graph, state) = b.build()?;
    Ok(Bench {
        model: Model::new(graph, &state)?,
        state,
        recorded: vec![
            ("intercept", ids[0]),
            ("slope", ids[1]),
            ("scale", ids[2]),
            ("nu", ids[3]),
        ],
        target_accept: 0.8,
        data: Json::default()
            .int("N", POINTS as i64)
            .vec("x", &x)
            .vec("y", &y),
    })
}

fn stochastic_volatility() -> Result<Bench> {
    let returns = stochastic_volatility::data().returns;
    let days = returns.len();
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 10.0);
    let phi_raw = b.beta(20.0, 1.5);
    let phi = phi_raw * 2.0 - 1.0;
    let sigma = b.half_cauchy(5.0);
    let eps_0 = b.gaussian(0.0, 1.0);
    let eps = b.gaussian_with_shape(&[days], 0.0, 1.0);
    b.set_value(mu, DoubleTensor::scalar(0.0));
    b.set_value(phi_raw, DoubleTensor::scalar(0.9));
    b.set_value(sigma, DoubleTensor::scalar(0.5));
    b.set_value(eps_0, DoubleTensor::scalar(0.0));
    b.set_value(eps, DoubleTensor::vector(vec![0.0; days]));
    let h_0 = mu + sigma / (1.0 - phi * phi).sqrt() * eps_0;
    let level = mu * (1.0 - phi);
    let shocks = sigma * eps;
    let h = b.scan(h_0, &[shocks], &[level, phi], |_, h, shock, c| {
        let (level, phi) = (c[0], c[1]);
        level + phi * h + shock[0]
    });
    let r = b.gaussian_with_shape(&[days], 0.0, (h * 0.5).exp());
    b.observe(r, DoubleTensor::vector(returns.clone()));
    let recorded = vec![
        ("mu", mu.id()),
        ("phi", phi.id()),
        ("sigma", sigma.id()),
        ("h", h.id()),
    ];
    let (graph, state) = b.build()?;
    Ok(Bench {
        model: Model::new(graph, &state)?,
        state,
        recorded,
        target_accept: 0.95,
        data: Json::default().int("T", days as i64).vec("r", &returns),
    })
}

fn gp_regression() -> Result<Bench> {
    let (x, y) = gp_regression::data(1);
    let b = GraphBuilder::new();
    let amplitude = b.half_gaussian(2.0);
    let length_scale = b.half_gaussian(2.0);
    let noise = b.half_gaussian(1.0);
    b.set_value(amplitude, DoubleTensor::scalar(1.0));
    b.set_value(length_scale, DoubleTensor::scalar(1.0));
    b.set_value(noise, DoubleTensor::scalar(0.5));
    GaussianProcess::new(Kernel::exp_quad(&b, amplitude, length_scale))
        .with_noise(noise)
        .marginal(
            DoubleTensor::vector(x.clone()),
            DoubleTensor::vector(y.clone()),
        );
    let ids = [amplitude.id(), length_scale.id(), noise.id()];
    let (graph, state) = b.build()?;
    Ok(Bench {
        model: Model::new(graph, &state)?,
        state,
        recorded: vec![
            ("amplitude", ids[0]),
            ("length_scale", ids[1]),
            ("noise", ids[2]),
        ],
        target_accept: 0.8,
        data: Json::default()
            .int("N", x.len() as i64)
            .vec("x", &x)
            .vec("y", &y),
    })
}

fn gaussian_mixture() -> Result<Bench> {
    let observations = gaussian_mixture::data(300, 7);
    let b = GraphBuilder::new();
    let first = b.gaussian_with_shape(&[1], 0.0, 10.0);
    let gap = b.half_gaussian_with_shape(&[1], 10.0);
    let means = b.concat(0, &[first, gap]).cumsum(0);
    let sigmas = b.half_gaussian_with_shape(&[2], 5.0);
    let weights = b.dirichlet(DoubleTensor::vector(vec![2.0, 2.0]));
    let y: DoubleVar = b.mixture_with_shape(
        &[observations.len()],
        DistributionKind::Gaussian,
        &[means.id(), sigmas.id()],
        weights,
    );
    b.observe(y, DoubleTensor::vector(observations.clone()));
    b.set_value(first, DoubleTensor::vector(vec![0.0]));
    b.set_value(gap, DoubleTensor::vector(vec![1.0]));
    b.set_value(sigmas, DoubleTensor::vector(vec![1.0, 1.0]));
    b.set_value(weights, DoubleTensor::vector(vec![0.5, 0.5]));
    let recorded = vec![
        ("means", means.id()),
        ("sigmas", sigmas.id()),
        ("weights", weights.id()),
    ];
    let (graph, state) = b.build()?;
    Ok(Bench {
        model: Model::new(graph, &state)?,
        state,
        recorded,
        target_accept: 0.8,
        data: Json::default()
            .int("N", observations.len() as i64)
            .vec("y", &observations),
    })
}

fn item_response() -> Result<Bench> {
    // the example's data, without the held-out predictions
    const STUDENTS: usize = 100;
    const ITEMS: usize = 8;
    let mut rng = KeanuRng::seed_from_u64(74);
    let abilities: Vec<_> = (0..STUDENTS).map(|_| rng.next_gaussian()).collect();
    let difficulty = [-1.8, -1.2, -0.6, -0.2, 0.3, 0.7, 1.2, 1.8];
    let discrimination = [1.4, 0.8, 1.0, 1.3, 0.8, 1.0, 1.4, 1.2];
    let (mut students, mut items, mut responses) = (Vec::new(), Vec::new(), Vec::new());
    for (student, &ability) in abilities.iter().enumerate() {
        for item in 0..ITEMS {
            let p = 1.0 / (1.0 + (-(ability - difficulty[item]) * discrimination[item]).exp());
            let correct = rng.next_f64() < p;
            if (student + item) % ITEMS != 0 {
                students.push(student as i64);
                items.push(item as i64);
                responses.push(correct);
            }
        }
    }
    let b = GraphBuilder::new();
    let ability = b.gaussian_with_shape(&[STUDENTS], 0.0, 1.0);
    let difficulty_mean = b.gaussian(0.0, 1.0);
    let difficulty_scale = b.log_normal(0.0, 0.5);
    let difficulty_offset = b.gaussian_with_shape(&[ITEMS], 0.0, 1.0);
    let item_difficulty = difficulty_mean + difficulty_scale * difficulty_offset;
    let log_discrimination_mean = b.gaussian(0.0, 0.4);
    let log_discrimination_scale = b.log_normal(0.3_f64.ln(), 0.5);
    let discrimination_offset = b.gaussian_with_shape(&[ITEMS], 0.0, 1.0);
    let item_discrimination =
        (log_discrimination_mean + log_discrimination_scale * discrimination_offset).exp();
    let student_index = b.int_constant(IntTensor::vector(students.clone()));
    let item_index = b.int_constant(IntTensor::vector(items.clone()));
    let logits = item_discrimination.gather(0, item_index)
        * (ability.gather(0, student_index) - item_difficulty.gather(0, item_index));
    b.observe(
        b.bernoulli_logit(logits),
        BoolTensor::vector(responses.clone()),
    );
    b.set_value(ability, DoubleTensor::zeros(&[STUDENTS]));
    b.set_value(difficulty_mean, DoubleTensor::scalar(0.0));
    b.set_value(difficulty_scale, DoubleTensor::scalar(1.0));
    b.set_value(difficulty_offset, DoubleTensor::zeros(&[ITEMS]));
    b.set_value(log_discrimination_mean, DoubleTensor::scalar(0.0));
    b.set_value(log_discrimination_scale, DoubleTensor::scalar(0.3));
    b.set_value(discrimination_offset, DoubleTensor::zeros(&[ITEMS]));
    let recorded = vec![
        ("ability", ability.id()),
        ("difficulty_mean", difficulty_mean.id()),
        ("difficulty_scale", difficulty_scale.id()),
        ("difficulty", item_difficulty.id()),
        ("log_discrimination_mean", log_discrimination_mean.id()),
        ("log_discrimination_scale", log_discrimination_scale.id()),
        ("discrimination", item_discrimination.id()),
    ];
    let (graph, state) = b.build()?;
    let responses: Vec<i64> = responses.iter().map(|&c| i64::from(c)).collect();
    Ok(Bench {
        model: Model::new(graph, &state)?,
        state,
        recorded,
        target_accept: 0.8,
        data: Json::default()
            .int("N", responses.len() as i64)
            .int("S", STUDENTS as i64)
            .int("I", ITEMS as i64)
            .ints("student", &students)
            .ints("item", &items)
            .ints("correct", &responses),
    })
}

fn hmm() -> Result<Bench> {
    let observations = hmm::simulate(hmm::STEPS, hmm::DATA_SEED).observations;
    let b = GraphBuilder::new();
    let rows = [
        b.dirichlet(DoubleTensor::vector(vec![1.0, 1.0])),
        b.dirichlet(DoubleTensor::vector(vec![1.0, 1.0])),
    ];
    let transition = b.stack(0, &rows);
    let first = b.gaussian_with_shape(&[1], 0.0, 5.0);
    let gap = b.half_gaussian_with_shape(&[1], 5.0);
    let means = b.concat(0, &[first, gap]).cumsum(0);
    let sigma = b.half_gaussian(2.0);
    let y: DoubleVar = b.hidden_markov_model(
        observations.len(),
        DistributionKind::Gaussian,
        &[means.id(), sigma.id()],
        DoubleTensor::vector(hmm::INITIAL.to_vec()),
        transition,
    );
    b.observe(y, DoubleTensor::vector(observations.clone()));
    for row in rows {
        b.set_value(row, DoubleTensor::vector(vec![0.5, 0.5]));
    }
    b.set_value(first, DoubleTensor::vector(vec![0.0]));
    b.set_value(gap, DoubleTensor::vector(vec![1.0]));
    b.set_value(sigma, DoubleTensor::scalar(1.0));

    let recorded = vec![
        ("transition", transition.id()),
        ("means", means.id()),
        ("sigma", sigma.id()),
    ];
    let (graph, state) = b.build()?;
    Ok(Bench {
        model: Model::new(graph, &state)?,
        state,
        recorded,
        target_accept: 0.8,
        data: Json::default()
            .int("T", observations.len() as i64)
            .vec("y", &observations),
    })
}

fn disease_mapping() -> Result<Bench> {
    let areas = disease_mapping::data(6)?;
    let b = GraphBuilder::new();
    let beta0 = b.gaussian(0.0, 1.0);
    let beta1 = b.gaussian(0.0, 1.0);
    let sigma = b.half_gaussian(1.0);
    let rho = b.beta(1.0, 1.0);
    let bym2 = b.bym2(areas.adjacency.clone(), sigma, rho)?;
    let log_expected = b.constant(DoubleTensor::vector(
        areas.expected.iter().map(|e| e.ln()).collect(),
    ));
    let covariate = b.constant(DoubleTensor::vector(areas.covariate.clone()));
    let y = b.poisson_log(log_expected + beta0 + covariate * beta1 + bym2.effect);
    b.observe(y, IntTensor::vector(areas.counts.clone()));
    b.set_value(sigma, DoubleTensor::scalar(0.5));
    b.set_value(rho, DoubleTensor::scalar(0.5));
    b.set_value(beta0, DoubleTensor::scalar(0.0));
    b.set_value(beta1, DoubleTensor::scalar(0.0));
    b.set_value(bym2.spatial, DoubleTensor::zeros(&[disease_mapping::AREAS]));
    b.set_value(
        bym2.unstructured,
        DoubleTensor::zeros(&[disease_mapping::AREAS]),
    );
    let recorded = vec![
        ("beta0", beta0.id()),
        ("beta1", beta1.id()),
        ("sigma", sigma.id()),
        ("rho", rho.id()),
        ("effect", bym2.effect.id()),
    ];
    let (graph, state) = b.build()?;
    let edges = areas.adjacency.edges();
    let scaling = areas.adjacency.scaling_factors()?;
    Ok(Bench {
        model: Model::new(graph, &state)?,
        state,
        recorded,
        target_accept: 0.8,
        data: Json::default()
            .int("N", disease_mapping::AREAS as i64)
            .int("N_edges", edges.len() as i64)
            .ints(
                "node1",
                &edges.iter().map(|e| e.0 as i64).collect::<Vec<_>>(),
            )
            .ints(
                "node2",
                &edges.iter().map(|e| e.1 as i64).collect::<Vec<_>>(),
            )
            .num("scaling_factor", scaling[0])
            .vec("E", &areas.expected)
            .vec("x", &areas.covariate)
            .ints("y", &areas.counts),
    })
}

fn lotka_volterra() -> Result<Bench> {
    // the example's data
    const INITIAL: [f64; 2] = [1.2, 0.7];
    const TRUE_RATES: [f64; 4] = [0.9, 0.55, 0.65, 0.45];
    const TIMES: usize = 30;
    const TRAIN: usize = 25;
    const NOISE: f64 = 0.04;
    let derivative = |[prey, predator]: [f64; 2]| -> [f64; 2] {
        let [alpha, beta, gamma, delta] = TRUE_RATES;
        [
            prey * (alpha - beta * predator),
            predator * (delta * prey - gamma),
        ]
    };
    let step = 0.0025;
    let mut s = INITIAL;
    let mut truth = Vec::with_capacity(2 * TIMES);
    for _ in 0..TIMES {
        for _ in 0..100 {
            let k1 = derivative(s);
            let offset =
                |rate: [f64; 2], scale: f64| std::array::from_fn(|i| s[i] + step * scale * rate[i]);
            let k2 = derivative(offset(k1, 0.5));
            let k3 = derivative(offset(k2, 0.5));
            let k4 = derivative(offset(k3, 1.0));
            s = std::array::from_fn(|i| {
                s[i] + step * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0
            });
        }
        truth.extend(s);
    }
    let mut rng = KeanuRng::seed_from_u64(21);
    let observed: Vec<f64> = truth[..2 * TRAIN]
        .iter()
        .map(|value| value * (NOISE * rng.next_gaussian()).exp())
        .collect();

    let b = GraphBuilder::new();
    let centres = DoubleTensor::vector(vec![0.8, 0.6, 0.6, 0.5]);
    let rates = b.log_normal(centres.ln(), 0.35);
    b.set_value(rates, centres);
    let train_times: Vec<f64> = (1..=TRAIN).map(|i| 0.25 * i as f64).collect();
    let times = b.constant(DoubleTensor::vector(train_times.clone()));
    let populations = b.ode(
        b.constant(DoubleTensor::vector(INITIAL.to_vec())),
        times,
        b.scalar(0.0),
        &[rates],
        OdeOptions::default(),
        |body, _, population, parameters| {
            let rates = parameters[0];
            let prey = population.take(&[0]);
            let predator = population.take(&[1]);
            body.stack(
                0,
                &[
                    prey * (rates.take(&[0]) - rates.take(&[1]) * predator),
                    predator * (rates.take(&[3]) * prey - rates.take(&[2])),
                ],
            )
        },
    );
    let observation = b.log_normal(populations.ln(), NOISE);
    b.observe(
        observation,
        DoubleTensor::new(&[TRAIN, 2], observed.clone())?,
    );
    let rates = rates.id();
    let (graph, state) = b.build()?;
    let rows: Vec<Vec<f64>> = observed.chunks(2).map(<[f64]>::to_vec).collect();
    Ok(Bench {
        model: Model::new(graph, &state)?,
        state,
        recorded: vec![("rates", rates)],
        target_accept: 0.9,
        data: Json::default()
            .int("T", TRAIN as i64)
            .vec("ts", &train_times)
            .vec("y0", &INITIAL)
            .matrix("y", &rows)
            .num("noise", NOISE),
    })
}

/// Builds one benchmark model.
type Build = fn() -> Result<Bench>;

const MODELS: &[(&str, Build)] = &[
    ("eight_schools", eight_schools),
    ("radon", radon),
    ("robust_regression", robust_regression),
    ("stochastic_volatility", stochastic_volatility),
    ("gp_regression", gp_regression),
    ("gaussian_mixture", gaussian_mixture),
    ("item_response", item_response),
    ("hmm", hmm),
    ("disease_mapping", disease_mapping),
    ("lotka_volterra", lotka_volterra),
];

fn run(name: &str, bench: Bench, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("data.json"), bench.data.render()).unwrap();
    // NUTS moves every latent of the model and records these nodes
    let recorded: Vec<NodeId> = bench.recorded.iter().map(|(_, id)| *id).collect();
    let nuts = Nuts::default()
        .with_adapt_count(WARMUP)
        .with_target_acceptance_prob(bench.target_accept)
        .with_save_statistics(true);
    let start = Instant::now();
    let chains = run_chains(CHAINS, SEED, &bench.state, |state, rng| {
        nuts.sample_with_statistics(
            &bench.model,
            state,
            &recorded,
            WARMUP + DRAWS,
            SamplingOptions::default().with_drop(WARMUP),
            rng,
        )
    })?;
    let seconds = start.elapsed().as_secs_f64();

    // the draws, one column per element
    let mut header = String::from("chain,draw");
    let mut columns: Vec<Vec<Vec<f64>>> = Vec::new(); // [column][chain][draw]
    for (label, id) in &bench.recorded {
        let shape = chains[0].0.f64(*id).unwrap().as_slice()[0].shape().to_vec();
        let size: usize = shape.iter().product();
        for flat in 0..size {
            let mut index = vec![0; shape.len()];
            let mut rest = flat;
            for d in (0..shape.len()).rev() {
                index[d] = rest % shape[d];
                rest /= shape[d];
            }
            if index.is_empty() {
                write!(header, ",{label}").unwrap();
            } else {
                let parts: Vec<String> = index.iter().map(usize::to_string).collect();
                write!(header, ",{label}.{}", parts.join(".")).unwrap();
            }
            columns.push(
                chains
                    .iter()
                    .map(|(samples, _)| samples.f64(*id).unwrap().element(&index).unwrap())
                    .collect(),
            );
        }
    }
    let mut csv = header + "\n";
    for chain in 0..CHAINS {
        for draw in 0..DRAWS {
            write!(csv, "{chain},{draw}").unwrap();
            for column in &columns {
                write!(csv, ",{:e}", column[chain][draw]).unwrap();
            }
            csv.push('\n');
        }
    }
    std::fs::write(dir.join("keanu.csv"), csv).unwrap();

    let leapfrog: usize = chains
        .iter()
        .map(|(_, s)| s.tree_size.iter().sum::<usize>())
        .sum();
    let warmup_leapfrog: usize = chains
        .iter()
        .map(|(_, s)| s.tree_size[..WARMUP].iter().sum::<usize>())
        .sum();
    let divergences: usize = chains.iter().map(|(_, s)| s.divergence_count()).sum();
    let stats = Json::default()
        .num("seconds", seconds)
        .int("leapfrog", leapfrog as i64)
        .int("warmup_leapfrog", warmup_leapfrog as i64)
        .int("divergences", divergences as i64)
        .int("chains", CHAINS as i64)
        .int("warmup", WARMUP as i64)
        .int("draws", DRAWS as i64);
    std::fs::write(dir.join("keanu.json"), stats.render()).unwrap();
    println!(
        "{name:>22}: {seconds:7.2} s, {leapfrog:>9} leapfrog steps ({:.1} µs each), {divergences} divergences",
        seconds * 1e6 * CHAINS as f64 / leapfrog as f64
    );
    Ok(())
}

/// The cost of one log density and gradient evaluation at the starting point, in µs (the fastest of 7 batches).
fn gradient_cost(bench: &Bench) -> f64 {
    use keanu::graph::Parameterisation;
    let param = Parameterisation::new(&bench.model, &bench.state).unwrap();
    let mut state = bench.state.clone();
    let z = param.from_state(&state).unwrap();
    if std::env::var("KEANU_BENCH_USE_COMPILED").is_ok() {
        let mut compiled = param.compile(&bench.state).unwrap();
        let n: usize = std::env::var("KEANU_BENCH_ITERS").map_or(2000, |r| r.parse().unwrap());
        let start = Instant::now();
        for _ in 0..n {
            std::hint::black_box(
                param
                    .log_prob_and_gradient_compiled(&mut compiled, &z, true, &bench.state)
                    .unwrap(),
            );
        }
        return start.elapsed().as_secs_f64() * 1e6 / n as f64;
    }
    let mut times = Vec::new();
    let n: usize = std::env::var("KEANU_BENCH_ITERS").map_or(2000, |r| r.parse().unwrap());
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..n {
            std::hint::black_box(param.log_prob_and_gradient(&z, true, &mut state).unwrap());
        }
        times.push(start.elapsed().as_secs_f64() * 1e6 / n as f64);
    }
    times.sort_by(f64::total_cmp);
    times[0]
}

/// Compares the compiled log probability and gradient with the general path's at a few points, and times both.
fn check_compiled(name: &str, bench: &Bench) {
    use keanu::graph::Parameterisation;
    let param = Parameterisation::new(&bench.model, &bench.state).unwrap();
    let Some(mut compiled) = param.compile(&bench.state) else {
        println!("{name:>22}: not compiled");
        return;
    };
    let mut state = bench.state.clone();
    let start = param.from_state(&state).unwrap();
    let mut identical = true;
    let mut worst: f64 = 0.0;
    for k in 0..6 {
        let z: Vec<f64> = start
            .iter()
            .enumerate()
            .map(|(i, &x)| x + 0.3 * k as f64 * ((i as f64 * 1.7 + k as f64).sin()))
            .collect();
        let (lp, g) = param.log_prob_and_gradient(&z, true, &mut state).unwrap();
        let (clp, cg) = param
            .log_prob_and_gradient_compiled(&mut compiled, &z, true, &bench.state)
            .unwrap();
        if lp.to_bits() != clp.to_bits() && !(lp.is_nan() && clp.is_nan()) {
            identical = false;
            worst = worst.max((lp - clp).abs());
        }
        match (&g, &cg) {
            (Some(g), Some(cg)) => {
                for (a, b) in g.iter().zip(cg) {
                    if a.to_bits() != b.to_bits() {
                        identical = false;
                        worst = worst.max((a - b).abs() / a.abs().max(1.0));
                    }
                }
            }
            (None, None) => {}
            _ => {
                identical = false;
                worst = f64::INFINITY;
            }
        }
    }
    let time = |f: &mut dyn FnMut()| {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..500 {
                f();
            }
            best = best.min(t.elapsed().as_secs_f64() * 1e6 / 500.0);
        }
        best
    };
    let general = time(&mut || {
        std::hint::black_box(
            param
                .log_prob_and_gradient(&start, true, &mut state)
                .unwrap(),
        );
    });
    let fast = time(&mut || {
        std::hint::black_box(
            param
                .log_prob_and_gradient_compiled(&mut compiled, &start, true, &bench.state)
                .unwrap(),
        );
    });
    println!(
        "{name:>22}: {} (worst {worst:.1e}); general {general:8.2} µs, compiled {fast:8.2} µs",
        if identical { "identical" } else { "DIFFERENT" }
    );
}

fn main() {
    if std::env::var("KEANU_BENCH_CHECK").is_ok() {
        for (name, build) in MODELS {
            check_compiled(name, &build().unwrap());
        }
        return;
    }
    if let Ok(filter) = std::env::var("KEANU_BENCH_GRADIENT") {
        for (name, build) in MODELS {
            if !filter.is_empty() && !filter.split(',').any(|f| f == *name) {
                continue;
            }
            let bench = build().unwrap();
            let gradient = gradient_cost(&bench);
            if std::env::var("KEANU_BENCH_NO_NUTS").is_ok() {
                println!("{name:>22}: {gradient:8.2} µs per gradient");
                continue;
            }
            // one chain on this thread: the cost of a leapfrog step, gradient and NUTS's own work together
            let recorded: Vec<NodeId> = bench.recorded.iter().map(|(_, id)| *id).collect();
            let mut state = bench.state.clone();
            let start = Instant::now();
            let (_, stats) = Nuts::default()
                .with_adapt_count(300)
                .with_target_acceptance_prob(bench.target_accept)
                .with_save_statistics(true)
                .sample_with_statistics(
                    &bench.model,
                    &mut state,
                    &recorded,
                    600,
                    SamplingOptions::default().with_drop(300),
                    &mut KeanuRng::seed_from_u64(SEED),
                )
                .unwrap();
            let leapfrog: usize = stats.tree_size.iter().sum();
            let per_step = start.elapsed().as_secs_f64() * 1e6 / leapfrog as f64;
            println!(
                "{name:>22}: {gradient:8.2} µs per gradient, {per_step:8.2} µs per leapfrog step in one chain"
            );
        }
        return;
    }
    let mut args = std::env::args().skip(1).filter(|a| a != "--bench");
    let out =
        PathBuf::from(args.next().unwrap_or_else(|| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/../benchmarks/ppl/out").into()
        }));
    let only: Vec<String> = args.collect();
    for (name, build) in MODELS {
        if !only.is_empty() && !only.iter().any(|o| o == name) {
            continue;
        }
        // KEANU_BENCH_REPEAT reruns each model, for profiling
        let repeat: usize = std::env::var("KEANU_BENCH_REPEAT").map_or(1, |r| r.parse().unwrap());
        for _ in 0..repeat {
            let bench = build().unwrap_or_else(|e| panic!("{name}: {e}"));
            run(name, bench, &out.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }
}
