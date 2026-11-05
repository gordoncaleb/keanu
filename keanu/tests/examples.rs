//! The examples, checked against the results the Java examples' tests assert.
//!
//! The Java tests run short Metropolis–Hastings chains and pass for one particular seed; with a different random
//! number generator those chain lengths don't converge reliably (checked over many seeds), so these tests run
//! chains 20 times longer than the Java defaults and assert the same tolerances. Where Metropolis–Hastings with
//! prior proposals can't mix (the Challenger model's strongly correlated parameters), the posterior is checked with
//! NUTS and the Java-style run is only checked to complete. The examples with discrete latents (coal mining
//! disasters, text messaging and the Bernoulli form of cheating students) use the default compound sampler and are
//! checked against their exact posteriors instead, computed in closed form or on a grid.
//!
//! The hierarchical examples with no Java counterpart (eight schools, radon) are checked against published or
//! independently computed posteriors, with multi-chain R-hat and divergence counts; each passed over seeds 1 to 8.
//! The GP regression example is checked against the function its data were generated from, and the stochastic
//! volatility and hidden Markov model examples against the parameters their synthetic data were simulated with.

#![allow(dead_code)]

#[path = "../examples/ab_testing.rs"]
mod ab_testing;
#[path = "../examples/challenger_disaster.rs"]
mod challenger_disaster;
#[path = "../examples/cheating_students.rs"]
mod cheating_students;
#[path = "../examples/coal_mining_disasters.rs"]
mod coal_mining_disasters;
#[path = "../examples/custom_distribution.rs"]
mod custom_distribution;
#[path = "../examples/eight_schools.rs"]
mod eight_schools;
#[path = "../examples/gaussian_mixture.rs"]
mod gaussian_mixture;
#[path = "../examples/gp_regression.rs"]
mod gp_regression;
#[path = "../examples/hmm.rs"]
mod hmm;
#[path = "../examples/multi_state.rs"]
mod multi_state;
#[path = "../examples/pharmacokinetics.rs"]
mod pharmacokinetics;
#[path = "../examples/radon.rs"]
mod radon;
#[path = "../examples/sir.rs"]
mod sir;
#[path = "../examples/stochastic_volatility.rs"]
mod stochastic_volatility;
#[path = "../examples/text_messaging.rs"]
mod text_messaging;

const LONGER: usize = 20;

fn assert_close(actual: f64, expected: f64, within: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= within,
        "{what}: {actual} is not within {within} of {expected}"
    );
}

/// The exact posterior of a Poisson change-point model with Exponential(`prior_rate`) priors on both rates and a
/// uniform prior on the switch τ (times before τ have the early rate), which is conjugate given τ: each segment's
/// marginal likelihood is b Γ(S + 1) / (n + b)^(S + 1) for n counts summing to S, and its rate's posterior mean is
/// (S + 1) / (n + b). Returns the posterior means of τ and the two rates, and the posterior mode of τ.
fn exact_change_point(
    times: &[i64],
    counts: &[i64],
    switches: &[i64],
    prior_rate: f64,
) -> [f64; 4] {
    use keanu::tensor::special::ln_gamma;
    let segment = |n: f64, sum: f64| {
        prior_rate.ln() + ln_gamma(sum + 1.0) - (sum + 1.0) * (n + prior_rate).ln()
    };
    let mut rows = Vec::new();
    for &tau in switches {
        let (mut n1, mut s1, mut n2, mut s2) = (0.0, 0.0, 0.0, 0.0);
        for (&t, &y) in times.iter().zip(counts) {
            if t < tau {
                n1 += 1.0;
                s1 += y as f64;
            } else {
                n2 += 1.0;
                s2 += y as f64;
            }
        }
        let log_weight = segment(n1, s1) + segment(n2, s2);
        rows.push((
            tau as f64,
            log_weight,
            (s1 + 1.0) / (n1 + prior_rate),
            (s2 + 1.0) / (n2 + prior_rate),
        ));
    }
    let max = rows.iter().map(|r| r.1).fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = rows.iter().map(|r| (r.1 - max).exp()).sum();
    let mut means = [0.0; 3];
    for &(tau, log_weight, early, late) in &rows {
        let p = (log_weight - max).exp() / total;
        means[0] += p * tau;
        means[1] += p * early;
        means[2] += p * late;
    }
    let mode = rows.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
    [means[0], means[1], means[2], mode]
}

/// Checks a sampled change-point model: every R-hat below 1.01, and each posterior mean within four Monte Carlo
/// standard errors of the exact one.
fn check_change_point(
    chains: &keanu::algorithms::ChainSamples,
    nodes: [keanu::graph::NodeId; 3],
    exact: [f64; 4],
) {
    for (i, (node, what)) in nodes
        .into_iter()
        .zip(["switch", "early rate", "late rate"])
        .enumerate()
    {
        let summary = chains.summarize(node, &[]).unwrap();
        assert!(summary.r_hat < 1.01, "{what}: R-hat {}", summary.r_hat);
        assert_close(summary.mean, exact[i], 4.0 * summary.mcse_mean, what);
    }
}

#[test]
fn coal_mining_disasters_switch_year() {
    // AE2: the default (compound) sampler, four chains, every R-hat below 1.01. The draws were Metropolis–Hastings'
    // and are now the compound sampler's, so the checks are against the exact posterior; Java asserted exactly 1890
    // for its seed, and the posterior's mode is at 1890 with neighbouring years close behind
    let r = coal_mining_disasters::run(1, coal_mining_disasters::DEFAULT_SAMPLES).unwrap();
    assert!(
        (1888..=1892).contains(&r.switch_year_mode),
        "switch year {}",
        r.switch_year_mode
    );
    let data = coal_mining_disasters::data();
    let (years, counts): (Vec<i64>, Vec<i64>) = data.into_iter().unzip();
    let switches: Vec<i64> = (years[0]..=years[years.len() - 1] + 1).collect();
    let exact = exact_change_point(&years, &counts, &switches, 1.0);
    assert_eq!(exact[3], 1890.0);
    check_change_point(
        &r.chains,
        [r.switch_point, r.early_rate, r.late_rate],
        exact,
    );
}

#[test]
fn text_messaging_switch_point_and_rates() {
    // the compound sampler's draws, checked against the exact posterior (the late rate's mode, meaningful for
    // Metropolis–Hastings' repeated values, is replaced by its mean)
    let r = text_messaging::run(1, text_messaging::DEFAULT_SAMPLES).unwrap();
    assert!(
        (41..=45).contains(&r.switch_point_mode),
        "switch point {}",
        r.switch_point_mode
    );
    let (days, messages) = text_messaging::data();
    let switches: Vec<i64> = (0..days.len() as i64).collect();
    let exact = exact_change_point(&days, &messages, &switches, 0.1);
    check_change_point(
        &r.chains,
        [r.switch_point, r.early_rate, r.late_rate],
        exact,
    );
    assert_close(r.early_rate_mean, 18.0, 2.0, "early rate mean");
    assert_close(r.late_rate_mean, 23.0, 2.0, "late rate mean");
}

#[test]
fn ab_testing_posteriors() {
    let p = ab_testing::run(1, ab_testing::DEFAULT_SAMPLES * LONGER).unwrap();
    assert!((0.02..=0.07).contains(&p.p_a), "p_a {}", p.p_a);
    assert!((0.02..=0.07).contains(&p.p_b), "p_b {}", p.p_b);
}

#[test]
fn cheating_students_with_bernoulli() {
    // The compound sampler's draws (they were Metropolis–Hastings'), checked against the exact posterior mean of p:
    // the number of true answers is Binomial(100, p / 2 + 1 / 4) and the total is observed with N(0, 1) noise, so
    // p(p | y) ∝ Σ_a Binomial(a; 100, p / 2 + 1 / 4) N(35; a, 1), integrated on a grid
    let estimate = cheating_students::run_with_bernoulli(100, 35.0, 1, 2000).unwrap();
    let grid = 4000;
    let (mut total, mut first) = (0.0, 0.0);
    for i in 0..grid {
        let p = (i as f64 + 0.5) / grid as f64;
        let q: f64 = 0.5 * p + 0.25;
        let mut log_binomial = 100.0 * (1.0 - q).ln();
        let mut density = 0.0;
        for a in 0..=100 {
            if a > 0 {
                log_binomial += ((101 - a) as f64 / a as f64).ln() + (q / (1.0 - q)).ln();
            }
            density += (log_binomial - 0.5 * (35.0 - a as f64).powi(2)).exp();
        }
        total += density;
        first += p * density;
    }
    let exact = first / total;
    assert_close(
        estimate.mean,
        exact,
        4.0 * estimate.mcse,
        "posterior mean of p",
    );
    assert_close(estimate.mean, 0.2, 0.05, "probability of cheating");
}

#[test]
fn cheating_students_with_binomial() {
    let p = cheating_students::run_with_binomial(100, 35, 1, 100 * LONGER).unwrap();
    assert_close(p, 0.2, 0.05, "probability of cheating");
}

#[test]
fn challenger_disaster_beta_is_small_and_positive() {
    let p = challenger_disaster::run_with_nuts(1, 4000).unwrap();
    assert_close(p.beta, 0.2, 0.2, "beta");
    assert_close(p.alpha, -15.0, 15.0, "alpha");
    // the Java style run (Metropolis–Hastings with prior proposals) still completes
    challenger_disaster::run(1, challenger_disaster::DEFAULT_SAMPLES).unwrap();
}

/// The posterior means of the true rates, computed independently by integrating over a dense grid of
/// φ = (logit(a / (a + b)), ln(a + b)): E[θⱼ | y] = ∫ (a + yⱼ) / (a + b + n) p(φ | y) dφ, with
/// p(φ | y) ∝ a b (a + b)^-5/2 ∏ⱼ B(a + yⱼ, b + n - yⱼ) / B(a, b). They agree to 10⁻⁵ whether or not the hyperprior
/// is truncated at ln(a + b) = 20, as the example's is.
const EXACT_TRUE_RATE_MEANS: [f64; 10] = [
    0.44791, 0.46776, 0.48264, 0.51737, 0.56201, 0.47768, 0.46776, 0.49256, 0.53721, 0.49752,
];

#[test]
fn custom_distribution_posterior_matches_numerical_integration() {
    // Sampled in the reparameterised coordinates, the chains mix well. Over seeds 1 to 40, a rate's mean was off by
    // 0.0005 (root mean square) and at most 0.0022 with NUTS (10,000 samples), and by 0.003 and at most 0.0125 with
    // Metropolis–Hastings (40,000 samples, the first half dropped). The tolerances are about twice the largest
    // errors, and eight to ten times the typical ones.
    let r = custom_distribution::run(1, 40_000, 10_000).unwrap();
    for (j, (&exact, (&nuts, &mh))) in EXACT_TRUE_RATE_MEANS
        .iter()
        .zip(r.nuts.iter().zip(&r.metropolis_hastings))
        .enumerate()
    {
        assert_close(nuts, exact, 0.005, &format!("NUTS true rate {j}"));
        assert_close(
            mh,
            exact,
            0.025,
            &format!("Metropolis–Hastings true rate {j}"),
        );
    }
}

#[test]
fn custom_distribution_gradient_matches_finite_differences() {
    use keanu::prelude::*;
    // the hyperprior and the rate offsets, without the observations: the gradient with respect to φ includes the
    // offsets' gradient with respect to their parameter
    let b = GraphBuilder::new();
    let phi: DoubleVar = b.custom(custom_distribution::HyperPrior, &[2], &[]);
    let offsets: DoubleVar = b.custom(custom_distribution::RateOffsets, &[3], &[phi.id()]);
    let at_phi = vec![0.3, 2.5];
    let at_offsets = vec![-1.2, 0.4, 2.0];
    b.set_value(phi, DoubleTensor::vector(at_phi.clone()));
    b.set_value(offsets, DoubleTensor::vector(at_offsets.clone()));
    let (phi, offsets) = (phi.id(), offsets.id());
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let gradient = model.log_prob_gradient(&state).unwrap();
    for (node, at) in [(phi, &at_phi), (offsets, &at_offsets)] {
        let analytic = gradient.get(node).unwrap().as_slice().to_vec();
        for i in 0..at.len() {
            let h = 1e-6;
            let log_prob = |delta: f64| {
                let mut s = state.clone();
                let mut v = at.clone();
                v[i] += delta;
                s.set_and_cascade(model.graph(), node, DoubleTensor::vector(v))
                    .unwrap();
                model.log_prob(&s).unwrap()
            };
            let numeric = (log_prob(h) - log_prob(-h)) / (2.0 * h);
            assert_close(analytic[i], numeric, 1e-6, "custom gradient");
        }
    }

    // beyond the truncation the log prob is -infinity
    let mut s = state.clone();
    let beyond = custom_distribution::MAX_LN_SIZE + 1.0;
    s.set_and_cascade(model.graph(), phi, DoubleTensor::vector(vec![0.3, beyond]))
        .unwrap();
    assert_eq!(model.log_prob(&s).unwrap(), f64::NEG_INFINITY);
}

#[test]
fn custom_distribution_that_cannot_be_sampled_reports_an_error() {
    use keanu::prelude::*;
    let b = GraphBuilder::new();
    let hyper: DoubleVar = b.custom(custom_distribution::HyperPrior, &[2], &[]);
    let hyper = hyper.id();
    let (graph, mut state) = b.build().unwrap();
    let error = state
        .lazy_eval(&graph, &[hyper], &mut KeanuRng::seed_from_u64(1))
        .unwrap_err();
    assert!(error.to_string().contains("cannot be sampled"), "{error}");
}

#[test]
fn custom_node_type_must_match_the_handle() {
    use keanu::prelude::*;
    let b = GraphBuilder::new();
    let _wrong: IntVar = b.custom(custom_distribution::HyperPrior, &[2], &[]);
    assert!(b.build().is_err());
}

#[test]
fn eight_schools_non_centred_converges_and_centred_diverges() {
    // AE3, with 4 chains of 1000 draws after 1000 adaptation steps each
    let r = eight_schools::run(1, 1000).unwrap();
    assert!(
        r.centred.divergences > 0,
        "centred eight schools reported no divergences"
    );
    let fit = &r.non_centred;
    assert_eq!(fit.divergences, 0, "non-centred eight schools diverged");
    for row in fit.samples.summary().unwrap().rows() {
        assert!(
            row.stats.r_hat < 1.01,
            "R-hat of {:?}{:?} is {}",
            row.node,
            row.index,
            row.stats.r_hat
        );
    }
    // the published posterior mean of mu is 4.4 (Stan and PyMC agree); its posterior sd is about 3.3 and its Monte
    // Carlo standard error here about 0.06
    let mu = fit.samples.summarize(fit.mu, &[]).unwrap();
    assert_close(mu.mean, 4.4, 1.0, "mu");
}

/// AE3 of the applied modelling gaps plan: the PSIS-LOO elpd of the centred eight schools model agrees with
/// ArviZ's. ArviZ's `centered_eight` example data (the same model and priors, 4 chains of 500 draws from PyMC) gives
/// `az.loo` elpd_loo −30.76 with standard error 1.43 (arviz-stats 1.3.3; ArviZ's documentation shows about −30.7,
/// se 1.3, for older versions of the data). The estimate is asserted within one of those standard errors, and every
/// school has a k̂.
#[test]
fn eight_schools_loo_agrees_with_arviz() {
    let r = eight_schools::run(2, 1000).unwrap();
    for (name, fit) in [("centred", &r.centred), ("non-centred", &r.non_centred)] {
        let loo = &fit.loo;
        assert_eq!(loo.points.len(), 8);
        assert_eq!(loo.points[7].name, "y[7]");
        assert!(
            loo.pareto_k().iter().all(|k| k.is_finite()),
            "{name}: {loo}"
        );
        assert_close(loo.elpd, -30.76, 1.43, &format!("{name} elpd_loo"));
        assert_close(loo.se, 1.43, 0.3, &format!("{name} se"));
    }
}

/// Posterior means of (mu_a, sigma_a, b, sigma_y) and the 85 county intercepts of the radon model, computed
/// independently: the intercepts integrated out analytically (each county's residuals are jointly Gaussian with
/// covariance sigma_y² I + sigma_a² J), random-walk Metropolis on (mu_a, b, ln sigma_a, ln sigma_y) for 400,000
/// steps, and each intercept's mean the average of its conditional posterior mean. Two independent runs agreed to
/// 0.001. They agree with the PyMC radon notebook; Gelman & Hill's REML fit (`lmer`, flat priors) gives mu_a 1.46,
/// b −0.69, sigma_a 0.33 and sigma_y 0.76.
const RADON_HYPERPARAMETER_MEANS: [f64; 4] = [1.4923, 0.3198, -0.6626, 0.7268];
const RADON_COUNTY_MEANS: [f64; radon::COUNTIES] = [
    1.227, 0.982, 1.507, 1.537, 1.473, 1.508, 1.874, 1.705, 1.199, 1.524, //
    1.461, 1.606, 1.275, 1.858, 1.433, 1.276, 1.386, 1.257, 1.38, 1.611, //
    1.652, 1.11, 1.468, 1.878, 1.831, 1.395, 1.644, 1.379, 1.345, 1.14, //
    1.757, 1.395, 1.745, 1.531, 1.128, 1.896, 0.858, 1.653, 1.622, 1.849, //
    1.783, 1.475, 1.575, 1.27, 1.364, 1.372, 1.342, 1.294, 1.655, 1.657, //
    1.789, 1.657, 1.413, 1.367, 1.575, 1.375, 1.128, 1.656, 1.593, 1.44, //
    1.237, 1.736, 1.56, 1.741, 1.448, 1.618, 1.72, 1.271, 1.398, 0.945, //
    1.51, 1.565, 1.58, 1.29, 1.579, 1.718, 1.689, 1.403, 1.148, 1.374, //
    1.928, 1.615, 1.604, 1.614, 1.415,
];

#[test]
fn radon_varying_intercepts_converge_to_the_reference_posterior() {
    let r = radon::run(1, 1000).unwrap();
    assert_eq!(r.divergences, 0, "radon diverged");
    let summary = r.samples.summary().unwrap();
    // the four hyperparameters and the 85 intercepts
    assert_eq!(summary.rows().len(), 4 + radon::COUNTIES);
    for row in summary.rows() {
        assert!(
            row.stats.r_hat < 1.01,
            "R-hat of {:?}{:?} is {}",
            row.node,
            row.index,
            row.stats.r_hat
        );
    }
    // Each mean within a quarter of its posterior sd of the reference. The bulk ESS of every element was above 600
    // over seeds 1 to 6, so the Monte Carlo standard error is below sd / 24 and the tolerance at least six times it.
    let check = |node, index: &[usize], expected: f64, what: &str| {
        let s = summary.get(node, index).unwrap();
        assert_close(s.mean, expected, 0.25 * s.sd, what);
    };
    for (node, expected, what) in [
        (r.mu_a, RADON_HYPERPARAMETER_MEANS[0], "mu_a"),
        (r.sigma_a, RADON_HYPERPARAMETER_MEANS[1], "sigma_a"),
        (r.b, RADON_HYPERPARAMETER_MEANS[2], "b"),
        (r.sigma_y, RADON_HYPERPARAMETER_MEANS[3], "sigma_y"),
    ] {
        check(node, &[], expected, what);
    }
    for (j, &expected) in RADON_COUNTY_MEANS.iter().enumerate() {
        check(r.a, &[j], expected, &format!("county {j} intercept"));
    }
}

#[test]
fn gaussian_mixture_recovers_the_components() {
    use keanu::algorithms::diagnostics::RHAT_THRESHOLD;
    let observations = gaussian_mixture::data(300, 7);
    let fit = gaussian_mixture::run(&observations, 4, 1000, 11).unwrap();
    let chains = &fit.chains;
    for c in 0..2 {
        // each true parameter within four posterior standard deviations (plus the Monte Carlo error) of its
        // posterior mean; the data are one draw, so the posterior is centred near, not on, the truth
        for (node, truth, what) in [
            (fit.means, gaussian_mixture::TRUE_MEANS[c], "mean"),
            (fit.sigmas, gaussian_mixture::TRUE_SIGMA, "sigma"),
            (fit.weights, gaussian_mixture::TRUE_WEIGHTS[c], "weight"),
        ] {
            let stats = chains.summarize(node, &[c]).unwrap();
            let within = 4.0 * stats.sd + 4.0 * stats.mcse_mean;
            assert_close(stats.mean, truth, within, &format!("component {c} {what}"));
            assert!(
                stats.r_hat < RHAT_THRESHOLD,
                "component {c} {what}: R-hat {}",
                stats.r_hat
            );
            assert!(
                stats.ess_bulk > 400.0,
                "component {c} {what}: bulk ESS {}",
                stats.ess_bulk
            );
        }
    }
    // the ordering holds in every draw, so the chains never swap labels
    for chain in chains.chains() {
        let means = chain.f64(fit.means).unwrap();
        assert!(
            means
                .as_slice()
                .iter()
                .all(|m| m.as_slice()[0] < m.as_slice()[1])
        );
    }
}

/// The mixture written with explicit assignments and marginalised automatically recovers the components as the
/// `Mixture` form does, and the recovered assignments put measurements far from the other group in their own.
#[test]
fn gaussian_mixture_with_marginalised_assignments_recovers_the_components() {
    use keanu::algorithms::diagnostics::RHAT_THRESHOLD;
    let observations = gaussian_mixture::data(300, 7);
    let result = gaussian_mixture::run_marginalized(&observations, 4, 500, 13).unwrap();
    let fit = &result.fit;
    for c in 0..2 {
        for (node, truth, what) in [
            (fit.means, gaussian_mixture::TRUE_MEANS[c], "mean"),
            (fit.sigmas, gaussian_mixture::TRUE_SIGMA, "sigma"),
            (fit.weights, gaussian_mixture::TRUE_WEIGHTS[c], "weight"),
        ] {
            let stats = fit.chains.summarize(node, &[c]).unwrap();
            let within = 4.0 * stats.sd + 4.0 * stats.mcse_mean;
            assert_close(stats.mean, truth, within, &format!("component {c} {what}"));
            assert!(
                stats.r_hat < RHAT_THRESHOLD,
                "component {c} {what}: R-hat {}",
                stats.r_hat
            );
        }
    }
    for (i, &y) in observations.iter().enumerate() {
        let row = &result.assignments.as_slice()[2 * i..2 * i + 2];
        assert!((row[0] + row[1] - 1.0).abs() < 1e-9, "measurement {i}");
        if y < -3.5 {
            assert!(row[0] > 0.99, "measurement {y}: {row:?}");
        } else if y > 4.5 {
            assert!(row[1] > 0.99, "measurement {y}: {row:?}");
        }
    }
}

#[test]
fn gp_regression_recovers_the_sine_curve() {
    // 4 chains of 500 draws after 500 adaptation steps each. Over seeds 1 to 10 every grid point was within 2
    // predictive sd of sin(x) (the worst at 1.98 sd, for seed 6) and the largest hyperparameter R-hat was 1.007.
    // With 20 training points instead of 40, half of those seeds had a grid point beyond 2 sd: the GP flattens the
    // sine's peaks, and 2 sd at each of 21 correlated grid points is close to a simultaneous band.
    let r = gp_regression::run(2, 500).unwrap();
    for (name, id) in [
        ("amplitude", r.amplitude),
        ("length scale", r.length_scale),
        ("noise", r.noise),
    ] {
        let s = r.samples.summarize(id, &[]).unwrap();
        assert!(s.r_hat < 1.01, "R-hat of {name} is {}", s.r_hat);
    }
    // the noise was generated with sd 0.2, and its posterior sd is a few hundredths
    let noise = r.samples.summarize(r.noise, &[]).unwrap();
    assert_close(noise.mean, gp_regression::NOISE, 0.1, "noise");
    for ((x, m), s) in r.grid.iter().zip(&r.mean).zip(&r.sd) {
        let truth = gp_regression::truth(*x);
        assert!(
            (m - truth).abs() < 2.0 * s,
            "predictive mean {m} ± {s} at x = {x} vs {truth}"
        );
        // the data pin the function down: the predictive sd is well below the function's amplitude of 1
        assert!(*s < 0.3, "predictive sd {s} at x = {x}");
    }
}

#[test]
fn stochastic_volatility_data_is_reproducible() {
    use stochastic_volatility::{DATA_SEED, DAYS, data, simulate, to_csv};
    let simulated = simulate(DAYS, DATA_SEED);
    if std::env::var_os("KEANU_REGENERATE").is_some() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data/returns.csv");
        std::fs::write(path, to_csv(&simulated)).unwrap();
    }
    let bundled = data();
    assert_eq!(bundled.returns.len(), DAYS);
    for (what, a, b) in [
        (
            "log variance",
            &bundled.log_variance,
            &simulated.log_variance,
        ),
        ("return", &bundled.returns, &simulated.returns),
    ] {
        for (t, (a, b)) in a.iter().zip(b).enumerate() {
            assert_close(*a, *b, 1e-11, &format!("day {t} {what}"));
        }
    }
}

#[test]
fn stochastic_volatility_recovers_the_simulated_parameters() {
    // 4 chains of 500 draws after 500 adaptation steps each, on the 200 bundled days. Over seeds 1 to 7, 0 to 2 of
    // the 2000 draws diverged and the largest R-hat (of the three parameters and the 200 log variances) was 1.0065 to
    // 1.0109; seed 8, with a chain that lingered where phi nears 1, had 44 divergences and an R-hat of 1.023
    // (re-measured on 2026-11-05 with Stan's divergence criterion: unchanged). The Stan User's Guide's
    // parameterisation doesn't reach zero divergences either (see the example's documentation), so the tolerance
    // stays.
    let draws = 500;
    let r = stochastic_volatility::run(1, draws).unwrap();
    // a few trajectories in a thousand diverge where phi nears 1 (see the example's documentation)
    let total = stochastic_volatility::CHAINS * draws;
    assert!(
        r.divergences * 100 < total,
        "{} of {total} draws diverged",
        r.divergences
    );
    // with a bulk effective sample size of about 600, an R-hat's own sampling spread is a few thousandths, and this
    // is the largest of 203; 1.02 allows for that while catching chains that disagree
    let summary = r.samples.summary().unwrap();
    for row in summary.rows() {
        assert!(
            row.stats.r_hat < 1.02,
            "R-hat of {:?}{:?} is {}",
            row.node,
            row.index,
            row.stats.r_hat
        );
    }
    // the data are one draw from the model, so the posterior is centred near, not on, the truth: each true value
    // within four posterior standard deviations (plus the Monte Carlo error) of its posterior mean
    for (node, truth, what) in [
        (r.mu, stochastic_volatility::TRUE_MU, "mu"),
        (r.phi, stochastic_volatility::TRUE_PHI, "phi"),
        (r.sigma, stochastic_volatility::TRUE_SIGMA, "sigma"),
    ] {
        let s = summary.get(node, &[]).unwrap();
        assert_close(s.mean, truth, 4.0 * s.sd + 4.0 * s.mcse_mean, what);
    }
}

#[test]
fn hmm_recovers_the_transition_probabilities() {
    // 4 chains of 1000 draws after 1000 adaptation steps each, on 400 simulated steps. Over seeds 1 to 8 the largest
    // R-hat was 1.003, and every posterior mean of the transition matrix was within 2 posterior sd of the truth.
    let r = hmm::run(1, 1000).unwrap();
    let summary = r.samples.summary().unwrap();
    for row in summary.rows() {
        assert!(
            row.stats.r_hat < 1.01,
            "R-hat of {:?}{:?} is {}",
            row.node,
            row.index,
            row.stats.r_hat
        );
    }
    // each true value within four posterior standard deviations (plus the Monte Carlo error) of its posterior mean
    let check = |node, index: &[usize], truth: f64, what: &str| {
        let s = summary.get(node, index).unwrap();
        assert_close(s.mean, truth, 4.0 * s.sd + 4.0 * s.mcse_mean, what);
    };
    for i in 0..2 {
        for j in 0..2 {
            let truth = hmm::TRUE_TRANSITION[i][j];
            check(r.transition, &[i, j], truth, &format!("P[{i}][{j}]"));
        }
        check(r.means, &[i], hmm::TRUE_MEANS[i], &format!("mean {i}"));
    }
    check(r.sigma, &[], hmm::TRUE_SIGMA, "sigma");
    // the ordering holds in every draw, so the chains never swap labels
    for chain in r.samples.chains() {
        let means = chain.f64(r.means).unwrap();
        assert!(
            means
                .as_slice()
                .iter()
                .all(|m| m.as_slice()[0] < m.as_slice()[1])
        );
    }
}

/// Asserts every R-hat of a summary is below `limit`.
fn assert_r_hat(samples: &keanu::algorithms::ChainSamples, limit: f64, what: &str) {
    for row in samples.summary().unwrap().rows() {
        assert!(
            row.stats.r_hat < limit,
            "{what}: R-hat of {:?}{:?} is {}",
            row.node,
            row.index,
            row.stats.r_hat
        );
    }
}

/// The gradient of the log probability in unconstrained space against central differences, at the model's point
/// and at `points` others (unconstrained, one value per coordinate), to `tolerance` relative.
fn assert_gradient_matches_finite_differences(
    model: &keanu::graph::Model,
    state: &keanu::graph::State,
    points: &[Vec<f64>],
    tolerance: f64,
    what: &str,
) {
    use keanu::graph::Parameterisation;
    let param = Parameterisation::new(model, state).unwrap();
    let mut state = state.clone();
    let mut all = vec![param.from_state(&state).unwrap()];
    all.extend(points.iter().cloned());
    for z in all {
        let (_, gradient) = param.log_prob_and_gradient(&z, true, &mut state).unwrap();
        let gradient = gradient.unwrap();
        for m in 0..z.len() {
            let at = |delta: f64, state: &mut keanu::graph::State| {
                let mut point = z.clone();
                point[m] += delta;
                param.log_prob(&point, true, state).unwrap()
            };
            let h = 1e-5;
            let fd = (at(h, &mut state) - at(-h, &mut state)) / (2.0 * h);
            assert!(
                (gradient[m] - fd).abs() <= tolerance * fd.abs().max(1.0),
                "{what} at {z:?}: d/dz[{m}] is {} by sensitivities, {fd} by finite differences",
                gradient[m]
            );
        }
    }
}

#[test]
fn sir_gradient_matches_finite_differences() {
    // the solver at tight tolerances, so that differences of the log probability resolve its gradient
    let options = keanu::graph::ode::OdeOptions::default().with_tolerances(1e-12, 1e-12);
    let (model, state, _) = sir::model(&sir::BOARDING_SCHOOL, options).unwrap();
    let points = [vec![0.5, -0.6, -1.9], vec![0.2, -0.2, -1.0]];
    assert_gradient_matches_finite_differences(&model, &state, &points, 1e-6, "SIR");
}

#[test]
fn sir_recovers_simulated_transmission_and_recovery_rates() {
    // 4 chains of 300 draws after 300 adaptation steps, on 14 days simulated with beta = 1.7, gamma = 0.55 and
    // phi = 8. Over seeds 1 to 8 the largest R-hat was 1.015, nothing diverged, and both 90% intervals covered the
    // true rates.
    let (beta, gamma) = (1.7, 0.55);
    let cases = sir::simulate(beta, gamma, 8.0, 14, 3).unwrap();
    let r = sir::fit(&cases, sir::CHAINS, 1, 300).unwrap();
    assert_r_hat(&r.samples, 1.02, "SIR on simulated data");
    for (node, truth, what) in [
        (r.nodes.beta, beta, "beta"),
        (r.nodes.gamma, gamma, "gamma"),
    ] {
        let s = r.samples.summarize(node, &[]).unwrap();
        assert!(
            s.q5 <= truth && truth <= s.q95,
            "{what}: {truth} is outside the 90% interval [{}, {}]",
            s.q5,
            s.q95
        );
    }
}

#[test]
fn sir_fits_the_boarding_school_outbreak_as_the_stan_case_study() {
    // 4 chains of 300 draws after 300 adaptation steps. The Stan case study (Grinsztajn et al., 2021) reports
    // posterior means of about 1.73 for beta, 0.54 for gamma, 3.2 for R0 and 1.85 days to recover. Over seeds 1 to
    // 8 the largest R-hat was 1.017, nothing diverged, and the means were 1.734 to 1.739, 0.539 to 0.545, 3.20 to
    // 3.24 and 1.85 to 1.87.
    let r = sir::run(1, 300).unwrap();
    assert_r_hat(&r.samples, 1.02, "SIR on the boarding school data");
    assert!(r.divergences <= 4, "{} divergences", r.divergences);
    let n = r.nodes;
    for (node, expected, within, what) in [
        (n.beta, 1.73, 0.05, "beta"),
        (n.gamma, 0.54, 0.03, "gamma"),
        (n.r0, 3.2, 0.2, "R0"),
        (n.recovery_time, 1.85, 0.1, "recovery time"),
    ] {
        let s = r.samples.summarize(node, &[]).unwrap();
        assert_close(s.mean, expected, within, what);
    }
}

#[test]
fn pharmacokinetics_ode_agrees_with_the_closed_form() {
    use pharmacokinetics::Solution;
    let data = pharmacokinetics::simulate(pharmacokinetics::DATA_SEED).unwrap();
    // the log probabilities and gradients agree to the solver's tolerance at several points
    let (ode, mut ode_state, _) = pharmacokinetics::model(&data, Solution::Ode).unwrap();
    let (analytic, mut analytic_state, _) =
        pharmacokinetics::model(&data, Solution::Analytic).unwrap();
    let ode_param = keanu::graph::Parameterisation::new(&ode, &ode_state).unwrap();
    let analytic_param = keanu::graph::Parameterisation::new(&analytic, &analytic_state).unwrap();
    for z in [
        vec![0.0, 1.1, 3.4, -1.2],
        vec![0.4, 1.0, 3.6, -2.3],
        vec![-0.3, 0.8, 3.2, -0.5],
    ] {
        let (a, ga) = ode_param
            .log_prob_and_gradient(&z, true, &mut ode_state)
            .unwrap();
        let (b, gb) = analytic_param
            .log_prob_and_gradient(&z, true, &mut analytic_state)
            .unwrap();
        assert_close(a, b, 1e-4 * b.abs().max(1.0), "log probability");
        for (i, (x, y)) in ga.unwrap().iter().zip(gb.unwrap()).enumerate() {
            assert_close(*x, y, 1e-4 * y.abs().max(1.0), &format!("gradient {i}"));
        }
    }
    // and their NUTS fits agree, each covering the truth: 4 chains of 300 draws after 300 adaptation steps (over
    // seeds 1 to 8 the largest R-hat was 1.019)
    let r = pharmacokinetics::run(1, 300).unwrap();
    assert_r_hat(&r.ode.samples, 1.02, "PK (ODE)");
    assert_r_hat(&r.analytic.samples, 1.02, "PK (closed form)");
    for (ode, analytic, truth, what) in [
        (
            r.ode.nodes.ka,
            r.analytic.nodes.ka,
            pharmacokinetics::TRUE_KA,
            "ka",
        ),
        (
            r.ode.nodes.cl,
            r.analytic.nodes.cl,
            pharmacokinetics::TRUE_CL,
            "CL",
        ),
        (
            r.ode.nodes.v,
            r.analytic.nodes.v,
            pharmacokinetics::TRUE_V,
            "V",
        ),
        (
            r.ode.nodes.sigma,
            r.analytic.nodes.sigma,
            pharmacokinetics::TRUE_SIGMA,
            "sigma",
        ),
    ] {
        let a = r.ode.samples.summarize(ode, &[]).unwrap();
        let b = r.analytic.samples.summarize(analytic, &[]).unwrap();
        // the same posterior: means within four Monte Carlo standard errors of their difference
        let mcse = (a.mcse_mean.powi(2) + b.mcse_mean.powi(2)).sqrt();
        assert_close(a.mean, b.mean, 4.0 * mcse, what);
        assert_close(a.mean, truth, 4.0 * a.sd + 4.0 * a.mcse_mean, what);
    }
}

#[test]
fn multi_state_recovers_the_simulated_intensities() {
    // 4 chains of 300 draws after 300 adaptation steps, on 300 simulated subjects. Over seeds 1 to 8 every R-hat
    // was below 1.01 and every 90% interval covered its true intensity.
    let r = multi_state::run(1, 300).unwrap();
    assert_r_hat(&r.samples, 1.02, "multi-state");
    let n = r.nodes;
    for (node, truth, what) in [
        (n.illness, multi_state::TRUE_ILLNESS, "q01"),
        (n.healthy_death, multi_state::TRUE_HEALTHY_DEATH, "q02"),
        (n.ill_death, multi_state::TRUE_ILL_DEATH, "q12"),
    ] {
        let s = r.samples.summarize(node, &[]).unwrap();
        assert!(
            s.q5 <= truth && truth <= s.q95,
            "{what}: {truth} is outside the 90% interval [{}, {}]",
            s.q5,
            s.q95
        );
    }
    // the five-year death probability is a function of the intensities: exp(5 Q)[0, 2] at the true ones
    let (q01, q02, q12) = (
        multi_state::TRUE_ILLNESS,
        multi_state::TRUE_HEALTHY_DEATH,
        multi_state::TRUE_ILL_DEATH,
    );
    let q = keanu::tensor::DoubleTensor::new(
        &[3, 3],
        vec![-(q01 + q02), q01, q02, 0.0, -q12, q12, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let truth = (&q * 5.0).matrix_exp().unwrap().as_slice()[2];
    let s = r.samples.summarize(n.dead_in_five_years, &[]).unwrap();
    assert_close(
        s.mean,
        truth,
        4.0 * s.sd + 4.0 * s.mcse_mean,
        "P(dead in 5 years)",
    );
}
