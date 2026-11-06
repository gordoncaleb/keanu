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
//! The SMC model comparison is checked against the closed-form evidence of its two regressions, and the stochastic
//! SIR fitted by ABC against the rates its outbreak was simulated with.

#![allow(dead_code)]

#[path = "../examples/ab_testing.rs"]
mod ab_testing;
#[path = "../examples/abc_stochastic_sir.rs"]
mod abc_stochastic_sir;
#[path = "../examples/challenger_disaster.rs"]
mod challenger_disaster;
#[path = "../examples/cheating_students.rs"]
mod cheating_students;
#[path = "../examples/co2_forecast.rs"]
mod co2_forecast;
#[path = "../examples/coal_mining_disasters.rs"]
mod coal_mining_disasters;
#[path = "../examples/cure_model.rs"]
mod cure_model;
#[path = "../examples/custom_distribution.rs"]
mod custom_distribution;
#[path = "../examples/disease_mapping.rs"]
mod disease_mapping;
#[path = "../examples/drift_diffusion.rs"]
mod drift_diffusion;
#[path = "../examples/eight_schools.rs"]
mod eight_schools;
#[path = "../examples/extreme_values.rs"]
mod extreme_values;
#[path = "../examples/gaussian_mixture.rs"]
mod gaussian_mixture;
#[path = "../examples/gp_regression.rs"]
mod gp_regression;
#[path = "../examples/hmm.rs"]
mod hmm;
#[path = "../examples/multi_state.rs"]
mod multi_state;
#[path = "../examples/occupancy.rs"]
mod occupancy;
#[path = "../examples/pharmacokinetics.rs"]
mod pharmacokinetics;
#[path = "../examples/radon.rs"]
mod radon;
#[path = "../examples/sir.rs"]
mod sir;
#[path = "../examples/smc_model_comparison.rs"]
mod smc_model_comparison;
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
fn gp_regression_with_an_hsgp_recovers_the_sine_curve() {
    // 4 chains of 500 draws after 500 adaptation steps each, sampling the 25 weights with the hyperparameters. The
    // amplitude is weakly identified by one stretch of a sine (bulk ESS about 400): over seeds 1 to 6 the largest
    // R-hat was 1.012, every grid point was within 2.01 predictive sd of sin(x), the largest predictive sd was 0.083
    // and the noise's posterior mean was between 0.179 and 0.211.
    let r = gp_regression::run_hsgp(1, 500).unwrap();
    for (name, id) in [
        ("amplitude", r.amplitude),
        ("length scale", r.length_scale),
        ("noise", r.noise),
    ] {
        let s = r.samples.summarize(id, &[]).unwrap();
        assert!(s.r_hat < 1.02, "R-hat of {name} is {}", s.r_hat);
    }
    let noise = r.samples.summarize(r.noise, &[]).unwrap();
    assert_close(noise.mean, gp_regression::NOISE, 0.05, "noise");
    for ((x, m), s) in r.grid.iter().zip(&r.mean).zip(&r.sd) {
        let truth = gp_regression::truth(*x);
        assert!(
            (m - truth).abs() < 2.5 * s,
            "HSGP predictive mean {m} ± {s} at x = {x} vs {truth}"
        );
        assert!(*s < 0.15, "HSGP predictive sd {s} at x = {x}");
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
fn co2_forecast_covers_the_held_out_months() {
    // 4 chains of 250 draws after 250 adaptation steps each, on 96 simulated months. Over data seeds 1 to 12 (4
    // chains of 150 draws each), the 90% intervals covered 18 to 24 of the 24 held-out months (94% of them
    // altogether) and the 50% intervals 3 to 19 (56%): calibrated, with the spread of one series whose months are
    // correlated. The bundled seed (3) covered 21 and 14; over chain seeds 1 to 5 with 250 draws the largest R-hat
    // was 1.006 to 1.010, with no divergences.
    let draws = 250;
    let (r, held_out) = co2_forecast::run(1, draws).unwrap();
    assert!(
        r.divergences * 100 < co2_forecast::CHAINS * draws,
        "{} divergences",
        r.divergences
    );
    let summary = r.samples.summary().unwrap();
    for row in summary.rows() {
        assert!(
            row.stats.r_hat < 1.02,
            "R-hat of {:?} is {}",
            row.node,
            row.stats.r_hat
        );
    }
    // the decomposition adds up, and the seasonal cycle has the simulated amplitude of about 3 to 4 ppm
    let parts = &r.decomposition;
    let seasonal = parts.component("seasonal").unwrap();
    for t in 0..co2_forecast::TRAINING_MONTHS {
        let sum: f64 = parts.components.iter().map(|(_, b)| b.mean[t]).sum();
        assert_close(sum, parts.total.mean[t], 1e-9, "trend + seasonal");
    }
    let last_year = &seasonal.mean[co2_forecast::TRAINING_MONTHS - 12..];
    let range = last_year.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - last_year.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!((5.0..9.0).contains(&range), "seasonal range {range}");
    // the 24 held-out months: a calibrated forecast's 90% intervals miss about 2.4 of them and its 50% intervals
    // about 12, but the months of one series are correlated, so the bounds are loose
    let covered_90 = co2_forecast::covered(&r.forecast, &held_out);
    assert!(covered_90 >= 18, "{covered_90} of 24 in the 90% intervals");
    let half = r
        .posterior
        .forecast(co2_forecast::HORIZON, &[], 0.5)
        .unwrap();
    let covered_50 = co2_forecast::covered(&half, &held_out);
    assert!(
        (6..=21).contains(&covered_50),
        "{covered_50} of 24 in the 50% intervals"
    );
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
    // decoding: each step's more probable regime, from the posterior state marginals averaged over the draws,
    // matches the simulated one at almost every step, and the probabilities are probabilities
    let truth = hmm::simulate(hmm::STEPS, hmm::DATA_SEED).states;
    let correct = r.decoded.iter().zip(&truth).filter(|(a, b)| a == b).count();
    assert!(
        correct as f64 >= 0.95 * hmm::STEPS as f64,
        "{correct} correct"
    );
    for p in r.state_probabilities.as_slice().chunks(2) {
        assert!((p[0] + p[1] - 1.0).abs() < 1e-9 && p[0] >= 0.0 && p[1] >= 0.0);
    }
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

/// The NUTS fit of the SIR model to the boarding school data (4 chains of 300 draws after 300 adaptation steps),
/// shared by the tests that check it.
fn sir_boarding_school() -> &'static sir::Sir {
    static FIT: std::sync::OnceLock<sir::Sir> = std::sync::OnceLock::new();
    FIT.get_or_init(|| sir::run(1, 300).unwrap())
}

#[test]
fn sir_fits_the_boarding_school_outbreak_as_the_stan_case_study() {
    // 4 chains of 300 draws after 300 adaptation steps. The Stan case study (Grinsztajn et al., 2021) reports
    // posterior means of about 1.73 for beta, 0.54 for gamma, 3.2 for R0 and 1.85 days to recover. Over seeds 1 to
    // 8 the largest R-hat was 1.017, nothing diverged, and the means were 1.734 to 1.739, 0.539 to 0.545, 3.20 to
    // 3.24 and 1.85 to 1.87.
    let r = sir_boarding_school();
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

/// AE5: the Laplace approximation of the SIR posterior, with its Hessian by finite differences through the ODE,
/// has means and standard deviations within 20% of NUTS's for β, γ, R₀ and the recovery time. With 2000 draws
/// (seed 1) the largest difference was 15%, in β's standard deviation (0.045 against 0.053); the means were within
/// 2%. The dispersion φ's posterior is strongly right-skewed (φ = 1 / φ⁻¹, which the data inform little), which a
/// Gaussian in log φ⁻¹ represents only roughly: its mean was 19% above NUTS's (11.4 against 9.5) and its standard
/// deviation 15% (5.9 against 5.1), so φ is checked to 25%. The fit takes about 60 ms and the draws (each solving
/// the ODE) about 1 s.
#[test]
fn sir_laplace_approximation_agrees_with_nuts() {
    let (laplace, l) = sir::laplace(&sir::BOARDING_SCHOOL, 2000, 1).unwrap();
    let r = sir_boarding_school();
    for (node, laplace_node, within, what) in [
        (r.nodes.beta, l.beta, 0.2, "beta"),
        (r.nodes.gamma, l.gamma, 0.2, "gamma"),
        (r.nodes.r0, l.r0, 0.2, "R0"),
        (r.nodes.recovery_time, l.recovery_time, 0.2, "recovery time"),
        (r.nodes.phi, l.phi, 0.25, "phi"),
    ] {
        let nuts = r.samples.summarize(node, &[]).unwrap();
        let draws = laplace.f64(laplace_node).unwrap();
        let mean = draws.mean().unwrap().as_slice()[0];
        let sd = draws.variance().unwrap().as_slice()[0].sqrt();
        assert!(
            (mean / nuts.mean - 1.0).abs() < within,
            "{what}: Laplace mean {mean}, NUTS {}",
            nuts.mean
        );
        assert!(
            (sd / nuts.sd - 1.0).abs() < within,
            "{what}: Laplace sd {sd}, NUTS {}",
            nuts.sd
        );
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

#[test]
fn occupancy_formulations_recover_psi_and_p_and_agree() {
    // the occupancy latents under the compound sampler, marginalised under NUTS, and the zero-inflated binomial
    // under NUTS: each covers the true ψ and p with its 90% interval and converges, the posterior means agree within
    // their Monte Carlo errors (plus a hundredth for the differences in the forms' adaptation), and so does the
    // posterior number of occupied sites
    let surveys = occupancy::data(3);
    let fits = [
        occupancy::run_latent(&surveys, 4, 500, 11).unwrap(),
        occupancy::run_marginalized(&surveys, 4, 500, 11).unwrap(),
        occupancy::run_zero_inflated(&surveys, 4, 500, 11).unwrap(),
    ];
    let names = ["latent", "marginalised", "zero-inflated"];
    for (fit, name) in fits.iter().zip(names) {
        for (node, truth, what) in [
            (fit.psi, occupancy::TRUE_PSI, "psi"),
            (fit.p, occupancy::TRUE_P, "p"),
        ] {
            let s = fit.chains.summarize(node, &[]).unwrap();
            assert!(s.r_hat < 1.02, "{name} {what}: R-hat {}", s.r_hat);
            assert!(
                s.q5 < truth && truth < s.q95,
                "{name} {what}: {truth} outside [{}, {}]",
                s.q5,
                s.q95
            );
        }
    }
    for (fit, name) in fits[1..].iter().zip(&names[1..]) {
        for (a, b, what) in [(fits[0].psi, fit.psi, "psi"), (fits[0].p, fit.p, "p")] {
            let (x, y) = (
                fits[0].chains.summarize(a, &[]).unwrap(),
                fit.chains.summarize(b, &[]).unwrap(),
            );
            let within = 4.0 * (x.mcse_mean.powi(2) + y.mcse_mean.powi(2)).sqrt() + 0.01;
            assert_close(x.mean, y.mean, within, &format!("{what}: latent vs {name}"));
        }
        assert_close(
            fits[0].occupied_sites,
            fit.occupied_sites,
            2.0,
            &format!("occupied sites: latent vs {name}"),
        );
    }
}

/// Radon's NUTS chains started from multi-path Pathfinder draws converge with a 150-step warm-up (R-hat below 1.01
/// for all 89 parameters; the largest was 1.0078 over seeds 1 to 8). Chains from the default starting values do as
/// well on this model with the same warm-up (they are already near the posterior), so this shows that Pathfinder's
/// starting points work, not that they shorten radon's warm-up. Pathfinder's own approximation is poor here (k̂ of
/// 1.7 to 2.1: the centred model's unconstrained density grows without bound as sigma_a shrinks with the intercepts
/// together, which the L-BFGS path heads for), which is why its draws start the chains rather than replace them.
/// AE6: trust-region Newton and L-BFGS reach the same MAP of the radon model, in its non-centred form (89
/// coordinates, from the default values; the centred form has no MAP, see [`radon::non_centred_model`]).
///
/// Newton needs far fewer gradient evaluations, and its optimum is tighter (the gradient's norm there is about 7e-6
/// against L-BFGS's 3e-2, and its fitness 2e-5 higher), but it is not cheaper overall: each conjugate gradient iteration costs a Hessian-vector product,
/// about two gradient evaluations. Measured from the defaults: L-BFGS used 130 gradient evaluations; Newton 28
/// gradient evaluations and 173 exact Hessian-vector products, about 374 gradient evaluations' worth (17 ms against
/// 44 ms, in release). Newton pays off when gradients are expensive relative to second-order sweeps, or when a
/// tight optimum is needed; L-BFGS remains the default.
#[test]
fn radon_map_by_trust_region_newton_and_lbfgs() {
    use keanu::algorithms::optimizer::{GradientOptimizer, Lbfgs, OptimizedResult, TrustRegion};
    let radon = radon::non_centred_model().unwrap();
    let nodes = [radon.mu_a, radon.sigma_a, radon.b, radon.sigma_y, radon.a];
    let run = |algorithm: keanu::algorithms::optimizer::Algorithm| {
        let mut state = radon.state.clone();
        let result = GradientOptimizer::new(&radon.model)
            .algorithm(algorithm)
            .max_a_posteriori(&mut state)
            .unwrap();
        let param = keanu::graph::Parameterisation::new(&radon.model, &state).unwrap();
        let z = param.from_state(&state).unwrap();
        let (_, gradient) = param.log_prob_and_gradient(&z, false, &mut state).unwrap();
        let gradient_norm = gradient.unwrap().iter().map(|g| g * g).sum::<f64>().sqrt();
        // the state is left at the optimum, with the intercepts computed from it
        let values: Vec<f64> = nodes
            .iter()
            .flat_map(|&id| state.f64(id).unwrap().as_slice().to_vec())
            .collect();
        (result, gradient_norm, values)
    };
    let (lbfgs, lbfgs_gradient, lbfgs_values): (OptimizedResult, f64, Vec<f64>) =
        run(Lbfgs::default().into());
    let (newton, newton_gradient, newton_values) = run(TrustRegion::default().into());
    assert!(lbfgs.converged && newton.converged);
    assert_eq!(lbfgs.hessian_vector_products, 0);
    // the same optimum, which Newton resolves more tightly
    assert!((newton.fitness - lbfgs.fitness).abs() < 1e-6 * lbfgs.fitness.abs());
    assert!(newton.fitness >= lbfgs.fitness);
    assert!(newton_gradient < 1e-4 && newton_gradient * 100.0 < lbfgs_gradient);
    for (x, y) in lbfgs_values.iter().zip(&newton_values) {
        assert!((x - y).abs() < 1e-2 * y.abs().max(1.0), "{x} vs {y}");
    }
    // each trial point costs one fitness and one gradient evaluation
    let (lbfgs_gradients, newton_gradients) = (lbfgs.evaluations / 2, newton.evaluations / 2);
    assert!(
        newton_gradients * 3 < lbfgs_gradients,
        "{newton_gradients} vs {lbfgs_gradients} gradient evaluations"
    );
    // counting an exact Hessian-vector product as two gradient evaluations, Newton costs more here
    let newton_cost = newton_gradients + 2 * newton.hessian_vector_products;
    assert!(
        newton_cost > lbfgs_gradients,
        "{newton_cost} vs {lbfgs_gradients} gradient evaluations' worth"
    );
}

#[test]
fn radon_from_pathfinder_converges_with_a_short_warm_up() {
    let (r, pareto_k) = radon::run_from_pathfinder(2, 150, 1000).unwrap();
    assert_eq!(r.divergences, 0, "radon diverged");
    let summary = r.samples.summary().unwrap();
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
    assert!(pareto_k > 0.7, "k̂ {pareto_k}");
}

#[test]
fn smc_model_comparison_matches_the_closed_form_evidence() {
    // 4 runs of 1000 particles per model. Every estimate was within 0.03 of its closed form with 2000 particles,
    // and the log Bayes factor of the quadratic over the linear model is about 8.
    let [linear, quadratic] = smc_model_comparison::run(1000, 1).unwrap();
    for fit in [linear, quadratic] {
        assert_close(
            fit.log_evidence,
            fit.exact_log_evidence,
            0.15,
            &format!("log evidence of degree {}", fit.degree),
        );
    }
    let log_bayes_factor = quadratic.log_evidence - linear.log_evidence;
    let exact = quadratic.exact_log_evidence - linear.exact_log_evidence;
    assert_close(log_bayes_factor, exact, 0.2, "log Bayes factor");
    assert!(exact > 5.0, "{exact}");
    // PSIS-LOO on the particles agrees that the quadratic model predicts better
    assert!(
        quadratic.elpd_loo > linear.elpd_loo,
        "{} vs {}",
        quadratic.elpd_loo,
        linear.elpd_loo
    );
}

#[test]
fn abc_recovers_the_stochastic_sir_rates() {
    // 4 runs of 1000 particles on an outbreak simulated with beta = 1 and gamma = 0.3. Over data seeds 1 to 8 both
    // 90% intervals covered the true rates and every R-hat was below 1.002.
    let (beta, gamma) = abc_stochastic_sir::TRUTH;
    let counts = abc_stochastic_sir::observed(2);
    let fit = abc_stochastic_sir::fit(&counts, 1000, 2).unwrap();
    // the intervals must also be informative: a third of the prior's width or less (2.8 for beta, 0.95 for gamma)
    for (node, truth, width, what) in [
        (fit.nodes.beta, beta, 0.9, "beta"),
        (fit.nodes.gamma, gamma, 0.3, "gamma"),
    ] {
        let s = fit.samples.summarize(node, &[]).unwrap();
        assert!(
            s.q5 <= truth && truth <= s.q95 && s.q95 - s.q5 < width,
            "{what}: {truth}, 90% interval [{}, {}]",
            s.q5,
            s.q95
        );
        assert!(s.r_hat < 1.01, "{what}: R-hat {}", s.r_hat);
    }
}

#[test]
fn disease_mapping_recovers_the_bym2_parameters_and_the_covariate_effect() {
    // BYM2 Poisson counts on a 10 × 10 lattice simulated from known values: the 90% intervals cover σ, ρ and both
    // coefficients, every parameter converges, and the posterior mean area effects follow the true ones
    let areas = disease_mapping::data(6).unwrap();
    let fit = disease_mapping::run(&areas, 4, 2000, 7).unwrap();
    for (node, truth, what) in [
        (fit.beta0, disease_mapping::TRUE_BETA[0], "beta0"),
        (fit.beta1, disease_mapping::TRUE_BETA[1], "beta1"),
        (fit.sigma, disease_mapping::TRUE_SIGMA, "sigma"),
        (fit.rho, disease_mapping::TRUE_RHO, "rho"),
    ] {
        let s = fit.chains.summarize(node, &[]).unwrap();
        assert!(s.r_hat < 1.01, "{what}: R-hat {}", s.r_hat);
        assert!(
            s.q5 < truth && truth < s.q95,
            "{what}: {truth} outside [{}, {}]",
            s.q5,
            s.q95
        );
    }
    let correlation = disease_mapping::effect_correlation(&areas, &fit).unwrap();
    assert!(
        correlation > 0.9,
        "the effects' correlation with the truth is {correlation}"
    );
}

#[test]
fn drift_diffusion_recovers_the_condition_drifts_and_each_subjects_parameters() {
    // response times of four subjects in two conditions simulated from known values: the 90% intervals cover both
    // population drifts, the starting point and every subject's boundary separation and non-decision time; every
    // parameter (each subject's drifts included) converges, with no divergences; and the easy condition's drift is
    // clearly the larger
    use drift_diffusion::{SUBJECTS, TRUE_A, TRUE_DRIFT, TRUE_T0, TRUE_W};
    let trials = drift_diffusion::data(drift_diffusion::DATA_SEED).unwrap();
    let fit = drift_diffusion::run(&trials, 4, 1000, 2).unwrap();
    assert_eq!(fit.divergences, 0);
    let mut checks = vec![
        (fit.drift, vec![0], TRUE_DRIFT[0], "hard drift".to_owned()),
        (fit.drift, vec![1], TRUE_DRIFT[1], "easy drift".to_owned()),
        (fit.w, vec![], TRUE_W, "w".to_owned()),
    ];
    for s in 0..SUBJECTS {
        checks.push((fit.a, vec![s], TRUE_A[s], format!("a[{s}]")));
        checks.push((fit.t0, vec![s], TRUE_T0[s], format!("t0[{s}]")));
    }
    for (node, index, truth, what) in checks {
        let summary = fit.chains.summarize(node, &index).unwrap();
        assert!(summary.r_hat < 1.01, "{what}: R-hat {}", summary.r_hat);
        assert!(
            summary.q5 < truth && truth < summary.q95,
            "{what}: {truth} outside [{}, {}]",
            summary.q5,
            summary.q95
        );
    }
    for s in 0..SUBJECTS {
        for c in 0..2 {
            let summary = fit.chains.summarize(fit.subject_drift, &[s, c]).unwrap();
            assert!(
                summary.r_hat < 1.01,
                "drift[{s}, {c}]: R-hat {}",
                summary.r_hat
            );
        }
    }
    let hard = fit.chains.summarize(fit.drift, &[0]).unwrap();
    let easy = fit.chains.summarize(fit.drift, &[1]).unwrap();
    assert!(
        easy.q5 > hard.q95,
        "easy {:?}, hard {:?}",
        (easy.q5, easy.q95),
        (hard.q5, hard.q95)
    );
}

#[test]
fn extreme_values_recovers_the_gev_parameters_and_return_levels() {
    // sixty annual maxima simulated from a known GEV: the 90% intervals cover μ, σ, ξ and the 10-, 50- and 100-year
    // return levels, every quantity converges, and NUTS rarely meets the parameter-dependent support's edge
    use extreme_values::{PERIODS, TRUE_MU, TRUE_SIGMA, TRUE_XI, return_level};
    let maxima = extreme_values::data(extreme_values::DATA_SEED);
    let fit = extreme_values::run(&maxima, 4, 1000, 1).unwrap();
    assert!(fit.divergences <= 40, "{} divergences", fit.divergences);
    let mut checks = vec![
        (fit.mu, vec![], TRUE_MU, "mu".to_owned()),
        (fit.sigma, vec![], TRUE_SIGMA, "sigma".to_owned()),
        (fit.xi, vec![], TRUE_XI, "xi".to_owned()),
    ];
    for (i, &period) in PERIODS.iter().enumerate() {
        let truth = return_level(TRUE_MU, TRUE_SIGMA, TRUE_XI, period);
        checks.push((
            fit.return_levels,
            vec![i],
            truth,
            format!("{period}-year return level"),
        ));
    }
    for (node, index, truth, what) in checks {
        let summary = fit.chains.summarize(node, &index).unwrap();
        assert!(summary.r_hat < 1.01, "{what}: R-hat {}", summary.r_hat);
        assert!(
            summary.q5 < truth && truth < summary.q95,
            "{what}: {truth} outside [{}, {}]",
            summary.q5,
            summary.q95
        );
    }
    // the return levels increase with the period, in every draw
    let draws = fit.chains.element(fit.return_levels, &[0]).unwrap();
    let longer = fit.chains.element(fit.return_levels, &[2]).unwrap();
    for (short, long) in draws.iter().zip(&longer) {
        assert!(short.iter().zip(long).all(|(s, l)| s < l));
    }
}

#[test]
fn cure_model_recovers_the_cure_fraction_and_the_relapse_times() {
    // relapse times of 400 patients, right-censored, simulated from a mixture cure model: the 90% intervals cover the
    // cure model's coefficients, the Weibull scale's coefficients and the shape, with no divergences and R-hat below
    // 1.01
    use cure_model::{TRUE_BETA, TRUE_GAMMA, TRUE_K};
    let patients = cure_model::data(cure_model::DATA_SEED);
    let fit = cure_model::run(&patients, 4, 1000, 1).unwrap();
    assert_eq!(fit.divergences, 0);
    let checks = [
        (fit.gamma, vec![0], TRUE_GAMMA[0], "gamma0"),
        (fit.gamma, vec![1], TRUE_GAMMA[1], "gamma1"),
        (fit.beta, vec![0], TRUE_BETA[0], "beta0"),
        (fit.beta, vec![1], TRUE_BETA[1], "beta1"),
        (fit.k, vec![], TRUE_K, "k"),
    ];
    for (node, index, truth, what) in checks {
        let summary = fit.chains.summarize(node, &index).unwrap();
        assert!(summary.r_hat < 1.01, "{what}: R-hat {}", summary.r_hat);
        assert!(
            summary.q5 < truth && truth < summary.q95,
            "{what}: {truth} outside [{}, {}]",
            summary.q5,
            summary.q95
        );
    }
}
