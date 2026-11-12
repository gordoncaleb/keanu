import { Engine, latest } from "./engine.js";
import { curveChart, histogramChart, intervalChart } from "./charts.js";
import { money, parseAmount, percent, statusLine } from "./format.js";

const engine = new Engine();
const $ = (id) => document.getElementById(id);

const FIT = { units: 500, draws: 500, seed: 11 };

const count = (v) => Math.round(v).toLocaleString("en-US");
const months = (v) => `${v.toFixed(0)} months`;

const survivalChart = curveChart($("survival-chart"), {
  title: "Share of pumps still running, by age",
  description: "For the fleet's load mix. The true curve is the one the synthetic fleet was generated with.",
  height: 260,
  xFormat: (v) => `${v}`,
  xLabel: (v) => `${v} months`,
  xHeading: "Age in months",
  tableStep: 6,
  yFormat: (v) => percent(v, 1),
  tickFormat: (v) => percent(v, 0),
  lines: [
    { field: "truth", key: "truth", label: "True curve" },
    { field: "naive", key: "compare", label: "Ignoring running pumps" },
  ],
});
const claimsChart = histogramChart($("claims-chart"), {
  title: "Warranty claims on the new fleet",
  description: "Across the posterior and simulated failures. Bars past the budget are highlighted.",
  height: 220,
  xFormat: (v) => count(v),
  binHeading: "Claims",
  binLabel: (lo, hi) => `${count(Math.ceil(lo))} to ${count(Math.floor(hi))} claims`,
});
const lifeChart = intervalChart($("life-chart"), {
  title: "Age by which 1 in 10 pumps fails",
  description: "The B10 life, against the truth and a fit that ignores the pumps still running.",
  format: (v) => months(v),
  tickFormat: (v) => `${v}`,
  name: (load) => `${load}, model`,
  rowHeading: "Load",
  labelWidth: 104,
  legend: { observed: "Ignoring running pumps", truth: "True value (synthetic data)" },
});

const fitted = { done: false };

function inputs() {
  const warranty = Number($("warranty").value);
  const fleet = Math.round(parseAmount($("fleet").value));
  const cost = parseAmount($("cost").value);
  const budget = parseAmount($("budget").value);
  const ok =
    Number.isFinite(warranty) && warranty >= 0 && warranty <= 120 &&
    Number.isFinite(fleet) && fleet >= 0 && fleet <= 1e7 &&
    Number.isFinite(cost) && cost >= 0 &&
    Number.isFinite(budget) && budget >= 0;
  return ok ? { warranty, fleet, cost, budget } : null;
}

async function refresh(args) {
  if (!fitted.done) return;
  if (!args) {
    $("answer").setAttribute("data-stale", "");
    return;
  }
  const { warranty, fleet, cost, budget } = args;
  const r = await engine.report(warranty, fleet, cost, budget);
  $("claims").textContent = count(r.expected_claims);
  $("claims-range").textContent = `${count(r.claims_low)} to ${count(r.claims_high)}`;
  $("cost-total").textContent = money(r.expected_cost);
  const p = r.probability_over_budget;
  const chance = p > 0.999 ? "more than 99.9%" : p < 0.001 ? "less than 0.1%" : percent(p, p < 0.1 ? 1 : 0);
  $("over").textContent = chance;
  // "an 8%", "an 11%", "an 80%", "an 18%" are read with a vowel sound
  $("over-article").textContent = /^(8|11|18)/.test(chance) ? "an" : "a";
  $("answer").removeAttribute("data-stale");
  $("naive-claims").textContent = `${count(r.naive_expected_claims)}, against ${count(r.true_expected_claims)} in truth`;
  survivalChart.update(
    r.survival.map((s) => ({ x: s.month, mean: s.mean, low: s.low, high: s.high, truth: s.truth, naive: s.naive })),
    [{ value: warranty, label: "Warranty ends", tone: "threshold" }],
  );
  const budgetClaims = cost > 0 ? budget / cost : Infinity;
  claimsChart.update(
    r.claims_histogram,
    [
      { value: r.expected_claims, label: "Expected" },
      { value: budgetClaims, label: "Budget", tone: "threshold" },
    ],
    (lo, hi) => hi > budgetClaims,
  );
}
const requestReport = latest(refresh);
const current = () => requestReport(inputs());

for (const id of ["warranty", "fleet", "cost", "budget"]) $(id).addEventListener("input", current);
for (const id of ["fleet", "cost", "budget"]) {
  $(id).addEventListener("blur", () => {
    const value = parseAmount($(id).value);
    if (value >= 0) $(id).value = value.toLocaleString("en-US");
  });
}

async function fit() {
  const status = statusLine($("status"));
  const label = `Learning pump lifetimes from ${FIT.units} pumps`;
  status.progress(label, 0);
  try {
    const { seconds } = await engine.fit("ReliabilityDemo", [FIT.units, FIT.draws, FIT.seed], {
      steps: 25,
      onProgress: (p) => status.progress(label, p),
    });
    fitted.done = true;
    status.done(
      `Fitted to ${FIT.units} pumps in ${seconds.toFixed(1)} s, in your browser, from ${FIT.draws} posterior ` +
        `draws. Changing the warranty reuses them.`,
    );
    const base = await engine.report(12, 1000, 1, 0);
    $("history").textContent = `${base.units}, of which ${base.failed} have failed`;
    const fleet = base.lives.find((l) => l.load === "Whole fleet").b10;
    $("b10").textContent = `${months(fleet.mean)}, likely ${fleet.low.toFixed(0)} to ${fleet.high.toFixed(0)}`;
    $("shape").textContent = `${base.shape.mean.toFixed(2)}, likely ${base.shape.low.toFixed(2)} to ${base.shape.high.toFixed(2)}`;
    $("load").textContent = `${percent(base.high_load_factor.mean, 0)} as long`;
    lifeChart.update(
      base.lives.map((l) => ({
        label: l.load,
        mean: l.b10.mean,
        low: l.b10.low,
        high: l.b10.high,
        observed: l.naive_b10,
        truth: l.true_b10,
      })),
    );
    current();
  } catch (error) {
    status.error(`The fit failed: ${error.message}`);
  }
}

fit();
