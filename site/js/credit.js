import { Engine, latest } from "./engine.js";
import { histogramChart, intervalChart } from "./charts.js";
import { money, percent, statusLine } from "./format.js";

const engine = new Engine();
const $ = (id) => document.getElementById(id);

const FIT = { historyLoans: 3000, bookLoans: 250, draws: 500, seed: 7 };

const defaultsChart = histogramChart($("defaults-chart"), {
  title: "Number of defaults on the book",
  description: "How many of the open loans default, across the posterior and simulated outcomes.",
  height: 220,
  xFormat: (v) => `${Math.round(v)}`,
  binHeading: "Defaults",
  binLabel: (lo) => `${Math.round(lo)} defaults`,
});
const lossChart = histogramChart($("loss-chart"), {
  title: "Credit losses",
  description: "45% of the amount outstanding on each defaulted loan.",
  height: 220,
  xFormat: (v) => money(v, { compact: true }),
  binHeading: "Loss",
  binLabel: (lo, hi) => `${money(lo)} to ${money(hi)}`,
});
const gradesChart = intervalChart($("grades-chart"), {
  title: "What the history taught the model, in a normal year",
  description: "Default probability by grade at average debt-to-income, against the history and the true rates.",
  format: (v) => percent(v, 1),
  tickFormat: (v) => percent(v, 0),
  name: (grade) => `Grade ${grade}, model`,
  rowHeading: "Grade",
});

const fitted = { done: false };

function stress() {
  return Number($("stress").value);
}

function threshold() {
  const value = Math.round(Number($("threshold").value));
  return Number.isFinite(value) && value >= 0 ? value : 0;
}

function describeStress(s) {
  const odds = Math.exp(s);
  $("stress-value").textContent =
    Math.abs(s) < 1e-9 ? "No stress" : `Default odds ×${odds.toFixed(2)}`;
}

async function refresh({ stress, threshold }) {
  if (!fitted.done) return;
  const r = await engine.report(stress, threshold);
  $("defaults").textContent = r.expected_defaults.mean.toFixed(0);
  const p = r.probability_defaults_exceed;
  const chance = p > 0.999 ? "more than 99.9%" : p < 0.001 ? "less than 0.1%" : percent(p, p < 0.1 ? 1 : 0);
  $("exceed").textContent = chance;
  // "an 8%", "an 11%", "an 80%", "an 18%" are read with a vowel sound
  $("exceed-article").textContent = /^(8|11|18)/.test(chance) ? "an" : "a";
  $("expected-loss").textContent = money(r.expected_loss);
  $("var").textContent = money(r.value_at_risk_99);
  $("answer").removeAttribute("data-stale");
  $("exposure").textContent = money(r.book_exposure);
  $("loss-share").textContent = percent(r.expected_loss / r.book_exposure, 1);
  $("es").textContent = money(r.expected_shortfall_99);
  defaultsChart.update(
    r.defaults_histogram,
    [{ value: threshold + 1, label: `More than ${threshold}`, tone: "threshold" }],
    (lo) => lo > threshold + 0.5,
  );
  lossChart.update(
    r.loss_histogram,
    [
      { value: r.expected_loss, label: "Average" },
      { value: r.value_at_risk_99, label: "1 year in 100", tone: "threshold" },
    ],
    (lo) => lo >= r.value_at_risk_99,
  );
}
const requestReport = latest(refresh);
const current = () => requestReport({ stress: stress(), threshold: threshold() });

$("scenario").addEventListener("change", () => {
  const value = $("scenario").value;
  if (value !== "custom") $("stress").value = value;
  describeStress(stress());
  current();
});
$("stress").addEventListener("input", () => {
  const s = stress();
  const preset = [...$("scenario").options].find((o) => o.value !== "custom" && Math.abs(Number(o.value) - s) < 1e-9);
  $("scenario").value = preset ? preset.value : "custom";
  $("scenario").querySelector('[value="custom"]').hidden = Boolean(preset);
  describeStress(s);
  current();
});
$("threshold").addEventListener("input", current);

async function fit() {
  const status = statusLine($("status"));
  status.progress("Learning default rates from 3,000 past loans", 0);
  try {
    const { seconds } = await engine.fit(
      "CreditDemo",
      [FIT.historyLoans, FIT.bookLoans, FIT.draws, FIT.seed],
      { steps: 25, onProgress: (p) => status.progress("Learning default rates from 3,000 past loans", p) },
    );
    fitted.done = true;
    status.done(
      `Fitted to ${FIT.historyLoans.toLocaleString("en-US")} past loans in ${seconds.toFixed(1)} s, in your ` +
        `browser, from ${FIT.draws} posterior draws. Changing the scenario reuses them.`,
    );
    $("book-size").textContent = FIT.bookLoans;
    const base = await engine.report(0, threshold());
    $("dti").textContent = `×${base.dti_odds_ratio.mean.toFixed(2)}`;
    gradesChart.update(
      base.grades.map((g) => ({
        label: g.grade,
        mean: g.default_probability.mean,
        low: g.default_probability.low,
        high: g.default_probability.high,
        observed: g.observed_rate,
        truth: g.true_probability,
      })),
    );
    current();
  } catch (error) {
    status.error(`The fit failed: ${error.message}`);
  }
}

describeStress(stress());
fit();
