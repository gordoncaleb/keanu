import { Engine, latest } from "./engine.js";
import { histogramChart, intervalChart } from "./charts.js";
import { percent, statusLine } from "./format.js";

const engine = new Engine();
const $ = (id) => document.getElementById(id);

const DRAWS = 1000;

const liftChart = histogramChart($("lift-chart"), {
  title: "How much better B is",
  description: "B's conversion rate relative to A's, across the posterior. Right of the line, B is better.",
  height: 220,
  xFormat: (v) => signedPercent(v, 0),
  binHeading: "Lift",
  binLabel: (lo, hi) => `${signedPercent(lo, 1)} to ${signedPercent(hi, 1)}`,
});
const ratesChart = intervalChart($("rates-chart"), {
  title: "Each version's conversion rate",
  description: "What the test says about each rate, against the raw share of visitors who converted.",
  format: (v) => percent(v, 2),
  tickFormat: (v) => percent(v, 2),
  legend: { estimate: "Model, with 90% interval", observed: "Observed conversion share" },
  name: (version) => `Version ${version}`,
  rowHeading: "Version",
  labelWidth: 40,
  fromZero: false,
});

function signedPercent(v, digits) {
  const text = percent(Math.abs(v), digits);
  return v > 0 ? `+${text}` : v < 0 ? `−${text}` : text;
}

/** Percentage points, the unit of a difference between two rates. */
function points(v) {
  return `${(v * 100).toFixed(v < 0.0001 ? 4 : v < 0.001 ? 3 : 2)} points`;
}

function arms() {
  const read = (id) => Math.round(Number($(id).value));
  return {
    visitorsA: read("visitors-a"),
    conversionsA: read("conversions-a"),
    visitorsB: read("visitors-b"),
    conversionsB: read("conversions-b"),
  };
}

function invalid({ visitorsA, conversionsA, visitorsB, conversionsB }) {
  for (const [name, visitors, conversions] of [
    ["A", visitorsA, conversionsA],
    ["B", visitorsB, conversionsB],
  ]) {
    if (!(visitors >= 1)) return `Version ${name} needs at least one visitor.`;
    if (!(conversions >= 0) || conversions > visitors)
      return `Version ${name}'s conversions must be between 0 and its ${visitors.toLocaleString("en-US")} visitors.`;
  }
  return null;
}

const status = statusLine($("status"));

async function run(input) {
  const problem = invalid(input);
  if (problem) {
    $("answer").setAttribute("data-stale", "");
    status.error(problem);
    return;
  }
  $("answer").setAttribute("data-stale", "");
  const { seconds } = await engine.fit(
    "AbTestDemo",
    [input.visitorsA, input.conversionsA, input.visitorsB, input.conversionsB, DRAWS, 1],
    { steps: 200 },
  );
  const r = await engine.report();
  const p = r.probability_b_better;
  const chance = p > 0.999 ? "more than 99.9%" : p < 0.001 ? "less than 0.1%" : percent(p, p < 0.1 || p > 0.9 ? 1 : 0);
  $("better").textContent = chance;
  $("better-article").textContent = /^(8|11|18)/.test(chance) ? "an" : "a";
  $("lift").textContent = signedPercent(r.lift.mean, 0);
  $("answer").removeAttribute("data-stale");
  status.done(`Refitted in ${(seconds * 1000).toFixed(0)} ms, in your browser, from ${r.draws} posterior draws.`);

  $("rate-a").textContent = percent(r.rate_a.mean, 2);
  $("rate-b").textContent = percent(r.rate_b.mean, 2);
  $("loss-b").textContent = points(r.expected_loss_b);
  $("loss-a").textContent = points(r.expected_loss_a);
  const [safer, riskier] = r.expected_loss_b < r.expected_loss_a ? ["B", "A"] : ["A", "B"];
  const ratio = Math.max(r.expected_loss_a, r.expected_loss_b) / Math.max(Math.min(r.expected_loss_a, r.expected_loss_b), 1e-12);
  $("decision").textContent =
    ratio < 2
      ? "Neither version has a clearly smaller expected loss yet. More visitors would settle it."
      : `Choosing ${safer} risks ${ratio > 100 ? "over 100" : Math.round(ratio)} times less lost conversion than ` +
        `choosing ${riskier}.`;

  liftChart.update(r.lift_histogram, [{ value: 0, label: "No difference" }], (lo) => lo >= 0);
  ratesChart.update([
    { label: "A", ...r.rate_a, observed: input.conversionsA / input.visitorsA },
    { label: "B", ...r.rate_b, observed: input.conversionsB / input.visitorsB },
  ]);
}

const refit = latest(async (input) => {
  try {
    await run(input);
  } catch (error) {
    status.error(`The fit failed: ${error.message}`);
  }
});

for (const id of ["visitors-a", "conversions-a", "visitors-b", "conversions-b"]) {
  $(id).addEventListener("input", () => refit(arms()));
}
refit(arms());
