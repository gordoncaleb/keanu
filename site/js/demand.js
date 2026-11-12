import { Engine, latest } from "./engine.js";
import { bandChart, histogramChart, intervalChart } from "./charts.js";
import { percent, statusLine } from "./format.js";

const engine = new Engine();
const $ = (id) => document.getElementById(id);

const FIT = { periods: 13, draws: 300, seed: 1 };
const MAX_PERIOD = 56;

const units = (v) => Math.round(v).toLocaleString("en-US");
const unitsLabel = (v) => `${units(v)} ${Math.round(v) === 1 ? "unit" : "units"}`;
const times = (v) => `×${v.toFixed(2)}`;

const salesChart = bandChart($("sales-chart"), {
  title: "Daily sales: a year of history, then the next four weeks",
  description: "Expected sales on each day of the history, then the forecast of each day's sales.",
  yFormat: (v) => units(v),
  tickFormat: (v) => units(v),
});
const demandChart = histogramChart($("demand-chart"), {
  title: "Total demand over the period",
  description: "Across the posterior and simulated futures. The bars past the stock line are the outcomes that run short.",
  height: 220,
  xFormat: (v) => units(v),
  binHeading: "Units",
  binLabel: (lo, hi) => `${units(lo)} to ${units(hi)} units`,
});
const weekdayChart = intervalChart($("weekday-chart"), {
  title: "What the history taught the model about the week",
  description: "Each day's sales as a multiple of an average day, against the history and the true pattern.",
  format: times,
  tickFormat: (v) => `×${v.toFixed(1)}`,
  labelWidth: 92,
  fromZero: false,
  name: (day) => `${day}, model`,
  rowHeading: "Day",
  legend: { truth: "True effect (synthetic data)" },
});

const fitted = { done: false };

function period() {
  const value = Math.round(Number($("period").value));
  return Number.isFinite(value) ? Math.min(MAX_PERIOD, Math.max(1, value)) : 28;
}

function onHand() {
  const value = Math.round(Number($("on-hand").value));
  return Number.isFinite(value) && value >= 0 ? value : 0;
}

async function refresh({ period, service }) {
  if (!fitted.done) return;
  const r = await engine.report(period, service);
  render(r);
}

function render(r) {
  $("stock").textContent = units(r.stock);
  $("expected").textContent = units(r.expected_demand);
  $("safety").textContent = units(r.safety_stock);
  $("order").textContent = unitsLabel(Math.max(0, r.stock - onHand()));
  $("short").textContent = r.expected_short < 0.5 ? "less than a unit" : unitsLabel(r.expected_short);
  $("left-over").textContent = units(r.expected_left_over);
  $("answer").removeAttribute("data-stale");
  $("range").textContent = `${units(r.demand.low)} to ${units(r.demand.high)} units`;
  $("fill").textContent = percent(r.fill_rate, 1);
  demandChart.update(
    r.demand_histogram,
    [
      { value: r.expected_demand, label: "Expected" },
      { value: r.actual_demand, label: "Actually sold" },
      { value: r.stock, label: "Stock", tone: "threshold" },
    ],
    (lo) => lo >= r.stock,
  );
}

const requestReport = latest(refresh);
const current = () => requestReport({ period: period(), service: Number($("service").value) });

$("period").addEventListener("input", current);
$("service").addEventListener("change", current);
$("on-hand").addEventListener("input", current);

async function fit() {
  const status = statusLine($("status"));
  const label = "Learning demand from a year of daily sales";
  status.progress(label, 0);
  try {
    const { seconds } = await engine.fit("DemandDemo", [FIT.periods, FIT.draws, FIT.seed], {
      steps: 25,
      onProgress: (p) => status.progress(label, p),
    });
    fitted.done = true;
    const r = await engine.report(period(), Number($("service").value));
    status.done(
      `Fitted to 364 days of sales in ${seconds.toFixed(1)} s, in your browser, from ${FIT.draws} posterior draws ` +
        `and ${r.futures.toLocaleString("en-US")} simulated futures. Changing the period or service level reuses them.`,
    );
    const g = r.growth;
    const signed = (v) => `${v < 0 ? "−" : "+"}${percent(Math.abs(v), 0)}`;
    $("growth").textContent = `${signed(g.mean)}, likely ${signed(g.low)} to ${signed(g.high)}`;
    const busiest = r.weekdays.reduce((a, b) => (b.multiplier.mean > a.multiplier.mean ? b : a));
    const above = busiest.multiplier.mean - 1;
    $("busiest").textContent = `${busiest.day}, ${percent(above, 0)} above average`;
    salesChart.update(r.history, r.forecast);
    weekdayChart.update(
      r.weekdays.map((w) => ({
        label: w.day,
        mean: w.multiplier.mean,
        low: w.multiplier.low,
        high: w.multiplier.high,
        observed: w.observed,
        truth: w.truth,
      })),
    );
    render(r);
  } catch (error) {
    status.error(`The fit failed: ${error.message}`);
  }
}

fit();
