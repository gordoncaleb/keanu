import { Engine, latest } from "./engine.js";
import { bandChart, histogramChart } from "./charts.js";
import { money, percent, parseAmount, parseSeries, statusLine } from "./format.js";

const engine = new Engine();
const $ = (id) => document.getElementById(id);

const DRAWS = { garch: 500, stochastic: 300 };
const HORIZON = 20;

const state = { model: "garch", returns: null, fitted: false, report: null };

const volatilityChart = bandChart($("volatility-chart"), {
  title: "Annualised volatility, then a 20-day forecast",
  description: "Each observed day, and the days ahead, with the model's uncertainty.",
  yFormat: (v) => percent(v, 1),
  tickFormat: (v) => percent(v, 0),
});
const lossChart = histogramChart($("loss-chart"), {
  title: "Tomorrow's profit and loss",
  description: "Simulated from the posterior. Losses to the right; the bars past the line are the rare days.",
  height: 230,
  xFormat: (v) => money(-v, { signed: true, compact: true }),
  binHeading: "Profit or loss",
  binLabel: (lo, hi) => `${money(-hi, { signed: true })} to ${money(-lo, { signed: true })}`,
});

function position() {
  return parseAmount($("position").value);
}

async function refresh() {
  if (!state.fitted) return;
  const value = position();
  if (!(value > 0)) {
    $("answer").setAttribute("data-stale", "");
    return;
  }
  state.report = await engine.report(value, HORIZON);
  render();
}
const requestReport = latest(refresh);

function render() {
  const r = state.report;
  if (!r) return;
  const tail = r.tail[Number($("level").value)];
  $("var").textContent = money(tail.value_at_risk);
  $("es").textContent = money(tail.expected_shortfall);
  $("answer").removeAttribute("data-stale");
  const today = r.volatility.at(-1).mean;
  $("today").textContent = percent(today, 1);
  $("long-run").replaceChildren(
    percent(r.long_run_volatility.mean, 1),
    small(`likely ${percent(r.long_run_volatility.low, 0)} to ${percent(r.long_run_volatility.high, 0)}`),
  );
  $("half-life").replaceChildren(
    `${r.half_life_days.mean.toFixed(0)}`,
    small(`likely ${r.half_life_days.low.toFixed(0)} to ${r.half_life_days.high.toFixed(0)}`),
  );
  $("naive").textContent = money(tail.naive_value_at_risk);
  const ratio = tail.value_at_risk / tail.naive_value_at_risk;
  const calmer = today < r.long_run_volatility.mean;
  $("comparison").textContent =
    Math.abs(ratio - 1) < 0.05
      ? "Here the normal model happens to agree: today's volatility is close to the sample's average."
      : `The normal model's ${money(tail.naive_value_at_risk)} uses the whole sample's volatility. The fitted model says ` +
        `${money(tail.value_at_risk)}, ${Math.round(Math.abs(1 - ratio) * 100)}% ${ratio < 1 ? "less" : "more"}, ` +
        `because volatility is ${calmer ? "calmer" : "higher"} right now than its long-run level.`;
  volatilityChart.update(r.volatility, r.forecast.map((d) => d.volatility));
  lossChart.update(
    r.loss_histogram,
    [{ value: tail.value_at_risk, label: `1 day in ${$("level").selectedOptions[0].textContent}`, tone: "threshold" }],
    (lo) => lo >= tail.value_at_risk,
  );
}

function small(text) {
  const node = document.createElement("small");
  node.textContent = text;
  return node;
}

async function fit() {
  state.fitted = false;
  $("answer").setAttribute("data-stale", "");
  const status = statusLine($("status"));
  const name = state.model === "garch" ? "GARCH(1,1)" : "Stochastic volatility";
  status.progress(`Fitting ${name}`, 0);
  try {
    const stochastic = state.model === "stochastic";
    const { seconds } = await engine.fit(
      "VolatilityDemo",
      [new Float64Array(state.returns), stochastic, DRAWS[state.model], 1],
      { steps: stochastic ? 2 : 50, onProgress: (p) => status.progress(`Fitting ${name}`, p) },
    );
    state.fitted = true;
    status.done(
      `${name} fitted to ${state.returns.length} days in ${seconds.toFixed(seconds < 1 ? 2 : 1)} s, ` +
        `in your browser, from ${DRAWS[state.model]} posterior draws.`,
    );
    await refresh();
  } catch (error) {
    status.error(`The fit failed: ${error.message}`);
  }
}

for (const button of document.querySelectorAll("[data-model]")) {
  button.addEventListener("click", () => {
    if (state.model === button.dataset.model) return;
    state.model = button.dataset.model;
    for (const b of document.querySelectorAll("[data-model]")) b.setAttribute("aria-pressed", String(b === button));
    fit();
  });
}

$("position").addEventListener("input", () => requestReport());
$("position").addEventListener("blur", () => {
  const value = position();
  if (value > 0) $("position").value = value.toLocaleString("en-US");
});
$("level").addEventListener("change", render);

$("fit-returns").addEventListener("click", () => {
  const message = $("data-message");
  try {
    const { returns, note } = parseSeries($("returns").value);
    if (returns.length < 30) throw new Error(`That gives ${returns.length} daily returns; the models need at least 30.`);
    state.returns = returns;
    message.removeAttribute("data-tone");
    message.textContent = note;
    fit();
  } catch (error) {
    message.dataset.tone = "error";
    message.textContent = error.message;
  }
});

$("use-sample").addEventListener("click", async () => {
  state.returns = await engine.call("VolatilityDemo", "sampleReturns");
  $("data-message").removeAttribute("data-tone");
  $("data-message").textContent = "Back to the simulated sample.";
  fit();
});

state.returns = await engine.call("VolatilityDemo", "sampleReturns");
fit();
