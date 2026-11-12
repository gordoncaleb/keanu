import { Engine, latest } from "./engine.js";
import { bandChart, intervalChart } from "./charts.js";
import { percent, statusLine } from "./format.js";

const engine = new Engine();
const $ = (id) => document.getElementById(id);

const FIT = { draws: 1000, seed: 7, samples: 6 };

const level = (v) => `${v.toFixed(1)} mg/L`;
const hours = (h) => `${+h.toFixed(1)} h`;

const steadyChart = bandChart($("steady-chart"), {
  title: "Blood level over one dosing interval, once doses have built up",
  description: "From one dose to the next, with the target range between the two lines.",
  yFormat: level,
  tickFormat: (v) => `${v}`,
  xFormat: hours,
  xLabel: (h) => `${+h.toFixed(1)} hours after a dose`,
  xHeading: "Hours after a dose",
  referenceLabel: "True level (synthetic patient)",
});
const firstDoseChart = bandChart($("first-dose-chart"), {
  title: "Blood level after the first 300 mg dose",
  description: "The six samples the model learned from, its fit, and the curve the samples were simulated from.",
  yFormat: level,
  tickFormat: (v) => `${v}`,
  xFormat: hours,
  xLabel: (h) => `${+h.toFixed(1)} hours after the dose`,
  xHeading: "Hours after the dose",
  referenceLabel: "True level (synthetic patient)",
  pointsLabel: "Measured samples",
});
const parametersChart = intervalChart($("parameters-chart"), {
  title: "How this patient differs from the typical patient",
  description: "Each quantity as a share of the population's median, for this patient and across the population.",
  format: (v) => `${Math.round(v * 100)}% of typical`,
  tickFormat: (v) => percent(v, 0),
  name: (label) => label,
  rowHeading: "Quantity",
  labelWidth: 190,
  legend: { estimate: "Estimate, with 90% interval", truth: "True value (synthetic patient)" },
});

const state = { fitted: false, population: false };

function dose() {
  const value = Number($("dose").value);
  return Number.isFinite(value) && value > 0 ? value : NaN;
}

function interval() {
  return Number($("interval").value);
}

/** A probability for the sentence, with "under 1%" and "over 99%" at the ends. */
function chance(p) {
  if (p < 0.005) return p === 0 ? "almost no" : "under 1%";
  if (p > 0.995) return "over 99%";
  return percent(p, 0);
}

// "an 8%", "an 11%", "an 18%", "an 80%" and "an under 1%" are read with a vowel sound; "almost no" needs none
function article(text) {
  if (text === "almost no") return "";
  return /^(8|11|18|under|over)/.test(text) ? "an" : "a";
}

function setChance(id, p) {
  const text = chance(p);
  $(id).textContent = text;
  $(`${id}-article`).textContent = article(text);
}

function range(interval) {
  return `likely ${interval.low.toFixed(1)} to ${interval.high.toFixed(1)}`;
}

function small(text) {
  const node = document.createElement("small");
  node.textContent = text;
  return node;
}

async function refresh({ dose, interval, population }) {
  if (!state.fitted) return;
  if (!(dose > 0)) {
    $("answer").setAttribute("data-stale", "");
    return;
  }
  const r = await engine.report(dose, interval, population);
  $("in-range").textContent = percent(r.chosen.time_in_range, 0);
  setChance("low", r.chosen.probability_trough_low);
  setChance("high", r.chosen.probability_peak_high);
  $("suggestion-lead").textContent = population ? "Without the samples," : "For this patient,";
  $("suggestion-verb").textContent = population ? "would look best" : "works best";
  $("suggested").textContent = `${r.suggested.dose} mg`;
  $("suggested-interval").textContent = interval;
  $("answer").removeAttribute("data-stale");

  $("peak").replaceChildren(level(r.peak.mean), small(range(r.peak)));
  $("trough").replaceChildren(level(r.trough.mean), small(range(r.trough)));
  const halfLife = r.parameters.find((p) => p.name === "Half-life").patient;
  $("half-life").replaceChildren(`${halfLife.mean.toFixed(1)} hours`, small(range(halfLife)));
  $("true-in-range").textContent = percent(r.true_time_in_range, 0);
  const clearance = r.parameters.find((p) => p.name === "Clearance");
  $("comparison").textContent = population
    ? `The population alone cannot tell a slow clearer from a fast one, so every dose risks being too low for some ` +
      `patients and too high for others. It would suggest ${r.suggested.dose} mg, in range ` +
      `${percent(r.suggested.time_in_range, 0)} of the time on average.`
    : `This patient clears about ${clearance.patient.mean.toFixed(1)} L an hour, against ` +
      `${clearance.typical.toFixed(1)} for the typical patient, so the drug builds up more than usual. ` +
      `Switch to the population alone to see the decision without the samples.`;

  const targets = [
    { value: r.min_effective, label: `Minimum effective, ${r.min_effective} mg/L` },
    { value: r.toxic, label: `Toxic above ${r.toxic} mg/L`, tone: "threshold" },
  ];
  steadyChart.update(r.steady_state.concentration, [], {
    x: r.steady_state.hours,
    reference: r.steady_state.truth,
    levels: targets,
  });
  firstDoseChart.update(r.first_dose_curve.concentration, [], {
    x: r.first_dose_curve.hours,
    reference: r.first_dose_curve.truth,
    points: r.samples.map((s) => ({ x: s.hours, y: s.concentration })),
  });
  parametersChart.update(
    r.parameters.flatMap((p) => {
      const typical = p.typical;
      const share = (i) => ({ mean: i.mean / typical, low: i.low / typical, high: i.high / typical });
      return [
        { label: `${p.name}, this patient`, ...share(p.patient), truth: p.truth / typical },
        { label: `${p.name}, population`, ...share(p.population) },
      ];
    }),
  );
}
const requestReport = latest(refresh);
const current = () => requestReport({ dose: dose(), interval: interval(), population: state.population });

$("dose").addEventListener("input", current);
$("interval").addEventListener("change", current);
for (const button of document.querySelectorAll("[data-population]")) {
  button.addEventListener("click", () => {
    state.population = button.dataset.population === "true";
    for (const b of document.querySelectorAll("[data-population]")) b.setAttribute("aria-pressed", String(b === button));
    current();
  });
}

async function fit() {
  const status = statusLine($("status"));
  const label = `Learning from ${FIT.samples} blood samples`;
  status.progress(label, 0);
  try {
    const { seconds } = await engine.fit("DosingDemo", [FIT.draws, FIT.seed], {
      steps: 25,
      onProgress: (p) => status.progress(label, p),
    });
    state.fitted = true;
    status.done(
      `Fitted to ${FIT.samples} blood samples in ${seconds.toFixed(1)} s, in your browser, from ` +
        `${FIT.draws.toLocaleString("en-US")} posterior draws. Changing the dose reuses them.`,
    );
    current();
  } catch (error) {
    status.error(`The fit failed: ${error.message}`);
  }
}

fit();
