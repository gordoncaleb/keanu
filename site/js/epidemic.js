import { Engine, latest } from "./engine.js";
import { bandChart, histogramChart } from "./charts.js";
import { percent, statusLine } from "./format.js";

const engine = new Engine();
const $ = (id) => document.getElementById(id);

const FIT = { days: 35, draws: 500, seed: 3 };

const count = (v) => Math.round(v).toLocaleString("en-US");
const absoluteDay = (i) => `${i + 1}`;

const casesChart = bandChart($("cases-chart"), {
  title: "Reported cases a day",
  description: "The model's reported cases, with reporting noise, against the 35 days observed, then forecast.",
  yFormat: count,
  dayLabel: absoluteDay,
  observedLabel: "Reported",
});
const occupancyChart = bandChart($("occupancy-chart"), {
  title: "Hospital beds in use",
  description: "Each day's patients in hospital, across the plausible outbreaks, against the beds available.",
  yFormat: count,
  dayLabel: absoluteDay,
});
const peakChart = histogramChart($("peak-chart"), {
  title: "Peak number of patients",
  description: "The highest number of beds in use at once, across the plausible outbreaks.",
  height: 220,
  xFormat: count,
  binHeading: "Peak patients",
  binLabel: (lo, hi) => `${count(lo)} to ${count(hi)} patients`,
});

const fitted = { done: false };

function beds() {
  const value = Math.round(Number($("beds").value));
  return Number.isFinite(value) && value >= 0 ? value : 0;
}

const cut = () => Number($("cut").value);
const fromDay = () => Number($("from-day").value);

function describeIntervention() {
  const c = cut();
  $("cut-value").textContent = c > 0 ? `${percent(c, 0)} fewer contacts` : "No change in contacts";
  $("from-day-value").textContent = `Day ${fromDay()}`;
}

function chance(p) {
  return p > 0.999 ? "more than 99.9%" : p < 0.001 ? "less than 0.1%" : percent(p, p < 0.1 ? 1 : 0);
}

/** A posterior mean, with its 90% interval and the true value underneath. */
function showEstimate(node, interval, format, truth) {
  const note = document.createElement("small");
  note.textContent = `90% interval ${format(interval.low)} to ${format(interval.high)}, true ${truth}`;
  node.replaceChildren(format(interval.mean), note);
}

async function refresh({ beds, cut, fromDay }) {
  if (!fitted.done) return;
  const r = await engine.report(beds, cut, fromDay);
  const fill = chance(r.probability_exceeds);
  $("fill").textContent = fill;
  // "an 8%", "an 11%", "an 80%", "an 18%" are read with a vowel sound
  $("fill-article").textContent = /^(8|11|18)/.test(fill) ? "an" : "a";
  const peakDay = Math.round(r.peak_day.mean);
  $("peak-day").textContent = peakDay >= r.occupancy.length ? `day ${peakDay} or later` : `day ${peakDay}`;
  $("peak").textContent = count(r.peak_occupancy.mean);
  $("answer").removeAttribute("data-stale");

  $("days-over").textContent = r.days_over_capacity.toFixed(r.days_over_capacity < 10 ? 1 : 0);
  $("without").textContent = chance(r.probability_exceeds_without);
  const attack = document.createElement("small");
  attack.textContent = `90% interval ${percent(r.attack_rate.low, 0)} to ${percent(r.attack_rate.high, 0)}`;
  $("attack").replaceChildren(percent(r.attack_rate.mean, 0), attack);

  const days = r.days_observed;
  casesChart.update(r.cases.slice(0, days), r.cases.slice(days), { observed: r.observed });
  occupancyChart.update(r.occupancy.slice(0, days), r.occupancy.slice(days), {
    limit: { value: beds, label: `${count(beds)} beds` },
  });
  peakChart.update(
    r.peak_histogram,
    [
      { value: beds, label: `${count(beds)} beds`, tone: "threshold" },
      ...(cut > 0 ? [{ value: r.peak_occupancy_without.mean, label: "No change in contacts" }] : []),
    ],
    (lo, hi) => hi > beds,
  );
}
const requestReport = latest(refresh);
const current = () => requestReport({ beds: beds(), cut: cut(), fromDay: fromDay() });

$("beds").addEventListener("input", current);
for (const id of ["cut", "from-day"]) {
  $(id).addEventListener("input", () => {
    describeIntervention();
    current();
  });
}

async function fit() {
  const status = statusLine($("status"));
  const label = `Learning how the outbreak spreads from ${FIT.days} days of cases`;
  status.progress(label, 0);
  try {
    const { seconds } = await engine.fit("EpidemicDemo", [FIT.days, FIT.draws, FIT.seed], {
      steps: 25,
      onProgress: (p) => status.progress(label, p),
    });
    fitted.done = true;
    status.done(
      `Fitted to ${FIT.days} days of reported cases in ${seconds.toFixed(1)} s, in your browser, from ` +
        `${FIT.draws} posterior draws. Changing the beds or the contacts reuses them.`,
    );
    const base = await engine.report(beds(), cut(), fromDay());
    const truth = base.truth;
    showEstimate($("r0"), base.r0, (v) => v.toFixed(2), truth.r0.toFixed(1));
    showEstimate($("recovery"), base.recovery_days, (v) => v.toFixed(1), truth.recovery_days.toFixed(0));
    showEstimate($("reporting"), base.reporting, (v) => percent(v, 0), percent(truth.reporting, 0));
    current();
  } catch (error) {
    status.error(`The fit failed: ${error.message}`);
  }
}

describeIntervention();
fit();
