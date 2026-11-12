// Number formatting, input parsing and the fit status line, shared by the demo pages.

const dollars = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const compactDollars = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  notation: "compact",
  maximumFractionDigits: 1,
});

/** $12,345 (or $12.3K compact); with `signed`, a leading + or − (a minus sign, not a hyphen). */
export function money(value, { signed = false, compact = false } = {}) {
  const format = compact ? compactDollars : dollars;
  const text = format.format(Math.abs(value));
  if (!signed || Math.round(value) === 0) return value < 0 && !signed ? `−${text}` : text;
  return `${value < 0 ? "−" : "+"}${text}`;
}

/** 12.3% from 0.123. */
export function percent(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

/** A positive amount typed with optional $, commas and k/m suffixes; NaN if unreadable. */
export function parseAmount(text) {
  const match = String(text).trim().toLowerCase().replace(/[$,\s]/g, "").match(/^(\d*\.?\d+)([km]?)$/);
  if (!match) return NaN;
  return Number(match[1]) * { "": 1, k: 1e3, m: 1e6 }[match[2]];
}

/**
 * Daily returns from pasted text: one number per line (the last number on a line, so CSV rows work) or comma
 * separated. All-positive values above 1 are taken as prices and become log returns; otherwise values with
 * magnitudes above 1 are taken as percentages.
 */
export function parseSeries(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim());
  let values;
  if (lines.length === 1) values = lines[0].split(/[,;\s]+/).map(Number);
  else
    values = lines.map((line) => {
      const numbers = line.match(/-?\d*\.?\d+(?:e-?\d+)?/gi);
      return numbers ? Number(numbers.at(-1)) : NaN;
    });
  const skipped = values.filter((v) => !Number.isFinite(v)).length;
  values = values.filter(Number.isFinite);
  if (values.length < 2) throw new Error("Paste at least 30 numbers: daily closing prices or daily returns.");
  const skippedNote = skipped ? ` Skipped ${skipped} line${skipped === 1 ? "" : "s"} without a number.` : "";
  if (values.every((v) => v > 1)) {
    const returns = values.slice(1).map((v, i) => Math.log(v / values[i]));
    return { returns, note: `Read ${values.length} prices as ${returns.length} daily log returns.${skippedNote}` };
  }
  if (values.some((v) => Math.abs(v) > 1)) {
    return { returns: values.map((v) => v / 100), note: `Read ${values.length} returns in percent.${skippedNote}` };
  }
  return { returns: values, note: `Read ${values.length} returns as fractions (0.01 is 1%).${skippedNote}` };
}

/** Drives a status paragraph: a progress bar with a label, then a sentence, or an error. */
export function statusLine(node) {
  return {
    progress(label, value) {
      node.removeAttribute("data-state");
      let bar = node.querySelector("progress");
      if (!bar) {
        node.replaceChildren();
        bar = document.createElement("progress");
        bar.max = 1;
        const text = document.createElement("span");
        node.append(bar, text);
      }
      bar.value = value;
      node.querySelector("span").textContent = `${label}, ${Math.round(value * 100)}%`;
    },
    done(text) {
      node.removeAttribute("data-state");
      node.replaceChildren(text);
    },
    error(text) {
      node.dataset.state = "error";
      node.replaceChildren(text);
    },
  };
}
