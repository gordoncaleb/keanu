// Small SVG charts drawn against the page's CSS tokens. Each chart re-renders to its container's width, has a
// hover and keyboard tooltip, and keeps a table view so no value depends on hovering.

const SVG = "http://www.w3.org/2000/svg";

function el(name, attributes = {}, parent) {
  const node = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  parent?.appendChild(node);
  return node;
}

function html(name, attributes = {}, parent) {
  const node = document.createElement(name);
  for (const [key, value] of Object.entries(attributes)) {
    if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  parent?.appendChild(node);
  return node;
}

/** Round, readable axis ticks between lo and hi. */
function ticks(lo, hi, count = 5) {
  const span = hi - lo || Math.abs(hi) || 1;
  const raw = span / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => span / s <= count) ?? magnitude * 10;
  const first = Math.ceil(lo / step) * step;
  const out = [];
  for (let v = first; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(12));
  return out;
}

/** Ticks from 0 that reach at least `hi`, for a value axis whose top must clear the data. */
function coverTicks(hi, count) {
  const out = ticks(0, hi, count);
  if (out.length < 2) return [0, hi || 1];
  const step = out[1] - out[0];
  while (out.at(-1) < hi - step * 1e-9) out.push(+(out.at(-1) + step).toFixed(12));
  return out;
}

function scale(d0, d1, r0, r1) {
  const k = (r1 - r0) / (d1 - d0 || 1);
  return (v) => r0 + (v - d0) * k;
}

/** A chart frame: a figure with a title, an SVG that tracks the container width, a tooltip and a table view. */
class Frame {
  constructor(container, { title, description, height = 260 }) {
    this.container = container;
    this.height = height;
    container.replaceChildren();
    container.classList.add("chart");
    const head = html("div", { class: "chart-head" }, container);
    html("h3", { text: title }, head);
    if (description) html("p", { class: "chart-note", text: description }, head);
    this.legend = html("div", { class: "legend" }, container);
    this.plot = html("div", { class: "plot" }, container);
    this.tooltip = html("div", { class: "tooltip", role: "status", hidden: "" }, this.plot);
    this.details = html("details", { class: "table-view" }, container);
    html("summary", { text: "Show as a table" }, this.details);
    this.table = html("div", { class: "table-scroll" }, this.details);
    this.render = () => {};
    new ResizeObserver(() => this.render()).observe(this.plot);
  }

  svg() {
    this.plot.querySelector("svg")?.remove();
    const width = Math.max(this.plot.clientWidth, 280);
    const svg = el("svg", { width, height: this.height, viewBox: `0 0 ${width} ${this.height}`, role: "img" });
    this.plot.prepend(svg);
    return { svg, width, height: this.height };
  }

  /** Shows the tooltip at plot coordinates (x, y): a strong value line, then secondary lines. */
  show(x, y, rows) {
    const tip = this.tooltip;
    tip.replaceChildren();
    for (const row of rows) {
      const line = html("div", { class: "tip-row" }, tip);
      if (row.key) html("span", { class: `tip-key ${row.key}` }, line);
      html("strong", { text: row.value }, line);
      if (row.label) html("span", { text: row.label }, line);
    }
    tip.hidden = false;
    const width = this.plot.clientWidth;
    const left = Math.min(Math.max(x + 14, 0), width - tip.offsetWidth - 4);
    tip.style.left = `${x + 14 + tip.offsetWidth > width ? x - tip.offsetWidth - 14 : left}px`;
    tip.style.top = `${Math.max(y - tip.offsetHeight - 8, 0)}px`;
  }

  hide() {
    this.tooltip.hidden = true;
  }

  setLegend(items) {
    this.legend.replaceChildren();
    for (const item of items) {
      const entry = html("span", { class: "legend-item" }, this.legend);
      html("span", { class: `legend-key ${item.key}` }, entry);
      html("span", { text: item.label }, entry);
    }
  }

  setTable(columns, rows) {
    const table = html("table");
    const head = html("tr", {}, html("thead", {}, table));
    for (const column of columns) html("th", { text: column, scope: "col" }, head);
    const body = html("tbody", {}, table);
    for (const row of rows) {
      const tr = html("tr", {}, body);
      for (const cell of row) html("td", { text: cell }, tr);
    }
    this.table.replaceChildren(table);
  }
}

function axes(svg, { left, right, top, bottom, x, y, xTicks, yTicks, xFormat, yFormat }) {
  const grid = el("g", { class: "grid" }, svg);
  for (const t of yTicks) {
    el("line", { x1: left, x2: right, y1: y(t), y2: y(t) }, grid);
    el("text", { x: left - 8, y: y(t), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, grid)
      .textContent = yFormat(t);
  }
  el("line", { x1: left, x2: right, y1: bottom, y2: bottom, class: "baseline" }, svg);
  for (const t of xTicks) {
    el("text", { x: x(t), y: bottom + 18, class: "tick", "text-anchor": "middle" }, svg).textContent = xFormat(t);
  }
  return grid;
}

/**
 * A mean line inside a 90% band over days, continuing into a forecast after a "today" rule.
 * `history` and `forecast` are arrays of { mean, low, high }.
 */
export function bandChart(container, options) {
  const frame = new Frame(container, options);
  let data = null;
  frame.setLegend([
    { key: "line", label: "Posterior mean" },
    { key: "band", label: "90% interval" },
  ]);
  frame.render = () => {
    if (!data) return;
    const { history, forecast } = data;
    const all = [...history, ...forecast];
    const { svg, width, height } = frame.svg();
    const pad = { left: 52, right: 16, top: 16, bottom: 30 };
    const n = all.length;
    const x = scale(0, n - 1, pad.left, width - pad.right);
    const hi = Math.max(...all.map((d) => d.high));
    const yTicks = coverTicks(hi, 4);
    const y = scale(0, yTicks.at(-1), height - pad.bottom, pad.top);
    const dayTicks = ticks(0, n - 1, Math.max(3, Math.floor(width / 110))).filter((t) => Number.isInteger(t));
    axes(svg, {
      left: pad.left, right: width - pad.right, top: pad.top, bottom: height - pad.bottom,
      x, y, xTicks: dayTicks, yTicks, xFormat: (t) => (t < history.length ? `${t + 1}` : `+${t - history.length + 1}`),
      yFormat: options.tickFormat ?? options.yFormat,
    });
    const area = (rows, offset) =>
      rows.map((d, i) => `${x(i + offset)},${y(d.high)}`).join(" ") +
      " " +
      rows.map((d, i) => `${x(i + offset)},${y(d.low)}`).reverse().join(" ");
    const line = (rows, offset) => rows.map((d, i) => `${i ? "L" : "M"}${x(i + offset)},${y(d.mean)}`).join("");
    el("polygon", { points: area(history, 0), class: "band" }, svg);
    // the forecast starts from the last observed day so the two bands join
    const joined = [history.at(-1), ...forecast];
    el("polygon", { points: area(joined, history.length - 1), class: "band forecast" }, svg);
    el("path", { d: line(history, 0), class: "series-line" }, svg);
    el("path", { d: line(joined, history.length - 1), class: "series-line forecast" }, svg);
    const today = x(history.length - 1);
    el("line", { x1: today, x2: today, y1: pad.top, y2: height - pad.bottom, class: "rule" }, svg);
    const label = el("text", { x: today + 6, y: pad.top + 10, class: "annotation" }, svg);
    label.textContent = "Forecast";
    if (today + 6 + label.getComputedTextLength() > width - pad.right) {
      label.setAttribute("x", today - 6);
      label.setAttribute("text-anchor", "end");
    }

    const cross = el("line", { y1: pad.top, y2: height - pad.bottom, class: "crosshair", visibility: "hidden" }, svg);
    const dot = el("circle", { r: 4, class: "crosshair-dot", visibility: "hidden" }, svg);
    const hit = el("rect", {
      x: pad.left, y: pad.top, width: width - pad.left - pad.right, height: height - pad.top - pad.bottom,
      class: "hit", tabindex: 0, "aria-label": `${options.title}: use the arrow keys to read each day`,
    }, svg);
    let focused = history.length - 1;
    const at = (i) => {
      focused = Math.max(0, Math.min(n - 1, i));
      const d = all[focused];
      cross.setAttribute("x1", x(focused));
      cross.setAttribute("x2", x(focused));
      dot.setAttribute("cx", x(focused));
      dot.setAttribute("cy", y(d.mean));
      cross.setAttribute("visibility", "visible");
      dot.setAttribute("visibility", "visible");
      const day = focused < history.length ? `Day ${focused + 1}` : `${focused - history.length + 1} days ahead`;
      frame.show(x(focused), y(d.high), [
        { key: "line", value: options.yFormat(d.mean), label: day },
        { value: `${options.yFormat(d.low)} to ${options.yFormat(d.high)}`, label: "90% interval" },
      ]);
    };
    const away = () => {
      cross.setAttribute("visibility", "hidden");
      dot.setAttribute("visibility", "hidden");
      frame.hide();
    };
    hit.addEventListener("pointermove", (e) => {
      const box = svg.getBoundingClientRect();
      const px = e.clientX - box.left;
      at(Math.round(((px - pad.left) / (width - pad.left - pad.right)) * (n - 1)));
    });
    hit.addEventListener("pointerleave", away);
    hit.addEventListener("focus", () => at(focused));
    hit.addEventListener("blur", away);
    hit.addEventListener("keydown", (e) => {
      const move = { ArrowLeft: -1, ArrowRight: 1, Home: -n, End: n }[e.key];
      if (move) {
        e.preventDefault();
        at(focused + move);
      }
    });
  };
  return {
    update(history, forecast) {
      data = { history, forecast };
      frame.render();
      frame.setTable(
        ["Day", "Mean", "5%", "95%"],
        [
          ...history.map((d, i) => [`${i + 1}`, options.yFormat(d.mean), options.yFormat(d.low), options.yFormat(d.high)]),
          ...forecast.map((d, i) => [`+${i + 1}`, options.yFormat(d.mean), options.yFormat(d.low), options.yFormat(d.high)]),
        ],
      );
    },
  };
}

/**
 * A histogram { start, width, fraction } with vertical markers [{ value, label }]. Bars for which `emphasis(lo, hi)`
 * is true use the series color; the rest are muted.
 */
export function histogramChart(container, options) {
  const frame = new Frame(container, options);
  let data = null;
  frame.render = () => {
    if (!data) return;
    const { histogram, markers, emphasis } = data;
    const { svg, width, height } = frame.svg();
    const pad = { left: 52, right: 16, top: 34, bottom: 30 };
    const lo = histogram.start[0];
    const hi = histogram.start.at(-1) + histogram.width;
    const x = scale(lo, hi, pad.left, width - pad.right);
    const top = Math.max(...histogram.fraction);
    const yTicks = coverTicks(top, 3);
    const y = scale(0, yTicks.at(-1), height - pad.bottom, pad.top);
    axes(svg, {
      left: pad.left, right: width - pad.right, top: pad.top, bottom: height - pad.bottom, x, y,
      xTicks: ticks(lo, hi, Math.max(3, Math.floor(width / 120))), yTicks,
      xFormat: options.xFormat, yFormat: (t) => `${+(t * 100).toFixed(1)}%`,
    });
    const bars = el("g", {}, svg);
    histogram.fraction.forEach((f, i) => {
      const b0 = histogram.start[i];
      const b1 = b0 + histogram.width;
      const left = x(b0) + 1;
      const barWidth = Math.max(x(b1) - x(b0) - 2, 1);
      const group = el("g", { class: "bar-group", tabindex: f > 0 ? 0 : -1 }, bars);
      el("rect", { x: x(b0), y: pad.top, width: x(b1) - x(b0), height: height - pad.top - pad.bottom, class: "hit" }, group);
      if (f > 0) {
        const barHeight = Math.max(height - pad.bottom - y(f), 1);
        // a 4px rounded data end anchored to the baseline
        const r = Math.min(4, barWidth / 2, barHeight);
        const yTop = height - pad.bottom - barHeight;
        el("path", {
          d: `M${left},${height - pad.bottom}V${yTop + r}Q${left},${yTop} ${left + r},${yTop}H${left + barWidth - r}` +
            `Q${left + barWidth},${yTop} ${left + barWidth},${yTop + r}V${height - pad.bottom}Z`,
          class: emphasis?.(b0, b1) ? "bar" : "bar muted",
        }, group);
      }
      const show = () =>
        frame.show(x(b0 + histogram.width / 2), y(f), [
          { value: `${+(f * 100).toFixed(2)}%`, label: options.binLabel(b0, b1) },
        ]);
      group.addEventListener("pointerenter", show);
      group.addEventListener("focus", show);
      group.addEventListener("pointerleave", () => frame.hide());
      group.addEventListener("blur", () => frame.hide());
    });
    // markers, labelled at the top, staggered when they would collide
    let lastLabelEnd = -Infinity;
    let row = 0;
    for (const marker of [...markers].sort((a, b) => a.value - b.value)) {
      if (marker.value < lo || marker.value > hi) continue;
      const mx = x(marker.value);
      el("line", { x1: mx, x2: mx, y1: pad.top - 4, y2: height - pad.bottom, class: `marker ${marker.tone ?? ""}` }, svg);
      const label = el("text", { x: mx + 4, y: 0, class: "annotation" }, svg);
      label.textContent = marker.label;
      const length = label.getComputedTextLength();
      row = mx < lastLabelEnd + 6 ? row + 1 : 0;
      label.setAttribute("y", 12 + row * 13);
      if (mx + 4 + length > width - pad.right) {
        label.setAttribute("x", mx - 4);
        label.setAttribute("text-anchor", "end");
      }
      lastLabelEnd = mx + 4 + length;
    }
  };
  return {
    update(histogram, markers, emphasis) {
      data = { histogram, markers, emphasis };
      frame.render();
      frame.setTable(
        [options.binHeading, "Probability"],
        histogram.fraction
          .map((f, i) => [options.binLabel(histogram.start[i], histogram.start[i] + histogram.width), `${(f * 100).toFixed(2)}%`])
          .filter((_, i) => histogram.fraction[i] > 0),
      );
    },
  };
}

/**
 * Horizontal 90% intervals per row { label, low, mean, high, observed?, truth? }, with optional observed (ring) and
 * true (tick) marks. `options.legend` overrides the legend labels; `options.name(label)` names a row in tooltips.
 */
export function intervalChart(container, options) {
  const frame = new Frame(container, { ...options, height: 0 });
  const labels = {
    estimate: "Model, with 90% interval",
    observed: "Observed in the history",
    truth: "True rate (synthetic data)",
    ...options.legend,
  };
  const name = options.name ?? ((label) => label);
  let rows = null;
  frame.render = () => {
    if (!rows) return;
    frame.height = 36 + rows.length * 40;
    const { svg, width, height } = frame.svg();
    const pad = { left: options.labelWidth ?? 64, right: 16, top: 8, bottom: 28 };
    const values = rows.flatMap((r) => [r.low, r.high, r.observed, r.truth].filter((v) => v !== undefined));
    const hi = Math.max(...values);
    const count = Math.max(3, Math.floor(width / 110));
    let xTicks;
    if (options.fromZero === false) {
      // zoom to the data, with a little room either side
      const lo = Math.min(...values);
      const margin = (hi - lo) * 0.08 || Math.abs(hi) * 0.1 || 1;
      xTicks = ticks(lo - margin, hi + margin, count);
    } else {
      xTicks = coverTicks(hi, count);
    }
    const x = scale(Math.min(xTicks[0], ...values), Math.max(xTicks.at(-1), hi), pad.left, width - pad.right);
    const grid = el("g", { class: "grid" }, svg);
    for (const t of xTicks) {
      el("line", { x1: x(t), x2: x(t), y1: pad.top, y2: height - pad.bottom }, grid);
      el("text", { x: x(t), y: height - pad.bottom + 18, class: "tick", "text-anchor": "middle" }, grid).textContent =
        (options.tickFormat ?? options.format)(t);
    }
    rows.forEach((r, i) => {
      const cy = pad.top + 20 + i * 40;
      const group = el("g", { class: "bar-group", tabindex: 0 }, svg);
      el("rect", { x: 0, y: cy - 20, width, height: 40, class: "hit" }, group);
      el("text", { x: pad.left - 12, y: cy, class: "row-label", "text-anchor": "end", "dominant-baseline": "middle" }, group)
        .textContent = r.label;
      el("line", { x1: x(r.low), x2: x(r.high), y1: cy, y2: cy, class: "whisker" }, group);
      if (r.truth !== undefined) el("line", { x1: x(r.truth), x2: x(r.truth), y1: cy - 9, y2: cy + 9, class: "truth" }, group);
      if (r.observed !== undefined) el("circle", { cx: x(r.observed), cy, r: 5, class: "observed" }, group);
      el("circle", { cx: x(r.mean), cy, r: 5, class: "estimate" }, group);
      const show = () => {
        const tip = [
          { key: "line", value: options.format(r.mean), label: name(r.label) },
          { value: `${options.format(r.low)} to ${options.format(r.high)}`, label: "90% interval" },
        ];
        if (r.observed !== undefined) tip.push({ value: options.format(r.observed), label: labels.observed.toLowerCase() });
        if (r.truth !== undefined) tip.push({ value: options.format(r.truth), label: labels.truth.toLowerCase() });
        frame.show(x(r.high), cy, tip);
      };
      group.addEventListener("pointerenter", show);
      group.addEventListener("focus", show);
      group.addEventListener("pointerleave", () => frame.hide());
      group.addEventListener("blur", () => frame.hide());
    });
  };
  return {
    update(next) {
      rows = next;
      const hasObserved = rows.some((r) => r.observed !== undefined);
      const hasTruth = rows.some((r) => r.truth !== undefined);
      frame.setLegend([
        { key: "dot", label: labels.estimate },
        ...(hasObserved ? [{ key: "ring", label: labels.observed }] : []),
        ...(hasTruth ? [{ key: "tick", label: labels.truth }] : []),
      ]);
      frame.render();
      frame.setTable(
        [options.rowHeading ?? "", "Model", "5%", "95%", ...(hasObserved ? [labels.observed] : []), ...(hasTruth ? [labels.truth] : [])],
        rows.map((r) => [
          r.label,
          options.format(r.mean),
          options.format(r.low),
          options.format(r.high),
          ...(hasObserved ? [r.observed === undefined ? "" : options.format(r.observed)] : []),
          ...(hasTruth ? [r.truth === undefined ? "" : options.format(r.truth)] : []),
        ]),
      );
    },
  };
}

/**
 * A mean line inside a 90% band over a numeric x axis, with optional comparison lines and vertical markers.
 * `points` are { x, mean, low, high, ...fields }; `options.lines` are [{ field, key, label }], extra series drawn
 * from point[field] with `key` a line style ("truth" or "compare"). `options.xLabel(x)` names a point.
 */
export function curveChart(container, options) {
  const frame = new Frame(container, options);
  const lines = options.lines ?? [];
  let data = null;
  frame.setLegend([
    { key: "line", label: "Posterior mean" },
    { key: "band", label: "90% interval" },
    ...lines.map((l) => ({ key: `line ${l.key}`, label: l.label })),
  ]);
  frame.render = () => {
    if (!data) return;
    const { points, markers } = data;
    const { svg, width, height } = frame.svg();
    const pad = { left: 52, right: 16, top: 16, bottom: 30 };
    const x0 = points[0].x;
    const x1 = points.at(-1).x;
    const x = scale(x0, x1, pad.left, width - pad.right);
    const hi = Math.max(...points.flatMap((p) => [p.high, ...lines.map((l) => p[l.field])]));
    const yTicks = coverTicks(hi, 4);
    const y = scale(0, yTicks.at(-1), height - pad.bottom, pad.top);
    axes(svg, {
      left: pad.left, right: width - pad.right, top: pad.top, bottom: height - pad.bottom, x, y,
      xTicks: ticks(x0, x1, Math.max(3, Math.floor(width / 110))), yTicks,
      xFormat: options.xFormat, yFormat: options.tickFormat ?? options.yFormat,
    });
    const area =
      points.map((p) => `${x(p.x)},${y(p.high)}`).join(" ") +
      " " +
      points.map((p) => `${x(p.x)},${y(p.low)}`).reverse().join(" ");
    el("polygon", { points: area, class: "band" }, svg);
    const path = (field) => points.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p[field])}`).join("");
    for (const l of lines) el("path", { d: path(l.field), class: `series-line ${l.key}` }, svg);
    el("path", { d: path("mean"), class: "series-line" }, svg);
    for (const marker of markers) {
      if (marker.value < x0 || marker.value > x1) continue;
      const mx = x(marker.value);
      el("line", { x1: mx, x2: mx, y1: pad.top, y2: height - pad.bottom, class: `marker ${marker.tone ?? ""}` }, svg);
      const label = el("text", { x: mx + 6, y: pad.top + 10, class: "annotation" }, svg);
      label.textContent = marker.label;
      if (mx + 6 + label.getComputedTextLength() > width - pad.right) {
        label.setAttribute("x", mx - 6);
        label.setAttribute("text-anchor", "end");
      }
    }

    const n = points.length;
    const cross = el("line", { y1: pad.top, y2: height - pad.bottom, class: "crosshair", visibility: "hidden" }, svg);
    const dot = el("circle", { r: 4, class: "crosshair-dot", visibility: "hidden" }, svg);
    const hit = el("rect", {
      x: pad.left, y: pad.top, width: width - pad.left - pad.right, height: height - pad.top - pad.bottom,
      class: "hit", tabindex: 0, "aria-label": `${options.title}: use the arrow keys to read each point`,
    }, svg);
    let focused = 0;
    const at = (i) => {
      focused = Math.max(0, Math.min(n - 1, i));
      const p = points[focused];
      cross.setAttribute("x1", x(p.x));
      cross.setAttribute("x2", x(p.x));
      dot.setAttribute("cx", x(p.x));
      dot.setAttribute("cy", y(p.mean));
      cross.setAttribute("visibility", "visible");
      dot.setAttribute("visibility", "visible");
      frame.show(x(p.x), y(p.high), [
        { key: "line", value: options.yFormat(p.mean), label: options.xLabel(p.x) },
        { value: `${options.yFormat(p.low)} to ${options.yFormat(p.high)}`, label: "90% interval" },
        ...lines.map((l) => ({ key: `line ${l.key}`, value: options.yFormat(p[l.field]), label: l.label.toLowerCase() })),
      ]);
    };
    const away = () => {
      cross.setAttribute("visibility", "hidden");
      dot.setAttribute("visibility", "hidden");
      frame.hide();
    };
    hit.addEventListener("pointermove", (e) => {
      const box = svg.getBoundingClientRect();
      const px = e.clientX - box.left;
      at(Math.round(((px - pad.left) / (width - pad.left - pad.right)) * (n - 1)));
    });
    hit.addEventListener("pointerleave", away);
    hit.addEventListener("focus", () => at(focused));
    hit.addEventListener("blur", away);
    hit.addEventListener("keydown", (e) => {
      const move = { ArrowLeft: -1, ArrowRight: 1, Home: -n, End: n }[e.key];
      if (move) {
        e.preventDefault();
        at(focused + move);
      }
    });
  };
  return {
    update(points, markers = []) {
      data = { points, markers };
      frame.render();
      const step = options.tableStep ?? 1;
      frame.setTable(
        [options.xHeading ?? "", "Mean", "5%", "95%", ...lines.map((l) => l.label)],
        points
          .filter((_, i) => i % step === 0)
          .map((p) => [
            options.xLabel(p.x),
            options.yFormat(p.mean),
            options.yFormat(p.low),
            options.yFormat(p.high),
            ...lines.map((l) => options.yFormat(p[l.field])),
          ]),
      );
    },
  };
}
