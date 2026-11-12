// Runs Keanu demos off the main thread. One demo object at a time; fits advance in small chunks so progress
// reaches the page between them.
//
// Messages in:  { id, type: "fit", demo: "volatility" | "credit", args: {...} }
//               { id, type: "report", args: {...} }
// Messages out: { id, type: "progress", value } during a fit, then { id, type: "done", result } or
//               { id, type: "error", message }.

import init, { CreditDemo, VolatilityDemo } from "../pkg/keanu_wasm.js";

const ready = init();
let demo = null;
let fitting = 0; // the id of the fit in progress, so a newer fit cancels an older one

const constructors = {
  volatility: ({ returns, stochastic, draws, seed }) =>
    new VolatilityDemo(new Float64Array(returns), stochastic, draws, seed),
  credit: ({ historyLoans, bookLoans, draws, seed }) =>
    new CreditDemo(historyLoans, bookLoans, draws, seed),
};

const chunk = { volatility: (args) => (args.stochastic ? 2 : 50), credit: () => 25 };

function yieldToMessages() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

async function fit(id, name, args) {
  await ready;
  fitting = id;
  if (demo) demo.free();
  demo = null;
  const started = performance.now();
  const next = constructors[name](args);
  const steps = chunk[name](args);
  while (!next.step(steps)) {
    postMessage({ id, type: "progress", value: next.progress() });
    // let a newer fit request arrive and take over
    await yieldToMessages();
    if (fitting !== id) {
      next.free();
      return;
    }
  }
  demo = next;
  postMessage({ id, type: "done", result: { seconds: (performance.now() - started) / 1000 } });
}

function report(id, args) {
  if (!demo) throw new Error("Nothing has been fitted yet.");
  const json =
    demo instanceof VolatilityDemo
      ? demo.report(args.position, args.horizon)
      : demo.report(args.stress, args.threshold);
  postMessage({ id, type: "done", result: JSON.parse(json) });
}

onmessage = async ({ data: { id, type, demo: name, args } }) => {
  try {
    if (type === "fit") await fit(id, name, args);
    else if (type === "report") report(id, args);
    else if (type === "sample-returns") {
      await ready;
      postMessage({ id, type: "done", result: Array.from(VolatilityDemo.sampleReturns()) });
    }
  } catch (error) {
    postMessage({ id, type: "error", message: String(error?.message ?? error) });
  }
};
