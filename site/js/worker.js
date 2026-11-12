// Runs Keanu demos off the main thread. It knows no demo by name: every class exported by keanu_wasm has a
// constructor that starts a fit, step(n) returning true once finished, progress(), and report(...) returning JSON.
//
// Messages in:  { id, type: "fit", demo: "CreditDemo", args: [...], steps }   construct, then step until done
//               { id, type: "report", args: [...] }                            the fitted demo's report(...)
//               { id, type: "call", demo: "VolatilityDemo", method, args }     a static method
// Messages out: { id, type: "progress", value } during a fit, then { id, type: "done", result } or
//               { id, type: "error", message }.

import init, * as wasm from "../pkg/keanu_wasm.js";

const ready = init();
let demo = null;
let fitting = 0; // the id of the newest fit; an older fit stops when it sees a newer one

function yieldToMessages() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function demoClass(name) {
  const cls = wasm[name];
  if (typeof cls !== "function") throw new Error(`There is no demo called ${name}.`);
  return cls;
}

async function fit(id, name, args, steps) {
  await ready;
  fitting = id;
  if (demo) demo.free();
  demo = null;
  const started = performance.now();
  const next = new (demoClass(name))(...args);
  while (!next.step(steps)) {
    postMessage({ id, type: "progress", value: next.progress() });
    await yieldToMessages();
    if (fitting !== id) {
      next.free();
      return;
    }
  }
  demo = next;
  postMessage({ id, type: "done", result: { seconds: (performance.now() - started) / 1000 } });
}

onmessage = async ({ data: { id, type, demo: name, method, args = [], steps = 25 } }) => {
  try {
    if (type === "fit") await fit(id, name, args, steps);
    else if (type === "report") {
      if (!demo) throw new Error("Nothing has been fitted yet.");
      postMessage({ id, type: "done", result: JSON.parse(demo.report(...args)) });
    } else if (type === "call") {
      await ready;
      const value = demoClass(name)[method](...args);
      postMessage({ id, type: "done", result: ArrayBuffer.isView(value) ? Array.from(value) : value });
    }
  } catch (error) {
    postMessage({ id, type: "error", message: String(error?.message ?? error) });
  }
};
