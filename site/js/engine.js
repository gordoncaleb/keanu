// A promise-based handle on the demo worker.

export class Engine {
  constructor() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.pending = new Map();
    this.nextId = 1;
    this.worker.onmessage = ({ data }) => {
      const request = this.pending.get(data.id);
      if (!request) return;
      if (data.type === "progress") request.onProgress?.(data.value);
      else {
        this.pending.delete(data.id);
        if (data.type === "done") request.resolve(data.result);
        else request.reject(new Error(data.message));
      }
    };
  }

  #send(message, onProgress) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage({ id, ...message });
    });
  }

  /** Fits a demo; resolves with { seconds } once finished. A newer fit abandons this one (it never resolves). */
  fit(demo, args, onProgress) {
    for (const [id, request] of this.pending) {
      if (request.onProgress) this.pending.delete(id);
    }
    return this.#send({ type: "fit", demo, args }, onProgress);
  }

  /** The fitted demo's report for `args`. */
  report(args) {
    return this.#send({ type: "report", args });
  }

  /** The bundled sample of daily returns. */
  sampleReturns() {
    return this.#send({ type: "sample-returns" });
  }
}

/**
 * Calls `run` with the latest value, at most one call in flight: values arriving meanwhile collapse to the newest,
 * so a dragged slider never queues up stale reports.
 */
export function latest(run) {
  let busy = false;
  let queued;
  let hasQueued = false;
  const pump = async (value) => {
    busy = true;
    try {
      await run(value);
    } finally {
      busy = false;
      if (hasQueued) {
        hasQueued = false;
        pump(queued);
      }
    }
  };
  return (value) => {
    if (busy) {
      queued = value;
      hasQueued = true;
    } else pump(value);
  };
}
