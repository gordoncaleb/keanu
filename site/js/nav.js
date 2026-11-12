// The site's examples by domain, and the sidebar built from them. This list is the one place a demo is
// registered: the sidebar on every page and the landing page's index both read it.

export const DOMAINS = [
  {
    name: "Finance",
    examples: [
      {
        href: "var.html",
        title: "Value at risk",
        question: "How much could this position lose tomorrow?",
        summary: "Volatility models of daily returns, value at risk and expected shortfall. Paste your own prices.",
        ready: true,
      },
      {
        href: "credit.html",
        title: "Credit portfolio",
        question: "How many of our loans will default, and what will it cost?",
        summary: "Default rates learned from a loan history, then a loan book priced under a downturn you choose.",
        ready: true,
      },
    ],
  },
  {
    name: "Marketing",
    examples: [
      {
        href: "ab-test.html",
        title: "A/B test",
        question: "Should we ship the new version?",
        summary: "The chance a variant really converts better, how much better, and what shipping it risks.",
        ready: true,
      },
    ],
  },
  {
    name: "Operations",
    examples: [
      {
        href: "demand.html",
        title: "Demand forecasting",
        question: "How much stock do we need next month?",
        summary: "A seasonal forecast of daily sales with its uncertainty, turned into a stock level.",
        ready: false,
      },
      {
        href: "reliability.html",
        title: "Equipment reliability",
        question: "When will this fleet start failing?",
        summary: "Failure times learned from units that failed and units still running, and the warranty cost.",
        ready: true,
      },
    ],
  },
  {
    name: "Health",
    examples: [
      {
        href: "epidemic.html",
        title: "Epidemic forecasting",
        question: "Will the hospitals run out of beds?",
        summary: "An outbreak model fitted to early case counts, projecting peak demand for beds.",
        ready: false,
      },
      {
        href: "dosing.html",
        title: "Drug dosing",
        question: "What dose keeps this patient in the safe range?",
        summary: "A drug's concentration over time learned from a few blood samples, for a dose you choose.",
        ready: false,
      },
    ],
  },
];

function element(name, attributes = {}, text) {
  const node = document.createElement(name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

/** The current page's file name, "index.html" for the site root. */
function currentPage() {
  const last = location.pathname.split("/").pop();
  return last === "" ? "index.html" : last;
}

/** Fills the page's sidebar: the wordmark, then each domain's examples. */
export function renderSidebar(aside = document.getElementById("sidebar")) {
  if (!aside) return;
  const here = currentPage();
  aside.replaceChildren();
  const home = element("a", { class: "wordmark", href: "index.html" }, "Keanu");
  if (here === "index.html") home.setAttribute("aria-current", "page");
  // On narrow screens the list folds behind a toggle; on wide screens it is always open.
  const toggle = element("button", { class: "menu-toggle", type: "button", "aria-expanded": "false", "aria-controls": "examples" }, "Examples");
  const nav = element("nav", { id: "examples", "aria-label": "Examples" });
  for (const domain of DOMAINS) {
    const group = element("div", { class: "nav-group" });
    const heading = element("h2", {}, domain.name);
    const list = element("ul");
    for (const example of domain.examples) {
      const item = element("li");
      if (example.ready) {
        const link = element("a", { href: example.href }, example.title);
        if (example.href === here) link.setAttribute("aria-current", "page");
        item.append(link);
      } else {
        item.append(element("span", { class: "pending", title: "Being built" }, example.title));
      }
      list.append(item);
    }
    group.append(heading, list);
    nav.append(group);
  }
  toggle.addEventListener("click", () => {
    const open = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(open));
  });
  aside.append(home, toggle, nav);
}

renderSidebar();
