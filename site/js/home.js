import { DOMAINS } from "./nav.js";

// The landing page's index: each domain's examples, with the question each one answers.
const index = document.getElementById("index");
for (const domain of DOMAINS) {
  const section = document.createElement("section");
  section.className = "domain";
  const heading = document.createElement("h2");
  heading.textContent = domain.name;
  const list = document.createElement("ul");
  list.className = "demos";
  for (const example of domain.examples) {
    const item = document.createElement("li");
    const target = document.createElement(example.ready ? "a" : "div");
    if (example.ready) target.href = example.href;
    else target.className = "pending";
    const question = document.createElement("h3");
    question.textContent = example.question;
    const summary = document.createElement("p");
    summary.textContent = example.summary;
    const title = document.createElement("span");
    title.className = "timing";
    title.textContent = example.ready ? example.title : `${example.title}, being built`;
    target.append(question, summary, title);
    item.append(target);
    list.append(item);
  }
  section.append(heading, list);
  index.append(section);
}
