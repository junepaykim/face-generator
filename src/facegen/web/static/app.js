// facegen: dashboard refresh, tick-all and confirmations. No libraries, nothing loaded from the web.
"use strict";

// The dashboard's status block reloads itself every few seconds.
(function refresh() {
  const box = document.querySelector("[data-refresh]");
  if (!box) return;
  const url = box.dataset.refresh;
  setInterval(async () => {
    if (document.hidden) return;
    try {
      const res = await fetch(url, { headers: { "Accept": "text/html" } });
      if (res.ok) box.innerHTML = await res.text();
    } catch (e) { /* the server may be restarting */ }
  }, 3000);
})();

// Players: tick all on the page, count the ticks, and enable the bulk buttons.
(function ticks() {
  const form = document.getElementById("bulk");
  if (!form) return;
  const boxes = () => Array.from(form.querySelectorAll('input[name="uid"]'));
  const all = form.querySelector("[data-tick-all]");
  const label = form.querySelector("[data-ticked]");
  const buttons = form.querySelectorAll("[data-needs-tick]");
  const update = () => {
    const n = boxes().filter((b) => b.checked).length;
    if (label) label.textContent = n + " ticked";
    buttons.forEach((b) => { b.disabled = n === 0; });
    if (all) all.checked = n > 0 && n === boxes().length;
  };
  if (all) all.addEventListener("change", () => { boxes().forEach((b) => { b.checked = all.checked; }); update(); });
  form.addEventListener("change", update);
  update();
})();

// Forms with data-confirm ask first.
document.addEventListener("submit", (e) => {
  const msg = e.target.dataset && e.target.dataset.confirm;
  if (msg && !window.confirm(msg)) e.preventDefault();
});
