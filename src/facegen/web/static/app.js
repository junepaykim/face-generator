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

// Times in the settings history, shown in this PC's local time.
function localTimes(root) {
  root.querySelectorAll("time[datetime]").forEach((t) => {
    const d = new Date(t.getAttribute("datetime"));
    if (!isNaN(d)) {
      t.textContent = d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
    }
  });
}

// Settings: each change is saved as it is made; there is no Apply button. The server keeps
// what every change replaced, so Undo, Redo and the Recent changes list can put it back.
(function autosave() {
  const form = document.querySelector("form[data-autosave]");
  if (!form) return;
  const token = form.querySelector('input[name="csrf_token"]').value;
  const status = document.querySelector("[data-save-status]");
  const history = document.getElementById("history");
  const undoButton = document.querySelector("[data-undo]");
  const redoButton = document.querySelector("[data-redo]");
  const inputs = new Map(Array.from(form.querySelectorAll("[data-key]")).map((el) => [el.name, el]));
  const read = (el) => (el.type === "checkbox" ? (el.checked ? "on" : "") : el.value.trim());
  // What the server holds for each field, as the field would show it.
  const saved = new Map(Array.from(inputs, ([name, el]) => [name, read(el)]));
  const dirty = () => Array.from(inputs.values()).filter((el) => read(el) !== saved.get(el.name));
  let running = 0;
  let leaving = false;
  let chain = Promise.resolve();

  const say = (text, kind) => {
    status.textContent = text;
    status.className = "save-status" + (kind ? " " + kind : "");
  };
  const fieldOf = (el) => el.closest("[data-field]");
  const showError = (el, message) => {
    const field = fieldOf(el);
    field.classList.toggle("error", Boolean(message));
    field.querySelector("[data-error]").textContent = message ? " · " + message : "";
  };
  const showChanged = (el, changed) => {
    const field = fieldOf(el);
    field.classList.toggle("changed", changed);
    field.querySelector("[data-dot]").hidden = !changed;
    field.querySelector("[data-use-default]").hidden = !changed;
  };

  async function saveDirty() {
    const fields = dirty();
    if (!fields.length) return;
    running += 1;
    say("Saving…");
    const body = new FormData();
    const posted = new Map(fields.map((el) => [el.name, read(el)]));
    posted.forEach((value, name) => body.append(name, value));
    try {
      const res = await fetch("/settings/save", {
        method: "POST",
        body,
        headers: { "X-CSRF-Token": token, "Accept": "application/json" },
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      fields.forEach((el) => showError(el, data.errors && data.errors[el.name]));
      if (!data.ok) {
        say(data.error, "bad");
        return;
      }
      for (const [name, f] of Object.entries(data.fields)) {
        const el = inputs.get(name);
        if (!el) continue;
        const shown = el.type === "checkbox" ? (f.value ? "on" : "") : f.text;
        // Show the value as stored, unless it was edited again while this save was on its way.
        if (read(el) === posted.get(name)) {
          if (el.type === "checkbox") el.checked = Boolean(f.value);
          else el.value = f.text;
        }
        saved.set(name, shown);
        showChanged(el, f.changed);
      }
      say(data.message, data.changed ? "ok" : "");
      history.innerHTML = data.history_html;
      localTimes(history);
      undoButton.disabled = !data.can_undo;
      redoButton.disabled = !data.can_redo;
    } catch (e) {
      say("Not saved: facegen did not answer. Is it still running?", "bad");
    } finally {
      running -= 1;
    }
  }
  // Saves run one at a time, in order; each sends whatever differs from the server then.
  const save = () => (chain = chain.then(saveDirty));

  form.addEventListener("change", save);
  form.addEventListener("submit", (e) => { e.preventDefault(); save(); });
  form.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.target.matches("input")) { e.preventDefault(); save(); }
  });
  form.addEventListener("click", (e) => {
    const button = e.target.closest("[data-use-default]");
    if (!button) return;
    e.preventDefault();
    const el = fieldOf(button).querySelector("[data-key]");
    if (el.type === "checkbox") el.checked = el.dataset.default === "on";
    else el.value = el.dataset.default;
    save();
  });

  // Undo, Redo, Reset and "undo back to here" wait for a save on its way, so they act on it.
  document.addEventListener("submit", async (e) => {
    const target = e.target;
    if (leaving || !target.matches("form[data-settings-action]")) return;
    e.preventDefault();
    const submitter = e.submitter;
    await save();
    if (dirty().length && !window.confirm("Some changes could not be saved (see the fields marked in red). Leave them unsaved?")) return;
    leaving = true;
    target.requestSubmit(submitter);
  });
  window.addEventListener("beforeunload", (e) => {
    if (!leaving && (running || dirty().length)) { e.preventDefault(); e.returnValue = ""; }
  });
  localTimes(document);
})();
