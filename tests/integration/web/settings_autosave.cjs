// Exercise the shipped script with controlled form events and delayed fetch responses.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const tick = () => new Promise(setImmediate);
const later = () => new Promise((ok) => setTimeout(ok, 5));

function page({ storage = new Map(), hash = "" } = {}) {
  const listeners = [];
  const clicks = [];
  const docListeners = { submit: [], keydown: [] };
  const form = {
    addEventListener(type, fn) {
      if (type === "change") listeners.push(fn);
      if (type === "click") clicks.push(fn);
    },
    querySelector() { return { value: "token" }; },
    querySelectorAll(selector) {
      if (selector === "[data-key]") return [mode, cutoff, theme, ...skin, candidates];
      if (selector === "[data-step-of]") return [minus, plus];
      if (selector.includes('name="preset.quality"')) return [preset];
      return [];
    },
  };
  const fields = [];
  function fieldBox(first) {
    const box = {
      changed: false,
      classList: { toggle(name, on) { if (name === "changed") box.changed = on; } },
      querySelectorAll: () => [],
      querySelector: (s) => (s === "[data-key]" ? first() : {}),
    };
    fields.push(box);
    return box;
  }
  const field = fieldBox(() => ({}));
  function input(name, value, radio = false) {
    return {
      name, value, type: radio ? "radio" : "text", closest: (s) => s === "[data-field]" ? field : null,
      matches: (s) => radio && s === "[data-preset]",
      change(next) {
        this.value = next;
        if (radio) this.checked = true;
        for (const fn of listeners) fn({ target: this });
      },
    };
  }
  const mode = input("cutoffs.mode", "auto");
  const cutoff = input("cutoffs.everywhere.a_ca", "165");
  const preset = input("preset.quality", "medium", true);
  const theme = input("display.theme", "auto");  // Settings' field; the sidebar has none (17d)
  // A row of buttons (Phase 17j): one radio per value, all named as the setting.
  const skinField = fieldBox(() => skin[0]);
  const skin = ["0", "1", "2"].map((value) => ({
    name: "my_club.soften_skin", value, type: "radio", checked: value === "2", dataset: { default: "2" },
    closest: (s) => s === "[data-field]" ? skinField : null, matches: () => false,
  }));
  const pick = (value) => {
    skin.forEach((r) => { r.checked = r.value === value; });
    for (const fn of listeners) fn({ target: skin.find((r) => r.value === value) });
  };
  const useDefault = { closest: (s) => (s === "[data-use-default]" ? useDefault : s === "[data-field]" ? skinField : null) };
  // A stepper: − and + beside a number with its limits.
  const candidates = {
    name: "groups.a.candidates", value: "3", type: "number", min: "1", max: "8", dataset: { default: "3" },
    closest: (s) => s === "[data-field]" ? field : null, matches: () => false,
  };
  const stepper = (step) => {
    const b = { hidden: true, disabled: false, dataset: { stepOf: "groups.a.candidates", step } };
    b.closest = (s) => (s === "[data-step-of]" ? b : null);
    return b;
  };
  const minus = stepper("-1");
  const plus = stepper("1");
  const click = (target) => { for (const fn of clicks) fn({ target, preventDefault() {} }); };
  const status = {};
  const buttons = {};
  const legend = { hidden: true };
  const opener = {
    textContent: "History (2)", classList: { toggle(name, on) { opener.on = on; } },
    setAttribute(k, v) { opener[k] = v; }, addEventListener(type, fn) { opener.click = fn; }, focus() {},
  };
  const closer = { hidden: true, addEventListener(type, fn) { closer.click = fn; }, focus() {} };
  const drawer = { hidden: false, querySelector: (s) => (s === "[data-history-close]" ? closer : null), contains: () => false };
  const history = { innerHTML: "", querySelectorAll: () => [] };
  const texts = { textContent: JSON.stringify({ "settings.history": "History ({count:,})" }) };
  const document = {
    documentElement: { classList: { add() {} } }, body: { appendChild() {} }, createElement: () => ({ setAttribute() {} }),
    addEventListener(type, fn) { if (docListeners[type]) docListeners[type].push(fn); },
    getElementById: (id) => ({ texts, history }[id] || null),
    querySelectorAll(s) {
      if (s === "form[data-autosave]") return [form];
      return [];
    },
    querySelector(s) {
      if (s === "form[data-autosave]") return form;
      if (s === "[data-save-status]") return status;
      if (s === "[data-changed-legend]") return legend;
      if (s === "[data-field].changed") return fields.find((f) => f.changed) || null;
      if (s === "[data-history-open]") return opener;
      if (s === "[data-history-drawer]") return drawer;
      return ["[data-undo]", "[data-redo]"].includes(s) ? buttons : null;
    },
  };
  let reloads = 0;
  const requests = [];
  const window = { addEventListener() {}, confirm: () => true };
  const fetch = (url, options) => new Promise((resolve, reject) => {
    const values = Object.fromEntries(options.body);
    requests.push({
      values, reject,
      reply(data) {
        const fields = Object.fromEntries(Object.entries(values).map(([k, v]) => [k, { value: v, text: v }]));
        resolve({ ok: true, json: async () => ({ ok: true, fields, ...data }) });
      },
    });
  });
  class FormData extends Map { append(key, value) { this.set(key, value); } }
  const sessionStorage = { getItem: (k) => storage.get(k) ?? null, setItem: (k, v) => { storage.set(k, v); } };
  vm.runInNewContext(fs.readFileSync(process.argv[2], "utf8"), {
    window, document, fetch, FormData, setInterval() {}, setTimeout, sessionStorage,
    location: { pathname: "/settings/advanced", hash, reload() { reloads++; } },
  });
  return {
    mode, cutoff, preset, theme, requests, status, reloads: () => reloads,
    skin, pick, checked: () => skin.filter((r) => r.checked).map((r) => r.value).join(),
    reset: () => click(useDefault), candidates, minus, plus, step: click, legend, opener, closer, drawer,
    key: (key) => docListeners.keydown.forEach((fn) => fn({ key })),
    submit: (event) => docListeners.submit.forEach((fn) => fn(event)),
  };
}

async function main() {
  const p = page();
  p.mode.change("manual");
  p.cutoff.change("170");
  assert.equal(p.requests.length, 1);
  p.requests[0].reply({ reload: true });
  await tick();
  assert.deepEqual(p.requests[1].values, { "cutoffs.everywhere.a_ca": "170" });
  p.cutoff.value = "175";  // still being typed, no change event before the response
  p.requests[1].reply({});
  await tick();
  assert.equal(p.reloads(), 0);
  assert.deepEqual(p.requests[2].values, { "cutoffs.everywhere.a_ca": "175" });
  p.requests[2].reply({});
  await tick();
  assert.equal(p.reloads(), 1);

  const q = page();
  q.mode.change("manual");
  q.cutoff.change("250");
  q.requests[0].reply({ reload: true });
  await tick();
  q.requests[1].reply({ ok: false, error: "out of range" });
  await tick();
  assert.equal(q.status.textContent, "out of range");
  assert.equal(q.reloads(), 0);
  assert.equal(q.cutoff.value, "250");
  q.cutoff.change("180");
  q.preset.change("high");
  q.requests[2].reply({ presets: { quality: "medium" } });
  await tick();
  assert.equal(q.preset.checked, true);
  q.requests[3].reject(new Error("offline"));
  await tick();
  assert.equal(q.reloads(), 0);
  q.cutoff.change("181");
  assert.deepEqual(q.requests[4].values, { "cutoffs.everywhere.a_ca": "181", "preset.quality": "high" });
  q.requests[4].reply({ presets: { quality: "high" } });
  await tick();
  assert.equal(q.reloads(), 1);

  const r = page();
  r.cutoff.change("175");
  r.theme.change("dark");
  assert.equal(r.requests.length, 1);
  r.requests[0].reply({});
  await tick();
  assert.deepEqual(r.requests[1].values, { "display.theme": "dark" });
  r.requests[1].reply({ reload: true });
  await tick();
  assert.equal(r.reloads(), 1);

  // Phase 17j. Recent changes: a drawer that History (N) opens, Esc closes, and that stays open
  // in the next page (after an Undo).
  const storage = new Map();
  const s = page({ storage, hash: "#tab-quality" });
  assert.equal(s.drawer.hidden, true);
  assert.equal(s.closer.hidden, false);
  s.opener.click({ preventDefault() {} });
  assert.equal(s.drawer.hidden, false);
  assert.equal(s.opener["aria-expanded"], "true");
  s.key("Escape");
  assert.equal(s.drawer.hidden, true);
  assert.equal(s.opener["aria-expanded"], "false");
  s.opener.click({ preventDefault() {} });
  assert.equal(storage.get("kitshot.history-open"), "1");

  // A row of buttons saves the value picked; one picked while that save is on its way wins.
  s.pick("0");
  assert.deepEqual(s.requests[0].values, { "my_club.soften_skin": "0" });
  s.pick("1");
  s.requests[0].reply({ history_count: 3, fields: { "my_club.soften_skin": { value: 0, text: "0", changed: true } } });
  await tick();
  assert.equal(s.checked(), "1");
  assert.equal(s.opener.textContent, "History (3)");
  assert.equal(s.legend.hidden, false);  // a setting differs: the dot is explained
  assert.deepEqual(s.requests[1].values, { "my_club.soften_skin": "1" });
  s.requests[1].reply({ fields: { "my_club.soften_skin": { value: 1, text: "1", changed: true } } });
  await tick();
  assert.equal(s.checked(), "1");
  // ↺ puts the default back and saves it; with nothing changed the legend goes.
  s.reset();
  assert.equal(s.checked(), "2");
  assert.deepEqual(s.requests[2].values, { "my_club.soften_skin": "2" });
  s.requests[2].reply({ fields: { "my_club.soften_skin": { value: 2, text: "2", changed: false } } });
  await tick();
  assert.equal(s.legend.hidden, true);

  // A stepper: + and − save one more or one less, within the limits; each turns off at its end.
  assert.equal(s.plus.hidden, false);
  assert.equal(s.minus.hidden, false);
  s.step(s.plus);
  assert.equal(s.candidates.value, "4");
  assert.deepEqual(s.requests[3].values, { "groups.a.candidates": "4" });
  s.requests[3].reply({});
  await tick();
  s.candidates.value = "7";
  s.step(s.plus);
  assert.equal(s.candidates.value, "8");
  assert.equal(s.plus.disabled, true);
  s.requests[4].reply({});
  await tick();
  s.step(s.plus);
  assert.equal(s.candidates.value, "8");
  await tick();
  assert.equal(s.requests.length, 5);  // nothing new to save
  s.candidates.value = "2";
  s.step(s.minus);
  s.step(s.minus);
  assert.equal(s.candidates.value, "1");
  assert.equal(s.minus.disabled, true);
  assert.deepEqual(s.requests[5].values, { "groups.a.candidates": "1" });
  s.requests[5].reply({});
  await tick();

  // "Undo" on an entry in the drawer waits for a save on its way, then comes back to the tab.
  s.cutoff.change("170");
  const submitted = [];
  const next = { value: "/settings/advanced" };
  s.submit({
    target: { matches: (q) => q === "form[data-settings-action]", querySelector: () => next, requestSubmit(b) { submitted.push(b); } },
    submitter: "undo-through", preventDefault() {},
  });
  await tick();
  assert.deepEqual(submitted, []);
  s.requests[6].reply({});
  await tick();
  await tick();
  assert.deepEqual(submitted, []);  // in a task of its own: the click's submit is still on its way
  await later();
  assert.deepEqual(submitted, ["undo-through"]);
  assert.equal(next.value, "/settings/advanced#tab-quality");
  const u = page({ storage });
  assert.equal(u.drawer.hidden, false);
  assert.equal(u.opener["aria-expanded"], "true");
}

main().catch((err) => { console.error(err); process.exitCode = 1; });
