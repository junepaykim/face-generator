// Exercise the shipped script with controlled form events and delayed fetch responses.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const tick = () => new Promise(setImmediate);

function page() {
  const listeners = [];
  const form = {
    addEventListener(type, fn) { if (type === "change") listeners.push(fn); },
    querySelector() { return { value: "token" }; },
    querySelectorAll(selector) {
      if (selector === "[data-key]") return [mode, cutoff];
      if (selector.includes('name="preset.quality"')) return [preset];
      return [];
    },
  };
  const field = { classList: { toggle() {} }, querySelectorAll: () => [], querySelector: () => ({}) };
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
  const sidebar = {
    dataset: { display: "display.theme" }, value: "auto", type: "select-one", closest: () => null,
    addEventListener(type, fn) { if (type === "change") this.change = fn; },
  };
  const status = {};
  const buttons = {};
  const document = {
    documentElement: {}, body: { appendChild() {} }, createElement: () => ({ setAttribute() {} }),
    addEventListener() {}, getElementById: () => null,
    querySelectorAll(s) {
      if (s === "form[data-autosave]") return [form];
      if (s === "select[data-display]") return [sidebar];
      return [];
    },
    querySelector(s) {
      if (s === "form[data-autosave]") return form;
      if (s === "[data-save-status]") return status;
      return ["[data-undo]", "[data-redo]"].includes(s) ? buttons : null;
    },
  };
  let reloads = 0;
  const requests = [];
  const window = { addEventListener() {} };
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
  vm.runInNewContext(fs.readFileSync(process.argv[2], "utf8"), {
    window, document, fetch, location: { reload() { reloads++; } }, FormData, setInterval() {},
  });
  return { mode, cutoff, preset, sidebar, requests, status, reloads: () => reloads };
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
  r.sidebar.value = "dark";
  r.sidebar.change();
  assert.equal(r.requests.length, 1);
  r.requests[0].reply({});
  await tick();
  assert.deepEqual(r.requests[1].values, { "display.theme": "dark" });
  r.requests[1].reply({ reload: true });
  await tick();
  assert.equal(r.reloads(), 1);
}

main().catch((err) => { console.error(err); process.exitCode = 1; });
