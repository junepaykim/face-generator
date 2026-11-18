// facegen: dashboard refresh, tick-all, confirmations, settings autosave and the language and
// theme switches. No libraries, nothing loaded from the web.
"use strict";

// The page's own texts, in its language (the message catalog, Phase 12a). "{count}" and the
// like are filled in here.
const TEXTS = (() => {
  try { return JSON.parse(document.getElementById("texts").textContent); } catch (e) { return {}; }
})();
function text(key, values) {
  let s = TEXTS[key] || key;
  for (const [name, value] of Object.entries(values || {})) s = s.split("{" + name + "}").join(String(value));
  return s;
}
const LANG = document.documentElement.lang || undefined;

// The language and theme switches at the foot of the sidebar: saved as a setting (one change
// that Settings can undo), then the page loads again in the new language or theme.
(function displaySwitches() {
  const token = document.getElementById("csrf");
  document.querySelectorAll("select[data-display]").forEach((select) => {
    select.addEventListener("change", async () => {
      const body = new FormData();
      body.append(select.dataset.display, select.value);
      try {
        const res = await fetch("/settings/save", {
          method: "POST", body,
          headers: { "X-CSRF-Token": token.value, "Accept": "application/json" },
        });
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error || res.status);
        location.reload();
      } catch (e) {
        window.alert(text("js.switch_failed"));
      }
    });
  });
})();

// Live pages (Phase 12d). The server says when a part of the page has changed (server-sent
// events from /events, one topic per part: "status" on the dashboard, "log" on the Log page);
// the page then fetches that part again. The sidebar's state dot and review count follow on
// every page. When the career changes (the first sync, or a save of another career), the whole
// page is out of date: it reloads. Without EventSource the part is fetched every 3 s instead.
(function live() {
  const box = document.querySelector("[data-refresh]");
  const topic = box ? box.dataset.topic : null;
  const career = () => {
    const mark = box && box.querySelector("[data-career]");
    return mark ? mark.dataset.career : null;
  };
  let busy = false;
  let again = false;
  let stale = false;

  // A part comes back as sections with ids. A section the server sends as it did last time
  // keeps its elements, so no picture in it is drawn again; the latest pictures keep theirs
  // and only a new one is added. New pictures are decoded before they show.
  const key = (el) => {
    const copy = el.cloneNode(true);  // the moving progress bar is not a change
    copy.querySelectorAll("progress[data-since]").forEach((bar) => bar.removeAttribute("value"));
    return copy.outerHTML;
  };
  const sent = new Map();
  if (box) Array.from(box.children).forEach((el) => { if (el.id) sent.set(el.id, key(el)); });

  async function refresh() {
    if (!box) return;
    if (document.hidden) { stale = true; return; }
    if (busy) { again = true; return; }
    busy = true;
    try {
      const res = await fetch(box.dataset.refresh, { headers: { "Accept": "text/html" } });
      if (res.ok) {
        const before = career();
        const parts = document.createRange().createContextualFragment(await res.text());
        const wanted = [];
        const fresh = [];
        const after = [];
        let refocus = null;
        for (const el of Array.from(parts.children)) {
          const old = el.id ? box.querySelector(":scope > #" + CSS.escape(el.id)) : null;
          const html = el.id ? key(el) : null;
          if (old && sent.get(el.id) === html) { wanted.push(old); continue; }
          if (el.id) sent.set(el.id, html);
          if (old && old.hasAttribute("data-pictures")) {
            const merged = mergePictures(old, el);
            if (merged) { wanted.push(old); fresh.push(...merged.fresh); after.push(merged.done); continue; }
          }
          // Folded sections the user opened stay open.
          if (old) old.querySelectorAll("details[open][id]").forEach((d) => {
            const twin = el.querySelector("#" + CSS.escape(d.id));
            if (twin) twin.open = true;
          });
          // So does what the user is typing (the guide's folder box).
          if (old) old.querySelectorAll("input[name]:not([type=hidden])").forEach((input) => {
            const twin = el.querySelector(`input[name="${CSS.escape(input.name)}"]`);
            if (!twin || twin.type !== input.type) return;
            if (input.value !== input.defaultValue) twin.value = input.value;
            if (document.activeElement === input) {
              refocus = [twin, input.selectionStart, input.selectionEnd];
            }
          });
          wanted.push(el);
          fresh.push(...el.querySelectorAll("img"));
        }
        await decoded(fresh);
        place(box, wanted);
        after.forEach((done) => done());
        if (refocus) {
          const [input, start, end] = refocus;
          input.focus();
          try { input.setSelectionRange(start, end); } catch (e) { /* not a text box */ }
        }
        localTimes(box);
        progress();
        if (before !== null && career() !== before) location.reload();
      }
    } catch (e) { /* the server may be restarting */ }
    finally {
      busy = false;
      if (again) { again = false; refresh(); }
    }
  }

  // Children of `parent` in this order, moving or removing only what is out of place.
  function place(parent, wanted) {
    wanted.forEach((el, i) => {
      const at = parent.children[i];
      if (at !== el) parent.insertBefore(el, at || null);
    });
    Array.from(parent.children).slice(wanted.length).forEach((el) => el.remove());
  }

  // Images loaded and decoded before they show, so none flashes in (at most 1.5 s).
  function decoded(images) {
    const all = images.map((img) => {
      img.loading = "eager";
      return img.decode().catch(() => {});
    });
    return Promise.race([Promise.all(all), new Promise((ok) => setTimeout(ok, 1500))]);
  }

  // The latest pictures: keep the cards shown, add a new one at the front, and let the others
  // glide to their new places.
  function mergePictures(oldSection, newSection) {
    const grid = oldSection.querySelector(".thumbs");
    const incoming = newSection.querySelector(".thumbs");
    if (!grid || !incoming) return null;
    const shown = new Map(Array.from(grid.children, (card) => [card.dataset.id, card]));
    const offset = (card) => {  // where a card sits within the grid, whatever moves above it
      const a = card.getBoundingClientRect();
      const g = grid.getBoundingClientRect();
      return { x: a.left - g.left, y: a.top - g.top };
    };
    const was = new Map(Array.from(shown, ([id, card]) => [id, offset(card)]));
    const fresh = [];
    const cards = Array.from(incoming.children, (card) => {
      const kept = shown.get(card.dataset.id);
      if (kept) return kept;
      card.classList.add("fresh");
      card.addEventListener("animationend", function ended(e) {
        if (e.animationName !== "thumb-ring") return;  // the last of its animations
        card.classList.remove("fresh");
        card.removeEventListener("animationend", ended);
      });
      fresh.push(...card.querySelectorAll("img"));
      return card;
    });
    return {
      fresh,
      done() {
        place(grid, cards);
        if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
        cards.forEach((card) => {
          const from = was.get(card.dataset.id);
          if (!from) return;
          const to = offset(card);
          const dx = from.x - to.x;
          const dy = from.y - to.y;
          if (!dx && !dy) return;
          card.animate(
            [{ transform: `translate(${dx}px, ${dy}px)` }, { transform: "none" }],
            { duration: 420, easing: "cubic-bezier(.2, .7, .2, 1)" },
          );
        });
      },
    };
  }
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && stale) { stale = false; refresh(); }
  });

  function sidebar(data) {
    const dot = document.querySelector(".side .dot-state");
    if (dot && data.state) {
      dot.className = "dot-state state-" + data.state;
      if (data.label) dot.title = data.label;
    }
    const link = document.querySelector('.side nav a[href="/review"]');
    if (!link) return;
    let count = link.querySelector(".count");
    if (data.review) {
      if (!count) {
        count = document.createElement("span");
        count.className = "count";
        link.appendChild(count);
      }
      count.textContent = data.review;
    } else if (count) {
      count.remove();
    }
  }

  if (!window.EventSource) {
    if (box) setInterval(refresh, 3000);
    return;
  }
  // The part as the page was drawn: its fingerprint, so the first event only fetches it again
  // if something changed between drawing the page and the stream opening.
  const seen = {};
  if (box && topic) seen[topic] = box.dataset.rev;
  const source = new EventSource("/events");
  ["status", "log", "pictures"].forEach((name) => {
    source.addEventListener(name, (e) => {
      let data;
      try { data = JSON.parse(e.data); } catch (err) { return; }
      if (name === "status") sidebar(data);
      if (name === topic && seen[name] !== data.rev) refresh();
      seen[name] = data.rev;
    });
  });
})();

// The picture being generated: its bar moves with the time since it started, against the usual
// time per picture (data-since, data-each), between the server's updates.
function progress() {
  document.querySelectorAll("progress[data-since][data-each]").forEach((bar) => {
    const since = Date.parse(bar.dataset.since);
    const each = parseFloat(bar.dataset.each);
    if (isNaN(since) || !(each > 0)) return;
    bar.value = Math.min(99, Math.max(0, ((Date.now() - since) / 1000 / each) * 100));
  });
}
setInterval(progress, 1000);
progress();

// Hover texts for facegen's own words (Phase 12): a group chip says what the group is, and so
// do a few terms such as auto-update and pinned (data-tip). One floating box, placed by the
// word, so no table or card cuts it off; keyboard focus shows it too.
(function hoverTips() {
  const tip = document.createElement("div");
  tip.className = "hover-tip";
  tip.setAttribute("role", "tooltip");
  tip.hidden = true;
  document.body.appendChild(tip);
  let current = null;
  const hide = () => { current = null; tip.hidden = true; };
  const show = (el) => {
    current = el;
    tip.textContent = el.dataset.tip;
    tip.style.visibility = "hidden";
    tip.hidden = false;
    const word = el.getBoundingClientRect();
    const box = tip.getBoundingClientRect();
    const left = Math.min(Math.max(8, word.left), window.innerWidth - box.width - 8);
    let top = word.bottom + 6;
    if (top + box.height > window.innerHeight - 8) top = word.top - box.height - 6;
    tip.style.left = left + "px";
    tip.style.top = Math.max(8, top) + "px";
    tip.style.visibility = "";
  };
  document.addEventListener("mouseover", (e) => {
    const el = e.target.closest("[data-tip]");
    if (el && el !== current) show(el);
    else if (!el && current) hide();
  });
  document.addEventListener("focusin", (e) => {
    const el = e.target.closest("[data-tip]");
    if (el) show(el);
  });
  document.addEventListener("focusout", hide);
  window.addEventListener("scroll", hide, true);
  // A live update can take the word away under the mouse.
  setInterval(() => { if (current && !current.isConnected) hide(); }, 500);
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
    if (label) label.textContent = text("js.ticked", { count: n });
    buttons.forEach((b) => { b.disabled = n === 0; });
    if (all) all.checked = n > 0 && n === boxes().length;
  };
  if (all) all.addEventListener("change", () => { boxes().forEach((b) => { b.checked = all.checked; }); update(); });
  form.addEventListener("change", update);
  update();
})();

// Review: a player's candidates side by side, with the picture FM27 shows now (Phase 12i). The
// pictures load when the dialog opens; without a dialog the links open each picture.
document.addEventListener("click", (e) => {
  const opener = e.target.closest("[data-compare]");
  if (opener) {
    const dialog = document.getElementById(opener.dataset.compare);
    if (!dialog || typeof dialog.showModal !== "function") return;
    e.preventDefault();
    dialog.querySelectorAll("img[data-src]").forEach((img) => {
      img.src = img.dataset.src;
      img.removeAttribute("data-src");
    });
    dialog.querySelectorAll(".compare-pic.picked").forEach((f) => f.classList.remove("picked"));
    dialog.showModal();
    const at = opener.dataset.at && dialog.querySelector(`.compare-pic[data-at="${opener.dataset.at}"]`);
    if (at) {
      at.classList.add("picked");
      at.scrollIntoView({ block: "nearest", inline: "center" });
    }
    return;
  }
  // A click beside the dialog's box, or on Close, closes it.
  const dialog = e.target.closest("dialog.compare");
  if (dialog && (e.target === dialog || e.target.closest("[data-close]"))) dialog.close();
});

// Forms with data-confirm ask first.
document.addEventListener("submit", (e) => {
  const msg = e.target.dataset && e.target.dataset.confirm;
  if (msg && !window.confirm(msg)) e.preventDefault();
});

// Times in the settings history, shown in this PC's local time.
function localTimes(root) {
  root.querySelectorAll("time[datetime]").forEach((t) => {
    const d = new Date(t.getAttribute("datetime"));
    if (isNaN(d)) return;
    // data-short: the time for today, the date for an earlier day (under a thumbnail).
    if (t.hasAttribute("data-short")) {
      const today = d.toDateString() === new Date().toDateString();
      t.textContent = today
        ? d.toLocaleTimeString(LANG, { hour: "2-digit", minute: "2-digit" })
        : d.toLocaleDateString(LANG, { month: "short", day: "numeric" });
      t.title = d.toLocaleString(LANG);
      return;
    }
    t.textContent = d.toLocaleString(LANG, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
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
  // The Basic page shows no default beside its fields.
  const showChanged = (el, changed) => {
    const field = fieldOf(el);
    field.classList.toggle("changed", changed);
    field.querySelectorAll("[data-dot], [data-use-default]").forEach((x) => { x.hidden = !changed; });
  };
  // A preset (Basic page) is a radio group: the choice sent, then the one the values match.
  const chosen = new Map();
  const radios = (kind) => form.querySelectorAll(`[data-preset][name="preset.${kind}"]`);
  const showPreset = (kind, name) => {
    radios(kind).forEach((r) => { r.checked = r.value === name; });
    const custom = form.querySelector(`[data-custom="${kind}"]`);
    if (custom) custom.hidden = name !== null;
  };

  async function saveDirty() {
    const fields = dirty();
    const picked = new Map(chosen);
    chosen.clear();
    if (!fields.length && !picked.size) return;
    running += 1;
    say(text("js.saving"));
    const body = new FormData();
    const posted = new Map(fields.map((el) => [el.name, read(el)]));
    posted.forEach((value, name) => body.append(name, value));
    picked.forEach((value, name) => body.append(name, value));
    try {
      const res = await fetch("/settings/save", {
        method: "POST",
        body,
        headers: { "X-CSRF-Token": token, "Accept": "application/json" },
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      fields.forEach((el) => showError(el, data.errors && data.errors[el.name]));
      picked.forEach((_, name) => {
        const radio = form.querySelector(`[data-preset][name="${name}"]`);
        if (radio) showError(radio, data.errors && data.errors[name]);
      });
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
      syncTicks();
      Object.entries(data.presets || {}).forEach(([kind, name]) => showPreset(kind, name));
      say(data.message, data.changed ? "ok" : "");
      // A new language or theme shows once the page loads again.
      if (data.reload) { leaving = true; location.reload(); return; }
      if (history) {
        history.innerHTML = data.history_html;
        localTimes(history);
      }
      undoButton.disabled = !data.can_undo;
      redoButton.disabled = !data.can_redo;
    } catch (e) {
      say(text("js.no_answer"), "bad");
    } finally {
      running -= 1;
    }
  }
  // Saves run one at a time, in order; each sends whatever differs from the server then.
  const save = () => (chain = chain.then(saveDirty));

  // A list of groups shown as tick boxes: the hidden field holds "a, c" and is what is saved.
  const syncTicks = () => {
    form.querySelectorAll("[data-group-of]").forEach((box) => {
      const field = inputs.get(box.dataset.groupOf);
      if (field) box.checked = field.value.split(/[,\s]+/).includes(box.value);
    });
  };
  form.addEventListener("change", (e) => {
    const box = e.target.closest("[data-group-of]");
    if (!box) return;
    const name = box.dataset.groupOf;
    const field = inputs.get(name);
    field.value = Array.from(form.querySelectorAll(`[data-group-of="${name}"]`))
      .filter((b) => b.checked).map((b) => b.value).join(", ");
  }, true);

  form.addEventListener("change", (e) => {
    if (e.target.matches("[data-preset]")) chosen.set(e.target.name, e.target.value);
  }, true);
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
    syncTicks();
    save();
  });

  // Undo, Redo, Reset and "undo back to here" wait for a save on its way, so they act on it.
  document.addEventListener("submit", async (e) => {
    const target = e.target;
    if (leaving || !target.matches("form[data-settings-action]")) return;
    e.preventDefault();
    const submitter = e.submitter;
    await save();
    if (dirty().length && !window.confirm(text("js.unsaved_confirm"))) return;
    leaving = true;
    target.requestSubmit(submitter);
  });
  window.addEventListener("beforeunload", (e) => {
    if (!leaving && (running || dirty().length)) { e.preventDefault(); e.returnValue = ""; }
  });
})();

// A filter form that applies itself: a list choice at once, the search box on Enter (Gallery).
// Fields left at their default (an empty box, a list's first choice) stay out of the address.
document.querySelectorAll("form[data-autosubmit]").forEach((form) => {
  form.addEventListener("change", (e) => { if (e.target.matches("select")) form.requestSubmit(); });
  form.addEventListener("submit", () => {
    form.querySelectorAll("input[name], select[name]").forEach((el) => {
      el.disabled = el.matches("select") ? el.selectedIndex === 0 : !el.value.trim();
    });
  });
  // Back to a page kept by the browser: the fields work again.
  window.addEventListener("pageshow", () => {
    form.querySelectorAll("[name]").forEach((el) => { el.disabled = false; });
  });
});

// Times on every page (the settings history, the dashboard's pictures) in this PC's local time.
localTimes(document);
