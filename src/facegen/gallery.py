"""A local HTML page of a career's generated pictures, for the user to look at.

Written to data/masters/<career key>/index.html and opened from disk: the pictures never
leave this PC (AGENTS.md hard rule 6). It shows every picture with its candidates, their
check results and the prompt, so the user can judge the pick and the quality check. The web
UI (Phase 3) replaces it.
"""

from __future__ import annotations

import html
import json
import sqlite3
from pathlib import Path
from typing import Any

from facegen.traits import CharacterSheet

_CSS = """
:root { --bg:#f6f7f9; --fg:#1d2330; --muted:#5d6675; --line:#d9dde4; --card:#fff;
        --ok:#1f7a45; --okbg:#e3f4ea; --bad:#a32a2a; --badbg:#fbe5e5; --pick:#2f5fd0; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14171c; --fg:#e7eaf0; --muted:#9aa3b2; --line:#2c323c; --card:#1c2027;
          --ok:#7fd6a0; --okbg:#173325; --bad:#f19b9b; --badbg:#3a1d1d; --pick:#8fb0ff; } }
body { margin:0; padding:16px 24px 48px; background:var(--bg); color:var(--fg);
       font:14px/1.45 system-ui, sans-serif; }
h1 { margin:0 0 4px; font-size:22px; }
.note { color:var(--muted); max-width:90ch; }
.player { display:flex; gap:16px; padding:14px 0; border-bottom:1px solid var(--line);
          flex-wrap:wrap; }
.main { width:320px; }
.main img { width:320px; aspect-ratio:3/4; object-fit:cover; display:block; border-radius:6px; }
.info { width:280px; font-size:13px; }
.info h2 { margin:0 0 4px; font-size:16px; }
.info .muted { color:var(--muted); }
.cands { display:flex; gap:6px; flex-wrap:wrap; align-items:flex-start; }
.pic { margin:0; width:180px; background:var(--card); border:1px solid var(--line);
       border-radius:6px; overflow:hidden; }
.pic.chosen { border:2px solid var(--pick); }
.pic img { display:block; width:180px; aspect-ratio:3/4; object-fit:cover; }
figcaption { display:flex; flex-wrap:wrap; gap:3px; padding:4px; }
.b { font-size:11px; padding:1px 5px; border-radius:3px; color:var(--muted);
     background:var(--line); }
.b.ok { color:var(--ok); background:var(--okbg); }
.b.bad { color:var(--bad); background:var(--badbg); }
details { margin-top:6px; } summary { cursor:pointer; color:var(--muted); }
pre { white-space:pre-wrap; font-size:11px; color:var(--muted); }
"""


def _badge(text: str, cls: str = "") -> str:
    return f'<span class="b {cls}">{html.escape(text)}</span>'


def _kit_note(render: dict[str, Any]) -> str:
    """Where the kit's colours came from, when it was not the hand-written table."""
    if render.get("kit_source") == "catalog":
        return " <i>(colours from FM27's kit image)</i>"
    if render.get("kit_source") == "bundled":
        return " <i>(bundled colours from FM27's kit image)</i>"
    return "" if render.get("kit_known", True) else " <i>(fallback kit)</i>"


def _rel(path: str | None, career_key: str) -> str:
    if not path:
        return ""
    return path.removeprefix(f"masters/{career_key}/")


def collect(conn: sqlite3.Connection, career_id: int) -> list[dict[str, Any]]:
    """Every picture of the career, newest players first, current and rejected ones."""
    rows = conn.execute(
        "SELECT i.*, p.name, p.age, p.is_goalkeeper, c.name AS club, t.sheet, t.base_seed"
        " FROM images i"
        " JOIN players p ON p.career_id = i.career_id AND p.unique_id = i.unique_id"
        " LEFT JOIN clubs c ON c.career_id = i.career_id AND c.club_id = i.club_id"
        " LEFT JOIN traits t ON t.career_id = i.career_id AND t.unique_id = i.unique_id"
        "   AND t.version = i.traits_version"
        " WHERE i.career_id = ? AND (i.is_current = 1 OR i.master_path LIKE '%-rejected.png')"
        " ORDER BY i.is_current, p.name, i.version DESC",
        (career_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def render_page(title: str, career_key: str, entries: list[dict[str, Any]]) -> str:
    blocks = []
    for e in entries:
        check = json.loads(e["check_result"] or "{}")
        render = json.loads(e["render"] or "{}")
        sheet = (
            CharacterSheet.from_json(e["sheet"], e["base_seed"]).describe() if e["sheet"] else ""
        )
        chosen = check.get("chosen", {})
        cands = []
        for c in check.get("candidates", []):
            is_pick = (c["attempt"], c["index"]) == (chosen.get("attempt"), chosen.get("index"))
            badges = [
                _badge("passed" if c["passed"] else "failed", "ok" if c["passed"] else "bad"),
                _badge(f"score {c['score']:.2f}"),
                _badge(
                    f"{c['rating']} {c['p_general']:.2f}",
                    "ok" if c["rating"] == "general" else "bad",
                ),
            ]
            if is_pick:
                badges.insert(0, _badge("chosen", "ok"))
            badges += [_badge(f, "bad") for f in c.get("failures", [])]
            tip = html.escape(
                f"seed {c['seed']}\ntraits {json.dumps(c.get('traits', {}))}\n"
                f"tags {', '.join(list(c.get('tags', {}))[:40])}"
            )
            cands.append(
                f'<figure class="pic{" chosen" if is_pick else ""}">'
                f'<a href="{html.escape(_rel(c["path"], career_key))}" target="_blank">'
                f'<img src="{html.escape(_rel(c["path"], career_key))}" loading="lazy" '
                f'title="{tip}" alt=""></a><figcaption>{"".join(badges)}</figcaption></figure>'
            )
        detail = render.get("detail")
        status = (
            _badge("published", "ok")
            if e["published_path"]
            else _badge("not published", "bad" if not e["is_current"] else "")
        )
        lines = [
            f"<h2>{html.escape(e['name'])}</h2>",
            (
                f'<div class="muted">{e["unique_id"]} · age {e["age"]} · '
                f"{html.escape(e['club'] or 'no club')}</div>"
            ),
            (
                f"<div>{status} {_badge('group ' + e['group_name'].upper())} "
                f"{_badge(str(e['age_band']))} {_badge(e['reason'])}</div>"
            ),
            f"<div>kit: {html.escape(str(render.get('kit', '')))}"
            + _kit_note(render)
            + f"<br>pose: {html.escape(str(render.get('pose', '')))}"
            + (
                f" · backdrop: {html.escape(str(render['backdrop']))}"
                if render.get("backdrop")
                else ""
            )
            + "</div>",
            f'<div class="muted">{html.escape(sheet)}</div>',
        ]
        if detail:
            lines.append(
                f"<div>detail pass: {'used' if detail.get('used') else 'rejected by the check'}"
                f", {detail.get('seconds', '?')} s</div>"
            )
        reference = render.get("reference")
        if reference:
            lines.append(
                f"<div>reference: v{reference['version']} at {reference['scale']}</div>"
                if reference.get("used")
                else f"<div>reference: none ({html.escape(str(reference.get('why')))})</div>"
            )
        if not check.get("passed", True):
            lines.append(f"<div>{_badge(str(check.get('failures', 'failed')), 'bad')}</div>")
        lines.append(
            "<details><summary>prompt</summary><pre>"
            + html.escape(f"{e['prompt']}\n\nnegative: {e['negative_prompt']}\nseed {e['seed']}")
            + "</pre></details>"
        )
        master = html.escape(_rel(e["master_path"], career_key))
        blocks.append(
            '<div class="player">'
            f'<div class="main"><a href="{master}" target="_blank">'
            f'<img src="{master}" loading="lazy" alt=""></a></div>'
            f'<div class="info">{"".join(lines)}</div>'
            f'<div class="cands">{"".join(cands)}</div></div>'
        )
    done = sum(1 for e in entries if e["is_current"])
    rejected = len(entries) - done
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Generated pictures</title><style>{_CSS}</style></head><body>
<h1>Generated pictures · {html.escape(title)}</h1>
<p class="note">{done} current pictures, {rejected} sent to review. Left: the picture that is
(or would be) published. Right: every candidate with its check; the chosen one is outlined.
Hover a candidate for its seed, trait confidences and tags. Local file only; these pictures
do not leave this PC.</p>
{"".join(blocks)}
</body></html>"""


def write_gallery(
    conn: sqlite3.Connection, career_id: int, career_key: str, title: str, data_dir: Path
) -> Path:
    page = render_page(title, career_key, collect(conn, career_id))
    path = data_dir / "masters" / career_key / "index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path
