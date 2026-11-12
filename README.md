# FM27 Face Generator

Generates anime-style portraits for Football Manager 27 players on a local GPU and
publishes them into FM27's graphics folder: everyone at your club, the best newgens
elsewhere, and anyone you pick by hand.

> **Status: generation MVP (Phase 2).** The tool reads a save, builds the picture queue,
> draws the pictures on the GPU and publishes them into FM27 by hand (`facegen run`). The
> web UI and automation come later; see [docs/implementation-plan.md](docs/implementation-plan.md).

## What it will do

- Read the FM27 save file to find players, their club, and their current and potential
  ability.
- Give every player at your club a picture, replacing any existing photo.
- Sort newgens elsewhere into three groups (A, B, C) with different picture quality and
  queue priority.
- Give each player a stored character sheet (hair, eyes and so on) and draw an
  upper-body portrait in their club's kit.
- Re-draw tracked players (group A, plus newgens from your academy wherever they go) in
  a new kit after a transfer, keeping a similar face.
- Publish pictures where FM27 loads them, per career.
- Serve a small local web page to change group thresholds while it runs, pick
  pre-existing players to generate, and request a new face for any player.

## Requirements

- The Windows PC that runs FM27, with WSL2.
- An NVIDIA GPU with 16 GB of VRAM.
- Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync
```

```bash
uv run pytest
```

## Usage

```bash
uv run facegen sync --dry-run
```

Copies the newest save from FM27's `games` folder into `data/snapshots/`, reads the copy,
and prints the group counts and the queue it would create, storing nothing. Drop
`--dry-run` to store the result, add `--csv data/players.csv` for a per-player table, and
use `--save PATH` to pick a save.

```bash
uv run facegen status
```

Generating pictures (close FM27 first; the GPU needs the room):

```bash
uv run facegen run --group a --my-club --dry-run
```

Prints each queued job's character sheet, kit, pose and prompt without drawing anything.

```bash
uv run facegen run --group a --my-club
```

Draws the queued pictures, checks each one, keeps masters in `data/masters/<career>/`, and
publishes into `<FM27 user dir>/graphics/facegen/<career>/` with a `config.xml`. Then in
FM27: Settings → Reload Skin. Add `--no-publish` to look first, then `facegen publish`.
`--limit N` stops after N jobs; `--now` draws even while FM27 runs. Each run writes a local
page, `data/masters/<career>/index.html`, showing every picture, its candidates and checks.
Running it again changes nothing.

Kit colours per club are in `config/kits.toml`; a club not listed gets a neutral kit. Height
and build words go to the tenth of players at each end; the cut-offs are in
`config/default.toml` under `[body]`.

## Documentation

| Document | Contents |
|---|---|
| [docs/spec.md](docs/spec.md) | What the tool does: groups, picture style, updates, UI |
| [docs/architecture.md](docs/architecture.md) | Components, database, files on disk, technology |
| [docs/implementation-plan.md](docs/implementation-plan.md) | Phases, tasks and exit tests |
| [docs/research.md](docs/research.md) | Verified facts about the save format, `fmsave` and this PC |
| [docs/decisions.md](docs/decisions.md) | Decision log and open questions |
| [AGENTS.md](AGENTS.md) | Rules for coding agents working in this repository |

## Layout

```
config/default.toml   default settings, including group thresholds
config/kits.toml      kit colours per club
docs/                 specification, architecture, plan, research, decisions
src/facegen/          the Python package
tests/                unit tests
data/                 runtime state and generated pictures (created on first run, not in git)
```
