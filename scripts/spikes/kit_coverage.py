"""How many clubs FM27's own kits cover, and how each would be drawn (read only).

Reads FM27's database from the install folder with Kitshot's reader (datasource/game_kits.py;
nothing is cached) and prints, for the clubs asked about: whether the database has exactly one
home shirt, whether its style is one Kitshot knows the look of, and the kit words. Run it after
an FM27 update and record the result in docs/research.md ("FM27's kits").

Example:
  uv run --no-sync python scripts/spikes/kit_coverage.py --clubs 1708 1736 602
  uv run --no-sync python scripts/spikes/kit_coverage.py --clubs-file clubs.txt --all
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from kitshot.core.kit_values import SHIRT_STYLES, game_kits
from kitshot.datasource.game_kits import find_database, load_game_kits
from kitshot.platform.fm_paths import detect_install_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--install-dir", type=Path, help="FM27's install folder (found if left out)"
    )
    parser.add_argument("--clubs", type=int, nargs="*", default=[], help="unique club IDs")
    parser.add_argument("--clubs-file", type=Path, help="a file of club IDs, one per line")
    parser.add_argument("--all", action="store_true", help="also count every club in the database")
    args = parser.parse_args()
    install = args.install_dir or detect_install_dir()
    db = find_database(install)
    if db is None:
        print(f"No FM27 database under {install}", file=sys.stderr)
        return 2
    kits = load_game_kits(install, None)
    if kits is None:
        print(f"FM27's database {db.version} could not be read (see the log line above)")
        return 1
    print(f"FM27's database {db.version}: {db.path}\nSHA-256 {kits.sha256}")
    print(f"{len(kits.shirts):,} clubs with one home shirt")
    if args.all:
        known = Counter(home.style in SHIRT_STYLES for home, _ in kits.shirts.values())
        print(f"  style known: {known[True]:,} ({100 * known[True] / len(kits.shirts):.1f}%)")
    clubs = list(args.clubs)
    if args.clubs_file:
        clubs += [int(x) for x in args.clubs_file.read_text().split() if x.strip()]
    if not clubs:
        return 0
    found = known = 0
    for club in clubs:
        shirts = kits.shirts.get(club)
        if shirts is None:
            print(f"{club:>10}  no home shirt in the database")
            continue
        home, keeper = shirts
        found += 1
        known += home.style in SHIRT_STYLES
        k = game_kits(home, keeper)
        hexes = "/".join("".join(f"{c:02X}" for c in rgb) for rgb in home.colours)
        mark = "" if home.style in SHIRT_STYLES else "  (style not known: no layout)"
        print(
            f"{club:>10}  style {home.style:>2} {hexes}  {k.home.shirt_words()}"
            f" | keeper {k.goalkeeper.shirt[0]}{mark}"
        )
    print(f"\n{found}/{len(clubs)} with a home shirt, {known}/{len(clubs)} with a known style")
    return 0


if __name__ == "__main__":
    sys.exit(main())
