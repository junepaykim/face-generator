#!/usr/bin/env bash
# Starts `facegen serve` and opens the web UI in the Windows browser once it answers.
# If facegen is already running, it only opens the browser.
# Meant for a Windows desktop shortcut:  wsl.exe -d <distro> -- bash <repo>/scripts/launch.sh
set -u

cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
# wsl.exe runs this without a login shell, so add the folders that hold uv and cargo.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
printf '\033]0;facegen\007'   # window title

# The UI address is fixed in config/default.toml ([ui] host and port).
port=$(awk -F= '/^\[/{ui=($0=="[ui]")} ui && $1~/^port *$/{gsub(/[ "]/,"",$2); print $2}' config/default.toml)
url="http://127.0.0.1:${port:-8765}/"

answers() { curl -s -o /dev/null --noproxy '*' --max-time 2 "$url"; }
open_browser() { (cd /mnt/c && cmd.exe /c start "" "$url" >/dev/null 2>&1); }

if answers; then
    echo "facegen is already running; opening $url"
    open_browser
    exit 0
fi

# Open the browser once the server answers (the first start after an update can take a while
# while uv rebuilds), and give up if it never does.
(
    for _ in $(seq 300); do
        sleep 1
        if answers; then open_browser; exit 0; fi
    done
) &
opener=$!

uv run facegen serve "$@"
status=$?
kill "$opener" 2>/dev/null

if [ "$status" -ne 0 ]; then
    echo
    echo "facegen serve stopped with exit code $status."
    read -r -p "Press Enter to close this window." _
fi
exit "$status"
