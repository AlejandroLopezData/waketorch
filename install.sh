#!/usr/bin/env bash
# WakeTorch installer (macOS / Linux)
#   ./install.sh             install: creates .venv, installs dependencies and the `waketorch` command
#   ./install.sh uninstall   removes the command and .venv (your projects are NOT touched)
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
BIN_DIR="$HOME/.local/bin"
LINK="$BIN_DIR/waketorch"
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

# ── Look & feel ──────────────────────────────────────────────────────────
TTY=0; [ -t 1 ] && TTY=1
if [ "$TTY" = 1 ]; then
    ORANGE='\033[38;5;208m'; RED='\033[31m'; GREEN='\033[32m'; DIM='\033[2m'; RESET='\033[0m'
else
    ORANGE=''; RED=''; GREEN=''; DIM=''; RESET=''
fi
UTF=0
case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in *[Uu][Tt][Ff]-8* | *[Uu][Tt][Ff]8*) UTF=1 ;; esac
if [ "$UTF" = 1 ]; then
    FRAMES=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏); FULL="█"; EMPTY="░"; OK="✔"; MARK="▲"
else
    FRAMES=('|' '/' '-' '\'); FULL="#"; EMPTY="-"; OK="OK"; MARK=">"
fi

say()  { printf "${ORANGE}%s${RESET} %s\n" "$MARK" "$1"; }
fail() { printf "\r\033[K${RED}x${RESET} %s\n" "$1" >&2; exit 1; }

bar() {  # bar PERCENT  ->  ████████░░░░░░░░
    local pct=$1 width=24 filled i out=""
    filled=$((pct * width / 100))
    for ((i = 0; i < width; i++)); do
        if [ "$i" -lt "$filled" ]; then out+="$FULL"; else out+="$EMPTY"; fi
    done
    printf '%s' "$out"
}

# run "Label" START SPAN command...
# Runs the command with a spinner, the package being installed and a progress bar
# that moves from START% to START+SPAN% while pip collects packages.
run() {
    local label=$1 base=$2 span=$3
    shift 3
    : > "$LOG"
    if [ "$TTY" = 0 ]; then  # no terminal (CI, pipes): plain lines
        say "$label"
        "$@" > "$LOG" 2>&1 || { tail -n 15 "$LOG" >&2; fail "$label failed. ${HINT:-}"; }
        return 0
    fi
    "$@" > "$LOG" 2>&1 &
    local pid=$! i=0 seen pct pkg
    while kill -0 "$pid" 2>/dev/null; do
        seen=$(grep -c '^Collecting' "$LOG" 2>/dev/null || true)
        [ "$seen" -gt 30 ] && seen=30
        pct=$((base + span * seen / 30))
        pkg=$(grep '^Collecting' "$LOG" 2>/dev/null | tail -n 1 | awk '{print $2}' | sed 's/[<>=!~;(\[].*//' || true)
        printf "\r\033[K${ORANGE}%s${RESET} %s %3d%%  %s ${DIM}%s${RESET}" \
            "${FRAMES[i % ${#FRAMES[@]}]}" "$(bar "$pct")" "$pct" "$label" "$pkg"
        i=$((i + 1))
        sleep 0.1
    done
    if ! wait "$pid"; then
        printf '\r\033[K'
        tail -n 15 "$LOG" >&2
        fail "$label failed. ${HINT:-}"
    fi
    printf "\r\033[K${GREEN}%s${RESET} %s %3d%%  %s\n" "$OK" "$(bar $((base + span)))" $((base + span)) "$label"
}

# ── Uninstall ────────────────────────────────────────────────────────────
if [ "${1:-}" = "uninstall" ]; then
    rm -f "$LINK"
    rm -rf "$ROOT/.venv"
    say "Removed the waketorch command and .venv. Your projects were not touched."
    exit 0
fi

# ── 1) Python 3.10 or newer ──────────────────────────────────────────────
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" > /dev/null 2>&1 \
        && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2> /dev/null; then
        PYTHON="$candidate"
        break
    fi
done
[ -n "$PYTHON" ] || fail "Python 3.10 or newer is required. On macOS: brew install python@3.12"
echo
say "WakeTorch installer ($($PYTHON --version))"
echo

# ── 2) Environment + dependencies (from pyproject.toml) ──────────────────
HINT="On Debian/Ubuntu: sudo apt install python3-venv" run "Creating the environment" 5 10 "$PYTHON" -m venv .venv
run "Updating pip" 15 5 .venv/bin/python -m pip install --disable-pip-version-check --progress-bar off --upgrade pip
say "PyTorch is large: the dependencies can take a few minutes."
run "Installing dependencies" 20 75 .venv/bin/python -m pip install --disable-pip-version-check --progress-bar off -e .

# ── 3) `waketorch` command ───────────────────────────────────────────────
mkdir -p "$BIN_DIR"
ln -sf "$ROOT/.venv/bin/waketorch" "$LINK"

# ── 4) Make sure ~/.local/bin is in PATH ─────────────────────────────────
NEEDS_RESTART=0
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        case "${SHELL:-}" in
            */zsh) RC="$HOME/.zshrc" ;;
            *)     RC="$HOME/.bashrc" ;;
        esac
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$RC"
        NEEDS_RESTART=1
        ;;
esac
if [ "$TTY" = 1 ]; then
    printf "${GREEN}%s${RESET} %s 100%%  %s\n" "$OK" "$(bar 100)" "Creating the waketorch command"
else
    say "Creating the waketorch command"
fi

echo
say "Done. Run it with:  waketorch"
[ "$NEEDS_RESTART" = "1" ] && say "Open a new terminal first (or run: source $RC)"
[ "$(uname)" = "Linux" ] && say "If recording does not work: sudo apt install libportaudio2"
exit 0