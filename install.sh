#!/usr/bin/env bash
# WakeTorch installer (macOS / Linux)
#   ./install.sh             install: creates .venv, installs dependencies and the `waketorch` command
#   ./install.sh uninstall   removes the command and .venv (your projects are NOT touched)
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
BIN_DIR="$HOME/.local/bin"
LINK="$BIN_DIR/waketorch"

say()  { printf '\033[38;5;208m▲\033[0m %s\n' "$1"; }
fail() { printf '\033[31m✘\033[0m %s\n' "$1" >&2; exit 1; }

if [ "${1:-}" = "uninstall" ]; then
    rm -f "$LINK"
    rm -rf "$ROOT/.venv"
    say "Removed the waketorch command and .venv. Your projects were not touched."
    exit 0
fi

# 1) Python 3.10 or newer
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 \
       && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
[ -n "$PYTHON" ] || fail "Python 3.10 or newer is required. On macOS: brew install python@3.12"
say "Using $($PYTHON --version)"

# 2) Virtual environment + dependencies (from pyproject.toml)
say "Creating the environment (.venv)"
"$PYTHON" -m venv .venv
say "Installing dependencies (PyTorch is large, this can take a few minutes)"
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -e .

# 3) `waketorch` command
mkdir -p "$BIN_DIR"
ln -sf "$ROOT/.venv/bin/waketorch" "$LINK"
say "Command created: $LINK"

# 4) Make sure ~/.local/bin is in PATH
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        case "${SHELL:-}" in
            */zsh) RC="$HOME/.zshrc" ;;
            *)     RC="$HOME/.bashrc" ;;
        esac
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$RC"
        say "Added ~/.local/bin to your PATH in $RC"
        NEEDS_RESTART=1
        ;;
esac

echo
say "Done. Run it with:  waketorch"
[ "${NEEDS_RESTART:-0}" = "1" ] && say "Open a new terminal first (or run: source $RC)"
exit 0