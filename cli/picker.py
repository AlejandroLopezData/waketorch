"""Elegir archivos o carpetas: selector nativo del sistema o rutas arrastradas a la terminal.

- macOS: Finder (osascript, viene con el sistema)
- Windows: Explorador de archivos (PowerShell, viene con el sistema)
- Linux: zenity, kdialog o tkinter, el que haya instalado
- Sin entorno gráfico (SSH, etc.): no hay selector y la app usa "arrastrar o pegar una ruta".
"""
import functools
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

AUDIO_PATTERNS = ("*.wav", "*.flac", "*.ogg", "*.mp3", "*.m4a")

_TK_SCRIPT = """
import sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
kind, prompt = sys.argv[1], sys.argv[2]
if kind == "folder":
    out = [filedialog.askdirectory(title=prompt, mustexist=True)]
else:
    out = list(filedialog.askopenfilenames(title=prompt))
print("\\n".join(p for p in out if p))
"""


class PickerUnavailable(Exception):
    """No se pudo abrir el selector de archivos del sistema."""


def native_name():
    return {"Darwin": "Finder", "Windows": "File Explorer"}.get(platform.system(), "file manager")


@functools.lru_cache(maxsize=1)
def _has_tk():
    try:
        return subprocess.run([sys.executable, "-c", "import tkinter"], capture_output=True,
                              timeout=15).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _linux_tool():
    for tool in ("zenity", "kdialog"):
        if shutil.which(tool):
            return tool
    return "tk" if _has_tk() else None


def _powershell():
    return shutil.which("powershell") or shutil.which("pwsh")


def available():
    """True si se puede abrir un selector de archivos nativo en este equipo."""
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return False  # el diálogo se abriría en la otra máquina
    system = platform.system()
    if system == "Darwin":
        return shutil.which("osascript") is not None
    if system == "Windows":
        return _powershell() is not None
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    return _linux_tool() is not None


# ───────────────────────── comandos por sistema ─────────────────────────
def _applescript(kind, prompt):
    prompt = prompt.replace("\\", "\\\\").replace('"', '\\"')
    lines = ["tell current application to activate"]
    if kind == "folder":
        lines.append(f'return POSIX path of (choose folder with prompt "{prompt}")')
    else:
        lines += [f'set picked to choose file with prompt "{prompt}" with multiple selections allowed',
                  'set out to ""', "repeat with f in picked",
                  "set out to out & POSIX path of f & linefeed", "end repeat", "return out"]
    return [part for line in lines for part in ("-e", line)]


def _powershell_script(kind, prompt):
    prompt = prompt.replace("'", "''")
    head = "[Console]::OutputEncoding = [Text.Encoding]::UTF8; Add-Type -AssemblyName System.Windows.Forms; "
    if kind == "folder":
        return head + ("$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
                       f"$d.Description = '{prompt}'; if ($d.ShowDialog() -eq 'OK') {{ $d.SelectedPath }}")
    flt = "Audio files|" + ";".join(AUDIO_PATTERNS) + "|All files|*.*"
    return head + ("$d = New-Object System.Windows.Forms.OpenFileDialog; "
                   f"$d.Title = '{prompt}'; $d.Multiselect = $true; $d.Filter = '{flt}'; "
                   "if ($d.ShowDialog() -eq 'OK') { $d.FileNames -join [Environment]::NewLine }")


def _command(kind, prompt):
    """Devuelve (comando, función que detecta 'cancelado', separador de rutas o None)."""
    system = platform.system()
    if system == "Darwin":
        return (["osascript", *_applescript(kind, prompt)],
                lambda p: p.returncode != 0 and "-128" in (p.stderr or ""), None)
    if system == "Windows":
        ps = _powershell()
        if not ps:
            raise PickerUnavailable("PowerShell not found")
        return [ps, "-NoProfile", "-STA", "-Command", _powershell_script(kind, prompt)], (lambda p: False), None
    tool = _linux_tool()
    home = str(Path.home())
    if tool == "zenity":
        cmd = ["zenity", "--file-selection", f"--title={prompt}", "--separator=|"]
        if kind == "folder":
            cmd.append("--directory")
        else:
            cmd += ["--multiple", "--file-filter=Audio files | " + " ".join(AUDIO_PATTERNS),
                    "--file-filter=All files | *"]
        return cmd, (lambda p: p.returncode == 1), "|"
    if tool == "kdialog":
        if kind == "folder":
            cmd = ["kdialog", "--title", prompt, "--getexistingdirectory", home]
        else:
            cmd = ["kdialog", "--title", prompt, "--getopenfilename", home,
                   " ".join(AUDIO_PATTERNS) + "|Audio files\n*|All files", "--multiple", "--separate-output"]
        return cmd, (lambda p: p.returncode == 1), None
    if tool == "tk":
        return [sys.executable, "-c", _TK_SCRIPT, kind, prompt], (lambda p: False), None
    raise PickerUnavailable("no file picker found (install zenity or tkinter)")


def pick(kind, prompt="Choose"):
    """Abre el selector. kind: 'folder' | 'files'.

    Devuelve una lista de rutas, o None si el usuario cancela. Lanza PickerUnavailable si no se pudo abrir.
    """
    cmd, cancelled, separator = _command(kind, prompt)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError as exc:
        raise PickerUnavailable(str(exc)) from exc
    if cancelled(proc):
        return None
    if proc.returncode != 0:
        lines = [l for l in (proc.stderr or "").splitlines() if l.strip()]
        raise PickerUnavailable(lines[-1][:100] if lines else f"exit code {proc.returncode}")
    out = proc.stdout.strip()
    parts = out.split(separator) if separator else out.splitlines()
    paths = [Path(p.strip()) for p in parts if p.strip()]
    return paths or None


# ───────────────────────── rutas arrastradas / pegadas ─────────────────────────
def _tokens(text, windows=False):
    if windows:  # rutas con \ y entre comillas dobles
        return [quoted or plain for quoted, plain in re.findall(r'"([^"]+)"|(\S+)', text)]
    try:
        return shlex.split(text)  # entiende comillas y espacios escapados con \
    except ValueError:  # comillas sin cerrar
        return text.split()


def parse_dropped(text):
    """Convierte lo que pega la terminal al arrastrar archivos en una lista de rutas.

    Entiende: varias rutas juntas, espacios escapados (\\ ), comillas simples o dobles, ~ y file://.
    """
    text = (text or "").strip()
    if not text:
        return []
    whole = Path(text.strip("\"'")).expanduser()
    if whole.exists():  # una sola ruta escrita a mano con espacios sin escapar
        return [whole]
    windows = os.name == "nt"
    paths = []
    for token in _tokens(text, windows):
        token = token.strip()
        if token.lower().startswith("file://"):
            token = unquote(urlparse(token).path)
            if windows and re.match(r"^/[A-Za-z]:", token):
                token = token[1:]
        if token:
            paths.append(Path(token).expanduser())
    return paths