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
name = sys.argv[3] if len(sys.argv) > 3 else ""
start = sys.argv[4] if len(sys.argv) > 4 else ""
if kind == "folder":
    out = [filedialog.askdirectory(title=prompt, mustexist=True)]
elif kind == "save":
    out = [filedialog.asksaveasfilename(title=prompt, initialfile=name, initialdir=start or None,
                                        defaultextension=".onnx", filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")])]
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
def _esc(text):
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def _applescript(kind, prompt, name=None, start=None):
    prompt = _esc(prompt)
    lines = ["tell current application to activate"]
    if kind == "save":
        location = f' default location ((POSIX file "{_esc(start)}") as alias)' if start else ""
        lines += [f'set f to choose file name with prompt "{prompt}" default name "{_esc(name or "")}"{location}',
                  "return POSIX path of f"]
    elif kind == "folder":
        lines.append(f'return POSIX path of (choose folder with prompt "{prompt}")')
    else:
        lines += [f'set picked to choose file with prompt "{prompt}" with multiple selections allowed',
                  'set out to ""', "repeat with f in picked",
                  "set out to out & POSIX path of f & linefeed", "end repeat", "return out"]
    return [part for line in lines for part in ("-e", line)]


def _powershell_script(kind, prompt, name=None, start=None):
    prompt = prompt.replace("'", "''")
    head = "[Console]::OutputEncoding = [Text.Encoding]::UTF8; Add-Type -AssemblyName System.Windows.Forms; "
    if kind == "save":
        name = str(name or "").replace("'", "''")
        start = str(start or "").replace("'", "''")
        return head + ("$d = New-Object System.Windows.Forms.SaveFileDialog; "
                       f"$d.Title = '{prompt}'; $d.FileName = '{name}'; $d.InitialDirectory = '{start}'; "
                       "$d.Filter = 'ONNX model|*.onnx|All files|*.*'; $d.OverwritePrompt = $true; "
                       "if ($d.ShowDialog() -eq 'OK') { $d.FileName }")
    if kind == "folder":
        return head + ("$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
                       f"$d.Description = '{prompt}'; if ($d.ShowDialog() -eq 'OK') {{ $d.SelectedPath }}")
    flt = "Audio files|" + ";".join(AUDIO_PATTERNS) + "|All files|*.*"
    return head + ("$d = New-Object System.Windows.Forms.OpenFileDialog; "
                   f"$d.Title = '{prompt}'; $d.Multiselect = $true; $d.Filter = '{flt}'; "
                   "if ($d.ShowDialog() -eq 'OK') { $d.FileNames -join [Environment]::NewLine }")


def _command(kind, prompt, name=None, start=None):
    """Devuelve (comando, función que detecta 'cancelado', separador de rutas o None)."""
    system = platform.system()
    if system == "Darwin":
        return (["osascript", *_applescript(kind, prompt, name, start)],
                lambda p: p.returncode != 0 and "-128" in (p.stderr or ""), None)
    if system == "Windows":
        ps = _powershell()
        if not ps:
            raise PickerUnavailable("PowerShell not found")
        return [ps, "-NoProfile", "-STA", "-Command", _powershell_script(kind, prompt, name, start)], (lambda p: False), None
    tool = _linux_tool()
    home = str(Path.home())
    if tool == "zenity":
        cmd = ["zenity", "--file-selection", f"--title={prompt}", "--separator=|"]
        if kind == "save":
            cmd += ["--save", "--confirm-overwrite", f"--filename={Path(start or home) / (name or '')}"]
        elif kind == "folder":
            cmd.append("--directory")
        else:
            cmd += ["--multiple", "--file-filter=Audio files | " + " ".join(AUDIO_PATTERNS),
                    "--file-filter=All files | *"]
        return cmd, (lambda p: p.returncode == 1), "|"
    if tool == "kdialog":
        if kind == "save":
            cmd = ["kdialog", "--title", prompt, "--getsavefilename", str(Path(start or home) / (name or "")),
                   "*.onnx|ONNX model"]
        elif kind == "folder":
            cmd = ["kdialog", "--title", prompt, "--getexistingdirectory", home]
        else:
            cmd = ["kdialog", "--title", prompt, "--getopenfilename", home,
                   " ".join(AUDIO_PATTERNS) + "|Audio files\n*|All files", "--multiple", "--separate-output"]
        return cmd, (lambda p: p.returncode == 1), None
    if tool == "tk":
        return [sys.executable, "-c", _TK_SCRIPT, kind, prompt, name or "", str(start or "")], (lambda p: False), None
    raise PickerUnavailable("no file picker found (install zenity or tkinter)")


def pick(kind, prompt="Choose", name=None, start=None):
    """Abre el selector. kind: 'folder' | 'files' | 'save'.

    Devuelve una lista de rutas, o None si el usuario cancela. Lanza PickerUnavailable si no se pudo abrir.
    """
    cmd, cancelled, separator = _command(kind, prompt, name, start)
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


def pick_save(name, start=None, prompt="Save as"):
    """Ventana de "Guardar como" del sistema. Devuelve la ruta elegida (carpeta + nombre) o None si cancela."""
    paths = pick("save", prompt, name=name, start=start)
    return paths[0] if paths else None


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