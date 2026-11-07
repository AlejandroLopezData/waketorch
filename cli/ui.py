"""Capa visual de WakeTorch: tema, prompts y helpers. Misma API que antes."""
import sys

import questionary
from questionary import Style
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.theme import Theme

# ── Paleta de fuego ──────────────────────────────────────────────
FLAME, AMBER, GOLD, EMBER = "#FF6B1A", "#FFC53D", "#FF9F1C", "#E8391D"
MUTED, TRACK = "#8A8F98", "#3A3F47"
OK, WARN, BAD = "#4ADE80", "#FFC53D", "#F87171"

console = Console(theme=Theme({
    "accent": f"bold {FLAME}",
    "gold": AMBER,
    "muted": MUTED,
    "track": TRACK,
    "ok": OK,
    "warn": WARN,
    "bad": BAD,
}))

PROMPT_STYLE = Style([
    ("qmark", f"fg:{FLAME} bold"),
    ("question", "bold"),
    ("answer", f"fg:{AMBER} bold"),
    ("pointer", f"fg:{FLAME} bold"),
    ("highlighted", f"fg:{GOLD} bold"),
    ("selected", f"fg:{AMBER}"),
    ("instruction", f"fg:{MUTED}"),
    ("muted", f"fg:{MUTED}"),
    ("separator", f"fg:{MUTED}"),
])

TONES = {"green": "ok", "yellow": "warn", "red": "bad"}
BORDERS = {"cyan": FLAME, "accent": FLAME, "green": OK, "yellow": WARN, "red": BAD}


def max_width():
    """Ancho útil: toda la terminal, con tope para pantallas muy anchas."""
    return min(console.width, 100)


def narrow():
    return console.width < 80


def tone(level):
    """'green' | 'yellow' | 'red' -> nombre de estilo del tema."""
    return TONES.get(level, level)


# ── Prompts ──────────────────────────────────────────────────────
def item(name, hint=""):
    """Título de opción con una pista en gris (para usar en select)."""
    return [("", name), ("class:muted", f"   {hint}")] if hint else name


def select(message, choices, default=None):
    items = [questionary.Choice(title=title, value=value) for title, value in choices]
    start = next((c for c in items if c.value == default), None)
    return questionary.select(message, choices=items, default=start, qmark="◆", pointer="❯",
                              instruction="(↑↓ move · enter select)", style=PROMPT_STYLE).ask()


def text(message, default=""):
    return questionary.text(message, default=default, qmark="◆", style=PROMPT_STYLE).ask()


def confirm(message, default=True):
    return questionary.confirm(message, default=default, qmark="◆", style=PROMPT_STYLE).ask()


def number(message, default):
    while True:
        raw = text(message, str(default))
        if raw is None:
            return None
        raw = raw.strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        error("Enter a whole number greater than 0.")


def pause(message="Press Enter to continue"):
    console.input(f"\n[muted]{message} ↵[/muted]")


# ── Piezas visuales ──────────────────────────────────────────────
def light(level):
    return f"[{tone(level)}]●[/{tone(level)}]"


def meter(fraction, level="green", width=12):
    """Barra fina de progreso: ━━━━━━━━────"""
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return f"[{tone(level)}]{'━' * filled}[/{tone(level)}][track]{'━' * (width - filled)}[/track]"


def panel(body, title=None, style="cyan"):
    console.print(Panel(body, title=f"[bold]{title}[/bold]" if title else None, title_align="left",
                        border_style=BORDERS.get(style, style), box=box.ROUNDED, padding=(0, 1), width=max_width()))


def hint(message):
    console.print(f"[muted]{message}[/muted]")


def brand(project=None):
    label = "[accent]▲ WakeTorch[/accent]"
    if project:
        label += f" [muted]›[/muted] [bold]{project}[/bold]"
    console.print(Rule(label, align="left", style=FLAME, characters="─"))


_FLASH = []  # mensajes breves que se muestran al principio de la SIGUIENTE pantalla


def flash(kind, message):
    """Guarda un mensaje ('ok' | 'warn' | 'error') para mostrarlo arriba en la próxima pantalla."""
    _FLASH.append((kind, message))


def flush_flash():
    if not _FLASH:
        return
    icons = {"ok": "[ok]✔[/ok]", "warn": "[warn]▲[/warn]", "error": "[bad]✘[/bad]"}
    for kind, message in _FLASH:
        console.print(f"{icons.get(kind, '•')} {message}")
    console.print()
    _FLASH.clear()


def screen(project=None, section=None, big=False):
    """Nueva pantalla: reemplaza la anterior y dibuja el logo.

    big=True usa la portada grande si la terminal es lo bastante alta; si no, la cabecera compacta.
    """
    from . import cover  # import tardío: cover importa ui

    sync = console.is_terminal
    if sync:  # "synchronized output": el terminal pinta el cambio de golpe, sin parpadeo
        sys.stdout.write("\x1b[?2026h")
    console.clear()
    if big and console.width >= 76 and console.height >= 40:
        cover.full(project, section)
    else:
        cover.header(project, section)
    if sync:
        sys.stdout.write("\x1b[?2026l")
        sys.stdout.flush()
    flush_flash()


def success(message):
    console.print(f"[ok]✔[/ok] {message}")


def warn(message):
    console.print(f"[warn]▲[/warn] {message}")


def error(message):
    console.print(f"[bad]✘[/bad] {message}")