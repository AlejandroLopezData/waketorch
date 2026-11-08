"""Portada de WakeTorch, al estilo de Claude Code: caja redondeada, mascota y consejos.

El diseño se adapta al ancho real de la terminal cada vez que se dibuja:
  >= 76 columnas  ->  dos columnas (mascota | consejos)
  44-75           ->  una columna apilada
  < 44            ->  texto simple sin caja
"""
from rich import box
from rich.align import Align
from rich.console import Group
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .ui import FLAME, MUTED, TRACK, console, flush_flash, max_width

TAGLINE = "Any language. Your voice. Your model."
VERSION = "0.1.0"
GREY = "#B4B8BF"

# Mascota: una antorcha pequeña y sólida (8 columnas)
ORANGE = "#EE4C2C"

ORANGE = "#EE4C2C"

MASCOT = [
    ("        ▗▞▀  ", ORANGE),
    ("      ▗▞▀  ▄  ", ORANGE),
    ("    ▗▞▀   ▗██  ", ORANGE),
    ("  ▗▞▀      ██▌  ", ORANGE),
    (" ▐▛       ▐██▌  ", ORANGE),
    (" █▌       ███   ", ORANGE),
    (" ▝█▄    ▄██▘   ", ORANGE),
    ("  ▝▀███▀▘    ", ORANGE),
]
TIPS = [
    "Record your wake word",
    "Add other sounds (negatives)",
    "Train: it takes a few minutes",
    "Try it live with your mic",
]


def _mascot():
    t = Text()  # sin justify: Align centra el bloque entero y no lo deforma
    for line, color in MASCOT:
        t.append(line + "\n", style=color)
    t.rstrip()
    return t


def _tips():
    t = Text()
    for i, tip in enumerate(TIPS, 1):
        t.append(f"{i}  ", style=f"bold {FLAME}")
        t.append(tip + ("\n" if i < len(TIPS) else ""))
    return t


def _title(version):
    return Text.assemble(("WakeTorch ", "bold"), (f"v{version}", MUTED))


def _panel(body, version, padding=(1, 2)):
    return Panel(body, title=_title(version), title_align="left", border_style=FLAME,
                 box=box.ROUNDED, padding=padding, width=max_width())


def _wide(version):
    left = Group(
        Align.center(Text("Welcome back!", style="bold")),
        Text(""),
        Align.center(_mascot()),
        Text(""),
        Align.center(Text("Any language. Your voice.\nYour model.", style=MUTED, justify="center")),
    )
    right = Group(
        Text("Tips for getting started", style=f"bold {FLAME}"),
        _tips(),
        Text(""),
        Rule(style=TRACK, characters="─"),
        Text("What you get", style=f"bold {FLAME}"),
        Text("A light .onnx model that runs\non any CPU: PC, Raspberry Pi…", style=MUTED),
    )
    height = 13  # alto de la columna izquierda (para la línea divisoria)
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(ratio=1, vertical="middle")
    grid.add_column(width=1)
    grid.add_column(ratio=1, vertical="middle")
    grid.add_row(left, Text("\n".join("│" * 1 for _ in range(height)), style=TRACK), right)
    return _panel(grid, version)


def _stacked(version):
    body = Group(
        Align.center(_mascot()),
        Text(""),
        Align.center(Text("Welcome back!", style="bold")),
        Align.center(Text(TAGLINE, style=MUTED)),
        Text(""),
        Text("Tips for getting started", style=f"bold {FLAME}"),
        _tips(),
    )
    return _panel(body, version)


def _tiny(version):
    t = Text()
    t.append("\n▲ ", style=f"bold {FLAME}")
    t.append("WakeTorch ", style="bold")
    t.append(f"v{version}\n", style=MUTED)
    t.append(f"{TAGLINE}\n", style=MUTED)
    return t


def build_cover(version):
    width = console.width
    if width >= 76:
        return _wide(version)
    if width >= 44:
        return _stacked(version)
    return _tiny(version)


# Mini antorcha de 3 líneas para la cabecera que siempre está visible
MINI = [(" ▗▖ ", FLAME), ("▐██▌", FLAME), (" ▀▀ ", "#7A7F88")]


def header(project=None, section=None, one_line=False):
    """Cabecera compacta: se redibuja al inicio de cada pantalla."""
    crumbs = [p for p in (project, section) if p]
    if one_line or console.width < 44 or console.height < 20:  # una sola línea
        t = Text.assemble(("▲ ", f"bold {FLAME}"), ("WakeTorch", "bold"))
        for part in crumbs:
            t.append(f" › {part}", style=MUTED)
        console.print(t)
        console.print(Rule(style=TRACK))
        return
    icon = Text()
    for line, color in MINI:
        icon.append(line + "\n", style=color)
    icon.rstrip()
    info = Text("\n")
    info.append("WakeTorch ", style="bold")
    info.append(f"v{VERSION}\n", style=MUTED)
    info.append(" › ".join(crumbs) or TAGLINE, style=MUTED)
    grid = Table.grid(padding=(0, 2))
    grid.add_column()
    grid.add_column()
    grid.add_row(icon, info)
    console.print(Padding(grid, (0, 1)))
    console.print(Rule(style=TRACK))


def full(project=None, section=None):
    """Portada grande + ruta (proyecto › sección) en una línea."""
    console.print(build_cover(VERSION))
    crumbs = " › ".join(p for p in (project, section) if p)
    if crumbs:
        console.print(f"[muted]{crumbs}[/muted]\n")


def show_cover(version="0.1.0"):
    """Pantalla de inicio: portada completa si hay altura; si no, la cabecera compacta."""
    global VERSION
    VERSION = version
    console.clear()
    if console.height < 30:
        header()
    else:
        console.print(build_cover(version))
    flush_flash()
    console.print("[muted]↑↓ navigate · enter select · ctrl+c exit[/muted]\n")