"""Gráfico de pérdida dibujado con caracteres braille (sin plotext: no depende de su versión)."""
from rich.console import Group
from rich.table import Table
from rich.text import Text

from . import ui
from .ui import AMBER, FLAME, MUTED, console

# bit de cada punto dentro de una celda braille de 2 columnas x 4 filas
_BITS = {(0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (1, 0): 0x08,
         (0, 3): 0x40, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80}
LABEL_W = 8  # ancho de la columna de valores del eje Y


def _history_table(history):
    table = Table(title="Training history", box=None, header_style="muted", padding=(0, 2))
    for column in ("step", "train loss", "validation loss", "FRR", "FPR"):
        table.add_column(column, justify="right")
    for h in history:
        table.add_row(str(h["step"]), f"{h['train_loss']:.4f}", f"{h['val_loss']:.4f}",
                      f"{h['frr']:.3f}", f"{h['fpr']:.4f}")
    console.print(table)


def _line(x0, y0, x1, y1):
    """Puntos de una recta entre dos puntos (Bresenham)."""
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            return
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _render(steps, series, best_step, width, height):
    """series: [(valores, estilo)] -> Text con el gráfico completo."""
    cols, rows = width * 2, height * 4
    all_y = [v for values, _ in series for v in values]
    data_min = min(all_y)  # mínimo real de los datos, antes de añadir márgenes
    ymin, ymax = data_min, max(all_y)
    if ymax - ymin < 1e-9:
        ymax, ymin = ymax + 0.5, ymin - 0.5
    pad = (ymax - ymin) * 0.05
    ymin, ymax = ymin - pad, ymax + pad
    if data_min >= 0:  # una loss no puede ser negativa: el margen no debe bajar de cero
        ymin = max(ymin, 0.0)
    xmin, xmax = steps[0], steps[-1]

    def px(x):
        return round((x - xmin) / ((xmax - xmin) or 1) * (cols - 1))

    def py(y):
        return rows - 1 - round((y - ymin) / (ymax - ymin) * (rows - 1))

    cells = {}  # (cx, cy) -> [máscara, estilo]

    def dot(x, y, style):
        key = (x // 2, y // 4)
        cell = cells.setdefault(key, [0, style])
        cell[0] |= _BITS[(x % 2, y % 4)]
        cell[1] = style

    if best_step is not None and xmin <= best_step <= xmax:  # línea punteada del mejor paso
        bx = px(best_step)
        for y in range(0, rows, 3):
            dot(bx, y, MUTED)
    for values, style in series:
        pts = [(px(x), py(v)) for x, v in zip(steps, values)]
        for (a, b), (c, d) in zip(pts, pts[1:]):
            for x, y in _line(a, b, c, d):
                dot(x, y, style)
        if len(pts) == 1:
            dot(*pts[0], style)

    out = Text()
    for r in range(height):
        value = ymax - (ymax - ymin) * (r / max(height - 1, 1))
        label = f"{value:.3f}" if r in (0, height // 2, height - 1) else ""
        out.append(label.rjust(LABEL_W) + " │", style=MUTED)
        for c in range(width):
            mask, style = cells.get((c, r), (0, None))
            out.append(chr(0x2800 + mask) if mask else " ", style=style)
        out.append("\n")
    out.append(" " * LABEL_W + " └" + "─" * width + "\n", style=MUTED)
    left, right = f"step {steps[0]}", f"{steps[-1]}"
    out.append(" " * (LABEL_W + 2) + left + right.rjust(width - len(left)) + "\n", style=MUTED)
    out.append(" " * (LABEL_W + 2))
    out.append("━ ", style=FLAME)
    out.append("training   ")
    out.append("━ ", style=AMBER)
    out.append("validation")
    if best_step is not None:
        out.append("   ┆ ", style=MUTED)
        out.append(f"best (step {best_step})", style=MUTED)
    return out


def loss_curve(history, best_step=None, height=12):
    if len(history) < 2:
        ui.hint("Not enough evaluation points to draw a curve.")
        return
    width = min(ui.max_width() - 6 - LABEL_W - 2, 72)  # borde + relleno del panel + eje Y
    if width < 16:  # terminal demasiado estrecha: mejor la tabla
        _history_table(history)
        return
    try:
        steps = [h["step"] for h in history]
        chart = _render(steps, [([h["train_loss"] for h in history], FLAME),
                                ([h["val_loss"] for h in history], AMBER)],
                        best_step, width, height)
        ui.panel(Group(chart), "Loss during training")
    except Exception as exc:  # nunca dejar la pantalla sin información
        _history_table(history)
        ui.hint(f"(chart unavailable: {exc})")