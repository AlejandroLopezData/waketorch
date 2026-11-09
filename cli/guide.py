"""Guide: one compact page with advice for recording good audio.

EDIT THE TEXT BELOW (Markdown: ## titles, **bold**, - lists). The code underneath just shows it.
"""

GUIDE = """
## ● Your wake word
Recordings of **you saying it**. Aim for **30-50 takes**.
Vary the volume, the distance to the microphone and your tone of voice. Say only the wake word.

## ● Other sounds (negatives)
Everything that is **not** your wake word: normal speech, TV, music and words that sound similar.
Aim for **40-50 takes**. They are what prevents false alarms.

## ● Background noise
The sound of your place: a fan, the street, the kitchen. A few clips are enough, and long takes are fine.
Also people talking, the noise will be mixed with your wake word and negatives to make the model more robust.

## ● General advice
-More recordings are better than fewer. If you have time, record more than the minimum.
-Record in a quiet place, but don't be afraid of some background noise: it makes the model more robust.
"""

# ─────────────────────────── code (no need to edit) ───────────────────────────
from rich.markdown import Markdown  # noqa: E402
from rich.segment import Segment, Segments  # noqa: E402

from . import ui  # noqa: E402
from .ui import console  # noqa: E402


def guide_screen(project=None):
    """Shows the guide on one page; Enter goes back. Only splits into parts if the terminal is too short."""
    width = max(30, min(console.width - 4, 88))
    lines = console.render_lines(Markdown(GUIDE.strip("\n")), console.options.update(width=width), pad=False)
    page = max(8, console.height - 7)
    pages = [lines[i:i + page] for i in range(0, len(lines), page)] or [[]]
    for n, chunk in enumerate(pages, 1):
        ui.screen(project, "Quick recording guide", compact=True)
        out = []
        for line in chunk:
            out.extend(line)
            out.append(Segment.line())
        console.print(Segments(out), end="")
        ui.pause("Press Enter to go back" if n == len(pages) else "Press Enter for the next part")