from rich import box
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from wakeforge.config import ConfigError, load_config

from . import charts, data_tools, examples, runner, ui
from .versions import project_config_path, version_infos, version_label

CATEGORY_LABELS = {
    "wake_word": "Wake word",
    "negatives": "Other sounds",
    "noise": "Background noise",
}
CATEGORY_NOTES = {
    "wake_word": "aim for 30-50 takes",
    "negatives": "aim for 10+ minutes",
    "noise": "a few clips of your room",
}
# Meta visual de cada barra de progreso: (campo, objetivo)
GOALS = {"wake_word": ("files", 40), "negatives": ("minutes", 10), "noise": ("files", 3)}
TIPS = {
    "wake_word": "Say your wake word naturally. Vary the volume, the distance to the microphone and your tone of voice.",
    "negatives": "Record things that are NOT your wake word: normal speech, TV, music and words that sound similar.",
    "noise": "Record the background noise of your place: fan, street, kitchen. Long takes are fine.",
}
TAKES_DEFAULT = {"wake_word": 40, "negatives": 40, "noise": 1}

CONFIG_OPTIONS = [
    ("training", "steps", "How long to train. More steps = longer training."),
    ("audio", "window_seconds", "Seconds of audio the model listens to (0 = automatic)."),
    ("augmentation", "noise_prob", "How often background noise is mixed into the training audio."),
    ("training", "val_fraction", "Share of your recordings kept aside to measure quality."),
    ("model", "width", "Model size. Smaller = faster, bigger = more capacity."),
]


def render_stats(stats):
    """Tabla de audio que se adapta al ancho de la terminal."""
    width = ui.console.width
    table = Table(box=None, padding=(0, 2), header_style="muted")
    table.add_column("", no_wrap=True)
    table.add_column("clips", justify="right")
    if width >= 70:
        table.add_column("minutes", justify="right")
    table.add_column("progress", no_wrap=True)
    if width >= 90:
        table.add_column("")
    for category in data_tools.CATEGORIES:
        s = stats[category]
        field, goal = GOALS[category]
        level = data_tools.level(category, s)
        row = [f"[bold]{CATEGORY_LABELS[category]}[/bold]", str(s["files"])]
        if width >= 70:
            row.append(f"{s['minutes']:.1f}")
        row.append(f"{ui.light(level)} {ui.meter(s[field] / goal, level, 12 if width >= 56 else 6)}")
        if width >= 90:
            row.append(f"[muted]{CATEGORY_NOTES[category]}[/muted]")
        table.add_row(*row)
    return table


def pick_version(project, message, version=None):
    infos = version_infos(project)
    if not infos:
        ui.flash("warn", "There is no trained model yet. Choose 'Train a model' first.")
        return None
    if version is not None:
        return next((i for i in infos if i["version"] == version), None)
    if len(infos) == 1:
        return infos[0]
    choices = [(version_label(i), i["version"]) for i in reversed(infos)]
    picked = ui.select(message, choices, default=infos[-1]["version"])
    return None if picked is None else next(i for i in infos if i["version"] == picked)


def show_version(info, project=None):
    meta = info["meta"]
    if not meta:
        ui.warn("This model has no embedded metadata (it was trained with an older version). Train again.")
        return
    v = meta.get("validation") or {}
    frr, fpr = v.get("frr"), v.get("fpr")

    explain = not ui.narrow()
    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column(style="muted", no_wrap=True)
    table.add_column()
    if explain:
        table.add_column()
    _add = table.add_row
    table.add_row = lambda a, b, c="": _add(a, b, c) if explain else _add(a, b)
    if frr is not None:
        sens = 1 - frr
        level = "green" if sens >= 0.9 else "yellow" if sens >= 0.7 else "red"
        table.add_row("Sensitivity", f"[{ui.tone(level)}]{sens * 100:.0f}%[/] {ui.meter(sens, level, 10)}",
                      "[muted]of 100 times you say it, how many are detected[/muted]")
    if fpr is not None:
        level = "green" if fpr <= 0.01 else "yellow" if fpr <= 0.05 else "red"
        table.add_row("False alarms", f"[{ui.tone(level)}]{fpr * 100:.2f}%[/]",
                      "[muted]of 100 clips that are not your wake word, how many trigger it[/muted]")
    table.add_row("Threshold", f"{meta.get('threshold', 0):.2f}", "[muted]score needed to trigger[/muted]")
    table.add_row("Window", f"{meta.get('window_seconds', 0):.2f} s", "[muted]audio the model listens to[/muted]")
    if v.get("n_val_pos") is not None:
        table.add_row("Measured on", f"{v['n_val_pos']} + {v['n_val_neg']} clips",
                      "[muted]wake word + other sounds, kept aside from training[/muted]")
    table.add_row("File", f"[gold]{info['path']}[/gold]", "")
    ui.panel(table, f"Model v{info['version']}  ·  {info['date']:%Y-%m-%d %H:%M}", "green")

    if (v.get("n_val_pos") or 0) < 20:
        ui.warn("Few validation examples: treat these numbers as a rough guide. "
                "The best check is to try it with the microphone.")
    history = meta.get("history", [])
    height = 12
    if len(history) >= 2 and ui.console.height < 36:
        # poca altura: el gráfico va en su propia pantalla
        ui.pause("Press Enter to see the loss curve")
        ui.screen(project, "Loss curve")
        height = max(5, min(12, ui.console.height - 13))
    charts.loss_curve(history, v.get("step"), height=height)


def data_screen(project):
    while True:
        ui.screen(project, "Add audio", big=True)
        ui.panel(render_stats(data_tools.category_stats(project)), "Your audio")
        action = ui.select("Add audio", [
            ("●  Record your wake word", "rec:wake_word"),
            ("●  Record other sounds (negatives)", "rec:negatives"),
            ("●  Record background noise", "rec:noise"),
            ("↧  Import audio files or a folder", "import"),
            ("←  Back", "back"),
        ])
        if action in (None, "back"):
            return
        if action.startswith("rec:"):
            category = action.split(":", 1)[1]
            ui.screen(project, f"Record · {CATEGORY_LABELS[category]}")
            ui.panel(TIPS[category], f"Tip · {CATEGORY_LABELS[category]}")
            takes = ui.number("How many takes?", TAKES_DEFAULT[category])
            if takes is not None:
                code = runner.run_recorder(project, category, takes)
                if code in (0, None):
                    ui.flash("ok", "Recording session finished")
                else:  # error: dejar leer lo que mostró el grabador
                    ui.pause()
                    ui.flash("warn", "The recorder ended with an error")
        else:
            import_flow(project)


def import_flow(project):
    ui.screen(project, "Import audio")
    category = ui.select("What are these recordings?", [
        ("My wake word", "wake_word"),
        ("Other sounds (negatives)", "negatives"),
        ("Background noise", "noise"),
    ])
    if category is None:
        return
    source = ui.text("Path to an audio file or a folder (wav, flac, ogg, mp3, m4a):")
    if not source or not source.strip():
        return
    try:
        result = data_tools.import_audios(source, project, category)
    except FileNotFoundError as exc:
        ui.flash("error", str(exc))
        return
    ui.flash("ok", f"Added {result['added']} file(s) to {result['destination']}")
    if result["duplicates"]:
        ui.flash("warn", f"{result['duplicates']} duplicate(s) skipped.")
    if result["unreadable"]:
        ui.flash("warn", "Could not read: " + ", ".join(result["unreadable"][:5]))
    if result["not_audio"]:
        ui.flash("warn", f"{result['not_audio']} file(s) were not audio and were skipped.")


def train_screen(project):
    config_path = project_config_path(project)
    while True:
        try:
            cfg, problem = load_config(config_path), None
        except ConfigError as exc:
            cfg, problem = None, str(exc)

        ui.screen(project, "Train", big=True)
        ui.console.print(f"\n[muted]Settings file[/muted]  [gold]{config_path}[/gold]")
        ui.hint("Open it in any text editor to change how the model trains.")
        if problem:
            ui.panel(problem, "config.toml has problems", "red")
        else:
            explain = not ui.narrow()
            table = Table(box=None, padding=(0, 2), header_style="muted")
            for column in ("option", "now") + (("what it does",) if explain else ()):
                table.add_column(column)
            for section, key, meaning in CONFIG_OPTIONS:
                cells = [f"[gold]{section}.{key}[/gold]", f"[bold]{getattr(getattr(cfg, section), key)}[/bold]"]
                if explain:
                    cells.append(f"[muted]{meaning}[/muted]")
                table.add_row(*cells)
            ui.panel(table, "Training settings")

        choice = ui.select("Training", [
            ("▶   Train now", "full"),
            ("▷   Quick test (500 steps, just to check that everything works)", "quick"),
            ("▶▶  Long training (4000 steps)", "long"),
            ("✔   I edited config.toml, check it", "check"),
            ("←   Back", "back"),
        ], default="full")
        if choice in (None, "back"):
            return
        if choice == "check":
            if problem:
                ui.flash("error", "Still has problems. Fix them and check again.")
            else:
                ui.flash("ok", "config.toml is valid.")
            continue
        if problem:
            ui.flash("error", "Fix config.toml before training.")
            continue

        problems = data_tools.problems_before_training(data_tools.category_stats(project))
        if problems:
            for p in problems:
                ui.flash("warn", p)
            return

        overrides = {"full": [], "quick": ["training.steps=500"], "long": ["training.steps=4000"]}[choice]
        ui.screen(project, "Training")
        result = runner.run_training(project, overrides)
        if result is None:
            ui.pause()
            continue
        ui.success(f"Training finished: model version {result['version']}")
        info = next(i for i in version_infos(project) if i["version"] == result["version"])
        show_version(info, project)
        if ui.confirm("Show how to use this model in Python?", True):
            usage_screen(project, result["version"])
        ui.flash("ok", f"Training finished: model v{result['version']}")
        return


def results_screen(project):
    ui.screen(project, "Results")
    info = pick_version(project, "Which version?")
    if info is None:
        return
    ui.screen(project, f"Results · v{info['version']}")
    show_version(info, project)
    ui.hint("Tip: 'How to use my model' shows the code to use it in your own programs.")
    ui.pause("Press Enter to go back")


def usage_screen(project, version=None):
    ui.screen(project, "Use my model")
    info = pick_version(project, "Which version do you want to use?", version)
    if info is None:
        return
    meta = info["meta"] or {}
    rate = (meta.get("input") or {}).get("sample_rate", 16000)
    while True:
        ui.screen(project, f"Use my model · v{info['version']}", big=True)
        ui.panel(
            f"The model takes [bold]mono audio at {rate} Hz[/bold] (decimal numbers between -1 and 1) "
            f"and returns a [bold]score between 0 and 1[/bold].\n"
            f"It looks at the last [bold]{meta.get('window_seconds', 0):.1f} s[/bold] of audio. "
            f"If the score is above [bold]{meta.get('threshold', 0):.2f}[/bold], your wake word was said.\n"
            f"The window, the threshold and the sample rate are stored [bold]inside the .onnx file[/bold]: "
            f"you do not have to copy them anywhere.\n\n"
            f"[muted]Install what the examples need with:  pip install onnxruntime numpy soundfile sounddevice\n"
            f"If your microphone does not support {rate} Hz, resample the audio to {rate} Hz first.[/muted]",
            f"Using model v{info['version']}")
        kind = ui.select("Show me an example", [
            (label, key) for key, (label, _) in examples.EXAMPLES.items()
        ] + [("←  Back", "back")])
        if kind in (None, "back"):
            return
        show_code(project, info, kind)


def show_code(project, info, kind):
    """Muestra el ejemplo en pantalla limpia; si no cabe, lo divide en partes."""
    label = examples.EXAMPLES[kind][0]
    lines = examples.example_code(kind, info["path"]).rstrip("\n").split("\n")
    page = max(8, ui.console.height - 12)
    pages = [lines[i:i + page] for i in range(0, len(lines), page)]
    for n, chunk in enumerate(pages, 1):
        ui.screen(project, f"Example · {label}")
        title = label + (f"  ({n}/{len(pages)})" if len(pages) > 1 else "")
        ui.console.print(Panel(Syntax("\n".join(chunk), "python", theme="monokai", line_numbers=True,
                                      start_line=(n - 1) * page + 1, background_color="default"),
                               title=f"[bold]{title}[/bold]", title_align="left",
                               border_style=ui.FLAME, box=box.ROUNDED, width=ui.max_width()))
        last = n == len(pages)
        if last:
            ui.hint("Copy it into a .py file and run it with python.")
        ui.pause("Press Enter to go back" if last else "Press Enter for the next part")


def test_screen(project):
    ui.screen(project, "Try it")
    info = pick_version(project, "Which version do you want to test?")
    if info is None:
        return
    ui.screen(project, f"Try it · v{info['version']}")
    ui.panel("Say your wake word.  Press [bold]Ctrl+C[/bold] to stop.", f"Listening · model v{info['version']}")
    code = runner.run_listen(project, info["version"])
    if code not in (0, None):
        ui.pause()  # que se pueda leer el error mostrado arriba
        ui.flash("warn", "The test ended with an error")