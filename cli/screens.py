from rich import box
from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from wakeforge.config import ConfigError, load_config

from . import charts, data_tools, device_store, examples, picker, runner, ui
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


def _formats():
    return ", ".join(sorted(ext.lstrip(".") for ext in data_tools.AUDIO_EXTS))


def import_flow(project):
    """Añadir audio: una carpeta (se copia su contenido) o archivos sueltos."""
    ui.screen(project, "Import audio")
    category = ui.select("What are these recordings?", [
        ("My wake word", "wake_word"),
        ("Other sounds (negatives)", "negatives"),
        ("Background noise", "noise"),
    ])
    if category is None:
        return
    label = CATEGORY_LABELS[category]
    paths = _choose_sources(project, label)
    if paths:
        _import_all(project, category, paths)


def _choose_sources(project, label):
    """Cómo añadir el audio. Devuelve una lista de rutas, o None si el usuario vuelve atrás."""
    native = picker.available()
    while True:
        if not native:  # sin selector del sistema (p. ej. SSH): directamente arrastrar o pegar
            return _drop_prompt(project, label)
        ui.screen(project, f"Import audio · {label}")
        ui.hint(f"Accepted formats: {_formats()}. You are responsible for the audio you add.\n")
        method = ui.select(f"How do you want to add audio for “{label}”?", [
            (f"▤  Choose a folder in {picker.native_name()}", "folder"),
            (f"▤  Choose files in {picker.native_name()}", "files"),
            ("⇣  Drag & drop, or paste a path", "drop"),
            ("←  Back", "back"),
        ], default="folder")
        if method in (None, "back"):
            return None
        if method == "drop":
            paths = _drop_prompt(project, label)
            if paths:
                return paths
            continue
        ui.screen(project, f"Import audio · {label}")
        ui.panel(Align.center(Text(f"The {picker.native_name()} window is open.\nChoose what to add, or cancel to go back.",
                                   justify="center")), "Waiting for your selection")
        try:
            paths = picker.pick(method, f"Choose audio for {label}")
        except picker.PickerUnavailable as exc:
            native = False
            ui.flash("warn", f"Couldn't open the file picker ({exc}). Drag the files here instead.")
            continue
        if paths:
            return paths


def _drop_prompt(project, label):
    """Arrastrar archivos o una carpeta a la terminal: el sistema pega sus rutas como texto."""
    ui.screen(project, f"Import audio · {label} · Drag & drop")
    ui.panel(Align.center(Group(
        Text("⇣", style=f"bold {ui.FLAME}", justify="center"),
        Text("Drag your audio files or a folder onto this window,\nthen press Enter.", justify="center"),
        Text(f"\nAccepted formats: {_formats()}. You are responsible for the audio you add.\n"
             "You can also paste or type a path. Leave it empty to go back.", style=ui.MUTED, justify="center"),
    )), "Drag & drop")
    return picker.parse_dropped(ui.text("Drop here")) or None


def _import_all(project, category, paths):
    """Carpeta -> copia su contenido; archivo -> copia el archivo (lo hace data_tools.import_audios)."""
    total = {"added": 0, "duplicates": 0, "unreadable": [], "not_audio": 0, "missing": 0, "destination": None}
    with ui.console.status("[accent]Importing audio…[/accent]", spinner="dots", spinner_style=ui.FLAME):
        for path in paths:
            try:
                result = data_tools.import_audios(path, project, category)
            except FileNotFoundError:
                total["missing"] += 1
                continue
            total["added"] += result["added"]
            total["duplicates"] += result["duplicates"]
            total["not_audio"] += result["not_audio"]
            total["unreadable"] += result["unreadable"]
            total["destination"] = result["destination"]
    if total["added"]:
        ui.flash("ok", f"Added {total['added']} file(s) to {total['destination']}")
    else:
        ui.flash("warn", "Nothing new was added.")
    if total["duplicates"]:
        ui.flash("warn", f"{total['duplicates']} duplicate(s) skipped.")
    if total["unreadable"]:
        ui.flash("warn", "Could not read: " + ", ".join(total["unreadable"][:5]))
    if total["not_audio"]:
        ui.flash("warn", f"{total['not_audio']} file(s) were not audio and were skipped.")
    if total["missing"]:
        ui.flash("error", f"{total['missing']} path(s) were not found.")


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

        rec = device_store.recommended()
        recommended_item = []
        if rec and rec["viable"]:
            recommended_item = [(f"★   Train with the recommended steps ({rec['steps']}, about {rec['minutes']:.0f} min)", "rec")]
        choice = ui.select("Training", recommended_item + [
            ("▶   Train now", "full"),
            ("▷   Quick test (500 steps, just to check that everything works)", "quick"),
            ("▶▶  Long training (4000 steps)", "long"),
            ("✔   I edited config.toml, check it", "check"),
            ("←   Back", "back"),
        ], default="rec" if recommended_item else "full")
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

        overrides = {"full": [], "quick": ["training.steps=500"], "long": ["training.steps=4000"],
                     "rec": [f"training.steps={rec['steps']}"] if rec else []}[choice]
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
    """Muestra el ejemplo listo para copiar: sin bordes ni números de línea y sin recortar líneas largas.

    Si no cabe en la terminal (la pantalla alternativa no tiene scroll) lo divide en partes.
    """
    label = examples.EXAMPLES[kind][0]
    lines = examples.example_code(kind, info["path"]).rstrip("\n").split("\n")
    page = max(8, ui.console.height - 8)
    pages = [lines[i:i + page] for i in range(0, len(lines), page)]
    for n, chunk in enumerate(pages, 1):
        ui.screen(project, f"Example · {label}", compact=True)
        part = f"  [muted]part {n}/{len(pages)}[/muted]" if len(pages) > 1 else ""
        ui.console.print(f"[bold]{label}[/bold]{part}\n")
        # soft_wrap: una línea larga no se recorta ni se parte; copiar y pegar da código válido
        ui.console.print(Syntax("\n".join(chunk), "python", theme="monokai", background_color="default"),
                         soft_wrap=True)
        last = n == len(pages)
        if last:
            ui.hint("\nSelect and copy it into a .py file, then run it with python.")
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


def device_screen(project=None):
    """Muestra el último test guardado; si no existe (primera vez) lo ejecuta y lo guarda."""
    stored = device_store.load()
    if stored is None:
        stored = _run_and_store(project, first_time=True)
        if stored is None:
            return
    while True:
        if show_device(project, stored) != "rerun":
            return
        new = _run_and_store(project)
        if new is not None:
            stored = new


def _run_and_store(project, first_time=False):
    """Ejecuta el test, guarda el resultado y lo devuelve (None si falló o se canceló)."""
    ui.screen(project, "Check my computer")
    intro = "Measures how fast this computer can train and recommends how many steps to use.\nIt takes about a minute."
    if first_time:
        intro += "\nThe result is saved, so you only need to do this once."
    if project:
        intro += f"\nUsing the settings and data of [bold]{project}[/bold] for a more accurate estimate."
    ui.panel(intro, "Computer check")
    result = runner.run_device_check(project)
    if result is None:
        ui.pause()
        return None
    try:
        return device_store.save(result, project)
    except OSError as exc:  # no se pudo guardar: aun así se muestran los datos
        ui.flash("warn", f"Could not save the result ({exc}). It will be measured again next time.")
        return {"saved_at": None, "project": project, "result": result}


def _device_summary(stored):
    result = stored["result"]
    sysinfo = result["system"]
    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column(style="muted", no_wrap=True)
    table.add_column()
    ram = f"{sysinfo['ram_gb']:.1f} GB" if sysinfo.get("ram_gb") else "unknown"
    gpus = ", ".join(sysinfo["gpus"]) if sysinfo["gpus"] else "[muted]none found: it will use the CPU[/muted]"
    table.add_row("System", sysinfo["os"])
    table.add_row("CPU", f"{sysinfo['cpu']}  [muted]{sysinfo['cores']}[/muted]")
    table.add_row("Memory", ram)
    table.add_row("GPU", gpus)
    table.add_row("Software", f"Python {sysinfo['python']} · PyTorch {sysinfo['torch']}")
    table.add_row("Measured", f"{device_store.saved_label(stored)}  [muted]{result.get('source', '')}[/muted]")
    return table


def _device_speeds(result):
    best_ms = result["best"]["ms"]
    table = Table(box=None, padding=(0, 2), header_style="muted")
    table.add_column("configuration", no_wrap=True)
    table.add_column("ms / step", justify="right")
    table.add_column("speed", no_wrap=True)
    for r in result["results"]:
        if r["ms"] is None:
            table.add_row(f"[muted]{r['label']}[/muted]", "[bad]failed[/bad]", f"[muted]{r['error'] or ''}[/muted]")
            continue
        fastest = r["label"] == result["best"]["label"]
        level = "green" if fastest else "yellow"
        mark = "  [ok]◀ fastest[/ok]" if fastest else ""
        table.add_row(r["label"], f"{r['ms']:.1f}", f"{ui.meter(best_ms / r['ms'], level, 12)}{mark}")
    return table


def _device_estimate(result):
    rec = result["rec"]
    lines = [
        f"[muted]Quick test[/muted]    {rec['quick_steps']:>5} steps   about {rec['quick_minutes']:.1f} min",
        f"[accent]Recommended[/accent]  {rec['steps']:>5} steps   about {rec['minutes']:.1f} min",
    ]
    if rec["steps"] < result["max_steps"]:
        lines.append(f"[muted]The full {result['max_steps']} steps would take about {rec['max_minutes']:.0f} min[/muted]")
    lines.append("")
    if rec["viable"]:
        lines.append("[ok]✔[/ok] This computer can train the model locally.")
        lines.append("[muted]In 'Train a model' you will find an option with these steps.[/muted]")
    else:
        minimum = result["min_steps"] * rec["seconds_per_step"] / 60
        lines.append(f"[bad]✘[/bad] Not recommended: even the minimum ({result['min_steps']} steps) takes about {minimum:.0f} min.")
        lines.append("[muted]Consider training elsewhere (for example Google Colab) and bringing the .onnx file back.[/muted]")
    if result.get("bf16_slower"):
        lines.append("[warn]▲[/warn] On this GPU bf16 is slower than fp32, and train.py uses bf16 on CUDA.")
    return "\n".join(lines)


def show_device(project, stored):
    """Muestra el resultado guardado. Devuelve 'rerun' si el usuario quiere medir otra vez."""
    result = stored["result"]
    estimate = _device_estimate(result)
    verdict = "green" if result["rec"]["viable"] else "red"
    ui.screen(project, "Check my computer")
    if ui.console.height >= 36:  # cabe todo en una pantalla
        ui.panel(_device_summary(stored), "Your computer")
        ui.panel(_device_speeds(result), "Speed")
        ui.panel(estimate, "Estimate", verdict)
    else:  # terminal baja: primero la conclusión, luego los detalles
        ui.panel(estimate, "Estimate", verdict)
        ui.pause("Press Enter for the details")
        ui.screen(project, "Check my computer")
        ui.panel(_device_summary(stored), "Your computer")
        ui.panel(_device_speeds(result), "Speed")
    return ui.select("What do you want to do?", [
        ("←  Back", "back"),
        ("↻  Run the test again (about a minute)", "rerun"),
    ], default="back")