try:
    from rich.console import Group
    from rich.text import Text
    from wakeforge.projects import (
        create_project, find_legacy_recordings, import_legacy, list_projects, project_state,
    )

    from . import cover, data_tools, screens, ui
except ModuleNotFoundError as exc:
    # Solo faltan dependencias si el módulo que no se encuentra es externo.
    # Otros ImportError (p. ej. un nombre que no existe en ui.py) se muestran completos.
    if (exc.name or "").split(".")[0] in {"rich", "questionary", "plotext"}:
        raise SystemExit(
            f"A dependency is missing ({exc}).\n"
            "Install the CLI requirements with:  pip install rich questionary plotext"
        ) from exc
    raise

try:
    from wakeforge import __version__
except ImportError:
    __version__ = "0.1.0"

NEXT_STEP = {
    "data": "Add audio: record or import your wake word and some other sounds",
    "train": "Train your first model",
    "test": "Try your model with the microphone",
}


def banner():
    cover.show_cover(__version__)


def summary(name):
    try:
        state = project_state(name)
    except FileNotFoundError:
        return ""
    clips = sum(state["counts"].values())
    model = f"model v{state['version']}" if state["trained"] else "not trained"
    return f"{clips} clips · {model}"


def ask_new_project():
    while True:
        name = ui.text("Project name (for example: hey_computer)")
        if name is None or not name.strip():
            return None
        try:
            create_project(name.strip())
            ui.success(f"Project [bold]{name.strip()}[/bold] created")
            return name.strip()
        except (ValueError, FileExistsError) as exc:
            ui.error(str(exc))


def ask_legacy_import():
    name = ui.text("Name for the project that will hold your old recordings", "my_wakeword")
    if name is None or not name.strip():
        return None
    try:
        result = import_legacy(name.strip())
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        ui.flash("error", str(exc))
        return None
    ui.success(f"Imported your old recordings: {result['copied']}")
    return name.strip()


def choose_project():
    while True:
        banner()
        names = list_projects()
        choices = [(ui.item(name, summary(name)), name) for name in names]
        choices.append(("＋  New project", "__new__"))
        if find_legacy_recordings():
            choices.append(("↧  Import my old my_recordings/ folder", "__legacy__"))
        choices.append(("⚙  Check this computer", "__device__"))
        choices.append(("✕  Quit", "__quit__"))

        picked = ui.select("Choose a project", choices, default=names[0] if names else "__new__")
        if picked in (None, "__quit__"):
            return None
        if picked == "__new__":
            name = ask_new_project()
        elif picked == "__legacy__":
            name = ask_legacy_import()
        elif picked == "__device__":
            screens.device_screen()
            name = None
        else:
            name = picked
        if name:
            return name


def recommended(state):
    counts = state["counts"]
    if counts["wake_word"] < 4 or counts["negatives"] < 1:
        return "data"
    return "test" if state["trained"] else "train"


def model_line(state):
    if not state["trained"]:
        return "[muted]Model[/muted]   [muted]not trained yet[/muted]"
    m = state["metrics"] or {}
    parts = [f"[accent]v{state['version']}[/accent]"]
    if m.get("frr") is not None and m.get("fpr") is not None:
        parts.append(f"sensitivity [ok]{(1 - m['frr']) * 100:.0f}%[/ok]")
        parts.append(f"false alarms [ok]{m['fpr'] * 100:.1f}%[/ok]")
    parts.append(f"[muted]{len(state['versions'])} version(s)[/muted]")
    return "[muted]Model[/muted]   " + "  ·  ".join(parts)


def show_status(project, state):
    ui.console.print()
    ui.panel(Group(screens.render_stats(data_tools.category_stats(project)), Text(""),
                   Text.from_markup(model_line(state))), project)
    if not state["config_ok"]:
        ui.panel(state["config_error"], "config.toml has problems", "red")
    ui.console.print(f"[accent]➜ Next step[/accent]  {NEXT_STEP[recommended(state)]}\n")


def hub(project):
    actions = {
        "data": screens.data_screen,
        "train": screens.train_screen,
        "results": screens.results_screen,
        "test": screens.test_screen,
        "usage": screens.usage_screen,
        "device": screens.device_screen,
    }
    while True:
        ui.screen(project, big=True)
        state = project_state(project)
        show_status(project, state)
        choice = ui.select("What do you want to do?", [
            ("1  Add audio (record or import)", "data"),
            ("2  Train a model", "train"),
            ("3  See the results", "results"),
            ("4  Try the model with the microphone", "test"),
            ("5  How to use my model", "usage"),
            ("6  Check my computer", "device"),
            ("←  Back to projects", "back"),
        ], default=recommended(state))
        if choice in (None, "back"):
            return
        actions[choice](project)


def main():
    # Pantalla alternativa: la app ocupa toda la terminal y, al salir, tu terminal
    # queda como estaba antes, sin el historial de pasos.
    alt = ui.console.set_alt_screen(True)
    try:
        while True:
            project = choose_project()
            if project is None:
                break
            hub(project)
    except KeyboardInterrupt:
        pass
    finally:
        if alt:
            ui.console.set_alt_screen(False)
    ui.console.print("[muted]See you soon.[/muted]")


if __name__ == "__main__":
    main()