import json
import os
import subprocess
import sys
from collections import deque

from rich.table import Column
from rich.progress import (BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

from . import ROOT, ui
from .ui import AMBER, FLAME, MUTED, OK, console, panel

EVENT_PREFIX = "WF_EVENT "


def child_env():
    env = dict(os.environ)
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_interactive(cmd):
    process = subprocess.Popen(cmd, cwd=ROOT, env=child_env())
    while True:
        try:
            return process.wait()
        except KeyboardInterrupt:
            continue


def run_recorder(project, category, takes):
    return run_interactive([sys.executable, "-m", "record.recorder",
                            "-p", project, "-c", category, "-n", str(takes)])


def run_listen(project, version, device=None):
    cmd = [sys.executable, str(ROOT / "scripts" / "listen.py"),
           "--project", project, "--version", str(version)]
    if device:
        cmd += ["--device", device]
    return run_interactive(cmd)


def _warnings_from(log):
    found = []
    for line in log:
        if "Warning:" in line:
            found.append(line.split("Warning:", 1)[1].strip())
    return found


def run_training(project, overrides=()):
    cmd = [sys.executable, str(ROOT / "scripts" / "train.py"),
           "--project", project, "--no-test", "--progress-json"]
    for override in overrides:
        cmd += ["--set", override]

    process = subprocess.Popen(cmd, cwd=ROOT, env=child_env(), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               encoding="utf-8", errors="replace")
    log, result = [], None
    recent_losses = deque(maxlen=20)
    last_eval = ""

    columns = [SpinnerColumn("dots", style=FLAME), TextColumn("{task.description}", table_column=Column(no_wrap=True)),
               BarColumn(bar_width=None, complete_style=FLAME, finished_style=OK, pulse_style=AMBER,
                         style="#3A3F47"),
               TaskProgressColumn(style=AMBER)]
    if console.width >= 80:
        columns += [TextColumn(f"[{MUTED}]elapsed[/]"), TimeElapsedColumn(),
                    TextColumn(f"[{MUTED}]left[/]")]
    columns.append(TimeRemainingColumn())
    progress = Progress(*columns, console=console, expand=True)
    console.print()
    try:
        with progress:
            task = progress.add_task("[muted]Loading and preparing audio...[/muted]", total=None)
            for raw in process.stdout:
                line = raw.rstrip("\n")
                event = None
                if line.startswith(EVENT_PREFIX):
                    try:
                        event = json.loads(line[len(EVENT_PREFIX):])
                    except ValueError:
                        event = None
                if event is None:
                    log.append(line)
                    continue

                stage = event.get("stage")
                if stage == "training" and event.get("step") == 0:
                    progress.update(task, total=event["steps"], completed=0,
                                    description="[accent]Training[/accent]")
                elif stage == "training":
                    recent_losses.append(event["loss"])
                    loss = sum(recent_losses) / len(recent_losses)
                    progress.update(task, completed=event["step"],
                                    description=f"[accent]Training[/accent]  [muted]loss[/muted] {loss:.3f}{last_eval}")
                elif stage == "eval":
                    last_eval = (f"  [muted]│ val FRR[/muted] {event['frr']:.2f}"
                                 f"  [muted]FPR[/muted] {event['fpr']:.3f}") if console.width >= 90 else ""
                elif stage == "exporting":
                    progress.update(task, description="[accent]Exporting the ONNX model...[/accent]")
                elif stage == "result":
                    result = event
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        ui.warn("Training cancelled. Nothing was saved.")
        return None

    if process.returncode != 0 or result is None:
        tail = "\n".join(log[-15:]) or "(no output)"
        panel(tail, "Training failed", "red")
        return None

    warnings = _warnings_from(log)
    if warnings:
        panel("\n".join(f"• {w}" for w in dict.fromkeys(warnings)), "Things to improve", "yellow")
    return {"version": result["version"], "model": result["model"]}