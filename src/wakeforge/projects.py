"""Gestión de proyectos: cada wake word vive en su propia carpeta.

    projects/<name>/
        config.toml          copia de config/default.toml con el nombre del proyecto
        my_recordings/       wake_word/  negatives/  noise/
        outputs/             model_v1.onnx, model_v2.onnx ... (metadata is embedded in each file)

No depende de torch (se puede usar desde el CLI sin cargar el entrenamiento).

    python -m wakeforge.projects create NAME
    python -m wakeforge.projects list
    python -m wakeforge.projects state NAME
    python -m wakeforge.projects import-legacy NAME [--source my_recordings]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from .audio import list_audio
from .config import ConfigError, load_config
from .modelfile import list_versions, read_metadata

ROOT = Path(__file__).resolve().parents[2]
PROJECTS_DIR = ROOT / "projects"
TEMPLATE = ROOT / "config" / "default.toml"
SUBDIRS = ("wake_word", "negatives", "noise")
BAD_CHARS = '/\\:*?"<>|'

RECORDINGS_README = """# my_recordings

Put your audio files here (wav, flac, ogg, mp3, m4a).

- wake_word/   your wake word, 30-50 takes (different volumes, distances and tones of voice)
- negatives/   normal speech, TV, music, words that sound similar to your wake word
- noise/       background noise from your environment (fan, street, kitchen...)

Files are never uploaded anywhere: this folder is ignored by git.
"""


def check_name(name: str) -> None:
    if not name or name != name.strip():
        raise ValueError("Project name cannot be empty or start/end with spaces.")
    if name in (".", "..") or name.startswith(".") or len(name) > 64:
        raise ValueError("Project name cannot start with '.' and must be at most 64 characters.")
    if any(c in name for c in BAD_CHARS):
        raise ValueError(f"Project name cannot contain any of: {BAD_CHARS}")


def project_path(name: str, projects_dir: Path = PROJECTS_DIR) -> Path:
    return Path(projects_dir) / name


def list_projects(projects_dir: Path = PROJECTS_DIR) -> list[str]:
    projects_dir = Path(projects_dir)
    if not projects_dir.is_dir():
        return []
    return sorted(p.name for p in projects_dir.iterdir() if (p / "config.toml").is_file())


def _render_template(template: Path, name: str) -> str:
    lines = Path(template).read_text(encoding="utf-8").split("\n")
    in_project, found = False, False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[project]":
            in_project = True
        elif stripped.startswith("["):
            in_project = False
        elif in_project and re.match(r"\s*name\s*=", line):
            lines[i] = f'name = "{name}"'
            found = True
    if not found:
        raise ConfigError(f"{template} has no 'name' option in [project].")
    return "\n".join(lines)


def create_project(name: str, projects_dir: Path = PROJECTS_DIR, template: Path = TEMPLATE) -> Path:
    check_name(name)
    path = project_path(name, projects_dir)
    if path.exists():
        raise FileExistsError(f"Project '{name}' already exists: {path}")
    if not Path(template).is_file():
        raise FileNotFoundError(f"Config template not found: {template}")

    try:
        for sub in SUBDIRS:
            (path / "my_recordings" / sub).mkdir(parents=True)
        (path / "outputs").mkdir()
        (path / "my_recordings" / "README.md").write_text(RECORDINGS_README, encoding="utf-8")
        (path / "config.toml").write_text(_render_template(template, name), encoding="utf-8")
        load_config(path / "config.toml")
    except Exception:
        shutil.rmtree(path, ignore_errors=True)
        raise
    return path


def project_state(name: str, projects_dir: Path = PROJECTS_DIR) -> dict:
    path = project_path(name, projects_dir)
    if not (path / "config.toml").is_file():
        raise FileNotFoundError(f"Project '{name}' not found in {projects_dir}")

    state = {"name": name, "path": str(path), "config_ok": True, "config_error": None,
             "counts": {}, "trained": False, "version": None, "versions": [],
             "trained_at": None, "metrics": None}
    try:
        cfg = load_config(path / "config.toml")
        recordings, outputs, model_file = Path(cfg.project.recordings_dir), Path(cfg.project.output_dir), cfg.export.filename
    except (ConfigError, OSError) as e:
        state.update(config_ok=False, config_error=str(e))
        recordings, outputs, model_file = path / "my_recordings", path / "outputs", "model.onnx"

    state["counts"] = {sub: len(list_audio(recordings / sub)) for sub in SUBDIRS}
    versions = list_versions(outputs, model_file)
    if versions:
        latest = max(versions)
        model = versions[latest]
        state.update(trained=True, version=latest, versions=sorted(versions),
                     trained_at=datetime.fromtimestamp(model.stat().st_mtime).isoformat(timespec="seconds"))
        try:
            v = read_metadata(model)["validation"]
            state["metrics"] = {k: v.get(k) for k in ("frr", "fpr", "threshold", "n_val_pos", "n_val_neg")}
        except (KeyError, TypeError, ValueError):
            pass
    return state


def find_legacy_recordings(root: Path = ROOT) -> Path | None:
    legacy = Path(root) / "my_recordings"
    return legacy if any(list_audio(legacy / sub) for sub in SUBDIRS) else None


def import_legacy(name: str, source: Path | None = None, projects_dir: Path = PROJECTS_DIR) -> dict:
    source = Path(source) if source else find_legacy_recordings()
    if source is None or not source.is_dir():
        raise FileNotFoundError("No recordings found to import (expected my_recordings/ with audio inside).")
    path = create_project(name, projects_dir)
    copied = {}
    for sub in SUBDIRS:
        src = source / sub
        if src.is_dir():
            shutil.copytree(src, path / "my_recordings" / sub, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(".gitkeep"))
        copied[sub] = len(list_audio(path / "my_recordings" / sub))
    return {"path": str(path), "copied": copied}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m wakeforge.projects")
    ap.add_argument("--projects-dir", default=str(PROJECTS_DIR))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for cmd in ("create", "state", "import-legacy"):
        p = sub.add_parser(cmd)
        p.add_argument("name")
        if cmd == "import-legacy":
            p.add_argument("--source", default=None)
    args = ap.parse_args(argv)
    pd = Path(args.projects_dir)

    if args.cmd == "list":
        for n in list_projects(pd):
            print(n)
    elif args.cmd == "create":
        print(f"Created {create_project(args.name, pd)}")
    elif args.cmd == "state":
        print(json.dumps(project_state(args.name, pd), indent=2))
    elif args.cmd == "import-legacy":
        print(json.dumps(import_legacy(args.name, args.source, pd), indent=2))


if __name__ == "__main__":
    main()