import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wakeforge import load_config, train
from wakeforge.config import ConfigError
from wakeforge.modelfile import list_versions, write_metadata
from wakeforge.projects import PROJECTS_DIR, list_projects


def project_config_path(project):
    path = PROJECTS_DIR / project / "config.toml"
    if not path.is_file():
        available = ", ".join(list_projects()) or "none"
        raise SystemExit(f"Project '{project}' not found in {PROJECTS_DIR} (available: {available}).")
    return path


def _collect_metadata(stage, version, out):
    meta_path = stage / "metadata.json"
    if not meta_path.is_file():
        raise RuntimeError("Training finished without producing metadata.json.")
    meta = json.loads(meta_path.read_text())
    split_path = stage / "split.json"
    if split_path.is_file():
        meta["split"] = json.loads(split_path.read_text())
    meta["version"] = version
    meta.setdefault("config", {}).setdefault("project", {})["output_dir"] = str(out)
    return meta


def train_config(config_path, overrides=(), on_progress=None):
    base = load_config(config_path, overrides)
    out = Path(base.project.output_dir)
    latest_name = base.export.filename
    version = max(list_versions(out, latest_name), default=0) + 1
    model_name = f"{Path(latest_name).stem}_v{version}{Path(latest_name).suffix}"

    stage = out / f".run_v{version}"
    shutil.rmtree(stage, ignore_errors=True)
    cfg = load_config(config_path, [
        *overrides,
        f"project.output_dir={str(stage)!r}",
        f"export.filename={model_name!r}",
    ])

    try:
        train(cfg, on_progress)
        meta = _collect_metadata(stage, version, out)
        write_metadata(stage / model_name, meta)
        shutil.move(str(stage / model_name), str(out / model_name))
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    return {"version": version, "output_dir": out, "model": out / model_name, "metadata": meta}


def train_project(project, overrides=(), on_progress=None):
    result = train_config(project_config_path(project), overrides, on_progress)
    result["project"] = project
    return result


def emit_event(event):
    print("WF_EVENT " + json.dumps(event, default=str), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train a wake word model and export it to ONNX.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--project", help="project name inside projects/")
    group.add_argument("--config", help="path to a config.toml (instead of a project)")
    ap.add_argument("--set", action="append", default=[], metavar="section.option=value",
                    help="override an option; can be repeated")
    ap.add_argument("--no-test", action="store_true", help="do not open the microphone test afterwards")
    ap.add_argument("--progress-json", action="store_true",
                    help="print progress events as 'WF_EVENT {json}' lines (used by the CLI)")
    args = ap.parse_args(argv)

    print(f"Training project: {args.project or args.config}\n")
    on_progress = emit_event if args.progress_json else None

    try:
        if args.project:
            result = train_project(args.project, args.set, on_progress)
        else:
            result = train_config(args.config, args.set, on_progress)
    except (ConfigError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.progress_json:
        emit_event({"stage": "result", "version": result["version"], "model": str(result["model"])})

    v = result["metadata"].get("validation", {})
    print()
    print("=" * 60)
    print(f"TRAINING COMPLETED  (version {result['version']})")
    print("=" * 60)
    print(f"Model     : {result['model']}")
    print(f"Window    : {result['metadata'].get('window_seconds', 0):.2f} s | threshold: {result['metadata'].get('threshold', 0):.2f}")
    print(f"Validation: FRR {v.get('frr', float('nan')):.3f} | FPR {v.get('fpr', float('nan')):.4f}")

    if args.project and not args.no_test:
        print("\nOpening microphone test...\n")
        subprocess.run([sys.executable, str(Path(__file__).with_name("listen.py")),
                        "--project", args.project])


if __name__ == "__main__":
    main()