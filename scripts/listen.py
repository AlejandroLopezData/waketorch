import argparse
import json
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wakeforge.config import ConfigError, load_config
from wakeforge.modelfile import META_KEY, list_versions
from wakeforge.projects import PROJECTS_DIR, list_projects


def resolve_project(name):
    names = list_projects()
    available = ", ".join(names) or "none"
    if name is None:
        raise SystemExit(f"Specify a project with --project NAME (available: {available}).")
    if name not in names:
        raise SystemExit(f"Project '{name}' not found in {PROJECTS_DIR} (available: {available}).")
    return name


def load_model(project, version=None):
    config_path = PROJECTS_DIR / project / "config.toml"
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        raise SystemExit(f"The config of project '{project}' is invalid:\n{exc}") from exc

    versions = list_versions(cfg.project.output_dir, cfg.export.filename)
    if not versions:
        raise SystemExit(
            f"Project '{project}' has no trained model yet.\n"
            f"Train it with:  python scripts/train.py --project {project}"
        )
    chosen = version if version is not None else max(versions)
    if chosen not in versions:
        listed = ", ".join(f"v{v}" for v in sorted(versions))
        raise SystemExit(f"Version {chosen} not found for project '{project}' (available: {listed}).")
    model_path = versions[chosen]

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(f"Could not import 'onnxruntime' ({exc}).  pip install onnxruntime") from exc

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    raw = session.get_modelmeta().custom_metadata_map.get(META_KEY)
    if raw is None:
        raise SystemExit(
            f"{model_path.name} has no embedded metadata (it was exported before metadata was "
            "stored inside the model). Train again to create a new version."
        )
    return session, json.loads(raw), model_path


def import_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(
            f"Could not import 'sounddevice' ({exc}).\n"
            "Install it with:  pip install sounddevice\n"
            "On Linux, PortAudio is also required:  sudo apt install libportaudio2"
        ) from exc
    return sd


def parse_device(value):
    if value is None or str(value).strip().lower() in ("", "default"):
        return None
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def pick_capture_rate(sd, device, target):
    try:
        sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=target)
        return target
    except sd.PortAudioError:
        return int(sd.query_devices(device, "input")["default_samplerate"])


def to_model_audio(buffer, rate, target, window_samples):
    if rate != target:
        from scipy.signal import resample_poly

        g = gcd(rate, target)
        x = resample_poly(buffer, target // g, rate // g).astype(np.float32)
    else:
        x = buffer
    if len(x) < window_samples:
        return np.pad(x, (window_samples - len(x), 0))
    return x[-window_samples:]


REARM_RATIO = 0.5


class Trigger:
    def __init__(self, threshold, refractory, rearm_ratio=REARM_RATIO):
        self.threshold = threshold
        self.refractory = refractory
        self.rearm_level = threshold * rearm_ratio
        self.armed = True
        self.last = None

    def update(self, score, now):
        if not self.armed and score < self.rearm_level and now - self.last >= self.refractory:
            self.armed = True
        if self.armed and score >= self.threshold:
            self.armed = False
            self.last = now
            return True
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description="Live test of a trained wake word with the microphone.")
    ap.add_argument("-p", "--project", help="project name inside projects/")
    ap.add_argument("--version", type=int, help="model version to use (default: the latest)")
    ap.add_argument("--threshold", type=float, help="override the threshold saved with the model")
    ap.add_argument("--step", type=float, help="seconds between analyses (default: from the model metadata)")
    ap.add_argument("--cooldown", type=float, help="minimum seconds between detections (default: the model window)")
    ap.add_argument("--device", help="microphone index or name")
    args = ap.parse_args(argv)

    project = resolve_project(args.project)
    session, meta, model_path = load_model(project, args.version)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    target_rate = int(meta["input"]["sample_rate"])
    window_samples = int(meta["window_samples"])
    window_seconds = float(meta["window_seconds"])
    threshold = args.threshold if args.threshold is not None else float(meta["threshold"])
    step_seconds = args.step if args.step is not None else float(meta.get("suggested_hop_seconds", 0.1))
    refractory = args.cooldown if args.cooldown is not None else window_seconds
    trigger = Trigger(threshold, refractory)

    sd = import_sounddevice()
    device = parse_device(args.device)

    try:
        rate = pick_capture_rate(sd, device, target_rate)
    except Exception as exc:
        raise SystemExit(f"Could not access the microphone: {exc}") from exc

    if rate != target_rate:
        try:
            import scipy.signal  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                f"Your microphone does not support {target_rate} Hz (it runs at {rate} Hz) and "
                "resampling needs scipy.  pip install scipy"
            ) from exc

    step_samples = int(rate * step_seconds)
    buffer = np.zeros(int(rate * window_seconds), dtype=np.float32)

    print(f"Project  : {project}")
    print(f"Model    : {model_path.name} (version {meta.get('version', '?')})")
    print(f"Window   : {window_seconds:.2f} s | step: {step_seconds:.2f} s | threshold: {threshold:.2f}")
    print(f"Microphone at {rate} Hz" + (f" (resampled to {target_rate} Hz)" if rate != target_rate else ""))
    print("\nSay your wake word... Ctrl+C to quit.\n")

    try:
        with sd.InputStream(samplerate=rate, channels=1, dtype="float32",
                            blocksize=step_samples, device=device) as stream:
            while True:
                data, _ = stream.read(step_samples)
                chunk = data[:, 0]
                n = len(chunk)
                buffer[:-n] = buffer[n:]
                buffer[-n:] = chunk

                audio = to_model_audio(buffer, rate, target_rate, window_samples)
                score = float(np.asarray(
                    session.run([output_name], {input_name: audio.reshape(1, -1).astype(np.float32)})[0]
                ).reshape(-1)[0])

                bar = "#" * int(score * 20)
                print(f"\rScore: {score:.4f} [{bar:<20}]", end="", flush=True)

                if trigger.update(score, time.monotonic()):
                    print(f"\n\n>>> WAKE WORD DETECTED (score {score:.2f}) <<<\n")

    except KeyboardInterrupt:
        print("\n\nMicrophone closed.")
    except Exception as exc:
        print(f"\n\nERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())