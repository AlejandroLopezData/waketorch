import ast
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import get_args, get_origin

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class ConfigError(ValueError):
    pass


def get_unique_project_name(output_dir: str | Path, name: str) -> str:
    output_dir = Path(output_dir)

    candidate = name
    n = 1

    while (output_dir / candidate).exists():
        candidate = f"{name}({n})"
        n += 1

    return candidate


@dataclass
class ProjectCfg:
    name: str
    recordings_dir: str
    output_dir: str


@dataclass
class AudioCfg:
    sample_rate: int
    window_seconds: float
    window_margin: float
    window_min: float
    window_max: float
    window_percentile: float
    trim_silence: bool
    trim_top_db: float


@dataclass
class FeaturesCfg:
    n_mels: int
    n_fft: int
    win_length: int
    hop_length: int
    f_min: float
    f_max: float


@dataclass
class ModelCfg:
    width: int
    kernel_size: int
    dilations: list[int]
    dropout: float


@dataclass
class TrainingCfg:
    steps: int
    batch_size: int
    lr: float
    weight_decay: float
    val_fraction: float
    positive_ratio: float
    label_smoothing: float
    ema_decay: float
    eval_every: int
    max_fpr: float
    min_threshold: float
    grad_clip: float
    negative_stride: float
    seed: int
    num_workers: int
    device: str


@dataclass
class AugmentationCfg:
    gain_db: float
    noise_prob: float
    snr_min_db: float
    snr_max_db: float
    speed_factors: list[float]
    speed_prob: float
    freq_mask: int
    time_mask: int


@dataclass
class ExportCfg:
    filename: str
    opset: int
    verify: bool


SECTIONS = {
    "project": ProjectCfg,
    "audio": AudioCfg,
    "features": FeaturesCfg,
    "model": ModelCfg,
    "training": TrainingCfg,
    "augmentation": AugmentationCfg,
    "export": ExportCfg,
}


@dataclass
class Config:
    project: ProjectCfg
    audio: AudioCfg
    features: FeaturesCfg
    model: ModelCfg
    training: TrainingCfg
    augmentation: AugmentationCfg
    export: ExportCfg

    def to_dict(self) -> dict:
        return asdict(self)


_TYPE_NAMES = {int: "an integer", float: "a number", bool: "true or false", str: "a string"}
_TYPE_PLURALS = {int: "integers", float: "numbers", bool: "booleans", str: "strings"}


def _type_name(tp) -> str:
    if get_origin(tp) is list:
        return f"a list of {_TYPE_PLURALS[get_args(tp)[0]]}"
    return _TYPE_NAMES[tp]


def _convert(value, tp):
    if get_origin(tp) is list:
        if not isinstance(value, list):
            raise TypeError
        return [_convert(v, get_args(tp)[0]) for v in value]
    if tp is bool:
        if isinstance(value, bool):
            return value
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif tp is str:
        if isinstance(value, str):
            return value
    raise TypeError


def _parse_override(text: str):
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _build_section(name: str, cls, data, problems: list[str]):
    if not isinstance(data, dict):
        problems.append(f"[{name}] must be a table section")
        return None
    valid = {f.name: f.type for f in fields(cls)}
    start = len(problems)
    for key in valid:
        if key not in data:
            problems.append(f"[{name}] missing option '{key}' ({_type_name(valid[key])})")
    for key in data:
        if key not in valid:
            problems.append(f"[{name}] unknown option '{key}'")
    values = {}
    for key, tp in valid.items():
        if key in data:
            try:
                values[key] = _convert(data[key], tp)
            except TypeError:
                problems.append(f"[{name}] {key}: expected {_type_name(tp)}, got {data[key]!r}")
    return cls(**values) if len(problems) == start else None


def load_config(path: str | Path, overrides: list[str] | tuple = ()) -> Config:
    if path is None:
        raise ConfigError("A path to config.toml is required (there are no default values).")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path} is not valid TOML: {e}") from e

    problems: list[str] = []
    for item in overrides:
        key, sep, val = item.partition("=")
        section, dot, name = key.strip().partition(".")
        if not sep or not dot:
            problems.append(f"invalid --set '{item}': use the format section.option=value")
            continue
        raw.setdefault(section, {})
        if isinstance(raw[section], dict):
            raw[section][name] = _parse_override(val)

    for section in raw:
        if section not in SECTIONS:
            problems.append(f"unknown section [{section}] (valid: {', '.join(SECTIONS)})")

    built = {}
    for name, cls in SECTIONS.items():
        if name not in raw:
            problems.append(f"missing section [{name}]")
            continue
        built[name] = _build_section(name, cls, raw[name], problems)

    if problems:
        raise ConfigError(f"Invalid config ({path}):\n  - " + "\n  - ".join(problems))

    cfg = Config(**built)
    base = path.resolve().parent
    for attr in ("recordings_dir", "output_dir"):
        folder = Path(getattr(cfg.project, attr)).expanduser()
        setattr(cfg.project, attr, str(folder if folder.is_absolute() else base / folder))
    validate(cfg, path)
    return cfg


def validate(cfg: Config, path: str | Path = "config") -> None:
    p, a, f, m, t, g, e = (cfg.project, cfg.audio, cfg.features, cfg.model,
                           cfg.training, cfg.augmentation, cfg.export)
    bad = []
    check = lambda ok, msg: None if ok else bad.append(msg)

    check(p.name.strip() != "" and not any(c in p.name for c in '/\\:*?"<>|'),
          "project.name must be a valid folder name (no / \\ : * ? \" < > |)")

    check(a.sample_rate > 0, "audio.sample_rate must be > 0")
    check(a.window_seconds >= 0, "audio.window_seconds must be >= 0 (0 = automatic)")
    check(a.window_margin >= 0, "audio.window_margin must be >= 0")
    check(0 < a.window_min <= a.window_max, "audio.window_min must be > 0 and <= window_max")
    check(0 < a.window_percentile <= 100, "audio.window_percentile must be in (0, 100]")
    check(a.trim_top_db > 0, "audio.trim_top_db must be > 0")

    check(f.n_mels > 0 and f.hop_length > 0 and f.win_length > 0,
          "features: n_mels, hop_length and win_length must be > 0")
    check(f.n_fft >= f.win_length, "features.n_fft must be >= features.win_length")
    check(0 <= f.f_min < f.f_max <= a.sample_rate / 2,
          "features: requires 0 <= f_min < f_max <= sample_rate / 2")

    check(m.width > 0, "model.width must be > 0")
    check(m.kernel_size > 0 and m.kernel_size % 2 == 1, "model.kernel_size must be odd and > 0")
    check(len(m.dilations) > 0 and all(d >= 1 for d in m.dilations),
          "model.dilations must contain integers >= 1")
    check(0 <= m.dropout < 1, "model.dropout must be in [0, 1)")

    check(t.steps >= 20, "training.steps must be >= 20 (the learning-rate schedule needs a few steps to warm up)")
    check(t.batch_size >= 1 and t.eval_every >= 1, "training: batch_size and eval_every must be >= 1")
    check(t.lr > 0 and t.weight_decay >= 0, "training: lr must be > 0 and weight_decay >= 0")
    check(0 < t.val_fraction < 0.5, "training.val_fraction must be between 0 and 0.5")
    check(0 < t.positive_ratio < 1, "training.positive_ratio must be between 0 and 1")
    check(0 <= t.label_smoothing < 0.5, "training.label_smoothing must be in [0, 0.5)")
    check(0 <= t.ema_decay < 1, "training.ema_decay must be in [0, 1)")
    check(0 < t.max_fpr < 1, "training.max_fpr must be between 0 and 1")
    check(0 < t.min_threshold < 1, "training.min_threshold must be between 0 and 1")
    check(t.grad_clip > 0, "training.grad_clip must be > 0")
    check(0 < t.negative_stride <= 1, "training.negative_stride must be in (0, 1]")
    check(t.num_workers >= 0, "training.num_workers must be >= 0")
    check(t.device in ("auto", "cpu", "mps") or t.device.startswith("cuda"),
          "training.device must be 'auto', 'cpu', 'mps' or 'cuda'")

    check(g.gain_db >= 0, "augmentation.gain_db must be >= 0")
    check(0 <= g.noise_prob <= 1 and 0 <= g.speed_prob <= 1,
          "augmentation: noise_prob and speed_prob must be in [0, 1]")
    check(g.snr_min_db <= g.snr_max_db, "augmentation.snr_min_db cannot be greater than snr_max_db")
    check(len(g.speed_factors) > 0 and all(x > 0 for x in g.speed_factors),
          "augmentation.speed_factors must contain values > 0")
    check(g.freq_mask >= 0 and g.time_mask >= 0,
          "augmentation: freq_mask and time_mask must be >= 0 (0 = disabled)")

    check(e.filename.endswith(".onnx"), "export.filename must end with .onnx")
    check(e.opset >= 13, "export.opset must be >= 13")

    if bad:
        raise ConfigError(f"Inconsistent config values ({path}):\n  - " + "\n  - ".join(bad))