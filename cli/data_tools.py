import hashlib
import re
import shutil
import wave
from pathlib import Path

from wakeforge.audio import AUDIO_EXTS, list_audio
from wakeforge.config import ConfigError, load_config
from wakeforge.projects import PROJECTS_DIR, SUBDIRS

CATEGORIES = SUBDIRS
SOUNDFILE_EXTS = {".flac", ".ogg", ".mp3"}


def recordings_dir(project):
    try:
        return Path(load_config(PROJECTS_DIR / project / "config.toml").project.recordings_dir)
    except (ConfigError, OSError):
        return PROJECTS_DIR / project / "my_recordings"


def duration_seconds(path):
    path = Path(path)
    try:
        if path.suffix.lower() == ".wav":
            with wave.open(str(path), "rb") as w:
                return w.getnframes() / w.getframerate()
        if path.suffix.lower() in SOUNDFILE_EXTS:
            import soundfile as sf

            return sf.info(str(path)).duration
    except Exception:
        return None
    return None


def is_readable(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".wav" or suffix in SOUNDFILE_EXTS - {".mp3"}:
        return duration_seconds(path) is not None
    return True


def category_stats(project):
    base = recordings_dir(project)
    stats = {}
    for category in CATEGORIES:
        files = list_audio(base / category)
        seconds = sum(d for d in (duration_seconds(p) for p in files) if d)
        stats[category] = {"files": len(files), "minutes": seconds / 60}
    return stats


def level(category, stat):
    if category == "wake_word":
        return "red" if stat["files"] < 4 else "yellow" if stat["files"] < 20 else "green"
    if category == "negatives":
        return "red" if stat["minutes"] < 1 else "yellow" if stat["minutes"] < 10 else "green"
    return "yellow" if stat["files"] == 0 else "green"


def problems_before_training(stats):
    problems = []
    if stats["wake_word"]["files"] < 4:
        problems.append(f"Record at least 4 takes of your wake word (you have {stats['wake_word']['files']}).")
    if stats["negatives"]["files"] < 1:
        problems.append("Add some other sounds (normal speech, TV, music) as negatives.")
    return problems


def _sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _next_index(folder, category):
    if not Path(folder).is_dir():
        return 1
    rx = re.compile(rf"^{re.escape(category)}_(\d+)\.[^.]+$", re.IGNORECASE)
    numbers = [int(m.group(1)) for p in Path(folder).iterdir() if (m := rx.match(p.name))]
    return max(numbers, default=0) + 1


def import_audios(source, project, category):
    source = Path(str(source).strip().strip("\"'")).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Not found: {source}")
    if source.is_file():
        files = [source]
    else:
        files = sorted(p for p in source.rglob("*") if p.suffix.lower() in AUDIO_EXTS)

    dest = recordings_dir(project) / category
    dest.mkdir(parents=True, exist_ok=True)
    seen = {_sha1(p) for p in list_audio(dest)}
    index = _next_index(dest, category)
    result = {"added": 0, "duplicates": 0, "unreadable": [], "not_audio": 0, "destination": dest}

    for path in files:
        if path.suffix.lower() not in AUDIO_EXTS:
            result["not_audio"] += 1
            continue
        if not is_readable(path):
            result["unreadable"].append(path.name)
            continue
        digest = _sha1(path)
        if digest in seen:
            result["duplicates"] += 1
            continue
        shutil.copy2(path, dest / f"{category}_{index:04d}{path.suffix.lower()}")
        seen.add(digest)
        index += 1
        result["added"] += 1
    return result