from datetime import datetime

from wakeforge.config import ConfigError, load_config
from wakeforge.modelfile import list_versions, read_metadata
from wakeforge.projects import PROJECTS_DIR


def project_config_path(project):
    return PROJECTS_DIR / project / "config.toml"


def version_infos(project):
    try:
        cfg = load_config(project_config_path(project))
    except (ConfigError, OSError):
        return []
    infos = []
    for version, path in sorted(list_versions(cfg.project.output_dir, cfg.export.filename).items()):
        try:
            meta = read_metadata(path)
        except ValueError:
            meta = None
        infos.append({"version": version, "path": path, "meta": meta,
                      "date": datetime.fromtimestamp(path.stat().st_mtime)})
    return infos


def version_label(info):
    label = f"v{info['version']}   {info['date']:%Y-%m-%d %H:%M}"
    v = (info["meta"] or {}).get("validation") or {}
    if v.get("frr") is not None and v.get("fpr") is not None:
        label += f"   sensitivity {(1 - v['frr']) * 100:.0f}%   false alarms {v['fpr'] * 100:.1f}%"
    elif info["meta"] is None:
        label += "   (no metadata: train again)"
    return label