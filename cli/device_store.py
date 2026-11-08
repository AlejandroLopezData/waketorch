"""Guarda el último test del equipo para no repetirlo: PROJECTS_DIR/.device_check.json

Es un archivo oculto (empieza por punto) para que list_projects() no lo confunda con un proyecto.
Si prefieres verlo, cambia FILE_NAME.
"""
import json
import os
from datetime import datetime

from wakeforge.projects import PROJECTS_DIR

FILE_NAME = ".device_check.json"
SCHEMA = 1
_REQUIRED = ("system", "results", "best", "rec")


def path():
    return PROJECTS_DIR / FILE_NAME


def load():
    """Devuelve {'saved_at', 'project', 'result'} o None si no existe o no es válido."""
    try:
        data = json.loads(path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return None
    result = data.get("result")
    if not isinstance(result, dict) or any(key not in result for key in _REQUIRED):
        return None
    return {"saved_at": data.get("saved_at"), "project": data.get("project"), "result": result}


def save(result, project=None):
    """Escribe el resultado (de forma atómica) y devuelve lo guardado. Puede lanzar OSError."""
    result = {k: v for k, v in result.items() if k != "stage"}
    stored = {"saved_at": datetime.now().isoformat(timespec="seconds"), "project": project, "result": result}
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path().with_suffix(".tmp")
    tmp.write_text(json.dumps({"schema": SCHEMA, **stored}, indent=2), encoding="utf-8")
    os.replace(tmp, path())
    return stored


def recommended():
    """Recomendación de pasos guardada, solo si el equipo puede entrenar en local."""
    stored = load()
    rec = stored["result"].get("rec") if stored else None
    return rec if rec and rec.get("viable") else None


def saved_label(stored):
    try:
        return f"{datetime.fromisoformat(stored['saved_at']):%Y-%m-%d %H:%M}"
    except (TypeError, ValueError):
        return "unknown date"