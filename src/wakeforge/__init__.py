"""wakeforge: entrena tu propio wake word en casa."""
from .config import Config, load_config

__all__ = ["Config", "load_config", "train"]


def train(cfg: Config, on_progress=None):
    """Entrena con las grabaciones del proyecto y exporta el ONNX. `on_progress(event: dict)` es opcional."""
    from .train import train as _train  # import perezoso: torch solo se carga al entrenar
    return _train(cfg, on_progress)