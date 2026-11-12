from .config import Config, load_config

__all__ = ["Config", "load_config", "train"]


def train(cfg: Config, on_progress=None):
    from .train import train as _train  # import lazy: torch solo se carga al entrenar
    return _train(cfg, on_progress)