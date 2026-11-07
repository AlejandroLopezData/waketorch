import re

from . import ROOT

EXAMPLES = {
    "file": ("Test it with an audio file", "detect_from_file.py"),
    "mic": ("Listen to the microphone", "detect_from_mic.py"),
}


def example_code(kind, model_path):
    code = (ROOT / "examples" / EXAMPLES[kind][1]).read_text(encoding="utf-8")
    return re.sub(r"^MODEL_PATH = .*$", f"MODEL_PATH = {str(model_path)!r}", code, count=1, flags=re.MULTILINE)