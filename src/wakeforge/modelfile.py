import json
import re
from pathlib import Path

META_KEY = "wakeforge"

_METADATA_PROPS_FIELD = 14
_TAG_KEY = 0x0A
_TAG_VALUE = 0x12


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _parse_entry(chunk: bytes) -> tuple[str, str]:
    key = value = ""
    pos = 0
    while pos < len(chunk):
        tag, pos = _read_varint(chunk, pos)
        size, pos = _read_varint(chunk, pos)
        data = chunk[pos:pos + size].decode("utf-8")
        pos += size
        if tag == _TAG_KEY:
            key = data
        elif tag == _TAG_VALUE:
            value = data
    return key, value


def _read_props(path: Path) -> dict[str, str]:
    buf = Path(path).read_bytes()
    props: dict[str, str] = {}
    pos = 0
    try:
        while pos < len(buf):
            tag, pos = _read_varint(buf, pos)
            field, wire = tag >> 3, tag & 7
            if wire == 0:
                _, pos = _read_varint(buf, pos)
            elif wire == 1:
                pos += 8
            elif wire == 5:
                pos += 4
            elif wire == 2:
                size, pos = _read_varint(buf, pos)
                if field == _METADATA_PROPS_FIELD:
                    key, value = _parse_entry(buf[pos:pos + size])
                    props[key] = value
                pos += size
            else:
                raise ValueError
        if pos != len(buf):
            raise ValueError
    except (ValueError, IndexError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path} is not a valid ONNX file.") from exc
    return props


def write_metadata(path: str | Path, meta: dict) -> None:
    path = Path(path)
    if META_KEY in _read_props(path):
        raise ValueError(f"{path} already has embedded metadata.")
    value = json.dumps(meta, default=str).encode("utf-8")
    key = META_KEY.encode("utf-8")
    entry = bytes([_TAG_KEY]) + _varint(len(key)) + key + bytes([_TAG_VALUE]) + _varint(len(value)) + value
    field = bytes([(_METADATA_PROPS_FIELD << 3) | 2]) + _varint(len(entry)) + entry
    with open(path, "ab") as f:
        f.write(field)


def read_metadata(path: str | Path) -> dict | None:
    raw = _read_props(Path(path)).get(META_KEY)
    return json.loads(raw) if raw is not None else None


def list_versions(output_dir: str | Path, model_filename: str = "model.onnx") -> dict[int, Path]:
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return {}
    stem, suffix = Path(model_filename).stem, Path(model_filename).suffix
    rx = re.compile(rf"^{re.escape(stem)}_v(\d+){re.escape(suffix)}$")
    return {int(m.group(1)): p for p in output_dir.iterdir() if (m := rx.match(p.name))}