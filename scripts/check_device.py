import argparse
import contextlib
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from wakeforge import load_config
from wakeforge.augment import build_specaug
from wakeforge.data import prepare_data
from wakeforge.features import LogMelFrontend
from wakeforge.model import WakeWordNet
from wakeforge.projects import PROJECTS_DIR, TEMPLATE
from wakeforge.train import EMA

EVENT_PREFIX = "WF_EVENT "  # mismo protocolo que train.py --progress-json
_JSON = False


def emit(event):
    """Con --json envía un evento por línea (para que la CLI lo muestre). Si no, no hace nada."""
    if _JSON:
        sys.__stdout__.write(EVENT_PREFIX + json.dumps(event) + "\n")
        sys.__stdout__.flush()


def run_command(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def cpu_name():
    system = platform.system()
    if system == "Darwin":
        name = run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Windows":
        name = os.environ.get("PROCESSOR_IDENTIFIER")
        if name:
            return name
    return platform.processor() or platform.machine() or "unknown"


def physical_cores():
    system = platform.system()
    if system == "Darwin":
        value = run_command(["sysctl", "-n", "hw.physicalcpu"])
        return int(value) if value and value.isdigit() else None
    if system == "Linux":
        value = run_command(["sh", "-c", "lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l"])
        return int(value) if value and value.isdigit() and int(value) > 0 else None
    return None


def total_ram_gb():
    system = platform.system()
    try:
        if system == "Darwin":
            value = run_command(["sysctl", "-n", "hw.memsize"])
            return int(value) / 2 ** 30 if value else None
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return int(line.split()[1]) / 2 ** 20
        if system == "Windows":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullTotalPhys / 2 ** 30
    except Exception:
        return None
    return None


def recommend(step_ms, data_ms, budget_minutes, max_steps, min_steps, quick_steps, margin):
    seconds_per_step = (step_ms + data_ms) / margin / 1000
    fit = int(budget_minutes * 60 / seconds_per_step)
    steps = min(max_steps, fit)
    quick = min(quick_steps, steps)
    return {
        "seconds_per_step": seconds_per_step,
        "fit": fit,
        "steps": steps,
        "minutes": steps * seconds_per_step / 60,
        "quick_steps": quick,
        "quick_minutes": quick * seconds_per_step / 60,
        "max_minutes": max_steps * seconds_per_step / 60,
        "viable": fit >= min_steps,
    }


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_step(cfg, window_samples, device, precision, threads, warmup=3, steps=15, time_limit=25.0):
    if threads is not None:
        torch.set_num_threads(threads)
    t = cfg.training
    front_device = torch.device("cpu") if device.type == "mps" else device
    frontend = LogMelFrontend(cfg.audio.sample_rate, cfg.features).to(front_device).eval()
    net = WakeWordNet(cfg.features.n_mels, cfg.model).to(device)
    specaug = build_specaug(cfg.augmentation)
    specaug = specaug.to(device) if specaug is not None else None
    ema = EMA(net, t.ema_decay)
    opt = torch.optim.AdamW(net.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    wav = torch.randn(t.batch_size, window_samples) * 0.1
    y = (torch.rand(t.batch_size) > 0.5).float().to(device)
    ls = t.label_smoothing

    times, started = [], time.perf_counter()
    for i in range(warmup + steps):
        sync(device)
        start = time.perf_counter()
        net.train()
        with torch.no_grad():
            x = frontend(wav.to(front_device)).to(device)
            if specaug is not None:
                x = specaug(x)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            logits = net(x)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y * (1 - 2 * ls) + ls)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), t.grad_clip)
        opt.step()
        ema.update(net)
        loss.item()
        sync(device)
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1000)
            if len(times) >= 3 and time.perf_counter() - started > time_limit:
                break
    return statistics.median(times)


def measure_data_ms(cfg, data):
    t = cfg.training
    labels = torch.tensor(data.train_ds.labels)
    npos, nneg = int((labels == 1).sum()), int((labels == 0).sum())
    weights = torch.where(labels == 1, t.positive_ratio / npos, (1 - t.positive_ratio) / nneg)
    batches = 8
    sampler = WeightedRandomSampler(weights, batches * t.batch_size, replacement=True)
    loader = DataLoader(data.train_ds, t.batch_size, sampler=sampler, drop_last=True, num_workers=t.num_workers)
    iterator, times = iter(loader), []
    for i in range(batches):
        start = time.perf_counter()
        next(iterator)
        if i > 0:
            times.append((time.perf_counter() - start) * 1000)
    return statistics.mean(times)


def describe_system():
    """Imprime los datos del equipo y devuelve (hay_aceleradores, datos_como_dict)."""
    ram = total_ram_gb()
    physical = physical_cores()
    cpu = cpu_name()
    cores = f"{os.cpu_count()} logical" + (f", {physical} physical" if physical else "")
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": cpu,
        "cores": cores,
        "ram_gb": ram,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpus": [],
    }
    print("1) Your computer")
    print(f"   OS       {info['os']}")
    print(f"   CPU      {cpu} | {cores}")
    print(f"   RAM      {f'{ram:.1f} GB' if ram else 'unknown'}")
    print(f"   Python   {info['python']} | PyTorch {info['torch']}")
    accelerators = False
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"   GPU      CUDA {i}: {p.name} | {p.total_memory / 2 ** 30:.1f} GB | compute capability {p.major}.{p.minor}")
        info["gpus"].append(f"{p.name} · {p.total_memory / 2 ** 30:.1f} GB")
        accelerators = True
    if torch.cuda.is_available():
        try:
            print(f"   bfloat16 supported by the GPU: {torch.cuda.is_bf16_supported()}")
        except Exception:
            pass
    if torch.backends.mps.is_available():
        print("   GPU      Apple GPU (MPS) available")
        info["gpus"].append("Apple GPU (MPS)")
        accelerators = True
    if not accelerators:
        print("   GPU      no compatible GPU found: training will use the CPU")
    return accelerators, info


def run(args):
    """Hace el test. Devuelve (código_de_salida, datos_para_la_CLI o None)."""
    accelerators, system_info = describe_system()

    print("\n2) Measuring (this takes about a minute)")
    data_ms = 20.0
    emit({"stage": "preparing"})
    if args.project:
        cfg = load_config(PROJECTS_DIR / args.project / "config.toml")
        data = prepare_data(cfg)
        window_samples = data.window_samples
        data_ms = measure_data_ms(cfg, data)
        source = f"project '{args.project}': window {window_samples / cfg.audio.sample_rate:.2f} s, data loading {data_ms:.1f} ms/step"
    else:
        cfg = load_config(TEMPLATE)
        seconds = cfg.audio.window_seconds if cfg.audio.window_seconds > 0 else 1.5
        window_samples = int(round(seconds * cfg.audio.sample_rate))
        source = f"default settings: assumed window {seconds:.2f} s, data loading assumed {data_ms:.0f} ms/step"
    print(f"   {source}")
    print(f"   batch {cfg.training.batch_size}, model width {cfg.model.width}\n")

    candidates = []
    if torch.cuda.is_available():
        cuda = torch.device("cuda")
        candidates += [("CUDA, fp32", cuda, "fp32", None, 25.0), ("CUDA, bf16 (what train.py uses)", cuda, "bf16", None, 25.0)]
    if torch.backends.mps.is_available():
        candidates.append(("Apple GPU (MPS)", torch.device("mps"), "fp32", None, 25.0))
    default_threads = torch.get_num_threads()
    cpu_limit = 8.0 if accelerators else 25.0
    for n in sorted({1, default_threads}):
        candidates.append((f"CPU, {n} thread(s)", torch.device("cpu"), "fp32", n, cpu_limit))

    results = []
    for index, (label, device, precision, threads, limit) in enumerate(candidates):
        emit({"stage": "measuring", "index": index, "total": len(candidates), "label": label})
        try:
            ms = measure_step(cfg, window_samples, device, precision, threads, time_limit=limit)
            results.append((label, ms, None))
        except Exception as exc:
            results.append((label, None, str(exc).splitlines()[0][:80]))
        torch.set_num_threads(default_threads)

    ok = [r for r in results if r[1] is not None]
    best = min(ok, key=lambda r: r[1]) if ok else None
    print(f"   {'configuration':34s} {'ms per step':>12s}")
    for label, ms, error in results:
        mark = "  <-- fastest" if best and label == best[0] else ""
        print(f"   {label:34s} {ms:12.1f}{mark}" if ms is not None else f"   {label:34s} {'failed':>12s}  ({error})")

    print("\n3) Estimate")
    if best is None:
        message = "Every configuration failed, so no estimate is possible."
        print(f"   {message}", file=sys.stderr if _JSON else sys.stdout)
        return 1, None
    rec = recommend(best[1], data_ms, args.budget_minutes, args.max_steps, args.min_steps, args.quick_steps, args.margin)
    print(f"   Fastest: {best[0]} at {best[1]:.1f} ms/step (+{data_ms:.0f} ms data), counting on {args.margin * 100:.0f}% of that speed once warm")
    print(f"   Steps that fit in {args.budget_minutes:g} min: {rec['fit']}")
    print(f"   Quick test           {rec['quick_steps']:5d} steps   about {rec['quick_minutes']:.1f} min")
    print(f"   RECOMMENDED          {rec['steps']:5d} steps   about {rec['minutes']:.1f} min   (limit: {args.max_steps} steps or {args.budget_minutes:g} min)")
    if rec["steps"] < args.max_steps:
        print(f"   The full {args.max_steps} steps would take about {rec['max_minutes']:.0f} min")
    if rec["viable"]:
        print("\n   Verdict: this computer can train the model locally.")
        target = f"-p {args.project}" if args.project else "-p YOUR_PROJECT"
        print(f"   Use:  python scripts/train.py {target} --set training.steps={rec['steps']}")
    else:
        print(f"\n   Verdict: NOT recommended. Even the minimum ({args.min_steps} steps) would take "
              f"about {args.min_steps * rec['seconds_per_step'] / 60:.0f} min. Consider training elsewhere (for example Google Colab) "
              "and bringing the .onnx file back.")
    bf16 = next((r for r in results if "bf16" in r[0]), None)
    fp32 = next((r for r in results if r[0] == "CUDA, fp32"), None)
    bf16_slower = bool(bf16 and fp32 and (bf16[1] is None or (fp32[1] is not None and bf16[1] > fp32[1] * 1.1)))
    if bf16_slower:
        print("   Note: on this GPU bf16 is slower than or fails against fp32, and train.py currently uses bf16 on CUDA.")

    payload = {
        "system": system_info,
        "source": source,
        "batch": cfg.training.batch_size,
        "width": cfg.model.width,
        "results": [{"label": label, "ms": ms, "error": error} for label, ms, error in results],
        "best": {"label": best[0], "ms": best[1]},
        "data_ms": data_ms,
        "margin": args.margin,
        "budget_minutes": args.budget_minutes,
        "max_steps": args.max_steps,
        "min_steps": args.min_steps,
        "rec": rec,
        "bf16_slower": bf16_slower,
    }
    return 0, payload


def main():
    global _JSON
    ap = argparse.ArgumentParser(description="Check this computer and estimate how many training steps it can do.")
    ap.add_argument("-p", "--project", help="measure with this project's config and data (optional)")
    ap.add_argument("--budget-minutes", type=float, default=10.0, help="time you accept to wait (default 10)")
    ap.add_argument("--max-steps", type=int, default=2000, help="upper limit of recommended steps (default 2000)")
    ap.add_argument("--min-steps", type=int, default=300, help="below this, local training is not recommended (default 300)")
    ap.add_argument("--quick-steps", type=int, default=500, help="steps of the quick test (default 500)")
    ap.add_argument("--margin", type=float, default=0.8, help="speed kept after heating up, 0-1 (default 0.8)")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable mode for the CLI: progress events and one final result, no text report")
    args = ap.parse_args()
    _JSON = args.json

    quiet = contextlib.redirect_stdout(io.StringIO()) if args.json else contextlib.nullcontext()
    with quiet:
        code, payload = run(args)
    if args.json and payload is not None:
        emit({"stage": "result", **payload})
    return code


if __name__ == "__main__":
    raise SystemExit(main())