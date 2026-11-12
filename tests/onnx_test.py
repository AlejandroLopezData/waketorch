import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

META_KEYS = ("waketorch", "wakeforge")
SKIP_DIRS = {".venv", "venv", "node_modules", ".git", "__pycache__", "site-packages"}
SLOWDOWNS = (1, 2, 5, 10, 20, 50)


# ----------------------------------------------------------------- device info
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
                    if line.startswith(("model name", "Model", "Hardware")):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Windows":
        name = os.environ.get("PROCESSOR_IDENTIFIER")
        if name:
            return name
    return platform.processor() or platform.machine() or "unknown"


def total_ram_gb():
    system = platform.system()
    try:
        if system == "Darwin":
            value = run_command(["sysctl", "-n", "hw.memsize"])
            return int(value) / 1e9 if value else None
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return int(line.split()[1]) * 1024 / 1e9
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
            return status.ullTotalPhys / 1e9
    except Exception:
        return None
    return None


def peak_memory_mb():
    """Peak memory of this process in MB (1 MB = 1e6 bytes)."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (peak if platform.system() == "Darwin" else peak * 1024) / 1e6
    except ImportError:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / 1e6
    except Exception:
        return None


# ----------------------------------------------------------------- model helpers
def find_model(arg):
    """Use the given path, or look for a .onnx file under the current folder."""
    if arg:
        path = Path(arg).expanduser()
        if not path.is_file():
            raise SystemExit(f"File not found: {path}")
        return path
    found = [p for p in Path.cwd().rglob("*.onnx") if not SKIP_DIRS.intersection(p.parts)]
    if not found:
        raise SystemExit("No .onnx file found here. Run:  python onnx_test.py path/to/model.onnx")
    if len(found) > 1:
        listing = "\n".join(f"  {p}" for p in sorted(found)[:15])
        raise SystemExit(f"Found several .onnx files, pick one:\n{listing}\n\n"
                         f"Example:  python onnx_test.py {sorted(found)[0]}")
    print(f"Using {found[0]}\n")
    return found[0]


def make_session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def model_stats(path):
    """Hardware-independent numbers: parameters and multiply-accumulates (MACs) per check.
    Needs the optional `onnx` package (pip install onnx). Counts Conv, Gemm and MatMul only."""
    try:
        import onnx
        from onnx import shape_inference
    except ImportError:
        return None
    try:
        graph = shape_inference.infer_shapes(onnx.load(str(path))).graph
        init = {i.name: list(i.dims) for i in graph.initializer}
        shapes = dict(init)
        for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
            t = vi.type.tensor_type
            if t.HasField("shape"):
                shapes[vi.name] = [d.dim_value if d.dim_value > 0 else 1 for d in t.shape.dim]
        params = sum(math.prod(d) for d in init.values())
        macs, ops = 0, {}
        for node in graph.node:
            ops[node.op_type] = ops.get(node.op_type, 0) + 1
            out = shapes.get(node.output[0])
            if not out:
                continue
            if node.op_type == "Conv" and node.input[1] in init:
                macs += math.prod(out) * math.prod(init[node.input[1]][1:])
            elif node.op_type == "Gemm" and node.input[1] in init:
                macs += math.prod(out[:-1]) * math.prod(init[node.input[1]])
            elif node.op_type == "MatMul":
                a = shapes.get(node.input[0])
                if a:
                    macs += math.prod(out) * a[-1]
        uncounted = sorted(set(ops) & {"LSTM", "GRU", "RNN", "ConvTranspose"})
        return {"params": params, "macs": macs, "ops": ops, "uncounted": uncounted}
    except Exception:
        return None


# ----------------------------------------------------------------- measurements
def measure_latency(path, x, threads, min_runs=30, max_runs=300, time_limit=5.0):
    session = make_session(path, threads)
    name = session.get_inputs()[0].name
    for _ in range(10):
        session.run(None, {name: x})
    times, started = [], time.perf_counter()
    while len(times) < max_runs:
        start = time.perf_counter()
        session.run(None, {name: x})
        times.append((time.perf_counter() - start) * 1000)
        if len(times) >= min_runs and time.perf_counter() - started > time_limit:
            break
    t = np.array(times)
    return {"median": float(np.median(t)), "p95": float(np.percentile(t, 95)),
            "worst": float(t.max()), "runs": len(times)}


def run_realtime(path, x, threads, hop, seconds):
    session = make_session(path, threads)
    name = session.get_inputs()[0].name
    for _ in range(10):
        session.run(None, {name: x})
    checks = max(1, int(seconds / hop))
    late = 0
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    next_tick = wall_start
    for _ in range(checks):
        session.run(None, {name: x})
        finished = time.perf_counter()
        next_tick += hop
        if finished > next_tick:
            late += 1
        else:
            time.sleep(next_tick - finished)
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    return {"cpu_percent": cpu / wall * 100, "late": late, "checks": checks}


def memory_child(path, window):
    """Runs in a fresh process: one session, one thread, like a real app."""
    baseline = peak_memory_mb()
    session = make_session(path, 1)
    name = session.get_inputs()[0].name
    x = np.zeros((1, window), dtype=np.float32)
    for _ in range(100):
        session.run(None, {name: x})
    print(json.dumps({"baseline": baseline, "peak": peak_memory_mb()}))
    return 0


def measure_app_memory(path, window):
    try:
        out = subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve()), str(path), "--mem-child", str(window)],
            text=True, stderr=subprocess.DEVNULL, timeout=120)
        return json.loads(out.strip().splitlines()[-1])
    except Exception:
        return None


def verdict(cpu_percent, late_fraction):
    if late_fraction > 0.05 or cpu_percent >= 90:
        return "TOO SLOW", "It cannot keep up at this rate."
    if cpu_percent >= 60:
        return "BARELY", "It keeps up, but with little room to spare."
    if cpu_percent >= 30:
        return "OK", "It runs in real time. A heavy app running at the same time could get in the way."
    return "COMFORTABLE", "It runs in real time with plenty of room to spare."


def spread(values, fmt="{:.2f}"):
    """median [min - max] across the repeated runs."""
    med = statistics.median(values)
    if len(values) == 1:
        return fmt.format(med)
    return f"{fmt.format(med)} [{fmt.format(min(values))} - {fmt.format(max(values))}]"


def human(n):
    for unit, size in (("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            return f"{n / size:.2f} {unit}"
    return f"{n:.0f}"


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Estimate whether a device can run a wake word .onnx model in real time.")
    ap.add_argument("model", nargs="?", help="path to the .onnx file (optional: searches the current folder if omitted)")
    ap.add_argument("--runs", type=int, default=5, help="how many times to repeat the whole measurement (default 5)")
    ap.add_argument("--hop", type=float, default=0.1, help="seconds between checks while listening (default 0.1)")
    ap.add_argument("--seconds", type=float, default=10.0, help="length of each real-time test (default 10)")
    ap.add_argument("--all-threads", action="store_true", help="also measure with every CPU core (extra info only)")
    ap.add_argument("--window-seconds", type=float, default=1.5, help="audio window, only if the model has no metadata")
    ap.add_argument("--sample-rate", type=int, default=16000, help="sample rate, only if the model has no metadata")
    ap.add_argument("--json", metavar="FILE", help="also save all results to a JSON file")
    ap.add_argument("--mem-child", type=int, metavar="WINDOW", help=argparse.SUPPRESS)
    args = ap.parse_args()

    path = find_model(args.model)
    if args.mem_child:
        return memory_child(path, args.mem_child)

    try:
        probe = make_session(path, 1)
    except Exception as exc:
        raise SystemExit(f"Could not load the model: {str(exc).splitlines()[0]}") from exc
    if len(probe.get_inputs()) != 1:
        raise SystemExit("This check supports models with a single audio input.")

    meta_map = probe.get_modelmeta().custom_metadata_map
    raw = next((meta_map[k] for k in META_KEYS if k in meta_map), None)
    meta = json.loads(raw) if raw else None
    window = int(meta["window_samples"]) if meta else int(args.window_seconds * args.sample_rate)
    rate = int(meta["input"]["sample_rate"]) if meta else args.sample_rate
    x = (np.random.randn(1, window) * 0.1).astype(np.float32)
    try:
        probe.run(None, {probe.get_inputs()[0].name: x})
    except Exception as exc:
        raise SystemExit(f"The model did not accept a test input of {window} samples: {str(exc).splitlines()[0]}") from exc

    ram = total_ram_gb()
    cores = os.cpu_count() or 1
    size_mb = path.stat().st_size / 1e6
    cpu = cpu_name()
    hop_ms = args.hop * 1000
    stats = model_stats(path)

    print("1) This device")
    print(f"   OS       {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"   CPU      {cpu} | {cores} cores")
    print(f"   RAM      {f'{ram:.1f} GB' if ram else 'unknown'}")
    print(f"   Runtime  Python {platform.python_version()} | onnxruntime {ort.__version__}")

    print("\n2) The model (these numbers do not depend on the device)")
    print(f"   File     {path.name} | {size_mb:.3f} MB ({size_mb * 1000:.0f} KB)")
    print(f"   Input    {window} samples = {window / rate:.2f} s of audio at {rate} Hz"
          + ("" if meta else "  (assumed: no metadata in the model)"))
    if stats:
        per_second = stats["macs"] / args.hop
        print(f"   Params   {human(stats['params'])}")
        print(f"   Work     ~{human(stats['macs'])} MACs per check, ~{human(per_second)} MACs per second "
              f"at {1 / args.hop:g} checks/s")
        if stats["uncounted"]:
            print(f"   Note     {', '.join(stats['uncounted'])} not counted: real work is higher than shown")
    else:
        print("   Params   (install the optional package to see parameters and MACs:  pip install onnx)")

    thread_options = sorted({1, cores}) if args.all_threads else [1]
    latency = {n: [] for n in thread_options}
    live_runs = []

    print(f"\n3) Measuring: {args.runs} runs, 1 thread (a small device), synthetic noise input")
    for i in range(1, args.runs + 1):
        for n in thread_options:
            latency[n].append(measure_latency(path, x, n))
        live = run_realtime(path, x, 1, args.hop, args.seconds)
        live_runs.append(live)
        print(f"   run {i}/{args.runs}: {latency[1][-1]['median']:.2f} ms per check | "
              f"CPU {live['cpu_percent']:.0f}% of one core | missed {live['late']}/{live['checks']}")

    print("\n4) Time for one check (milliseconds, median across runs [min - max])")
    print(f"   {'threads':>8s} {'typical':>20s} {'slow (95%)':>20s} {'worst':>8s}")
    for n in thread_options:
        medians = [r["median"] for r in latency[n]]
        p95s = [r["p95"] for r in latency[n]]
        worst = max(r["worst"] for r in latency[n])
        print(f"   {n:8d} {spread(medians):>20s} {spread(p95s):>20s} {worst:8.2f}")

    median_ms = statistics.median(r["median"] for r in latency[1])
    p95_ms = statistics.median(r["p95"] for r in latency[1])
    cpu_values = [r["cpu_percent"] for r in live_runs]
    cpu_median = statistics.median(cpu_values)
    # In a real listening loop each check costs more than in a tight loop (idle gaps cool the CPU
    # caches and lower its clock), so the estimates use the larger of the two measurements.
    share = max(median_ms / hop_ms, cpu_median / 100)
    effective_ms = share * hop_ms
    late_total = sum(r["late"] for r in live_runs)
    checks_total = sum(r["checks"] for r in live_runs)
    word, explanation = verdict(cpu_median, late_total / checks_total)

    print(f"\n5) Real-time test: {args.seconds:g} s per run, 1 thread")
    print(f"   CPU used          {spread(cpu_values, '{:.1f}')}% of one core")
    print(f"   Missed deadlines  {late_total} of {checks_total} (all runs)")
    print(f"   Headroom          {1 / share:.0f}x ({effective_ms:.1f} ms effective of a {hop_ms:.0f} ms budget; "
          f"{median_ms:.2f} ms in a tight loop)")

    print("\n6) Memory of a minimal app (fresh process, one session, one thread)")
    mem = measure_app_memory(path, window)
    app_peak = mem["peak"] if mem else None
    added = (mem["peak"] - mem["baseline"]) if mem and mem["peak"] and mem["baseline"] else None
    if mem and app_peak:
        print(f"   Whole process     ~{app_peak:.0f} MB (Python + onnxruntime + model)")
        if added is not None:
            print(f"   Added by model    ~{added:.0f} MB (on top of Python + onnxruntime)")
    else:
        print("   Could not measure memory on this system.")

    print("\n7) Estimate for other devices")
    print("   If one core of the device is N times slower than this one, the model would use:")
    print(f"   {'slower':>8s} {'time per check':>16s} {'CPU at ' + f'{1 / args.hop:g}' + ' checks/s':>22s}   verdict")
    scenarios = []
    for n in SLOWDOWNS:
        cpu_est = share * n * 100
        w, _ = verdict(cpu_est, 0)
        if cpu_est >= 100:
            w = "TOO SLOW"
        scenarios.append({"slowdown": n, "time_ms": effective_ms * n, "cpu_percent": cpu_est, "verdict": w})
        print(f"   {str(n) + 'x':>8s} {effective_ms * n:13.1f} ms {cpu_est:21.0f}%   {w}")
    limit = 1 / share
    print(f"   Rule of thumb: it keeps up on any core up to ~{limit:.0f}x slower than this one, and stays under")
    print(f"   30% CPU up to ~{0.3 * limit:.0f}x slower. Assumes time scales with core speed: keep a 2-3x safety margin.")

    print("\nRESULT")
    print(f"   {word}: {explanation}")

    mem_txt = f"~{app_peak:.0f} MB" if app_peak else "?"
    macs_txt = f"{human(stats['macs'])}" if stats else "?"
    params_txt = f"{human(stats['params'])}" if stats else "?"
    print("\nFor your README (table row, measured with 1 thread):")
    print("   | Device | Model size | Params | MACs per check | Time per check | CPU at "
          f"{1 / args.hop:g} checks/s | RAM (minimal app) |")
    print("   |---|---|---|---|---|---|---|")
    print(f"   | {cpu} | {size_mb:.2f} MB | {params_txt} | {macs_txt} | {median_ms:.1f} ms | "
          f"{cpu_median:.1f}% of one core | {mem_txt} |")
    print("\nCopy and share this line to compare devices:")
    print(f"   {cpu} | {platform.system()} | onnxruntime {ort.__version__} | model {size_mb:.2f} MB | "
          f"{median_ms:.1f} ms per check (1 thr) | {cpu_median:.1f}% CPU at {1 / args.hop:g} checks/s | "
          f"{mem_txt} RAM | {word}")

    if args.json:
        data = {
            "device": {"os": platform.system(), "release": platform.release(), "machine": platform.machine(),
                       "cpu": cpu, "cores": cores, "ram_gb": ram, "python": platform.python_version(),
                       "onnxruntime": ort.__version__},
            "model": {"file": path.name, "size_mb": size_mb, "window_samples": window, "sample_rate": rate,
                      "params": stats["params"] if stats else None, "macs_per_check": stats["macs"] if stats else None},
            "settings": {"runs": args.runs, "hop": args.hop, "seconds": args.seconds},
            "latency_ms": {str(n): latency[n] for n in thread_options},
            "realtime": live_runs,
            "memory_mb": mem,
            "scenarios": scenarios,
            "verdict": word,
        }
        Path(args.json).write_text(json.dumps(data, indent=2))
        print(f"\nSaved results to {args.json}")

    return 0 if word != "TOO SLOW" else 1


if __name__ == "__main__":
    raise SystemExit(main())