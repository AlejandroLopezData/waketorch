import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

META_KEYS = ("waketorch", "wakeforge")
SKIP_DIRS = {".venv", "venv", "node_modules", ".git", "__pycache__", "site-packages"}


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
        peak_bytes = peak if platform.system() == "Darwin" else peak * 1024
        return peak_bytes / 1e6
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


def find_model(arg):
    """Use the given path, or look for a .onnx file under the current folder."""
    if arg:
        path = Path(arg).expanduser()
        if not path.is_file():
            raise SystemExit(f"File not found: {path}")
        return path
    found = []
    for p in Path.cwd().rglob("*.onnx"):
        if not SKIP_DIRS.intersection(p.parts):
            found.append(p)
    if not found:
        raise SystemExit("No .onnx file found here. Run:  python onnx_test.py path/to/model.onnx")
    if len(found) > 1:
        listing = "\n".join(f"  {p}" for p in sorted(found)[:15])
        raise SystemExit(f"Found several .onnx files, pick one:\n{listing}\n\nExample:  python onnx_test.py {sorted(found)[0]}")
    print(f"Using {found[0]}\n")
    return found[0]


def make_session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


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


def verdict(cpu_per_thread, late_fraction):
    if late_fraction > 0.05 or cpu_per_thread >= 90:
        return "TOO SLOW", "It cannot keep up at this rate."
    if cpu_per_thread >= 60:
        return "BARELY", "It keeps up, but with little room to spare."
    if cpu_per_thread >= 30:
        return "OK", "It runs in real time. A heavy app running at the same time could get in the way."
    return "COMFORTABLE", "It runs in real time with plenty of room to spare."


def spread(values, fmt="{:.2f}"):
    """median [min - max] across the repeated runs."""
    med = statistics.median(values)
    if len(values) == 1:
        return fmt.format(med)
    return f"{fmt.format(med)} [{fmt.format(min(values))} - {fmt.format(max(values))}]"


def main():
    ap = argparse.ArgumentParser(description="Measure whether this device can run a wake word .onnx model in real time.")
    ap.add_argument("model", nargs="?", help="path to the .onnx file (optional: searches the current folder if omitted)")
    ap.add_argument("--runs", type=int, default=5, help="how many times to repeat the whole measurement (default 5)")
    ap.add_argument("--hop", type=float, default=0.1, help="seconds between checks while listening (default 0.1)")
    ap.add_argument("--seconds", type=float, default=10.0, help="length of each real-time test (default 10)")
    ap.add_argument("--window-seconds", type=float, default=1.5, help="audio window, only if the model has no metadata")
    ap.add_argument("--sample-rate", type=int, default=16000, help="sample rate, only if the model has no metadata")
    ap.add_argument("--json", metavar="FILE", help="also save all results to a JSON file")
    args = ap.parse_args()

    path = find_model(args.model)
    baseline_mb = peak_memory_mb()  # memory before loading any model

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

    print("1) This device")
    print(f"   OS       {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"   CPU      {cpu} | {cores} cores")
    print(f"   RAM      {f'{ram:.1f} GB' if ram else 'unknown'}")
    print(f"   Runtime  Python {platform.python_version()} | onnxruntime {ort.__version__}")

    print("\n2) The model")
    print(f"   File     {path.name} | {size_mb:.3f} MB ({size_mb * 1000:.0f} KB)")
    print(f"   Input    {window} samples = {window / rate:.2f} s of audio at {rate} Hz"
          + ("" if meta else "  (assumed: no metadata in the model)"))
    print(f"   Rate     one check every {args.hop:g} s = {1 / args.hop:g} checks per second")
    print(f"   Test     {args.runs} repeated runs, synthetic noise input")

    thread_options = sorted({1, cores})
    latency = {n: [] for n in thread_options}
    live_runs = []
    best_threads = None

    print(f"\n3) Measuring ({args.runs} runs)...")
    for i in range(1, args.runs + 1):
        for n in thread_options:
            latency[n].append(measure_latency(path, x, n))
        if best_threads is None:  # decide the thread count once, after the first run
            best_threads = min(thread_options, key=lambda n: latency[n][0]["median"])
        live = run_realtime(path, x, best_threads, args.hop, args.seconds)
        live_runs.append(live)
        med = latency[best_threads][-1]["median"]
        print(f"   run {i}/{args.runs}: {med:.2f} ms per check | CPU {live['cpu_percent']:.0f}% of one core | "
              f"missed {live['late']}/{live['checks']}")

    # Aggregate
    print("\n4) Time for one check (milliseconds, median across runs [min - max])")
    print(f"   {'threads':>8s} {'typical':>20s} {'slow (95%)':>20s} {'worst':>8s}")
    best_by_median = min(thread_options, key=lambda n: statistics.median(r["median"] for r in latency[n]))
    for n in thread_options:
        medians = [r["median"] for r in latency[n]]
        p95s = [r["p95"] for r in latency[n]]
        worst = max(r["worst"] for r in latency[n])
        mark = "  <-- fastest" if n == best_by_median and len(thread_options) > 1 else ""
        print(f"   {n:8d} {spread(medians):>20s} {spread(p95s):>20s} {worst:8.2f}{mark}")

    peak_mb = peak_memory_mb()
    overhead_mb = (peak_mb - baseline_mb) if (peak_mb and baseline_mb) else None

    cpu_values = [r["cpu_percent"] for r in live_runs]
    late_total = sum(r["late"] for r in live_runs)
    checks_total = sum(r["checks"] for r in live_runs)
    cpu_median = statistics.median(cpu_values)
    word, explanation = verdict(cpu_median / best_threads, late_total / checks_total)
    best_median = statistics.median(r["median"] for r in latency[best_threads])
    best_p95 = statistics.median(r["p95"] for r in latency[best_threads])
    realtime_factor = (args.hop * 1000) / best_median

    print(f"\n5) Listening for {args.seconds:g} s per run, like a real microphone ({best_threads} thread(s))")
    print(f"   CPU used          {spread(cpu_values, '{:.0f}')}% of one core")
    print(f"   Missed deadlines  {late_total} of {checks_total} (all runs)")
    print(f"   Headroom          {realtime_factor:.0f}x faster than needed ({best_median:.2f} ms vs {args.hop * 1000:.0f} ms budget)")
    print(f"   Memory            {f'{peak_mb:.0f} MB' if peak_mb else 'unknown'} peak for the whole Python process"
          + (f", about {overhead_mb:.0f} MB more than before loading the model" if overhead_mb is not None else ""))

    print("\nRESULT")
    print(f"   {word}: {explanation}")
    if word in ("BARELY", "TOO SLOW"):
        suggested = max(round(best_median / 0.3 / 1000 * 20) / 20, args.hop * 2)
        print(f"   Try checking less often, for example every {suggested:.2f} s (the model still hears the whole word).")

    line = (f"{cpu} | {platform.system()} | onnxruntime {ort.__version__} | model {size_mb:.2f} MB | "
            f"{best_median:.1f} ms per check ({best_threads} thr) | {cpu_median:.0f}% CPU at {1 / args.hop:g} checks/s | "
            f"{f'{peak_mb:.0f}' if peak_mb else '?'} MB RAM | {word}")
    row = (f"| {cpu} | {size_mb:.2f} MB | {best_median:.1f} ms | {best_p95:.1f} ms | "
           f"{cpu_median:.0f}% | {f'{peak_mb:.0f}' if peak_mb else '?'} MB |")

    print("\nFor your README (table row):")
    print("   | Device | Model size | Time per check (median) | Time (95%) | CPU at "
          f"{1 / args.hop:g} checks/s | RAM (process) |")
    print("   |---|---|---|---|---|---|")
    print(f"   {row}")
    print("\nCopy and share this line to compare devices:")
    print(f"   {line}")

    if args.json:
        data = {
            "device": {"os": platform.system(), "release": platform.release(), "machine": platform.machine(),
                       "cpu": cpu, "cores": cores, "ram_gb": ram, "python": platform.python_version(),
                       "onnxruntime": ort.__version__},
            "model": {"file": path.name, "size_mb": size_mb, "window_samples": window, "sample_rate": rate},
            "settings": {"runs": args.runs, "hop": args.hop, "seconds": args.seconds, "threads": best_threads},
            "latency_ms": {str(n): latency[n] for n in thread_options},
            "realtime": live_runs,
            "memory_mb": {"peak": peak_mb, "baseline": baseline_mb, "added_by_model": overhead_mb},
            "verdict": word,
        }
        Path(args.json).write_text(json.dumps(data, indent=2))
        print(f"\nSaved results to {args.json}")

    return 0 if word != "TOO SLOW" else 1


if __name__ == "__main__":
    raise SystemExit(main())