import argparse
import dataclasses
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, WeightedRandomSampler

from wakeforge import load_config
from wakeforge.augment import build_specaug
from wakeforge.data import prepare_data
from wakeforge.export import export_onnx
from wakeforge.features import LogMelFrontend
from wakeforge.model import ExportModel, WakeWordNet
from wakeforge.projects import PROJECTS_DIR
from wakeforge.train import EMA


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def bench(fn, device, repeats=3, warmup=1):
    for _ in range(warmup):
        fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync(device)
    return (time.perf_counter() - start) / repeats * 1000


def train_step_cost(net, x, y, device):
    def run():
        net.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(net(x), y)
        loss.backward()
    return bench(run, device)


def main():
    ap = argparse.ArgumentParser(description="Find out where the training time goes.")
    ap.add_argument("-p", "--project", required=True)
    ap.add_argument("--steps", type=int, default=12, help="measured steps for the breakdown")
    ap.add_argument("--device", default="cpu", help="device for the step breakdown (cpu, mps, cuda)")
    ap.add_argument("--steps-only", action="store_true", help="skip the micro-benchmarks and print only the step breakdown")
    ap.add_argument("--set", action="append", default=[], metavar="section.option=value")
    args = ap.parse_args()

    cfg = load_config(PROJECTS_DIR / args.project / "config.toml", args.set)
    t = cfg.training
    device = torch.device(args.device)
    print(f"torch {torch.__version__} | threads: {torch.get_num_threads()} | cpu cores: {os.cpu_count()} | "
          f"mps: {torch.backends.mps.is_available()} | cuda: {torch.cuda.is_available()}")

    data = prepare_data(cfg)
    train_ds = data.train_ds
    labels = torch.tensor(train_ds.labels)
    npos, nneg = int((labels == 1).sum()), int((labels == 0).sum())
    weights = torch.where(labels == 1, t.positive_ratio / npos, (1 - t.positive_ratio) / nneg)
    warmup = 3
    sampler = WeightedRandomSampler(weights, (args.steps + warmup) * t.batch_size, replacement=True)
    loader = DataLoader(train_ds, t.batch_size, sampler=sampler, drop_last=True, num_workers=t.num_workers)

    front_device = torch.device("cpu") if device.type == "mps" else device
    frontend = LogMelFrontend(cfg.audio.sample_rate, cfg.features).to(front_device).eval()
    net = WakeWordNet(cfg.features.n_mels, cfg.model).to(device)
    specaug = build_specaug(cfg.augmentation)
    specaug = specaug.to(device) if specaug is not None else None
    ema = EMA(net, t.ema_decay)
    opt = torch.optim.AdamW(net.parameters(), lr=t.lr)
    ls = t.label_smoothing

    totals = defaultdict(float)
    first_step = 0.0
    iterator = iter(loader)
    for i in range(args.steps + warmup):
        marks = [time.perf_counter()]

        def mark():
            sync(device)
            marks.append(time.perf_counter())

        wav, y = next(iterator)
        wav, y = wav.to(front_device), y.to(device)
        mark()
        with torch.no_grad():
            x = frontend(wav).to(device)
        mark()
        if specaug is not None:
            with torch.no_grad():
                x = specaug(x)
        mark()
        loss = F.binary_cross_entropy_with_logits(net(x), y * (1 - 2 * ls) + ls)
        mark()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mark()
        torch.nn.utils.clip_grad_norm_(net.parameters(), t.grad_clip)
        mark()
        opt.step()
        mark()
        ema.update(net)
        mark()
        loss.item()
        mark()

        if i == 0:
            first_step = marks[-1] - marks[0]
        if i >= warmup:
            names = ("data + transfer", "spectrogram", "specaugment", "forward + loss", "backward",
                     "grad clipping", "optimizer", "EMA update", "loss.item()")
            for name, a, b in zip(names, marks, marks[1:]):
                totals[name] += b - a

    total = sum(totals.values())
    print(f"\n1) Time per training step on {device} (batch {t.batch_size}, window {data.window_samples} samples)")
    for name, value in totals.items():
        print(f"   {name:16s} {value / args.steps * 1000:9.1f} ms   {value / total * 100:5.1f}%")
    print(f"   {'TOTAL':16s} {total / args.steps * 1000:9.1f} ms   -> {total / args.steps * t.steps / 60:.1f} min for {t.steps} steps")
    print(f"   first step (warm-up, may include kernel compilation): {first_step:.2f} s")

    if args.steps_only:
        return

    frames = 1 + data.window_samples // cfg.features.hop_length
    x = torch.randn(t.batch_size, cfg.features.n_mels, frames)
    y = (torch.rand(t.batch_size) > 0.5).float()
    cpu = torch.device("cpu")

    print("\n2) Network forward+backward alone, CPU (ms per step)")
    n_blocks = len(cfg.model.dilations)
    variants = [
        ("your config", cfg.model),
        ("dilations all 1", dataclasses.replace(cfg.model, dilations=[1] * n_blocks)),
        ("width 32", dataclasses.replace(cfg.model, width=32)),
    ]
    for label, model_cfg in variants:
        print(f"   {label:18s} {train_step_cost(WakeWordNet(cfg.features.n_mels, model_cfg), x, y, cpu):9.1f}")

    print("\n3) Same, changing the number of CPU threads (your config)")
    default_threads = torch.get_num_threads()
    for n in sorted({1, 2, 4, os.cpu_count() or 4}):
        torch.set_num_threads(n)
        print(f"   {n} thread(s)        {train_step_cost(WakeWordNet(cfg.features.n_mels, cfg.model), x, y, cpu):9.1f}")
    torch.set_num_threads(default_threads)

    print("\n4) Spectrogram: current convolution vs FFT (ms per batch, CPU)")
    wav = torch.randn(t.batch_size, data.window_samples) * 0.1
    fft_mel = torchaudio.transforms.MelSpectrogram(
        cfg.audio.sample_rate, n_fft=cfg.features.n_fft, win_length=cfg.features.win_length,
        hop_length=cfg.features.hop_length, n_mels=cfg.features.n_mels,
        f_min=cfg.features.f_min, f_max=cfg.features.f_max)
    conv_front = LogMelFrontend(cfg.audio.sample_rate, cfg.features).eval()
    with torch.no_grad():
        print(f"   convolution (current) {bench(lambda: conv_front(wav), cpu):9.1f}")
        print(f"   FFT                   {bench(lambda: fft_mel(wav), cpu):9.1f}")

    if torch.backends.mps.is_available():
        print("\n5) Network forward+backward on the Apple GPU (MPS, ms per step)")
        mps = torch.device("mps")
        try:
            print(f"   your config       {train_step_cost(WakeWordNet(cfg.features.n_mels, cfg.model).to(mps), x.to(mps), y.to(mps), mps):9.1f}")
        except Exception as exc:
            print(f"   failed: {exc}")

    print("\n6) Fixed cost: ONNX export + verification (seconds)")
    export_model = ExportModel(LogMelFrontend(cfg.audio.sample_rate, cfg.features), WakeWordNet(cfg.features.n_mels, cfg.model))
    with tempfile.TemporaryDirectory() as tmp:
        start = time.perf_counter()
        export_onnx(export_model.eval(), data.window_samples, Path(tmp) / "m.onnx", cfg.export.opset, True)
        print(f"   {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()