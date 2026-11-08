<div align="center">

# WakeTorch

### Train your own wake word model in minutes — in any language.

Guided CLI · Real voice samples · Fast training · CPU inference · ONNX export  
Open source · Local · Lightweight

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![ONNX](https://img.shields.io/badge/ONNX-005CED?style=flat&logo=onnx&logoColor=white) ![macOS](https://img.shields.io/badge/macOS-000000?style=flat&logo=apple&logoColor=white) ![Windows](https://img.shields.io/badge/Windows-0078D4?style=flat&logo=windows&logoColor=white) ![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat&logo=linux&logoColor=black) ![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-C51A4A?style=flat&logo=raspberrypi&logoColor=white) ![License](https://img.shields.io/badge/License-Apache--2.0-green?style=flat) ![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-orange?style=flat)

**WORK IN PROGRESS — This project is still under active development.**

WakeTorch is an early-stage project. Features, functionality, and documentation
may change as development progresses. Bugs and incomplete features are expected.

**Any language. Your voice. Your wake word.**

[Model & Documentation](https://alejandrolopezdata.github.io/waketorch/)

</div>


## Why WakeTorch?

> **You shouldn't need a huge dataset, a GPU, or a PhD to train a wake word.**

* **Any language** — train the wake word you actually want, in the language you actually speak.
* **Minutes, not hours** — the guided CLI takes you from recordings to a working model in minutes.
* **Your voice, not synthetic TTS** — train on real recordings for a truly personal wake word.
* **Less data, more signal** — built-in augmentation makes small datasets go further.
* **One model, everywhere** — export to ONNX and run lightweight inference on CPU.
* **Yours from start to finish** — local training, local inference, no cloud required.


## Installation

**Requirements:** [Python](https://www.python.org/downloads/) 3.10+ (3.12 recommended; on Windows, tick **Add python.exe to PATH** when installing) and a microphone.

### 1. Download

Download the ZIP from [GitHub](https://github.com/alejandrolopezdata/waketorch) (**Code → Download ZIP**) and unzip it,
or clone it:

```bash
git clone https://github.com/alejandrolopezdata/waketorch.git
```

### 2. Install

**macOS / Linux**

```bash
cd ~/Downloads/waketorch-main
chmod +x install.sh
./install.sh
```

**Windows** (Command Prompt)

```bat
cd %USERPROFILE%\Downloads\waketorch-main
install.bat
```

You can also just double-click `install.bat` in the unzipped folder.

### 3. Run

Open a **new** terminal and type:

```bash
waketorch
```

### Uninstall

`./install.sh uninstall` (macOS / Linux) or `install.bat uninstall` (Windows). Your projects are not touched.

## Hardware recommendations

You don't need a powerful computer, but training speed depends a lot on your hardware. Running a trained model is far lighter than training it: once you have the `.onnx` file, any computer can run it on the CPU.

| Computer | Training | Notes |
|---|---|---|
| **Mac with Apple Silicon** (M1 or later) | Recommended | Uses the Apple GPU automatically. Measured on an M2: a 500-step test takes about 36 seconds. |
| **Windows / Linux with an NVIDIA GPU** (4 GB+) | Expected to work well | Uses CUDA automatically. Not measured yet. |
| **Modern CPU only** (6+ cores) | Works, but slower | Estimated 25-60 minutes for the full training. The check below recommends fewer steps to keep it short. |
| **Older or low-power CPU** (2 cores, Celeron, Pentium...) | Not recommended | Estimated more than 1.5 hours. Train on another computer and copy the `.onnx` file. |
| **Intel / AMD integrated graphics, Radeon** | Same as CPU only | PyTorch does not use these GPUs. |
| **Mac with an Intel processor** | Probably not supported | Recent PyTorch versions are not published for these Macs. |

You also need about 4 GB of RAM (8 GB recommended) and around 5 GB of free disk space, mostly for PyTorch.

> Only the Apple M2 has been measured so far. The other rows are estimates, and we would love to replace them with real numbers: run the check below and share your results.

### Check if your computer can train it

Before training, run the device check. It takes about a minute:

```bash
python scripts/check_device.py
```

Run it from the project folder, in the same Python environment where you installed WakeTorch. To measure with your own project's settings and recordings, add `-p YOUR_PROJECT`.

The check:

1. Detects your operating system, CPU, RAM and GPU (NVIDIA or Apple).
2. Runs a short, real training benchmark on every option it finds (GPU and CPU).
3. Works out how many training steps fit in 10 minutes, up to the standard 2000, and tells you whether your computer is up to the job.

On a fast computer it recommends the full 2000 steps. On a slower one it recommends fewer, so training finishes in about 10 minutes, and it prints the exact command to use. If even the minimum (300 steps) would take too long, it says so and suggests training on another computer.

The guided CLI will soon offer this recommendation directly while you train.

### Automatic optimization

You don't have to configure anything for it to run well on modest hardware. Training picks the fastest option it finds (NVIDIA GPU, then Apple GPU, then CPU). On CPU it also measures whether using 1 thread or all of them is faster, because on some machines fewer threads are several times faster.
