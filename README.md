<div align="center">

# WakeTorch

### Train your own wake word model in minutes — in any language.

Guided CLI · Real voice samples · Fast training · ONNX export · Open source
Local · Lightweight

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) 
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=flat&logo=onnx&logoColor=white) 
![macOS](https://img.shields.io/badge/macOS-000000?style=flat&logo=apple&logoColor=white) 
![Windows](https://img.shields.io/badge/Windows-0078D4?style=flat&logo=windows&logoColor=white) 
![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat&logo=linux&logoColor=black) 
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-C51A4A?style=flat&logo=raspberrypi&logoColor=white) 
![License](https://img.shields.io/badge/License-Apache--2.0-green?style=flat) ![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-orange?style=flat)
![Version](https://img.shields.io/badge/Version-0.1.0-blue?style=flat)

**Any language. Your voice. Your wake word.**

[Model & Documentation](https://alejandrolopezdata.github.io/waketorch/) | 
[Usage Tutorial](https://alejandrolopezdata.github.io/waketorch/)</div>


## Why WakeTorch?

> **You shouldn't have to rack your brain to train a wake word.**

* **Any language** — train the wake word you actually want, in the language you actually speak.
* **Minutes, not hours** — the guided CLI takes you from recordings to a working model in minutes.
* **Yours from start to finish** — local training, local inference, no cloud required.
* **One model, everywhere** — export to ONNX and run lightweight inference on CPU.


## Installation

**Requirements:** [Python](https://www.python.org/downloads/) 3.10+ (3.12 recommended) and a microphone (if you want to record directly).

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
cd Downloads\waketorch-main
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

## Hardware

You don't need a powerful computer. Training is the heavy part; once you have the `.onnx` file, any computer can run it on the CPU.
* **NVIDIA GPU:** highly recommended.
* **Generic GPU:** recommended.
* **Mac with Apple Silicon:** recommended.
* **CPU only:** works, but training is slower.

Not sure about yours? Run the device check (about a minute):

![Test Computer](static/app_check_computer.png)

* It detects your CPU, GPU, and RAM, runs a short benchmark, and recommends the best parameters for your system.
