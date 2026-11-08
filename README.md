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

**Requirements:** Python 3.10+ (3.12 recommended) and a microphone.

### 1. Download

Download the ZIP from [GitHub](https://github.com/alejandrolopezdata/waketorch) (**Code → Download ZIP**) and unzip it,
or clone it:

```bash
git clone https://github.com/alejandrolopezdata/waketorch.git
```

### 2. Install

**macOS**

```bash
brew install python@3.12        # skip if you already have Python 3.10+
cd ~/Downloads/waketorch-main
chmod +x install.sh
./install.sh
```

**Linux** (Debian / Ubuntu)

```bash
sudo apt install python3 python3-venv python3-pip libportaudio2
cd ~/Downloads/waketorch-main
chmod +x install.sh
./install.sh
```

**Windows** (Command Prompt)

Install Python from [python.org](https://www.python.org/downloads/) and tick **Add python.exe to PATH**, then:

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

### Troubleshooting

* **Command not found** — open a new terminal window.
* **Don't move the folder after installing** — if you do, run the installer again.