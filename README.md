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

WakeTorch runs on **macOS, Windows and Linux**. The installer creates a private environment, installs
every dependency and adds a `waketorch` command to your terminal.

**You will need:**

* **Python 3.10 or newer** (3.12 is recommended).
* **A microphone**, to record your wake word.
* **An internet connection** for the installation. PyTorch is a large download, so the first install
  can take a few minutes.

### 1. Download WakeTorch

Pick one option.

**Option A — Download the ZIP (no Git needed)**

1. Open the repository page on GitHub: <https://github.com/alejandrolopezdata/waketorch>
2. Click the green **Code** button, then **Download ZIP**.
3. Unzip it. You will get a folder called `waketorch-main`.

**Option B — Clone with Git**

```bash
git clone https://github.com/alejandrolopezdata/waketorch.git
```

This creates a folder called `waketorch`. In the commands below, use `waketorch` instead of `waketorch-main`.

### 2. Install it

#### macOS

1. **Install Python** if you don't have 3.10 or newer. With [Homebrew](https://brew.sh):

   ```bash
   brew install python@3.12
   ```

   Or download it from <https://www.python.org/downloads/>.

2. **Open Terminal** (press `Cmd + Space`, type "Terminal" and press Enter) and go to the folder:

   ```bash
   cd ~/Downloads/waketorch-main
   ```

3. **Run the installer:**

   ```bash
   chmod +x install.sh
   ./install.sh
   ```

4. **Open a new Terminal window** and run:

   ```bash
   waketorch
   ```

> **Microphone access:** the first time you record, macOS asks for permission. If it doesn't,
> go to **System Settings → Privacy & Security → Microphone** and enable your terminal app.

#### Windows

1. **Install Python** if you don't have 3.10 or newer. Download it from
   <https://www.python.org/downloads/> and, in the installer, **tick "Add python.exe to PATH"**.

2. **Unzip the download** (right-click the ZIP → **Extract All**) and open the `waketorch-main` folder.

3. **Run the installer:** double-click **`install.bat`**.

   Or, from Command Prompt (CMD) or PowerShell:

   ```bat
   cd %USERPROFILE%\Downloads\waketorch-main
   install.bat
   ```

   > If Windows shows "Windows protected your PC", click **More info → Run anyway**.

4. **Open a new terminal window** (CMD, PowerShell or Windows Terminal) and run:

   ```bat
   waketorch
   ```

> **Tip:** [Windows Terminal](https://aka.ms/terminal) gives the best look. The classic console works too.

#### Linux

1. **Install Python and the system packages** you need. On Debian / Ubuntu:

   ```bash
   sudo apt update
   sudo apt install python3 python3-venv python3-pip libportaudio2
   ```

   `libportaudio2` is required to record from the microphone. On other distributions, install
   Python 3.10+, the `venv` module and PortAudio with your package manager.

2. **Open a terminal** and go to the folder:

   ```bash
   cd ~/Downloads/waketorch-main
   ```

3. **Run the installer:**

   ```bash
   chmod +x install.sh
   ./install.sh
   ```

4. **Open a new terminal** and run:

   ```bash
   waketorch
   ```

### What the installer does

* Finds a compatible Python (3.10 or newer).
* Creates a private environment in the `.venv` folder, so nothing is installed system-wide.
* Installs all the dependencies from `pyproject.toml`.
* Creates the `waketorch` command and adds it to your `PATH`.

Your recordings, projects and models stay on your computer, inside the `projects/` folder.
Nothing is uploaded anywhere.

### Update

* **Cloned with Git:** run `git pull`, then run the installer again.
* **Downloaded the ZIP:** download the new ZIP, copy your `projects/` folder into it, and run the installer.

### Uninstall

| System | Command |
|---|---|
| macOS / Linux | `./install.sh uninstall` |
| Windows | `install.bat uninstall` |

This removes the `waketorch` command and the `.venv` folder. Your projects are not touched.

### Troubleshooting

* **`waketorch: command not found` / "not recognized"** — open a **new** terminal window. The installer
  updates your `PATH`, and open windows don't see the change.
* **"Python 3.10 or newer is required"** — install a newer Python (see step 2) and run the installer again.
* **The installation fails on Linux** — make sure `python3-venv` is installed (`sudo apt install python3-venv`).
* **Recording doesn't work on Linux** — install PortAudio: `sudo apt install libportaudio2`.
* **Don't move or delete the `waketorch-main` folder after installing.** The command runs the code from
  there. If you move it, run the installer again from the new location.