# New to Python?

This guide walks you from *no* Python on your machine to a working Particula
installation.  You will  

1. pick a code editor  
2. pick a python package manager,  
3. create / activate an isolated environment,  
4. install Particula, and  
5. explore the documentation.

---

## 1 · Learn a few Python basics 📚  

New to programming?  Spend an evening with the free course
**“Python for Everybody.”**  
Videos + quizzes + e‑textbook: <https://www.py4e.com>

---

## 2 · Install the essential tools  

Choose **one code editor**.  You can always try others later, but this is a good starting point.  
The table shows the most common choices:

| Tool | Why you need it | Where to get it |
| ---- | --------------- | -------------- |
| **Visual Studio Code** | Full‑featured editor with great Python support | [Download VS Code](https://code.visualstudio.com/) |
| **Spyder** | Scientific IDE, MATLAB‑like | [Download Spyder](https://www.spyder-ide.org/) |
| **Google Colab** | Nothing to install – runs in the browser | [Google Colab](https://colab.research.google.com/) |

---

## 3 · Install a Package Manager

Pick **one package manager**.  Conda is used in the step‑by‑step below, but uv
or pip will also work (guides linked).

| Package Manager | Why you need it | Where to get it |
| --------------- | --------------- | -------------- |
| **uv** | Rust‑powered, lightning‑fast | [uv installation guide](https://docs.astral.sh/uv/) |
| **pip** | Comes with Python | [pip installation guide](https://pip.pypa.io/en/stable/installation/) |
| **Conda / Miniconda** | Easiest way to manage multiple Pythons | [Miniconda installers](https://www.anaconda.com/docs/getting-started/miniconda/main#miniconda) |

Install **Git** if you plan to contribute code (optional for pure user):

- Windows/macOS: <https://git-scm.com/downloads>  
- Linux: `sudo apt install git` or your distro’s package manager

---

## 4 · Install Particula in an isolated environment

Follow their dedicated guides:

* [Install with uv](uv.md)  
* [Install with pip](pip.md)
* [Install with conda](conda.md)

---

## 5 · Next steps  

- Ready to dive deeper?  Browse the documentation and example gallery.  
- Want to contribute code?  See the [Contributor Setup](../../../contribute/setup.md) and install Particula in editable `[dev,extra]` mode.  

---

## Troubleshooting 🛠️  

### Common pitfalls  

- **`command not found` for `python`, `conda`, `uv`, or `pip`**  
  The tool is not on your system PATH. Close/re‑open the terminal or follow the
  installer’s instructions to add it to your environment variables.  

- **`No module named particula`**  
  You’re running a Python interpreter where Particula isn’t installed.  
  Activate the correct environment (`conda activate particula`,
  `source .venv/bin/activate`, etc.) or select it inside your editor.  

- **C compiler missing**  
  Some optional dependencies need a compiler.  

  - Windows → “Build Tools for Visual Studio”  
  - macOS → `xcode-select --install`  
  - Linux → `sudo apt install build-essential` (or your distro equivalent)  

- **“Permission denied” / read‑only file system**
  Work in a directory where you have write permission,
  or add `--user` when using `pip` (less reproducible),  
  or create the environment in your home folder.  

### Still stuck? Ask a 🤖  

Copy the error message into a Large‑Language‑Model chat
(e.g. OpenAI ChatGPT, Claude, Gemini) and request an explanation plus possible
fixes.
