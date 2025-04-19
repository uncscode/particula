# New to Python?

This guide walks you from *no* Python on your machine to a working Particula
installation.  You will  

1. pick a code editor  
2. pick a python package manager,  
3. create / activate an isolated environment,  
4. install Particula, and  
5. Explore the documentation.

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

* [Install with uv](Setup_UV.md)  
* [Install with pip](Setup_PIP.md)
* [Install with conda](Setup_Conda.md)

---

## 5 · Next steps  

• Ready to dive deeper?  Browse the documentation and example gallery.  
• Want to contribute code?  See the  
  [Contributor Workflow](Contributor_Workflow.md) and install Particula in
  editable `[dev,extra]` mode.  
