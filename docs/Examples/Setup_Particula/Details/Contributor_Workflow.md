# Contributor Workflow

*Developer Setup Step‑by‑Step*
---

This guide is a ground‑up walkthrough for first‑time contributors.  
You will fork Particula on GitHub (or GitHub Desktop), clone it locally,
create a private Python environment, install the package in editable
mode, and learn the branch → code → commit → PR cycle.

Interested in contributing to the Particula project? This section explains how to set up a development environment and the workflow for contributing changes. We assume you have a GitHub account and Git installed (see the [Beginner Setup](New_to_Python.md) if not). Following these steps will allow you to run the latest development version of Particula and prepare your contributions for a pull request.

## 1. Fork the repository on GitHub

First, **fork the Particula repository** to your own GitHub account. Forking creates your personal copy of the project:

- Visit the Particula GitHub repo: **<https://github.com/uncscode/particula>**.  
- Click the **“Fork”** button in the top-right corner of the page.  
- GitHub will create a fork under your account (e.g., `github.com/<your-username>/particula`).

*(If you’re new to forking, see GitHub’s guide on how to fork a repository for more details.)*

## 2. Get the code on your computer

Choose **one** of the two methods below.

**A. Git (command‑line)**  
1. Open a terminal and move to the folder where you keep projects.  
2. Clone your fork (replace `<your-username>`):  
   ```bash
   git clone https://github.com/<your-username>/particula.git
   ```  
3. Change into the project directory:  
   ```bash
   cd particula
   ```

**B. GitHub Desktop (GUI)**  
1. Install GitHub Desktop from <https://desktop.github.com/>.  
2. File → Clone repository… → URL tab → paste  
   `https://github.com/<your-username>/particula.git`.  
3. Click **Clone**; GitHub Desktop puts the files on disk and lets you open the
   folder in your code editor.

Either path leaves you with a `particula/` folder containing the source code.

## Set Up a Development Environment (`.venv`)

Create an isolated Python environment so development dependencies stay separate from other projects.
We recommend the lightning‑fast uv tool—see the [uv setup guide](Examples/Setup_Particula/Details/Setup_UV.md) for details.

- **Using uv:** You can create and activate the env in one step:  
   ```bash
   uv venv .venv      # creates & auto‑activates .venv for uv commands
   ```  
   This will create `.venv` and automatically make it active for subsequent `uv` commands.
- **Install Editable:** with virtual environment active, install Particula in **development mode** with the required dev dependencies:
   ```bash
   uv pip install -e ".[dev,extra]"
   ```

The `pip install -e ".[dev,extra]"` command tells pip to install the package in editable mode (`-e`) from the current directory (`.`) including the `[dev,extra]` optional dependencies (which include development and extra tools). This will pull in things like testing frameworks, linters, etc., as defined by Particula’s `pyproject.toml`. If using uv, run `uv pip install -e ".[dev,extra]"` equivalently.

> **Tip:** The `.[dev,extra]` syntax installs all standard and extra dependencies needed for development (such as documentation or additional features). You can inspect `pyproject.toml` for the exact extras defined.

## Development Workflow: Branch, Code, Commit, PR

You are now ready to create a **feature branch**, write code, commit, push,
and open a pull request.
