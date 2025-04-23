# Setup via pip

`pip` is Python’s default package manager—already on most systems and easy to
learn.  If you have Python, you have pip!  Use it inside a *virtual
environment* to keep Particula’s dependencies isolated from other projects.

If you need to install Python, use the [miniconda distribution](https://www.anaconda.com/docs/getting-started/miniconda/main) (which includes pip).  If you prefer to install Python separately, see the [Python.org](https://www.python.org/downloads/) page for instructions.

## 1. Ensure pip is up‑to‑date

Upgrade pip itself (recommended):

```bash
python -m pip install --upgrade pip
```

## 2. Create a virtual environment (recommended)

Create an isolated environment named `.venv`:

```bash
python -m venv .venv
```

Activate it on Linux / macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows (CMD or PowerShell):

```bash
.\.venv\Scripts\activate
```

## 3. Install Particula

Install the core package:

```bash
pip install particula
```

Need the tutorial extras (plots, progress‑bars, etc.)?  Install them with:

```bash
pip install "particula[extra]"
```

## 4. Upgrade / Uninstall

Upgrade Particula:

```bash
pip install -U particula
```

Uninstall Particula:

```bash
pip uninstall particula
```

## 6. Developing Particula from source

If you want to contribute to Particula, see the [Contributor Setup](Contributor_Setup.md) section for details on setting up a development environment and workflow.

Install Particula **editable + dev extras**:

```bash
pip install -e ".[dev,extra]"
```

The package is now linked to your working copy—changes you make in the repository are picked up immediately.

