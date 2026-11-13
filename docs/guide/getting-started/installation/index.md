# Setup Particula

These docs use the **v0.3.0 native development API**. For these examples,
clone the repository and install the matching checkout:

```bash
git clone https://github.com/uncscode/particula.git
cd particula
python -m pip install -e ".[extra]"
```

Then [run your first CPU simulation](../first-simulation.md).
The package-manager commands below install the published distribution, which may
predate the native API. Documentation does not announce a new release.

Choose the route that matches your comfort level:

* I already use Python every day → **_Quick‑start for Experienced Users_**
* I’m new to Python → [**_Guide for New Users_**](new-to-python.md)
* I want to contribute to Particula → [**_Contributing to Particula_**](../../../contribute/setup.md)

---

## Quick‑start for Experienced Users

### Create Isolated Environment
  
   *See the [conda](conda.md), [pip](pip.md), or [uv](uv.md) guides for exact commands.*

### Install Particula

If your Python Environment is already set up, install Particula directly using one of the following methods:

---

### :simple-uv: **uv** Installation

[**uv**](uv.md) is a fast rust-based Python package manager.

```bash
uv pip install particula                   # uv ≈ much faster pip drop‑in 
```

with optional extras:
```bash
uv pip install "particula[extra]"
```

---

### :simple-pypi: PyPI Installation

[**pip**](pip.md) is Python’s default package manager and installs packages from the Python Package Index (PyPI).

```bash
pip install particula                   # PyPI
```

with optional extras:
```bash
pip install "particula[extra]"
```

---

### :simple-condaforge: Conda Installation

[**conda**](conda.md) is a cross‑platform environment & package manager; Particula is distributed via the community‑maintained conda‑forge channel.

```bash
conda install -c conda-forge particula
```

Optional extras:
```bash
conda install -c conda-forge particula matplotlib pandas tqdm pint 
```

---

