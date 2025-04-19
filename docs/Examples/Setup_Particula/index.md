# Setup Particula

Welcome!  Choose the route that matches your comfort level:

* I already use Python every day → **_Quick‑start for Experienced Users_**
* I’m new to Python → [**_Guide for New Users_**](Details/New_to_Python.md)
* I want to contribute to Particula → [**_Contributing to Particula_**](Details/Contributor_Workflow.md)

---

## Quick‑start for Experienced Users

### Create Isolated Environment
  
   *See the [conda](Details/Setup_Conda.md), [pip](Details/Setup_PIP.md), or [uv](Details/Setup_UV.md) guides for exact commands.*

### Install Particula

If your Python Environment is already set up, install Particula directly using one of the following methods:

#### :simple-uv: Fast Rust-Based Python Package Manager

```bash
uv pip install particula                   # uv ≈ pip drop‑in
```

with optional extras:
```bash
uv pip install "particula[extra]"
```

#### :simple-pypi: PyPI Installation

```bash
pip install particula                   # PyPI
```

with optional extras:
```bash
pip install "particula[extra]"
```


#### :simple-condaforge: Conda Installation

```bash
conda install -c conda-forge particula
```

Optional extras:
```bash
conda install -c conda-forge particula matplotlib pandas tqdm pint 
```

---

