# Setup Particula

To run the examples from source, clone the repository and install Particula:

```bash
git clone https://github.com/uncscode/particula.git
cd particula
python -m pip install -e ".[extra]"
```

Then [run your first CPU simulation](../first-simulation.md).

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

## Local documentation

Wheels built from this 0.3.0 source include version-matched documentation. After
installing such a release with `pip install particula`, no checkout or network
connection is needed to read or search it:

```bash
particula-docs
python -m particula_docs --json
rg "ParticleData" "$(python -m particula_docs)/guide"
```

Use the Python interpreter from your Particula environment. The locator uses
only the standard library and does not import simulation code, NumPy, or Warp.
The JSON output records the package version, documentation path, file count,
uncompressed source size, and manifest path. `bundle-manifest.json` records
SHA-256 hashes and byte sizes for every bundled source file.

The bundle contains `docs/index.md` and the `guide/`, `theory/`, `contribute/`,
`examples/`, `data/`, and `images/` trees, preserving source bytes, directory
structure, and relative links between included files. Python examples remain
readable source; running them may require optional dependencies and hardware.
Generated API pages, caches, website builds/saved website results, and `.trash/`
are excluded. Links to repository tooling or rendered website resources still
require those resources; this is a searchable source bundle, not a built site.

For editable installations, the locator points directly to the repository's
maintained `docs/` directory, so edits appear immediately. `docs/` is the only
maintained documentation source; `particula/docs/` is generated in temporary
build staging and must not be edited. Older releases without the locator can
use their matching source documentation or the online documentation after
checking version compatibility.
