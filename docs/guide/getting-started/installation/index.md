# Setup Particula

## Run the first CPU simulation

With Python 3.12 or newer in an active virtual environment, install Particula 0.3.5 or newer:

```bash
python -m pip install "particula>=0.3.5"
```

Then [download and run the first CPU simulation](../first-simulation.md).
If you already work in a development checkout, install its matching source
from the repository root with
`python -m pip install -e .` and use the checkout command in the guide.

## Other installation options

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

Wheels built from this source include version-matched documentation. After
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
`studies/`, `data/`, and `images/` trees, preserving source bytes, directory
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

## CPU dependencies and application logging

The development package requires Python 3.12 or newer and installs NumPy,
SciPy, typing extensions and Warp. CPU simulations do not require a CUDA GPU;
Warp remains a declared dependency even for CPU use. A separate distribution
without Warp is not currently provided.

Importing Particula leaves application logging configuration unchanged and
does not create package-local log files. Configure Python logging in your
application when output is wanted, for example with `logging.basicConfig`.
The `particula` logger accepts application handlers and levels. Calling
`particula.logger_setup.setup()` explicitly retains the legacy file-logging
configuration and requires write access to its configured destination.
