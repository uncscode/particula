# Setup via conda

`conda` is a cross‑platform environment & package manager.  Using an isolated
Conda *environment* keeps Particula’s dependencies from colliding with other
projects.

## 1. Install Miniconda (one‑time)

Download the latest *Miniconda* installer for your OS from  
<https://docs.conda.io/en/latest/miniconda.html> and run it (accept the
defaults).  After installation, open a new terminal so the `conda` command is
on your path.

## 2. Create & activate an environment

Create an environment named `particula` with Python:

```bash
conda create -n particula python=3.12
```

Activate it:

```bash
conda activate particula
```

Your prompt now starts with `(particula)`—everything you install will stay
inside this environment.

## 3. Add the conda‑forge channel (recommended)

Particula is published on the community‑maintained **conda‑forge** channel:

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

## 4. Install Particula

Install the core package:

```bash
conda install particula
```

Need the tutorial extras (plots, progress‑bars, etc.)?

```bash
conda install matplotlib pandas pint tqdm
```

## 5. Upgrade / Uninstall

Upgrade Particula:

```bash
conda update particula
```

Uninstall Particula:

```bash
conda remove particula
```

## 6. Developing Particula from source

Working from a fork?  After activating your `particula` environment, install
Particula **editable + dev extras** with pip (inside Conda it’s safe):

Install editable:

```bash
pip install -e ".[dev,extra]"
```

If you want to contribute to Particula, see the [Contributor Setup](Contributor_Setup.md) section for details on setting up a development environment and workflow.

