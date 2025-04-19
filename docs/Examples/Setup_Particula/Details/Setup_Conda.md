# Setting Up a Conda Environment

With VS Code, Git, and Miniconda ready, you can now create an isolated Python environment for Particula. Using a dedicated environment ensures that Particula’s dependencies don’t interfere with other projects.

1. **Open a terminal** (VS Code has an integrated terminal you can use, or use your system’s terminal).
2. **Create a new conda environment** named `particula-env` (you can choose any name) with the latest Python version. For example:  
   ```bash
   conda create -n particula-env python=3.12 -y
   ```  
   This will create a new environment with Python 3.12. The `-y` flag auto-confirms the installation of packages.
3. **Activate the environment**:  
   ```bash
   conda activate particula-env
   ```  
   After activation, your terminal prompt should prefixed with `(particula-env)` indicating you’re now using the new environment. All subsequent commands will use this environment’s Python and libraries.

## Installing Particula in the Environment

With the `particula-env` environment active, you can install Particula itself. There are two main ways to install:

- **Using conda:** Install from conda-forge (recommended for beginners, as it ensures compatibility):  
  ```bash
  conda install -c conda-forge particula
  ```  
  This command fetches Particula from the community-maintained **conda-forge** channel. Conda will handle all dependency resolution and installation.
- **Using pip:** Alternatively, if you prefer or if you want the absolute latest version, you can use pip within the conda environment:  
  ```bash
  pip install particula
  ```  
  Since your conda environment is isolated, it’s safe to use pip here. This will download Particula from PyPI.

Both methods will result in Particula being installed in your `particula-env`. You can verify the installation by launching Python from the activated environment and importing Particula:

```bash
(particula-env) $ python -c "import particula; print('Particula version:', particula.__version__)"
```

If you see a version number printed without errors, congratulations – Particula is installed and ready! You can now begin using Particula’s API or run example scripts. We recommend using VS Code’s integrated terminal and Python support to write and run your code. If you need a refresher on Python basics before diving in, consider the official [Python Beginner’s Guide](https://wiki.python.org/moin/BeginnersGuide) for additional help.


```markdown
# Installing Particula with Conda

Conda is a cross‑platform package‑ and environment‑manager that isolates dependencies cleanly. citeturn0search14

## 1 · Prerequisites

* **Miniconda** or **Anaconda** ≥ 23.  
  Download from the official site and run the installer (accept defaults unless you know you need otherwise). citeturn0search2

## 2 · Create & activate an environment

```bash
conda create -n particula python=3.12
conda activate particula
```
Conda installs the requested Python version plus its own runtime files. citeturn0search6

## 3 · Add the conda‑forge channel

Particula is distributed via **conda‑forge**, a community‑maintained repository. citeturn0search7turn0search16

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

## 4 · Install Particula

```bash
conda install particula
```

### Extras used in the tutorials

```bash
conda install matplotlib pandas pint tqdm
```

## 5 · Upgrade / uninstall

```bash
conda update particula
conda remove particula
```

## 6 · Troubleshooting

* **UnsatisfiableError** – add `-c conda-forge` explicitly or ensure strict priority.
* **Proxy issues** – set the environment variables `HTTP_PROXY` / `HTTPS_PROXY` before running conda commands.
