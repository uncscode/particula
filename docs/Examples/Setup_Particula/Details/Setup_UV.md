# Installing Particula with uv

**uv** (by Astral) is a drop‑in, Rust‑powered replacement for pip that resolves, builds, and installs packages much faster. It’s compatible with pip and can be used in any virtual environment. For more details, see the [uv documentation](https://astral.sh/uv/).

# Create an environment & install Particula

```bash
uv venv .venv            # creates a .venv virtual environment in the current directory
uv pip install particula # installs Particula inside the .venv
```

`uv pip` accepts the same flags as pip, so extras work too:

```bash
uv pip install "particula[extra]"
```

## Upgrade / uninstall

```bash
uv pip install -U particula
uv pip uninstall particula
```

## Specifying a Python version

If you want to use a specific Python version, you can specify it with the `--python` flag:

```bash
uv venv .venv --python=python3.12
```

## Setup for Contributing

If you want to contribute to Particula, see the [Contributing to Particula](Contributor_Workflow.md) section for details on setting up a development environment and workflow.

Once in your development environment, you can install Particula in editable mode with the required dev dependencies:

```bash
uv pip install -e ".[dev,extra]"
```
