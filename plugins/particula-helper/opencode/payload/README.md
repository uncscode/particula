# Particula Helper

Six skills for environment setup, aerosol science guidance, simulation design,
local execution, analysis, and debugging. The host supplies filesystem and
Python execution tools. Installing this plugin does not install Particula,
change Python environments, start a server, or run simulations.

## Install

After this distribution is merged into `uncscode/particula`:

```bash
codex plugin marketplace add uncscode/particula
codex plugin add particula-helper@particula

claude plugin marketplace add uncscode/particula
claude plugin install particula-helper@particula

opencode plugin add 'github:uncscode/particula#main::path:plugins/particula-helper/opencode'
```

Choose the commands for your host, then start a new session and ask:
"Use particula-setup to check my project's existing .venv."
See the [installation and compatibility guide](https://uncscode.github.io/particula/AI_Assistants/).

## Environment and science

Reuse the user's selected environment. `requirements.txt` is an opt-in pinned
starter for a fresh environment, not an instruction to replace an existing one.
Condensation is the introductory CPU example. Scientific checks are numerical
checks, not experimental validation. See `references/compatibility.md` for
the known coagulation limitation and host verification boundaries.

## Maintaining this package

Edit the canonical `skills/`, `scripts/`, and `references/` here. From the
repository root run:

```bash
python scripts/build_particula_plugins.py --sync
python scripts/build_particula_plugins.py --check --output /tmp/particula-plugins
```

The committed OpenCode payload is generated so Git subdirectory installation
works without accessing parent folders or running a prepare/install hook.
CI rejects drift. Plugin versions are independent of the Python library;
change all three plugin manifests and the OpenCode package version together.
Release archives are attached to normal Particula releases, not published to
an npm registry or public plugin directory automatically.
