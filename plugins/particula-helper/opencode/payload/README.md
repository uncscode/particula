# Particula Helper

Six scientific skills for environment setup, aerosol science guidance, simulation
design, execution, analysis and debugging, plus five repository development skills. The host supplies filesystem and
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
See the [installation and compatibility guide](https://uncscode.github.io/particula/guide/ai-assistants/).

## Environment and science

Reuse the user's selected environment. The native starters require an installation with the 0.3.0 native APIs; `requirements.txt` is an opt-in pinned
starter for a fresh environment, not an instruction to replace an existing one.
Condensation is the introductory CPU example. Scientific checks are numerical
checks, not experimental validation. See `references/compatibility.md` for
the known coagulation limitation and host verification boundaries.

Locate matching installed documentation with the selected interpreter's
`python -m particula_docs --json`, then search the returned directory. Wheel
installs include local guides, theory, contributor guidance, Python examples,
and data/images; editable installs point to the maintained repository docs.
See `references/sources.md` for search commands and older-version fallback.

## Repository development skills

Use these in a selected Particula source checkout. They do not apply to every
Python project or replace the six study-oriented skills.

| Skill | Use |
| --- | --- |
| [particula-plan](https://github.com/uncscode/particula/blob/main/plugins/particula-helper/skills/particula-plan/SKILL.md) | Scope changes with concrete code references and acceptance evidence |
| [particula-develop](https://github.com/uncscode/particula/blob/main/plugins/particula-helper/skills/particula-develop/SKILL.md) | Implement native library, CPU process and GPU changes |
| [particula-test](https://github.com/uncscode/particula/blob/main/plugins/particula-helper/skills/particula-test/SKILL.md) | Select suites and preserve independent scientific validation |
| [particula-docs](https://github.com/uncscode/particula/blob/main/plugins/particula-helper/skills/particula-docs/SKILL.md) | Maintain docstrings, Starlight content and saved example outputs |
| [particula-review](https://github.com/uncscode/particula/blob/main/plugins/particula-helper/skills/particula-review/SKILL.md) | Review scientific/API correctness and prepare requested commits/PRs |

In the repository, `AGENTS.md` routes to these canonical files. Shared guidance
lives in `references/repository-conventions.md`, `scientific-boundaries.md`,
`repository-validation.md`, and `docstrings.md`. This replaces the retired
root development workflow; the OpenCode host adapter here remains supported.
No skill requires the historical root workflow directory. Installed copies
must defer to the selected checkout's current instructions and configuration.

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
