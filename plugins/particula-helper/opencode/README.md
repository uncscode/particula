# Particula Helper

Seven researcher skills for environment setup, aerosol science guidance, simulation
design, execution, analysis, debugging and issue reporting, plus six repository
development skills. The host supplies filesystem and
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

## Report a research concern

Use [particula-report](https://github.com/uncscode/particula/blob/main/plugins/particula-helper/skills/particula-report/SKILL.md)
to investigate a suspected bug, unexpected result or perceived scientific error.
The agent searches version-matched source and existing issues, prepares a report
using `references/issue-report.md`, and asks you to review it before posting to
GitHub. A full checkout, proven root cause or fix is not required. Without
GitHub access, it provides copyable Markdown.

## Repository development skills

Use these in a selected Particula source checkout. They do not apply to every
Python project or replace the seven researcher-oriented skills.

| Skill | Use |
| --- | --- |
| [particula-plan](https://github.com/uncscode/particula/blob/main/plugins/particula-helper/skills/particula-plan/SKILL.md) | Scope changes with concrete code references and acceptance evidence |
| [particula-develop](https://github.com/uncscode/particula/blob/main/plugins/particula-helper/skills/particula-develop/SKILL.md) | Implement native library, CPU process and GPU changes |
| [particula-test](https://github.com/uncscode/particula/blob/main/plugins/particula-helper/skills/particula-test/SKILL.md) | Select suites and preserve independent scientific validation |
| [particula-docs](https://github.com/uncscode/particula/blob/main/plugins/particula-helper/skills/particula-docs/SKILL.md) | Organize the topic graph and maintain guides, theory and docstrings |
| [particula-examples](https://github.com/uncscode/particula/blob/main/plugins/particula-helper/skills/particula-examples/SKILL.md) | Create/edit worked examples with explained code, plots or process visuals, reflections and topic links |
| [particula-review](https://github.com/uncscode/particula/blob/main/plugins/particula-helper/skills/particula-review/SKILL.md) | Review scientific/API correctness and prepare requested commits/PRs |

For examples, `references/example-authoring.md` provides an adaptable structure:
question and setup, explained code, visual evidence, interpretation, closing
reflections and a next investigation. Introduce every code block in prose and
complete registration, topic categorization and navigation links.

In the repository, `AGENTS.md` routes to these canonical files. Shared guidance
lives in `references/repository-conventions.md`, `scientific-boundaries.md`,
`repository-validation.md`, and `docstrings.md`. This replaces the retired
root development workflow; the OpenCode host adapter here remains supported.
Documentation organization and narrative review share `references/documentation-topics.md`
and `references/scientific-narratives.md`. No skill requires the historical root workflow directory. Installed copies
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
change all three plugin manifests, the OpenCode package version and both root
version entries in its shrinkwrap file together. Update release notes and
regenerate the payload. After validation, a version increase on upstream main
automatically publishes `particula-helper-vVERSION` with the tested archives.
Plugin releases do not replace the library release as GitHub Latest or publish
the Python package. Existing releases and historical drafts are preserved.
Archives are also attached to normal Particula releases; npm registry and
public plugin directory publication are separate actions.
