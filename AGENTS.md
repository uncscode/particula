# AGENTS.md — Particula development

Particula is a Python aerosol particle simulator. This checkout targets
**0.3.0** and requires **Python 3.12+**. Upstream is
https://github.com/uncscode/particula; a local `origin` may be a fork.
`particula/__init__.py` owns the version read dynamically by Flit.
A version bump does not publish a release.

## Task guidance

Repository guidance is maintained as skills in
[Particula Helper](plugins/particula-helper/README.md), packaged for Codex,
Claude and OpenCode. Read the relevant skill before doing that work; do not
load every skill for a small task. These links work directly from a checkout
without installing the plugin. Installed plugin skills operate on the user's
selected checkout, not on their cached installation directory.

| Task | Skill |
| --- | --- |
| Plan a library feature, maintenance change or research task | [particula-plan](plugins/particula-helper/skills/particula-plan/SKILL.md) |
| Implement/refactor library or GPU code | [particula-develop](plugins/particula-helper/skills/particula-develop/SKILL.md) |
| Write/run tests, lint or release validation | [particula-test](plugins/particula-helper/skills/particula-test/SKILL.md) |
| Edit docstrings, guides, theory or published examples | [particula-docs](plugins/particula-helper/skills/particula-docs/SKILL.md) |
| Review changes or prepare requested commits/PRs | [particula-review](plugins/particula-helper/skills/particula-review/SKILL.md) |

For using Particula in a scientific study, the plugin separately provides
[setup](plugins/particula-helper/skills/particula-setup/SKILL.md),
[science guidance](plugins/particula-helper/skills/particula-guide/SKILL.md),
[simulation design](plugins/particula-helper/skills/particula-build/SKILL.md),
[execution](plugins/particula-helper/skills/particula-run/SKILL.md),
[analysis](plugins/particula-helper/skills/particula-analyze/SKILL.md), and
[debugging](plugins/particula-helper/skills/particula-debug/SKILL.md).
`particula-build` designs studies; `particula-develop` changes the library.

## Authority and working rules

- Follow the user's scope and this checkout's instructions. Current
  `pyproject.toml`, CI and owning implementation establish executable behavior;
  use the maintained guides to understand intended contracts. If they disagree,
  investigate and report the conflict instead of treating archived text as law.
- Use the selected environment, normally the existing `.venv/bin/python`,
  `.venv/bin/pytest`, and `.venv/bin/ruff`. Preserve unrelated working changes.
- Keep scientific units, assumptions, conservation and failure boundaries
  explicit. Never weaken a numerical test or substitute CPU for required CUDA
  to obtain a passing result.
- Use direct development commands from the
  [validation reference](plugins/particula-helper/references/repository-validation.md).
  The retired ADW/OpenCode development system is not required. Its archive is
  historical reference only; do not load it as active agent instructions.
- Edit the root `readme.md` only when the user explicitly requests README
  changes. A general feature, documentation, packaging, or release task does
  not authorize README additions. Keep the README focused on the project
  overview, installation, a quick start, and links. Put detailed usage,
  implementation, release, and troubleshooting guidance in the owning `docs/`
  page or skill reference instead.
- Keep shared conventions and detailed contracts in the linked skills/references
  and owning documentation rather than copying them back into this file.

### CodeGraph

In an indexed repository (`.codegraph/` exists), use CodeGraph before text
search or file reads to locate or understand code. Load the `codegraph` skill
at `~/.codex/skills/codegraph/SKILL.md` for setup and navigation. Prefer
`codegraph_explore` with this repository's path and a concrete file or symbol;
`codegraph explore` is the CLI fallback. Reuse existing indexes. Initialize an
unindexed repository when needed and keep `.codegraph/` ignored by Git.
Use ordinary search for documentation, configuration, or unavailable coverage.

## Maintained references

- [Repository conventions](plugins/particula-helper/references/repository-conventions.md):
  layout, Python style, public/concrete API boundaries and commit/PR conventions.
- [Scientific boundaries](plugins/particula-helper/references/scientific-boundaries.md):
  population measures, native ownership, process semantics, GPU state and evidence.
- [Validation](plugins/particula-helper/references/repository-validation.md):
  suite selection, numerical oracles, warning policy, coverage and release gates.
- [Docstrings](plugins/particula-helper/references/docstrings.md): units, shapes,
  equations and accurate Google-style API descriptions.
- [Data model](docs/guide/data-model/index.md),
  [processes](docs/guide/processes/index.md),
  [GPU support](docs/guide/gpu/support.md), and
  [migration](docs/guide/migration/v0.3.0/index.md): detailed supported contracts.
- [Documentation authoring](website/AUTHORING.md) and
  [documentation placement](docs/contribute/documentation.md): Python sources,
  saved results and Astro/Starlight builds; no paired notebook maintenance.
- [Conda release checks](conda/README.md) and
  [release evidence](docs/contribute/validation/releases/v0.3.0.md): validation
  is distinct from release approval and external feedstock success.

**Last updated:** 2026-11-13
