# Pull Request Conventions

**Project:** particula  
**Last Updated:** 2026-06-06

Keep PRs focused, reviewable, and backed by targeted validation.

## Title Format

Use a short conventional title:

```text
<type>: <summary>
```

Examples:

```text
fix: correct charged wall-loss zero-field behavior
test: add staggered condensation regression cases
docs: update Jupytext notebook workflow
```

## PR Body

Include:

- Summary of behavior or documentation changed.
- Scientific or architectural context when relevant.
- Tests and lint commands run.
- Known limitations or follow-up work.

## Validation

Common validation commands:

```bash
pytest
pytest --cov=particula --cov-report=term-missing
ruff check particula/ --fix
ruff format particula/
ruff check particula/
mypy particula/ --ignore-missing-imports
```

For notebook PRs, include Jupytext sync and notebook execution status. For
performance-sensitive changes, include targeted benchmark results or explain why
the benchmark was not run.

## Scope

Prefer small PRs. If a change spans multiple modules or introduces a new model,
include architecture notes and focused tests for each affected module.
