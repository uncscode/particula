# Documentation Guide

**Project:** particula  
**Last Updated:** 2026-07-04

particula documentation is split between agent-facing guides in
`.opencode/guides/` and user-facing MkDocs content in `docs/`.

## Locations

- `.opencode/guides/`: Agent and contributor guidance.
- `docs/`: User-facing MkDocs content.
- `docs/Examples/`: Tutorials, examples, runnable scripts, and paired Jupytext
  notebooks.
- `docs/Theory/`: Conceptual and theoretical explanations.
- `docs/API/`: Generated or API-oriented documentation.
- `adw-docs/`: Legacy source of migrated guide content, kept for reference.

## MkDocs

Validate documentation changes with the repository-local wrapper:

```bash
python3 .opencode/tools/build_mkdocs.py --validate-only --strict
```

Build docs directly with MkDocs when you need a full local site build:

```bash
mkdocs build
```

Serve locally with:

```bash
mkdocs serve
```

## Example Source Workflow

Examples in `docs/Examples/` may be plain runnable `.py` scripts, paired
`.py`/`.ipynb` notebooks, or both behind a topic landing page. Keep the
published runnable entrypoint current when an example intentionally exposes one.

For notebook-backed examples, use Jupytext paired sync: edit the `.py` percent
file first, then sync and execute the `.ipynb`.

```bash
ruff check docs/Examples/path/to/file.py --fix
ruff format docs/Examples/path/to/file.py
python3 .opencode/tools/validate_notebook.py docs/Examples/path/to/file.ipynb --sync
python3 .opencode/tools/run_notebook.py docs/Examples/path/to/file.ipynb
```

Commit both paired files when a notebook exists.

## Docstrings

Use Google-style docstrings. Include units and scientific citations where they
help users understand model behavior or equations.

## When to Update Docs

- Public API changes.
- New or changed scientific models.
- Changes to examples or notebooks.
- New validation rules or testing workflows.
- Architecture changes affecting module boundaries.

## Link Hygiene

Prefer relative links within the repository. After documentation changes, run
`python3 .opencode/tools/build_mkdocs.py --validate-only --strict` when
feasible to catch broken documentation structure and invalid links.
