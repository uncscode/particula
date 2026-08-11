# Documentation Guide

**Project:** particula  
**Last Updated:** 2026-08-09

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

For a dated feature closeout, also run and record the exact required command:

```bash
mkdocs build --strict
```

Closeout tables must report actual dated command outcomes, including required
coverage and optional-device pass-or-clean-skip results. Do not mark a phase
shipped from planned commands or inferred evidence.

When the current agent cannot run the wrapper directly, delegate validation to
the `docs-validator` subagent with the changed documentation paths and request
strict MkDocs validation. The subagent should check links and anchors, run the
equivalent of `mkdocs build --strict`, and return a concise pass/fail result with
the command outcome. Treat a reported strict-validation failure as blocking;
informational notices outside the changed-file scope should be reported but do
not require unrelated fixes.

Example task prompt:

```text
Validate these changed documentation files: <paths>. Check Markdown links and
anchors, run the equivalent of `mkdocs build --strict`, and report pass/fail
with the command result. Do not modify files unless a validation fix is needed.
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

Concrete modules that are intentionally unexported are implementation-boundary
APIs, not user-facing features. Document their bounded contract in module
docstrings and colocated tests when needed, but do not add them to public API
references, homepage navigation, or user-facing examples unless they become a
supported public interface. A public direct-kernel entry point may require
concrete-only configuration or caller-owned sidecars. In that case, document
the package-exported step as the supported boundary and identify the concrete
imports only as required setup; do not promote those records or helpers as
package APIs or imply hidden transfer, fallback, or high-level integration.

## Link Hygiene

Prefer relative links within the repository. After documentation changes, run
`python3 .opencode/tools/build_mkdocs.py --validate-only --strict` when
feasible to catch broken documentation structure and invalid links.
