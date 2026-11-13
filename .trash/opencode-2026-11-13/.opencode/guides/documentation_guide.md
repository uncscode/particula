# Documentation Guide

**Project:** particula  
**Last Updated:** 2026-11-13

particula documentation is split between agent-facing guides in
`.opencode/guides/` and user-facing Starlight content in `docs/`.

Follow `docs/contribute/documentation.md` for the canonical five-section
placement, consolidation, retirement, and validation rules. Extend an owning
topic before adding a file; implementation phases are not public feature pages.

## Locations

- `.opencode/guides/`: Agent and contributor guidance.
- `docs/`: User-facing Starlight content.
- `docs/examples/`: Tutorials, examples, runnable percent-cell Python sources.
- `docs/guide/`: Supported usage, installation, migration, and GPU contracts.
- `docs/theory/`: Scientific equations, assumptions, and primary references.
- `docs/contribute/`: Contributor workflows, architecture, and durable validation.
- `docs/API/`: Generated or API-oriented documentation.
- `adw-docs/`: Legacy source of migrated guide content, kept for reference.

## Starlight

Validate documentation changes with the repository-local wrapper:

```bash
npm --prefix website run build
```

For a dated feature closeout, also run and record the exact required command:

```bash
npm --prefix website run build
```

Closeout tables must report actual dated command outcomes, including required
coverage and optional-device pass-or-clean-skip results. Do not mark a phase
shipped from planned commands or inferred evidence.

For resident-GPU closeouts, run the required groups sequentially and record the
date, Warp availability, qualified devices, literal commands, test counts,
coverage percentages, and per-target term-missing rows. Derive coverage targets
from the completed phases' executable-module diff; documentation, examples, and
tests are not executable coverage targets. A missing artifact, failed required
command, unavailable required runner, or aggregate coverage below the required
threshold blocks shipment. CUDA is optional pass-or-clean-skip evidence, never
CPU-fallback evidence. Do not claim a status of shipped or complete until the
actual strict-build and required test outcomes are available. A Not Started to
Active transition is allowed when recording a valid blocker or in-progress
remediation.

When the current agent cannot run the wrapper directly, delegate validation to
the `docs-validator` subagent with the changed documentation paths and request
Starlight validation. The subagent should check links and anchors, run the
equivalent of `npm --prefix website run build`, and return a concise pass/fail result with
the command outcome. Treat a reported strict-validation failure as blocking;
informational notices outside the changed-file scope should be reported but do
not require unrelated fixes.

Example task prompt:

```text
Validate these changed documentation files: <paths>. Check Markdown links and
anchors, run the equivalent of `npm --prefix website run build`, and report pass/fail
with the command result. Do not modify files unless a validation fix is needed.
```

Build docs directly with Starlight when you need a full local site build:

```bash
npm --prefix website run build
```

Serve locally with:

```bash
npm --prefix website run dev
```

## Example Source Workflow

Examples in `docs/examples/` and `docs/theory/` use Python sources with
percent cells for Markdown and code. Register entry points, device requirements,
timeouts and helper/data dependencies in `website/examples.json`.

```bash
.venv/bin/python scripts/docs_examples.py refresh EXAMPLE_KEY
.venv/bin/python scripts/docs_examples.py check
npm --prefix website run build
```

Inspect figures, tables and textual output, then commit the source and saved
bundle in `website/results/`. A failed refresh leaves the previous bundle intact
and blocks acceptance. Never reduce workloads or use a CPU fallback to qualify
CUDA execution. Historical viewers consume declared recorded data. Builds do
not execute examples. See `docs/contribute/Feature_Workflow/Details/Add_Example.md`.

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
`npm --prefix website run build` when
feasible to catch broken documentation structure and invalid links.
