# {{PROJECT_NAME}} developer docs (Python)

This scaffold provides ready-to-use developer documentation for {{PROJECT_NAME}} built around the Python tooling stack. Keep the placeholders intact until you run the docs apply step; the values will be filled from your docs manifest.

## What lives here

- [Testing Guide](./testing_guide.md) — how to structure, name, and run tests with pytest and coverage
- [Code Style](./code_style.md) — naming, imports, typing, and formatting rules
- [Linting Guide](./linting_guide.md) — lint, format, and type-check commands
- [Docstring Guide](./docstring_guide.md) — {{DOCSTRING_STYLE}} docstrings with examples
- [Commit Conventions](./commit_conventions.md) — semantic commit rules and examples
- [PR Conventions](./pr_conventions.md) — PR template, checklist, and reviewer tips

## Quick start

```bash
# Install dependencies (example)
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
{{TEST_COMMAND}}

# Run coverage
{{TEST_COMMAND}} --cov={{PACKAGE_NAME}} --cov-report=term-missing

# Lint, format, and type-check
{{LINT_COMMAND}}
{{FORMAT_COMMAND}}
{{TYPE_CHECK_COMMAND}}
```

- Minimum Python version: {{PYTHON_VERSION}}
- Line length target: {{LINE_LENGTH}} characters
- Coverage target: {{COVERAGE_THRESHOLD}}%

## How to use these stubs

1) **Scaffold** the docs for Python:
   ```bash
   adw setup docs scaffold --language python
   ```
2) **Edit** `.adw-docs-manifest.yaml` (or your manifest) to set placeholder values.
3) **Apply** the manifest to replace placeholders:
   ```bash
   adw setup docs apply
   ```
4) **Customize** sections as your project evolves. Keep placeholders for values controlled by the manifest so future apply steps remain idempotent.

## Contributing to docs

- Keep code examples syntactically valid Python that respects the style rules in [Code Style](./code_style.md).
- Prefer concise, actionable guidance over long essays; link to the other guides instead of duplicating content.
- When adding new placeholders, also add defaults and descriptions to `adw/templates/keyword_manifest.yaml`.

## Related guides

- [Testing Guide](./testing_guide.md)
- [Code Style](./code_style.md)
- [Linting Guide](./linting_guide.md)
- [Docstring Guide](./docstring_guide.md)
- [Commit Conventions](./commit_conventions.md)
- [PR Conventions](./pr_conventions.md)
