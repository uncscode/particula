# Linting Guide

This guide covers linting, formatting, and type-checking for {{PACKAGE_NAME}}. Run these commands locally before pushing to keep CI green.

## Commands (with placeholders)

```bash
# Lint for style and correctness
{{LINT_COMMAND}}

# Format code
{{FORMAT_COMMAND}}

# Type check
{{TYPE_CHECK_COMMAND}}
```

- Keep these placeholders intact; they will be replaced by your manifest when you apply templates.
- Use `-v` flags sparingly; default output is usually enough to fix issues.

## Ruff (lint + format)

- `{{LINT_COMMAND}}` should report errors and warnings; fix them rather than silencing.
- `{{FORMAT_COMMAND}}` auto-formats to the project standard (line length {{LINE_LENGTH}}).
- For intentional ignores, prefer narrow `# noqa: XYZ` on the specific line and add a brief comment.

## Mypy (type checking)

- Run `{{TYPE_CHECK_COMMAND}}` to ensure type coverage for public APIs.
- Add precise types to function signatures and dataclasses; avoid `Any` unless absolutely necessary.
- When you must silence an error, use `# type: ignore[error-code]` with a reason.

## Pre-commit hooks

Configure hooks to prevent regressions:
```bash
pre-commit install
pre-commit run --all-files
```

Recommended hooks:
- ruff (lint)
- ruff-format (format)
- mypy (type check)
- trailing-whitespace / end-of-file-fixer

## Workflow tips

- Run linters before tests to catch fast failures.
- Keep imports ordered to avoid churn (see [Code Style](./code_style.md)).
- If tooling versions change, update the commands in your manifest so downstream templates stay in sync.

## Troubleshooting

- If ruff flags unused imports in tests, consider using fixtures or explicit ignores only where necessary.
- If mypy warns about third-party stubs, add or update `py.typed` or install the relevant `types-` package.
- For formatting disagreements, prefer the formatter’s output; avoid manual tweaks that will be reverted.
