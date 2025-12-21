# Code Style

This guide summarizes Python coding conventions for {{PACKAGE_NAME}}. Match these rules to keep diffs small and reviews focused on behavior.

## Core rules

- Line length: **{{LINE_LENGTH}}** characters for code and docs.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Imports: standard library first, third-party next, local last; separate groups with a blank line and keep alphabetical inside each group.
- Typing: add type hints to public functions and data structures; prefer `typing` generics (e.g., `dict[str, int]`).
- Errors: raise specific exceptions with actionable messages; avoid bare `except`.
- Logging: use the module-level logger (`logger = logging.getLogger(__name__)`) instead of `print`.

## Module layout example

```python
"""Module description."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from third_party_lib import Client

from {{PACKAGE_NAME}}.core.models import Item

logger = logging.getLogger(__name__)


def process(items: Iterable[Item]) -> list[Item]:
    """Validate and normalize items before persistence."""

    normalized: list[Item] = []
    for item in items:
        normalized.append(item.normalize())
    return normalized
```

## Functions and classes

- Prefer small, single-purpose functions.
- Keep argument lists short; use dataclasses or TypedDict when passing structured data.
- Document behavior and edge cases in docstrings (see [Docstring Guide](./docstring_guide.md)).

## Collections and defaults

- Use immutable defaults; never use mutable objects as default args.

```python
def build_config(overrides: dict[str, str] | None = None) -> dict[str, str]:
    base = {"timeout": "5", "retries": "3"}
    if overrides:
        base.update(overrides)
    return base
```

## Boolean clarity

- Prefer positive conditionals and early returns to reduce nesting.

```python
def should_retry(status_code: int) -> bool:
    if status_code in {400, 401}:
        return False
    return status_code >= 500
```

## Imports and dependency boundaries

- Avoid circular imports; move imports inside functions when necessary to prevent cycles.
- Keep third-party dependencies centralized; avoid importing heavy modules in hot paths unless required.

## Formatting and linting

- Run `{{FORMAT_COMMAND}}` to format code and `{{LINT_COMMAND}}` to enforce style and correctness.
- Fix lint warnings instead of silencing them; use `# noqa` only when justified and documented.

## Testing alignment

- Mirror code style in tests: same line length, import order, and typing conventions. See [Testing Guide](./testing_guide.md) for naming and layout.
