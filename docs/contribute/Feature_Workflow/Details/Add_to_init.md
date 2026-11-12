# Adding to `__init__.py`

## Purpose

Make new public APIs available for easy imports.

## Rules

- Only import important and relevant classes and functions.
- Do not import everything from a module (e.g., `from particula.<area> import *`).
- Only include public names (no leading `_`). These helpers are not intended for public use.
- Keep imports minimal and grouped logically.
- Maintain alphabetical order within groups.

## Steps

1. Open `particula/<area>/__init__.py`.
2. Add under the right section (nonexecutable placeholder template):

```text
from particula.<area> import ClassName, function_name
```

Documented concrete configuration and sidecar imports do not authorize package
re-exports. Preserve the approved public surface and test its exports.
