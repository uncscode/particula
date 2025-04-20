# Add to __init__

## Purpose

Expose your new public symbols at the package level for easy import.

## Procedure

1. Open particula/<area>/__init__.py.
2. Add `from particula.<area> import <ClassName>, <function_name>` under the appropriate section.
3. Append the symbols to the module’s `__all__` list.

