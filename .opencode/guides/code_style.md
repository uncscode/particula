# Code Style Guide

**Project:** particula  
**Last Updated:** 2026-06-06

This guide captures the particula-specific Python style rules migrated from
`adw-docs/code_style.md`.

## Python Version

- Minimum supported Python version: 3.12.
- Prefer Python 3.12+ syntax, including built-in generics and `|` unions.

## Naming

- Functions and methods: `snake_case`, for example `calculate_density`.
- Variables: `snake_case`, with clear scientific meaning and units when useful.
- Classes: `PascalCase`, for example `Aerosol` and `AerosolBuilder`.
- Constants: `UPPER_CASE`, for example `BOLTZMANN_CONSTANT`.
- Modules and packages: `snake_case`.
- Private helpers: `_leading_underscore`.

Avoid abbreviations unless they are standard in the domain, such as `rh` for
relative humidity.

## Formatting

- Maximum line length: 80 characters.
- Indentation: 4 spaces, never tabs.
- String quote preference: double quotes.
- Use trailing commas in multi-line literals and function signatures.
- Use two blank lines between top-level functions/classes and one blank line
  between methods.

## Imports

Group imports in this order:

1. Standard library
2. Third-party libraries
3. Local particula imports

```python
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from particula.util.constants import BOLTZMANN_CONSTANT
from particula.util.validate_inputs import validate_inputs
```

Use absolute imports for particula modules. Prefer `TYPE_CHECKING` imports for
types that would otherwise create import cycles or unnecessary runtime imports.

## Type Hints

Use type hints for public APIs and complex internal functions.

```python
from numpy.typing import NDArray
import numpy as np


def calculate_density(
    mass: float | NDArray[np.float64],
    volume: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Calculate density from mass and volume."""
    return mass / volume
```

Use `NDArray[np.float64]` for NumPy arrays where dtype matters. Add types
incrementally, prioritizing public APIs.

## Docstrings

Use Google-style docstrings. Module docstrings should describe the module and
include citations for scientific methods when applicable.

```python
"""Activity coefficients for organic-water mixtures.

Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics via an efficient
reduced-complexity model.
Atmospheric Chemistry and Physics.
https://doi.org/10.5194/acp-19-13383-2019
"""
```

## Comments

Use comments sparingly. Comments should explain why something is done, cite a
source, or clarify units.

```python
# From Gorkowski et al. (2019), Equation 7.
blending_weight = (1.0 + np.exp(-s * (x - x0))) ** (-1)

temperature = 298.15  # K
pressure = 101325.0  # Pa
```

Do not comment obvious operations.

## Scientific Code

Prefer vectorized NumPy operations for array calculations.

```python
def calculate_all(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate values using vectorized NumPy operations."""
    return values**2 + 2 * values + 1
```

Use constants from `particula.util.constants` instead of hardcoding physical
constant values.

## Input Validation

Use `@validate_inputs` for public functions that require numerical constraints.

```python
from particula.util.validate_inputs import validate_inputs


@validate_inputs({"mass": "positive", "volume": "positive"})
def calculate_density(mass: float, volume: float) -> float:
    """Calculate density from positive mass and volume."""
    return mass / volume
```

Supported validators include `"positive"`, `"nonnegative"`, and `"finite"`.

## NVIDIA Warp Notes

For Warp-heavy code, use `Any` for Warp arrays when static type checkers cannot
represent Warp runtime types. Detect Warp arrays by duck typing, not
`isinstance(obj, wp.array)`.

```python
from typing import Any

import numpy as np


def to_numpy(values: Any) -> np.ndarray:
    """Convert Warp-like or array-like values to NumPy."""
    if hasattr(values, "numpy") and callable(values.numpy):
        return values.numpy()
    return np.asarray(values)
```

GPU implementations must match the Python/NumPy reference physics. Do not use
reduced-order approximations unless the model explicitly requires them.

## Anti-Patterns

- Do not use mutable default arguments.
- Do not use `from numpy import *` or broad wildcard imports.
- Do not hardcode physical constants that already exist in `util.constants`.
- Do not bury domain equations in uncommented code.
- Do not rely on Python loops for large numerical arrays when vectorization is practical.

## Summary

- Python 3.12+.
- 80-character lines.
- Google-style docstrings.
- `snake_case` functions/modules and `PascalCase` classes.
- NumPy for scientific computation.
- Explicit units and citations for physical calculations.
- `validate_inputs` for public numerical validation.
