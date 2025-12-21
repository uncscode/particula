# Code Style Guide

**Version:** 0.2.6
**Last Updated:** 2025-11-30

## Overview

This guide documents Python-specific coding standards for the **particula** repository.

> **See Also:** [Code Culture](code_culture.md) - Development philosophy, code review practices, and the "smooth is safe, safe is fast" principle including the 100-line rule for PRs.

### Language Version

**Minimum Version**: Python 3.9

**Current Development Version**: Python 3.13

From `pyproject.toml`:
```toml
requires-python = ">=3.9"
```

## Naming Conventions

### Functions/Methods: `snake_case`

Use lowercase with underscores for function and method names.

**Example:**
```python
def calculate_density(mass: float, volume: float) -> float:
    """Calculate density from mass and volume."""
    return mass / volume

def bat_activity_coefficients(molar_mass_ratio, organic_mole_fraction):
    """Calculate activity coefficients for the BAT model."""
    pass

def convert_to_oh_equivalent(oxygen2carbon, molar_mass_ratio):
    """Convert oxygen to carbon ratio to OH equivalent."""
    pass
```

**Key Points:**
- Use descriptive names that clearly indicate purpose
- Avoid abbreviations unless widely recognized (e.g., `rh` for relative humidity)
- Long names are okay if they improve clarity

### Variables: `snake_case`

Use lowercase with underscores for variable names.

**Example:**
```python
# Good
particle_diameter = 1.5e-6  # meters
molar_mass_ratio = 18.015 / 200.0
oxygen2carbon = 0.5
density_air = 1.184  # kg/m³

# Avoid
particleDiameter = 1.5e-6  # camelCase
MMR = 18.015 / 200.0  # unclear abbreviation
o2c = 0.5  # unclear abbreviation
```

**Key Points:**
- Use full words for clarity
- Include units in comments when applicable
- Use descriptive names even for temporary variables

### Constants: `UPPER_CASE`

Use uppercase with underscores for module-level constants.

**Example:**
```python
# particula/util/constants.py
AVOGADRO_NUMBER = 6.02214076e23  # mol^-1
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
GAS_CONSTANT = 8.314462618  # J/(mol·K)
STANDARD_TEMPERATURE = 298.15  # K
STANDARD_PRESSURE = 101325.0  # Pa
```

### Classes: `PascalCase`

Use PascalCase (CapitalizedWords) for class names.

**Example:**
```python
class Aerosol:
    """Aerosol class combining gas and particle phases."""
    pass

class AerosolBuilder:
    """Builder for creating Aerosol instances."""
    pass

class FitValues(NamedTuple):
    """Named tuple for BAT model fit values."""
    a1: List[float]
    a2: List[float]
    s: List[float]
```

**Key Points:**
- No underscores in class names
- Use descriptive names that indicate the class's purpose
- Builder pattern: append "Builder" to the class name

### Modules and Packages: `snake_case`

Use lowercase with underscores for module and package names.

**Example:**
```
particula/
├── activity_coefficients.py
├── bat_blending.py
├── vapor_pressure_strategies.py
├── atmosphere_builders.py
```

**Key Points:**
- Keep module names short but descriptive
- Use singular nouns unless the module contains a collection
- Avoid dots in module names (use underscores instead)

### Private Variables and Methods: `_leading_underscore`

Use a leading underscore for internal/private variables and methods.

**Example:**
```python
class Calculator:
    def __init__(self):
        self._internal_cache = {}  # Private instance variable
        
    def _helper_method(self, x):
        """Internal helper method."""
        return x * 2
        
    def public_method(self, x):
        """Public API method."""
        return self._helper_method(x) + 1
```

### Type Variables: `PascalCase`

Use PascalCase for type variables.

**Example:**
```python
from typing import TypeVar

T = TypeVar('T')
NumberType = TypeVar('NumberType', int, float)
```

## Code Formatting

### Line Length

**Maximum**: 80 characters

From `pyproject.toml`:
```toml
[tool.ruff]
line-length = 80

[tool.ruff.format]
docstring-code-line-length = 80
```

**Break long lines:**
```python
# Good - function arguments broken across lines
def bat_activity_coefficients(
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    organic_mole_fraction: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate activity coefficients."""
    pass

# Good - long expressions broken with parentheses
result = (
    long_variable_name_one 
    + long_variable_name_two 
    + long_variable_name_three
)
```

### Indentation

Use **4 spaces** per indentation level (never tabs).

```python
def function():
    if condition:
        do_something()
        if nested_condition:
            do_nested_thing()
```

### Blank Lines

- **Two blank lines** between top-level functions and classes
- **One blank line** between methods in a class
- **One blank line** to separate logical sections within functions (sparingly)

```python
"""Module docstring."""

import numpy as np


CONSTANT_VALUE = 42


def first_function():
    """First function."""
    pass


def second_function():
    """Second function."""
    pass


class MyClass:
    """A class."""
    
    def method_one(self):
        """Method one."""
        pass
    
    def method_two(self):
        """Method two."""
        pass
```

### Imports

**Order** (enforced by ruff):
1. Standard library imports
2. Third-party imports
3. Local application imports

**Format:**
```python
# Standard library
import os
import sys
from typing import List, Optional, Tuple, Union

# Third-party
import numpy as np
from numpy.typing import NDArray

# Local
from particula.activity.bat_blending import bat_blending_weights
from particula.util.validate_inputs import validate_inputs
```

**Key Points:**
- One import per line for explicit imports
- Group imports with blank lines
- Sort imports alphabetically within each group (ruff handles this)
- Use absolute imports for particula modules

### String Quotes

**Preference**: Double quotes `"`

From `pyproject.toml`:
```toml
[tool.ruff.format]
quote-style = "double"  # implicit default
```

**Example:**
```python
# Good
message = "Hello, world!"
docstring = """Multi-line docstring."""

# Also acceptable for avoiding escapes
message_with_quote = 'He said "Hello"'
```

### Trailing Commas

Use trailing commas in multi-line structures (enforced by ruff format):

```python
# Lists
values = [
    1.0,
    2.0,
    3.0,  # Trailing comma
]

# Function arguments
def function(
    arg1: float,
    arg2: float,
    arg3: float,  # Trailing comma
):
    pass
```

**Benefits:**
- Cleaner diffs when adding items
- Prevents errors from missing commas

## Type Hints

### When to Use Type Hints

**Always use type hints for:**
- Public function/method parameters
- Public function/method return types
- Class attributes (when not obvious)

**Example:**
```python
from typing import Union, Optional, Tuple
from numpy.typing import NDArray
import numpy as np

def calculate_density(
    mass: Union[float, NDArray[np.float64]],
    volume: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate density from mass and volume.
    
    Args:
        mass: Mass in kg.
        volume: Volume in m³.
        
    Returns:
        Density in kg/m³.
    """
    return mass / volume
```

### Type Hint Style

**Use Union for multiple types:**
```python
# Good
value: Union[float, int]
array: Union[float, NDArray[np.float64]]

# Note: Python 3.10+ supports `float | int`, but we target 3.9+
```

**Use Optional for None:**
```python
# Good
def process(data: Optional[np.ndarray] = None) -> float:
    if data is None:
        data = np.array([1.0, 2.0, 3.0])
    return np.mean(data)
```

**Use NDArray for numpy arrays:**
```python
from numpy.typing import NDArray
import numpy as np

def process_array(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Process numpy array."""
    return arr * 2.0
```

### Relaxed Type Checking

From `pyproject.toml`:
```toml
[tool.ruff.lint]
extend-ignore = [
  "ANN",   # ignore all missing-type-*/missing-return-type checks
]
```

This means:
- Type hints are encouraged but not strictly enforced
- You can add type hints incrementally
- Focus on public APIs first

## Docstrings

### Style: Google

From `pyproject.toml`:
```toml
[tool.ruff.lint.pydocstyle]
convention = "google"
```

**See**: [Docstring Guide](docstring_guide.md) for comprehensive documentation standards.

### Basic Structure

```python
def function(arg1: float, arg2: str) -> bool:
    """Short one-line summary (imperative mood).
    
    Optional longer description that provides more detail about what
    the function does, how it works, or when to use it.
    
    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: When arg1 is negative.
    """
    if arg1 < 0:
        raise ValueError("arg1 must be non-negative")
    return arg1 > 0
```

### Module Docstrings

Every module should have a docstring at the top:

```python
"""Activity coefficients for organic-water mixtures.

Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019
"""

import numpy as np
# ... rest of module
```

## Comments

### When to Comment

**Do comment:**
- Complex algorithms or calculations
- Non-obvious business logic
- Workarounds for bugs or limitations
- Citations for equations or methods
- Units for physical quantities

**Don't comment:**
- Obvious code (let the code speak)
- Redundant docstrings

### Comment Style

```python
# Good - explains the "why"
# Use safe exponential to prevent overflow in BAT calculation
safe_exp = get_safe_exp()
result = safe_exp(large_value)

# Good - cites source
# From Gorkowski et al. (2019), Equation 7
blending_weight = (1.0 + np.exp(-s * (x - x0))) ** (-1)

# Good - includes units
temperature = 298.15  # K
pressure = 101325.0  # Pa
density = 1.184  # kg/m³

# Avoid - redundant
# Calculate density
density = mass / volume  # This is obvious from the code
```

### Inline Comments

Use sparingly, only when code logic is non-obvious:

```python
# Good - clarifies non-obvious logic
if np.any(phase_sep_flag):  # Check for phase separation
    result = blended_value
else:
    result = homogeneous_value

# Avoid - states the obvious
x = x + 1  # Increment x
```

## Code Organization

### Module Structure

Organize modules in this order:

1. Module docstring
2. Imports (standard lib, third-party, local)
3. Module-level constants
4. Module-level variables (if any)
5. Functions (public first, then private)
6. Classes (public first, then private)

**Example:**
```python
"""Module for calculating particle properties.

Brief description of module purpose and any relevant citations.
"""

# Standard library
from typing import Union

# Third-party
import numpy as np
from numpy.typing import NDArray

# Local
from particula.util.constants import BOLTZMANN_CONSTANT

# Constants
DEFAULT_TEMPERATURE = 298.15  # K

# Public functions
def calculate_property(value: float) -> float:
    """Calculate property."""
    return _helper(value)

# Private functions
def _helper(value: float) -> float:
    """Internal helper."""
    return value * 2

# Public classes
class PropertyCalculator:
    """Calculator for properties."""
    pass
```

### File Organization

Keep modules focused and reasonably sized:
- **Prefer**: Multiple small, focused modules
- **Avoid**: Single large modules with many responsibilities

**Example** (from particula/activity):
```
activity/
├── activity_coefficients.py   # BAT activity coefficients
├── bat_blending.py            # Blending weights
├── bat_coefficients.py        # Fit coefficients
├── convert_functional_group.py  # Functional group conversion
├── gibbs.py                   # Gibbs free energy
├── gibbs_mixing.py            # Gibbs mixing
├── phase_separation.py        # Phase separation
```

## Best Practices

### Use Numpy for Numerical Computations

```python
# Good - vectorized numpy operations
import numpy as np

def calculate_all(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate for all values at once."""
    return values ** 2 + 2 * values + 1

# Avoid - Python loops for array operations
def calculate_all_slow(values: list) -> list:
    """Slow version with loops."""
    return [x**2 + 2*x + 1 for x in values]
```

### Input Validation

Use the `@validate_inputs` decorator for public functions:

```python
from particula.util.validate_inputs import validate_inputs

@validate_inputs({
    "mass": "positive",
    "volume": "positive",
    "temperature": "nonnegative",
})
def calculate_density(
    mass: float,
    volume: float,
    temperature: float = 298.15,
) -> float:
    """Calculate density with validation."""
    return mass / volume
```

**Available validators:**
- `"positive"`: Value must be > 0
- `"nonnegative"`: Value must be >= 0
- `"finite"`: Value must be finite (not inf or nan)

### Avoid Mutable Default Arguments

```python
# Bad
def append_to_list(item, items=[]):
    items.append(item)
    return items

# Good
def append_to_list(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

### Use Context Managers

```python
# Good - file is always closed
with open("data.txt", "r") as f:
    data = f.read()

# Avoid - file might not be closed on error
f = open("data.txt", "r")
data = f.read()
f.close()
```

### Explicit is Better Than Implicit

```python
# Good - clear and explicit
import numpy as np
array = np.array([1.0, 2.0, 3.0])
result = np.mean(array)

# Avoid - unclear what 'mean' is
from numpy import *
array = array([1.0, 2.0, 3.0])
result = mean(array)
```

## Anti-Patterns to Avoid

### Magic Numbers

```python
# Bad
if temperature > 373.15:
    print("Water is boiling")

# Good
WATER_BOILING_POINT = 373.15  # K at 1 atm

if temperature > WATER_BOILING_POINT:
    print("Water is boiling")
```

### Deeply Nested Code

```python
# Bad
def process(data):
    if data is not None:
        if len(data) > 0:
            if validate(data):
                if transform(data):
                    return compute(data)
    return None

# Good - early returns
def process(data):
    if data is None:
        return None
    if len(data) == 0:
        return None
    if not validate(data):
        return None
    if not transform(data):
        return None
    return compute(data)
```

### Unnecessary Comprehensions

```python
# Bad - unnecessary list comprehension
sum([x**2 for x in values])

# Good - generator expression
sum(x**2 for x in values)

# Best - numpy for numerical operations
np.sum(values**2)
```

## Testing Conventions

**See**: [Testing Guide](testing_guide.md) for comprehensive testing standards.

**Key points:**
- Test files: `*_test.py` suffix
- Test functions: `test_` prefix
- Test classes: `Test` prefix
- Assertions allowed in test files (S101 ignored)

## Summary

**Key Style Requirements:**
1. ✅ Use `snake_case` for functions, variables, modules
2. ✅ Use `PascalCase` for classes
3. ✅ Use `UPPER_CASE` for constants
4. ✅ Maximum line length: 80 characters
5. ✅ Google-style docstrings
6. ✅ Type hints for public APIs
7. ✅ Import order: stdlib → third-party → local
8. ✅ 4 spaces for indentation (never tabs)
9. ✅ Double quotes for strings
10. ✅ Validate inputs with `@validate_inputs` decorator

**Quick Reference:**
```python
"""Module docstring."""

from typing import Union
import numpy as np
from particula.util import validate_inputs

CONSTANT_VALUE = 42.0

@validate_inputs({"mass": "positive", "volume": "positive"})
def calculate_density(
    mass: Union[float, np.ndarray],
    volume: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Calculate density from mass and volume.
    
    Args:
        mass: Mass in kg.
        volume: Volume in m³.
        
    Returns:
        Density in kg/m³.
    """
    return mass / volume
```
