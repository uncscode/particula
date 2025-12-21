# Docstring Guide

**Version:** 0.2.6
**Last Updated:** 2025-11-30

## Overview

This guide documents the documentation style conventions for the **particula** repository. All functions, classes, methods, and modules should follow **Google-style** docstring format.

### Documentation Style

particula uses **Google-style** as the standard documentation format, as configured in `pyproject.toml`:

```toml
[tool.ruff.lint.pydocstyle]
convention = "google"  # accepts "google", "numpy", or "pep257"
```

### Integration with ADW

This guide is referenced by ADW commands to understand repository-specific docstring requirements. ADW uses this guide to generate properly formatted docstrings for new functions and classes.

### See Also

- **[Docstring Function Guide](docstring_function.md)** - Detailed function docstring examples
- **[Docstring Class Guide](docstring_class.md)** - Detailed class docstring examples
- **[Code Style Guide](code_style.md)** - General coding standards including type hints

## Format Structure

### Function Example

```python
from typing import Union
from numpy.typing import NDArray
import numpy as np

def calculate_density(
    mass: Union[float, NDArray[np.float64]],
    volume: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate density from mass and volume.
    
    Computes density using the fundamental relationship: density = mass / volume.
    Supports both scalar and array inputs for vectorized operations.
    
    Args:
        mass: Mass of the object in kilograms. Can be scalar or array.
        volume: Volume of the object in cubic meters. Can be scalar or array.
    
    Returns:
        Density in kg/m³. Returns scalar if both inputs are scalars,
        otherwise returns array matching the broadcast shape.
    
    Raises:
        ValueError: If volume is zero or negative.
    
    Examples:
        >>> calculate_density(10.0, 2.0)
        5.0
        
        >>> import numpy as np
        >>> masses = np.array([10.0, 20.0, 30.0])
        >>> volumes = np.array([2.0, 4.0, 6.0])
        >>> calculate_density(masses, volumes)
        array([5., 5., 5.])
    """
    if np.any(volume <= 0):
        raise ValueError("Volume must be positive")
    return mass / volume
```

### Module Docstring Example

All modules should have a docstring at the top explaining their purpose. If the module implements methods from a scientific paper, include a citation.

```python
"""Activity coefficients for organic-water mixtures.

Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019
"""
```

## Required Sections

### For Functions

All public functions must include:

- **Brief description**: One-line summary (first line after opening `"""`)
- **Args**: List all parameters with descriptions
  - Format: `parameter_name: Description of the parameter.`
  - Include type information if not obvious from type hints
  - Mention units for physical quantities
- **Returns**: Describe the return value
  - Format: Description of what is returned
  - Include type and shape information for arrays
- **Raises** (if applicable): List exceptions that can be raised
  - Format: `ExceptionType: Condition that raises this exception.`
- **Examples** (recommended): Usage examples with doctests
  - Use `>>>` for interactive Python examples
  - Include both scalar and array examples for scientific functions

### For Classes

All public classes must include:

- **Brief description**: One-line summary
- **Attributes**: List all public attributes
  - Format: `attribute_name: Description of the attribute.`
  - Include type information
- **Methods**: Document in their own docstrings (don't list in class docstring)

### For Modules

All modules should include:

- **Brief description**: Summary of module purpose
- **Citation** (if applicable): Scientific paper reference for implemented algorithms
  - Format: Author, Title, Journal, DOI/URL

## Line Length

**Docstring Line Length**: 80 characters maximum

From `pyproject.toml`:
```toml
[tool.ruff]
line-length = 80

[tool.ruff.format]
docstring-code-line-length = 80
```

This matches the general code line length limit for consistency.

## Type Hints vs Docstrings

particula uses **type hints** for type information. Do **NOT** duplicate type information in docstrings.

**Good:**
```python
def calculate_density(
    mass: float,
    volume: float,
) -> float:
    """Calculate density from mass and volume.
    
    Args:
        mass: Mass of the object in kilograms.
        volume: Volume of the object in cubic meters.
    
    Returns:
        Density in kg/m³.
    """
    return mass / volume
```

**Bad (redundant type info):**
```python
def calculate_density(mass, volume):
    """Calculate density from mass and volume.
    
    Args:
        mass (float): Mass of the object in kilograms.
        volume (float): Volume of the object in cubic meters.
    
    Returns:
        float: Density in kg/m³.
    """
    return mass / volume
```

## Validation with Input Decorator

For public functions accepting numerical inputs, use the `@validate_inputs` decorator from `particula.util.validate_inputs`:

```python
from particula.util.validate_inputs import validate_inputs

@validate_inputs({
    "mass": "positive",
    "volume": "positive",
})
def calculate_density(mass: float, volume: float) -> float:
    """Calculate density from mass and volume.
    
    Args:
        mass: Mass of the object in kilograms. Must be positive.
        volume: Volume of the object in cubic meters. Must be positive.
    
    Returns:
        Density in kg/m³.
    """
    return mass / volume
```

Available validators: `"positive"`, `"nonnegative"`, `"finite"`

## Quick Reference

### Checklist for New Documentation

- [ ] Brief description (one line)
- [ ] Detailed description (if non-trivial or implements scientific method)
- [ ] Args section with all parameters documented
- [ ] Returns section describing return value
- [ ] Raises section (if applicable)
- [ ] Examples section (recommended, required for complex functions)
- [ ] Type hints in function signature (NOT in docstring)
- [ ] Line lengths ≤ 80 characters
- [ ] Scientific citation in module docstring (if applicable)
- [ ] Units specified for physical quantities

## See Also

- **[Linting Guide](linting_guide.md)**: Docstring linting rules (ruff pydocstyle)
- **[Code Style Guide](code_style.md)**: General coding standards and type hints
- **[Docstring Function Guide](docstring_function.md)**: Detailed function docstring template
- **[Docstring Class Guide](docstring_class.md)**: Detailed class docstring template
