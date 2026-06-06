# Docstring Guide

**Project:** particula  
**Last Updated:** 2026-06-06

Use Google-style docstrings for modules, public functions, classes, and methods.
Docstrings should be accurate, concise, and useful to future scientific-code
maintainers.

## Module Docstrings

Module docstrings should state the purpose of the module and include citations
when the module implements a published method or equation.

```python
"""Activity coefficients for organic-water mixtures.

Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics via an efficient
reduced-complexity model.
Atmospheric Chemistry and Physics.
https://doi.org/10.5194/acp-19-13383-2019
"""
```

## Function Docstrings

```python
def calculate_density(mass: float, volume: float) -> float:
    """Calculate density from mass and volume.

    Args:
        mass: Particle mass in kg.
        volume: Particle volume in m^3.

    Returns:
        Density in kg/m^3.

    Raises:
        ValueError: If mass or volume is not positive.
    """
    return mass / volume
```

## Class Docstrings

Class docstrings should describe the role of the class and any important
construction constraints.

```python
class AerosolBuilder:
    """Build aerosol objects from gas and particle representations."""
```

## Scientific Documentation Rules

- Include units for physical quantities.
- Cite equations, models, and parameterizations when they come from literature.
- Explain assumptions that affect scientific validity.
- Do not duplicate obvious type information unless units or semantics need clarification.
- Keep examples short and executable when included.

## Style Rules

- First line is a short summary.
- Use `Args:`, `Returns:`, `Raises:`, and `Examples:` sections as needed.
- Use imperative or direct wording.
- Keep line length to 80 characters.
- Keep docstrings synchronized with implementation changes.

## Tests

Test functions should also have short docstrings when the behavior is not fully
obvious from the test name. Prefer docstrings that state the behavior being
validated, not the mechanics of the test.
