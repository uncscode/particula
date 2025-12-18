# AGENTS.md - particula Quick Reference

**For ADW (AI Developer Workflow) agents working on the particula repository.**

## Repository Overview

**Name:** particula  
**Description:** A simple, fast, and powerful particle simulator  
**Language:** Python 3.9+  
**Repository:** https://github.com/Gorkowski/particula.git  
**Version:** 0.2.6

## Quick Start

### Build & Test Commands

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=particula --cov-report=term-missing

# Run linters (auto-fix)
ruff check particula/ --fix
ruff format particula/
ruff check particula/

# Run type checker
mypy particula/ --ignore-missing-imports

# ADW tools
.opencode/tool/run_pytest.py      # Run tests with validation
.opencode/tool/run_linters.py     # Run linters following CI workflow
```

### Installation

```bash
# Development installation
pip install -e .[dev]

# Or with uv
uv pip install -e .[dev]
```

## Code Style Essentials

### Naming Conventions
- **Functions/Variables**: `snake_case` (e.g., `calculate_density`, `molar_mass_ratio`)
- **Classes**: `PascalCase` (e.g., `Aerosol`, `AerosolBuilder`)
- **Constants**: `UPPER_CASE` (e.g., `BOLTZMANN_CONSTANT`, `GAS_CONSTANT`)
- **Modules**: `snake_case` (e.g., `activity_coefficients.py`)
- **Test files**: `*_test.py` suffix (e.g., `activity_coefficients_test.py`)

### Code Formatting
- **Line length**: 80 characters
- **Indentation**: 4 spaces (never tabs)
- **Docstrings**: Google-style
- **String quotes**: Double quotes `"`
- **Import order**: stdlib → third-party → local

### Type Hints
```python
from typing import Union
from numpy.typing import NDArray
import numpy as np

def function(
    mass: Union[float, NDArray[np.float64]],
    volume: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Short description."""
    return mass / volume
```

## Testing

**Framework:** pytest  
**Coverage:** pytest-cov  
**Minimum tests:** 500 (current: 711)  
**Test pattern:** `*_test.py`

**Test structure:**
```
particula/
├── activity/
│   ├── tests/
│   │   ├── activity_coefficients_test.py
│   │   └── ...
│   └── activity_coefficients.py
└── ...
```

## Linting

**Linters:** ruff (check + format), mypy

**CI Workflow:**
1. `ruff check particula/ --fix` (apply fixes)
2. `ruff format particula/` (format code)
3. `ruff check particula/` (final check - must pass)

**Configuration:** See `pyproject.toml` → `[tool.ruff]`

**Test files:** Assertions allowed (`S101` ignored in `*_test.py`)

## Project Structure

```
particula/
├── activity/          # Activity coefficients, phase separation
├── dynamics/          # Coagulation, condensation, wall loss
│   ├── coagulation/
│   ├── condensation/
│   └── properties/
├── equilibria/        # Partitioning calculations
├── gas/               # Gas phase, species, vapor pressure
│   └── properties/
├── particles/         # Particle distributions, representations
│   ├── distribution_strategies/
│   └── properties/
├── util/              # Utilities, constants, validation
│   ├── chemical/
│   └── lf2013_coagulation/
├── integration_tests/ # Integration tests
└── __init__.py
```

## Documentation

**Location:** `docs/Agent/`

**Key guides:**
- `testing_guide.md` - Test framework and conventions
- `linting_guide.md` - Linting tools and configuration
- `code_style.md` - Naming, formatting, type hints
- `docstring_guide.md` - Google-style docstring format
- `commit_conventions.md` - Commit message format
- `pr_conventions.md` - Pull request conventions
- `architecture_reference.md` - Module structure

## Common Patterns

### Input Validation

```python
from particula.util.validate_inputs import validate_inputs

@validate_inputs({
    "mass": "positive",
    "volume": "positive",
})
def calculate_density(mass: float, volume: float) -> float:
    """Calculate density."""
    return mass / volume
```

**Available validators:** `"positive"`, `"nonnegative"`, `"finite"`

### Module Docstrings

```python
"""Short description of module purpose.

Optional longer description or citation:

Author, A. B., & Author, C. D. (2019).
Paper title.
Journal Name
https://doi.org/...
"""
```

### Function Docstrings

```python
def function(arg1: float, arg2: str) -> bool:
    """Short one-line summary.
    
    Optional longer description.
    
    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: When arg1 is negative.
    """
    pass
```

### Wall loss strategies

```python
import particula as par

strategy = par.dynamics.SphericalWallLossStrategy(
    wall_eddy_diffusivity=0.001,
    chamber_radius=0.5,
    distribution_type="discrete",
)

wall_loss = par.dynamics.WallLoss(wall_loss_strategy=strategy)

# Split time_step across sub_steps; concentrations clamp at zero each step
aerosol = wall_loss.execute(aerosol, time_step=1.0, sub_steps=2)
```

**Key points:**

- Strategies live in `particula.dynamics.wall_loss` and are exported through
  `particula.dynamics`.
- `WallLossStrategy` is the abstract base class for wall loss models.
- `WallLoss` runnable wraps a strategy, splits `time_step` across `sub_steps`,
  clamps concentrations to non-negative, and composes with other runnables
  via `|`.
- `SphericalWallLossStrategy` provides a spherical chamber implementation
  using existing wall loss coefficient utilities.
- Supported `distribution_type` values are `"discrete"`, `"continuous_pdf"`,
  and `"particle_resolved"`.

## ADW Workflows

**Available workflows:**
```bash
adw workflow list          # List all workflows
adw workflow complete 123  # Full workflow for issue #123
adw workflow patch 456     # Quick patch for issue #456
adw workflow test          # Run tests only
adw workflow document      # Update documentation
```

**Workflow phases:**
1. **plan** - Create implementation plan
2. **build** - Implement changes
3. **lint** - Run linters (auto-fix)
4. **test** - Run tests (min 500 required)
5. **review** - Code review
6. **document** - Update docs
7. **ship** - Create PR

## Important Files

**Configuration:**
- `pyproject.toml` - Package config, ruff, pytest
- `.github/workflows/test.yml` - Test CI
- `.github/workflows/lint.yml` - Lint CI
- `.pre-commit-config.yaml` - Pre-commit hooks

**ADW Tools:**
- `.opencode/tool/run_pytest.py` - Test runner with validation
- `.opencode/tool/run_linters.py` - Linter runner following CI workflow

## Dependencies

**Core:**
- numpy >= 2.0.0
- scipy >= 1.12

**Development:**
- pytest (testing)
- ruff (linting + formatting)
- mypy (type checking - optional)
- mkdocs, mkdocs-material (documentation)

## Key Principles

1. **Scientific Computing**: Use NumPy for vectorized operations
2. **Validation**: Use `@validate_inputs` decorator for public functions
3. **Testing**: Maintain >500 tests, use `*_test.py` pattern
4. **Documentation**: Google-style docstrings with citations
5. **Code Quality**: Ruff (check + format) + mypy
6. **Line Length**: 80 characters max
7. **Type Hints**: Required for public APIs

## Quick Checks

Before committing:
```bash
# Run linters
ruff check particula/ --fix && ruff format particula/ && ruff check particula/

# Run tests
pytest --cov=particula

# Or use ADW tools
.opencode/tool/run_linters.py && .opencode/tool/run_pytest.py
```

## Getting Help

**Documentation:**
- Full guides: `docs/Agent/`
- Examples: `docs/Examples/`
- Theory: `docs/Theory/`

**ADW Commands:**
```bash
adw health                # Check ADW system health
adw status                # Show active workflows
adw workflow list         # List available workflows
```

---

**Last Updated:** 2025-12-03  
**For questions about ADW:** See `docs/Agent/README.md`  
**For questions about particula:** See main `readme.md`
