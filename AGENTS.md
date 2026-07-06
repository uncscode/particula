# AGENTS.md - particula Quick Reference

**For ADW (AI Developer Workflow) agents working on the particula repository.**

## Repository Overview

**Name:** particula  
**Description:** A simple, fast, and powerful particle simulator  
**Language:** Python 3.12+  
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

# Slow + performance benchmarks (excluded from CI)
pytest particula/dynamics/condensation/tests/staggered_performance_test.py -v -m "slow and performance"

# Focused deterministic GPU mass-precision baseline tests
pytest particula/gpu/tests/mass_precision_cases_test.py -q

# ADW tools
.opencode/tools/run_pytest.py      # Run tests with validation
.opencode/tools/run_linters.py     # Run linters following CI workflow
```
Performance benchmarks verify O(n) scaling at 1k/10k/100k particles, theta-mode
comparisons (half/random/batch), and deterministic seeds. Note: staggered uses
Gauss-Seidel per-particle loops (sequential), so high overhead vs simultaneous
(vectorized) is expected and not enforced as a target.

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

For deterministic GPU precision-baseline work, use explicit `np.float64`
fixtures and keep reproduction commands focused on the affected module.

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

**Location:** `.opencode/guides/`

**Key guides:**
- `.opencode/guides/testing_guide.md` - Test framework and conventions
- `.opencode/guides/linting_guide.md` - Linting tools and configuration
- `.opencode/guides/code_style.md` - Naming, formatting, type hints
- `.opencode/guides/docstring_guide.md` - Google-style docstring format
- `.opencode/guides/commit_conventions.md` - Commit message format
- `.opencode/guides/pr_conventions.md` - Pull request conventions
- `docs/Features/Roadmap/mass-precision-study.md` - Final GPU
  mass-precision recommendation report, unchanged production baseline,
  candidate study scope, and focused reproduction commands
- `docs/contribute/Feature_Workflow/index.md` - Contributor workflow overview

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

strategy = par.dynamics.ChargedWallLossStrategy(
    wall_eddy_diffusivity=0.001,
    chamber_geometry="spherical",
    chamber_radius=0.5,
    wall_potential=0.05,  # V; image-charge still applies when set to 0
    wall_electric_field=0.0,  # V/m (tuple for rectangular geometry)
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
- `ChargedWallLossStrategy` adds image-charge enhancement (active even when
  `wall_potential` is 0), optional `wall_electric_field` drift, and falls back
  to the neutral coefficient when particle charge and field are zero.
- Use `ChargedWallLossBuilder` with `set_chamber_geometry` plus
  `set_chamber_radius` or `set_chamber_dimensions`, and
  `set_wall_potential`/`set_wall_electric_field`; `WallLossFactory` supports
  `strategy_type="charged"`.
- Supported `distribution_type` values are `"discrete"`, `"continuous_pdf"`,
  and `"particle_resolved"`.

### GPU environment round trips

```python
from particula.gpu import (
    from_warp_environment_data,
    to_warp_environment_data,
)

gpu_environment = to_warp_environment_data(environment, device="cpu")
restored = from_warp_environment_data(gpu_environment)
```

**Key points:**

- `particula.gpu` now exports `WarpEnvironmentData`,
  `to_warp_environment_data`, and `from_warp_environment_data`.
- Environment transfers are explicit helper calls; kernels and runnables do
  not perform hidden CPU↔GPU synchronization for environment state.
- `condensation_step_gpu(..., environment=...)` and
  `coagulation_step_gpu(..., environment=...)` accept scalar
  `temperature`/`pressure`, direct per-box Warp arrays shaped `(n_boxes,)`,
  hybrid scalar-plus-Warp-array direct inputs when `environment` is omitted,
  or explicit `WarpEnvironmentData` on the active device.
- Mixing scalar or Warp-array `temperature`/`pressure` with `environment=`
  still raises an early `ValueError`.
- Explicit environment and direct Warp-array inputs must use `(n_boxes,)`
  temperature and pressure arrays on the same device as the particle/gas data.
- Accepted direct/environment `temperature` and `pressure` inputs, plus direct
  coagulation `volume` inputs, are validated as positive finite physical
  values before launch.
- `EnvironmentData.temperature` and `pressure` use `(n_boxes,)`;
  `saturation_ratio` uses `(n_boxes, n_species)` on both CPU and Warp mirrors.
- Round trips preserve `temperature`, `pressure`, and `saturation_ratio` for
  CPU-backed tests and parity coverage now also runs on `cuda` when available.
- `from_warp_environment_data(..., sync=False)` assumes manual Warp
  synchronization before `.numpy()` access.

### GPU gas round trips

```python
import numpy as np

from particula.gpu import from_warp_gas_data, to_warp_gas_data

vapor_pressure = np.array([[2330.0, 120.0]])  # (1, n_species)
gpu_gas = to_warp_gas_data(
    gas_data,
    device="cpu",
    vapor_pressure=vapor_pressure,
)
restored = from_warp_gas_data(gpu_gas, name=gas_data.name)
```

**Key points:**

- `GasData.name` is CPU-owned ordered metadata. `WarpGasData` does not store
  names, so callers must preserve ordered names externally for semantic
  restores.
- Omitting `name` or passing `name=None` to `from_warp_gas_data()` restores
  placeholders such as `species_0`.
- `GasData.concentration` and GPU `vapor_pressure` use
  `(n_boxes, n_species)`, including `(1, n_species)` for single-box examples.
- `GasData.partitioning` is `bool` on CPU and `int32` on GPU, with explicit
  `bool → int32 → bool` conversion across the helper boundary.
- `WarpGasData.vapor_pressure` is GPU-only helper state. Pass it explicitly
  when needed, otherwise `to_warp_gas_data()` allocates zeros with the same
  shape, and `from_warp_gas_data()` always drops it on CPU restore.
- See `docs/Features/particle-data-migration.md` for the user-facing field
  authority table and `particula/gpu/tests/conversion_test.py` for the
  regression-backed contract.

### GPU mass-precision baseline study

```bash
pytest particula/gpu/tests/mass_precision_cases_test.py -q
pytest particula/gpu/tests/mass_precision_metrics_test.py -q
pytest particula/gpu/tests/benchmark_test.py --benchmark -k mass_precision -v -s
```

**Key points:**

- The deterministic baseline lives in
  `particula/gpu/tests/mass_precision_cases_test.py`.
- Baseline cases cover NPF, 5-10 nm, accumulation-mode, and droplet-scale
  masses without changing the production CPU/GPU storage schema.
- The focused comparison module lives in
  `particula/gpu/tests/mass_precision_metrics_test.py` and evaluates exactly
  three study-only reconstruction candidates against the fp64 baseline,
  including cached reconstruction, conservation, mixed-scale small-particle,
  and clamp-accounting checks.
- Mixed-scale review thresholds are now exercised explicitly so nanometer
  particles cannot be masked by droplet-scale aggregates in the same array.
- Optional throughput evidence lives on
  `particula/gpu/tests/benchmark_test.py` behind the existing `--benchmark`
  opt-in and still skips cleanly when Warp/CUDA is unavailable.
- Use `docs/Features/Roadmap/mass-precision-study.md` as the reference for
  case names, supported-vs-unsupported candidate scope, thresholds,
  memory-footprint examples, and focused reproduction.

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
- `.opencode/tools/run_pytest.py` - Test runner with validation
- `.opencode/tools/run_linters.py` - Linter runner following CI workflow

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
.opencode/tools/run_linters.py && .opencode/tools/run_pytest.py
```

## Notebook Editing (Jupytext Paired Sync)

Notebooks in `docs/Examples/` use Jupytext paired sync. Edit `.py` files, not `.ipynb`:

```bash
# 1. Edit the .py file (percent format with # %% cell markers)
# 2. Lint the .py file
ruff check docs/Examples/path/to/file.py --fix
ruff format docs/Examples/path/to/file.py

# 3. Sync to update .ipynb
python3 .opencode/tools/validate_notebook.py docs/Examples/path/to/file.ipynb --sync

# 4. Execute to validate and generate outputs
python3 .opencode/tools/run_notebook.py docs/Examples/path/to/file.ipynb

# 5. Commit both files (.py and .ipynb)
```

**Key points:**
- Always edit `.py` files, not `.ipynb` directly
- Lint before sync to catch syntax errors
- Execute after sync to validate code and generate website outputs
- Commit both files to keep them paired

**Full documentation:** `.opencode/guides/documentation_guide.md` (Jupytext Paired Sync Workflow section)

## Getting Help

**Documentation:**
- Full guides: `.opencode/guides/`
- Examples: `docs/Examples/`
- Theory: `docs/Theory/`

**ADW Commands:**
```bash
adw health                # Check ADW system health
adw status                # Show active workflows
adw workflow list         # List available workflows
```

---

**Last Updated:** 2026-07-05  
**For questions about ADW:** See `.opencode/guides/README.md`  
**For questions about particula:** See main `readme.md`
