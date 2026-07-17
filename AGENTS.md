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
**Minimum tests:** 500 (current: 713)  
**Test pattern:** `*_test.py`

**Registered markers:** `slow`, `performance`, `benchmark`, `warp`, `cuda`,
`gpu_parity`, `stochastic`  
**Collection policy:** plain `pytest` preserves normal collection;
`--benchmark` is the only collection-affecting option.

**GPU policy:** Warp CPU is the default parity backend when Warp is installed.
CUDA coverage is optional local/manual validation and must skip cleanly when
unavailable. Deterministic parity uses explicit `rtol`/`atol`, conservation
checks stay tight, and stochastic checks use aggregate statistics or
sigma-based bounds rather than exact per-seed equality.

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
- Import direct GPU step entry points from `particula.gpu.kernels`, not from
  top-level `particula.gpu`.
- The canonical runnable example for the supported low-level path is
  `docs/Examples/gpu_direct_kernels_quick_start.py`; it keeps CPU↔Warp
  transfers explicit, defaults to Warp `device="cpu"`, and defers
  `particula.gpu.kernels` imports until the Warp-enabled execution branch.
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
- Import the low-level step with
  `from particula.gpu.kernels import condensation_step_gpu`. The optional
  `CondensationScratchBuffers` sidecar is concrete-module-only at
  `particula.gpu.kernels.condensation`; it is not a second step entry point.
- `condensation_step_gpu(...)` requires keyword-only
  `thermodynamics=ThermodynamicsConfig`. Each successful call executes exactly
  four equal `time_step / 4.0` substeps. In each substep, it optionally refreshes
  composition-weighted surface tension, overwrites caller-owned
   `WarpGasData.vapor_pressure` from normalized current per-box temperature,
   refreshes environment properties, produces a raw proposal, then applies its
   P2-finalized, inventory-limited transfer. Vapor pressure is derived step
   state, not a caller-supplied physics source.
- The total-transfer buffer is cleared once after preflight, accumulates applied
  P2-finalized transfers, and is returned by identity when supplied. Work
  storage retains only the final raw proposal. Particle masses mutate in place,
  and each finalized transfer applies the matching particle-concentration-
  weighted delta to `gas.concentration`.
- Supplied `CondensationScratchBuffers` fields must be active-device,
  stable-shape `wp.float64`: transfer fields use
  `(n_boxes, n_particles, n_species)` and property fields use `(n_boxes,)`.
  Fields may be omitted independently, which uses fallback allocations.
- An optional active-device `wp.float64` `latent_heat` sidecar shaped
  `(n_species,)` applies a per-substep rate correction. Omitting it, or using
  a zero entry for a species, preserves that species' isothermal rate path. An
   optional caller-owned active-device `wp.float64` `energy_transfer` output
   shaped `(n_boxes, n_species)` requires `latent_heat`; after successful
   preflight it records signed whole-call P2-finalized transfer times latent
   heat and is not returned as a third tuple item. `thermal_work` has
   validated per-species shape `(n_species,)` but remains deferred, unused
   state.
- This direct kernel step does not establish CPU-strategy parity, a `Runnable`
  API, adaptive stepping, graph capture/replay, autodiff, or general accuracy
  claims. It executes exactly four equal substeps; vapor-pressure refresh does
  not read gas concentration, while each later mass-transfer proposal reads
  gas concentration coupled by its predecessors. See
  `docs/Features/condensation_strategy_system.md` and
  `docs/Features/Roadmap/condensation-stiffness-study.md` for its bounded
  contract and case-specific evidence.
- The deterministic fp64 direct-kernel parity matrix runs one-box and
  multi-box/multi-species fixtures against an independent NumPy fixed-four-
  substep, P2 inventory-finalized, gas-coupled oracle. It separately compares
  particle masses and gas concentrations, and covers uptake, evaporation,
  disabled partitioning, latent heat, zero gas, and inactive particle slots.
  Warp CPU is required when Warp is installed; CUDA is optional and skips
  cleanly when unavailable. Separate per-box/per-species particle-plus-gas
  inventory coverage uses `rtol=1e-12, atol=1e-30`. This is direct-kernel
  evidence only.
- Scalar temperatures, direct Warp temperature arrays, and
  `WarpEnvironmentData` all drive this refresh. Non-`wp.float64` temperature
  arrays are cast into a device-local float64 buffer; no host vapor-pressure
  evaluation or host-to-device vapor-pressure transfer occurs in the step.
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
- See `docs/Features/data-containers-and-gpu-foundations.md` for the canonical
  user-facing field authority and transfer-boundary contract,
  `docs/Features/particle-data-migration.md` for migration walkthroughs, and
  `particula/gpu/tests/conversion_test.py` for the regression-backed contract.

### GPU vapor-pressure refresh

- `refresh_vapor_pressure_gpu` is a concrete-module API: import it from
  `particula.gpu.kernels.thermodynamics`, not `particula.gpu.kernels`.
- It explicitly overwrites device-resident `WarpGasData.vapor_pressure` with
  shape `(n_boxes, n_species)` from a validated `ThermodynamicsConfig` and a
  device-local `wp.float64` temperature array in K with shape `(n_boxes,)`.
- Constant mode reads `parameters[:, 0]` as vapor pressure in Pa. Canonical
  Buck mode uses fixed water/ice equations; its four parameter slots are
  reserved and ignored.
- `condensation_step_gpu(..., thermodynamics=config)` invokes this primitive
  after successful entry-point validation and before condensation mass
  transfer. Call it directly only when vapor pressure must be refreshed outside
  a condensation step.

### GPU coagulation RNG state ownership

```python
import warp as wp

rng_states = wp.zeros((n_boxes,), dtype=wp.uint32, device=device)

# Seed once before a repeated-step loop or before graph capture.
coagulation_step_gpu(..., rng_seed=41, rng_states=rng_states,
                     initialize_rng=True)

for _ in range(n_steps):
    coagulation_step_gpu(..., rng_seed=41, rng_states=rng_states)
```

**Key points:**

- `coagulation_step_gpu(..., mechanism_config=...)` accepts a keyword-only
  `CoagulationMechanismConfig` imported from
  `particula.gpu.kernels.coagulation`; configuration APIs are deliberately not
  re-exported through `particula.gpu.kernels`.
- Omitting `mechanism_config` preserves Brownian, particle-resolved execution.
  This is the only executable mechanism/distribution combination. Malformed
  configuration, unsupported distributions, and reserved mechanisms fail during
  host-side preflight before runtime state access, allocation, mutation, RNG
  initialization, or kernel launch.
- Before Brownian execution, `particles.charge` must be a finite `wp.float64`
  Warp array with shape `(n_boxes, n_particles)` on the particle-data device.
  Non-finite charge is detected by one read-only device scan and private-status
  readback; failure occurs before environment or volume normalization,
  caller-output validation or allocation, RNG setup, and Brownian/apply work.
  Accepted merge application adds donor charge to the recipient and clears the
  donor. This bookkeeping does not enable charge-dependent selection or rates.
- Shipped baseline: caller-owned persistent `rng_states` are seeded once and
  then reused across repeated calls.
- Omitting `rng_states` keeps the convenience allocate-and-seed-per-call path.
- Reusing a fixed `rng_seed` with a persistent `rng_states` buffer does not
  trigger hidden reseeding; call with `initialize_rng=True` only when you
  intentionally want to reset the stream.
- For graph capture or repeated-step loops, initialize or reset persistent
  `rng_states` before capture or before entering the loop.
- Coagulation `rng_states` are Warp-resident sidecar state, not fields on
  `ParticleData`, `GasData`, `EnvironmentData`, or their Warp mirrors.
- Mixed NPF/droplet Brownian acceptance diagnostics live only in
  `particula/gpu/kernels/tests/coagulation_test.py`; keep attempted-count
  instrumentation test-local and leave the public `coagulation_step_gpu(...)`
  API unchanged for debug-only coverage.
- The bounded Brownian selector hardening is regression-covered only through
  test-local diagnostics: accepted pairs must remain sorted, in bounds, and
  drawn from originally active slots; zero/one-active inputs must early-return;
  exactly-two-active inputs must collapse to the sole valid pair; and accepted
  collisions must conserve total mass.
- See `docs/Features/data-containers-and-gpu-foundations.md` and
  `docs/Features/Roadmap/data-oriented-gpu.md` for the user-facing ownership
  and graph-capture guidance.

Focused diagnostic runs:

```bash
pytest particula/gpu/kernels/tests/coagulation_test.py -q -k mixed_scale
pytest particula/gpu/kernels/tests/coagulation_test.py -q -k "mixed_scale or sparse or degenerate or conservation" -Werror
```

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

## Example Editing

`docs/Examples/` may contain runnable `.py` entrypoints, paired notebooks, or
both. For notebook-backed examples, use Jupytext paired sync and edit `.py`
files, not `.ipynb`:

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
- Runnable script examples may be standalone `.py` files without notebooks
- Always edit `.py` files, not `.ipynb` directly, when a notebook pair exists
- Lint before sync to catch syntax errors
- Execute after sync to validate code and generate website outputs
- Commit both files to keep them paired when the example includes a notebook

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

**Last Updated:** 2026-07-14  
**For questions about ADW:** See `.opencode/guides/README.md`  
**For questions about particula:** See main `readme.md`
