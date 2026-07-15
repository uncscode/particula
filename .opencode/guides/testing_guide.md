# Testing Guide

**Project:** particula  
**Last Updated:** 2026-07-10

particula uses pytest as its primary testing framework. Tests should be close to
the code they validate and should exercise scientific correctness, edge cases,
and regression behavior.

## Framework

- **pytest:** test discovery and execution.
- **pytest-cov:** coverage reporting.
- **NumPy testing helpers:** numerical comparisons and tolerances.

## File Naming

All test files must use the `*_test.py` suffix.

```text
Correct:
  activity_coefficients_test.py
  coagulation_test.py
  vapor_pressure_test.py

Wrong:
  test_activity_coefficients.py
  activity_coefficients_tests.py
  streamTest.py
```

This pattern matters because pytest discovery, ruff per-file ignores, and agent
tooling all rely on it.

## Test Locations

Place tests in `tests/` subdirectories alongside source modules.

```text
particula/
├── activity/
│   ├── activity_coefficients.py
│   └── tests/
│       └── activity_coefficients_test.py
├── gas/
│   └── tests/
└── particles/
    └── tests/
```

Integration tests live in `particula/integration_tests/`.

## Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=particula --cov-report=term-missing

# Run a module's tests
pytest particula/activity/tests/

# Run a single file
pytest particula/activity/tests/activity_coefficients_test.py

# Run a single test
pytest particula/activity/tests/activity_coefficients_test.py::test_function_name

# Match CI warning behavior
pytest -Werror
```

## Marker Policy

Repository-wide pytest marker registration lives in `particula/conftest.py` and
`pyproject.toml`.

- Registered markers include `slow`, `performance`, `benchmark`, `warp`,
  `cuda`, `gpu_parity`, and `stochastic`.
- Marker registration is descriptive by default. Plain `pytest` preserves
  normal collection behavior unless a test module opts into its own
  `pytest.importorskip("warp")` or similar runtime guard.
- `--benchmark` remains the only collection-affecting pytest option in the
  repository. Benchmark-marked tests are skipped unless you pass that flag.

Use the GPU-oriented markers to describe intent clearly:

- `@pytest.mark.warp`: Warp-dependent or Warp-targeted coverage.
- `@pytest.mark.cuda`: CUDA-specific or CUDA-if-available coverage.
- `@pytest.mark.gpu_parity`: CPU/Warp/CUDA parity validation.
- `@pytest.mark.stochastic`: stochastic or tolerance-band regression coverage.

## Warnings

CI treats warnings as errors with `pytest -Werror`. Tests that pass locally may
fail in CI if they emit `RuntimeWarning`, `DeprecationWarning`, or similar.

Preferred handling order:

1. Fix the underlying warning condition.
2. Use `pytest.warns()` when warning emission is intentional behavior.
3. Use a specific warning filter only when the warning is expected and not the
   subject of the test.

```python
import pytest


def test_expected_warning():
    """Test that the warning is part of the public behavior."""
    with pytest.warns(RuntimeWarning, match="radius values are zero"):
        result = function_that_warns()
    assert result is not None
```

## Scientific Test Patterns

Use `numpy.testing` for numerical comparisons.

```python
import numpy as np
import numpy.testing as npt


def test_physical_property():
    """Test a known physical-property value."""
    temperature = 298.15  # K
    pressure = 101325.0  # Pa

    result = calculate_density(temperature, pressure)
    expected = 1.184

    npt.assert_allclose(result, expected, rtol=1e-3)
```

For conservation laws, compare initial and final totals with an appropriate
tolerance.

## Performance Benchmarks

The staggered condensation benchmark suite is heavy and excluded from normal CI.
Run it manually when changing staggered condensation behavior:

```bash
pytest particula/dynamics/condensation/tests/staggered_performance_test.py -v -m "slow and performance"
```

This suite verifies O(n) scaling at 1k/10k/100k particles, theta-mode behavior,
and deterministic seeded behavior. Staggered stepping uses sequential
Gauss-Seidel updates, so high overhead compared to simultaneous vectorized
stepping is expected.

## Wall Loss Coverage

Wall loss strategy tests should cover both package export paths where relevant:

- `particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py`
- `particula/dynamics/tests/wall_loss_strategies_test.py`

Coverage should include spherical and rectangular geometry, rectangular chamber
dimension validation, supported distribution types, zero concentration, empty
particle-resolved inputs, and parity with helper functions.

## NVIDIA Warp Tests

GPU code must match Python/NumPy reference implementations. Warp CPU is the
default parity backend whenever Warp is installed. CUDA coverage is optional
and local/manual when a CUDA-capable device is available; standard CI must skip
cleanly when CUDA is unavailable. Use the existing markers exactly as
registered: `warp`, `cuda`, `gpu_parity`, and `stochastic`.

### Release-validation command sets

Use focused Warp CPU runs for the default supported validation path. These are
the shipped release-validation commands whenever Warp is installed:

```bash
pytest particula/gpu/tests/cuda_availability_test.py -q
pytest particula/gpu/kernels/tests/environment_test.py -q
pytest particula/gpu/kernels/tests/thermodynamics_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_test.py -q -Werror
pytest particula/gpu/kernels/tests/coagulation_test.py -q -m "warp and gpu_parity"
pytest particula/gpu/kernels/tests/coagulation_test.py -q -m "warp and stochastic"
```

Use CUDA-targeted runs only for optional local/manual validation when a
CUDA-capable device is available. CUDA is additive evidence, not the default
path, and the same Warp-marked modules should continue to collect safely when
CUDA is absent:

```bash
pytest particula/gpu/kernels/tests/environment_test.py -q -m "warp and cuda"
pytest particula/gpu/kernels/tests/condensation_test.py -q -m "warp and cuda" -Werror
```

These commands match the shipped marker and helper contract:

- Warp CPU is the baseline parity backend when Warp is installed.
- CUDA validation is additional local/manual evidence until dedicated CI exists.
- Missing Warp or missing CUDA should produce expected skips, not release
  failures, when a command reaches a guarded suite.
- Warp-marked tests should avoid eager module-level `pytest.importorskip("warp")`
  patterns so marker deselection does not force a collection-time Warp
  dependency.
- Benchmark coverage stays opt-in behind `--benchmark` and remains separate
  from the default release-validation path above.

GPU thermodynamics refresh coverage belongs in
`particula/gpu/kernels/tests/thermodynamics_test.py`. Test explicit
device-resident refresh behavior against CPU vapor-pressure references, cover
constant and canonical Buck modes across the freezing boundary, and verify
invalid inputs leave the caller-owned vapor-pressure buffer unchanged. The
refresh primitive remains a concrete-module API and is not condensation
integration coverage.

### Device-aware tolerance policy

Keep GPU assertions in three separate classes. Deterministic parity,
conservation checks, and stochastic validation each need their own pass
criteria so stochastic expectations never relax conservation assertions or
imply exact replay requirements.

1. **Deterministic parity:** use explicit
   `numpy.testing.assert_allclose(..., rtol=..., atol=...)` bounds for CPU vs
   Warp CPU comparisons and for optional CUDA comparisons when run locally.
   This is the parity rule for deterministic reference agreement.
2. **Conservation checks:** keep mass or count drift tolerances tight and
   assert them separately from parity checks. Do not relax conservation bounds
   just because the surrounding kernel, seeded replay, or diagnostic fixture
   uses stochastic sampling.
3. **Stochastic validation:** compare aggregate behavior across repeated seeds
   or time steps with documented tolerance bands or sigma-based bounds. Use
   those bounded aggregate expectations instead of exact per-seed equality
   across CPU, Warp CPU, or CUDA.

Document the chosen tolerances in the test body or nearby comments when they
are not already obvious from the physics or baseline study.

For GPU Brownian coagulation acceptance work, keep attempted-vs-accepted
collision instrumentation private to
`particula/gpu/kernels/tests/coagulation_test.py`. The shipped
`coagulation_step_gpu(...)` API and production synchronization behavior should
stay unchanged when the work is diagnostic-only.

For the GPU condensation suite, keep shared helpers in support modules only when
discoverable `*_test.py` wrappers expose the runnable cases. The current entry
points are:

- `particula/gpu/kernels/tests/condensation_test.py`
- `particula/gpu/kernels/tests/condensation_stiffness_test.py`

```python
import numpy as np
import numpy.testing as npt
import pytest

wp = pytest.importorskip("warp")


def test_gpu_matches_numpy():
    """Warp computation matches the NumPy reference."""
    expected = numpy_reference(...)
    result = warp_result(...)
    npt.assert_allclose(result, expected, rtol=1e-10, atol=0.0)
```

For conservation checks, keep the assertion separate and tight:

```python
def test_total_mass_is_conserved():
    """Accepted collisions conserve total mass."""
    initial_total = np.sum(initial_mass)
    final_total = np.sum(final_mass)
    npt.assert_allclose(final_total, initial_total, rtol=1e-12, atol=0.0)
```

The public GPU condensation hook has a separate deterministic fp64 regression
pattern. For each box and partitioning species, compare the change in particle
inventory (particle mass weighted by particle concentration) plus the gas
change against zero. Keep that `rtol=1e-12, atol=1e-30` inventory assertion
separate from CPU-oracle particle and gas parity
(`rtol=2e-10, atol=1e-30`). The fixture must also retain disabled-partitioning,
zero-gas, and zero-concentration-slot cases; this is scoped direct-kernel
evidence, not CPU-strategy or runnable parity.

For stochastic kernels, assert bounded aggregate behavior instead of replaying
exact accepted-collision sequences or per-seed trajectories. A seeded range can
be used to gather repeated evidence, but the pass condition should be a
documented aggregate bound such as a tolerance interval or `3-sigma` window
around the expected mean rather than exact seed-by-seed replay.

Use constants from `particula.util.constants`; do not hardcode physical
constants in kernels.

For CPU↔GPU container helpers, add round-trip coverage that checks exact value
and shape preservation on the Warp CPU backend. For `EnvironmentData`, cover
single-box and multi-box cases, the default synchronized path, any supported
manual `sync=False` path, and malformed-schema failures surfaced by CPU-side
validation.

When evaluating experimental GPU integrator candidates, keep prototypes
test-local until they are production-qualified. Prefer deterministic fixed-shape
helpers, caller-owned scratch or buffer reuse, repeated-run equality checks,
finite and non-negative mass assertions, and explicit CPU-reference error bounds
recorded in the test or companion study note. If public runtime behavior is
unchanged, say so directly in the related documentation.

For deterministic GPU baseline studies, prefer explicit `np.float64` fixture
construction over random inputs so later precision comparisons have a stable
reference. The mass-precision baseline suite is the reference pattern:

```bash
pytest particula/gpu/tests/mass_precision_cases_test.py -q
```

Keep those tests warning-clean under `-Werror`, assert canonical
`(n_boxes, n_particles, n_species)` shapes where relevant, and document any
baseline assumptions in the matching roadmap page.

For the mixed NPF/droplet coagulation diagnostic coverage added in
`particula/gpu/kernels/tests/coagulation_test.py`, keep selector-validity and
low-active regressions focused on seeded invariants: accepted pairs stay
sorted, in bounds, and limited to originally active slots; zero/one-active
inputs early-return cleanly; exactly-two-active inputs fall back to the only
valid pair; and accepted collisions conserve total mass. Use focused local runs
such as:

```bash
pytest particula/gpu/kernels/tests/coagulation_test.py -q -k mixed_scale
pytest particula/gpu/kernels/tests/coagulation_test.py -q -k "mixed_scale or sparse or degenerate or conservation" -Werror
```

These checks are intended for seeded regression and warning-clean acceptance
sanity, not for exact CPU/CUDA equality or user-facing feature documentation.

## Test Quality

- Use descriptive test names such as `test_coagulation_conserves_total_mass`.
- Keep tests independent; do not rely on test execution order.
- Use parametrization for related input variants.
- Prefer focused assertions, but include enough checks to validate the behavior.
- Add regression tests when fixing bugs.

## Troubleshooting

- If tests are not discovered, check `*_test.py` naming and run `pytest --collect-only`.
- For the GPU condensation suite, verify collection via
  `pytest particula/gpu/kernels/tests/condensation_test.py --collect-only -q`
  and
  `pytest particula/gpu/kernels/tests/condensation_stiffness_test.py --collect-only -q`.
- If imports fail, install the package in development mode with `pip install -e .[dev]`.
- If coverage looks wrong, run `pytest --cov=particula --cov-report=term`.
- If CI fails but local tests pass, rerun locally with `pytest -Werror`.
- If Warp is not installed, Warp-marked suites may skip through
  `pytest.importorskip("warp")`; treat that as the expected missing-Warp path,
  not a regression.
- If CUDA is unavailable, CUDA-targeted coverage should skip cleanly instead of
  failing CPU-only or CI validation. Some guarded paths use the shared
  `Warp/CUDA not available` message, while others keep more specific skip
  reasons when that context is more useful.
- Use marker selection such as `-m "warp and gpu_parity"`,
  `-m "warp and stochastic"`, or `-m "warp and cuda"` for targeted local GPU
  validation, and keep `pytest particula/gpu/tests/benchmark_test.py
  --benchmark -v -s` separate as opt-in benchmark evidence rather than default
  release validation.
