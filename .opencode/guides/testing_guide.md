# Testing Guide

**Project:** particula  
**Last Updated:** 2026-08-11

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
# Run the full suite with repository-configured coverage
.opencode/tools/run_pytest.py

# Run a module's assertions
pytest particula/activity/tests/

# Run a single file's assertions
pytest particula/activity/tests/activity_coefficients_test.py

# Run a single assertion
pytest particula/activity/tests/activity_coefficients_test.py::test_function_name

```

For local development and agent validation, invoke the repository test runner
without a test target for the canonical full-suite coverage check:

```bash
.opencode/tools/run_pytest.py
```

The runner applies the repository coverage defaults, including the full-package
scope, terminal report, and minimum threshold. Do not pass local
`coverageSource`, `coverageThreshold`, or raw `--cov` controls for comprehensive
validation.

Folder, file, node, marker, and name-filter selections are focused assertion
checks. They do not support coverage evidence in this repository. Run them with
direct `pytest` and do not interpret them as coverage results. After focused
checks pass, run the untargeted repository test runner for comprehensive
coverage.

CI/CD may split tests and coverage into multiple steps for efficiency. Those
pipeline controls are implementation details and do not need to be reproduced
during local validation. A local wrapper's inability to express a CI/CD-only
coverage target does not block validation when the canonical untargeted run is
available.

Local runs should not add `-Werror`; the repository's local test tooling
already applies its configured warning policy.

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

CI/CD may explicitly add `-Werror` to enforce warnings as errors. Local runs
should rely on the configured warning policy instead of passing `-Werror` on
the command line. Test wrappers may intentionally reject that redundant
passthrough flag. Tests that pass locally may still fail in CI if they emit a
`RuntimeWarning`, `DeprecationWarning`, or similar warning outside the local
policy.

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

The full suite collects wall-loss strategy coverage from:

- `particula/dynamics/tests/wall_loss_strategies_test.py`

`particula/dynamics/wall_loss/tests/` is excluded from normal recursive
collection. When changing the concrete wall-loss strategy module, also run its
additional suite directly:

```bash
pytest particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py
```

Keep geometry, distribution-type, empty-input, and helper-parity scenarios in
the adjacent suites rather than restating their detailed cases here.

## NVIDIA Warp Tests

GPU code should be checked against independent Python or NumPy references.
Warp CPU is the required parity baseline whenever Warp is installed. CUDA is
optional local evidence and must skip cleanly when unavailable. Use the
registered `warp`, `cuda`, `gpu_parity`, and `stochastic` markers to describe
test intent; markers do not select a device by themselves.

Keep tests close to the layer they validate:

- `particula/gpu/kernels/tests/`: direct kernel behavior and parity.
- `particula/gpu/properties/tests/`: GPU property helpers.
- `particula/gpu/tests/`: conversions, exports, examples, and process sequences.
- `particula/execution/tests/`: resident sessions, scheduling, communication,
  checkpoints, restart, diagnostics, and integration.
- Adjacent CPU `tests/` directories: independent reference behavior.

Use the maintained examples and their regression tests as the source of truth
for concrete workflows and expected output. Update an example and its test
together instead of copying its detailed contract into this guide:

- Data containers: `docs/Examples/data_containers_and_gpu_foundations.py` and
  `particula/gpu/tests/data_containers_example_test.py`.
- Direct kernels: `docs/Examples/gpu_direct_kernels_quick_start.py` and
  `particula/gpu/tests/gpu_direct_kernels_example_test.py`.
- Direct coagulation: `docs/Examples/gpu_coagulation_direct.py` and
  `particula/gpu/tests/gpu_coagulation_direct_example_test.py`.
- Direct nucleation: `docs/Examples/Nucleation/gpu_direct_nucleation.py` and
  `particula/gpu/tests/gpu_direct_nucleation_example_test.py`.
- Complete direct sequence: `docs/Examples/gpu_complete_process_sequence.py`
  and `particula/gpu/tests/gpu_complete_process_sequence_example_test.py`.
- Resident session: `docs/Examples/gpu_resident_session.py` and
  `particula/execution/tests/gpu_resident_session_docs_test.py`.
- Resident loop: `docs/Examples/gpu_resident_multi_timestep.py` and
  `particula/tests/gpu_resident_multi_timestep_docs_test.py`.

`particula/gpu/tests/process_sequence_test.py` is the maintained reference for
composing direct GPU boundaries without restoring CPU state between calls.

Use focused, coverage-free commands while developing GPU code:

```bash
# Direct GPU code
pytest particula/gpu/ -q

# Resident execution
pytest particula/execution/tests/ -q

# Optional CUDA-only evidence
pytest particula/gpu/ particula/execution/tests/ -q -m "warp and cuda"
```

After focused checks pass, run `.opencode/tools/run_pytest.py` without a target
for repository-wide assertions and coverage. Run `mkdocs build --strict` when
GPU examples or user-facing documentation change.

Warp-dependent modules should skip clearly when Warp or CUDA is missing. Use a
fixture or test-local `pytest.importorskip("warp")` when a module also contains
CPU-only or collection-only checks. A module-level skip is appropriate only
when every test in that module requires Warp.

GPU tests should cover the following behavior where applicable:

- Single-box and multi-box inputs, including empty and inactive state.
- Supported scalar and device-array inputs, shapes, dtypes, and devices.
- Caller-owned array identity and fields that must remain unchanged.
- Exact no-op behavior for zero time, zero work, or disabled operations.
- Read-only preflight rejection before mutation.
- Conservation of mass, concentration, count, or charge as appropriate.
- Persistent RNG state across calls and explicit reset behavior.
- Export boundaries and runnable examples when public imports or docs change.

Use an independent oracle for parity tests. Do not calculate expected values
with the production helper under test. Compare each meaningful output
separately so an aggregate total cannot hide a component error. Keep direct
kernel tests separate from resident scheduler and integration tests.

### Resident execution closeout coverage

When resident lifecycle or communication changes affect diagnostics,
resources, checkpoints, scheduling, or resident communication, retain the
per-target term-missing rows and require the aggregate 80% gate for these
changed-module targets: `diagnostics.py`, `gpu_resources.py`,
`checkpoint.py`, `resident_scheduler.py`, and `resident_communication.py`.

```bash
pytest particula/execution/tests/ -q \
  --cov=particula.execution.diagnostics,particula.execution.gpu_resources,particula.execution.checkpoint,particula.execution.resident_scheduler,particula.execution.resident_communication \
  --cov-report=term-missing --cov-fail-under=80
```

### Release-validation command sets

Run the hardware-free GPU documentation contract test during release
validation, alongside the applicable focused tests and the repository's
untargeted coverage runner:

```bash
pytest particula/tests/gpu_coagulation_docs_test.py -q
```

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

Document chosen tolerances in the test body or nearby comments when they are not
obvious from the physics or a referenced baseline. A seeded range may gather
stochastic evidence, but its pass condition must use the documented aggregate
bound rather than exact seed-by-seed replay.

Prefer explicit `np.float64` fixtures for deterministic GPU baselines. Keep
diagnostic instrumentation test-local, and do not change production APIs solely
to expose test internals. Tests marked `benchmark` must remain opt-in behind
`--benchmark`; use the `slow` and `performance` markers for other expensive
tests.

## Test Quality

- Use descriptive test names such as `test_coagulation_conserves_total_mass`.
- Keep tests independent; do not rely on test execution order.
- Use parametrization for related input variants.
- Prefer focused assertions, but include enough checks to validate the behavior.
- Add regression tests when fixing bugs.

## Troubleshooting

- If tests are not discovered, check `*_test.py` naming and run `pytest --collect-only`.
- If imports fail, install the package in development mode with `pip install -e .[dev]`.
- If coverage looks wrong, rerun `.opencode/tools/run_pytest.py` without a
  focused target and inspect its full-package report.
- If CI fails but local tests pass, inspect the CI warning and compare it with
  the configured local warning policy; do not add `-Werror` to local wrapper
  arguments.
- If Warp is not installed, Warp-dependent tests should skip through their
  fixture, test-local, or module-level guard; treat that as the expected path.
- If CUDA is unavailable, CUDA-targeted tests should skip cleanly instead of
  failing CPU-only validation.
- Use marker selection such as `-m "warp and gpu_parity"`,
  `-m "warp and stochastic"`, or `-m "warp and cuda"` for targeted local GPU
  validation, and keep `pytest particula/gpu/tests/benchmark_test.py
  --benchmark -v -s` separate as opt-in benchmark evidence rather than default
  validation.
