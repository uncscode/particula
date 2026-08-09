# Testing Guide

**Project:** particula  
**Last Updated:** 2026-08-09

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

```

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

Use focused Warp runs for the default supported validation path. Warp CPU is
the required baseline whenever Warp is installed; the deterministic focused
command also exercises CUDA when it is available. These are the shipped
release-validation commands:

```bash
pytest particula/gpu/tests/cuda_availability_test.py -q
pytest particula/gpu/kernels/tests/environment_test.py -q
pytest particula/gpu/kernels/tests/thermodynamics_test.py -q
pytest particula/gpu/kernels/tests/dilution_test.py -q
pytest particula/gpu/kernels/tests/exhaustion_test.py -q
pytest particula/gpu/kernels/tests/nucleation_test.py -q
pytest particula/gpu/kernels/tests/slot_management_test.py -q
pytest particula/gpu/kernels/tests/wall_loss_test.py particula/gpu/kernels/tests/wall_loss_parity_test.py -q
pytest particula/gpu/kernels/tests/condensation_test.py -q
pytest particula/gpu/kernels/tests/coagulation_validation_test.py -q -m "warp and gpu_parity"
pytest particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py -q -m "warp and stochastic and not cuda"
pytest particula/gpu/kernels/tests/coagulation_test.py -q
# Validate private resident composition of shipped direct GPU boundaries.
pytest particula/gpu/tests/process_sequence_test.py -q
# Validate the published direct-GPU coagulation example and documentation.
pytest particula/gpu/tests/gpu_coagulation_direct_example_test.py -q
# Validate the explicit-transfer complete-process example.
pytest particula/gpu/tests/gpu_complete_process_sequence_example_test.py -q
# Validate documentation links and the closeout projection without Warp or CUDA.
pytest particula/tests/gpu_coagulation_docs_test.py -q
# Validate the E6 complete-process publication and blocked closeout projection.
pytest particula/tests/gpu_complete_process_sequence_docs_test.py -q
```

Use CUDA-targeted runs only for optional local/manual validation when a
CUDA-capable device is available. CUDA is additive evidence, not the default
path, and the same Warp-marked modules should continue to collect safely when
CUDA is absent:

```bash
pytest particula/gpu/kernels/tests/environment_test.py -q -m "warp and cuda"
pytest particula/gpu/kernels/tests/condensation_test.py -q -m "warp and cuda"
pytest particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py -q -m "warp and cuda"
```

For the full fixed-mask contract, see the
[GPU coagulation validation record](../../docs/Features/Roadmap/coagulation-validation.md).
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
- Marker selection describes test intent; it does not select a device. In
  particular, deterministic P2 public-step checks enumerate available Warp
  devices, so they exercise CUDA when it is available as well as the Warp CPU
  baseline.

`particula/gpu/tests/process_sequence_test.py` provides private, test-only
resident composition coverage for the existing exported direct condensation,
coagulation, dilution, wall-loss, and nucleation boundaries. It keeps the same
caller-owned Warp containers and sidecars resident between calls, checks
identity, accounting, no-op/preflight, and RNG-sidecar contracts, and guards
against intermediate CPU restoration. This is contributor test evidence only:
it does not create a production coordinator, public integration API, hidden
transfer, CPU fallback, or runnable. Run its CUDA-marked row separately when
available:

```bash
pytest particula/gpu/tests/process_sequence_test.py -q \
  -m "warp and cuda"
```

### Resident communication and checkpoint coverage

Resident communication is concrete-only execution coverage, not a public API or
an example workflow. Keep its tests under `particula/execution/tests/` and cover
both closed-map GAS and PARTICLES resource families, optional volume evolution,
and the canonical twelve-node schedule. Verify that normal-step metadata checks
do not repeat P1 payload validation, allocate, transfer, inspect host payloads,
or synchronize. Cover prelaunch rejection as non-mutating; after either barrier
writer launches, assert lifecycle fault/guard close behavior rather than
rollback.

Checkpoint tests must preserve schema-v1 noncommunication restart support and
verify schema-v2 checkpoints with no communication family or one complete
closed-map family. Restarts must require an exact device and recreate arrays and
bindings instead of reusing source identities.

Resident Brownian-coagulation and wall-loss coverage must keep each pinned RNG
sidecar by identity across valid dispatches and verify that normal dispatch
never reseeds or allocates it. Test immutable stream metadata during session
setup and first resource acquisition separately. When either resident stream
has been published, checkpoint and finalization must reject before payload
work; stream checkpoint continuation is intentionally unsupported. Wall-loss
coverage must also verify reset-like values reject before direct-kernel
resolution and that an empty resolved launch set is a no-op. Do not add reset
or inspection APIs, hidden transfers, or synchronization.

```bash
pytest particula/execution/tests/gpu_session_test.py \
  particula/execution/tests/gpu_resources_test.py \
  particula/execution/tests/coagulation_adapter_test.py \
  particula/execution/tests/coagulation_integration_test.py \
  particula/execution/tests/checkpoint_test.py \
  particula/execution/tests/process_graph_test.py \
  particula/execution/tests/resident_communication_test.py \
  particula/execution/tests/scheduler_test.py -q
```

`docs/Examples/gpu_complete_process_sequence.py` is a standalone, direct-Warp
example with a focused regression suite in
`particula/gpu/tests/gpu_complete_process_sequence_example_test.py`. The suite
checks the deterministic no-Warp path without requiring Warp, then checks the
enabled path's one explicit conversion of each CPU container, five-call order,
one synchronization, and one final restore. It also verifies that direct errors
propagate without an intermediate restore or CPU fallback. This example is
documentation for explicit caller-owned transfers and sidecars; it does not
create a scheduler, high-level runnable, backend selector, or integration API.

The coagulation validation matrix supports exactly the singleton masks `1`,
`2`, `4`, and `8`; two-way masks `3`, `5`, `6`, `9`, `10`, and `12`; and
four-way mask `15`. The three-way masks `7`, `11`, `13`, and `14` remain
deferred/fail closed. P1 uses `rtol=1e-7, atol=0` for Brownian pair-rate
comparisons. Brownian property and selector-majorant checks, along with other
applicable positive/additive rate, property, and majorant comparisons, use
`rtol=1e-6, atol=0`; physical zeros are exact. P2 keeps concentration-weighted,
per-box/per-species inventory at `rtol=1e-12, atol=1e-30` and separates that
ownership evidence from stochastic acceptance.

The P3 coagulation stochastic-validation matrix uses 100 fresh seeds for every
executable row and device. It compares aggregate accepted collisions with an
independent initial-state expectation using `3 * sqrt(expected_mean)`. This
bound is neither conservation evidence nor an exact accepted-pair, seed, RNG,
CPU/Warp, or CUDA replay requirement.

The record is limited to the existing direct path: it makes no mandatory-CUDA,
production-API, new-physics, performance, CPU-fallback, runnable, graph-capture,
autodiff, or adaptive-stepping conclusion.

GPU thermodynamics refresh coverage belongs in
`particula/gpu/kernels/tests/thermodynamics_test.py`. Test explicit
device-resident refresh behavior against CPU vapor-pressure references, cover
constant and canonical Buck modes across the freezing boundary, and verify
invalid inputs leave the caller-owned vapor-pressure buffer unchanged. The
refresh primitive remains a concrete-module API and is not condensation
integration coverage.

GPU dilution P2--P4 coverage belongs in
`particula/gpu/kernels/tests/dilution_test.py`. Mark it `warp`, defer Warp
imports so missing Warp skips cleanly, and import `dilution_step_gpu` from
`particula.gpu.kernels`. Cover scalar and metadata-valid per-box coefficient
forms, identity return, protected-field preservation, write-free scalar-zero
and zero-time paths, and the independent finite-step oracle
`c_new = c * exp(-alpha * time_step)` for particle and gas concentrations.
P4 runs a deterministic float64 one-/multi-box matrix against that NumPy
oracle on Warp CPU, with matching CUDA rows that skip cleanly when unavailable.
Keep particle and gas parity assertions separate, preserve caller-owned per-box
coefficient identity and values, and use exact equality for no-op checks. This
is direct-kernel test evidence only; it does not establish CPU-runnable parity.

GPU resampling coverage belongs in
`particula/gpu/kernels/tests/exhaustion_test.py`. Mark device cases `warp` and
`gpu_parity`, defer Warp imports, and import only `resampling_step_gpu` from
`particula.gpu.kernels`; `ResamplingBuffers` remains concrete-module-only at
`particula.gpu.kernels.exhaustion`. Test against an independent NumPy P2
equal-weight remapping oracle, including stable source ordering, original-slot
retention/release, diagnostics, and separate tight inventory conservation.
Preflight failures must preserve every caller array. Planning failures may alter
documented buffer lanes but must skip commit and preserve particles; all-zero
demand and empty-box calls must be exact write-free no-ops. Warp CPU is the
baseline and CUDA is optional with clean skips. Keep scaling evidence marked
`slow`, `performance`, and `benchmark`, behind `--benchmark`; it must not imply
a CPU fallback, resizing, policy-resolution, runnable, or broad performance
claim.

Private GPU nucleation P2 coverage belongs in
`particula/gpu/kernels/tests/nucleation_test.py`. Test the concrete module seam
only; do not add a package export, runnable, or user-facing example. Use an
independent NumPy float64 oracle for survival-included activation or kinetic
rate, `E_pot = J * dt`, common inventory admission, removal, and gate-code
precedence. Assert that successful planning changes only its documented
caller-owned planning, finalized-demand, and diagnostic sidecars; particle and
gas fields, plus P3-owned sidecar lanes, must remain unchanged. Exercise Warp
CPU when installed and optional CUDA with clean skips. Keep invalid-preflight
and derived-demand failures non-mutating, and keep this evidence separate from
future activation or transaction behavior.

GPU slot activation P4 coverage belongs in
`particula/gpu/kernels/tests/slot_management_test.py`. Defer Warp imports so
missing Warp skips cleanly, import only `activate_slots_gpu` from
`particula.gpu.kernels`, and retain `get_slot_diagnostics_gpu` as a
concrete-module-only P3 test helper. Compare deterministic float64 copies and
all caller-owned int32 sidecars exactly against independent CPU activation and
post-call diagnostics. Cover ascending-free-slot mapping, selected-prefix-only
validation, zero prefixes, zero boxes, zero capacity, exact capacity, repeated
activation, and sparse capacity. Reusable fixed-slot coverage should keep an
independent CPU oracle and separately assert exact values, shapes, dtypes,
devices, and sidecar identity. Include selected-prefix coverage and snapshot
accessible particle, request, and output arrays for preflight rejection.
Warp CPU is the baseline; CUDA parity is optional and must skip cleanly when
unavailable. The complementary CPU contract evidence is:

```bash
pytest particula/particles/tests/slot_management_test.py -q
```

This tests the bounded direct-Warp boundary only, not a runnable, implicit
transfer, storage resizing, or rollback after writer launch.

Direct GPU wall-loss parity coverage belongs in
`particula/gpu/kernels/tests/wall_loss_parity_test.py`. Keep the NumPy
CPU/system-state oracle independent of GPU helpers and compare complete
particle-resolved slots with the CPU system-state wall-loss functions. The Warp
diagnostic intentionally uses approved production GPU property helpers to mirror
the supported GPU calculation path; it is not independently derived. Cover
spherical and rectangular geometries, one- and multi-box inputs, per-box
environment state, particle scales, and inactive or unusable slots. Charged
coverage must use an independent charged CPU oracle, retain exact neutral
fallback checks for zero-charge slots, and preserve caller-owned rectangular
field storage. Keep deterministic coefficient agreement separate from
stochastic evidence. For stochastic removal, use aggregate survival counts
across repeated fresh seeds and a documented binomial bound; do not require
CPU/Warp RNG-stream, per-seed, or trajectory replay. Include distinct exact
no-op and invalid-preflight non-mutation checks for zero time and all-inactive
inputs, including caller-owned RNG state. This suite validates the existing
direct-kernel boundary only and must not imply a new public API or physics
capability.

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

For GPU coagulation acceptance work, including charged-hard-sphere-only and
canonical Brownian-plus-charged coverage, keep attempted-vs-accepted collision
instrumentation private to `particula/gpu/kernels/tests/coagulation_test.py`.
The shipped `coagulation_step_gpu(...)` API and production synchronization
behavior should stay unchanged when the work is diagnostic-only.

The direct ST1956 turbulent-shear singleton uses explicit positive finite
`turbulent_dissipation` (m²/s³) and `fluid_density` (kg/m³) P2 inputs. Cover
scalar and active-device `wp.float64` `(n_boxes,)` inputs. Executable masks are
the singletons `1`, `2`, `4`, and `8`; two-term masks `3`, `5`, `6`, `9`, `10`,
and `12`; and the four-term mask `15`. The three-term masks are deferred:
mask `7` rejects at capability preflight before particle metadata or enabled-term
validation, while masks `11`, `13`, and `14` validate particle metadata and all
enabled terms (including turbulent P2 inputs where enabled) before raising
`ValueError("Additive coagulation execution is deferred.")`. Deferred errors
occur before downstream normalization, output/RNG work, executable launches, or
mutation. Test turbulent, charged, and sedimentation validation according to the
enabled mechanism bits, while proving non-turbulent masks ignore turbulent
arguments and caller-owned particle, collision-output, and persistent-RNG state
is unchanged after preflight errors. Test the turbulent singleton's O(A)
two-largest-active-radii majorant against an independent NumPy oracle. For every
executable additive mask, public-path multi-pair tests must independently sum
enabled component rates and use finite, discriminating timesteps so omission of
one contribution changes bounded selection outcomes. Keep mass conservation
separate from stochastic acceptance checks and use aggregate tolerance or sigma
bounds rather than exact seeded pair replay.

For the GPU condensation suite, keep shared helpers in support modules only when
discoverable `*_test.py` wrappers expose the runnable cases. The current entry
points are:

- `particula/gpu/kernels/tests/condensation_test.py`
- `particula/gpu/kernels/tests/condensation_stiffness_test.py`
- `particula/gpu/kernels/tests/condensation_graph_capture_test.py`
- `particula/gpu/kernels/tests/condensation_autodiff_test.py`

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

The direct GPU condensation hook has deterministic fp64 parity-matrix coverage
in `particula/gpu/kernels/tests/condensation_test.py`. It runs one-box and
multi-box/multi-species fixtures against an independent NumPy fixed-four-
substep, P2 inventory-finalized, gas-coupled oracle. Compare final particle
masses and gas concentrations independently, with explicit tolerance bounds;
do not replace these checks with an aggregate inventory assertion. The matrix
covers uptake, evaporation, disabled partitioning, latent heat, zero gas, and
inactive particle slots. Warp CPU is required whenever Warp is installed;
the matching CUDA matrix is optional and must skip cleanly when unavailable.
This is scoped direct-kernel evidence, not CPU-strategy or runnable parity.

The public GPU condensation hook also has separate deterministic per-box,
per-species inventory regression coverage. Compare the change in particle
inventory (particle mass weighted by particle concentration) plus the gas
change against zero at `rtol=1e-12, atol=1e-30`. Keep this conservation
assertion separate from CPU-oracle particle and gas parity.

For direct-condensation P1--P4 coverage, keep parity, conservation, capture,
and derivative assertions separate. Warp CPU is the baseline for supported P1
parity/P2 conservation and bounded P4 raw-rate autodiff probes when Warp is
installed; CUDA is optional local/manual evidence. P3 is the exception: CPU
graph capture is capability-skipped, and only the CUDA public-step host
validation readback within capture is a strict expected failure because it is
not capture-safe. Setup and normal calls remain ordinary assertions. These
precise guarded skips/xfails document an unsupported capture capability, not a
CPU-baseline or replay-support claim. The P4 wrapper covers only an interior,
out-of-place raw-rate Tape derivative; it does not relax P2 boundary or
in-place-mutation limits.

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

Keep those tests warning-clean under the configured policy, assert canonical
`(n_boxes, n_particles, n_species)` shapes where relevant, and document any
baseline assumptions in the matching roadmap page.

For the mixed NPF/droplet coagulation diagnostic coverage added in
`particula/gpu/kernels/tests/coagulation_test.py`, keep selector-validity and
low-active regressions focused on seeded invariants: accepted pairs stay
sorted, in bounds, and limited to originally active slots; zero/one-active
inputs early-return cleanly; exactly-two-active inputs fall back to the only
valid pair; and accepted collisions conserve total mass. For exact charged-only
coverage, also assert signed-charge conservation and test aggregate stochastic
behavior against an independent oracle rather than exact pair replay. Use
focused local runs such as:

```bash
pytest particula/gpu/kernels/tests/coagulation_test.py -q -k mixed_scale
pytest particula/gpu/kernels/tests/coagulation_test.py -q -k "mixed_scale or sparse or degenerate or conservation"
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
- If CI fails but local tests pass, inspect the CI warning and compare it with
  the configured local warning policy; do not add `-Werror` to local wrapper
  arguments.
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
