# Testing Strategy

Every production phase ships its tests in the same change. Test coverage
thresholds must never be lowered; changed code must meet at least the configured
80% threshold. Tests use the `*_test.py` convention and run without Warp/CUDA.

## Per-Phase Coverage

- **P1 (complete, issue #1389):** `particula/dynamics/tests/dilution_test.py`
  covers the three helper equations; scalar and broadcast-array return shapes;
  exact no-ops; elementwise invalid-domain, `None`, unsupported-type, and
  incompatible-shape rejection; warning-clean extreme finite decay; input
  non-mutation; and the concrete-module-only package boundary for
  `get_dilution_step()`. The focused command is
  `pytest particula/dynamics/tests/dilution_test.py -q -Werror`.
- **P2 (complete, issue #1390):**
  `particula/dynamics/tests/dilution_test.py` covers direct container dilution
  for physical particle concentration and scalar/multi-species atmosphere gas
  groups, identity and protected-state retention, exact zero-input no-ops, and
  finite extreme-decay underflow. It also covers zero-dimensional NumPy scalar
  inputs; typed scalar-boundary errors; invalid particle, each gas-group, and
  converted-storage preflight failures without writes; and rollback/recovery
  after injected commit failures. Package-surface coverage confirms
  `dilute_aerosol` remains concrete-module-only.
- **P3 (complete, issue #1391):**
  `particula/dynamics/tests/dilution_runnable_test.py` covers strategy
  coefficient validation and rate delegation; P2 step validation propagation;
  direct strategy and substepped runnable exponential decay for particle,
  partitioning-gas, and gas-only concentrations; exact no-ops; finite,
  nonnegative large-decay results; and concrete-module-only import boundaries.
  Spy strategies verify equal slice count, invalid substep/time failure before
  strategy calls or mutation, preservation of original-aerosol identity despite
  a nonconforming strategy return, and `|`/`RunnableSequence` ordering.
- **P4 (complete, issue #1392):**
  `particula/dynamics/tests/dilution_test.py` extends direct coverage with
  malformed physical particle and both gas-group sources, particle backing
  storage, volume, and candidate-shape failures before setters run, including
  zero-duration/zero-coefficient ordering and rollback snapshots.
  `particula/dynamics/tests/dilution_runnable_test.py` verifies concrete
  supported-runnable preflight before the first substep while preserving valid
  custom strategy equal-substep delegation. The new
  `particula/dynamics/tests/dilution_exports_test.py` verifies direct dynamics
  imports, symbol identity, dynamic `__all__`, and `par.dynamics`
  construction/execution. Concrete helpers remain absent from that public API.
- **P5:** Execute the CPU example in the docs test path, validate links, and run
  all focused dilution tests.

## Acceptance Matrix

| Case | Particle expectation | Gas expectation | Protected state |
|---|---|---|---|
| Positive coefficient/time | Matches canonical CPU formula | Same fractional semantics | Unchanged |
| Zero flow/coefficient | Exact no-op | Exact no-op | Unchanged |
| Zero time | Exact no-op | Exact no-op | Unchanged |
| Zero concentration | Remains exactly zero | Remains exactly zero | Unchanged |
| Scalar/multi-species gas | Shapes preserved | Elementwise reference match | Metadata unchanged |
| Multiple substeps | Matches documented integrator | Same update count | Identity retained |
| Invalid input/state | No writes | No writes | Complete snapshot equal |

Use explicit tolerances appropriate to float64 calculations and exact equality
for no-op and protected-field assertions. Include regression tests for existing
helper imports and outputs so introducing the strategy cannot break users.

## Verification Commands

```bash
pytest particula/dynamics/tests/dilution_test.py -q
pytest particula/dynamics/tests/dilution_runnable_test.py -q
pytest particula/dynamics/tests/dilution_exports_test.py -q
ruff check particula/dynamics/ --fix
ruff format particula/dynamics/
ruff check particula/dynamics/
mypy particula/dynamics/ --ignore-missing-imports
```
