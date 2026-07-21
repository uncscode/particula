# Testing Strategy

Every production phase ships its tests in the same change. Test coverage
thresholds must never be lowered; changed code must meet at least the configured
80% threshold. Tests use the `*_test.py` convention and run without Warp/CUDA.

## Per-Phase Coverage

- **P1:** Extend `particula/dynamics/tests/dilution_test.py` with scalar and
  NumPy-array equations, broadcasting, exact zero-flow/zero-concentration
  behavior, invalid types/shapes, negative values, and NaN/Inf rejection.
- **P2:** Add container fixtures and assertions to the dilution tests (or a
  focused `dilution_strategy_test.py`) for particle distributions and
  scalar/multi-species gas. Compare against an independently calculated NumPy
  reference and snapshot mass, charge, density, distribution, volume, names,
  molar mass, partitioning, temperature, and pressure.
- **P3:** Add `particula/dynamics/tests/dilution_runnable_test.py` for strategy
  versus runnable agreement, expected concentration changes, exact no-ops,
  substep count/splitting, returned identity, and `|` composition.
- **P4:** Add public import and preflight immutability tests. Parameterize zero,
  negative, nonfinite, boolean/non-integer `sub_steps`, malformed shapes, and
  invalid existing state; verify both particle and gas snapshots remain exact.
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
ruff check particula/dynamics/ --fix
ruff format particula/dynamics/
ruff check particula/dynamics/
mypy particula/dynamics/ --ignore-missing-imports
```
