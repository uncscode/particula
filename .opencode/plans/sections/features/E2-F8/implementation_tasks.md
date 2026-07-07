# E2-F8 Implementation Tasks

## E2-F8-P1: Audit CPU Dynamics Container Boundaries and Baseline Docs

- [x] Confirm `ParticleData` and `GasData` shape contracts from their
  dataclasses.
- [x] Trace condensation data paths in `condensation_strategies.py` through
  unwrap, matching-type validation, and single-box validation.
- [x] Trace coagulation data paths in `coagulation_strategy_abc.py` through
  adapter helpers and mutation sites that access `particle.*[0]`.
- [x] Identify the minimum doc sections requiring updates.
- [x] Add small baseline tests only where they reduce ambiguity for later
  phases.
- [x] Land the three focused regressions for issue #1218 in the existing
  condensation and coagulation test modules.

## E2-F8-P2: Clarify Single-Box and Box-0 Behavior with Focused Tests

- Add public condensation tests that pass multi-box `ParticleData`/`GasData` to
  representative strategy methods and assert clear `ValueError` messages.
- Decide whether to retain the P1 box-0-only coagulation baseline as documented
  behavior or replace it with explicit unsupported-input validation.
- Prefer adding a small helper in `coagulation_strategy_abc.py` to require
  `n_boxes=1` before helpers mutate or compute from box 0 if the chosen P2
  direction is explicit rejection.
- If P2 changes runtime semantics, replace the transitional box-0 baseline with
  clear unsupported-input errors and update tests accordingly.
- Keep tests co-located with changed strategy code.

## E2-F8-P3: Improve Unsupported Multi-Box Errors and User Documentation

- Update `docs/Features/particle-data-migration.md` with a support table for
  CPU condensation and CPU coagulation.
- Explain that users who need multi-box CPU execution should loop over boxes at
  the caller level until strategy-level multi-box support exists.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` if roadmap language could
  imply all strategies execute every box today.
- Ensure error messages mention strategy-level support boundaries and `n_boxes`.
- Run focused dynamics tests and linters.

## Verification Commands

```bash
pytest particula/dynamics/condensation/tests/condensation_strategies_test.py -v
pytest particula/dynamics/coagulation/coagulation_strategy/tests/coagulation_strategy_abc_test.py -v
ruff check particula/ docs/Features --fix
ruff format particula/
```
