# E2-F8 Implementation Tasks

## E2-F8-P1: Audit CPU Dynamics Container Boundaries and Baseline Docs

- Confirm `ParticleData` and `GasData` shape contracts from their dataclasses.
- Trace condensation data paths in `condensation_strategies.py` through unwrap,
  matching-type validation, single-box validation, and box-0 array use.
- Trace coagulation data paths in `coagulation_strategy_abc.py` through adapter
  helpers and mutation sites that access `particle.*[0]`.
- Identify the minimum doc sections requiring updates.
- Add small baseline tests only if they reduce ambiguity for later phases.

## E2-F8-P2: Clarify Single-Box and Box-0 Behavior with Focused Tests

- Add public condensation tests that pass multi-box `ParticleData`/`GasData` to
  representative strategy methods and assert clear `ValueError` messages.
- Add coagulation tests for multi-box `ParticleData` strategy calls.
- Prefer adding a small helper in `coagulation_strategy_abc.py` to require
  `n_boxes=1` before helpers mutate or compute from box 0.
- Remove transitional box-0 behavior for unsupported multi-box inputs; tests
  should assert clear errors instead of documenting partial box-0 execution.
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
pytest particula/dynamics/condensation/tests/condensation_strategies_test.py
pytest particula/dynamics/coagulation/coagulation_strategy/tests/coagulation_strategy_abc_test.py
ruff check particula/ docs/Features --fix
ruff format particula/
```
