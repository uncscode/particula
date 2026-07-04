# E2-F8 Testing Strategy

## Test Goals

- Prove CPU condensation data-container paths reject multi-box inputs clearly.
- Prove CPU coagulation data-container paths do not imply full multi-box
  execution.
- Keep existing single-box data-container behavior working.
- Verify error messages are actionable and mention `n_boxes` or unsupported
  strategy-level multi-box execution.

## Co-Located Test Plan

- `particula/dynamics/condensation/tests/condensation_strategies_test.py`
  - Add representative public-method multi-box rejection tests.
  - Keep existing `_require_single_box` helper tests.
  - Assert `ParticleData` and `GasData` mismatches still raise `TypeError`.
- `particula/dynamics/coagulation/coagulation_strategy/tests/coagulation_strategy_abc_test.py`
  - Add multi-box `ParticleData` tests for strategy helpers and `step()` paths.
  - Prefer assertions that unsupported multi-box calls raise `ValueError`.
  - Assert the error message names unsupported multi-box execution rather than
    implying box-0 fallback behavior.

## Non-Regression Tests

- Existing single-box condensation data tests continue to pass.
- Existing coagulation adapter tests continue to pass for `n_boxes=1`.
- Legacy `ParticleRepresentation` and `GasSpecies` paths remain unchanged.

## Commands

```bash
pytest particula/dynamics/condensation/tests/condensation_strategies_test.py
pytest particula/dynamics/coagulation/coagulation_strategy/tests/coagulation_strategy_abc_test.py
pytest particula/dynamics/condensation/tests/condensation_strategies_test.py particula/dynamics/coagulation/coagulation_strategy/tests/coagulation_strategy_abc_test.py
```

## Acceptance for Tests

Every phase that changes strategy behavior or docs must include its co-located
unit tests. There is no standalone testing phase.
