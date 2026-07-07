# E2-F8 Testing Strategy

## Test Goals

- Preserve the current CPU condensation public data-container multi-box
  rejection boundary.
- Prove current CPU coagulation data-container paths do not imply full multi-box
  execution and still operate on box 0 only.
- Keep existing single-box data-container behavior working.
- Keep this phase audit-only: regressions should document baseline semantics
  without changing runtime behavior.

## Co-Located Test Plan

- `particula/dynamics/condensation/tests/condensation_strategies_test.py`
  - Add a representative public `mass_transfer_rate(...)` multi-box rejection
    regression.
  - Keep existing `_require_single_box` helper tests.
  - Assert `ParticleData` and `GasData` mismatches still raise `TypeError`.
- `particula/dynamics/coagulation/coagulation_strategy/tests/coagulation_strategy_abc_test.py`
  - Add multi-box `ParticleData` regressions for helper-backed reads when box 1+
    intentionally differs from box 0.
  - Add a particle-resolved `step()` regression showing box 0 mutates while
    later boxes remain unchanged.

## Non-Regression Tests

- Existing single-box condensation data tests continue to pass.
- Existing coagulation adapter tests continue to pass for `n_boxes=1`, with the
  new regressions documenting current multi-box box-0 fallback behavior.
- Legacy `ParticleRepresentation` and `GasSpecies` paths remain unchanged.

## Commands

```bash
pytest particula/dynamics/condensation/tests/condensation_strategies_test.py -v
pytest particula/dynamics/coagulation/coagulation_strategy/tests/coagulation_strategy_abc_test.py -v
pytest particula/dynamics/condensation/tests/condensation_strategies_test.py particula/dynamics/coagulation/coagulation_strategy/tests/coagulation_strategy_abc_test.py -v -Werror
```

## Acceptance for Tests

Every phase that changes strategy behavior or docs must include its co-located
unit tests. There is no standalone testing phase.
