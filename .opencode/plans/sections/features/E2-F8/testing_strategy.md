# E2-F8 Testing Strategy

## Test Goals

- Preserve the current CPU condensation public data-container multi-box
  rejection boundary through representative public entry points.
- Prove CPU coagulation data-container paths reject unsupported multi-box
  `ParticleData` execution instead of silently operating on box 0.
- Keep existing single-box data-container behavior working.
- Keep assertions focused on stable contract terms such as `n_boxes`,
  `single-box`, and unsupported multi-box CPU execution.

## Co-Located Test Plan

- `particula/dynamics/condensation/tests/condensation_strategies_test.py`
  - Keep `mass_transfer_rate(...)` multi-box rejection coverage and add a
    representative public `step()`-path rejection case.
  - Keep existing `_require_single_box` helper tests.
  - Assert `ParticleData` and `GasData` mismatches still raise `TypeError`.
- `particula/dynamics/coagulation/coagulation_strategy/tests/coagulation_strategy_abc_test.py`
  - Replace multi-box helper-backed box-0 regressions with rejection tests.
  - Replace particle-resolved multi-box mutation regressions with
    raise-before-mutation tests.
  - Retain single-box non-regression coverage, including supported no-op cases.

## Non-Regression Tests

- Existing single-box condensation data tests continue to pass.
- Existing coagulation adapter tests continue to pass for `n_boxes=1`, with the
  new regressions documenting explicit multi-box rejection behavior.
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
