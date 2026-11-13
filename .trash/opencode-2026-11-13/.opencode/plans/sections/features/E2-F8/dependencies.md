# E2-F8 Dependencies

## Required Dependencies

- **E2-F1:** Container schema foundation. E2-F8 relies on the established
  `ParticleData` and `GasData` shape contracts and should not redefine them.

## Related Sibling Features

- **E2-F2:** Environment containers may influence documentation wording for
  box/environment boundaries.
- **E2-F3 and E2-F4:** Gas/environment boundaries and vapor-pressure
  ownership should remain consistent with support-boundary docs.
- **E2-F6 and E2-F7:** Numerical evidence and GPU condensation work are
  adjacent but not prerequisites for this CPU support-boundary feature.

## Internal Code Dependencies

- `particula/particles/particle_data.py`
- `particula/gas/gas_data.py`
- `particula/dynamics/condensation/condensation_strategies.py`
- `particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py`
- Existing condensation and coagulation test modules.

## External Dependencies

No new external Python dependencies are expected. Existing pytest, NumPy, and
ruff tooling are sufficient.

## Dependency Risks

- If E2-F1 container names or fields change before implementation, tests and
  documentation examples must be adjusted.
- If sibling features alter environment/gas boundaries, docs should reference
  those final terms rather than duplicating obsolete language.
