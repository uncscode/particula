# Feature E5-F1: Latent Heat Strategy Pattern

**Parent Epic**: [E5: Non-Isothermal Condensation with Latent Heat](../epics/E5-non-isothermal-condensation.md)
**Status**: Planning
**Priority**: P1
**Owners**: @Gorkowski
**Start Date**: TBD
**Target Date**: TBD
**Last Updated**: 2026-03-02
**Size**: Medium (4 phases)

## Summary

Create a `LatentHeatStrategy` ABC with three concrete implementations (constant,
linear, power-law) following the dependency injection pattern established by
`VaporPressureStrategy`. Include builders and factory for each strategy type, and
export from the `particula.gas` namespace.

## Goals

1. Define `LatentHeatStrategy` ABC with abstract `latent_heat(temperature)`
   method returning `float | NDArray` in J/kg
2. Implement `ConstantLatentHeat`, `LinearLatentHeat`, and `PowerLawLatentHeat`
   concrete strategies with proper edge-case handling
3. Create builders extending `BuilderABC` with `pre_build_check()` validation
4. Create `LatentHeatFactory` extending `StrategyFactoryABC` with type mappings
5. Export all new public symbols from `particula/gas/__init__.py`

## Non-Goals

- Temperature evolution or feedback (out of scope for E5)
- Heat capacity strategies (separate concern)
- GPU/Warp implementations (deferred to E5-F7)

## Dependencies

- None blocking. All required infrastructure exists:
  - `VaporPressureStrategy` pattern (ABC + concrete in one file) provides the
    template
  - `BuilderABC` from `particula.abc_builder` for builder pattern
  - `StrategyFactoryABC` from `particula.abc_factory` for factory pattern

## Design

### Strategy Classes

All strategies live in a single file following the `VaporPressureStrategy`
pattern where the ABC and all concrete classes share one module.

```python
# particula/gas/latent_heat_strategies.py

class LatentHeatStrategy(ABC):
    """ABC for latent heat of vaporization parametrizations."""
    @abstractmethod
    def latent_heat(self, temperature) -> float | NDArray:
        """Return latent heat [J/kg] at given temperature [K]."""
        ...

class ConstantLatentHeat(LatentHeatStrategy):
    """Fixed latent heat, independent of temperature."""
    def __init__(self, latent_heat_ref: float):  # [J/kg]
        ...

class LinearLatentHeat(LatentHeatStrategy):
    """L(T) = L_ref - slope * (T - T_ref). Standard in cloud modeling."""
    def __init__(self, latent_heat_ref, slope, temperature_ref):
        ...

class PowerLawLatentHeat(LatentHeatStrategy):
    """L(T) = L_ref * (1 - T/T_c)^beta. Captures L->0 as T->T_c."""
    def __init__(self, latent_heat_ref, critical_temperature, beta):
        ...
```

### Builders

Each builder follows the `VaporPressureBuilder` pattern from
`particula/gas/vapor_pressure_builders.py`: extends `BuilderABC`, defines
required parameters, implements `set_*()` methods, and returns a strategy
from `build()`.

### Factory

`LatentHeatFactory` extends `StrategyFactoryABC` (same pattern as
`VaporPressureFactory` in `particula/gas/vapor_pressure_factories.py`):

```python
get_builders() -> {
    "constant": ConstantLatentHeatBuilder(),
    "linear": LinearLatentHeatBuilder(),
    "power_law": PowerLawLatentHeatBuilder(),
}
```

## Phase Checklist

- [ ] **E5-F1-P1**: Create `LatentHeatStrategy` ABC and `ConstantLatentHeat`
  with tests
  - Issue: TBD | Size: S (~60 LOC) | Status: Not Started
  - File: `particula/gas/latent_heat_strategies.py`
  - ABC: `LatentHeatStrategy` with abstract method
    `latent_heat(temperature) -> float | NDArray`
  - `ConstantLatentHeat(latent_heat_ref)` returns fixed value
  - Units: J/kg (per-species, consistent with thermodynamic convention)
  - Follow `VaporPressureStrategy` pattern: ABC + concrete classes in one file
  - Tests: `particula/gas/tests/latent_heat_strategies_test.py`
  - Test constant returns correct value, array broadcasting, type consistency

- [ ] **E5-F1-P2**: Add `LinearLatentHeat` and `PowerLawLatentHeat` strategies
  with tests
  - Issue: TBD | Size: M (~80 LOC) | Status: Not Started
  - `LinearLatentHeat(latent_heat_ref, slope, temperature_ref)`:
    `L(T) = L_ref - slope * (T - T_ref)`
  - `PowerLawLatentHeat(latent_heat_ref, critical_temperature, beta)`:
    `L(T) = L_ref * (1 - T/T_c)^beta`
  - Both handle edge cases: T > T_c clamps to 0 for power law
  - Tests: validate against known water values at 273 K, 293 K, 313 K
  - Tests: array input, scalar input, edge cases

- [ ] **E5-F1-P3**: Add latent heat builders with tests
  - Issue: TBD | Size: M (~90 LOC) | Status: Not Started
  - File: `particula/gas/latent_heat_builders.py`
  - Follow `VaporPressureBuilder` pattern (see
    `particula/gas/vapor_pressure_builders.py`, 517 lines, for reference)
  - `ConstantLatentHeatBuilder` -- extends `BuilderABC`, required param:
    `latent_heat_ref` [J/kg]; `build()` returns `ConstantLatentHeat`
  - `LinearLatentHeatBuilder` -- required params: `latent_heat_ref`, `slope`,
    `temperature_ref`; `build()` returns `LinearLatentHeat`
  - `PowerLawLatentHeatBuilder` -- required params: `latent_heat_ref`,
    `critical_temperature`, `beta`; `build()` returns `PowerLawLatentHeat`
  - Each builder uses `pre_build_check()` to validate required keys
  - Tests: `particula/gas/tests/latent_heat_builders_test.py`
  - Tests: build with valid params, missing-param errors, round-trip values

- [ ] **E5-F1-P4**: Add latent heat factory and gas exports with tests
  - Issue: TBD | Size: S (~60 LOC) | Status: Not Started
  - File: `particula/gas/latent_heat_factories.py`
  - `LatentHeatFactory` extends `StrategyFactoryABC` (from
    `particula/abc_factory.py`, 111 lines -- same pattern as
    `VaporPressureFactory` in `particula/gas/vapor_pressure_factories.py`)
  - Maps `"constant"` -> `ConstantLatentHeatBuilder`, `"linear"` ->
    `LinearLatentHeatBuilder`, `"power_law"` -> `PowerLawLatentHeatBuilder`
  - Export all strategies, builders, and factory from
    `particula/gas/__init__.py` (currently 146 lines)
  - Tests: `particula/gas/tests/latent_heat_factories_test.py`
  - Tests: factory creation for each type, round-trip parameter passing,
    invalid type error, import smoke tests

## Critical Testing Requirements

- **No Coverage Modifications**: Keep coverage thresholds as shipped (80%).
- **Self-Contained Tests**: Attach `*_test.py` files that prove each phase is
  complete.
- **Test-First Completion**: Write and pass tests before declaring phases ready
  for review.
- **80%+ Coverage**: Every phase must ship tests that maintain at least 80%
  coverage.

## Testing Strategy

### Unit Tests

| Test File | Phase | Coverage Target |
|-----------|-------|----------------|
| `particula/gas/tests/latent_heat_strategies_test.py` | P1, P2 | Strategy classes |
| `particula/gas/tests/latent_heat_builders_test.py` | P3 | Builder validation |
| `particula/gas/tests/latent_heat_factories_test.py` | P4 | Factory creation |

### Test Cases

1. **Constant**: Returns fixed value for scalar and array temperatures
2. **Linear**: Validated against known water values (L_ref=2.501e6, c=2.3e3,
   T_ref=273.15) at 273 K, 293 K, 313 K
3. **Power Law**: Validated with beta=0.38, T_c=647.1 K for water; edge case
   T >= T_c returns 0
4. **Array Broadcasting**: All strategies handle 1D arrays correctly
5. **Builders**: Build with valid params, missing-param errors, round-trip
6. **Factory**: Creation for each type, invalid type error

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Power law edge case at T >= T_c | Explicit `np.clip` to clamp ratio to [0, inf) |
| Units confusion (J/kg vs J/mol) | Document J/kg convention clearly in docstrings |
| API drift from VaporPressure pattern | Follow existing pattern exactly, reuse base classes |

## Success Criteria

1. `LatentHeatStrategy` ABC with 3 concrete implementations
2. All strategies return correct values for known water parametrizations
3. Builders validate required parameters and produce correct strategy objects
4. Factory maps string types to builders and creates strategies
5. All new symbols exported from `particula.gas` namespace
6. 80%+ test coverage for all new code

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-02 | Initial feature document created from E5 epic | ADW |
