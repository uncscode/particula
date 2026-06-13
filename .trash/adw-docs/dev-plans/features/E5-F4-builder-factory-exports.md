# Feature E5-F4: Builder, Factory, and Exports

**Parent Epic**: [E5: Non-Isothermal Condensation with Latent Heat](../epics/E5-non-isothermal-condensation.md)
**Status**: Planning
**Priority**: P1
**Owners**: @Gorkowski
**Start Date**: TBD
**Target Date**: TBD
**Last Updated**: 2026-03-02
**Size**: Small (2 phases)

## Summary

Create `CondensationLatentHeatBuilder` following the established builder pattern,
register the new strategy in `CondensationFactory`, and update all namespace
exports so the new condensation strategy is accessible through the
`particula.dynamics` public API.

## Goals

1. `CondensationLatentHeatBuilder` extending `BuilderABC` with condensation
   mixins plus latent heat setter methods
2. Register `"latent_heat"` type in `CondensationFactory`
3. Export `CondensationLatentHeat` and `CondensationLatentHeatBuilder` from
   `particula.dynamics.condensation` and `particula.dynamics` namespaces

## Non-Goals

- Modifying existing builders or factory types
- Adding new builder mixins (existing mixins are sufficient)
- GUI or CLI integration

## Dependencies

- **E5-F1** (latent heat strategies) must land for `LatentHeatStrategy` type
- **E5-F3** (strategy class) must land for `CondensationLatentHeat` to wrap
- Existing builder infrastructure:
  - `BuilderABC` from `particula.abc_builder`
  - `BuilderMolarMassMixin` from `particula.builder_mixin`
  - `BuilderDiffusionCoefficientMixin`, `BuilderAccommodationCoefficientMixin`,
    `BuilderUpdateGasesMixin` from `condensation_builder_mixin.py` (94 lines)
  - `CondensationIsothermalBuilder` (60 lines) provides the pattern template

## Design

### Builder Structure

```python
# particula/dynamics/condensation/condensation_builder/
#   condensation_latent_heat_builder.py

class CondensationLatentHeatBuilder(
    BuilderABC,
    BuilderMolarMassMixin,
    BuilderDiffusionCoefficientMixin,
    BuilderAccommodationCoefficientMixin,
    BuilderUpdateGasesMixin,
):
    """Builder for CondensationLatentHeat strategy.

    Required: molar_mass, diffusion_coefficient, accommodation_coefficient
    Optional: latent_heat_strategy, latent_heat, activity_strategy,
              surface_strategy, vapor_pressure_strategy,
              skip_partitioning_indices, update_gases
    """

    def __init__(self):
        super().__init__(required_parameters=[
            "molar_mass",
            "diffusion_coefficient",
            "accommodation_coefficient",
        ])
        self.latent_heat_strategy = None
        self.latent_heat = 0.0

    def set_latent_heat_strategy(self, strategy):
        self.latent_heat_strategy = strategy
        return self

    def set_latent_heat(self, value, units="J/kg"):
        self.latent_heat = value  # + unit conversion
        return self

    def build(self) -> CondensationLatentHeat:
        self.pre_build_check()
        return CondensationLatentHeat(
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            update_gases=self.update_gases,
            latent_heat_strategy=self.latent_heat_strategy,
            latent_heat=self.latent_heat,
        )
```

### Factory Update

```python
# particula/dynamics/condensation/condensation_factories.py
# Add to get_builders():
{
    "isothermal": CondensationIsothermalBuilder(),
    "isothermal_staggered": CondensationIsothermalStaggeredBuilder(),
    "latent_heat": CondensationLatentHeatBuilder(),  # NEW
}
```

### Export Updates

Three `__init__.py` files need updating:
1. `particula/dynamics/condensation/condensation_builder/__init__.py` --
   add `CondensationLatentHeatBuilder`
2. `particula/dynamics/condensation/__init__.py` (22 lines) --
   add `CondensationLatentHeat` and `CondensationLatentHeatBuilder`
3. `particula/dynamics/__init__.py` (169 lines) --
   add re-exports for new symbols

## Phase Checklist

- [ ] **E5-F4-P1**: Create `CondensationLatentHeatBuilder` with tests
  - Issue: TBD | Size: S (~60 LOC) | Status: Not Started
  - File: `particula/dynamics/condensation/condensation_builder/
    condensation_latent_heat_builder.py`
  - Follow `CondensationIsothermalBuilder` pattern (see
    `condensation_isothermal_builder.py`, 60 lines -- extends `BuilderABC` +
    `BuilderDiffusionCoefficientMixin`, `BuilderAccommodationCoefficientMixin`,
    `BuilderUpdateGasesMixin` from `condensation_builder_mixin.py`)
  - Adds `set_latent_heat_strategy(strategy: LatentHeatStrategy)` and
    `set_latent_heat(value: float)` (constant fallback) methods
  - Required parameters: `molar_mass`, `diffusion_coefficient`,
    `accommodation_coefficient` (from mixins)
  - Optional parameters: `latent_heat_strategy`, `latent_heat`, plus all
    strategy parameters from base (`activity_strategy`, `surface_strategy`,
    `vapor_pressure_strategy`, `skip_partitioning_indices`)
  - `build()` returns `CondensationLatentHeat` instance
  - Update `condensation_builder/__init__.py` to export new builder
  - Tests: `particula/dynamics/condensation/tests/
    condensation_builder_test.py` (new or extend)
  - Tests: build with strategy, build with constant fallback, missing
    required param errors, parameter propagation to built object

- [ ] **E5-F4-P2**: Register in factory and update exports with tests
  - Issue: TBD | Size: S (~40 LOC) | Status: Not Started
  - Update `CondensationFactory` in `particula/dynamics/condensation/
    condensation_factories.py` (currently 49 lines, maps 2 types) to add
    `"latent_heat"` -> `CondensationLatentHeatBuilder`
  - Update `particula/dynamics/condensation/__init__.py` (currently 22 lines)
    to export `CondensationLatentHeat` and `CondensationLatentHeatBuilder`
  - Update `particula/dynamics/__init__.py` (currently 169 lines) to re-export
    new symbols
  - Tests: extend `particula/dynamics/condensation/tests/
    condensation_factories_test.py` (currently 117 lines)
  - Tests: factory creation with `"latent_heat"` type, round-trip parameter
    passing, invalid type error, import smoke tests for all new exports

## Critical Testing Requirements

- **No Coverage Modifications**: Keep coverage thresholds as shipped (80%).
- **Self-Contained Tests**: Ship `*_test.py` files proving builder and factory
  work correctly.
- **Test-First Completion**: Write and pass tests before declaring phases ready
  for review.
- **Round-Trip Validation**: Parameters passed to builder must propagate
  correctly to the built strategy object.

## Testing Strategy

### Unit Tests

| Test File | Phase | Coverage Target |
|-----------|-------|----------------|
| `particula/dynamics/condensation/tests/condensation_builder_test.py` | P1 | Builder class |
| `particula/dynamics/condensation/tests/condensation_factories_test.py` | P2 | Factory + exports |

### Key Test Cases

1. **Builder with strategy**: Pass `LatentHeatStrategy` object, verify it
   propagates to built `CondensationLatentHeat`
2. **Builder with constant**: Pass `latent_heat=2.5e6`, verify
   `ConstantLatentHeat` wrapper is created
3. **Builder missing params**: Omit required param, expect `ValueError`
4. **Factory creation**: `factory.get_strategy("latent_heat", params)` returns
   `CondensationLatentHeat`
5. **Factory invalid type**: `factory.get_strategy("bad_type", {})` raises
6. **Import smoke tests**: `from particula.dynamics import
   CondensationLatentHeat, CondensationLatentHeatBuilder`

## Success Criteria

1. `CondensationLatentHeatBuilder` creates valid strategy objects
2. `CondensationFactory` supports `"latent_heat"` type
3. All new symbols importable from `particula.dynamics` namespace
4. Existing builder and factory tests continue to pass
5. 80%+ test coverage for new code

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-02 | Initial feature document created from E5 epic | ADW |
