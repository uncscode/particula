# Feature E5-F3: CondensationLatentHeat Strategy Class

**Parent Epic**: [E5: Non-Isothermal Condensation with Latent Heat](../epics/E5-non-isothermal-condensation.md)
**Status**: Planning
**Priority**: P1
**Owners**: @Gorkowski
**Start Date**: TBD
**Target Date**: TBD
**Last Updated**: 2026-03-02
**Size**: Large (5 phases)

## Summary

Create `CondensationLatentHeat`, a new condensation strategy class that extends
`CondensationStrategy` with non-isothermal mass transfer corrections. This class
reuses all inherited infrastructure (mean free path, Knudsen number,
first-order mass transport, pressure delta calculation) and replaces only the
final mass transfer rate computation with the latent-heat-corrected version from
E5-F2.

The strategy supports an optional `LatentHeatStrategy` for temperature-dependent
latent heat, a constant fallback value, or automatic isothermal behavior when
neither is provided. Energy released/absorbed during each step is tracked as a
diagnostic attribute.

## Goals

1. `CondensationLatentHeat` class extending `CondensationStrategy` with latent
   heat correction on mass transfer rate
2. Support all three distribution types: particle-resolved, discrete, continuous
3. Support both legacy (`ParticleRepresentation` + `GasSpecies`) and data-only
   (`ParticleData` + `GasData`) input paths
4. Track cumulative latent heat energy per step as a diagnostic
5. Isothermal fallback when no latent heat strategy or value is provided
6. API-compatible with `MassCondensation` runnable (no changes needed)

## Non-Goals

- Modifying the `MassCondensation` runnable
- Staggered variant (`CondensationLatentHeatStaggered`) -- trivial follow-on
- Temperature feedback to gas phase
- GPU/Warp implementation (deferred to E5-F7)

## Dependencies

- **E5-F1** (latent heat strategies) must land for strategy injection
- **E5-F2** (mass transfer functions) must land for
  `get_mass_transfer_rate_latent_heat()` and `get_latent_heat_energy_released()`
- `CondensationStrategy` ABC (line 169-683 in `condensation_strategies.py`,
  currently 1699 lines) is stable
- `CondensationIsothermal` (line 687-1041) provides the implementation template
- `get_thermal_conductivity()` in `particula.gas.properties` (46 lines) exists

## Design

### Class Structure

```python
# In particula/dynamics/condensation/condensation_strategies.py (extend)

class CondensationLatentHeat(CondensationStrategy):
    def __init__(self, ...,
                 latent_heat_strategy: LatentHeatStrategy | None = None,
                 latent_heat: float | NDArray = 0.0):
        super().__init__(...)
        self._latent_heat_strategy = self._resolve_latent_heat_strategy(
            latent_heat_strategy, latent_heat)
        self.last_latent_heat_energy = 0.0  # diagnostic

    def _resolve_latent_heat_strategy(self, strategy, constant):
        """Resolution priority: strategy > constant > isothermal."""
        if strategy is not None:
            return strategy
        if constant > 0:
            return ConstantLatentHeat(constant)
        return None  # isothermal fallback
```

### Method Flow

`mass_transfer_rate()` follows `CondensationIsothermal.mass_transfer_rate()`
(lines 761-833) exactly through steps 1-3, then branches:

1. Unwrap particle/gas, fill/clip radii (same as isothermal)
2. Compute `first_order_mass_transport` via inherited base method
3. Compute `pressure_delta` via inherited `calculate_pressure_delta()`
4. **New**: Evaluate L, get thermal conductivity, extract vapor pressure at
   surface, call `get_mass_transfer_rate_latent_heat()`

`step()` follows `CondensationIsothermal.step()` (lines 922-1040) with two
additions:
1. Uses latent-heat-corrected mass transfer rate from above
2. After computing mass transfer, calls `get_latent_heat_energy_released()`
   and stores in `self.last_latent_heat_energy`

### API Compatibility

- Return type of `step()`: `Tuple[ParticleRepresentation | ParticleData,
  GasSpecies | GasData]` (unchanged)
- The `MassCondensation` runnable only calls `strategy.step()` and
  `strategy.rate()` -- no modification needed
- Energy diagnostic is accessed via `strategy.last_latent_heat_energy` after
  calling `step()`

## Phase Checklist

- [ ] **E5-F3-P1**: Create `CondensationLatentHeat` class skeleton with tests
  - Issue: TBD | Size: M (~80 LOC) | Status: Not Started
  - File: `particula/dynamics/condensation/condensation_strategies.py` (extend
    existing file, currently 1699 lines)
  - Extends `CondensationStrategy` (inherits all base methods including
    `mean_free_path`, `knudsen_number`, `first_order_mass_transport`,
    `calculate_pressure_delta`, `_fill_zero_radius`, `_apply_skip_partitioning`)
  - Additional constructor parameters beyond base class:
    - `latent_heat_strategy: LatentHeatStrategy | None = None`
    - `latent_heat: float | NDArray = 0.0` (constant fallback)
  - Resolution logic in `_resolve_latent_heat_strategy()`:
    1. If `latent_heat_strategy` provided -> use it directly
    2. Else if `latent_heat > 0` -> wrap in `ConstantLatentHeat(latent_heat)`
    3. Else (both zero/None) -> store None, behave as isothermal
  - Instance attribute: `last_latent_heat_energy: float = 0.0` (diagnostic)
  - Tests: `particula/dynamics/condensation/tests/
    condensation_strategies_test.py` (extend existing, currently 1655 lines)
  - Tests: instantiation with all param combos, strategy resolution priority,
    fallback to isothermal when L=0, type errors for bad inputs

- [ ] **E5-F3-P2**: Implement `mass_transfer_rate()` and `rate()` with tests
  - Issue: TBD | Size: M (~100 LOC) | Status: Not Started
  - `mass_transfer_rate()` follows `CondensationIsothermal.mass_transfer_rate()`
    flow (lines 761-833) but replaces the final `get_mass_transfer_rate()` call
    with `get_mass_transfer_rate_latent_heat()`:
    1. Unwrap particle/gas, fill/clip radii (same as isothermal)
    2. Compute `first_order_mass_transport` via inherited base method
    3. Compute `pressure_delta` via inherited `calculate_pressure_delta()`
    4. Evaluate `L = latent_heat_strategy.latent_heat(temperature)`
    5. Get `kappa = get_thermal_conductivity(temperature)` (dry air, from
       `particula.gas.properties`, 46 lines)
    6. Extract `vapor_pressure_surface` from `calculate_pressure_delta()`
       intermediate
    7. Call `get_mass_transfer_rate_latent_heat(...)`
  - `rate()`: identical to `CondensationIsothermal.rate()` -- multiply
    `mass_transfer_rate` by particle concentration, apply skip-partitioning
  - Tests: numerical parity with `CondensationIsothermal` when L=0 (< 1e-15
    relative), reduced rate when L > 0, array shapes for single/multi-species

- [ ] **E5-F3-P3**: Implement `step()` for particle-resolved with energy
  tracking and tests
  - Issue: TBD | Size: M (~100 LOC) | Status: Not Started
  - `step()` follows `CondensationIsothermal.step()` flow (lines 922-1040)
    with two additions:
    1. Uses `mass_transfer_rate` from P2 (includes thermal correction)
    2. After computing `mass_transfer` via `get_mass_transfer()`, calls
       `get_latent_heat_energy_released(mass_transfer, L)` and stores result
       in `self.last_latent_heat_energy`
  - Return type: `Tuple[ParticleRepresentation | ParticleData, GasSpecies |
    GasData]` (same as isothermal)
  - Scope: particle-resolved distribution type only
  - Tests: mass conservation (gas + particle total < 1e-12), energy = sum(dm
    * L) to machine precision, sign conventions, single and multi-species

- [ ] **E5-F3-P4**: Add discrete and continuous distribution support with tests
  - Issue: TBD | Size: S (~60 LOC) | Status: Not Started
  - Extend `step()` to handle discrete (binned) and continuous (PDF)
    distribution types
  - Uses same `get_mass_transfer()` routing as isothermal -- the only
    difference is the mass transfer rate computation (already handled in P2)
  - Verify array shapes for 2D mass arrays (particles x species) with
    discrete bins
  - Tests: discrete distribution with 10, 50, 100 bins; continuous
    distribution; mass conservation across all types; isothermal parity
    when L=0

- [ ] **E5-F3-P5**: Add data-only path support and parity tests
  - Issue: TBD | Size: S (~50 LOC) | Status: Not Started
  - Ensure `ParticleData` + `GasData` input path works identically to
    `ParticleRepresentation` + `GasSpecies` legacy path
  - Follow the same pattern as `CondensationIsothermal.step()` lines
    1011-1040 for the per-particle mass division in data-only path
  - Parity test: legacy path vs data-only path with `rtol=1e-10`
  - Edge cases: zero particles (no crash), single species, very small
    particles (< MIN_PARTICLE_RADIUS_M), zero concentration particles

## Critical Testing Requirements

- **No Coverage Modifications**: Keep coverage thresholds as shipped (80%).
- **Self-Contained Tests**: Extend existing `condensation_strategies_test.py`
  (1655 lines) with new test classes/functions.
- **Test-First Completion**: Write and pass tests before declaring phases ready
  for review.
- **Mass Conservation**: All distribution types must preserve total mass to
  within 1e-12 relative tolerance.
- **Isothermal Parity**: When `latent_heat=0`, results must exactly match
  `CondensationIsothermal` (< 1e-15 relative).
- **Energy Bookkeeping**: `last_latent_heat_energy` must equal `sum(dm * L)`
  to machine precision.

## Testing Strategy

### Unit Tests

| Test File | Phase | Coverage Target |
|-----------|-------|----------------|
| `particula/dynamics/condensation/tests/condensation_strategies_test.py` | P1-P5 | Strategy class |

### Key Test Cases

1. **Instantiation**: All parameter combinations, strategy resolution priority
2. **Isothermal parity**: L=0 matches `CondensationIsothermal` to < 1e-15
3. **Rate reduction**: L > 0 always produces lower rate than isothermal
4. **Mass conservation**: gas + particle total mass unchanged (< 1e-12)
5. **Energy tracking**: `last_latent_heat_energy == sum(dm * L)` to < 1e-14
6. **Distribution types**: particle-resolved, discrete (10/50/100 bins),
   continuous
7. **Data-only parity**: `ParticleData`/`GasData` path matches legacy path
8. **Edge cases**: zero particles, single species, very small particles

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Large file (1699+ lines) | Add new class at end, keep helper functions modular |
| Vapor pressure surface extraction | May need refactoring `calculate_pressure_delta()` or computing separately |
| Energy diagnostic thread safety | Attribute is overwritten each step, document single-threaded assumption |

## Success Criteria

1. `CondensationLatentHeat` works with all 3 distribution types
2. Both legacy and data-only input paths produce identical results
3. Isothermal parity verified to machine precision
4. Mass conservation verified across all distribution types
5. Energy diagnostic tracks Q = sum(dm * L) to machine precision
6. `MassCondensation` runnable works unchanged with new strategy
7. 80%+ test coverage for new code

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-02 | Initial feature document created from E5 epic | ADW |
