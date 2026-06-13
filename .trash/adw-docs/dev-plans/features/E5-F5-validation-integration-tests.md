# Feature E5-F5: Validation and Integration Tests

**Parent Epic**: [E5: Non-Isothermal Condensation with Latent Heat](../epics/E5-non-isothermal-condensation.md)
**Status**: Planning
**Priority**: P1
**Owners**: @Gorkowski
**Start Date**: TBD
**Target Date**: TBD
**Last Updated**: 2026-03-02
**Size**: Small (2 phases)

## Summary

Comprehensive integration tests validating mass conservation, isothermal-limit
parity, and physical accuracy of the non-isothermal condensation implementation
against literature values. These tests go beyond the unit tests in E5-F2 and
E5-F3 by testing the full end-to-end flow through the strategy class with
realistic physical scenarios.

## Goals

1. Mass conservation integration tests across all 3 distribution types
   (particle-resolved, discrete, continuous) with single and multi-species
2. Isothermal-limit parity tests verifying `CondensationLatentHeat` with L=0
   matches `CondensationIsothermal` to machine precision
3. Physical validation against known cloud droplet growth rates from
   Rogers & Yau (1989)
4. Verification that non-isothermal rate <= isothermal rate for ALL positive L

## Non-Goals

- Performance benchmarks (future, marked slow)
- GPU parity tests (deferred to E5-F7)
- Testing latent heat strategies in isolation (covered by E5-F1)

## Dependencies

- **E5-F3** (strategy class) must be fully implemented with all distribution
  types
- **E5-F4** (builder/factory) must land for factory-based construction in tests
- Physical constants and properties:
  - Water: L_ref = 2.501e6 J/kg, c = 2.3e3 J/(kg*K), T_ref = 273.15 K
  - Air thermal conductivity from `get_thermal_conductivity()`
  - Water vapor diffusion coefficient ~2.5e-5 m^2/s at 293 K

## Design

### Mass Conservation Test Methodology

Total mass = sum(gas_mass) + sum(particle_mass * concentration)

Test across:
- Distribution types: particle-resolved (100, 1000, 10000 particles), discrete
  (10, 50, 100 bins), continuous
- Species: single-species and multi-species (3 species)
- Physical scenario: water vapor at T=293K, P=101325Pa
- Tolerance: `|total_initial - total_final| / total_initial < 1e-12`

### Isothermal Parity Test Methodology

Run identical scenarios through both `CondensationIsothermal` and
`CondensationLatentHeat(latent_heat=0)`:
- Compare all output arrays element-wise
- Tolerance: < 1e-15 relative error on all mass arrays
- Single-species and multi-species (3 species)

### Physical Validation Methodology

Validate against Rogers & Yau (1989) Chapter 7:
- Scenario: water droplet, S=1.005 (0.5% supersaturation), T=273.15K,
  P=101325Pa, r=10 um
- Expected: non-isothermal dm/dt ~ 60-70% of isothermal dm/dt for water
  (thermal resistance is significant for water's large L)
- Tolerance: < 5% relative error vs literature values
- Property check: non-isothermal rate <= isothermal rate for ALL positive L
  (test with L = 100, 1000, 10000, 2.5e6 J/kg)

## Phase Checklist

- [ ] **E5-F5-P1**: Mass conservation and isothermal-limit integration tests
  - Issue: TBD | Size: M (~100 LOC) | Status: Not Started
  - File: `particula/dynamics/condensation/tests/
    latent_heat_conservation_test.py`
  - Mass conservation: `abs(total_initial - total_final) / total_initial
    < 1e-12` where total = sum(gas_mass) + sum(particle_mass * concentration)
  - Test across all 3 distribution types: particle-resolved (100, 1000, 10000
    particles), discrete (10, 50, 100 bins), continuous
  - Isothermal limit: when `latent_heat=0`, `CondensationLatentHeat.step()`
    results must match `CondensationIsothermal.step()` to machine precision
    (< 1e-15 relative error on all mass arrays)
  - Test with both single-species and multi-species (3 species) setups
  - Use water vapor at T=293K, P=101325Pa as baseline physical scenario

- [ ] **E5-F5-P2**: Physical validation against known cloud droplet growth
  - Issue: TBD | Size: M (~80 LOC) | Status: Not Started
  - File: `particula/dynamics/condensation/tests/
    latent_heat_validation_test.py`
  - Validate against Rogers & Yau (1989) Chapter 7: water droplet growth rate
    at S=1.005 (0.5% supersaturation), T=273.15K, P=101325Pa, r=10 um
  - Expected: non-isothermal dm/dt should be ~60-70% of isothermal dm/dt for
    water (thermal resistance is significant for water's large L)
  - Tolerance: < 5% relative error vs literature values (accounting for
    differences in assumed air properties)
  - Property check: non-isothermal rate <= isothermal rate for ALL positive L
    values (test with L = 100, 1000, 10000, 2.5e6 J/kg)
  - Energy bookkeeping: `abs(Q_released - sum(dm * L)) < 1e-14 * abs(Q)`
    for each step

## Critical Testing Requirements

- **No Coverage Modifications**: Keep coverage thresholds as shipped (80%).
- **Self-Contained Tests**: New test files with complete test scenarios.
- **Test-First Completion**: Write and pass tests before declaring phases ready
  for review.
- **Mass Conservation**: < 1e-12 relative tolerance across all distribution
  types.
- **Isothermal Parity**: < 1e-15 relative error when L=0.
- **Physical Accuracy**: < 5% relative error vs Rogers & Yau literature values.

## Testing Strategy

### Integration Tests

| Test File | Phase | Coverage Target |
|-----------|-------|----------------|
| `particula/dynamics/condensation/tests/latent_heat_conservation_test.py` | P1 | Mass conservation + isothermal parity |
| `particula/dynamics/condensation/tests/latent_heat_validation_test.py` | P2 | Physical validation |

### Parametrized Test Matrix (P1)

| Distribution Type | Particles/Bins | Species | Scenario |
|-------------------|----------------|---------|----------|
| Particle-resolved | 100 | 1 | Water, 293K |
| Particle-resolved | 1000 | 1 | Water, 293K |
| Particle-resolved | 10000 | 3 | Water + organics, 293K |
| Discrete | 10 | 1 | Water, 293K |
| Discrete | 50 | 3 | Water + organics, 293K |
| Discrete | 100 | 1 | Water, 293K |
| Continuous | N/A | 1 | Water, 293K |

### Physical Validation Matrix (P2)

| Parameter | Value | Source |
|-----------|-------|--------|
| Temperature | 273.15 K | Rogers & Yau Ch. 7 |
| Pressure | 101325 Pa | Standard atmosphere |
| Supersaturation | 0.5% (S=1.005) | Cloud activation |
| Droplet radius | 10 um | Cloud droplet size |
| Latent heat | 2.501e6 J/kg | Water at 273 K |
| Expected ratio | ~0.6-0.7 (non-iso/iso) | Literature |

## Literature References

1. **Rogers, R. R., & Yau, M. K. (1989).** *A Short Course in Cloud Physics*.
   Ch. 7: Growth of cloud droplets by condensation.
2. **Seinfeld, J. H., & Pandis, S. N. (2016).** *Atmospheric Chemistry and
   Physics*, Ch. 13, Eq. 13.3.
3. **Topping, D., & Bane, M. (2022).** *Introduction to Aerosol Modelling*,
   Eq. 2.36.

## Success Criteria

1. Mass conservation verified to < 1e-12 across all distribution types
2. Isothermal parity verified to < 1e-15 when L=0
3. Physical validation within 5% of Rogers & Yau literature values
4. Non-isothermal rate <= isothermal rate for all positive L values
5. Energy bookkeeping exact to < 1e-14 relative
6. All tests fast (< 10 seconds total)

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-02 | Initial feature document created from E5 epic | ADW |
