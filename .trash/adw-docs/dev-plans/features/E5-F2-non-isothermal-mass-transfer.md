# Feature E5-F2: Non-Isothermal Mass Transfer Functions

**Parent Epic**: [E5: Non-Isothermal Condensation with Latent Heat](../epics/E5-non-isothermal-condensation.md)
**Status**: Planning
**Priority**: P1
**Owners**: @Gorkowski
**Start Date**: TBD
**Target Date**: TBD
**Last Updated**: 2026-03-02
**Size**: Small (3 phases)

## Summary

Add pure functions for non-isothermal mass transfer to the existing
`particula/dynamics/condensation/mass_transfer.py` module. These functions
implement the thermal resistance correction from Topping & Bane (2022) Eq. 2.36
and Seinfeld & Pandis (2016) Eq. 13.3, plus a latent heat energy release
tracking function.

The thermal resistance factor modifies the isothermal mass transfer rate by
adding a denominator correction that accounts for the latent heat released or
absorbed during phase change. When latent heat is zero, the non-isothermal rate
reduces exactly to the existing isothermal rate.

## Goals

1. `get_thermal_resistance_factor()` -- computes the thermodynamic denominator
   correction (purely thermodynamic, no particle size dependence)
2. `get_mass_transfer_rate_latent_heat()` -- non-isothermal mass transfer rate
   that reduces to `get_mass_transfer_rate()` when L=0
3. `get_latent_heat_energy_released()` -- energy bookkeeping function (Q = dm * L)
4. All functions are pure, use `@validate_inputs`, and handle array broadcasting

## Non-Goals

- Modifying existing isothermal functions
- Adding particle size dependence to the thermal resistance factor (size effects
  are in the numerator via `first_order_mass_transport`, already computed by the
  base strategy class)
- Temperature feedback (energy tracking is diagnostic only)

## Dependencies

- **E5-F1** (latent heat strategies) must land first so the strategy pattern is
  available, though the pure functions in this feature only take numeric values
  (not strategy objects) and could technically be developed in parallel.
- Existing functions in `mass_transfer.py` (461 lines, 6 functions) are stable.
- `GAS_CONSTANT` from `scipy.constants` (already used in mass_transfer.py).

## Design

### Thermal Resistance Factor

From Topping & Bane (2022) Eq. 2.36 denominator:

```
thermal_factor = [D * L * p_surf / (kappa * T)] * [L / (R_specific * T) - 1]
                 + R_specific * T
```

Where:
- `D` = diffusion coefficient of vapor in air [m^2/s]
- `L` = latent heat of vaporization [J/kg]
- `p_surf` = equilibrium vapor pressure at particle surface [Pa]
- `kappa` = thermal conductivity of air [W/(m*K)]
- `T` = temperature [K]
- `R_specific = R / M` = specific gas constant [J/(kg*K)]

**Key insight**: This denominator is purely thermodynamic -- it does NOT depend
on particle size. Size effects enter only through `k_cond` (the numerator's
first-order mass transport coefficient), which the base class already computes
with Knudsen/transition regime corrections.

### Non-Isothermal Mass Transfer Rate

```
dm/dt = isothermal_rate / thermal_factor
      = [K * delta_p * M / (R * T)] / thermal_factor
```

When `L = 0`:
- `thermal_factor = R_specific * T = (R/M) * T`
- The expression reduces exactly to `K * delta_p * M / (R * T)` which is
  `get_mass_transfer_rate()` -- this must be verified to machine precision.

### Energy Release Tracking

```
Q = dm * L    [J]
```

Sign convention: positive dm (condensation) = heat released (Q > 0),
negative dm (evaporation) = heat absorbed (Q < 0).

## Phase Checklist

- [ ] **E5-F2-P1**: Add thermal resistance factor function with tests
  - Issue: TBD | Size: S (~50 LOC) | Status: Not Started
  - File: `particula/dynamics/condensation/mass_transfer.py` (extend existing
    file, currently 461 lines with 6 functions)
  - Function: `get_thermal_resistance_factor(diffusion_coefficient,
    latent_heat, vapor_pressure_surface, thermal_conductivity, temperature,
    molar_mass) -> float | NDArray`
  - Computes the denominator correction (Topping & Bane Eq. 2.36 denominator)
  - Where `R_specific = R / molar_mass` (specific gas constant, J/(kg*K))
  - Note: `p_surf` is the equilibrium vapor pressure at the particle surface
    (from `calculate_pressure_delta()` partial_pressure_particle), NOT the
    partial pressure delta
  - Pure function, no class state, uses `@validate_inputs` decorator
  - Tests: `particula/dynamics/condensation/tests/mass_transfer_test.py`
    (extend existing file, currently 440 lines)
  - Tests: known values for water at 293 K, isothermal limit (L=0 should give
    `R_specific * T`), dimensional consistency, array broadcasting

- [ ] **E5-F2-P2**: Add non-isothermal mass transfer rate function with tests
  - Issue: TBD | Size: S (~50 LOC) | Status: Not Started
  - File: `particula/dynamics/condensation/mass_transfer.py` (extend)
  - Function: `get_mass_transfer_rate_latent_heat(pressure_delta,
    first_order_mass_transport, temperature, molar_mass, latent_heat,
    thermal_conductivity, vapor_pressure_surface) -> float | NDArray`
  - Computes: `isothermal_rate / thermal_factor` where `thermal_factor` is
    from `get_thermal_resistance_factor()`
  - When `latent_heat = 0`, the thermal_factor reduces to `R_specific * T`
    and the full expression reduces exactly to `get_mass_transfer_rate()`
    (isothermal limit). This MUST be tested to machine precision.
  - Uses `@validate_inputs` decorator with same validation as
    `get_mass_transfer_rate`
  - Tests: isothermal parity (L=0 matches `get_mass_transfer_rate` to
    < 1e-15 relative), known cloud droplet growth rate for water at S=1.01
    (1% supersaturation), multi-species array shapes (1D and 2D)

- [ ] **E5-F2-P3**: Add latent heat energy release tracking function with tests
  - Issue: TBD | Size: XS (~30 LOC) | Status: Not Started
  - File: `particula/dynamics/condensation/mass_transfer.py` (extend)
  - Function: `get_latent_heat_energy_released(mass_transfer, latent_heat)
    -> float | NDArray`
  - Note: takes `mass_transfer` (kg, the actual mass moved in a step) not
    `mass_transfer_rate` (kg/s). The time_step is already applied by
    `get_mass_transfer()` before this function is called.
  - Returns energy [J] released to gas phase per particle: `Q = dm * L`
  - Sign convention: positive dm = condensation = heat released (Q > 0),
    negative dm = evaporation = heat absorbed (Q < 0)
  - Uses `@validate_inputs` decorator
  - Tests: energy conservation (Q = dm * L to machine precision), sign
    conventions for condensation/evaporation, scalar and array shapes

## Critical Testing Requirements

- **No Coverage Modifications**: Keep coverage thresholds as shipped (80%).
- **Self-Contained Tests**: Extend existing `mass_transfer_test.py` (440 lines)
  with new test functions.
- **Test-First Completion**: Write and pass tests before declaring phases ready
  for review.
- **Isothermal Parity**: When `latent_heat=0`,
  `get_mass_transfer_rate_latent_heat()` must match `get_mass_transfer_rate()`
  to machine precision (< 1e-15 relative).
- **Energy Precision**: `Q = dm * L` to machine precision (< 1e-14 relative).

## Testing Strategy

### Unit Tests

| Test File | Phase | Coverage Target |
|-----------|-------|----------------|
| `particula/dynamics/condensation/tests/mass_transfer_test.py` | P1-P3 | All new functions |

### Key Test Cases

1. **Thermal resistance at L=0**: Should return `R_specific * T` exactly
2. **Thermal resistance for water at 293 K**: Known values from Seinfeld &
   Pandis
3. **Isothermal parity**: L=0 non-isothermal rate == isothermal rate to < 1e-15
4. **Rate reduction**: Non-isothermal rate < isothermal rate for all L > 0
5. **Energy sign convention**: Condensation (dm > 0) gives Q > 0
6. **Array broadcasting**: 1D and 2D inputs produce correct shapes

## Literature References

1. **Topping & Bane (2022)**, Eq. 2.36 -- Thermal resistance denominator
2. **Seinfeld & Pandis (2016)**, Eq. 13.3 -- Non-isothermal mass transfer
3. **Rogers & Yau (1989)**, Ch. 7 -- Cloud droplet growth rate validation

## Success Criteria

1. Three new pure functions added to `mass_transfer.py`
2. Isothermal limit verified to machine precision (< 1e-15 relative)
3. Non-isothermal rate always <= isothermal rate for positive L
4. Energy tracking exact (Q = dm * L to < 1e-14 relative)
5. All existing tests continue to pass
6. 80%+ coverage for new code

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-02 | Initial feature document created from E5 epic | ADW |
