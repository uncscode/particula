# Epic E5: Non-Isothermal Condensation with Latent Heat

**Status**: Planning
**Priority**: P1
**Owners**: @Gorkowski
**Start Date**: 2026-03-02
**Target Date**: TBD
**Last Updated**: 2026-03-01
**Size**: Medium (7 features, ~22 phases)

## Vision

Add a **non-isothermal condensation strategy** to particula that accounts for
the latent heat of vaporization during mass transfer. The current
`CondensationIsothermal` strategy assumes constant temperature, which is valid
for small organic aerosol growth but breaks down for cloud droplet formation and
any process where the heat released or absorbed during phase change creates a
significant thermal resistance to further mass transfer.

The non-isothermal mass transfer equation (Topping & Bane 2022, Eq. 2.36;
Seinfeld & Pandis 2016) modifies the isothermal rate with a denominator term
that accounts for latent heat effects:

```
dm/dt = [N * 4*pi*r * Di * (p_gas - p_surface)] /
        { [Di * Li * pi / (kappa * T)] * [Li/(R*T) - 1] + Ri * T }
```

This epic delivers:
1. **Latent heat strategy pattern** (fixed, linear, power-law) following the
   existing dependency injection pattern used by `VaporPressureStrategy`
2. **Non-isothermal mass transfer math** as composable pure functions
3. **`CondensationLatentHeat` strategy** for simultaneous particle updates
4. **Latent heat energy tracking** to the gas phase (energy bookkeeping for
   future temperature evolution feedback)
5. **Builder, factory, and export integration** following existing patterns
6. **Python-native implementation first**, then GPU/Warp translation as a
   follow-on feature

## Scope

### In Scope

- `LatentHeatStrategy` ABC with `ConstantLatentHeat`, `LinearLatentHeat`, and
  `PowerLawLatentHeat` concrete strategies in `particula/gas/`
- Builder and factory for latent heat strategies
- Pure functions for non-isothermal mass transfer rate in
  `particula/dynamics/condensation/mass_transfer.py`
- `CondensationLatentHeat` strategy class extending `CondensationStrategy`
- Tracking cumulative latent heat energy released/absorbed to gas phase per step
  (returned alongside updated particle + gas, or stored as diagnostic)
- Builder and factory integration for `CondensationLatentHeat`
- Export from `particula.dynamics` and `particula.gas` namespaces
- Support for all distribution types: particle-resolved first, then discrete
  and continuous
- Comprehensive tests with mass conservation and isothermal-limit parity
- Documentation, docstrings, and usage example notebook
- Python-native implementation only (Warp/GPU is a follow-on feature)

### Out of Scope

- **Temperature evolution/feedback**: The gas-phase ambient temperature is
  assumed large enough that it does not change. We track the energy released
  but do not feed it back into T. This is a separate follow-on epic.
- **Staggered variant** (`CondensationLatentHeatStaggered`): Single
  simultaneous variant only. Staggered can be added trivially later.
- **GPU/Warp acceleration**: Python-native first. GPU translation is a
  follow-on feature after this epic ships.
- **New runnable wrapper**: Reuse existing `MassCondensation` runnable.
- **Heat capacity strategies**: Out of scope for this epic; latent heat
  strategies are sufficient.

## Dependencies

- None blocking. All required infrastructure exists:
  - `CondensationStrategy` ABC is stable
  - `get_thermal_conductivity()` exists in `particula.gas`
  - `VaporPressureStrategy` pattern provides the dependency injection template
  - Builder/factory patterns are well established

## Literature References

1. **Topping, D., & Bane, M. (2022).** *Introduction to Aerosol Modelling*,
   Eq. 2.36. Wiley. DOI: 10.1002/9781119625728
2. **Seinfeld, J. H., & Pandis, S. N. (2016).** *Atmospheric Chemistry and
   Physics*, Ch. 13, Eq. 13.3. 3rd ed. Wiley.
3. **Rogers, R. R., & Yau, M. K. (1989).** *A Short Course in Cloud Physics*.
   3rd ed. Pergamon Press. (Linear latent heat approximation)
4. **Watson, K. M. (1943).** Thermodynamics of the liquid state.
   *Ind. Eng. Chem.* 35(4), 398-406. (Power-law latent heat form)

## Latent Heat Parametrizations

### Constant (species-specific fixed value)

```
L(T) = L_ref                        [J/kg] or [J/mol]
```

Simplest form. Suitable when temperature range is narrow.

### Linear (atmospheric approximation)

```
L(T) = L_ref - c * (T - T_ref)      [J/kg]
```

For water: L_ref = 2.501e6 J/kg, c = 2.3e3 J/(kg*K), T_ref = 273.15 K.
Valid for -40 C to +40 C. Standard in cloud modeling, CCN activation, and
parcel models. Derived from dL/dT = c_p,v - c_p,l ~ -2.3 kJ/(kg*K).

### Power Law (wide temperature range)

```
L(T) = L_ref * (1 - T/T_c)^beta     [J/kg]
```

Where T_c is the critical temperature and beta ~ 0.38 for water. Captures
L -> 0 as T -> T_c. Used in engineering thermodynamics and EOS-based models.

## Phase Checklist

### Feature E5-F1: Latent Heat Strategy Pattern

- [x] **E5-F1-P1**: Create `LatentHeatStrategy` ABC and `ConstantLatentHeat`
  with tests
  - Issue: #1122 | Size: S (~60 LOC) | Status: Complete
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
  - `LinearLatentHeat(latent_heat_ref, slope, temperature_ref)`:
    `L(T) = L_ref - slope * (T - T_ref)`
  - `PowerLawLatentHeat(latent_heat_ref, critical_temperature, beta)`:
    `L(T) = L_ref * (1 - T/T_c)^beta`
  - Both handle edge cases: T > T_c clamps to 0 for power law
  - Tests: validate against known water values at 273 K, 293 K, 313 K
  - Tests: array input, scalar input, edge cases

- [ ] **E5-F1-P3**: Add latent heat builders with tests
  - File: `particula/gas/latent_heat_builders.py`
  - Follow `VaporPressureBuilder` pattern (see
    `particula/gas/vapor_pressure_builders.py` for reference)
  - `ConstantLatentHeatBuilder` — extends `BuilderABC`, required param:
    `latent_heat_ref` [J/kg]; `build()` returns `ConstantLatentHeat`
  - `LinearLatentHeatBuilder` — required params: `latent_heat_ref`, `slope`,
    `temperature_ref`; `build()` returns `LinearLatentHeat`
  - `PowerLawLatentHeatBuilder` — required params: `latent_heat_ref`,
    `critical_temperature`, `beta`; `build()` returns `PowerLawLatentHeat`
  - Each builder uses `pre_build_check()` to validate required keys
  - Tests: `particula/gas/tests/latent_heat_builders_test.py`
  - Tests: build with valid params, missing-param errors, round-trip values

- [ ] **E5-F1-P4**: Add latent heat factory and gas exports with tests
  - File: `particula/gas/latent_heat_factories.py`
  - `LatentHeatFactory` extends `StrategyFactoryABC` (from
    `particula/abc_factory.py`, 111 lines — same pattern as
    `VaporPressureFactory` in `particula/gas/vapor_pressure_factories.py`)
  - Maps `"constant"` -> `ConstantLatentHeatBuilder`, `"linear"` ->
    `LinearLatentHeatBuilder`, `"power_law"` -> `PowerLawLatentHeatBuilder`
  - Export all strategies, builders, and factory from
    `particula/gas/__init__.py`
  - Tests: `particula/gas/tests/latent_heat_factories_test.py`
  - Tests: factory creation for each type, round-trip parameter passing,
    invalid type error, import smoke tests

### Feature E5-F2: Non-Isothermal Mass Transfer Functions

- [ ] **E5-F2-P1**: Add thermal resistance factor function with tests
  - File: `particula/dynamics/condensation/mass_transfer.py` (extend existing
    file, currently 461 lines with 6 functions)
  - Function: `get_thermal_resistance_factor(diffusion_coefficient,
    latent_heat, vapor_pressure_surface, thermal_conductivity, temperature,
    molar_mass) -> float | NDArray`
  - Computes the denominator correction (Topping & Bane Eq. 2.36 denominator):
    `[D * L * p_surf / (kappa * T)] * [L / (R_specific * T) - 1] + R_specific * T`
  - Where `R_specific = R / molar_mass` (specific gas constant, J/(kg·K))
  - Note: `p_surf` is the equilibrium vapor pressure at the particle surface
    (from `calculate_pressure_delta()` partial_pressure_particle), NOT the
    partial pressure delta. This is the saturation vapor pressure modified by
    Kelvin and activity effects.
  - Note: This denominator is purely thermodynamic — it does NOT depend on
    particle size. Size effects enter only through `k_cond` (the numerator's
    first-order mass transport coefficient), which the base class already
    computes with Knudsen/transition regime corrections via
    `first_order_mass_transport()`.
  - Pure function, no class state, uses `@validate_inputs` decorator
  - Tests: `particula/dynamics/condensation/tests/mass_transfer_test.py`
    (extend existing file, currently 440 lines)
  - Tests: known values for water at 293 K, isothermal limit (L=0 should give
    `R_specific * T`), dimensional consistency, array broadcasting

- [ ] **E5-F2-P2**: Add non-isothermal mass transfer rate function with tests
  - File: `particula/dynamics/condensation/mass_transfer.py` (extend)
  - Function: `get_mass_transfer_rate_latent_heat(pressure_delta,
    first_order_mass_transport, temperature, molar_mass, latent_heat,
    thermal_conductivity, vapor_pressure_surface) -> float | NDArray`
  - Computes: `(K * Δp * M) / (R * T * thermal_factor)` where
    `thermal_factor = get_thermal_resistance_factor(...)` from E5-F2-P1
  - Note: needs `vapor_pressure_surface` to pass through to
    `get_thermal_resistance_factor()`. This is the equilibrium vapor pressure
    at the droplet surface (activity × pure_vapor_pressure × kelvin_term).
  - When `latent_heat = 0`, the thermal_factor reduces to `R_specific * T`
    and the full expression reduces exactly to `get_mass_transfer_rate()`
    (isothermal limit). This MUST be tested to machine precision.
  - Uses `@validate_inputs` decorator with same validation as
    `get_mass_transfer_rate`
  - Tests: isothermal parity (L=0 matches `get_mass_transfer_rate` to
    < 1e-15 relative), known cloud droplet growth rate for water at S=1.01
    (1% supersaturation), multi-species array shapes (1D and 2D)

- [ ] **E5-F2-P3**: Add latent heat energy release tracking function with tests
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

### Feature E5-F3: `CondensationLatentHeat` Strategy Class

- [ ] **E5-F3-P1**: Create `CondensationLatentHeat` class skeleton with tests
  - File: `particula/dynamics/condensation/condensation_strategies.py` (extend
    existing file, currently 1699 lines with `CondensationStrategy` ABC,
    `CondensationIsothermal`, and `CondensationIsothermalStaggered`)
  - Extends `CondensationStrategy` (inherits all base methods including
    `mean_free_path`, `knudsen_number`, `first_order_mass_transport`,
    `calculate_pressure_delta`, `_fill_zero_radius`, `_apply_skip_partitioning`)
  - Additional constructor parameters beyond base class:
    - `latent_heat_strategy: LatentHeatStrategy | None = None`
      (None = isothermal fallback)
    - `latent_heat: float | NDArray = 0.0` (constant fallback if no strategy)
  - Resolution logic in `_resolve_latent_heat_strategy()`:
    1. If `latent_heat_strategy` provided → use it directly
    2. Else if `latent_heat > 0` → wrap in `ConstantLatentHeat(latent_heat)`
    3. Else (both zero/None) → store None, behave as isothermal
  - Instance attribute: `last_latent_heat_energy: float = 0.0` (diagnostic,
    set during `step()`, accessible after — does NOT affect the
    `MassCondensation` runnable which only calls `step()` and `rate()`)
  - Tests: `particula/dynamics/condensation/tests/
    condensation_strategies_test.py` (extend existing, currently 1655 lines)
  - Tests: instantiation with all param combos, strategy resolution priority,
    fallback to isothermal when L=0, type errors for bad inputs

- [ ] **E5-F3-P2**: Implement `mass_transfer_rate()` and `rate()` with tests
  - `mass_transfer_rate()` follows `CondensationIsothermal.mass_transfer_rate()`
    flow (lines 761-833) but replaces the final `get_mass_transfer_rate()` call
    with `get_mass_transfer_rate_latent_heat()`:
    1. Unwrap particle/gas, fill/clip radii (same as isothermal)
    2. Compute `first_order_mass_transport` via inherited base method (already
       includes size-dependent Knudsen/transition correction — no changes)
    3. Compute `pressure_delta` via inherited `calculate_pressure_delta()`
    4. Evaluate `L = latent_heat_strategy.latent_heat(temperature)` (if
       strategy is None, L = 0 and fall through to isothermal path)
    5. Get `kappa = get_thermal_conductivity(temperature)` — this is the
       **dry air** thermal conductivity from `particula.gas.properties`
       (already exists). For aerosol in air, this is correct because the
       heat transfer occurs through the air surrounding the particle.
    6. Extract `vapor_pressure_surface` from `calculate_pressure_delta()`
       intermediate (the particle-side partial pressure). This requires
       refactoring `calculate_pressure_delta()` to optionally return the
       surface pressure, OR computing it separately in this method.
    7. Call `get_mass_transfer_rate_latent_heat(pressure_delta,
       first_order_mass_transport, temperature, molar_mass, L, kappa,
       vapor_pressure_surface)`
  - `rate()`: identical to `CondensationIsothermal.rate()` — multiply
    `mass_transfer_rate` by particle concentration, apply skip-partitioning
  - Tests: numerical parity with `CondensationIsothermal` when L=0 (< 1e-15
    relative), reduced rate when L > 0 (thermal resistance always slows
    condensation), array shapes for single and multi-species

- [ ] **E5-F3-P3**: Implement `step()` for particle-resolved with energy
  tracking and tests
  - `step()` follows `CondensationIsothermal.step()` flow (lines 922-1040)
    with two additions:
    1. Uses `mass_transfer_rate` from P2 (which already includes thermal
       correction)
    2. After computing `mass_transfer` via `get_mass_transfer()`, calls
       `get_latent_heat_energy_released(mass_transfer, L)` and stores result
       in `self.last_latent_heat_energy`
  - Return type: `Tuple[ParticleRepresentation | ParticleData, GasSpecies |
    GasData]` (same as isothermal for API compatibility — the `MassCondensation`
    runnable does NOT need modification)
  - Energy diagnostic: `self.last_latent_heat_energy` is a float attribute
    set during each `step()` call, overwritten each time, accessible after.
    It is NOT returned in the tuple to keep API compatibility.
  - Scope: particle-resolved distribution type only (each particle has
    individual mass/radius)
  - Tests: mass conservation (gas + particle total unchanged to < 1e-12),
    energy = sum(dm * L) to machine precision, sign conventions,
    single-species and multi-species particle-resolved cases

- [ ] **E5-F3-P4**: Add discrete and continuous distribution support with tests
  - Extend `step()` to handle discrete (binned) and continuous (PDF)
    distribution types
  - These use the same `get_mass_transfer()` routing (single vs multiple
    species) as isothermal — the only difference is the mass transfer rate
    computation (already handled in P2)
  - Verify array shapes for 2D mass arrays (particles × species) with
    discrete bins
  - Tests: discrete distribution with 10, 50, 100 bins; continuous
    distribution; mass conservation across all types; isothermal parity
    when L=0

- [ ] **E5-F3-P5**: Add data-only path support and parity tests
  - Ensure `ParticleData` + `GasData` input path works identically to
    `ParticleRepresentation` + `GasSpecies` legacy path
  - Follow the same pattern as `CondensationIsothermal.step()` lines
    1011-1040 for the per-particle mass division in data-only path
  - Parity test: legacy path vs data-only path with `rtol=1e-10`
  - Edge cases: zero particles (no crash), single species, very small
    particles (< MIN_PARTICLE_RADIUS_M), zero concentration particles

### Feature E5-F4: Builder, Factory, and Exports

- [ ] **E5-F4-P1**: Create `CondensationLatentHeatBuilder` with tests
  - File: `particula/dynamics/condensation/condensation_builder/
    condensation_latent_heat_builder.py`
  - Follow `CondensationIsothermalBuilder` pattern (see
    `condensation_isothermal_builder.py`, 60 lines — extends `BuilderABC` +
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

### Feature E5-F5: Validation and Integration Tests

- [ ] **E5-F5-P1**: Mass conservation and isothermal-limit integration tests
  - File: `particula/dynamics/condensation/tests/
    latent_heat_conservation_test.py`
  - Mass conservation: `abs(total_initial - total_final) / total_initial
    < 1e-12` where total = sum(gas_mass) + sum(particle_mass × concentration)
  - Test across all 3 distribution types: particle-resolved (100, 1000, 10000
    particles), discrete (10, 50, 100 bins), continuous
  - Isothermal limit: when `latent_heat=0`, `CondensationLatentHeat.step()`
    results must match `CondensationIsothermal.step()` to machine precision
    (< 1e-15 relative error on all mass arrays)
  - Test with both single-species and multi-species (3 species) setups
  - Use water vapor at T=293K, P=101325Pa as baseline physical scenario

- [ ] **E5-F5-P2**: Physical validation against known cloud droplet growth
  - File: `particula/dynamics/condensation/tests/
    latent_heat_validation_test.py`
  - Validate against Rogers & Yau (1989) Chapter 7: water droplet growth rate
    at S=1.005 (0.5% supersaturation), T=273.15K, P=101325Pa, r=10µm
  - Expected: non-isothermal dm/dt should be ~60-70% of isothermal dm/dt for
    water (thermal resistance is significant for water's large L)
  - Tolerance: < 5% relative error vs literature values (accounting for
    differences in assumed air properties)
  - Property check: non-isothermal rate <= isothermal rate for ALL positive L
    values (test with L = 100, 1000, 10000, 2.5e6 J/kg)
  - Energy bookkeeping: `abs(Q_released - sum(dm * L)) < 1e-14 * abs(Q)`
    for each step

### Feature E5-F6: Documentation and Examples

- [ ] **E5-F6-P1**: Add docstrings and Theory document update
  - Google-style docstrings for all new public classes and functions (should
    already be done per-phase, this is a sweep to catch gaps)
  - Add literature citations in module docstrings (Topping & Bane Eq. 2.36,
    Seinfeld & Pandis Eq. 13.3, Rogers & Yau Ch. 7)
  - Update or create `docs/Theory/Technical/Dynamics/Condensation_Equations.md`
    with non-isothermal mass transfer derivation linking theory to code
  - Include explanation of why size effects (Knudsen correction) are in the
    numerator only, not the denominator
  - Run docstring linter (`ruff check`, `ruff format`) to validate

- [ ] **E5-F6-P2**: Create usage example notebook
  - File: `docs/Examples/Dynamics/non_isothermal_condensation_example.py`
    (py:percent format, synced to `.ipynb` via Jupytext)
  - Follow Jupytext paired sync workflow from AGENTS.md: edit .py → lint →
    sync → execute → commit both files
  - Demonstrate all three latent heat strategies (constant, linear, power law)
  - Compare isothermal vs non-isothermal growth curves on same plot
  - Show energy tracking diagnostic (`strategy.last_latent_heat_energy`)
  - Show cloud droplet activation scenario (water at 0.5% supersaturation)
  - Use `particula` public API only (no internal imports)

- [ ] **E5-F6-P3**: Update development documentation
  - Update `adw-docs/dev-plans/README.md` with epic status
  - Update `adw-docs/dev-plans/epics/index.md`
  - Update `adw-docs/dev-plans/features/index.md`
  - Add completion notes and lessons learned

### Feature E5-F7: Warp/GPU Translation (Follow-on)

- [ ] **E5-F7-P1**: Translate latent heat pure functions to `wp.func` kernels
  - Translate `get_thermal_resistance_factor` and
    `get_mass_transfer_rate_latent_heat` to Warp `wp.func` equivalents
  - Add Warp unit tests mirroring Python tests with numerical parity
  - Follow patterns established in E3-F3 for Warp integration

- [ ] **E5-F7-P2**: Translate `CondensationLatentHeat.step()` to Warp kernel
  - Create GPU kernel for non-isothermal condensation step
  - Include energy tracking in kernel output
  - Parity test: GPU vs CPU results to within floating-point tolerance

- [ ] **E5-F7-P3**: Update development documentation for GPU feature
  - Update epic and feature docs with GPU completion status
  - Add GPU usage examples to notebook
  - Document performance comparison (CPU vs GPU)

## Critical Testing Requirements

- **No Coverage Modifications**: Keep coverage thresholds as shipped (80%).
- **Self-Contained Tests**: Attach `*_test.py` files that prove each phase is
  complete.
- **Test-First Completion**: Write and pass tests before declaring phases ready
  for review.
- **80%+ Coverage**: Every phase must ship tests that maintain at least 80%
  coverage.
- **Mass Conservation**: All modes must preserve total mass to within 1e-12
  relative tolerance.
- **Isothermal Parity**: When latent_heat=0, results must exactly match
  `CondensationIsothermal`.
- **Energy Bookkeeping**: Latent heat energy released must equal `dm * L` to
  machine precision.

## Testing Strategy

### Unit Tests
- `particula/gas/tests/latent_heat_strategies_test.py` -- strategy classes
- `particula/gas/tests/latent_heat_builders_test.py` -- builders
- `particula/gas/tests/latent_heat_factories_test.py` -- factory
- `particula/dynamics/condensation/tests/mass_transfer_test.py` -- extend with
  thermal resistance and non-isothermal rate tests
- `particula/dynamics/condensation/tests/condensation_strategies_test.py` --
  extend with `CondensationLatentHeat` tests
- `particula/dynamics/condensation/tests/condensation_factories_test.py` --
  extend with factory tests

### Integration Tests
- `particula/dynamics/condensation/tests/latent_heat_conservation_test.py` --
  mass conservation and isothermal parity
- `particula/dynamics/condensation/tests/latent_heat_validation_test.py` --
  physical validation against literature values

### Performance Tests (future, marked slow)
- GPU vs CPU parity and performance comparison (E5-F7)

## Success Metrics

1. **Mass Conservation**: Total (gas + particle) mass preserved to < 1e-12
   relative error
2. **Isothermal Parity**: L=0 results match `CondensationIsothermal` to
   machine precision (< 1e-15 relative)
3. **Physical Accuracy**: Non-isothermal rate <= isothermal rate for all
   positive L (thermal resistance always reduces growth rate)
4. **Energy Conservation**: `Q_released = sum(dm * L)` to machine precision
5. **Coverage**: Maintain 80%+ test coverage for all new code
6. **API Consistency**: Follows existing strategy/builder/factory patterns;
   works with `MassCondensation` runnable unchanged

## Pseudocode Reference

```python
# Latent heat strategy (dependency injection, follows VaporPressureStrategy)
class LatentHeatStrategy(ABC):
    @abstractmethod
    def latent_heat(self, temperature) -> float | NDArray:
        """Return latent heat of vaporization [J/kg] at given temperature."""
        ...

class ConstantLatentHeat(LatentHeatStrategy):
    def __init__(self, latent_heat_ref: float):
        self.latent_heat_ref = latent_heat_ref  # [J/kg]

    def latent_heat(self, temperature):
        return self.latent_heat_ref  # ignores temperature

class LinearLatentHeat(LatentHeatStrategy):
    def __init__(self, latent_heat_ref, slope, temperature_ref):
        self.latent_heat_ref = latent_heat_ref  # [J/kg]
        self.slope = slope                       # [J/(kg·K)]
        self.temperature_ref = temperature_ref   # [K]

    def latent_heat(self, temperature):
        return self.latent_heat_ref - self.slope * (temperature - self.temperature_ref)

class PowerLawLatentHeat(LatentHeatStrategy):
    def __init__(self, latent_heat_ref, critical_temperature, beta):
        self.latent_heat_ref = latent_heat_ref          # [J/kg]
        self.critical_temperature = critical_temperature # [K]
        self.beta = beta                                 # ~0.38 for water

    def latent_heat(self, temperature):
        ratio = np.clip(1 - temperature / self.critical_temperature, 0, None)
        return self.latent_heat_ref * ratio**self.beta


# Non-isothermal mass transfer (pure functions in mass_transfer.py)
def get_thermal_resistance_factor(
    diffusion_coefficient, latent_heat, vapor_pressure_surface,
    thermal_conductivity, temperature, molar_mass,
) -> float | NDArray:
    """Thermodynamic denominator correction (Topping & Bane Eq. 2.36).

    Note: This is purely thermodynamic — no particle size dependence.
    Size effects enter only through k_cond in the numerator.
    """
    r_specific = GAS_CONSTANT / molar_mass  # J/(kg·K)
    return (
        (diffusion_coefficient * latent_heat * vapor_pressure_surface)
        / (thermal_conductivity * temperature)
        * (latent_heat / (r_specific * temperature) - 1)
        + r_specific * temperature
    )

def get_mass_transfer_rate_latent_heat(
    pressure_delta, first_order_mass_transport, temperature, molar_mass,
    latent_heat, thermal_conductivity, vapor_pressure_surface,
) -> float | NDArray:
    """Non-isothermal mass transfer rate.

    dm/dt = (K * Δp * M) / (R * T * thermal_factor)
    When latent_heat=0, thermal_factor = R_specific * T and this reduces
    exactly to get_mass_transfer_rate().
    """
    thermal_factor = get_thermal_resistance_factor(
        diffusion_coefficient, latent_heat, vapor_pressure_surface,
        thermal_conductivity, temperature, molar_mass,
    )
    return (first_order_mass_transport * pressure_delta * molar_mass
            / (GAS_CONSTANT * temperature * thermal_factor)
            * (GAS_CONSTANT / molar_mass * temperature))
    # Simplifies to: K * Δp * M / (R * T) / thermal_factor
    # which is: isothermal_rate / thermal_factor


# CondensationLatentHeat strategy
class CondensationLatentHeat(CondensationStrategy):
    def __init__(self, ..., latent_heat_strategy=None, latent_heat=0.0):
        super().__init__(...)
        # Resolution: strategy > constant > isothermal fallback
        self._latent_heat_strategy = self._resolve_latent_heat_strategy(
            latent_heat_strategy, latent_heat
        )
        self.last_latent_heat_energy = 0.0  # diagnostic, set in step()

    def mass_transfer_rate(self, particle, gas_species, temperature, pressure):
        # Steps 1-3: identical to CondensationIsothermal
        radius = ...  # fill zeros, clip to MIN_PARTICLE_RADIUS_M
        k_cond = self.first_order_mass_transport(radius, T, P)  # inherited
        pressure_delta = self.calculate_pressure_delta(...)      # inherited

        # Step 4: latent heat correction (new)
        if self._latent_heat_strategy is None:
            return get_mass_transfer_rate(...)  # isothermal path
        L = self._latent_heat_strategy.latent_heat(temperature)
        kappa = get_thermal_conductivity(temperature)  # dry air
        p_surf = ...  # equilibrium vapor pressure at particle surface
        return get_mass_transfer_rate_latent_heat(
            pressure_delta, k_cond, temperature, molar_mass, L, kappa, p_surf
        )

    def step(self, particle, gas_species, temperature, pressure, time_step):
        mass_rate = self.mass_transfer_rate(...)
        dm = get_mass_transfer(mass_rate, time_step, ...)
        # Energy tracking (new)
        L = self._latent_heat_strategy.latent_heat(temperature)
        self.last_latent_heat_energy = float(np.sum(
            get_latent_heat_energy_released(dm, L)
        ))
        # Update particle + gas (same as isothermal step)
        ...
```

## Additional Notes

- The `MassCondensation` runnable works unchanged with this strategy since
  it only calls `strategy.step()` and `strategy.rate()`.
- The latent heat energy diagnostic (`last_latent_heat_energy`) is a stepping
  stone toward full temperature feedback in a future epic.
- The linear latent heat approximation is the most commonly used form in
  atmospheric science and should be the default recommendation for cloud
  modeling use cases.
- Warp/GPU translation (E5-F7) is deliberately scoped as a follow-on feature
  to keep the initial implementation focused on correctness.

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-02 | Initial epic creation | ADW |
| 2026-03-02 | Split E5-F1-P3 into P3 (builders) + P4 (factory+exports); split E5-F3-P3 into P3 (particle-resolved step) + P4 (discrete+continuous) + P5 (data-only parity); added missing details: function signatures, file references, thermal conductivity source, vapor_pressure_surface parameter, test tolerances, literature targets | ADW |
