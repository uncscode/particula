# Feature F5: Wildfire Plume Evolution Simulation

**Status**: Planning
**Priority**: P2
**Owners**: TBD
**Start Date**: TBD
**Target Date**: TBD
**Last Updated**: 2026-01-19
**Size**: M (2 notebooks, ~400 LOC total)

## Vision

Create a comprehensive end-to-end simulation notebook demonstrating wildfire
smoke plume evolution from near-source emission to regional transport. This
notebook will showcase particula's ability to model complex multi-physics
aerosol dynamics including high-temperature condensation, multi-mechanism
coagulation (Brownian + turbulent shear + sedimentation), and plume dilution.

Wildfire smoke is a critical air quality and climate concern. Understanding how
smoke particles evolve helps researchers interpret field measurements, improve
emission inventories, and predict downwind impacts. This simulation bridges the
gap between combustion science and atmospheric modeling.

## Scope

### In Scope

- **Two simulation scenarios in one notebook**:
  1. **Near-source plume** (seconds to minutes): Hot emissions cooling rapidly,
     intense coagulation, initial particle growth
  2. **Regional transport** (hours): Dilution, continued aging, size
     distribution evolution

- **Multi-stage temperature profile**: Hot core (~800-1000 K) → warm plume
  (~400-500 K) → ambient (~298 K)

- **Species**:
  - Black carbon / soot (non-volatile core)
  - Organic aerosol (semi-volatile, temperature-dependent partitioning)
  - Optionally: inorganic species (potassium salts as biomass burning tracers)

- **Processes**:
  - Condensation with temperature-dependent vapor pressures
  - Brownian coagulation (dominant for small particles)
  - Turbulent shear coagulation (energetic plume environment)
  - Sedimentation (gravitational settling of large particles/aggregates)
  - Dilution (plume mixing with background air)

- **Particle representation**: Speciated mass bins or particle-resolved

- **Visualizations**:
  - Size distribution evolution (dN/dlogDp vs time)
  - Mass concentration by species
  - Temperature and dilution profiles
  - Coagulation kernel contributions

- **Beginner-friendly**: Detailed markdown explanations, concept boxes, step-by-
  step code walkthrough

### Out of Scope

- Gas-phase chemistry (oxidation of VOCs)
- Optical properties calculations
- Comparison to specific field campaign data
- 3D plume dispersion modeling

## Dependencies

- `particula.dynamics.Coagulation` with combined strategies
- `particula.dynamics.MassCondensation`
- `particula.dynamics.dilution` module
- `BrownianCoagulationStrategy`
- `TurbulentShearCoagulationStrategy`
- `SedimentationCoagulationStrategy`
- `CombineCoagulationStrategy` or `CombineCoagulationStrategyBuilder`

## Phase Checklist

- [ ] **F5-P1**: Create notebook structure and near-source simulation
  - Issue: TBD | Size: M | Status: Not Started
  - Set up notebook with imports, plotting style, species definitions
  - Define multi-stage temperature profile (cooling function)
  - Implement near-source simulation (0-60 seconds)
  - Configure combined coagulation (Brownian + Shear + Sedimentation)
  - Add condensation with temperature-dependent vapor pressure
  - Visualize initial particle growth and coagulation
  - Add detailed markdown explanations for beginners
  - Internal consistency checks (mass conservation, number concentration)

- [ ] **F5-P2**: Add regional transport simulation and documentation
  - Issue: TBD | Size: M | Status: Not Started
  - Extend simulation to regional transport timescales (hours)
  - Add dilution process (plume mixing with clean air)
  - Show size distribution evolution over long timescales
  - Compare coagulation mechanism contributions
  - Add summary visualizations (before/after comparisons)
  - Add "What You Learned" section and key takeaways
  - Update `docs/Examples/Simulations/index.md`
  - Update `docs/Examples/index.md` with new card

- [ ] **F5-P3**: Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Update `adw-docs/dev-plans/features/index.md`
  - Add entry to `adw-docs/dev-plans/README.md`
  - Move plan to completed/ folder

## Critical Testing Requirements

- **No Coverage Modifications**: Notebooks don't require test coverage, but any
  helper functions added to particula core must have 80%+ coverage.
- **Self-Contained**: Notebook should run end-to-end without errors.
- **Internal Consistency**: Mass conservation checks within notebook.
- **Reproducibility**: Fixed random seeds for consistent outputs.

## Testing Strategy

- Notebook will be validated by running all cells without errors
- Internal checks:
  - Total mass conservation (gas + particle) within 1%
  - Number concentration decreases monotonically (coagulation)
  - Temperature profile follows expected cooling curve
- No external `*_test.py` files needed for notebook-only feature

## Shipping Checklist

1. Update `adw-docs/dev-plans/features/index.md` with F5 entry
2. Update `docs/Examples/Simulations/index.md` with notebook link
3. Update `docs/Examples/index.md` with simulation card
4. Ensure notebook runs cleanly in fresh environment
5. Merge PR referencing this plan

## Technical Notes

### Temperature Profile

```python
def plume_temperature(time, T_initial=1000, T_ambient=298, tau=30):
    """Multi-stage cooling: exponential decay to ambient."""
    return T_ambient + (T_initial - T_ambient) * np.exp(-time / tau)
```

### Combined Coagulation Setup

```python
coag_strategy = par.dynamics.CombineCoagulationStrategyBuilder()
    .add_strategy(par.dynamics.BrownianCoagulationBuilder().build())
    .add_strategy(par.dynamics.TurbulentShearCoagulationBuilder()
        .set_turbulent_dissipation(0.1)  # m²/s³, energetic plume
        .build())
    .add_strategy(par.dynamics.SedimentationCoagulationBuilder().build())
    .build()
```

### Key Learning Outcomes

1. How temperature affects vapor pressures and condensation rates
2. Why different coagulation mechanisms dominate at different scales
3. How dilution affects particle concentrations and composition
4. Interpreting size distribution evolution in smoke plumes
