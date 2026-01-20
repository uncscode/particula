# Feature F7: Cloud Chamber Injection Cycles Simulation

**Status**: Completed  
**Priority**: P2  
**Owners**: ADW Workflow  
**Start Date**: 2026-01-19  
**Target Date**: TBD  
**Last Updated**: 2026-01-20  
**Completion Date**: 2026-01-20  
**Size**: L (1 notebook, ~1500 LOC)  
**Notebook**: `docs/Examples/Simulations/Notebooks/Cloud_Chamber_Cycles.ipynb`

## Vision

Create a comprehensive end-to-end simulation notebook demonstrating cloud
droplet activation and deactivation cycles in a rectangular cloud chamber. This
notebook will showcase particle-resolved speciated mass tracking through
multiple humidity cycles, demonstrating how water mass accumulates
preferentially in larger particles and how different seed compositions (varying
hygroscopicity) respond to cloud processing.

Cloud chambers are essential tools for studying aerosol-cloud interactions.
Understanding how particles of different sizes and compositions activate into
cloud droplets, and how repeated cycling affects the particle population, is
critical for interpreting chamber experiments and parameterizing cloud
microphysics in models. This simulation demonstrates particula's full
capabilities: particle-resolved tracking, wall loss, dilution, injection, and
condensation.

## Scope

### In Scope

- **4 complete cycles**:
  1. Humid injection → Droplet activation/growth (5+ μm diameter)
  2. Dry air dilution → Droplet deactivation/shrinkage
  3. Repeat for 4 total cycles

- **Multiple seed scenarios** (to show interesting physics):
  - Scenario A: Ammonium sulfate seeds (high κ ≈ 0.61) - different sizes
  - Scenario B: Sucrose seeds (moderate κ ≈ 0.1) - different sizes
  - Scenario C: Mixed population (AS + sucrose) - competition for water vapor

- **Environmental conditions**:
  - Activation phase: ~100.4% RH (0.4% supersaturation)
  - Deactivation phase: ~60-70% RH (droplets shrink but don't fully dry)
  - Target droplet size: 5+ μm diameter at peak activation

- **Processes**:
  - Condensation/evaporation (isothermal, humidity-driven)
  - Rectangular chamber wall loss
  - Dilution (clean dry air injection)
  - Water vapor injection (humid air pulses)

- **Particle representation**: Particle-resolved speciated mass
  - Track individual particle histories through all 4 cycles
  - Species: seed material (AS or sucrose) + water

- **Key physics to demonstrate**:
  - Size-dependent activation (Kelvin effect)
  - Hygroscopicity (κ) differences between seed types
  - Mass accumulation in larger particles over cycles
  - Wall loss preferentially removes larger droplets
  - Composition tracking (water mass fraction evolution)

- **Visualizations**:
  - Individual particle trajectories (size vs time)
  - Size distribution evolution through cycles
  - Water mass fraction per particle
  - Activated fraction vs dry diameter
  - Wall loss rates by size
  - Comparison between seed types

- **Chamber**: Rectangular geometry with realistic dimensions

- **Beginner-friendly**: Detailed explanations of cloud activation, Köhler
  theory basics, and chamber experiment design

### Out of Scope

- Full Köhler curve calculations (use κ-Köhler approximation)
- Temperature changes during cycles (isothermal assumption)
- Collision/coalescence of droplets
- Entrainment mixing
- Comparison to specific chamber experiments

## Dependencies

### Core Classes (all available in `particula`)

- `particula.particles.ParticleResolvedSpeciatedMass` — distribution strategy
- `particula.particles.ParticleResolvedSpeciatedMassBuilder` — builder pattern
- `particula.dynamics.MassCondensation` — runnable for condensation/evaporation
- `particula.dynamics.WallLoss` — runnable wrapping wall loss strategies
- `particula.dynamics.RectangularWallLossStrategy` — rectangular chamber geometry
- `particula.dynamics.RectangularWallLossBuilder` — builder for rectangular
- `particula.dynamics.get_dilution_rate` — dilution rate calculation
- `particula.particles.ActivityKappaParameter` — κ-based hygroscopic growth
- `particula.particles.ActivityKappaParameterBuilder` — builder for κ activity

### Optional (for advanced scenarios)

- `particula.dynamics.CondensationIsothermalStaggered` — staggered ODE stepping
  for improved mass conservation (E1 feature, already shipped)

## Phase Checklist

- [x] **F7-P1**: Create notebook structure and single-cycle simulation (~150 LOC)
  - Issue: #896 | Size: M | Status: Completed
  - Set up notebook with imports, plotting style, chamber parameters
  - Define seed species (ammonium sulfate, sucrose) with κ values
  - Create particle-resolved speciated mass representation using builder
  - Configure rectangular chamber wall loss via `RectangularWallLossBuilder`
  - Implement single activation-deactivation cycle
  - Show droplet growth to 5+ μm at 100.4% RH
  - Visualize individual particle trajectories
  - Add detailed markdown explanations for beginners
  - Internal consistency checks (mass conservation)

- [x] **F7-P2**: Extend to 4 cycles with multiple scenarios (~1303 LOC added)
  - Issue: #897 | Size: L | Status: Completed
  - Implemented `apply_particle_dilution()` helper for particle-resolved dilution
  - Implemented `run_cycle()` helper for activation-deactivation cycles
  - Implemented `run_multi_cycle()` wrapper for N-cycle simulations
  - Created Scenario A: Ammonium sulfate only (κ=0.61, 5 sizes)
  - Created Scenario B: Sucrose only (κ=0.10, 5 sizes)
  - Created Scenario C: Mixed AS + sucrose population (10 particles)
  - Comprehensive visualizations (13 sections total):
    - Particle size trajectories over 4 cycles per scenario
    - Activated fraction vs dry diameter comparison
    - Water mass fraction evolution
    - Mass accumulation in larger particles
    - Comparison overlay plot across all scenarios
  - Added internal validation checks for helper functions and history structure
  - Demonstrated key physics:
    - κ-Köhler theory: Higher κ → lower critical supersaturation
    - Size-dependent activation: Larger particles activate first
    - Competition for water vapor in mixed populations
    - Mass preferentially accumulates in larger particles

- [x] **F7-P3**: Add comprehensive visualizations and documentation (~150 LOC)
  - Issue: TBD | Size: M | Status: Completed
  - Add wall loss analysis (size-dependent losses)
  - Create summary comparison plots (all scenarios)
  - Add water mass fraction evolution visualization
  - Write "What You Learned" section with key takeaways
  - Add concept boxes for Köhler theory, κ-theory
  - Update `docs/Examples/Simulations/index.md`
  - Update `docs/Examples/index.md` with new card

- [x] **F7-P4**: Update development documentation
  - Issue: TBD | Size: XS | Status: Completed
  - Update `adw-docs/dev-plans/features/index.md`
  - Add entry to `adw-docs/dev-plans/README.md`
  - Move plan to completed/ folder

## Critical Testing Requirements

- **No Coverage Modifications**: Notebooks don't require test coverage, but any
  helper functions added to particula core must have 80%+ coverage.
- **Self-Contained**: Notebook should run end-to-end without errors.
- **Internal Consistency**: Mass conservation, droplet activation thresholds.
- **Reproducibility**: Fixed random seeds for consistent outputs.

## Testing Strategy

- Notebook will be validated by running all cells without errors
- Internal checks:
  - Total mass (seed + water) conserved within 1% (accounting for wall loss)
  - Droplets reach 5+ μm diameter at peak supersaturation
  - Higher κ seeds activate at lower supersaturation
  - Larger dry particles activate before smaller ones
  - Wall loss increases with droplet size
- No external `*_test.py` files needed for notebook-only feature

## Shipping Checklist

1. Update `adw-docs/dev-plans/features/index.md` with F7 entry
2. Update `docs/Examples/Simulations/index.md` with notebook link
3. Update `docs/Examples/index.md` with simulation card
4. Ensure notebook runs cleanly in fresh environment
5. Merge PR referencing this plan

## Technical Notes

### Chamber Configuration

```python
import particula as par

# Rectangular cloud chamber dimensions
chamber_length = 1.0  # m
chamber_width = 0.5   # m
chamber_height = 0.5  # m
chamber_volume = chamber_length * chamber_width * chamber_height  # m³

# Build wall loss strategy using builder pattern
wall_loss_strategy = (
    par.dynamics.RectangularWallLossBuilder()
    .set_chamber_dimensions((chamber_length, chamber_width, chamber_height))
    .set_wall_eddy_diffusivity(0.001)  # m²/s
    .set_distribution_type("particle_resolved")
    .build()
)

# Wrap in runnable for execution
wall_loss = par.dynamics.WallLoss(wall_loss_strategy=wall_loss_strategy)
```

### Seed Species Properties

```python
# Ammonium sulfate (NH4)2SO4
as_molar_mass = 0.13214  # kg/mol
as_density = 1770  # kg/m³
as_kappa = 0.61  # high hygroscopicity

# Sucrose C12H22O11
sucrose_molar_mass = 0.34230  # kg/mol
sucrose_density = 1587  # kg/m³
sucrose_kappa = 0.10  # moderate hygroscopicity

# Water
water_molar_mass = 0.018015  # kg/mol
water_density = 997  # kg/m³
```

### Activity Strategy Setup

```python
# Build κ-based activity strategy for ammonium sulfate
as_activity = (
    par.particles.ActivityKappaParameterBuilder()
    .set_kappa(as_kappa)
    .set_density(as_density)
    .set_molar_mass(as_molar_mass)
    .set_water_index(1)  # water is second species
    .build()
)

# Build κ-based activity strategy for sucrose
sucrose_activity = (
    par.particles.ActivityKappaParameterBuilder()
    .set_kappa(sucrose_kappa)
    .set_density(sucrose_density)
    .set_molar_mass(sucrose_molar_mass)
    .set_water_index(1)
    .build()
)
```

### Particle Representation Setup

```python
# Create particle-resolved speciated mass distribution
distribution_strategy = par.particles.ParticleResolvedSpeciatedMassBuilder().build()

# Build aerosol with particles of different dry diameters
# Each particle tracks [seed_mass, water_mass]
aerosol = (
    par.AerosolBuilder()
    .set_distribution_strategy(distribution_strategy)
    .set_activity_strategy(as_activity)
    # ... additional configuration
    .build()
)
```

### Cycle Implementation (F7-P2 Shipped)

The following helper functions were implemented in F7-P2 (#897):

```python
def apply_particle_dilution(
    particle_mass: np.ndarray,
    dilution_coefficient: float,
    dt: float,
) -> np.ndarray:
    """Apply dilution to particle masses for particle-resolved simulations.
    
    For particle-resolved distributions, dilution reduces total particle
    number/mass proportionally. Implemented as exponential decay.
    
    Args:
        particle_mass: Array of particle masses (n_particles, n_species).
        dilution_coefficient: Dilution rate coefficient in s⁻¹.
        dt: Time step in seconds.
        
    Returns:
        Updated particle masses after dilution.
    """
    dilution_factor = np.exp(-dilution_coefficient * dt)
    return particle_mass * dilution_factor


def run_cycle(
    aerosol: par.Aerosol,
    condensation: par.dynamics.MassCondensation,
    wall_loss: par.dynamics.WallLoss,
    humid_duration: int = 30,
    dry_duration: int = 60,
    dilution_coefficient: float = 0.01,
    rh_humid: float = 1.004,
    rh_dry: float = 0.65,
    water_index: int = 2,
) -> tuple[par.Aerosol, list[dict]]:
    """Run one activation-deactivation cycle with humid/dry phases."""
    # Implementation uses set_water_activity() helper and
    # apply_particle_dilution() during dry phase
    ...


def run_multi_cycle(
    aerosol: par.Aerosol,
    condensation: par.dynamics.MassCondensation,
    wall_loss: par.dynamics.WallLoss,
    n_cycles: int = 4,
    humid_duration: int = 30,
    dry_duration: int = 60,
    dilution_coefficient: float = 0.01,
    rh_humid: float = 1.004,
    rh_dry: float = 0.65,
    water_index: int = 2,
) -> tuple[par.Aerosol, list[dict]]:
    """Run N cycles sequentially with cumulative time tracking."""
    # Uses copy.deepcopy() to avoid mutating original aerosol
    ...
```

**Key Implementation Notes:**
- Dilution uses mass-based exponential decay (not concentration-based)
- `run_cycle()` modifies `aerosol.particle_mass` directly during dry phase
- History records `{"time": t, "phase": phase, "masses": masses_copy.copy()}`
- Cumulative time tracking across cycles for proper visualization

### Key Learning Outcomes

1. How supersaturation drives cloud droplet activation
2. Size-dependent activation (Kelvin effect / curvature)
3. Hygroscopicity (κ) controls critical supersaturation
4. Mass accumulates preferentially in larger particles
5. Wall loss in chambers is size-dependent
6. Particle-resolved tracking through complex processes
7. Competition for water vapor in mixed populations

## Change Log

| Date | Author | Change |
|------|--------|--------|
| 2026-01-19 | ADW | Initial plan created |
| 2026-01-19 | ADW | Added builder patterns, explicit dependencies, LOC estimates |
| 2026-01-19 | ADW | F7-P1 completed: Notebook structure and single-cycle simulation (#896) |
| 2026-01-20 | ADW | F7-P2 completed: 4 cycles with 3 scenarios, ~1303 LOC added (#897) |
