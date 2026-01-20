# Feature F6: Marine Aerosol Sea Spray Aging Simulation

**Status**: Planning
**Priority**: P2
**Owners**: TBD
**Start Date**: TBD
**Target Date**: TBD
**Last Updated**: 2026-01-19
**Size**: M (1 notebook, ~300 LOC)

## Vision

Create an end-to-end simulation notebook demonstrating marine aerosol evolution
from fresh sea spray to cloud-processed aged particles. This notebook will
showcase the Binary Activity Thermodynamics (BAT) model for organic-water
interactions, organic film coating effects on hygroscopic growth, and the
transition from marine boundary layer aging to marine cloud deck formation.

Marine aerosols are among the most abundant natural aerosol sources and play a
critical role in cloud formation over oceans. Understanding how organic films
from biological activity affect sea salt hygroscopicity and cloud activation is
essential for climate modeling. This simulation demonstrates particula's
thermodynamic capabilities for mixed organic-inorganic systems.

## Scope

### In Scope

- **Two-phase simulation**:
  1. **Marine boundary layer aging**: Fresh sea spray + organic coating
     accumulation, RH cycling, hygroscopic growth/shrinkage
  2. **Marine cloud deck formation**: Activation into cloud droplets,
     supersaturation, droplet growth

- **Species**:
  - Sodium chloride (NaCl) - sea salt core
  - Marine organics (fatty acids, e.g., palmitic acid; marine SOA)
  - Water

- **Processes**:
  - Hygroscopic growth and shrinkage (RH-dependent water uptake)
  - Organic film coating effects on water uptake kinetics
  - BAT model activity coefficients for organic-water system
  - Phase separation diagnostics (liquid-liquid phase separation, LLPS)
  - Condensation/evaporation
  - Coagulation (Brownian)

- **Key physics to demonstrate**:
  - How organic coatings reduce hygroscopicity (κ reduction)
  - BAT model activity coefficients and Gibbs free energy
  - Phase separation at intermediate RH
  - Cloud activation thresholds (critical supersaturation)

- **Visualizations**:
  - Growth factor vs RH curves
  - Activity coefficients from BAT model
  - Phase diagram / separation diagnostics
  - Size distribution before/after cloud processing
  - Composition evolution (organic film thickness)

- **Beginner-friendly**: Detailed explanations of hygroscopicity, κ-Köhler
  theory, and BAT model concepts

### Out of Scope

- Full Köhler theory implementation (use simplified κ approach)
- Detailed marine biogeochemistry
- Long-range transport (focus on local MBL processes)
- Comparison to specific field campaign data

## Dependencies

- `particula.activity.bat_activity_coefficients`
- `particula.activity.phase_separation` (find_phase_separation, q_alpha)
- `particula.activity.gibbs_of_mixing`
- `particula.dynamics.MassCondensation`
- `particula.dynamics.Coagulation` (Brownian)
- `ActivityKappaParameter` or `ActivityNonIdealBinary` strategies

## Phase Checklist

- [ ] **F6-P1**: Create notebook with marine boundary layer aging simulation
  - Issue: TBD | Size: M | Status: Not Started
  - Set up notebook structure with imports and plotting style
  - Define species: NaCl core, organic film (palmitic acid or similar), water
  - Configure BAT model activity strategy for organic-water system
  - Implement RH cycling in marine boundary layer (e.g., 60% → 90% → 70%)
  - Show hygroscopic growth curves with/without organic coating
  - Demonstrate BAT activity coefficients calculation
  - Add phase separation diagnostics visualization
  - Detailed markdown explanations of κ-theory and BAT model
  - Internal consistency checks

- [ ] **F6-P2**: Add marine cloud deck formation and documentation
  - Issue: TBD | Size: M | Status: Not Started
  - Extend simulation to supersaturated conditions (~100.2-100.5% RH)
  - Show cloud droplet activation and growth
  - Compare activation for different organic coating thicknesses
  - Visualize size distribution evolution through cloud processing
  - Add "What You Learned" section with key takeaways
  - Update `docs/Examples/Simulations/index.md`
  - Update `docs/Examples/index.md` with new card

- [ ] **F6-P3**: Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Update `adw-docs/dev-plans/features/index.md`
  - Add entry to `adw-docs/dev-plans/README.md`
  - Move plan to completed/ folder

## Critical Testing Requirements

- **No Coverage Modifications**: Notebooks don't require test coverage, but any
  helper functions added to particula core must have 80%+ coverage.
- **Self-Contained**: Notebook should run end-to-end without errors.
- **Internal Consistency**: Mass conservation and thermodynamic consistency.
- **Reproducibility**: Fixed random seeds for consistent outputs.

## Testing Strategy

- Notebook will be validated by running all cells without errors
- Internal checks:
  - Water activity approaches 1.0 at high RH (dilute solution)
  - Organic coating reduces overall κ compared to pure NaCl
  - Phase separation occurs at expected RH ranges (if applicable)
  - Mass conservation within 1%
- No external `*_test.py` files needed for notebook-only feature

## Shipping Checklist

1. Update `adw-docs/dev-plans/features/index.md` with F6 entry
2. Update `docs/Examples/Simulations/index.md` with notebook link
3. Update `docs/Examples/index.md` with simulation card
4. Ensure notebook runs cleanly in fresh environment
5. Merge PR referencing this plan

## Technical Notes

### Species Properties

```python
# Sea salt (NaCl)
nacl_molar_mass = 0.05844  # kg/mol
nacl_density = 2165  # kg/m³
nacl_kappa = 1.28  # hygroscopicity parameter

# Palmitic acid (C16H32O2) - representative marine organic
palmitic_molar_mass = 0.25642  # kg/mol
palmitic_density = 853  # kg/m³
palmitic_kappa = 0.1  # low hygroscopicity

# Water
water_molar_mass = 0.018015  # kg/mol
water_density = 997  # kg/m³
```

### BAT Model Integration

```python
from particula.activity import (
    bat_activity_coefficients,
    find_phase_separation,
    gibbs_of_mixing,
)

# Calculate activity coefficients for organic-water mixture
gamma_org, gamma_water = bat_activity_coefficients(
    molar_mass_ratio=org_mass / water_mass,
    oxygen_carbon_ratio=0.125,  # palmitic acid O:C
)
```

### Key Learning Outcomes

1. How organic coatings modify sea salt hygroscopicity
2. Using the BAT model for organic-water thermodynamics
3. Understanding phase separation in mixed aerosols
4. Cloud activation thresholds for internally mixed particles
5. Marine boundary layer aerosol processing
