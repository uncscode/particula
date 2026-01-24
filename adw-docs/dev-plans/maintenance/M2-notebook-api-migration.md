# Maintenance M2: Jupyter Notebook API Migration

**ID:** M2
**Priority:** P2
**Status:** Planning
**Last Updated:** 2026-01-21

## Vision

Migrate all documentation Jupyter notebooks to use the current API patterns
(builders, factories, `get_*` methods) and update descriptions for clarity.
Ensure all notebooks pass `run_notebook` validation without errors (except
intentional error demonstrations which should be caught and displayed).

## Scope

**Total Notebooks:** 46 notebooks across docs/

**Categories:**
1. **End-to-End Simulations** (5 notebooks) - Large, complex workflows
2. **Dynamics Examples** (18 notebooks) - Coagulation, condensation, wall loss
3. **Component Tutorials** (14 notebooks) - Aerosol, gas, particles, equilibria
4. **Theory/Technical** (9 notebooks) - Properties, DNS comparisons

**Affected Directories:**
- `docs/Examples/Simulations/Notebooks/`
- `docs/Examples/Dynamics/`
- `docs/Examples/Chamber_Wall_Loss/Notebooks/`
- `docs/Examples/Aerosol/`
- `docs/Examples/Gas_Phase/Notebooks/`
- `docs/Examples/Particle_Phase/Notebooks/`
- `docs/Examples/Equilibria/Notebooks/`
- `docs/Examples/Nucleation/Notebooks/`
- `docs/Theory/Technical/`
- `docs/Theory/Accelerating_Python/`

## Current State Assessment

Initial `run_notebook` validation identified these failure categories:

| Category | Issue | Example |
|----------|-------|---------|
| Import errors | `species_density` module moved | `equilibria_part1.ipynb`, `activity_part1.ipynb` |
| Module not found | `particula.activity.species_density` | `Organic_Partitioning_and_Coagulation.ipynb` |
| API changes | `get_pure_vapor_pressure` signature | `Wall_Loss_Tutorial.ipynb` |
| Shape errors | Array dimension mismatches | `Condensation_1_Bin.ipynb` |
| Timeout | Simulations too slow | `Soot_Formation_in_Flames.ipynb` |

**Passing notebooks (sample):**
- `Aerosol_Tutorial.ipynb`
- `Coagulation_1_PMF_Pattern.ipynb`
- `Aerosol_Distributions.ipynb`
- `Vapor_Pressure.ipynb`

## Migration Patterns

### Pattern 1: species_density Import Migration

The `species_density` module was moved from `particula.activity` to
`particula.particles.properties.organic_density_module`. Update imports as:

```python
# OLD
from particula.activity import species_density
density = species_density.organic_array(
    molar_mass, oxygen2carbon, hydrogen2carbon=None, nitrogen2carbon=None
)

# NEW
from particula.particles.properties.organic_density_module import (
    get_organic_density_array,
)
density = get_organic_density_array(
    molar_mass=molar_mass,
    oxygen2carbon=oxygen2carbon,
    hydrogen2carbon=None,
    nitrogen2carbon=None,
)
```

For single values, use `get_organic_density_estimate` instead of `organic_array`.

### Pattern 2: Builder Pattern Updates
```python
# OLD (direct instantiation)
strategy = CondensationIsothermal(molar_mass=0.12, ...)

# NEW (builder pattern)
strategy = (
    par.dynamics.CondensationIsothermalBuilder()
    .set_molar_mass(0.12, "kg/mol")
    .set_diffusion_coefficient(1e-5, "m^2/s")
    .build()
)
```

### Pattern 3: Factory Pattern Updates
```python
# OLD
vapor_pressure = VaporPressureAntoine(a=7.838, b=1558.19, c=196.881)

# NEW
vapor_pressure = par.gas.VaporPressureFactory().get_strategy(
    "antoine", {"a": 7.838, "b": 1558.19, "c": 196.881}
)
```

### Pattern 4: Error Display (Teaching Notebooks)
For notebooks that intentionally demonstrate errors:
```python
# Wrap in try/except to show error without stopping execution
try:
    # Code that should error
    result = invalid_operation()
except SomeError as e:
    print(f"Expected error: {e}")
```

## Dependencies

### Upstream Dependencies
- Activity/Equilibria refactor (E2) may change APIs further - coordinate timing
- Any pending API changes in `particula.activity` or `particula.equilibria`

### Downstream Dependencies
- Documentation site build (mkdocs)
- User tutorials and examples

## Phase Checklist

### CI/CD Infrastructure (Prerequisite)

- [x] **M2-P0:** Add notebook validation CI workflow
  - Created `.github/workflows/notebooks.yml` - triggers only on notebook changes
  - Created `.github/scripts/validate_notebooks.py` - validation script
  - Features:
    - Detects changed notebooks via git diff
    - Phase 1: Syntax validation (ast.parse on code cells)
    - Phase 2: Execution validation with 5-minute timeout
    - Timeout = skip (not fail) for long-running simulations
  - Size: S | Status: Complete

### End-to-End Simulations (One per Phase - Large Updates)

- [ ] **M2-P1:** Cloud_Chamber_Cycles.ipynb - Split and update
  - Current state: Single file with Part 1 (single cycle) + Part 2 (4-cycle
    multi-scenario comparison with 3 seed types)
  - Split into two notebooks:
    - `Cloud_Chamber_Single_Cycle.ipynb` - Sections 1-8 (setup through single
      cycle simulation and visualization)
    - `Cloud_Chamber_Multi_Cycle.ipynb` - Part 2 (sections 9+) with 4-cycle
      comparison across ammonium sulfate, sucrose, and mixed seed scenarios
  - Update API calls to current builder/factory patterns
  - Ensure single-cycle version runs within 5-min CI timeout
  - Multi-cycle version may timeout (acceptable - skip in CI, passes syntax)
  - Update cross-references between the two notebooks
  - Size: M | Status: Not Started

- [ ] **M2-P2:** Soot_Formation_in_Flames.ipynb
  - Migrate API calls, fix timeout (optimize or mark slow)
  - Update descriptions and learning objectives
  - Size: L | Status: Not Started

- [ ] **M2-P3:** Cough_Droplets_Partitioning.ipynb
  - Migrate API calls, update descriptions
  - Size: L | Status: Not Started

- [ ] **M2-P4:** Organic_Partitioning_and_Coagulation.ipynb
  - Fix `species_density` import, migrate API calls
  - Size: L | Status: Not Started

- [ ] **M2-P5:** Biomass_Burning_Cloud_Interactions.ipynb
  - Migrate API calls, update descriptions
  - Size: L | Status: Not Started

### Dynamics - Coagulation

- [ ] **M2-P6:** Coagulation main pattern notebooks
  - `docs/Examples/Dynamics/Coagulation/Coagulation_1_PMF_Pattern.ipynb` (verify)
  - `docs/Examples/Dynamics/Coagulation/Coagulation_3_Particle_Resolved_Pattern.ipynb`
  - `docs/Examples/Dynamics/Coagulation/Coagulation_4_Compared.ipynb`
  - Size: S | Status: Not Started

- [ ] **M2-P7:** Coagulation charge notebooks
  - `docs/Examples/Dynamics/Coagulation/Charge/Coagulation_with_Charge_objects.ipynb`
  - `docs/Examples/Dynamics/Coagulation/Charge/Coagulation_with_Charge_functional.ipynb`
  - Size: S | Status: Not Started

- [ ] **M2-P8:** Coagulation functional notebooks
  - `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_1_PMF.ipynb`
  - `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_2_PDF.ipynb`
  - `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_3_compared.ipynb`
  - `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_4_ParticleResolved.ipynb`
  - Size: S | Status: Not Started

### Dynamics - Condensation

- [ ] **M2-P9:** Condensation core notebooks
  - `docs/Examples/Dynamics/Condensation/Condensation_1_Bin.ipynb` - migrated to
    `CondensationIsothermalBuilder`; shape mismatch persists in plotting/execution
    (rate vs aerosol/bin shapes). Needs another pass to align shapes and rerun
    `run_notebook` with executed outputs saved.
  - `docs/Examples/Dynamics/Condensation/Condensation_2_MassBin.ipynb` - migrated
    to builder/factory patterns; executed outputs pending save (writeExecuted
    overwrite currently blocked).
  - `docs/Examples/Dynamics/Condensation/Condensation_3_MassResolved.ipynb` - mass
    edges fixed; uses builder/factory; executed outputs pending save (writeExecuted
    overwrite currently blocked).
  - Size: S | Status: In Progress

- [ ] **M2-P10:** Condensation advanced notebooks
  - `docs/Examples/Dynamics/Condensation/Staggered_Condensation_Example.ipynb`
  - `docs/Examples/Dynamics/Customization/Adding_Particles_During_Simulation.ipynb`
  - Size: S | Status: Not Started

### Dynamics - Wall Loss

- [ ] **M2-P11:** Wall Loss tutorial and strategies
  - `docs/Examples/Chamber_Wall_Loss/Notebooks/Wall_Loss_Tutorial.ipynb` (verify - may be current)
  - `docs/Examples/Chamber_Wall_Loss/Notebooks/Spherical_Wall_Loss_Strategy.ipynb`
  - `docs/Examples/Chamber_Wall_Loss/Notebooks/Rectangular_Wall_Loss_Strategy.ipynb`
  - Size: S | Status: Not Started

- [ ] **M2-P12:** Wall Loss builders and simulation
  - `docs/Examples/Chamber_Wall_Loss/Notebooks/wall_loss_builders_factory.ipynb`
  - `docs/Examples/Chamber_Wall_Loss/Notebooks/Chamber_Forward_Simulation.ipynb`
  - Size: S | Status: Not Started

### Component Tutorials - Gas Phase

- [ ] **M2-P13:** Gas Phase notebooks
  - `docs/Examples/Gas_Phase/Notebooks/AtmosphereTutorial.ipynb`
  - `docs/Examples/Gas_Phase/Notebooks/Gas_Species.ipynb`
  - `docs/Examples/Gas_Phase/Notebooks/Vapor_Pressure.ipynb` (verify)
  - Size: S | Status: Not Started

### Component Tutorials - Aerosol and Particle Phase

- [ ] **M2-P14:** Aerosol tutorials
  - `docs/Examples/Aerosol/Aerosol_Tutorial.ipynb` (verify)
  - `docs/Examples/Particle_Phase/Notebooks/Aerosol_Distributions.ipynb` (verify)
  - `docs/Examples/Particle_Phase/Notebooks/Distribution_Tutorial.ipynb`
  - Size: S | Status: Not Started

- [ ] **M2-P15:** Particle representation notebooks
  - `docs/Examples/Particle_Phase/Notebooks/Particle_Representation_Tutorial.ipynb`
  - `docs/Examples/Particle_Phase/Notebooks/Particle_Surface_Tutorial.ipynb`
  - Size: S | Status: Not Started

- [ ] **M2-P16:** Activity tutorials
  - `docs/Examples/Particle_Phase/Notebooks/Activity_Tutorial.ipynb`
  - `docs/Examples/Particle_Phase/Notebooks/Functional/Activity_Functions.ipynb`
  - Size: S | Status: Not Started

### Component Tutorials - Equilibria

- [ ] **M2-P17:** Equilibria notebooks - fix species_density imports
  - `docs/Examples/Equilibria/Notebooks/equilibria_part1.ipynb`
  - `docs/Examples/Equilibria/Notebooks/activity_part1.ipynb`
  - Apply Pattern 1 migration for `species_density` → `get_organic_density_array`
  - Size: S | Status: Not Started

### Component Tutorials - Nucleation

- [ ] **M2-P18:** Nucleation notebook
  - `docs/Examples/Nucleation/Notebooks/Custom_Nucleation_Single_Species.ipynb`
  - Size: XS | Status: Not Started

### Theory/Technical Notebooks

- [ ] **M2-P19:** Theory - Properties notebooks
  - `docs/Theory/Technical/Properties/mean_free_path.ipynb`
  - `docs/Theory/Technical/Properties/dynamic_viscosity.ipynb`
  - Size: XS | Status: Not Started

- [ ] **M2-P20:** Theory - Dynamics notebooks
  - `docs/Theory/Technical/Dynamics/ionparticle_coagulation.ipynb`
  - Size: XS | Status: Not Started

- [ ] **M2-P21:** Theory - DNS Cloud Droplet Coagulation (4 of 5)
  - `docs/Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/DNS_Fluid_and_Particle_Properties_Comparison.ipynb`
  - `docs/Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/DNS_Horizontal_Velocity_Comparison.ipynb`
  - `docs/Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/DNS_Kernel_Comparison.ipynb`
  - `docs/Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/DNS_Radial_Distribution_Comparison.ipynb`
  - Note: These 4 notebooks share common patterns - batch update is efficient
  - Size: S | Status: Not Started

- [ ] **M2-P22:** Theory - DNS Radial Velocity + Accelerating Python
  - `docs/Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/DNS_Radial_Relative_Velocity_Comparison.ipynb`
  - `docs/Theory/Accelerating_Python/Details/Taichi_Exploration.ipynb`
  - Size: S | Status: Not Started

### Documentation Update

- [ ] **M2-P23:** Update development documentation
  - Update index.md and README.md with final status
  - Add completion notes and lessons learned
  - Document migration patterns for future reference
  - Size: XS | Status: Not Started

## Critical Testing Requirements

- **CI Validation**: `.github/workflows/notebooks.yml` validates on every PR
- **Validation Tool**: Use `run_notebook` tool for local development
- **Timeout Handling**: 5-minute timeout in CI; notebooks exceeding this are
  skipped (not failed) but must pass syntax validation
- **Syntax Check**: All code cells must parse with `ast.parse()` even if
  execution times out
- **Error Display**: Notebooks with intentional errors must catch and display
  without throwing (use try/except pattern)
- **No Silent Failures**: All notebooks must either pass or have documented
  expected errors

## Testing Strategy

### Pre-Phase Validation (REQUIRED)

Before starting each phase, run `run_notebook` on all listed notebooks to
identify which actually need updates. Some notebooks marked as "verify" may
already pass and require no changes.

```bash
# Check current state before starting work
run_notebook({notebookPath: "path/to/notebook.ipynb", timeout: 120})
```

If a notebook passes without changes, mark it as verified in the phase
completion notes but still include it in the PR for documentation.

### Per-Phase Validation
```bash
# Run single notebook and save outputs (required for docs)
run_notebook({notebookPath: "path/to/notebook.ipynb", timeout: 120, writeExecuted: "path/to/notebook.ipynb"})

# Run directory batch and save outputs
run_notebook({notebookPath: "docs/Examples/Dynamics/Coagulation/", recursive: true, writeExecuted: "inplace"})
```

**Important:** Notebook outputs must be saved because they are rendered as
documentation on the website. The `writeExecuted` parameter saves the executed
notebook (with all cell outputs, plots, and figures) back to the file. Always
commit the executed notebooks with their outputs.

### Acceptance Criteria per Notebook
1. `run_notebook` passes (no uncaught errors)
2. Uses current builder/factory patterns
3. No deprecated imports
4. Descriptions are clear and accurate
5. **Cell outputs saved** - All outputs (text, plots, figures) are committed

## Size Estimates

| Phase | Description | Size | Notebooks |
|-------|-------------|------|-----------|
| P0 | CI setup | S | - |
| P1 | Cloud chamber split | M | 1 → 2 |
| P2-P5 | End-to-end simulations | L each | 1 each |
| P6-P8 | Coagulation (3 phases) | S each | 3, 2, 4 |
| P9-P10 | Condensation (2 phases) | S each | 3, 2 |
| P11-P12 | Wall Loss (2 phases) | S each | 3, 2 |
| P13 | Gas Phase | S | 3 |
| P14-P16 | Particle Phase (3 phases) | S each | 3, 2, 2 |
| P17 | Equilibria | S | 2 |
| P18 | Nucleation | XS | 1 |
| P19-P22 | Theory (4 phases) | XS-S | 2, 1, 4, 2 |
| P23 | Documentation | XS | - |

**Total: 24 phases (P0-P23), 46 notebooks**

## Success Criteria

- [x] CI workflow validates notebooks on PRs (`.github/workflows/notebooks.yml`)
- [ ] All 46 notebooks pass syntax validation (ast.parse)
- [ ] All notebooks either pass execution or timeout gracefully (5 min limit)
- [ ] No deprecated API patterns remain
- [ ] All imports use current module structure
- [ ] Error demonstrations use try/except pattern
- [ ] Descriptions updated for clarity where needed

## References

- CI workflow: `.github/workflows/notebooks.yml`
- Validation script: `.github/scripts/validate_notebooks.py`
- Reference notebook: `Cloud_Chamber_Cycles.ipynb` (recently updated)
- Builder patterns: `adw-docs/architecture/architecture_guide.md`
- Current API: `particula/__init__.py` exports

## Notes

- End-to-end simulations are done one at a time due to large scope
- Batch phases group similar notebooks for efficiency
- Some notebooks may need optimization for timeout issues (mark as slow or
  reduce iteration counts for CI)
- Coordinate with E2 (Activity/Equilibria refactor) if it lands during this
  maintenance work

---

**Instructions for ADW Workflow:**
When generating issues from this maintenance plan:
1. Create phases sequentially - P1-P5 one at a time (large end-to-end simulations)
2. Phases P6-P22 can be parallelized if resources allow (max 4 notebooks each)
3. Run pre-phase validation to identify which notebooks actually need changes
4. Use `run_notebook` tool to validate each phase completion
5. Apply labels: `maintenance`, `priority:P2`, `documentation`, `notebooks`
6. **Save notebook outputs** - The executed notebook outputs (cell outputs, plots,
   figures) must be saved and committed. These outputs are rendered as documentation
   on the website via mkdocs. Use `run_notebook` with `writeExecuted` parameter to
   save the executed notebook back to the original path.

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-21 | Initial maintenance plan created | ADW Workflow |
| 2026-01-21 | Added CI workflow for notebook validation (M2-P0) | ADW Workflow |
| 2026-01-21 | Plan review: split phases for max 4 notebooks each, fixed species_density migration path, added pre-phase validation guidance, corrected Taichi notebook path | ADW Workflow |
| 2026-01-22 | Added requirement to save notebook outputs - outputs are used as docs on website | ADW Workflow |
