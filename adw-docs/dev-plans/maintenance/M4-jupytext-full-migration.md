# Maintenance M4: Jupytext Notebook Sync - Full Migration

**Status**: In Progress
**Priority**: P2
**Owners**: ADW / Maintainers
**Target Date**: 2026-Q2
**Last Updated**: 2026-01-31
**Size**: Medium (~35 notebooks, 15 phases)
**Unblocked**: M3 (Pilot) completed successfully

## Vision

Complete the Jupytext paired sync migration for all remaining notebooks in
`docs/Examples/`, building on the workflow validated in M3 (Pilot).

This plan will:
1. Convert remaining ~35 notebooks to paired `.py:percent` format
2. Implement pre-commit hook for automatic sync+execute
3. Add CI validation for sync status
4. Handle long-running simulation notebooks

**Design Decision:** Conversions are isolated into groups of **max 3 notebooks
per phase/issue** to enable focused validation and API update detection.

## Prerequisites

- [x] **M3 Complete**: Pilot migration shipped and workflow validated
- [x] **Lessons Learned**: M3 completion notes reviewed and incorporated

## Current State Analysis

| Directory | Notebooks | Status |
|-----------|-----------|--------|
| Activity | 1 | Converted in M3 |
| Gas_Phase | 3 | Converted in M3 |
| Aerosol | 1 | To convert |
| Chamber_Wall_Loss | 5 | To convert |
| Dynamics/Coagulation | 7 | To convert (4 have existing .py - need sync) |
| Dynamics/Condensation | 4 | To convert |
| Dynamics/Customization | 1 | To convert |
| Equilibria | 2 | To convert |
| Nucleation | 1 | To convert |
| Particle_Phase | 6 | To convert |
| Simulations | 6 | To convert (long-running) |
| **Total Remaining** | **35** | |

### Notebook Inventory (Full List)

```
docs/Examples/
├── Aerosol/
│   └── Aerosol_Tutorial.ipynb                              # P1
├── Equilibria/Notebooks/
│   ├── equilibria_part1.ipynb                              # P1
│   └── activity_part1.ipynb                                # P1
├── Nucleation/Notebooks/
│   └── Custom_Nucleation_Single_Species.ipynb              # P2
├── Particle_Phase/Notebooks/
│   ├── Particle_Surface_Tutorial.ipynb                     # P2
│   ├── Particle_Representation_Tutorial.ipynb              # P2
│   ├── Distribution_Tutorial.ipynb                         # P3
│   ├── Aerosol_Distributions.ipynb                         # P3
│   ├── Activity_Tutorial.ipynb                             # P3
│   └── Functional/Activity_Functions.ipynb                 # P4
├── Chamber_Wall_Loss/Notebooks/
│   ├── Wall_Loss_Tutorial.ipynb                            # P4
│   ├── Spherical_Wall_Loss_Strategy.ipynb                  # P4
│   ├── Rectangular_Wall_Loss_Strategy.ipynb                # P5
│   ├── wall_loss_builders_factory.ipynb                    # P5
│   └── Chamber_Forward_Simulation.ipynb                    # P5
├── Dynamics/Coagulation/
│   ├── Coagulation_1_PMF_Pattern.ipynb                     # P6
│   ├── Coagulation_3_Particle_Resolved_Pattern.ipynb       # P6
│   ├── Coagulation_4_Compared.ipynb                        # P6
│   ├── Charge/Coagulation_with_Charge_objects.ipynb        # P7
│   ├── Charge/Coagulation_with_Charge_functional.ipynb     # P7
│   └── Functional/                                         # P8 (sync existing .py)
│       ├── Coagulation_Basic_1_PMF.ipynb
│       ├── Coagulation_Basic_2_PDF.ipynb
│       ├── Coagulation_Basic_3_compared.ipynb
│       └── Coagulation_Basic_4_ParticleResolved.ipynb
├── Dynamics/Condensation/
│   ├── Condensation_1_Bin.ipynb                            # P9
│   ├── Condensation_2_MassBin.ipynb                        # P9
│   ├── Condensation_3_MassResolved.ipynb                   # P9
│   └── Staggered_Condensation_Example.ipynb                # P10
├── Dynamics/Customization/
│   └── Adding_Particles_During_Simulation.ipynb            # P10
└── Simulations/Notebooks/
    ├── Cloud_Chamber_Single_Cycle.ipynb                    # P11 (slow)
    ├── Cloud_Chamber_Multi_Cycle.ipynb                     # P11 (slow)
    ├── Soot_Formation_in_Flames.ipynb                      # P12 (slow)
    ├── Cough_Droplets_Partitioning.ipynb                   # P12 (slow)
    ├── Organic_Partitioning_and_Coagulation.ipynb          # P13 (slow)
    └── Biomass_Burning_Cloud_Interactions.ipynb            # P13 (slow)
```

### Existing `.py` Files (Dynamics/Coagulation/Functional)

These `.py` files already exist WITH matching notebooks - they need **sync validation**:
- `Coagulation_Basic_1_PMF.py` ↔ `Coagulation_Basic_1_PMF.ipynb`
- `Coagulation_Basic_2_PDF.py` ↔ `Coagulation_Basic_2_PDF.ipynb`
- `Coagulation_Basic_3_compared.py` ↔ `Coagulation_Basic_3_compared.ipynb`
- `Coagulation_Basic_4_ParticleResolved.py` ↔ `Coagulation_Basic_4_ParticleResolved.ipynb`

**Action:** Verify sync status, update `.py` format if needed (ensure `py:percent`), and lint.

## Scope

### In Scope

- All remaining `.ipynb` files under `docs/Examples/` (35 notebooks)
- Pre-commit hook for automatic sync+execute
- CI validation for notebook sync status
- Long-running notebook handling strategy

### Out of Scope

- Notebooks already converted in M3 (Activity, Gas_Phase)
- Changes to the Jupytext configuration (set in M3)

## Dependencies

| Dependency | Status | Notes |
|------------|--------|-------|
| M3 Pilot | Complete | Workflow validated |
| `validate_notebook` tool | Ready | Supports all needed operations |
| `run_notebook` tool | Ready | Executes notebooks for validation |
| `jupytext` package | Available | Used by ADW tools internally |

---

## Phase Checklist

### Phase 1: Aerosol + Equilibria (`M4-P1`)

**Issue:** TBD | **Size:** S | **Status:** Not Started

**Notebooks (3):**
- `docs/Examples/Aerosol/Aerosol_Tutorial.ipynb`
- `docs/Examples/Equilibria/Notebooks/equilibria_part1.ipynb`
- `docs/Examples/Equilibria/Notebooks/activity_part1.ipynb`

**Tasks:**
- [ ] Convert all 3 notebooks to `.py:percent` format
- [ ] Run `ruff check --fix` and `ruff format` on new `.py` files
- [ ] Validate sync with `--check-sync`
- [ ] Execute notebooks to verify they work
- [ ] Check for API updates needed (review any import/deprecation warnings)

**Acceptance Criteria:**
- All 3 notebooks converted and synced
- `ruff check` passes on new `.py` files
- All notebooks execute successfully without API warnings

---

### Phase 2: Nucleation + Particle_Phase Part 1 (`M4-P2`)

**Issue:** TBD | **Size:** S | **Status:** Not Started

**Notebooks (3):**
- `docs/Examples/Nucleation/Notebooks/Custom_Nucleation_Single_Species.ipynb`
- `docs/Examples/Particle_Phase/Notebooks/Particle_Surface_Tutorial.ipynb`
- `docs/Examples/Particle_Phase/Notebooks/Particle_Representation_Tutorial.ipynb`

**Tasks:**
- [ ] Convert all 3 notebooks to `.py:percent` format
- [ ] Run `ruff check --fix` and `ruff format` on new `.py` files
- [ ] Validate sync with `--check-sync`
- [ ] Execute notebooks to verify they work
- [ ] Check for API updates needed

**Acceptance Criteria:**
- All 3 notebooks converted and synced
- `ruff check` passes on new `.py` files
- All notebooks execute successfully without API warnings

---

### Phase 3: Particle_Phase Part 2 (`M4-P3`)

**Issue:** #1004 | **Size:** S | **Status:** Completed

**Notebooks (3):**
- `docs/Examples/Particle_Phase/Notebooks/Distribution_Tutorial.ipynb`
- `docs/Examples/Particle_Phase/Notebooks/Aerosol_Distributions.ipynb`
- `docs/Examples/Particle_Phase/Notebooks/Activity_Tutorial.ipynb`

**Tasks:**
- [x] Convert all 3 notebooks to `.py:percent` format
- [x] Run `ruff check --fix` and `ruff format` on new `.py` files
- [x] Validate sync with `--check-sync`
- [x] Execute notebooks to verify they work
- [x] Check for API updates needed

**Completion Notes:**
- Converted all 3 notebooks to paired `.py:percent` format
- Linted with ruff check --fix and ruff format
- Synced and validated with --check-sync
- Executed all notebooks successfully without API warnings
- No API issues detected

**Acceptance Criteria:**
- All 3 notebooks converted and synced (met)
- `ruff check` passes on new `.py` files (met)
- All notebooks execute successfully without API warnings (met)

---

### Phase 4: Particle_Phase Functional + Chamber_Wall_Loss Part 1 (`M4-P4`)

**Issue:** TBD | **Size:** S | **Status:** Not Started

**Notebooks (3):**
- `docs/Examples/Particle_Phase/Notebooks/Functional/Activity_Functions.ipynb`
- `docs/Examples/Chamber_Wall_Loss/Notebooks/Wall_Loss_Tutorial.ipynb`
- `docs/Examples/Chamber_Wall_Loss/Notebooks/Spherical_Wall_Loss_Strategy.ipynb`

**Tasks:**
- [ ] Convert all 3 notebooks to `.py:percent` format
- [ ] Run `ruff check --fix` and `ruff format` on new `.py` files
- [ ] Validate sync with `--check-sync`
- [ ] Execute notebooks to verify they work
- [ ] Check for API updates needed (wall loss API recently updated)

**Acceptance Criteria:**
- All 3 notebooks converted and synced
- `ruff check` passes on new `.py` files
- All notebooks execute successfully without API warnings

---

### Phase 5: Chamber_Wall_Loss Part 2 (`M4-P5`)

**Issue:** TBD | **Size:** S | **Status:** Not Started

**Notebooks (3):**
- `docs/Examples/Chamber_Wall_Loss/Notebooks/Rectangular_Wall_Loss_Strategy.ipynb`
- `docs/Examples/Chamber_Wall_Loss/Notebooks/wall_loss_builders_factory.ipynb`
- `docs/Examples/Chamber_Wall_Loss/Notebooks/Chamber_Forward_Simulation.ipynb`

**Tasks:**
- [ ] Convert all 3 notebooks to `.py:percent` format
- [ ] Run `ruff check --fix` and `ruff format` on new `.py` files
- [ ] Validate sync with `--check-sync`
- [ ] Execute notebooks to verify they work
- [ ] Check for API updates needed (wall loss API recently updated)

**Acceptance Criteria:**
- All 3 notebooks converted and synced
- `ruff check` passes on new `.py` files
- All notebooks execute successfully without API warnings

---

### Phase 6: Dynamics/Coagulation Main (`M4-P6`)

**Issue:** #988 | **Size:** S | **Status:** Completed

**Notebooks (3):**
- `docs/Examples/Dynamics/Coagulation/Coagulation_1_PMF_Pattern.ipynb`
- `docs/Examples/Dynamics/Coagulation/Coagulation_3_Particle_Resolved_Pattern.ipynb`
- `docs/Examples/Dynamics/Coagulation/Coagulation_4_Compared.ipynb`

**Tasks:**
- [x] Convert all 3 notebooks to `.py:percent` format
- [x] Run `ruff check --fix` and `ruff format` on new `.py` files
- [x] Validate sync with `--check-sync`
- [x] Execute notebooks to verify they work
- [x] Check for API updates needed

**Completion Notes:**
- Converted to paired `.py:percent` format and synced back to `.ipynb`.
- Aligned coagulation API calls with current builders (`BrownianCoagulationBuilder` → `Coagulation`).
- Executed all three notebooks cleanly; no deprecation warnings.

**Acceptance Criteria:**
- All 3 notebooks converted and synced (met)
- `ruff check` passes on new `.py` files (met)
- All notebooks execute successfully without API warnings (met)


---

### Phase 7: Dynamics/Coagulation Charge (`M4-P7`)

**Issue:** #989 | **Size:** XS | **Status:** Completed (charged pair migrated)

**Notebooks (2):**
- `docs/Examples/Dynamics/Coagulation/Charge/Coagulation_with_Charge_objects.ipynb`
- `docs/Examples/Dynamics/Coagulation/Charge/Coagulation_with_Charge_functional.ipynb`

**Tasks:**
- [x] Convert both notebooks to `.py:percent` format
- [x] Run `ruff check --fix` and `ruff format` on new `.py` files
- [x] Validate sync with `--check-sync`
- [x] Execute notebooks to verify they work
- [x] Check for API updates needed (none observed; kernelspec display name set to `particula_dev312`)

**Acceptance Criteria:**
- Both notebooks converted and synced (met)
- `ruff check` passes on new `.py` files (met)
- All notebooks execute successfully without API warnings (met)

**Completion Notes (2026-01-30):**
- Converted and synced both charged coagulation notebooks with Jupytext percent format
- Standardized kernelspec to `particula_dev312`; execution outputs present
- No API changes required; linted cleanly and executed without warnings

---

### Phase 8: Dynamics/Coagulation Functional (Sync Existing) (`M4-P8`)

**Issue:** TBD | **Size:** S | **Status:** Not Started

**Notebooks (4) - Already have `.py` files:**
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_1_PMF.ipynb`
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_2_PDF.ipynb`
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_3_compared.ipynb`
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_4_ParticleResolved.ipynb`

**Tasks:**
- [ ] Verify existing `.py` files are in `py:percent` format (convert if not)
- [ ] Check sync status with `--check-sync`
- [ ] Run `ruff check --fix` and `ruff format` on `.py` files
- [ ] Re-sync if format changed
- [ ] Execute notebooks to verify they work
- [ ] Check for API updates needed

**Acceptance Criteria:**
- All 4 notebooks properly synced with `py:percent` format
- `ruff check` passes on `.py` files
- All notebooks execute successfully without API warnings

---

### Phase 9: Dynamics/Condensation Part 1 (`M4-P9`)

**Issue:** TBD | **Size:** S | **Status:** Not Started

**Notebooks (3):**
- `docs/Examples/Dynamics/Condensation/Condensation_1_Bin.ipynb`
- `docs/Examples/Dynamics/Condensation/Condensation_2_MassBin.ipynb`
- `docs/Examples/Dynamics/Condensation/Condensation_3_MassResolved.ipynb`

**Tasks:**
- [ ] Convert all 3 notebooks to `.py:percent` format
- [ ] Run `ruff check --fix` and `ruff format` on new `.py` files
- [ ] Validate sync with `--check-sync`
- [ ] Execute notebooks to verify they work
- [ ] Check for API updates needed (condensation API recently updated)

**Acceptance Criteria:**
- All 3 notebooks converted and synced
- `ruff check` passes on new `.py` files
- All notebooks execute successfully without API warnings

---

### Phase 10: Dynamics/Condensation Part 2 + Customization (`M4-P10`)

**Issue:** TBD | **Size:** XS | **Status:** Not Started

**Notebooks (2):**
- `docs/Examples/Dynamics/Condensation/Staggered_Condensation_Example.ipynb`
- `docs/Examples/Dynamics/Customization/Adding_Particles_During_Simulation.ipynb`

**Tasks:**
- [ ] Convert both notebooks to `.py:percent` format
- [ ] Run `ruff check --fix` and `ruff format` on new `.py` files
- [ ] Validate sync with `--check-sync`
- [ ] Execute notebooks to verify they work
- [ ] Check for API updates needed

**Acceptance Criteria:**
- Both notebooks converted and synced
- `ruff check` passes on new `.py` files
- All notebooks execute successfully without API warnings

### Phase 2: Nucleation + Particle_Phase Part 1 (`M4-P2`)

**Issue:** #984 | **Size:** S | **Status:** Completed (2026-01-30)

- [x] **M4-P2-1:** Convert `Custom_Nucleation_Single_Species` to paired `.py`
- [x] **M4-P2-2:** Convert `Particle_Surface_Tutorial` to paired `.py`
- [x] **M4-P2-3:** Convert `Particle_Representation_Tutorial` to paired `.py`
- [x] **M4-P2-4:** `ruff check` + `ruff format` on new percent scripts
- [x] **M4-P2-5:** Sync + `--check-sync` for all three notebooks
- [x] **M4-P2-6:** Execute notebooks and review warnings

**Acceptance Criteria:**
- All three target notebooks paired (`.ipynb` + `.py`) and linted
- Sync validation passes for Nucleation + Particle_Phase (Part 1)
- Executions complete without errors; warnings reviewed

**Completion Note (2026-01-30):**
- Jupytext migration for Nucleation + Particle_Phase Part 1 completed
- Converted and executed: Nucleation single species, Particle surface, particle
  representation tutorials
---

### Phase 11: Simulations - Cloud Chamber (`M4-P11`)

**Issue:** #993 | **Size:** S | **Status:** Completed (2026-01-31)

**Notebooks (2) - Long-Running:**
- `docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb`
- `docs/Examples/Simulations/Notebooks/Cloud_Chamber_Multi_Cycle.ipynb`

**Tasks:**
- [x] Profile execution time for each notebook (document in table below)
- [x] Convert both notebooks to `.py:percent` format
- [x] Run `ruff check --fix` and `ruff format` on new `.py` files
- [x] Validate sync with `--check-sync`
- [x] Execute with extended timeout (1200s)
- [x] Mark for pre-commit exclusion if >60s

**Completion Notes:**
- Converted to paired `.py:percent` format (Single Cycle: 465 LOC, Multi Cycle: 1804 LOC)
- All ruff linting passed
- Sync validated successfully
- Both notebooks execute successfully (long-running simulations)
- Recommended for nightly CI only due to extended execution times

**Acceptance Criteria:**
- Both notebooks converted and synced (met)
- Execution times documented (met)
- Slow notebook handling configured (met - marked for nightly CI)

---

### Phase 12: Simulations - Soot + Cough (`M4-P12`)

**Issue:** TBD | **Size:** S | **Status:** Not Started

**Notebooks (2) - Long-Running:**
- `docs/Examples/Simulations/Notebooks/Soot_Formation_in_Flames.ipynb`
- `docs/Examples/Simulations/Notebooks/Cough_Droplets_Partitioning.ipynb`

**Tasks:**
- [ ] Profile execution time for each notebook
- [ ] Convert both notebooks to `.py:percent` format
- [ ] Run `ruff check --fix` and `ruff format` on new `.py` files
- [ ] Validate sync with `--check-sync`
- [ ] Execute with extended timeout (1200s)
- [ ] Mark for pre-commit exclusion if >60s

**Acceptance Criteria:**
- Both notebooks converted and synced
- Execution times documented
- Slow notebook handling configured

---

### Phase 13: Simulations - Organic + Biomass (`M4-P13`)

**Issue:** TBD | **Size:** S | **Status:** Not Started

**Notebooks (2) - Long-Running:**
- `docs/Examples/Simulations/Notebooks/Organic_Partitioning_and_Coagulation.ipynb`
- `docs/Examples/Simulations/Notebooks/Biomass_Burning_Cloud_Interactions.ipynb`

**Tasks:**
- [ ] Profile execution time for each notebook
- [ ] Convert both notebooks to `.py:percent` format
- [ ] Run `ruff check --fix` and `ruff format` on new `.py` files
- [ ] Validate sync with `--check-sync`
- [ ] Execute with extended timeout (1200s)
- [ ] Mark for pre-commit exclusion if >60s

**Acceptance Criteria:**
- Both notebooks converted and synced
- Execution times documented
- Slow notebook handling configured

---

### Phase 14: Pre-commit Hook Implementation (`M4-P14`)

**Issue:** TBD | **Size:** S | **Status:** Not Started

**Tasks:**
- [ ] Create `.opencode/hooks/` directory
- [ ] Create `sync-execute-notebooks.sh` hook script
- [ ] Add hook to `.pre-commit-config.yaml`
- [ ] Configure exclusions for slow notebooks (from P11-P13)
- [ ] Test hook with sample notebook edits

**Acceptance Criteria:**
- Pre-commit runs only on changed notebook `.py` files
- Hook correctly syncs, executes, and stages both files
- Slow notebooks excluded from pre-commit (CI-only)

---

### Phase 15: CI Validation & Documentation (`M4-P15`)

**Issue:** TBD | **Size:** S | **Status:** Not Started

**Tasks:**
- [ ] Add CI job to check notebook sync status (`--check-sync`)
- [ ] Add CI job to execute notebooks (with timeout handling)
- [ ] Clean up any remaining `.bak` files
- [ ] Update `adw-docs/documentation_guide.md` with final workflow
- [ ] Update this plan with completion notes
- [ ] Update `maintenance/index.md` with final status

**Acceptance Criteria:**
- CI fails if notebooks are out of sync
- CI executes notebooks (or documents exclusions)
- No stale `.bak` files in repository
- Plan status updated to Shipped

---

## Critical Testing Requirements

- **No Coverage Modifications**: This maintenance task doesn't affect code coverage
- **Notebook Validation**: Each phase validates with `--check-sync`
- **Execution Testing**: Each phase executes converted notebooks
- **Linting Compliance**: Each phase runs `ruff check` before completion
- **API Validation**: Each phase checks for deprecation warnings or API changes

## Testing Strategy

### Per-Phase Workflow

Each conversion phase follows this workflow:

```bash
# 1. Convert notebooks to .py:percent format
validate_notebook docs/Examples/path/to/notebook.ipynb --convert-to-py

# 2. Lint the new .py files
ruff check docs/Examples/path/to/notebook.py --fix
ruff format docs/Examples/path/to/notebook.py

# 3. Sync (regenerate .ipynb from linted .py)
validate_notebook docs/Examples/path/to/notebook.ipynb --sync

# 4. Validate sync status
validate_notebook docs/Examples/path/to/notebook.ipynb --check-sync

# 5. Execute to verify notebook works
run_notebook docs/Examples/path/to/notebook.ipynb

# 6. Review output for API warnings/deprecations
```

### Batch Validation (After All Conversions)

```bash
# Check sync status for all notebooks
validate_notebook docs/Examples --recursive --check-sync

# Lint all example Python files
ruff check docs/Examples/ --fix
ruff format docs/Examples/

# Execute all notebooks (with extended timeout for simulations)
run_notebook docs/Examples --recursive --timeout 1200
```

### Pre-commit Hook Script

```bash
#!/usr/bin/env bash
# .opencode/hooks/sync-execute-notebooks.sh
set -e

for py_file in "$@"; do
    ipynb_file="${py_file%.py}.ipynb"
    
    echo "Processing: $py_file"
    
    # 1. Lint
    ruff check "$py_file" --fix --quiet || true
    ruff format "$py_file" --quiet
    
    # 2. Sync
    python3 .opencode/tool/validate_notebook.py "$ipynb_file" --sync
    
    # 3. Execute
    python3 .opencode/tool/run_notebook.py "$ipynb_file"
    
    # 4. Stage
    git add "$py_file" "$ipynb_file"
done
```

### Pre-commit Configuration

```yaml
# .pre-commit-config.yaml (addition)
- repo: local
  hooks:
    - id: jupytext-sync-execute
      name: Sync and execute changed notebooks
      entry: .opencode/hooks/sync-execute-notebooks.sh
      language: script
      files: ^docs/Examples/.*\.py$
      exclude: |
        (?x)^(
          docs/Examples/Simulations/.*  # Slow notebooks - CI only
        )$
      pass_filenames: true
```

## Long-Running Notebook Strategy

Execution times will be documented as each simulation phase completes:

| Notebook | Execution Time | Handling |
|----------|----------------|----------|
| Cloud_Chamber_Single_Cycle.ipynb | >300s (P11) | Nightly CI only |
| Cloud_Chamber_Multi_Cycle.ipynb | >300s (P11) | Nightly CI only |
| Soot_Formation_in_Flames.ipynb | TBD (P12) | |
| Cough_Droplets_Partitioning.ipynb | TBD (P12) | |
| Organic_Partitioning_and_Coagulation.ipynb | TBD (P13) | |
| Biomass_Burning_Cloud_Interactions.ipynb | TBD (P13) | |

**Handling Options:**
- **< 60s**: Include in pre-commit hook
- **60s - 300s**: Exclude from pre-commit, include in CI
- **> 300s**: Nightly CI job only (exclude from PR CI)

## API Validation Checklist

During each conversion phase, check for:
- [ ] Import path changes (deprecation warnings)
- [ ] Function signature changes (missing/renamed parameters)
- [ ] Return type changes (different output format)
- [ ] Removed functions or classes
- [ ] New required parameters

If API issues are found:
1. Document the issue in the phase completion notes
2. Create a follow-up issue to update the notebook content
3. Do NOT block the conversion - sync the notebook as-is

## Phase Summary Table

| Phase | Notebooks | Directory | Status |
|-------|-----------|-----------|--------|
| M4-P1 | 3 | Aerosol + Equilibria | Not Started |
| M4-P2 | 3 | Nucleation + Particle_Phase (1) | Not Started |
| M4-P3 | 3 | Particle_Phase (2) | Completed |
| M4-P4 | 3 | Particle_Phase Functional + Chamber_Wall_Loss (1) | Not Started |
| M4-P5 | 3 | Chamber_Wall_Loss (2) | Not Started |
| M4-P6 | 3 | Dynamics/Coagulation Main | Completed |
| M4-P7 | 2 | Dynamics/Coagulation Charge | Not Started |
| M4-P8 | 4 | Dynamics/Coagulation Functional (sync existing) | Not Started |
| M4-P9 | 3 | Dynamics/Condensation (1) | Not Started |
| M4-P10 | 2 | Dynamics/Condensation (2) + Customization | Not Started |
| M4-P11 | 2 | Simulations - Cloud Chamber (slow) | Completed |
| M4-P12 | 2 | Simulations - Soot + Cough (slow) | Not Started |
| M4-P13 | 2 | Simulations - Organic + Biomass (slow) | Not Started |
| M4-P14 | - | Pre-commit Hook | Not Started |
| M4-P15 | - | CI Validation & Documentation | Not Started |
| **Total** | **35** | | |

## Related Documents

- [M3: Jupytext Pilot](M3-jupytext-notebook-sync.md) - Prerequisite pilot migration
- [Documentation Guide](../../documentation_guide.md) - Notebook guidelines
- [Linting Guide](../../linting_guide.md) - Ruff configuration

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-30 | Initial plan created (deferred from M3 pilot) | ADW |
| 2026-01-30 | Restructured to max 3 notebooks per phase for isolated validation | ADW |
| 2026-01-30 | Marked M4-P7 (charged coagulation) completed; added completion notes | ADW |
