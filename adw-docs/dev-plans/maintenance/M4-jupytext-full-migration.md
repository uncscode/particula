# Maintenance M4: Jupytext Notebook Sync - Full Migration

**Status**: Blocked
**Priority**: P2
**Owners**: ADW / Maintainers
**Target Date**: 2026-Q2
**Last Updated**: 2026-01-30
**Size**: Medium (~35 notebooks, 6 phases)
**Blocked By**: M3 (Pilot must complete first)

## Vision

Complete the Jupytext paired sync migration for all remaining notebooks in
`docs/Examples/`, building on the workflow validated in M3 (Pilot).

This plan will:
1. Convert remaining ~35 notebooks to paired `.py:percent` format
2. Implement pre-commit hook for automatic sync+execute
3. Add CI validation for sync status
4. Handle long-running simulation notebooks

## Prerequisites

- [ ] **M3 Complete**: Pilot migration shipped and workflow validated
- [ ] **Lessons Learned**: M3 completion notes reviewed and incorporated

## Current State Analysis

| Directory | Notebooks | Status |
|-----------|-----------|--------|
| Activity | 1 | Converted in M3 |
| Gas_Phase | 3 | Converted in M3 |
| Aerosol | 1 | To convert |
| Chamber_Wall_Loss | 4 | To convert |
| Dynamics/Coagulation | 8 | To convert (4 have orphaned .py) |
| Dynamics/Condensation | 4 | To convert |
| Dynamics/Customization | 1 | To convert |
| Equilibria | 2 | To convert |
| Nucleation | 1 | To convert |
| Particle_Phase | 6 | To convert |
| Simulations | 6 | To convert (long-running) |
| **Total Remaining** | **~35** | |

### Long-Running Notebooks (Simulations)

These may require special handling (increased timeout or CI-only execution):
- `Cloud_Chamber_Single_Cycle.ipynb`
- `Cloud_Chamber_Multi_Cycle.ipynb`
- `Soot_Formation_in_Flames.ipynb`
- `Cough_Droplets_Partitioning.ipynb`
- `Organic_Partitioning_and_Coagulation.ipynb`
- `Biomass_Burning_Cloud_Interactions.ipynb`

### Orphaned `.py` Files (Dynamics/Coagulation/Functional)

These `.py` files exist without matching notebooks - need investigation:
- `Coagulation_Basic_1_PMF.py`
- `Coagulation_Basic_2_PDF.py`
- `Coagulation_Basic_3_compared.py`
- `Coagulation_Basic_4_ParticleResolved.py`

## Scope

### In Scope

- All remaining `.ipynb` files under `docs/Examples/` (~35)
- Pre-commit hook for automatic sync+execute
- CI validation for notebook sync status
- Long-running notebook handling strategy

### Out of Scope

- Notebooks already converted in M3 (Activity, Gas_Phase)
- Changes to the Jupytext configuration (set in M3)

## Dependencies

| Dependency | Status | Notes |
|------------|--------|-------|
| M3 Pilot | Blocked | Must complete and validate workflow first |
| `validate_notebook` tool | Ready | Already supports all needed operations |
| `run_notebook` tool | Ready | Executes notebooks for validation |
| `jupytext` package | Available | Used by ADW tools internally |

## Phase Checklist

### Phase 1: Convert Core Notebooks (`M4-P1`)

**Issue:** TBD | **Size:** M | **Status:** Not Started

- [ ] **M4-P1-1:** Convert `docs/Examples/Aerosol/` notebooks (1)
- [ ] **M4-P1-2:** Convert `docs/Examples/Equilibria/Notebooks/` (2)
- [ ] **M4-P1-3:** Convert `docs/Examples/Particle_Phase/Notebooks/` (6)
- [ ] **M4-P1-4:** Run `ruff check` and fix linting issues
- [ ] **M4-P1-5:** Validate all conversions with `--check-sync`
- [ ] **M4-P1-6:** Execute notebooks to verify they work

**Acceptance Criteria:**
- All 9 notebooks converted and synced
- `ruff check` passes on new `.py` files
- All notebooks execute successfully

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

### Phase 3: Convert Chamber & Nucleation Notebooks (`M4-P3`)

**Issue:** TBD | **Size:** S | **Status:** Not Started

- [ ] **M4-P3-1:** Convert `docs/Examples/Chamber_Wall_Loss/Notebooks/` (4)
- [ ] **M4-P3-2:** Convert `docs/Examples/Nucleation/Notebooks/` (1)
- [ ] **M4-P3-3:** Run `ruff check` and fix linting issues
- [ ] **M4-P3-4:** Validate and execute all conversions

**Acceptance Criteria:**
- All 5 notebooks converted and synced
- All notebooks execute successfully

### Phase 4: Convert Simulation Notebooks (`M4-P4`)

**Issue:** TBD | **Size:** M | **Status:** Not Started

- [ ] **M4-P4-1:** Profile execution time for each simulation notebook
- [ ] **M4-P4-2:** Convert `docs/Examples/Simulations/Notebooks/` (6)
- [ ] **M4-P4-3:** Run `ruff check` and fix linting issues
- [ ] **M4-P4-4:** Validate sync status
- [ ] **M4-P4-5:** Execute with extended timeout (1200s) and document times
- [ ] **M4-P4-6:** Mark slow notebooks for pre-commit exclusion if needed

**Acceptance Criteria:**
- All 6 simulation notebooks converted and synced
- Execution times documented
- Slow notebook handling strategy documented

### Phase 5: Pre-commit Hook Implementation (`M4-P5`)

**Issue:** TBD | **Size:** S | **Status:** Not Started

- [ ] **M4-P5-1:** Create `.opencode/hooks/` directory
- [ ] **M4-P5-2:** Create `sync-execute-notebooks.sh` hook script
- [ ] **M4-P5-3:** Add hook to `.pre-commit-config.yaml`
- [ ] **M4-P5-4:** Configure exclusions for slow notebooks
- [ ] **M4-P5-5:** Test hook with sample notebook edits

**Acceptance Criteria:**
- Pre-commit runs only on changed notebook `.py` files
- Hook correctly syncs, executes, and stages both files
- Slow notebooks excluded from pre-commit (CI-only)

### Phase 6: CI Validation & Documentation (`M4-P6`)

**Issue:** TBD | **Size:** S | **Status:** Not Started

- [ ] **M4-P6-1:** Add CI job to check notebook sync status (`--check-sync`)
- [ ] **M4-P6-2:** Add CI job to execute notebooks (with timeout handling)
- [ ] **M4-P6-3:** Clean up any remaining `.bak` files
- [ ] **M4-P6-4:** Update `adw-docs/documentation_guide.md` with final workflow
- [ ] **M4-P6-5:** Update this plan with completion notes
- [ ] **M4-P6-6:** Update `maintenance/index.md` with final status

**Acceptance Criteria:**
- CI fails if notebooks are out of sync
- CI executes notebooks (or documents exclusions)
- No stale `.bak` files in repository
- Plan status updated to Shipped

## Critical Testing Requirements

- **No Coverage Modifications**: This maintenance task doesn't affect code coverage
- **Notebook Validation**: Each phase validates with `--check-sync`
- **Execution Testing**: Each phase executes converted notebooks
- **Linting Compliance**: Each phase runs `ruff check` before completion

## Testing Strategy

### Validation Commands

```bash
# Convert all remaining notebooks
validate_notebook docs/Examples --recursive --convert-to-py

# Check sync status (CI-friendly)
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

| Notebook | Expected Time | Handling |
|----------|---------------|----------|
| Cloud_Chamber_*.ipynb | TBD | Profile in M4-P4 |
| Soot_Formation_*.ipynb | TBD | Profile in M4-P4 |
| Other Simulations | TBD | Profile in M4-P4 |

Options for slow notebooks:
1. **Exclude from pre-commit**: Run only in CI with extended timeout
2. **Mark with `.slow` suffix**: `notebook.slow.py` excluded by pattern
3. **Nightly CI job**: Execute slow notebooks on schedule, not per-PR

## Related Documents

- [M3: Jupytext Pilot](M3-jupytext-notebook-sync.md) - Prerequisite pilot migration
- [Documentation Guide](../../documentation_guide.md) - Notebook guidelines
- [Linting Guide](../../linting_guide.md) - Ruff configuration

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-30 | Initial plan created (deferred from M3 pilot) | ADW |
