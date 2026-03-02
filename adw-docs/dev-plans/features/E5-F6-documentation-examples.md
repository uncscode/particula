# Feature E5-F6: Documentation and Examples

**Parent Epic**: [E5: Non-Isothermal Condensation with Latent Heat](../epics/E5-non-isothermal-condensation.md)
**Status**: Planning
**Priority**: P2
**Owners**: @Gorkowski
**Start Date**: TBD
**Target Date**: TBD
**Last Updated**: 2026-03-02
**Size**: Small (3 phases)

## Summary

Complete documentation for the non-isothermal condensation feature: sweep all
new code for docstring gaps, add a Theory document explaining the non-isothermal
mass transfer derivation, create a usage example notebook demonstrating all three
latent heat strategies and comparing isothermal vs non-isothermal growth, and
update development plan documentation with final status.

## Goals

1. Google-style docstrings for all new public classes and functions (gap sweep)
2. Literature citations in module docstrings (Topping & Bane, Seinfeld & Pandis,
   Rogers & Yau)
3. Theory document explaining non-isothermal mass transfer derivation
4. Usage example notebook showing all latent heat strategies, isothermal vs
   non-isothermal comparison, and energy tracking
5. Updated development plan documentation

## Non-Goals

- Modifying implementation code (docstrings and comments only)
- API documentation generation (handled by mkdocs automatically)
- GPU/Warp documentation (deferred to E5-F7)

## Dependencies

- **E5-F1 through E5-F5** must all be complete before documentation sweep
- Jupytext tooling must be available for notebook sync workflow
- `mkdocs` for documentation builds

## Phase Checklist

- [ ] **E5-F6-P1**: Add docstrings and Theory document update
  - Issue: TBD | Size: S (~60 LOC) | Status: Not Started
  - Google-style docstrings for all new public classes and functions (should
    already be done per-phase, this is a sweep to catch gaps)
  - Add literature citations in module docstrings:
    - Topping & Bane (2022) Eq. 2.36
    - Seinfeld & Pandis (2016) Eq. 13.3
    - Rogers & Yau (1989) Ch. 7
  - Update or create `docs/Theory/Technical/Dynamics/Condensation_Equations.md`
    with non-isothermal mass transfer derivation linking theory to code
  - Include explanation of why size effects (Knudsen correction) are in the
    numerator only, not the denominator
  - Run docstring linter (`ruff check`, `ruff format`) to validate

- [ ] **E5-F6-P2**: Create usage example notebook
  - Issue: TBD | Size: M (~100 LOC) | Status: Not Started
  - File: `docs/Examples/Dynamics/non_isothermal_condensation_example.py`
    (py:percent format, synced to `.ipynb` via Jupytext)
  - Follow Jupytext paired sync workflow from AGENTS.md: edit .py -> lint ->
    sync -> execute -> commit both files
  - Demonstrate all three latent heat strategies (constant, linear, power law)
  - Compare isothermal vs non-isothermal growth curves on same plot
  - Show energy tracking diagnostic (`strategy.last_latent_heat_energy`)
  - Show cloud droplet activation scenario (water at 0.5% supersaturation)
  - Use `particula` public API only (no internal imports)

- [ ] **E5-F6-P3**: Update development documentation
  - Issue: TBD | Size: XS (~20 LOC) | Status: Not Started
  - Update `adw-docs/dev-plans/README.md` with epic status
  - Update `adw-docs/dev-plans/epics/index.md`
  - Update `adw-docs/dev-plans/features/index.md`
  - Add completion notes and lessons learned

## Critical Testing Requirements

- **No Coverage Modifications**: Documentation-only changes do not affect
  coverage.
- **Notebook Execution**: Example notebook must execute without errors.
- **Linting**: All docstrings must pass `ruff check` and `ruff format`.
- **Link Validation**: All markdown links must resolve.

## Testing Strategy

### Documentation Validation

| Check | Tool | Phase |
|-------|------|-------|
| Docstring format | `ruff check particula/ --fix` + `ruff format` | P1 |
| Theory document links | Manual review + mkdocs build | P1 |
| Notebook execution | `run_notebook` tool | P2 |
| Notebook sync | `validate_notebook --sync` | P2 |
| Dev-plan links | Manual review | P3 |

### Notebook Test Cases

1. **Strategy demos**: Each latent heat strategy produces expected output
2. **Comparison plot**: Isothermal vs non-isothermal curves visually distinct
3. **Energy tracking**: `last_latent_heat_energy` attribute accessible and
   non-zero for L > 0
4. **Cloud activation**: Water droplet grows at reduced rate with non-isothermal
   correction
5. **No internal imports**: Only `import particula as par` and submodules

## Success Criteria

1. All new public classes/functions have complete Google-style docstrings
2. Theory document explains the non-isothermal derivation with equations
3. Example notebook executes cleanly and produces meaningful plots
4. Development plan documentation reflects final status
5. `ruff check` and `ruff format` pass with no issues

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-02 | Initial feature document created from E5 epic | ADW |
