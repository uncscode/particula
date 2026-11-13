# E9-M5 → M6 handoff

M5 migrates examples and documentation; it does not remove compatibility APIs,
change scientific algorithms, bump a release, merge its PR or authorize an M6
implementation automatically. M6 must consume the reviewed M5 inventory and
completed validation record before starting facade removal.

## Delivered surface

- 58 original example sources, 46 notebook pairs, plus the new native
  state-access/replacement walkthrough. All supported rows are classified in
  `example_inventory.json`; full workloads and output inspection are recorded
  by phase in `implementation.md`.
- Native flat `Aerosol`, explicit distribution/count/volume semantics,
  process-owned physics, ordered gas mappings, held-state access and validated
  individual/coordinated replacement.
- CPU scientific, direct-transfer and resident examples retain their separate
  boundaries. Optional CUDA absence does not prove native capture, performance
  or GPU measurements. No automatic transfer/rebinding/recovery was introduced.
- README/home quick starts, migration pages, topic guides, contributor guidance,
  helper plugin starter and its generated OpenCode payload are reconciled.
- Historical exceptions are prose/table-only facade references in migration
  history and release roadmaps. There are no supported legacy constructor calls
  in runnable example sources. Generic contributor templates and historical
  design sketches are explicitly nonexecutable text.

## Preserve during removal

Keep the independent numerical checks and full notebook workloads. The organic
partitioning/coagulation simulation retains a known fixed-composition binned
transport defect (observed species errors approximately -59% to +35%), the
cloud dry-phase oscillations remain unqualified, and the soot property-table
assumptions are not newly qualified. Their prominent limitations are evidence
boundaries, not new accepted tolerances. M6 must not suppress them as part of API
cleanup. Follow the retained M3 method record when assessing model changes.

Replacement does not implicitly rebind process species maps or GPU/resident
state. Distinguish particle-material widths from full-gas widths and preserve
inert lanes. Count/volume normalization and radius-PDF quadrature remain part
of the scientific contract.

## Publication limitation

The additional full-publication strict build rendered the native API pages but
exited 1 with 1751 Griffe warnings in 110 production source files, all unchanged
by M5. The required base strict build is separate and intentionally does not
generate API pages or render notebooks. This handoff does not claim a clean full
publication strict build; production docstring cleanup is separate follow-up.
Notebook execution is evidenced by the per-phase runs, not by either rendering
command. Final required gate outcomes are recorded in `implementation.md`.
