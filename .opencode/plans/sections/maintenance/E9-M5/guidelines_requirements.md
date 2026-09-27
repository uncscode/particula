# Guidelines and Requirements

## Functional Requirements

1. Freeze the supported inventory after accepting M4 evidence. Each source has
   a topic, pair/entrypoint, phase owner, native API mapping, scientific oracle,
   command and evidence status. Include supported unaffected examples; exclude
   only with explicit maintainer rationale, never by silently dropping a page.
2. Native executable examples must not construct `ParticleRepresentation`,
   `GasSpecies`, `Atmosphere`, obsolete builders or facade conversion bridges.
   Preserve useful preset behavior using M2 APIs. Keep activity/surface/vapor
   pressure and other scientific strategies process-owned.
3. Derive units and normalization from M1, not legacy spelling. Cover non-unit
   volume, ordered mixed partitioning/nonpartitioning gas, particle mass lanes,
   environment and process configuration alignment. Preserve meaningful results
   and conservation with existing justified tolerances, not exact plots alone.
4. Advanced guidance demonstrates held-object getter identity, supplied-object
   setter identity, direct mutation and subsequent process validation, valid
   coordinated layout replacement and invalid replacement leaving all state
   unchanged. Exact method spelling is copied from validated M1/M2, not invented.
5. CPU runnable examples remain single-box and retain composition, equal
   substeps and documented failure boundaries. CPU replacement does not rebind
   prepared GPU/resident state. Retain explicit conversion/synchronization and
   concrete-only resident imports; no automatic retry, recovery or fallback.
6. For every affected pair: edit `.py`, run Ruff check/format, use prescribed
   notebook sync, execute the notebook, inspect outputs and retain both files.
   Standalone scripts execute directly; creating a notebook is not required.
7. Legacy snippets are permitted only in clearly labeled non-supported
   before/after history. Keep historical code out of runnable cells and current
   API directives; supported after examples must execute without legacy imports.

## Quality Bars

- Tests accompany each changed example function and fixture in the same phase;
  documentation assertions complement, never replace, numerical execution.
- Follow Google-style docstrings, units, citations and repository Ruff rules.
- Preserve scientific coverage instead of deleting legacy-backed assertions.
- Require strict documentation rendering, link review, full inventory execution
  evidence and untargeted coverage at the normal repository threshold.

## Constraints

Strict completed-and-validated chain: M1 -> M2 -> M3 -> M4 -> M5 -> M6 under E9.
P1–P7 also execute serially. Upstream contract changes reopen affected gates.
Use existing tools/dependencies; required unavailable evidence blocks handoff.
Optional CUDA absence is a recorded clean skip, not GPU performance evidence.
