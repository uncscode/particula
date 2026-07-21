# Outcomes and Guardrails

- **Primary Outcome:** Complete fixed-shape aerosol process coverage with CPU
  references for dilution and nucleation and parity-tested direct GPU dilution,
  neutral/charged wall loss, nucleation, and slot-management capabilities.
- **Secondary Goals:**
  - Preserve stable particle-array shapes and make activation/exhaustion
    outcomes observable through caller-owned per-box diagnostics.
  - Conserve particle-plus-gas inventory per box and species, including
    inventory-limited nucleation and full-slot behavior.
  - Require Warp CPU evidence, with optional CUDA evidence that skips cleanly.
  - Publish an explicit-transfer integrated example and cross-link all
    generated plan IDs in the roadmap.
- **Guardrails / Non-Goals:**
  - No user-facing backend selection, high-level GPU `Runnable`, process
    scheduler, or multi-box transport; those remain Epic G responsibilities.
  - No dynamic GPU allocation/resizing, hidden host transfers, CPU fallback,
    or unplanned container-ownership changes.
  - No graph-capture, differentiability, CFD coupling, or production
    performance claims.
  - No exact CPU/GPU RNG-sequence requirement; deterministic coefficients and
    statistically bounded outcomes are the validation targets.
  - No silent loss of nucleation demand or slot-exhaustion inventory.
