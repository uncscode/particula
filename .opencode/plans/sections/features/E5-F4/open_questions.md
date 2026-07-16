# Open Questions

- [ ] Should the initial exhaustive sedimentation majorant remain step-local, or
  should E5-F1's scratch contract expose reusable property/majorant storage?
  - Open: prefer step-local storage unless E5-F1 has already established a
    validated caller-owned work-buffer contract; do not expand public ownership
    solely for E5-F4.
- [ ] What explicit deterministic `rtol`/`atol` values are appropriate across
  aerosol and droplet-scale SP2016 fixtures?
  - Open: derive scale-specific fp64 tolerances from independent Warp CPU
    evidence during P1 and record them in tests rather than using one broad
    default.
- [ ] Does the E5-F1 configuration represent collision efficiency at all?
  - Resolved 2026-07-15: E5-F4 exposes no efficiency field. Sedimentation
    efficiency is fixed at 1; any other value or model is unsupported.
- [ ] Are Brownian-plus-sedimentation calls accepted when E5-F4 ships?
  - Resolved 2026-07-15: No. E5-F4 registers sedimentation-only execution;
    E5-F6 owns additive combination registration and total-majorant evidence.
- [ ] Is drag-corrected settling part of the SP2016 GPU support claim?
  - Resolved 2026-07-15: No. The supported path is the existing Stokes settling
    formula with Cunningham slip correction. Non-Stokes drag and DNS variants
    are explicitly out of scope.

Classifier diagnostics: none.
