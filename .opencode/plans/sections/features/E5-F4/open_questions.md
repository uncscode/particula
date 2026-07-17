# Open Questions

- [x] Should the initial exhaustive sedimentation majorant remain step-local, or
  should E5-F1's scratch contract expose reusable property/majorant storage?
  - Resolved 2026-07-16: keep effective density, settling velocity, active-index,
    and majorant work storage step-local. E5-F4 does not add a public scratch
    sidecar solely to avoid internal allocations.
- [x] What explicit deterministic `rtol`/`atol` values are appropriate across
  aerosol and droplet-scale SP2016 fixtures?
  - Resolved 2026-07-16: use the independent valid-slot velocity oracle at
    `rtol=1e-12, atol=0`; assert inactive and invalid scratch values exactly
    zero. Stochastic sampler assertions remain invariant-based rather than
    using an unrelated numerical tolerance.
- [x] Does the E5-F1 configuration represent collision efficiency at all?
  - Resolved 2026-07-15: E5-F4 exposes no efficiency field. Sedimentation
    efficiency is fixed at 1; any other value or model is unsupported.
- [x] Are Brownian-plus-sedimentation calls accepted when E5-F4 ships?
  - Resolved 2026-07-15: No. E5-F4 registers sedimentation-only execution;
    E5-F6 owns additive combination registration and total-majorant evidence.
- [x] Does P2 make sedimentation publicly executable?
  - Resolved 2026-07-17: No. Only the exact private sampler mask executes
    sedimentation. Public configurations and private mixed sedimentation masks
    reject before scratch/output initialization, RNG access, kernel launch, or
    particle mutation.
- [x] Is drag-corrected settling part of the SP2016 GPU support claim?
  - Resolved 2026-07-15: No. The supported path is the existing Stokes settling
    formula with Cunningham slip correction. Non-Stokes drag and DNS variants
    are explicitly out of scope.

Classifier diagnostics: none.
