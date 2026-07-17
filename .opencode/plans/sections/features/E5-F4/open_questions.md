# Open Questions

- [x] Should the initial exhaustive sedimentation majorant remain step-local, or
  should E5-F1's scratch contract expose reusable property/majorant storage?
  - Resolved 2026-07-16: keep effective density, settling velocity, active-index,
    and majorant work storage step-local. E5-F4 does not add a public scratch
    sidecar solely to avoid internal allocations.
- [x] What explicit deterministic `rtol`/`atol` values are appropriate across
  aerosol and droplet-scale SP2016 fixtures?
  - Resolved 2026-07-16: use `rtol=1e-6, atol=0` for nonzero properties, pair
    rates, and majorants in both separately exercised scales. Assert analytical
    zero rates exactly. Do not use NumPy's default `atol=1e-8`, which would hide
    aerosol-scale errors.
- [x] Does the E5-F1 configuration represent collision efficiency at all?
  - Resolved 2026-07-15: E5-F4 exposes no efficiency field. Sedimentation
    efficiency is fixed at 1; any other value or model is unsupported.
- [x] Are Brownian-plus-sedimentation calls accepted when E5-F4 ships?
  - Resolved 2026-07-15: No. E5-F4 registers sedimentation-only execution;
    E5-F6 owns additive combination registration and total-majorant evidence.
- [x] Does P2 make sedimentation publicly executable?
  - Resolved 2026-07-17: No. Issue #1348 adds only private exact-mask dispatch;
    public preflight rejection remains intact, and private mixed sedimentation
    masks return without scheduling or RNG advancement.
- [x] Is drag-corrected settling part of the SP2016 GPU support claim?
  - Resolved 2026-07-15: No. The supported path is the existing Stokes settling
    formula with Cunningham slip correction. Non-Stokes drag and DNS variants
    are explicitly out of scope.

Classifier diagnostics: none.
