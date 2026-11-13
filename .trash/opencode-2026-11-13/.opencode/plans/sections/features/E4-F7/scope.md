# Scope

Publish the final, evidence-backed low-level GPU condensation contract and make it reproducible. Documentation changes are gated on the shipped behavior and tests from E4-F1 through E4-F6.

## In Scope

- Add a canonical support matrix for vapor-pressure refresh, activity and effective surface-tension modes, exactly four fixed substeps, latent heat, gas coupling, diagnostics, scratch ownership, devices, shapes, and precision.
- Extend the canonical direct-kernel example with explicit conversions, ordered species metadata, final E4 configuration, reusable stable-shape buffers, and explicit restore/checkpoint behavior.
- Add troubleshooting for configuration, species order, shape, device, environment exclusivity, physical validation, inventory limiting, diagnostics, and missing Warp/CUDA.
- Publish focused Warp CPU commands and clearly label optional CUDA commands.
- Deliberately update issue 1272 documentation assertions after dependency evidence lands; retain guardrails against unsupported claims.
- Update README/example indexes and mark the E4 roadmap milestone shipped only when the complete exit bar passes.

## Out of Scope

- High-level `Aerosol` or `Runnable` GPU integration, automatic backend selection, implicit fallback, or hidden migration/synchronization.
- New condensation physics, kernel behavior, container fields, adaptive substeps, data-dependent loops, or CPU strategy objects inside Warp data.
- Claiming unsupported activity models such as BAT or staggered/Gauss-Seidel GPU condensation.
- Broad diagnostics beyond the outputs delivered by E4-F1 through E4-F6; the requested diagnostics scope is none.
- Treating optional CUDA availability as a prerequisite for the supported Warp CPU reproduction baseline.
- The **CPU/Warp parity-walkthrough follow-up**: independent CPU and Warp
  inputs with separate physics, inventory-conservation, and energy-bookkeeping
  tolerances remain deferred outside E4-F7.
