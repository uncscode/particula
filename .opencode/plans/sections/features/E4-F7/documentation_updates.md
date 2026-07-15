# Documentation Updates

- `docs/Features/data-containers-and-gpu-foundations.md` — published the canonical bounded direct-condensation contract: supported modes, schemas, explicit ownership, four coupled substeps, P2-finalized transfer semantics, diagnostics, validation/rollback boundaries, device policy, and explicit non-goals.
- `docs/Features/particle-data-migration.md` — replaced obsolete direct-call guidance with the required `thermodynamics=` signature, two-item return, in-place gas mutation, explicit conversion/restore ownership, accepted environment inputs, and a link to the canonical page.
- `particula/tests/condensation_latent_heat_docs_test.py` — added text-only publication regressions for the foundations configuration, lifecycle, validation/boundaries, and migration contracts; existing notebook and CPU-publication checks remain.

Issue #1314 intentionally did not update examples, indexes, README, roadmaps, troubleshooting material, runtime APIs, kernels, or containers.
