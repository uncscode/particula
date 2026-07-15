# Implementation Tasks

## Documentation

- [x] Derive the final argument/return/mutation table from `particula/gpu/kernels/condensation.py` after E4-F1 through E4-F6 merge. Completed by Issue #1314 documentation review.
- [x] Add supported and unsupported condensation modes to `docs/Features/data-containers-and-gpu-foundations.md`, including exactly four substeps, fp64 fixed shapes, caller-owned scratch, and device coverage.
- [x] Document explicit conversion, synchronization, metadata, and checkpoint ownership; prohibit hidden transfer, CPU-per-step vapor-pressure refresh, and implicit fallback.
- [x] Document gas partitioning, inventory limiting, conservation, latent-energy sign/units, and whole-call diagnostic aggregation.
- [x] Extend `docs/Features/particle-data-migration.md` with final call signatures and targeted validation failures.
- [x] Extend `docs/Examples/gpu_direct_kernels_quick_start.py` using lazy kernel imports, Warp CPU default, ordered species configuration, reusable buffers, and explicit restore. Completed by Issue #1315 with two sequential direct calls and caller-owned scratch, latent-heat, and energy sidecars.
- [ ] Add a parity walkthrough that constructs independent CPU and Warp inputs and separates physics, conservation, and energy tolerances. Deferred to a later phase; Issue #1315 provides bounded direct-kernel regression evidence, not a general parity walkthrough.
- [ ] Publish focused commands in the canonical page and concise discovery links in `readme.md` and `docs/Examples/index.md`.
- [ ] Update `docs/Features/Roadmap/data-oriented-gpu.md` only after all E4 evidence gates pass.

## Tooling / Tests

- [x] Update `particula/gpu/tests/gpu_direct_kernels_example_test.py` for exact output, explicit helper calls, reusable buffers, no-Warp behavior, failure propagation, and guarded Warp-CPU state, energy, and gas-coupling checks. Completed by Issue #1315.
- [x] Revise `particula/tests/condensation_latent_heat_docs_test.py` to recognize shipped low-level GPU support while retaining CPU-notebook and no-high-level-backend guardrails.
- [x] Add documentation assertions for direct kernel imports, no hidden transfer/fallback, four substeps, Warp CPU baseline, optional CUDA, fixed-shape fp64, and unsupported modes.
- [x] Validate the revised example's published path and focused command against its final test layout: `pytest particula/gpu/tests/gpu_direct_kernels_example_test.py -q -Werror`. Completed by Issue #1315.
- [ ] Run the complete cross-phase example, GPU contract, stiffness, CPU reference, documentation, and warning-clean suite set; record optional CUDA skips separately.
