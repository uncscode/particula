# Documentation Updates

- `docs/Features/data-containers-and-gpu-foundations.md` — published the canonical bounded direct-condensation contract: supported modes, schemas, explicit ownership, four coupled substeps, P2-finalized transfer semantics, diagnostics, validation/rollback boundaries, device policy, and explicit non-goals.
- `docs/Features/particle-data-migration.md` — replaced obsolete direct-call guidance with the required `thermodynamics=` signature, two-item return, in-place gas mutation, explicit conversion/restore ownership, accepted environment inputs, and a link to the canonical page.
- `particula/tests/condensation_latent_heat_docs_test.py` — added text-only publication regressions for the foundations configuration, lifecycle, validation/boundaries, and migration contracts; existing notebook and CPU-publication checks remain.

Issue #1314 intentionally did not update examples, indexes, README, roadmaps, troubleshooting material, runtime APIs, kernels, or containers.

- `docs/Examples/gpu_direct_kernels_quick_start.py` — Issue #1315 replaced the mixed walkthrough with a condensation-only, Warp-CPU-default quick-start. It lazily loads the public step and concrete sidecar types, explicitly converts/restores CPU data, reuses complete fp64 scratch plus latent-heat and energy sidecars for two calls, and returns inspectable final checkpoints.
- `docs/Examples/index.md` — described the quick-start as the low-level explicit-transfer, gas-coupled direct-condensation path without claiming hidden transfers, high-level integration, or required CUDA.
- `particula/gpu/tests/gpu_direct_kernels_example_test.py` — added focused no-Warp import isolation, lazy-loader, mocked sidecar/reuse, failure-path, and guarded real Warp-CPU gas/energy-coupling regressions.

Issue #1315 intentionally left the README, roadmap, canonical feature pages, runtime APIs, kernels, and containers unchanged.

- `docs/Features/data-containers-and-gpu-foundations.md` — Issue #1316 added direct-condensation troubleshooting for ordered species metadata, layouts/devices/fp64 sidecars, environment/direct-input validation, P2 inventory limiting, synchronization, and Warp/CUDA availability. It also added the canonical focused reproduction command matrix, with Warp `device="cpu"` as baseline and optional/local CUDA evidence.
- `docs/Features/particle-data-migration.md` — added a concise direct-condensation troubleshooting pointer that preserves the explicit low-level migration scope and links once to the canonical command matrix.
- `readme.md` — added one discovery link to the canonical GPU condensation command matrix.
- `particula/tests/condensation_latent_heat_docs_test.py` — added isolated text-only regressions for scoped troubleshooting language, command paths and flags, evidence-class separation, and migration/README link constraints; they do not import Warp or CUDA.

Issue #1316 changed documentation and text-only regression coverage only; no runtime APIs, kernels, containers, examples, or roadmap status changed.

- `docs/Examples/index.md` and `readme.md` — Issue #1317 added one distinct
  canonical low-level direct-condensation-contract discovery link per surface
  while retaining the P2 quick-start source and the single P3 command-matrix
  anchor.
- `docs/Features/Roadmap/data-oriented-gpu.md` — published `Shipped | E4` and
  a completed bounded direct-kernel summary: explicit transfers, caller-owned
  fp64 sidecars, four equal substeps, P2 coupling, Warp CPU baseline, and
  optional/local CUDA evidence. High-level integration remains future work.
- `particula/tests/condensation_latent_heat_docs_test.py` — added isolated
  text-only discovery-link and Epic D shipped-boundary regressions.

Issue #1317 completed E4-F7-P4 after inspecting the E4-F1 through E4-F6 and
E4-F7-P1 through P3 records and passing `pytest
particula/tests/condensation_latent_heat_docs_test.py -q -Werror` (22 passed).
No production kernel, container, or API changed.
