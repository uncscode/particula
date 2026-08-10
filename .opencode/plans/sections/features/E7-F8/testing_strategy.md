# Testing Strategy

Every production phase ships with co-located `*_test.py` tests. Coverage
thresholds are not lowered; changed modules must retain at least 80% coverage.

## Per-Phase Tests

- **P1 (complete, Issue #1520):** `particula/execution/tests/rng_test.py` covers
  frozen carriers, strict UTF-8 IDs, integral metadata, canonical manifests,
  lane permutations, collisions, immutable lookup, FNV vectors, and stable-ID
  reorder/unrelated-ID invariance. A subprocess import hook proves host-only
  registration does not load Warp or `particula.gpu`; Warp-guarded rows cover
  caller-buffer identity, initialization, and schema preflight preservation.
  Execution export tests deny the direct-only module from package and top-level
  export surfaces.
- **P2 (complete, Issue #1521):** `gpu_session_test.py`, `gpu_resources_test.py`,
  `coagulation_adapter_test.py`, `coagulation_integration_test.py`,
  `scheduler_test.py`, and `checkpoint_test.py` cover P1 metadata validation;
  first-acquisition-only coagulation-sidecar initialization and identity;
  literal-false resident dispatch; consecutive advancement and zero-time/no-active
  preservation; rejection-before-import/dispatch; and fail-closed checkpoint and
  finalize behavior after sidecar publication.
- **P3 (complete, Issue #1522):** `gpu_resources_test.py`,
  `process_adapters_test.py`, `scheduler_test.py`, `checkpoint_test.py`, and
  direct `wall_loss_test.py` cover canonical-manifest wall-loss initialization,
  distinct namespace/array identity and nonaliasing, transactional acquisition
  and repeat acquisition without reseeding, scheduler-selected one-box dispatch,
  disabled/empty/prelaunch-skipped/zero-time/no-work preservation, reset-like
  request rejection, checkpoint guard, and post-dispatch scheduler faulting.
- **P4 (complete, Issue #1523):** `rng_test.py`, `gpu_resources_test.py`,
  `gpu_session_test.py`, and execution export tests cover frozen host manifests;
  complete and selected process/box resets; exact-tuple, duplicate, malformed,
  unregistered, and unpublished selector rejection before writers; full-schema
  preflight preservation; ACTIVE closed-binding restrictions; empty-published
  write-free behavior; identity retention; repeated-seed non-reset; and guards
  against readback, synchronization, scheduling, acquisition, or allocation.
- **P5 (complete, Issue #1524, commit `727f8b471`):**
  `particula/execution/tests/rng_invariance_test.py` compares the `box-a`
  logical lane with an isolated same-device reference across active, removed,
  no-work, and physically permuted unrelated-box arrangements. It covers
  resident Brownian particle, collision-output, and RNG snapshots; selected
  neutral wall-loss particle and RNG snapshots; zero-time and empty-selection
  no-ops; prelaunch `initialize_rng=True` rejection; and distinct process
  sidecar identity. Warp CPU is required and CUDA rows remain optional.
- **P6 (complete, Issue #1525):** `checkpoint_test.py` and
   `gpu_resources_test.py` cover schema-v3 absent, single-stream, and
   two-stream continuation; exact Warp-CPU split-run equivalence for Brownian
   coagulation and neutral wall loss; fresh restored identities; no reseed on
   restart/reacquisition/dispatch; and explicit-reset-only rederivation. They
   retain v1/v2 and v2 communication compatibility, assert one capture
   synchronization and bounded stream readbacks, reject malformed continuation
   records before setup/allocation, and exercise capture/readback and fresh
   restart construction recovery lifecycles. CUDA remains optional.
- **P7 (complete, Issue #1526):**
  `gpu_resident_session_docs_test.py` reads the final foundations, checkpoint,
  roadmap, contributor reference, and feature narratives with UTF-8 normalized
  prose. It verifies the documented ownership and continuation boundaries plus
  the shipped feature record; `mkdocs build --strict` checks publication links.

## Validation Matrix

- Warp CPU is the required stochastic baseline when Warp is installed; CUDA is
  optional and skips cleanly when unavailable.
- Re-run direct kernel tests in
  `particula/gpu/kernels/tests/coagulation_test.py` and
  `particula/gpu/kernels/tests/wall_loss_test.py` to prevent API or RNG regressions.
- Use exact comparisons only within the same backend/device class and frozen
  configuration. Cross-backend behavior uses existing statistical,
  conservation, and tolerance-based evidence; exact CPU/CUDA replay is excluded.
- Snapshot particle, gas, outputs, RNG arrays, counters, and identities around
  every expected pre-launch failure. Post-launch failures assert session faulting
  rather than rollback.
- Transfer spies reject `to_warp_*`, `from_warp_*`, `.numpy()`, or
   `wp.synchronize()` during normal scheduler steps; checkpoint capture is the
   explicit one-synchronization/readback exception.
