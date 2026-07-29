# Open Questions

- [x] Should backend selection extend `RunnableABC` or introduce a separate
  typed execution/session protocol?
  - Resolved 2026-07-27: Introduce a separate typed execution/session protocol
    and retain `RunnableABC` as the CPU `Aerosol` adapter contract.
  - Rationale: The existing runnable accepts one `Aerosol`, while the planned
    resident path owns particle, gas, environment, work-buffer, and RNG state.
  - Evidence:
    - `particula/runnable.py:36` - `RunnableABC` is explicitly an `Aerosol`
      process abstraction.
    - `docs/Features/Roadmap/data-oriented-gpu.md:1524` - the GPU loop must keep
      three typed Warp containers resident across enabled dynamics.
  - Resolved by: plan-question-resolver

- [x] What is the minimum versioned checkpoint payload?
  - Resolved 2026-07-29: Store schema version plus immutable canonical bytes for
    every primary resident array, including GPU vapor pressure, and all acquired
    sidecars. Detached CPU inspection carriers and ordered gas-name metadata are
    included for inspection only; they are not restart authority.
  - Rationale: Inspection carriers are intentionally lossy, so exact same-device
    restart reconstructs fresh resident state from canonical bytes rather than a
    CPU restore. Acquired sidecars are checkpoint payload, not reconstructed
    scratch state.
  - Evidence:
    - `particula.execution.checkpoint` preserves canonical payload descriptors
      and bytes for primary arrays and acquired sidecars.
    - `docs/Features/gpu_resident_checkpoints.md` documents lossy inspection and
      canonical-byte restart authority.
  - Resolved by: plan-question-resolver

- [x] Where should explicit CPU fallback occur?
  - Resolved 2026-07-29: No complete CPU fallback or restore is implemented at
    the resident checkpoint boundary. Restart is explicit, same-device only, and
    reconstructs fresh resident state from canonical checkpoint bytes.
  - Rationale: Detached CPU inspection carriers are lossy and cannot serve as a
    complete restore source. The boundary never selects a device, migrates state,
    or falls back inside an adapter or scheduler step.
  - Evidence:
    - `docs/Features/gpu_resident_checkpoints.md` documents no CPU fallback or
      migration and exact-device restart.
  - Resolved by: plan-question-resolver

- [x] Which transport map representation best preserves fixed-shape and
  deterministic execution?
  - Resolved 2026-07-27: Use a fixed-capacity canonical edge list with canonical
    source/destination order and an active-edge count.
  - Rationale: This preserves stable allocation and deterministic traversal while
    avoiding dense storage for prescribed low-degree communication graphs.
  - Evidence:
    - `docs/Features/Roadmap/data-oriented-gpu.md:1573` - both fixed-shape maps
      and sparse edge lists are explicitly allowed.
  - Resolved by: PR #1452 decision

- [x] How are stream identities preserved when boxes are reordered?
  - Resolved 2026-07-27: Key each process stream by a stable caller-provided
    logical box ID plus a versioned process namespace, checkpoint that mapping,
    and permute row-aligned physical and RNG state together without reseeding.
  - Rationale: Current device rows are positional, so row index alone cannot
    preserve stream identity under reorder, insertion, or disabling.
  - Evidence:
    - `docs/Features/Roadmap/data-oriented-gpu.md:1592` - independent box streams
      must remain reproducible when box count or enablement changes.
    - `particula/gpu/warp_types.py:164` - environment state is row-batched and
      contains no logical box identifier.
   - Resolved by: plan-question-resolver

- [x] What checkpoint compatibility boundary is implemented for resident
  restart?
  - Resolved 2026-07-29: Restart fails closed unless the checkpoint is schema
    version `1`, carrier type `ResidentSession`, ACTIVE with complete valid
    payloads, and the target `Device` is exactly equal to the source device.
  - Rationale: Inspection carriers are lossy, so compatibility must be based on
    canonical payload bytes and explicit schema/device checks rather than a
    best-effort CPU restore or migration.
  - Evidence: E7-F4-P7 documentation and regression coverage in
    `particula/execution/tests/gpu_resident_session_docs_test.py`.
  - Resolved by: issue #1490
