# Open Questions

- [x] Does a repeated root seed reset an existing resident stream?
  - Resolved 2026-08-09: No. P2 stores immutable P1 metadata on the session and
    initializes the resident coagulation sidecar only on first acquisition;
    compatible reacquisition and resident dispatch never reseed it. A generic
    lifecycle-valid reset API remains deferred to P4.

- [x] Must restart reproduce trajectories across CPU, Warp CPU, and CUDA?
  - Resolved 2026-07-27: No. Exact continuation applies only to a compatible
    same-backend/device-class restart; cross-backend validation remains
    statistical and conservation based.
  - Evidence: issue #1451 explicitly excludes exact CPU/CUDA stochastic equality.

- [x] Is RNG state included in a valid E7-F4 checkpoint?
  - Resolved 2026-08-09: No. P2 checkpoint and finalize fail closed before
    payload conversion when the resident coagulation sidecar has been published.
    No stream metadata or words are serialized, and restart continuation is
    unsupported; a future persistence design requires a separate contract.

- [x] What public logical box ID type should the first stable API accept?
  - Resolved 2026-07-27: Accept unique, non-empty UTF-8 strings and define a
    documented finite encoded-byte length limit in the public contract.
  - Rationale: Semantic identifiers remain host-owned string metadata rather
    than device-row fields, while a finite limit bounds validation and checkpoint
    resource use.
  - Evidence:
    - `particula/gpu/conversion.py:490` - semantic species identity is already
      caller-owned string metadata outside numeric Warp containers.
  - Resolved by: PR #1452 decision

- [x] How should disabled per-box execution reach kernels that currently launch
  across all boxes?
  - Resolved 2026-07-27: Add an optional same-device binary `wp.int32` enable
    mask shaped `(n_boxes,)`; false lanes must return before physical-state or
    RNG reads/writes, and omission means all boxes enabled.
  - Rationale: A fixed-shape mask preserves row identity and stable launches,
    unlike gather/scatter compaction that would remap stochastic streams.
  - Evidence:
    - `particula/gpu/warp_types.py:135` - binary GPU masks use `wp.int32`.
    - `docs/Features/Roadmap/data-oriented-gpu.md:1596` - disabled boxes must not
      perturb independent per-box streams.
  - Resolved by: plan-question-resolver

- [x] Is durable on-disk serialization part of the first RNG checkpoint schema?
  - Resolved 2026-08-09: No. There is currently no RNG checkpoint schema:
    published resident RNG state rejects checkpoint/finalize and is neither
    serialized nor restartable. Any future in-memory representation must still
    defer file encoding.
  - Rationale: Current checkpoint boundaries restore CPU objects, while durable
    format, migration, and filesystem guarantees are explicitly out of scope.
  - Evidence:
    - `.opencode/plans/sections/features/E7-F4/scope.md:40` - disk/file
      serialization formats are excluded.
    - `particula/gpu/conversion.py:422` - current checkpoint helpers return
      in-memory CPU containers rather than encoded artifacts.
  - Resolved by: plan-question-resolver
