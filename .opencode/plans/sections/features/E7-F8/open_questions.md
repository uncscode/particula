# Open Questions

- [ ] Does a repeated root seed reset an existing resident stream?
  - Deferred: P1 has no resident ownership or reset API. Calling the direct
    `StreamRegistry.initialize()` again deterministically overwrites its two
    caller-owned arrays; P4 must define lifecycle-valid reset semantics.

- [x] Must restart reproduce trajectories across CPU, Warp CPU, and CUDA?
  - Resolved 2026-07-27: No. Exact continuation applies only to a compatible
    same-backend/device-class restart; cross-backend validation remains
    statistical and conservation based.
  - Evidence: issue #1451 explicitly excludes exact CPU/CUDA stochastic equality.

- [ ] Is RNG state included in a valid E7-F4 checkpoint?
  - Deferred: Issue #1520 introduces no checkpoint payload, checkpoint binding,
    or restart behavior. A later E7-F8 phase must define and validate any stream
    metadata/state persistence contract.

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
  - Resolved 2026-07-27: No. Store logical stream metadata and synchronized RNG
    words in the versioned in-memory checkpoint; defer any file encoding.
  - Rationale: Current checkpoint boundaries restore CPU objects, while durable
    format, migration, and filesystem guarantees are explicitly out of scope.
  - Evidence:
    - `.opencode/plans/sections/features/E7-F4/scope.md:40` - disk/file
      serialization formats are excluded.
    - `particula/gpu/conversion.py:422` - current checkpoint helpers return
      in-memory CPU containers rather than encoded artifacts.
  - Resolved by: plan-question-resolver
