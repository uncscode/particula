# Scope

E7-F3 adds the issue #1451 T3 adapter and contracts for backend-selected
Brownian coagulation after E7-F1 and E7-F6. It preserves the existing CPU
reference and direct-Warp implementations while making state, outputs, RNG
ownership, validation, mutation, and unsupported behavior explicit.

## In Scope

- Declare Brownian coagulation capabilities for CPU and Warp backends through
  the E7-F1 execution context and E7-F6 availability/error policy.
- Adapt the CPU `Coagulation` runnable with a Brownian strategy without changing
  its `Aerosol`, `time_step`, or `sub_steps` behavior.
- Adapt the shipped particle-resolved direct Warp `coagulation_step_gpu` path
  with explicit environment/volume inputs and Brownian-only configuration.
- Model caller-owned `collision_pairs`, `n_collisions`, and per-box
  `wp.uint32` RNG state, including seed-once, explicit reset, and reuse rules.
- Preserve particle and supplied-buffer identity and report mutation/output
  metadata through the E7-F1 result vocabulary.
- Reject unsupported distributions, mechanisms, devices, malformed state, and
  invalid time/configuration before adapter invocation or mutation where the
  selection layer owns validation.
- Add CPU/Warp CPU bounded parity, conservation, stochastic, persistence,
  identity, negative, and no-transfer tests; make CUDA rows optional.
- Keep the P3 adapter concrete-only: do not add public documentation, exports,
  conversion, synchronization, fallback, or API handoffs in this phase.

## Out of Scope

- Charged, sedimentation, turbulent-shear, combined, or three-way coagulation
  selection, even where a direct kernel can execute some configurations.
- Exact CPU/Warp/CUDA random-trajectory equality or a shared cross-backend RNG
  algorithm; validation is statistical and invariant-based.
- Session-wide RNG stream identity, box reorder/disable guarantees, and
  checkpoint/restart policy, which belong to E7-F8.
- Resident session allocation/checkpoint ownership (E7-F4) and deterministic
  full-process scheduling (E7-F5).
- Hidden CPU/GPU transfer, automatic synchronization, silent fallback, dynamic
  resizing/compaction, graph capture, performance claims, or physics rewrites.
