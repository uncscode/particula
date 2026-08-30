# Dependency Map

## Inbound

- **Epic G / E7 resident execution:** `particula.execution.gpu_session`,
  `gpu_resources`, `resident_scheduler`, `resident_communication`,
  `diagnostics`, and `checkpoint` define the stable session and ownership
  boundary.
- **Direct Warp kernels:** stable-shape condensation, coagulation, dilution,
  wall-loss, nucleation, communication, and refresh operations provide the
  executable process nodes.
- **Fixed-capacity policies:** E6-F5/E6-F6 slot activation and exhaustion
  behavior allow active populations to change without resizing captured arrays.
- **Persistent RNG policy:** E7-F8 stream ownership and schema-v3 checkpoint
  continuation define initialization, reset, replay, and restart semantics.
- **Existing benchmark policy:** opt-in `--benchmark` collection and CUDA
  availability gates are the baseline for performance evidence.

## Outbound

- Epic I differentiability and global optimization can use the recorded stable
  execution boundary and memory allowance, but must define its own tape and
  gradient contracts.
- Future high-level GPU interfaces may consume the evidence but may not infer
  hidden capture, fallback, or automatic recapture behavior.

## Sequencing

1. E8-F1 establishes the bounded captured-loop lifecycle.
2. E8-F2 freezes the setup/replay boundary against hidden host work.
3. E8-F3 completes reusable resource ownership required for reliable replay.
4. E8-F4 captures the prepared sequence and provides guarded replay after
   E8-F1--E8-F3 are stable.
5. E8-F5 validates CPU, uncaptured GPU, and captured GPU correctness after
   E8-F4; E8-F6 may proceed after the executable path and resource inventory
   are fixed and gates benchmark rows on E8-F5 correctness evidence.
6. E8-F7 profiles only the correctness-qualified path and consumes E8-F6
   timing and memory evidence.
7. E8-F8 publishes the runnable workflow, runbook, and epic closeout; it
   depends on all preceding tracks.
