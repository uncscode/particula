# Dependencies

## Upstream

- **E8 parent:** Supplies CUDA-only capture/profiling guardrails, success
  metrics, and the requirement to preserve scientific contracts.
- **E8-F1:** Captured resident lifecycle and compatibility signature.
- **E8-F2:** Prepared enqueue boundary that separates setup from replay.
- **E8-F3:** Stable registry-owned buffers, identities, and logical-byte report.
- **E8-F4:** Executable captured full loop and deterministic invalidation rules.
- **E8-F5:** Correctness-qualified CPU/uncaptured/captured workload fixtures.
- **E8-F6:** Representative scaling matrix, raw replay timings, budget checks,
  memory evidence, artifact metadata, and explicit unavailable rows.
- **Epic G / E7:** Resident scheduler, communication, diagnostics, checkpoints,
  and persistent RNG contracts remain semantic authority.
- **External optional tools:** A qualified NVIDIA CUDA device and supported
  Nsight Systems/Compute versions are required for corresponding metric rows.

## Downstream

- Epic H closeout consumes the profile, recommendation table, command matrix,
  and explicit unavailable/failed rows.
- Follow-up optimization issues may consume one bounded recommendation but must
  retain separate correctness, parity, and scientific-review gates.
- Epic I may use bottleneck and memory evidence as planning input; it may not
  infer autodiff performance or tape behavior from this feature.

## Phase Ordering

P1 freezes records and workload IDs before P2 or P3 emits artifacts. P2 and P3
may then run independently on identical fixtures. P4 requires both artifact
families or explicit unavailable records. P5 publishes only verified P1-P4
outputs. Profiling starts only after E8-F4 through E8-F6 contracts are stable.
The E8 child-plan table and dependency map assign profiling to E8-F7; E8-F8
consumes only E8-F7's verified normalized artifacts, provenance, and bounded
recommendations for its example, runbook, and closeout gates.
