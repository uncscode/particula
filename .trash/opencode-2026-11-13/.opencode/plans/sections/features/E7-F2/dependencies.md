# Dependencies

## Upstream Plan Gates

- **E7-F1 — Backend-selection and execution-context API:** supplies typed
  backend/process/device/capability requests, adapter registration, execution
  state/result semantics, CPU reference delegation, and neutral imports.
- **E7-F6 — Fallback, capability errors, exports, and API stability:** freezes
  backend availability checks, explicit transition rules, unsupported-process
  errors, and which integration APIs may be public.

Implementation must not start against guessed versions of these contracts.
Authoritative order is
`E7-F1 -> E7-F6 -> {E7-F2, E7-F3, E7-F4}`.

## Shipped Technical Dependencies

- CPU `MassCondensation` and `CondensationStrategy` implementations.
- `condensation_step_gpu`, `ThermodynamicsConfig`,
  `CondensationActivitySurfaceConfig`, and `CondensationScratchBuffers`.
- Fixed-shape CPU/Warp particle, gas, and environment containers and explicit
  conversion boundaries from E2.
- NumPy CPU references and Warp CPU; CUDA hardware is optional.

## Downstream Consumers and Siblings

- **E7-F3** is a parallel sibling and should follow the same adapter/result and
  no-fallback conventions without coupling condensation to RNG concerns.
- **E7-F4** will own resident state and reusable sidecar lifecycle; E7-F2 must
  expose ownership requirements without implementing sessions.
- **E7-F5** depends on E7-F2/F3/F4 and will schedule environment updates,
  vapor-pressure refresh, condensation, and later processes deterministically.
- **E7-F9** owns broad full-loop closeout evidence; E7-F2 still ships all
  condensation unit, contract, and bounded parity coverage with its phases.

## External Dependencies

Python 3.12+, NumPy, and optional NVIDIA Warp. Warp CPU is the required routine
GPU-contract backend; CUDA remains optional evidence and must skip cleanly.
