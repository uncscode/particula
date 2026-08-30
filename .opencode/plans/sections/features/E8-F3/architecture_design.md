# Architecture Design

## High-Level Design

The existing registry remains the sole storage authority. E8-F2 supplies a
closed requirement description; E8-F3 resolves it against canonical manifests,
stages every array and native view, validates the whole candidate for session
ownership and nonaliasing, and publishes one immutable capture resource set.

```text
exact ACTIVE ResidentSession + E8-F2 prepared requirements
                         |
                         v
GPUResourceRegistry.prepare_capture_resources()       SETUP ONLY
  resolve canonical role/capacity inventory
  validate supplied arrays and protected primaries
  allocate every omitted process/control/communication/diagnostic role
  initialize selected RNG streams explicitly
  validate complete cross-family nonaliasing and identity graph
  compute checked logical byte records in canonical order
                         |
                   atomic publish
                         v
CaptureResourceSet (frozen view; retains exact native records/arrays)
  + CaptureResourceByteReport (role -> family -> total logical bytes)
                         |
            +------------+------------+
            |                         |
            v                         v
E8-F2 prepared enqueue          E8-F1 capture/replay
exact views only                exact signature only
no acquire/allocate             no replacement/resize
```

Candidate allocation may happen before publication, so failure cannot leave a
partially usable capture set. Once published, all compatible accessors return
the same set and nested views by identity. Incompatible requirements,
capacities, communication modes/maps, diagnostic bindings, session drift, or
array replacement fail closed before replay.

## Data / API / Workflow Changes

- **Data model:** Extend manifest metadata only as needed to identify capacity
  sources and ownership. Add frozen concrete-only capture inventory and byte
  report records. Byte counts are logical bytes (`product(shape) * item_size`),
  not observed allocator reservation, and use checked integer arithmetic.
- **API surface:** Add setup-only and validation/accessor methods on
  `GPUResourceRegistry`. Keep them direct-import-only; do not alter
  `particula.execution.__all__`, `particula.gpu.kernels.__all__`, or top-level
  exports. Existing family acquisition and checkpoint enumeration stay valid.
- **Workflow hooks:** E8-F2 preparation must declare complete requirements and
  consume the published set. E8-F1 READY/capture transition requires its exact
  identity/signature. E8-F4 validates behavior; E8-F5 and E8-F6 consume the
  canonical inventory and byte report rather than rebuilding formulas.
- **Ownership:** Registry-allocated arrays live for the registry/capture
  lifetime. Accepted caller-supplied arrays remain caller-owned but are pinned
  by exact identity and may not be replaced. Diagnostic arrays become part of
  the capture set only when selected by the closed prepared plan.
- **Failure model:** Setup validation/allocation failures publish nothing and
  are retryable while the session remains active. RNG initialization follows
  existing writer-failure semantics. No rollback is promised after a device
  writer launches; replay failure handling remains owned by E8-F1/E8-F2.

## Security & Compliance

There are no network, credential, or permission changes. Robustness is
fail-closed: exact types, devices, shapes, capacities, contiguous storage,
nonoverlapping byte ranges, session lifecycle, and identities are validated
before publication or replay. Reports expose metadata and counts only, never
device pointers, payloads, or RNG words. Arithmetic must reject overflow rather
than undercounting memory. No hidden transfer, synchronization, CPU fallback,
device selection, retry, or automatic recapture is introduced.
