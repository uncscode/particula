# Architecture Design

## High-Level Design

### Implemented P1/P2/P3 seams

`GPUResourceRegistry.logical_resource_report()` is now the read-only,
direct-module-only inventory seam. Given explicit collision, GAS-edge, and
PARTICLES-edge capacities, it resolves every role in the existing six manifests
(including both communication families) into frozen pointer-free role, family,
and aggregate reports. It uses pinned dimensions, declaration metadata,
`_shape()`, and shared checked accounting helpers only; it neither allocates nor
acquires resources, reads payloads, consults bindings/configurations, or mutates
the registry. Logical bytes are schema bytes rather than allocator-reserved
bytes. The symbolic manifests and package/top-level exports are unchanged.

The registry remains the intended sole storage authority. E8-F2 supplies a
closed requirement description; P4 will resolve it against canonical manifests,
stage every array and native view, validate the whole candidate for session
ownership and nonaliasing, and publish one immutable capture resource set.

P2 adds `PreparedResourceViews` as the concrete prepared-consumer carrier.
The descriptor-only dilution family resolves two `(B,)` `wp.float64` roles:
normalized coefficient and factors. Before a prepared adapter uses a supplied
view, registry schema validation reads metadata only and rejects incompatible
views. A valid adapter retains each supplied resource identity; it does not
allocate, publish, replace, reacquire, or initialize/reset RNG state.

P3 adds `register_capture_resources()` and `selected_resource_report()` on the
concrete registry boundary. Registration accepts only the exact active session,
`None` or an already-published closed GAS/PARTICLES communication view, and an
ordered tuple of diagnostic registrations. It builds and retains immutable role
metadata and reports once, then exact repeats return the retained inventory.
All candidate validation precedes publication; a sorted per-allocation interval
sweep detects forbidden nonempty overlaps in O(R log R), while permitted
read-only accounting-input aliases remain valid. Neither method allocates,
launches, transfers, synchronizes, reads payloads, changes checkpoint resource
enumeration, nor changes scheduler behavior.

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

The diagram describes the deferred P4 whole-set path. Candidate allocation may
happen before publication, so failure cannot leave a partially usable capture
set. Once published, all compatible accessors will return the same set and
nested views by identity. Incompatible requirements, capacities,
communication modes/maps, diagnostic bindings, session drift, or array
replacement will fail closed before replay.

## Data / API / Workflow Changes

- **Data model:** P1 added separate frozen concrete-only inventory and byte
  report records keyed to the unchanged manifests, with declaration-level
  capacity-source and ownership metadata. Byte counts are logical bytes
  (`product(shape) * item_size`), not observed allocator reservation, and use
  checked integer arithmetic. P2 adds concrete-only `PreparedResourceViews`
  and dilution normalized-control/factor descriptors. P3 adds a selected
  communication/diagnostic inventory that retains references plus resolved
  metadata only; whole capture-set carriers remain future work.
- **API surface:** P2 adds private/concrete prepared-view validation and
  retention plumbing; P3 adds direct-module-only registration/reporting methods
  on `GPUResourceRegistry`. Keep them direct-import-only; do not alter
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
- **Failure model:** P2 supplied-view validation is read-only and rejects before
  prepared adapter use. P3 candidate construction and rejection are
  transactional: the first inventory is assigned only after all validation and
  metadata work succeeds, and nonexact repeats retain the existing inventory.
  Whole-set setup/allocation and RNG writer semantics remain future-phase work.

## Security & Compliance

There are no network, credential, or permission changes. Robustness is
fail-closed: exact types, devices, shapes, capacities, contiguous storage,
nonoverlapping byte ranges, session lifecycle, and identities are validated
before publication or replay. Reports expose metadata and counts only, never
device pointers, payloads, or RNG words. Arithmetic must reject overflow rather
than undercounting memory. No hidden transfer, synchronization, CPU fallback,
device selection, retry, or automatic recapture is introduced.
