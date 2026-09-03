# Architecture Design

## High-Level Design

### Implemented P1--P5 seams

`GPUResourceRegistry.logical_resource_report()` is now the read-only,
direct-module-only inventory seam. Given explicit collision, GAS-edge, and
PARTICLES-edge capacities, it resolves every role in the existing six manifests
(including both communication families) into frozen pointer-free role, family,
and aggregate reports. It uses pinned dimensions, declaration metadata,
`_shape()`, and shared checked accounting helpers only; it neither allocates nor
acquires resources, reads payloads, consults bindings/configurations, or mutates
the registry. Logical bytes are schema bytes rather than allocator-reserved
bytes. The symbolic manifests and package/top-level exports are unchanged.

The registry remains the intended sole storage authority. P4 resolves the
closed E8-F2 requirement description against canonical manifests, stages every
array and native view privately, validates the candidate for session ownership
and nonaliasing, and publishes one immutable capture resource set only after
successful completion.

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

P4 implements this whole-set path. Candidate allocation happens before
publication, so a candidate failure leaves no partially usable capture set and
preserves prior registry/P3/ordinary-publication identities; a clean retry is
permitted. Once published, `validate_capture_resource_set()` and compatible
preparation return the same set and nested views by identity without resource
work. Incompatible requirements, capacities,
communication modes/maps, diagnostic bindings, session drift, or array
replacement will fail closed before replay.

P5 makes publication a precondition rather than an optional enqueue seam.
Callers register inventory, construct exact views/capacities and requirements,
publish once, and only then construct `ResidentSimulationRequest`. Capture and
READY validation use the cached registry association and fixed identity checks;
they retain the same requirements, set, and immutable report in signatures and
prepared carriers. Missing, partial, stale, or identity-distinct publication
rejects before token entry or dispatch. A changed valid publication is
`CONFIGURATIONS` drift for CAPTURED admission and a READY rejection without a
lifecycle transition. Repeated compatible use does not recompute reports or
inventory, acquire/prepare resources, read payloads, synchronize, or alter RNG.

## Data / API / Workflow Changes

- **Data model:** P1 added separate frozen concrete-only inventory and byte
  report records keyed to the unchanged manifests, with declaration-level
  capacity-source and ownership metadata. Byte counts are logical bytes
  (`product(shape) * item_size`), not observed allocator reservation, and use
  checked integer arithmetic. P2 adds concrete-only `PreparedResourceViews`
  and dilution normalized-control/factor descriptors. P3 adds a selected
  communication/diagnostic inventory that retains references plus resolved
   metadata only. P4 adds frozen direct-module-only
   `CaptureResourceRequirements` and `CaptureResourceSet` carriers, retained
   by exact identity and without payload/RNG-word copies.
- **API surface:** P4 adds direct-module-only
   `prepare_capture_resources()` and `validate_capture_resource_set()` to
   `GPUResourceRegistry`. The latter is metadata-only; both preserve exact
   retained identities. The optional E8-F2 seam retains/compares a supplied P4
   set and prepared views only. Keep all of these direct-import-only; do not alter
  `particula.execution.__all__`, `particula.gpu.kernels.__all__`, or top-level
  exports. Existing family acquisition and checkpoint enumeration stay valid.
- **Workflow hooks:** E8-F2 preparation must declare complete requirements and
   consume the published set before final request construction. E8-F1 CAPTURED
   admission and E8-F2 READY preparation require its exact identity/signature,
   including `(requirements, set, report)` in `configurations`. E8-F4 validates
   behavior; E8-F5 and E8-F6 consume the
  canonical inventory and byte report rather than rebuilding formulas.
- **Ownership:** Registry-allocated arrays live for the registry/capture
  lifetime. Accepted caller-supplied arrays remain caller-owned but are pinned
  by exact identity and may not be replaced. Diagnostic arrays become part of
  the capture set only when selected by the closed prepared plan.
- **Failure model:** P2 supplied-view validation is read-only and rejects before
  prepared adapter use. P3 candidate construction and rejection are
  transactional: the first inventory is assigned only after all validation and
  metadata work succeeds, and nonexact repeats retain the existing inventory.
   P4 candidate construction is transactional: nonpublishing staged resources
   and only newly created RNG streams are initialized before a final, nonfallible
   publication. Rollback after an unexpected launched device writer remains
   outside the guarantee.

## Security & Compliance

There are no network, credential, or permission changes. Robustness is
fail-closed: exact types, devices, shapes, capacities, contiguous storage,
nonoverlapping byte ranges, session lifecycle, and identities are validated
before publication or replay. Reports expose metadata and counts only, never
device pointers, payloads, or RNG words. Arithmetic must reject overflow rather
than undercounting memory. No hidden transfer, synchronization, CPU fallback,
device selection, retry, or automatic recapture is introduced.
