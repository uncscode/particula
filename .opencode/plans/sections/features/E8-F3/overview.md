# Overview

## Problem Statement

The shipped `GPUResourceRegistry` pins major condensation, coagulation,
wall-loss, nucleation, and communication sidecars, but capture-ready execution
needs one complete inventory before capture begins. Validation status arrays,
normalized controls, selected-lane work, diagnostics, and other prepared-node
temporaries must not fall back to per-call allocation. The capture lifecycle
also needs a deterministic way to prove that the same arrays are reused and to
report their logical byte cost without reading device payloads.

## Current Increment

E8-F3-P1 shipped in issue #1561. `GPUResourceRegistry` now exposes a
direct-module-only immutable logical inventory/report for the six existing
canonical manifests. It resolves declared shapes, dtype, capacity provenance,
ownership, elements, and logical bytes without allocating, acquiring a resource
family, inspecting payloads, or changing bindings/configuration state. Checked
shape, stride, element, and byte arithmetic is shared by reporting, allocation,
and range validation; focused registry tests pass. Package and top-level exports
remain unchanged.

E8-F3-P2 shipped in issue #1562. The registry now defines the concrete-only
dilution descriptor family: `(B,)` `wp.float64` normalized-coefficient and
factor roles. `PreparedResourceViews` carries supplied prepared resources, and
registry validation read-only checks those views before use. Prepared adapters
retain the supplied resource identities exactly. This increment adds neither
allocation, publication, reacquisition, nor RNG behavior, and it adds no public
exports.

E8-F3-P3 shipped in issue #1563. `GPUResourceRegistry` now registers one
concrete-only capture selection: absent communication or one exact,
already-published closed GAS/PARTICLES view, plus ordered diagnostic
registrations. The retained inventory contains exact references and
deterministic schema/logical-byte reports. Registration is transactional and
host-metadata-only; its O(R log R) byte-interval sweep rejects forbidden
overlaps without payload I/O. Checkpoint enumeration, scheduler behavior, and
public exports are unchanged.

E8-F3-P4 shipped in issue #1564. `GPUResourceRegistry` now atomically stages
and publishes one concrete-only `CaptureResourceSet` from exact
`CaptureResourceRequirements`. The setup transaction validates metadata and
identities, stages omitted resources without publication, protects existing
bindings on failure, and returns the original set by identity on a compatible
repeat. `validate_capture_resource_set()` is a metadata-only retained-set
accessor. The narrowed optional resident-enqueue seam can retain and validate
the exact set/views, without adding READY admission or changing token/dispatch
behavior.

## Value Proposition

E8-F3 incrementally makes the registry the concrete authority for
capture-lifetime reusable storage. P1 supplies the inventory, P2 adds the
dilution descriptor/view validation seam, P3 pins selected communication and
diagnostic resources, and P4 supplies atomic whole-set preparation and exact
identity reuse. P5 retains broader admission and prepared-path policy work.

## User Stories

- As a graph-capture integrator, I want every required sidecar allocated before
  capture so replay contains only fixed device enqueues.
- As a resident-session owner, I want exact identity reuse checks so accidental
  replacement or aliasing fails before captured work launches.
- As a performance engineer, I want deterministic logical byte totals so later
  memory-budget and benchmark tracks use the same authoritative inventory.
