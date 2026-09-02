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

## Value Proposition

E8-F3 incrementally makes the registry the concrete authority for
capture-lifetime reusable storage. P1 supplies the inventory; P2 adds the
dilution descriptor/view validation seam and exact prepared-adapter retention.
Later phases own whole-set allocation, publication, and compatible reuse.

## User Stories

- As a graph-capture integrator, I want every required sidecar allocated before
  capture so replay contains only fixed device enqueues.
- As a resident-session owner, I want exact identity reuse checks so accidental
  replacement or aliasing fails before captured work launches.
- As a performance engineer, I want deterministic logical byte totals so later
  memory-budget and benchmark tracks use the same authoritative inventory.
