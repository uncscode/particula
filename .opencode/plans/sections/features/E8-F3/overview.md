# Overview

## Problem Statement

The shipped `GPUResourceRegistry` pins major condensation, coagulation,
wall-loss, nucleation, and communication sidecars, but capture-ready execution
needs one complete inventory before capture begins. Validation status arrays,
normalized controls, selected-lane work, diagnostics, and other prepared-node
temporaries must not fall back to per-call allocation. The capture lifecycle
also needs a deterministic way to prove that the same arrays are reused and to
report their logical byte cost without reading device payloads.

## Value Proposition

E8-F3 makes the registry the single concrete owner of capture-lifetime reusable
storage. One setup transaction preallocates the resources selected by the E8-F2
prepared timestep, pins exact identities and capacities, and publishes an
immutable inventory with deterministic per-role, per-family, and total byte
accounting. Repeated preparation and replay can then reject drift before launch
and perform zero allocation or replacement.

## User Stories

- As a graph-capture integrator, I want every required sidecar allocated before
  capture so replay contains only fixed device enqueues.
- As a resident-session owner, I want exact identity reuse checks so accidental
  replacement or aliasing fails before captured work launches.
- As a performance engineer, I want deterministic logical byte totals so later
  memory-budget and benchmark tracks use the same authoritative inventory.
