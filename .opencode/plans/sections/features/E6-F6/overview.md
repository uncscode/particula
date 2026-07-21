# Overview

## Problem Statement

Fixed-shape CPU and Warp particle arrays cannot admit every particle-creation
request once a box runs out of free slots. E6-F5 deliberately reports capacity
and fails atomically; without a separate exhaustion policy, downstream
nucleation would either stop, resize arrays, or silently discard demand.

## Value Proposition

E6-F6 provides a shared, conservation-checked policy layer. Deterministic
resampling is enabled by default and creates capacity without allocation;
representative-volume scaling is an independently selectable fallback that is
off by default. A complete read-only plan precedes every commit, so unsupported
or unsatisfiable requests fail without particle, volume, sidecar, or RNG writes.

## User Stories

- As a particle-resolved simulation user, I want full boxes handled by a
  documented default so new-particle demand is never silently lost.
- As a scientific developer, I want per-box number, species-mass, and charge
  accounting so resolution changes cannot create or destroy inventory.
- As a GPU developer, I want fixed-shape, caller-owned work buffers and CPU/Warp
  parity so exhaustion handling remains device-resident and allocation-stable.

This is feature track T6 under E6 and depends on E6-F5.
