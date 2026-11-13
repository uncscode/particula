# Overview

## Problem Statement

Before issues #1281 and #1282, GPU condensation had no explicit, validated
thermodynamic configuration boundary or device-resident vapor-pressure refresh.

## Value Proposition

Issue #1281 shipped E4-F1-P1: a caller-owned, fixed-shape GPU sidecar and
validator required at the `condensation_step_gpu()` boundary. Issue #1282
shipped E4-F1-P2: `refresh_vapor_pressure_gpu` in
`particula/gpu/kernels/thermodynamics.py` validates its Warp `float64` public
boundary, then uses one `(n_boxes, n_species)` launch to overwrite
`WarpGasData.vapor_pressure`. It supports constant pressures in Pa and the
canonical Buck water/ice equations, while retaining the fixed sidecar schema
and leaving condensation integration to P3.

Issue #1284 shipped E4-F1-P4 integration coverage without production changes.
It verifies that the public condensation boundary safely reuses caller-owned
thermodynamics, vapor-pressure, and mass-transfer buffers, preserves the legacy
positional mass-transfer slot, and rejects missing or cross-device sidecars
atomically.

## User Stories

- As a simulation developer, I want a required, caller-owned thermodynamic
  sidecar so future thermodynamic behavior has an explicit GPU boundary.
- As a model author, I want explicit model modes and parameter validation so an
  unsupported thermodynamic strategy fails predictably.
- As a GPU user, I want an explicit on-device refresh API with no host formula
  calculation or implicit transfer.
- As a simulation developer, I want every successful condensation step to
  derive its caller-owned vapor-pressure buffer from the selected current
  per-box temperature before mass transfer.
- As a model author, I want constant and Buck output that preserves species
  ordering and matches CPU references at the freezing boundary.
- As a simulation developer, I want repeated public calls to reuse my
  thermodynamic sidecar and output buffer without stale values or hidden
  mutation when validation fails.
