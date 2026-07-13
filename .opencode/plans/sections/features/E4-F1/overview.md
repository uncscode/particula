# Overview

## Problem Statement

GPU condensation currently consumes `WarpGasData.vapor_pressure` as state
fixed during CPU-to-GPU transfer; omitted values are zero-filled. When the
per-box temperature changes, the pressure can therefore be stale or physically
invalid. Issue #1272 requires current thermodynamics to drive condensation
without a host round trip.

## Value Proposition

E4-F1 defines a numeric, fixed-shape GPU configuration for explicitly supported
vapor-pressure models and refreshes the `(n_boxes, n_species)` pressure buffer
on-device from current temperature. Constant and Buck models will match CPU
references, preserve species ordering, and fail before mutation on invalid
configuration. This foundation gates sibling features E4-F2 and E4-F3.

## User Stories

- As a simulation developer, I want vapor pressure recomputed from current GPU
  temperature so repeated condensation steps do not consume stale host data.
- As a model author, I want explicit model modes and parameter validation so an
  unsupported thermodynamic strategy fails predictably.
- As a GPU user, I want Warp CPU/CUDA-compatible formulas so execution remains
  device-resident and numerically traceable to CPU physics.
