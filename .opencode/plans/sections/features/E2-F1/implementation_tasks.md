# Implementation Tasks

## E2-F1-P1: Inventory Current Schemas

- Read and summarize fields from `ParticleData`, `GasData`, `WarpParticleData`,
  and `WarpGasData`.
- Capture current validation rules, dtype expectations, and leading dimension
  conventions.
- Capture CPU/GPU transfer behavior from `particula/gpu/conversion.py`, including
  `GasData.name` placeholders and `WarpGasData.vapor_pressure` loss on return.
- Cross-check inventory against existing particle, gas, Warp type, and conversion
  tests.

## E2-F1-P2: Decide Field Ownership

- Create a schema decision record with columns for field, owner, CPU shape, GPU
  shape, dtype, mutability, round-trip behavior, and downstream consumers.
- Decide whether `ParticleData.density` stays shared as `(n_species,)` for now.
- Decide whether `ParticleData.volume` remains particle-container-owned or is
  documented as shared/per-box simulation state until an environment container
  exists.
- Decide whether `vapor_pressure` is gas state, environment-derived state,
  process scratch/cache, or a GPU-only helper with explicit loss semantics.
- Decide how species names are preserved across GPU workflows.

## E2-F1-P3: Document Shape Conventions

- Publish canonical shape tables for single-box and multi-box workflows.
- State that single-box workflows retain a leading box dimension of size 1.
- State particle-resolved and binned concentration/count conventions.
- Document CPU strategy boundaries where data containers can store multi-box
  state but dynamics may still require `n_boxes == 1`.

## E2-F1-P4: Publish Handoff Map

- Add downstream handoff notes for E2-F2 through E2-F9.
- Link the decision record from roadmap and migration docs.
- Run documentation/link checks available in the repository.
- Include any reviewer-requested clarifications from earlier phases.
