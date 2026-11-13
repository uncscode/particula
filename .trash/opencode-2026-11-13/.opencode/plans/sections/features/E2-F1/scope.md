# Scope

## In Scope

- Inventory current schemas for:
  - `particula/particles/particle_data.py::ParticleData`
  - `particula/gas/gas_data.py::GasData`
  - `particula/gpu/warp_types.py::WarpParticleData`
  - `particula/gpu/warp_types.py::WarpGasData`
- Decide authoritative ownership for shared and per-box fields, including:
  - particle masses, concentration/counts, charge, shared density shaped
    `(n_species,)`, and authoritative per-box simulation volume on
    `ParticleData`
  - gas names, molar masses, concentration, and partitioning, while keeping vapor
    pressure out of CPU `GasData` and `EnvironmentData` ownership
  - environment temperature, pressure, and canonical species-resolved
    `saturation_ratio`
- Document shape conventions for:
  - single-box workflows with leading `n_boxes == 1`
  - multi-box workflows with leading `n_boxes`
  - particle-resolved workflows
  - binned/distribution workflows
  - CPU arrays and Warp array dimensions
- Publish a downstream ownership handoff for E2-F2 through E2-F9.
- Add or update lightweight documentation tests only if the chosen docs format
  has checkable examples or tables.

## Out of Scope

- Implementing `EnvironmentData` or `WarpEnvironmentData`; that belongs to
  downstream environment tracks.
- Migrating condensation or coagulation kernels from scalar temperature and
  pressure to environment containers.
- Changing numerical algorithms, dtype policy, or mass/precision behavior.
- Refactoring public APIs beyond documentation-only links or exports needed for
  the decision record.
- Creating a standalone testing-only phase; any checkable artifacts should ship
  alongside the phase that introduces them.

## Explicit Non-Goals

- Do not remove existing `ParticleData` or `GasData` compatibility paths.
- Do not imply that all CPU dynamics support multi-box execution merely because
  containers can hold multi-box state.
- Do not make GPU-only fields silently authoritative without documenting CPU
  round-trip behavior.
- Do not preserve species names inside `WarpGasData`; require explicit CPU-side
  names or external index-map metadata on restoration.
