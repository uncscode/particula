# E2-F9 Architecture and Design

## Design Approach

This feature is a documentation integration layer over existing runtime
architecture. It should not change source semantics. The guide should organize
the foundation around stable concepts:

1. CPU data ownership: `ParticleData` owns particle arrays and `GasData` owns
   gas-species arrays.
2. Shape conventions: every multi-box-capable array should show its explicit
   leading `n_boxes` dimension and shared arrays should be called out.
3. GPU boundary: Warp schemas and transfer helpers are explicit conversion
   points, not transparent runtime behavior.
4. Support boundaries: current condensation strategies and GPU kernels still
   have scalar or single-box compatibility constraints.
5. Roadmap handoff: planned environment-state, graph-capture, precision, and
   autodiff work remain future dependencies.

## Shape Tables to Include

- `ParticleData.masses`: `(n_boxes, n_particles, n_species)`.
- `ParticleData.concentration`: `(n_boxes, n_particles)`.
- `ParticleData.charge`: `(n_boxes, n_particles)`.
- `ParticleData.density`: `(n_species,)`, currently shared across boxes.
- `ParticleData.volume`: `(n_boxes,)`.
- `GasData.concentration`: `(n_boxes, n_species)`.
- `GasData.molar_mass`: `(n_species,)`.
- Planned environment fields such as temperature and pressure: `(n_boxes,)`,
  marked as roadmap/planned unless dependency implementation exists.

## GPU Transfer Caveats

- `WarpGasData` excludes species names, stores `partitioning` as integer data,
  and adds `vapor_pressure` compared with CPU `GasData`.
- `from_warp_gas_data` requires names for faithful reconstruction and does not
  round-trip `vapor_pressure` into `GasData`.
- `gpu_context` is a convenience helper for `ParticleData`; complex flows should
  use explicit conversion helpers.
- `sync=False` should be documented as advanced/manual synchronization mode.
