# Infrastructure Reuse

- `particula/gpu/warp_types.py:24-78` defines `WarpParticleData`, including
  multi-box masses, concentration, charge, shared density, and authoritative
  `(n_boxes,)` volume. Extend behavior around this struct; do not alter its
  schema.
- `particula/gpu/warp_types.py:81-161` defines `WarpGasData` and its
  `(n_boxes, n_species)` concentration layout. Preserve names as CPU-owned
  metadata and keep vapor pressure derived resident state.
- `particula/gpu/warp_types.py:164-184` defines per-box environment arrays.
  Communication must not silently conflate volume changes with temperature or
  pressure updates.
- `particula/particles/particle_data.py:47-137` and
  `particula/gas/gas_data.py:46-138` provide CPU multi-box shape conventions for
  independent reference calculations and checkpoint validation.
- `particula/gpu/kernels/dilution.py:1-17,40-96` demonstrates per-box
  concentration kernels, full preflight before update launch, identity-preserving
  mutation, and no-op semantics. Reuse its validation and launch-order patterns;
  do not misuse dilution as conservative inter-box transport.
- `particula/gpu/kernels/environment.py` provides Warp-array recognition and
  same-device validation patterns used by direct kernels.
- `particula/gpu/conversion.py` is the only existing explicit CPU/Warp transfer
  boundary. Normal communication steps must not call it.
- E7-F4 `ResidentSession`, dimensions, `SidecarRegistry`, and checkpoint resource
  manifest are the lifecycle and allocation seams. Register communication maps,
  staging ledgers, slot plans, and diagnostics once.
- E7-F5 `ProcessNode`, `TimestepPlan`, resolved graph, and scheduler mutation
  window are the orchestration seams. Add typed communication/volume node kinds
  rather than bypassing the scheduler.
- E7-F6 capability errors and fallback policy govern unsupported configuration
  and device handling. Runtime launch failures propagate and fault the session.
- Shipped fixed-slot discovery/activation and exhaustion patterns under
  `particula/particles/slot_management.py`, `particula/particles/exhaustion.py`,
  and `particula/gpu/kernels/slot_management.py` provide deterministic free-slot,
  preflight, and capacity-failure precedents. Reuse semantics where applicable;
  do not introduce resizing.
- `particula/gpu/tests/process_sequence_test.py` and module-level direct-kernel
  tests provide transfer spies, identity assertions, Warp CPU parametrization,
  and optional CUDA skip patterns.
- `docs/Features/Roadmap/data-oriented-gpu.md:1573-1585` is the authoritative
  prescribed-communication design boundary.
