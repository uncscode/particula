# Support

## Current shipped support boundaries

The containers are multi-box capable, but current execution support is narrower
than storage support.

| Area | Shipped support | Notes |
| --- | --- | --- |
| CPU `ParticleData` / `GasData` storage | Multi-box-capable | Leading `n_boxes` axis is part of the stored schema. |
| CPU condensation with data containers | `n_boxes == 1` only | Multi-box CPU execution still requires a caller-managed per-box loop. |
| CPU coagulation with data containers | `n_boxes == 1` only | Multi-box CPU execution is not yet a built-in runtime path. |
| CPU↔GPU transfer | Explicit helper calls only | No hidden container movement or hidden environment synchronization. |
| Warp/CUDA support | Optional | Warp `device="cpu"` is the baseline when Warp is installed; CUDA is additive local evidence and unavailable devices skip cleanly. |
| Low-level GPU slot activation | Shipped fixed-capacity direct-kernel contract | `activate_slots_gpu` maps selected request prefixes to ascending free slots with caller-owned same-device sidecars; no hidden transfer, resizing, or high-level runnable is provided. |
| Low-level GPU condensation direct-kernel path | Shipped bounded direct-kernel contract | Executes four fixed coupled substeps with active-device sequence validation inventory and gas coupling. This is direct-kernel evidence, not broad GPU-condensation support. |
| Low-level GPU coagulation direct-kernel path | Direct, particle-resolved direct-kernel contract | Exact mask and rejection boundaries follow below. This path establishes no Runnable support, CPU parity, or performance claim. |
| Resident execution and native CUDA capture | Explicit bounded lifecycle | See [resident example](https://github.com/uncscode/particula/blob/main/docs/Examples/gpu_resident_multi_timestep.py) and [capture limitations](resident/graph-capture.md); no implicit CPU fallback or measured performance claim. |

Additional shipped boundaries:

- Executable coagulation masks are singletons `1`, `2`, `4`, `8`; unordered
  pairs `3`, `5`, `6`, `9`, `10`, `12`; and four-way mask `15`. Tuples
  normalize to canonical order. Non-turbulent three-way mask `7` rejects at
  capability preflight before particle metadata or enabled-term validation.
  Turbulent three-way masks `11`, `13`, `14` proceed through particle metadata
  and enabled-term validation, then reject before downstream normalization,
  allocation, RNG setup, kernel launch, or mutation. Turbulent masks require
  explicit turbulent inputs.
- Supplied coagulation particle state, collision outputs, and persistent RNG are
  caller-owned same-device Warp resources. Persistent RNG reuse has no implicit
  transfer or synchronization; omitted state allocates call-local storage, and
  `initialize_rng=True` explicitly resets the caller-owned buffer.
- `ParticleData.volume` remains the authoritative per-box simulation-volume
  owner; it does not move into `EnvironmentData`.
- `EnvironmentData` is the shipped CPU thermodynamic owner for `temperature`,
  `pressure`, and `saturation_ratio`.
- `WarpGasData.vapor_pressure` is helper state only and has no CPU `GasData`
  field.
- Coagulation `mechanism_config` is host metadata, while supplied
  `collision_pairs`, `n_collisions`, and `rng_states` are caller-owned
  same-device Warp sidecars. None are fields on `ParticleData`, `GasData`,
  `EnvironmentData`, or any Warp container schema.
- Omitted coagulation collision outputs and RNG state are call-local convenience
  allocations. Supplied RNG state advances in place and resets only with
  `initialize_rng=True`.
- Kernels and runnables do not perform hidden simulation-state transfers.
  Explicitly supplied step and output buffers do mutate in place: particle
  masses, gas concentration, scratch transfer fields, total transfer, and
  energy output retain their documented write semantics.
  Callers remain responsible for synchronization before
  host observation or restoration.
  Validation `.numpy()` readbacks at their documented preflight and per-substep
  boundaries may synchronize without transferring simulation state.
  CUDA preflight validation-flag readbacks may
  synchronize without transferring simulation state.
- Direct condensation does not add a high-level `Aerosol`/`Runnable` path,
  automatic backend selection or fallback, implicit transfer or synchronization,
  adaptive stepping, new container fields, kernels, or physics, BAT, or
  staggered/Gauss-Seidel support. It also does not claim general CPU-strategy
  parity or broader unproven physics support.

