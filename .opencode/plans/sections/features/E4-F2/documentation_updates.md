# Documentation Updates

- Update `docs/Features/data-containers-and-gpu-foundations.md` with numeric
  activity/surface configuration, ownership, and transfer-boundary behavior.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` with supported ideal,
  kappa, static, and selected effective-surface modes.
- Document BAT and all unimplemented CPU strategies as CPU-only and explain
  their explicit failure mode.
- Add or update a direct-kernel example only if the public call signature
  changes; do not imply high-level `Aerosol`/`Runnable` GPU support.
- Record parity tolerances, Warp CPU/CUDA policy, and focused reproduction
  commands near the affected GPU feature documentation.
- Mark E4-F2 phases and these plan sections shipped after implementation.

## P1 Status
- Issue #1287 added internal formula helpers and co-located tests only. No
  public API or user documentation changed; supported-mode documentation remains
  deferred to the integration and documentation phases.
