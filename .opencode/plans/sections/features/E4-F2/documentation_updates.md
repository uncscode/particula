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
- Record completed phase evidence in the canonical plan sections after
  implementation.

## P1 Status
- Issue #1287 added internal formula helpers and co-located tests only. No
  public API or user documentation changed; supported-mode documentation remains
  deferred to the integration and documentation phases.

## P2 Status
- Issue #1288 added an internal device helper and co-located tests only. No
  public API or user documentation changed; supported-mode documentation remains
  deferred to the integration and documentation phases.

## P3 Status
- Issue #1289 added the concrete-kernel `CondensationActivitySurfaceConfig`
  API documentation and co-located test coverage. Feature documentation remains
   intentionally out of scope; no user-facing documentation was changed.

## P4 Status
- Issue #1290 updated
  `docs/Features/data-containers-and-gpu-foundations.md` with the supported
  numeric activity/surface modes, fixed-shape caller-owned fp64 sidecars,
  direct-step ownership and mutation contract, and fp64 parity device/tolerance
  policy.
- Issue #1290 updated `docs/Features/Roadmap/data-oriented-gpu.md` to describe
  the shipped direct-kernel ideal/kappa and static/composition-weighted scope.
  BAT and every other unsupported activity or surface strategy remain CPU-only
  and fail with `ValueError`; no high-level GPU runnable support is implied.
