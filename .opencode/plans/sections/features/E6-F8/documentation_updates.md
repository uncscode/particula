# Documentation Updates

## P1/P3 delivered (#1438, #1439, #1440)

- `particula/gpu/kernels/nucleation.py` documents the concrete-only read-only
  P1 boundary, frozen record ownership, fixed-shape sidecars, and later-phase
  deferrals. Its P2 documentation records survival-included rates, common
  inventory-limited admission, P2/P3 sidecar ownership, and the sidecar-only
  mutation boundary. P3 documentation records private exact count conversion,
  E6-F5 diagnostic reuse, retained over-capacity counts, caller-owned sidecars,
  and the pre-launch preservation/post-launch rollback boundary.
- `.opencode/guides/architecture/architecture_outline.md` and
  `.opencode/guides/architecture_reference.md` now describe the shipped private
  P3 staging seam and continue to defer E6-F6 policy, activation, particle/gas
  mutation, and an integrated GPU step. No user-facing documentation or API
  export was added.

## Deferred P7 work

- Update `docs/Theory/Technical/Dynamics/Nucleation_Equations.md` with the exact
  CPU-to-Warp correspondence, SI units, validity bounds, admission equation,
  represented-mass accounting, and unsupported physics.
- Add or update a `docs/Features/` direct GPU nucleation page documenting
  `nucleation_step_gpu`, concrete-module configuration/scratch APIs, fixed
  shapes, dtypes, devices, ownership, mutation, and failure ordering.
- Add an explicit CPU-to-Warp setup and restore example under
  `docs/Examples/Nucleation/`; keep transfers outside the direct step and make
  Warp CPU the default documented backend.
- Update `AGENTS.md` with intended imports, sidecar contracts, E6-F5/F6/F7
  dependencies, conservation/parity commands, and no-fallback boundaries.
- Cross-link E6-F7's CPU reference and E6-F9's integrated process example;
  state that E6-F8 does not provide a high-level GPU runnable or scheduler.
- Update E6 and E6-F8 plan sections with final issue numbers, measured
  tolerances, shipped status, and any resolved sidecar naming decisions.
