# Documentation Updates

## P1--P6 delivered (#1438, #1439, #1440, #1441, #1442, #1443)

- `particula/gpu/kernels/nucleation.py` documents the concrete-only read-only
  P1 boundary, frozen record ownership, fixed-shape sidecars, and later-phase
  deferrals. Its P2 documentation records survival-included rates, common
  inventory-limited admission, P2/P3 sidecar ownership, and the sidecar-only
  mutation boundary. P3 documentation records private exact count conversion,
  E6-F5 diagnostic reuse, retained over-capacity counts, caller-owned sidecars,
   and the pre-launch preservation/post-launch rollback boundary. P4
   documentation adds immutable P2/P3 handoffs, caller-owned exhaustion
   controls/buffers, resampling-first/scaling-fallback selection, final P4
   diagnostics, expected-rejection snapshots, and the entered-primitive
   no-cross-primitive-rollback boundary.
- `.opencode/guides/architecture/architecture_outline.md` and
  `.opencode/guides/architecture_reference.md` now describe the shipped private
   P4 orchestration seam, its ownership/failure boundaries, and its continued
   deferral of activation, particle/gas mutation, and an integrated GPU step.
   `docs/Features/Roadmap/data-oriented-gpu.md` and
   `docs/Features/nucleation_strategy_system.md` record the same private P4
    contract. Issue #1442 adds the public-boundary docstring for the lazily
    exported direct step and export regression coverage; configurations,
    sidecars, and helpers remain concrete-only. No feature documentation or
    example closeout was performed. Issue #1443 adds only the test-module and
    test-local helper docstrings for independent direct-Warp parity/conservation
    evidence; it makes no general documentation or API-contract change.

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
