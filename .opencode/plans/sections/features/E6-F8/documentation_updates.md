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

## P7 shipped

- Updated `docs/Theory/Technical/Dynamics/Nucleation_Equations.md`,
  `docs/Features/`, `docs/Examples/Nucleation/`, `AGENTS.md`, and the E6-F8
  plan sections to publish the bounded direct-transfer contract, explicit
  example, validation commands, and shipped status.

## P7 shipped

Updated feature, roadmap, theory, architecture, AGENTS, example index, and the
direct-Warp example. The implementation remains bounded and E6-F9 downstream.
