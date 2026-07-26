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

## P7 shipped (#1444)

- Updated the feature, roadmap, theory, architecture, `AGENTS.md`, and
  `docs/Examples/Nucleation/` to publish the package-exported direct step,
  concrete-only records and sidecars, explicit transfer/synchronization
  ownership, bounded P1--P5 ordering, conservation tolerance, and failure/no-op
  boundaries.
- Added `docs/Examples/Nucleation/gpu_direct_nucleation.py` and its Warp-guarded
  regression in `particula/gpu/tests/gpu_direct_nucleation_example_test.py`.
  `particula/tests/nucleation_docs_test.py` covers publication language, import
  boundaries, links, and exclusions. No production kernel, export, physics,
  capacity-policy, or parity-oracle behavior changed.
- E6-F9 remains the downstream explicit-transfer integration consumer; P7 adds
  neither a GPU Runnable nor scheduler/backend orchestration.
