# Documentation Updates

## P1 Completion (issue #1528)

- Public documentation deliberately remains unchanged. The diagnostics protocol
  is concrete-only in `particula.execution.diagnostics`, adds no package or
  top-level exports, and is documented only by its concrete module contract and
  co-located tests.

## P2 Completion (issue #1529)

- Public documentation remains unchanged. Concrete checkpoint docstrings now
  define schema-v3 continuation metadata, including valid empty current-word
  payloads, and clarify that canonical primary bytes plus registry-owned
  sidecars, ledgers, diagnostics, and closed-map state are authoritative.
  Arbitrary caller outputs remain outside checkpoint authority.

- Add `docs/Examples/gpu_resident_multi_timestep.py` as the canonical complete
  E7 example using backend selection, resident state, multiple boxes/processes,
  diagnostics, explicit checkpoints, and finalization.
- Add `particula/tests/gpu_resident_multi_timestep_docs_test.py` to execute and
  inspect the example, including transfer-boundary assertions.
- Update `docs/Features/data-containers-and-gpu-foundations.md` with the public
  execution/session imports, state authority, diagnostic shapes/units,
  checkpoint payload, restart guarantees, support matrix, and limitations.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` with dated E7-F1 through
  E7-F9 evidence and mark Epic G complete only after the exact exit bar passes.
- Update `AGENTS.md` quick reference with canonical imports and focused commands.
- Update applicable `.opencode/guides/` testing/documentation guidance with the
  Warp CPU baseline, optional CUDA policy, and docs-regression command.
- Update E7/E7-F9 structured plan sections to shipped status during finalization.
- Link issue #1451 closeout to the support matrix, example, recorded tolerances,
  test commands, pass/skip results, and remaining Epic H/I deferrals.

Documentation must not describe graph capture, performance, CUDA availability,
cross-backend exact RNG replay, hidden fallback, or unsupported process modes as
shipped capabilities.
