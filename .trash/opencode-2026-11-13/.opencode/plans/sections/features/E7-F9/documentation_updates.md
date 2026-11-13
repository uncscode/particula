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

## P4 Completion (issue #1531)

- Public documentation remains unchanged. The implementation adds only internal
  resident multi-box lifecycle, logical-ID, RNG, and wall-loss regression
  coverage in `particula/execution/tests/multi_box_loop_test.py`; it introduces
   no production or public behavior.

## P5 Completion (issue #1532)

- Public and user documentation remain unchanged. The implementation adds only
  regression coverage in `particula/execution/tests/transport_loop_test.py`,
  `particula/execution/tests/restart_loop_test.py`, and
  `particula/gpu/kernels/tests/communication_test.py`; it changes no production
   API, checkpoint schema, scheduler ordering, or exports.

## P6 Completion (issue #1533)

- Added `docs/Examples/gpu_resident_multi_timestep.py` as the canonical runnable
  three-box resident-scheduler example. It gives actionable lazy no-Warp guidance
  with no CPU fallback; validates availability; performs one source setup upload,
  two source steps, caller-owned diagnostic observation, manual exact-device
  checkpoint/restart, and cached source finalization.
- Added `particula/tests/gpu_resident_multi_timestep_docs_test.py` as the
  executable documentation regression for disabled/import/failure/enabled paths,
  transfer boundaries, resident identities, diagnostics, restart, and
  finalization. CUDA remains optional.

- P7/finalization must update
  `docs/Features/data-containers-and-gpu-foundations.md` with concrete-only,
  direct-import execution/session seams, state authority, diagnostic
  shapes/units, checkpoint payload, restart guarantees, support matrix, and
  limitations.
- P7/finalization must update `docs/Features/Roadmap/data-oriented-gpu.md` with
  dated E7-F1 through E7-F9 evidence and mark Epic G complete only after the
  exact exit bar passes.
- P7/finalization must update the `AGENTS.md` quick reference and applicable
  `.opencode/guides/` testing/documentation guidance with canonical direct
  imports, the Warp CPU baseline, optional CUDA policy, and docs-regression
  command.
- Only after recorded validation evidence may finalization update E7/E7-F9
  structured-plan status to shipped and link issue #1451 closeout to the support
  matrix, recorded tolerances, test commands, pass/skip results, and remaining
  Epic H/I deferrals.

Documentation must not describe graph capture, performance, CUDA availability,
cross-backend exact RNG replay, hidden fallback, or unsupported process modes as
shipped capabilities.

## P7 Evidence Closeout (issue #1534, 2026-08-11)

- Required artifacts and evidence are recorded: focused assertions (289),
  exports (15), resident fast suite (891), full-package coverage (6,254 tests,
  93%), changed-module coverage (891 tests, 95%; `diagnostics.py` 79%,
  `gpu_resources.py` 87%, `checkpoint.py` 87%, `resident_scheduler.py` 86%),
  strict MkDocs, and optional CUDA rows all passed.
- The derived P1--P6 executable coverage targets are
  `particula/execution/diagnostics.py`,
  `particula/execution/gpu_resources.py`,
  `particula/execution/checkpoint.py`, and
  `particula/execution/resident_scheduler.py`; P7 Markdown is not a coverage
  target. Epic G and P7 are shipped.
