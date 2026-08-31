# Success Criteria

- [x] P1 provides one concrete-only graph-capture declaration module defining
  capability and compatibility vocabulary without new public package exports.
  Lifecycle, invalidation, and recapture vocabulary remain P2-P3 scope.
- [x] P1 capability resolution distinguishes CPU and Warp CPU, unavailable
  runtime or device, missing capture API, and availability without importing
  Warp or falling back.
- [x] P1 compatibility signatures cover every documented structural field:
  device, dimensions, primary identities, exact request/graph/schedule/order,
  process configuration, sidecars, diagnostics, communication, and RNG state
  identities.
- [x] P1 comparison reports deterministic first drift for the implemented
  signature groups and leaves stable RNG-array identity compatible; exhaustive
  drift and active-slot coverage remains P3 test scope.
- [x] P2 lifecycle metadata covers all six states, legal host-only transitions,
  exact carrier validation, first-reason retention, and idempotent paths; it
  rejects every other lifecycle transition without inspecting a resident binding.
- [x] Recapture is explicit and creates a new record only after eligibility
  checks; there is no automatic replacement, migration, fallback, retry, or
  graph-handle checkpointing.
- [x] P2 read-only classification preserves accepted lifecycle metadata by
  identity, while writer-may-have-launched classification produces immutable
  fault metadata without asserting resident-session mutation or rollback.
- [x] Persistent coagulation and wall-loss RNG sidecars advance by identity and
  are never implicitly initialized or reset by replay/recapture checks.
- [ ] Focused execution tests pass; the untargeted repository runner supplies
  full-package coverage without lowering thresholds; documentation builds
  strictly.
- [ ] E8-F2 and E8-F3 can consume the documented contract without redefining
  lifecycle states or recapture triggers.

Validation record for #1550 (2026-08-30): focused documentation/export checks
passed (17 passed); the untargeted `.opencode/tools/run_pytest.py` passed with
6381 passed, 9 skipped, 1 xfailed, and 93.59% coverage. `mkdocs build --strict`
is unavailable because no supported MkDocs runner is available. The strict-build
criterion and delivery handoff remain unchecked.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Documented structural recapture triggers with tests | 0 centralized | 100% of signature fields | `graph_capture_test.py` parametrization |
| Illegal lifecycle transitions accepted | Not defined | 0 | Lifecycle transition matrix tests |
| Hidden CPU fallback/automatic recapture paths | 0 intended, not contract-backed | 0 | Unit, integration, and export tests |
| Focused capture-contract test failures | N/A | 0 | Focused pytest commands |
| Full-package coverage threshold | Repository configured | Met or exceeded; never lowered | `.opencode/tools/run_pytest.py` |
| CUDA-unavailable behavior | Ad hoc test harness | Clean skip, never CPU fallback | CUDA-marked capability rows |

Performance improvement and numerical captured-loop parity are deliberately not
E8-F1 completion metrics; E8-F4, E8-F5, and E8-F8 own that evidence.
