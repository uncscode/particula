# Success Criteria

- [ ] One concrete-only graph-capture contract module defines capability,
  compatibility, lifecycle, invalidation, and recapture vocabulary without new
  public package exports.
- [ ] Capability resolution distinguishes unsupported Warp CPU, unavailable
  Warp/CUDA runtime or device, missing capture APIs, and invalid resident state.
- [ ] A compatibility signature covers every documented structural field:
  device, dimensions, primary identities, exact request/graph/schedule/order,
  process configuration, sidecars, diagnostics, communication, and RNG state
  identities.
- [ ] Every structural drift case deterministically invalidates before launch;
  stable-shape payload changes and active-slot occupancy changes do not.
- [ ] Lifecycle transition tests cover all legal transitions and reject replay
  or recapture from invalid, faulted, terminal, or open-step bindings.
- [ ] Recapture is explicit and creates a new record only after eligibility
  checks; there is no automatic replacement, migration, fallback, retry, or
  graph-handle checkpointing.
- [ ] Read-only preflight failures preserve active resident and capture state;
  possible post-launch writer failures retain existing no-rollback fault rules.
- [ ] Persistent coagulation and wall-loss RNG sidecars advance by identity and
  are never implicitly initialized or reset by replay/recapture checks.
- [ ] Focused execution tests pass; the untargeted repository runner supplies
  full-package coverage without lowering thresholds; documentation builds
  strictly.
- [ ] E8-F2 and E8-F3 can consume the documented contract without redefining
  lifecycle states or recapture triggers.

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
