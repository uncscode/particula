# Success Criteria

- [ ] One setup-only operation resolves and publishes every reusable array
  required by the selected E8-F2 prepared timestep before READY/capture.
- [ ] The inventory covers process scratch/output, normalized controls,
  validation/status and selected-lane work, the selected communication family,
  diagnostics/accounting, and coagulation/wall-loss RNG state.
- [ ] Compatible reacquisition returns the exact capture set, native records,
  arrays, capacities, configuration, and byte report by identity.
- [ ] After first successful setup, prepared enqueue/replay performs zero
  allocation, resource acquisition, RNG initialization, payload readback,
  transfer, synchronization, or storage replacement.
- [ ] Whole-set setup is fail-closed: validation/allocation failure publishes no
  partial capture set, and a clean retry can succeed while the session is active.
- [ ] Session, primary, sidecar, capacity, communication-map, diagnostic, or
  requirement drift rejects before allocator or writer activity.
- [ ] Byte accounting is deterministic, overflow-safe, and exact for logical
  manifest bytes, including canonical zero dimensions and dynamic capacities.
- [ ] Reports expose no pointers, payload values, or RNG words and explicitly do
  not claim allocator reservation/fragmentation, checkpoint copies, or tape.
- [ ] Existing family acquisition, checkpoint/restart, RNG, diagnostics,
  communication, resident execution, and export contracts remain compatible.
- [ ] Focused resident tests, full applicable execution tests, untargeted
  repository coverage, linters, and strict documentation validation pass
  without lowering thresholds or changing default collection.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Capture-required roles represented | Major E7 families; prepared temporaries incomplete | 100% of E8-F2 requirement set | Inventory completeness test |
| Device allocations after capture-set publication | Direct/prepared seams may allocate | 0 | Forbidden allocator spies |
| Resource acquisitions during prepared enqueue/replay | Possible through incomplete preparation | 0 | Registry spy/launch trace |
| Identity replacements accepted after publication | Family-level rejection exists | 0 across full capture set | Drift/replacement tests |
| Partial capture sets after failed setup | No whole-set contract | 0 | Injected-failure tests |
| Logical byte report variance for identical requirements | No report | 0 bytes | Snapshot/formula tests |
| Full-package coverage threshold | Repository configured | Met or exceeded | `.opencode/tools/run_pytest.py` |

Performance speedup and allocator-reserved memory are not E8-F3 metrics; E8-F5,
E8-F6, and E8-F8 own those measurements.
