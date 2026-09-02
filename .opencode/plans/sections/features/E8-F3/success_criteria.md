# Success Criteria

## P1 Completion

- [x] The unchanged six canonical manifests resolve into immutable,
  direct-module-only logical reports in manifest order, including both
  communication families.
- [x] Reports are read-only and independent of acquisition, bindings,
  configuration payloads, device payload inspection, and allocator reservation.
- [x] Checked arithmetic now protects shape, stride, element, and byte
  calculations used by reporting, allocation, and range validation.
- [x] Focused registry tests for the P1 inventory/reporting contract pass with
  no package or top-level export changes.

## P2 Completion

- [x] The concrete-only dilution descriptor family defines `(B,)` `wp.float64`
  normalized-coefficient and factors roles.
- [x] `PreparedResourceViews` supports read-only validation of supplied views
  before prepared-adapter use, and accepted supplied resources are retained by
  exact identity.
- [x] P2 introduces no allocation, publication, reacquisition, RNG lifecycle
  change, or public export.

## P3 Completion

- [x] One concrete-only registration pins absent communication or one exact
  already-published closed GAS/PARTICLES view plus ordered diagnostic
  registrations by identity.
- [x] Selected-resource reports retain deterministic resolved schema and logical
  byte metadata, including zero extents, without payload inspection.
- [x] Candidate validation is transactional, exact repeats reuse the retained
  inventory, and forbidden overlaps are detected in O(R log R) host metadata
  work while permitted read-only accounting aliases remain allowed.
- [x] Diagnostics validation, communication setup, checkpoint enumeration,
  scheduler behavior, and public exports remain unchanged.

- [x] One setup-only operation resolves, stages, and atomically publishes the
  selected reusable capture resources without changing READY/capture admission.
- [x] The P4 set retains the selected process/control, communication,
  diagnostics/accounting, and applicable coagulation/wall-loss RNG resources by
  exact identity.
- [x] Compatible preparation and metadata-only validation return the exact
  capture set, requirements, native records, arrays, capacities, configuration,
  and byte report by identity.
- [ ] After first successful setup, prepared enqueue/replay performs zero
  allocation, resource acquisition, RNG initialization, payload readback,
  transfer, synchronization, or storage replacement.
- [x] Whole-set setup is fail-closed: candidate failure publishes no partial
  capture set, preserves prior publication, and permits a clean retry.
- [x] Exact requirements, pinned session, P3 inventory, capacities, schemas,
  retained views/records, communication/diagnostic bindings, and stream-resource
  identity are validated before candidate resource work.
- [ ] Byte accounting is deterministic, overflow-safe, and exact for logical
  manifest bytes, including canonical zero dimensions and dynamic capacities.
- [ ] Reports expose no pointers, payload values, or RNG words and explicitly do
  not claim allocator reservation/fragmentation, checkpoint copies, or tape.
- [x] Existing family acquisition, checkpoint/restart, diagnostics,
  communication, resident execution, and direct-only export contracts remain
  compatible; P4 preserves existing RNG sidecars and initializes only new ones.
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
