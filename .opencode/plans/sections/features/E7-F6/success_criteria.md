# Success Criteria

- [x] Every unsupported backend/device/process/capability outcome raises the
  documented typed error with stable reason metadata.
- [x] Importing the public execution API succeeds when Warp is absent.
- [x] P3 fallback is disabled by default and requires an explicit typed request.
- [x] P3 explicit fallback occurs only before upload/mutation with exact
  CPU-authoritative state or at a caller-asserted restored boundary.
- [x] P3 runtime/kernel/adapter exceptions propagate without CPU retry.
- [x] P3 rejected and failed requests perform zero hidden conversion,
  synchronization, kernel launch, fallback adapter call, or state mutation.
- [x] P3 fallback results distinguish requested and selected backend and retain
  the capability reason without modifying native result metadata.
- [x] Public `__all__` tests accept intended names and reject registries,
  concrete adapters, sidecars, configs, and helper kernels; they also preserve
  ordered identity re-exports and CPU-only import neutrality.
- [x] Existing low-level GPU imports remain callable and are documented
  experimental; no breaking removal occurs.
- [x] P6 documentation regressions validate the stable/concrete-only boundary,
  closed fallback outcomes, no-movement resolver wording, the executable
  selection-only fence, guarded resident pseudocode, and Markdown links; the
  focused execution and documentation tests and strict documentation build pass.
- [x] E7-F6 documentation does not claim automatic movement, recovery, retry,
  or runtime behavior changes. CUDA, transport, detailed RNG/restart policy,
  broad orchestration, and direct-kernel follow-up remain downstream.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Silent backend transitions | Undefined | 0 | Transfer/adapter spy tests |
| Typed capability/availability failure branches | Fragmented errors | 100% enumerated | Error matrix tests |
| Public execution imports in CPU-only subprocess | Not available | 100% approved surface | Import tests |
| Post-invocation CPU retries | Undefined | 0 | Sentinel integration tests |
| Changed execution-module coverage | N/A | >= 80% | pytest-cov |
| Existing direct GPU import regressions | 0 expected | 0 | Kernel export tests |
