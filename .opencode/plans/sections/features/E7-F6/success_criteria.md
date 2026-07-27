# Success Criteria

- [ ] Every unsupported backend/device/process/capability outcome raises the
  documented typed error with stable reason metadata.
- [ ] Importing the public execution API succeeds when Warp is absent.
- [ ] Fallback is disabled by default and requires an explicit typed request.
- [ ] Explicit fallback occurs only before upload/mutation with CPU-authoritative
  state or after a caller-requested restore boundary.
- [ ] Runtime/kernel/adapter exceptions propagate without CPU retry.
- [ ] Rejected and failed requests perform zero hidden conversion,
  synchronization, kernel launch, fallback adapter call, or state mutation.
- [ ] Results distinguish requested and selected backend and retain the fallback reason.
- [ ] Public `__all__` tests accept intended names and reject registries,
  concrete adapters, sidecars, configs, and helper kernels.
- [ ] Existing low-level GPU imports remain callable and are documented
  experimental; no breaking removal occurs.
- [ ] Focused tests pass with at least 80% execution-package coverage, CUDA skips
  cleanly, Ruff/mypy pass, and strict documentation builds.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Silent backend transitions | Undefined | 0 | Transfer/adapter spy tests |
| Typed capability/availability failure branches | Fragmented errors | 100% enumerated | Error matrix tests |
| Public execution imports in CPU-only subprocess | Not available | 100% approved surface | Import tests |
| Post-invocation CPU retries | Undefined | 0 | Sentinel integration tests |
| Changed execution-module coverage | N/A | >= 80% | pytest-cov |
| Existing direct GPU import regressions | 0 expected | 0 | Kernel export tests |
