# Success Criteria

## Pass / Fail Criteria

- [ ] One documented import surface is selected and locked by regression tests:
  `particula.gpu.kernels` for `condensation_step_gpu` and
  `coagulation_step_gpu`.
- [ ] Supported direct imports succeed without launching kernels, and rejected
  raw/internal-symbol imports are guarded by explicit negative assertions when
  they remain unsupported.
- [ ] The quick-start runs on Warp CPU from a clean entrypoint, demonstrates
  both condensation and coagulation direct-step usage, and keeps all
  `to_warp_*` / `from_warp_*` transfer boundaries explicit.
- [ ] Missing Warp, missing CUDA, device mismatch, and mixed `environment=`
  input failure modes are documented in the shipped example or adjacent GPU
  docs with deterministic smoke-test coverage where practical.
- [ ] Focused kernel regression tests still pass after the import-path and
  example changes, and CUDA remains optional rather than required for default
  validation.
- [ ] No backend selector, hidden synchronization, or broad raw-kernel API
  promotion is introduced while closing the feature.

## Evidence Metrics

| Metric | Completion Signal | Evidence Source |
| --- | --- | --- |
| Import-path stability | Supported path resolves and unsupported path behavior is intentional | `particula/gpu/tests/kernel_exports_test.py` |
| Public-surface narrowness | `particula.gpu` stays non-reexporting and `particula.gpu.kernels.__all__` is limited to the two step functions | Export assertions in `kernel_exports_test.py` |
| Runnable quick-start | Example executes on Warp CPU and exits cleanly when Warp is absent | `particula/gpu/tests/gpu_direct_kernels_example_test.py` |
| Transfer-boundary clarity | Example shows explicit transfer helpers for particle and gas data, with direct scalar thermodynamic inputs | Reviewed docs/example diff plus smoke-test assertions |
| Troubleshooting coverage | Device mismatch and optional CUDA guidance ship with the example/docs | `docs/Examples/` plus roadmap/foundation docs |

## Definition of Done

Reviewers can identify one supported low-level import path, run the quick-start
on Warp CPU, and see exactly where data ownership crosses the CPU↔GPU boundary
without reading raw kernel internals.
