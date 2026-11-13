# Condensation Parity Walkthrough Ownership Record

This independent condensation walkthrough record is the source of truth for routing deferred capabilities
from the bounded direct-condensation walkthrough.

## Supported evidence boundary

The [walkthrough source](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_condensation_parity_walkthrough.py)
compares an independent fp64, fixed-four-substep NumPy oracle with the low-level
direct GPU kernel. Its [walkthrough regression test](https://github.com/Gorkowski/particula/blob/main/examples_tests/gpu_condensation_parity_walkthrough_test.py)
keeps that bounded comparison executable.

Physics, conservation, and energy are separately evaluated evidence. Warp CPU
is the supported baseline when installed; CUDA is optional, additive evidence.
This record does not establish high-level CPU strategy parity, `Runnable`
parity, or general CPU workflow parity. The
[fixed-four direct-kernel evidence boundary](../numerics/condensation-stiffness.md#production-hook-evidence-boundary)
states the related production-hook evidence limit.

## Scope boundaries and current owners

This walkthrough establishes direct fixed-four-substep kernel evidence only.
It does not qualify the following independent paths:

| Capability | Canonical owner and evidence requirement |
| --- | --- |
| `thermal_work` consumption, temperature feedback, adaptive stepping | Future approved numerical-method work with its own numerical/validation contract |
| Backend selection, native/runnable integration, resident loops | [Execution](../../../guide/gpu/execution.md); use its explicit supported paths and separate tests, without inferring general CPU-strategy parity |
| Graph capture and host-validation separation | [Graph-capture runbook](../../../guide/gpu/resident/graph-capture.md); capture qualification requires separate evidence |
| Performance and memory budget | [Resident benchmarks](resident-benchmarks.md) and [qualification gates](graph-capture-qualification.md); no reviewed measurement is supplied by the walkthrough |
| Broad state/multi-step autodiff | [Pending initial-state optimization](../../roadmap.md); distinct from the bounded raw-rate interior probe |
| Phase-aware surface tension and BAT activity in the direct kernel | Future approved physics expansion and validation contract |

## Focused reproduction commands

```bash
python docs/examples/gpu/direct/gpu_condensation_parity_walkthrough.py
pytest examples_tests/gpu_condensation_parity_walkthrough_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_stiffness_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_graph_capture_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_autodiff_test.py -q -Werror
```
