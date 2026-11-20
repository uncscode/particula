# Condensation Parity Walkthrough Ownership Record

This independent condensation walkthrough record is the source of truth for routing deferred capabilities
from the bounded direct-condensation walkthrough.

## Supported evidence boundary

[Supported evidence boundary in the study](../../../studies/validation/condensation/direct-kernel/reference.md#supported-evidence-boundary)

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
python docs/studies/validation/condensation/direct-kernel/gpu_condensation_parity_walkthrough.py
pytest examples_tests/gpu_condensation_parity_walkthrough_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_stiffness_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_graph_capture_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_autodiff_test.py -q -Werror
```
