# Condensation Parity Walkthrough Ownership Record

This E5-F8-P3 record is the source of truth for routing deferred capabilities
from the bounded direct-condensation walkthrough.

## Supported evidence boundary

The [walkthrough source](../../Examples/gpu_condensation_parity_walkthrough.py)
compares an independent fp64, fixed-four-substep NumPy oracle with the low-level
direct GPU kernel. Its [walkthrough regression test](../../../particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py)
keeps that bounded comparison executable.

Physics, conservation, and energy are separately evaluated evidence. Warp CPU
is the supported baseline when installed; CUDA is optional, additive evidence.
This record does not establish high-level CPU strategy parity, `Runnable`
parity, or general CPU workflow parity. The
[fixed-four direct-kernel evidence boundary](condensation-stiffness-study.md#p1--p4-production-hook-evidence-boundary)
states the related P1--P4 production-hook limit.

## Deferred capability ownership

| Deferred capability | Downstream owner | Entry gate | E5-F8 non-claim |
| --- | --- | --- | --- |
| thermal_work consumption | Future approved condensation numerical-method work | An approved numerical-method plan and numerical/validation contract are required. | thermal_work consumption is outside the walkthrough/E5-F8 scope. |
| temperature feedback | Future approved condensation numerical-method work | An approved numerical-method plan and numerical/validation contract are required. | temperature feedback is outside the walkthrough/E5-F8 scope. |
| adaptive stepping | Future approved condensation numerical-method work | An approved numerical-method plan and numerical/validation contract are required. | adaptive stepping is outside the walkthrough/E5-F8 scope. |
| backend selection | Epic G | Backend-selection and GPU-resident integration work is required. | backend selection is outside the walkthrough/E5-F8 scope. |
| high-level Aerosol/Runnable integration | Epic G | Backend-selection and GPU-resident integration work is required. | high-level Aerosol/Runnable integration is outside the walkthrough/E5-F8 scope. |
| GPU-resident/full-workflow coupling | Epic G | Backend-selection and GPU-resident integration work is required. | GPU-resident/full-workflow coupling is outside the walkthrough/E5-F8 scope. |
| general CPU workflow/strategy parity | Epic G | Backend-selection and GPU-resident integration work is required. | general CPU workflow/strategy parity is outside the walkthrough/E5-F8 scope. |
| graph capture/replay | Epic H | Graph-capturable stable-shape loop validation and capture/performance exit work are required. | graph capture/replay is outside the walkthrough/E5-F8 scope. |
| host-validation/capture separation | Epic H | Graph-capturable stable-shape loop validation and capture/performance exit work are required. | host-validation/capture separation is outside the walkthrough/E5-F8 scope. |
| performance/benchmarking | Epic H | Benchmark/memory-budget exit work is required. | performance/benchmarking is outside the walkthrough/E5-F8 scope. |
| memory-budget work | Epic H | Benchmark/memory-budget exit work is required. | memory-budget work is outside the walkthrough/E5-F8 scope. |
| broad state/multi-step autodiff | Epic I | Differentiability and gradient-validation work is required; it is distinct from the shipped raw-rate interior probe. | broad state/multi-step autodiff is outside the walkthrough/E5-F8 scope. |
| phase-aware surface tension | Approved condensation-physics expansion | An approved plan and physics-validation contract are required. | phase-aware surface tension is outside the walkthrough/E5-F8 scope. |
| BAT activity | Approved condensation-physics expansion | An approved plan and physics-validation contract are required. | BAT activity is outside the walkthrough/E5-F8 scope. |

## Focused reproduction commands

```bash
python docs/Examples/gpu_condensation_parity_walkthrough.py
pytest particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_stiffness_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_graph_capture_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_autodiff_test.py -q -Werror
pytest particula/tests/condensation_parity_walkthrough_docs_test.py -q -Werror
```

## Downstream roadmap owners

Deferred integration and workflow work belongs to [Epic G](data-oriented-gpu.md#epic-g-backend-selection-and-gpu-resident-simulation).
Capture, validation separation, and performance work belong to
[Epic H](data-oriented-gpu.md#epic-h-graph-capture-and-performance). Broad
state autodiff belongs to
[Epic I](data-oriented-gpu.md#epic-i-differentiability-and-global-optimization).
