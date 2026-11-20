# GPU Coagulation Validation Record

This maintainer and release-review record documents evidence for the existing
direct GPU-coagulation path. It does not introduce a production API, physics,
public export, runnable integration, CPU fallback, graph capture/replay,
autodiff, adaptive stepping, or a performance claim.

## Supported Evidence Matrix

[Supported Evidence Matrix in the study](../../../studies/validation/coagulation/direct-coagulation/reference.md#supported-evidence-matrix)

## Deferred Masks

The three-way masks `7`, `11`, `13`, and `14` are deferred and fail closed.
They are not supported combinations, and this record makes no claim for any
other combination.

## Evidence Boundaries

[Evidence Boundaries in the study](../../../studies/validation/coagulation/direct-coagulation/reference.md#evidence-boundaries)

## Warning-Clean Reproduction Commands

```bash
pytest particula/gpu/kernels/tests/coagulation_validation_test.py -q -m "warp and gpu_parity" -Werror
pytest particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py -q -m "warp and stochastic and not cuda" -Werror
pytest particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py -q -m "warp and cuda" -Werror
pytest particula/gpu/kernels/tests/coagulation_test.py -q -Werror
```

Missing Warp or unavailable CUDA is valid only when an existing guarded test
skips. A skip is not CPU/Warp or CUDA parity evidence.

## Historical performance context

[Recorded coagulation benchmark evidence](../../../studies/benchmarks/historical-coagulation/index.md) describes
a dated, machine-specific tradeoff. It is separate from the physics validation
above and from current resident/capture qualification.

## Adaptive outer intervals on native CUDA

[Adaptive outer intervals on native CUDA in the study](../../../studies/validation/coagulation/adaptive/trajectories.md#adaptive-outer-intervals-on-native-cuda)
