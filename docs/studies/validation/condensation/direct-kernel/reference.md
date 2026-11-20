# Fixed-four-substep condensation reference

See the [qualification record](../../../../contribute/validation/gpu/condensation-parity.md) for the associated implementation decision and reproduction gates.

## Supported evidence boundary

The [walkthrough source](https://github.com/uncscode/particula/blob/main/docs/studies/validation/condensation/direct-kernel/gpu_condensation_parity_walkthrough.py)
compares an independent fp64, fixed-four-substep NumPy oracle with the low-level
direct GPU kernel. Its [walkthrough regression test](https://github.com/Gorkowski/particula/blob/main/examples_tests/gpu_condensation_parity_walkthrough_test.py)
keeps that bounded comparison executable.

Physics, conservation, and energy are separately evaluated evidence. Warp CPU
is the supported baseline when installed; CUDA is optional, additive evidence.
This record does not establish high-level CPU strategy parity, `Runnable`
parity, or general CPU workflow parity. The
[fixed-four direct-kernel evidence boundary](../stiffness.md#production-hook-evidence-boundary)
states the related production-hook evidence limit.
