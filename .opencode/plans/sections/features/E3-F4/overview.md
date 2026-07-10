# Overview

Feature `E3-F4` resolves the public low-level GPU kernel import path for direct
condensation and coagulation step usage. The feature is part of parent epic
`E3` and focuses on making the existing GPU kernel surface discoverable without
introducing high-level backend selection or hidden CPU/GPU transfer behavior.

The current codebase already exposes direct step functions from
`particula.gpu.kernels`:

- `condensation_step_gpu`
- `coagulation_step_gpu`

Top-level `particula.gpu` remains focused on availability, data transfer, and
context helpers such as `WARP_AVAILABLE`, `to_warp_particle_data`,
`to_warp_gas_data`, `to_warp_environment_data`, and `gpu_context`. Phase
`E3-F4-P1` finalized `particula.gpu.kernels` as the supported public import
path for the two direct step functions, kept `particula.gpu` intentionally
non-reexporting, excluded lower-level helper kernels from the package-level
public surface, and added focused regression coverage in
`particula/gpu/tests/kernel_exports_test.py`.

## Goals

- Select and document the stable import path for direct low-level GPU kernels.
- Add regression tests so the chosen import/export path cannot drift silently.
- Document the direct-kernel import contract without implying a broader
  top-level quick-start API than the shipped code supports.
- Demonstrate explicit transfer boundaries with `ParticleData`, `GasData`,
  transfer helpers, `gpu_context`, and `WARP_AVAILABLE`.
- Add troubleshooting guidance for missing Warp, missing CUDA, device mismatch,
  and invalid environment input combinations.

## Motivation

Epic `E3` is hardening the data-oriented GPU path. Earlier feature tracks cover
RNG persistence and coagulation sampling/performance evidence. This feature
turns the low-level API into an explicit, tested, documented user path while
preserving the current design principle: users opt into GPU-resident data and
explicitly transfer data at the boundary.
