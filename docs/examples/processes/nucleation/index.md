# Nucleation

The supported path is a deterministic, CPU-only, one-box `Nucleation` runnable
using fixed-capacity slots and partitioning gas. It preserves the supplied
`Aerosol` identity and uses `ExhaustionControls` when capacity is exhausted.

## Supported CPU runnable

- [Source/download: CPU nucleation](https://github.com/uncscode/particula/blob/main/docs/examples/processes/nucleation/cpu_nucleation.py)
- Run: `python docs/examples/processes/nucleation/cpu_nucleation.py`
- [Feature contract](../../../guide/processes/nucleation.md)

## Direct-Warp low-level example

The [direct-Warp source/download](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_direct_nucleation.py) demonstrates the
package-exported `nucleation_step_gpu` with caller-owned transfers, sidecars,
and synchronization. It requires Warp and defaults to Warp CPU; it is not a
replacement for the supported CPU runnable or a CPU fallback.

- Run: `python -Werror docs/examples/gpu/direct/gpu_direct_nucleation.py`

## Illustrative custom workflow

The [single-species notebook](custom_nucleation_single_species.ipynb)
is an illustrative custom workflow. Its direct facade mutation and custom mass
source are not the supported CPU nucleation API.
