# Nucleation

The supported path is a deterministic, CPU-only, one-box `Nucleation` runnable
using fixed-capacity slots and partitioning gas. It preserves the supplied
`Aerosol` identity and uses `ExhaustionControls` when capacity is exhausted.

## Supported CPU runnable

- [Source/download: CPU nucleation](cpu_nucleation.py)
- Run: `python docs/Examples/Nucleation/cpu_nucleation.py`
- [Feature contract](../../Features/nucleation_strategy_system.md)

## Direct-Warp low-level example

The [direct-Warp source/download](gpu_direct_nucleation.py) demonstrates the
package-exported `nucleation_step_gpu` with caller-owned transfers, sidecars,
and synchronization. It requires Warp and defaults to Warp CPU; it is not a
replacement for the supported CPU runnable or a CPU fallback.

- Run: `python -Werror docs/Examples/Nucleation/gpu_direct_nucleation.py`

## Illustrative custom workflow

The [single-species notebook](Notebooks/Custom_Nucleation_Single_Species.ipynb)
is an illustrative custom workflow. Its direct facade mutation and custom mass
source are not the supported E6-F7 API.
