# Nucleation

The supported path is a deterministic, CPU-only, one-box `Nucleation` runnable
using fixed-capacity slots and partitioning gas. It preserves the supplied
`Aerosol` identity and uses `ExhaustionControls` when capacity is exhausted.

## Supported CPU runnable

- [Source/download: CPU nucleation](cpu_nucleation.py)
- Run: `python docs/Examples/Nucleation/cpu_nucleation.py`
- [Feature contract](../../Features/nucleation_strategy_system.md)

## Illustrative custom workflow

The [single-species notebook](Notebooks/Custom_Nucleation_Single_Species.ipynb)
is an illustrative custom workflow. Its direct facade mutation and custom mass
source are not the supported E6-F7 API.
