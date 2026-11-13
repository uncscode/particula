# Examples

Start small, then build toward full scientific workloads. These examples target
native v0.3.0 development APIs; use the [installation guide](../guide/getting-started/installation/index.md).
Notebook pages display saved results in the full documentation build. Standalone
sources remain runnable from the repository checkout.

## Fundamentals

1. [First CPU simulation](../guide/getting-started/first-simulation.md)
2. [Native aerosol](fundamentals/index.md)
3. [Data containers and explicit transfers](fundamentals/data-containers.md)
4. [Gas and environment](fundamentals/gas/index.md)
5. [Particle distributions and properties](fundamentals/particles/index.md)

## Processes and thermodynamics

- [Condensation and coagulation](processes/index.md)
- [Wall loss](processes/wall-loss/index.md)
- [Nucleation](processes/nucleation/index.md)
- [Activity](thermodynamics/activity/index.md)
- [Equilibria](thermodynamics/equilibria/index.md)
- [CPU dilution](../guide/processes/dilution.md#boundaries-and-example)

## GPU examples

These standalone examples use explicit caller-owned transfers and device state.
Warp CPU is the baseline when installed; CUDA capture is a separate path.

| Purpose | Source | Requirement |
| --- | --- | --- |
| Direct condensation | [Quick start](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_direct_kernels_quick_start.py) | Optional Warp; CPU device by default |
| Direct coagulation | [Brownian coagulation](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_coagulation_direct.py) | Optional Warp; CPU device by default |
| Combine direct processes | [Five-process sequence](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_complete_process_sequence.py) | Optional Warp; illustrative explicit calls |
| Inspect numerical agreement | [Condensation parity](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_condensation_parity_walkthrough.py) | Validation demonstration, not general strategy parity |
| Manage session state | [Resident lifecycle](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/resident/gpu_resident_session.py) | Optional Warp; explicit restart/finalization |
| Run several resident steps | [Resident loop](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/resident/gpu_resident_multi_timestep.py) | Optional Warp; explicit scheduling |
| Capture and replay | [Graph capture](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/resident/gpu_resident_graph_capture.py) | Native CUDA only; no CPU fallback |

## Full simulations

[Scientific simulations](simulations/index.md) retain their complete workloads
and limitations. They are longer studies, not installation smoke tests. Start
with a process tutorial before adapting a full chamber or atmospheric case.

## Tutorials

The fundamentals and process indexes above are the tutorial entry points. For
underlying equations, use [Theory](../theory/index.md); for support boundaries,
use the [Guide](../guide/index.md).
