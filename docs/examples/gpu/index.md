# GPU examples

Run these sources from the repository root after installing Particula
and optional Warp support. Read the [GPU guide](../../guide/gpu/index.md) first.
Warp CPU is the baseline for the direct and uncaptured resident examples;
graph capture requires native CUDA and does not fall back to CPU.

| Learning objective | Source | Scope |
| --- | --- | --- |
| Explicit transfers and a first direct step | [Direct quick start](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_direct_kernels_quick_start.py) | Tutorial |
| Direct Brownian coagulation | [Coagulation](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_coagulation_direct.py) | Tutorial |
| Five explicit process calls | [Complete process sequence](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_complete_process_sequence.py) | Illustration, not a scheduler |
| Compare fixed-four-substep condensation | [Parity walkthrough](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_condensation_parity_walkthrough.py) | Bounded numerical validation |
| Activate nucleated particles | [Direct nucleation](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_direct_nucleation.py) | Fixed-capacity direct step |
| Checkpoint and restart state | [Resident session](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/resident/gpu_resident_session.py) | Explicit lifecycle |
| Schedule several resident steps | [Multi-timestep loop](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/resident/gpu_resident_multi_timestep.py) | Resident workflow |
| Capture and replay a stable loop | [Graph capture](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/resident/gpu_resident_graph_capture.py) | Native CUDA only |

For example:

```bash
python docs/examples/gpu/direct/gpu_direct_kernels_quick_start.py
```

Each source documents its inputs and assertions. Clean optional-device skips
are not parity or performance evidence. Consult [validation records](../../contribute/validation/index.md)
for the evidence boundaries.
