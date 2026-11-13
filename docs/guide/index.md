# Guide

Start with a small CPU simulation, then choose a process or execution path.
These guides describe the v0.3.0 native development API; follow the checkout
instructions when your installed release predates it.

## Start here

1. [Install Particula](../Examples/Setup_Particula/index.md).
2. [Run a first CPU simulation](getting-started/first-simulation.md).
3. Learn [container construction](../Features/particle-data-migration/construction.md)
   and [state access and ownership](../Features/particle-data-migration/containers.md).
4. Choose a [scientific process](../Features/native_scientific_processes.md)
   and a related [example](../Examples/index.md).

## Choose a path

| Task | Read next |
| --- | --- |
| Configure particle, gas, and environment state | [Data model](../Features/data-containers-and-gpu-foundations.md) |
| Simulate growth, collisions, sources, or losses | [Processes](../Features/index.md) |
| Use explicit Warp transfers and device execution | [GPU selection](../Features/backend_selection.md), then [resident checkpoints](../Features/gpu_resident_checkpoints.md) |
| Capture a resident loop on native CUDA | [Graph-capture runbook](../Features/gpu_graph_capture.md) |
| Upgrade legacy scripts | [v0.3.0 migration](../Features/particle-data-migration/index.md) |

CPU process examples are the first-use path. Direct Warp examples require
explicit transfers; resident execution has additional lifecycle rules. Native
CUDA capture has no CPU fallback. See each guide's supported scope before
combining execution paths.
