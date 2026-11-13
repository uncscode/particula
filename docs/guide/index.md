# Guide

Start with a small CPU simulation, then choose a process or execution path.

## Start here

1. [Install Particula](getting-started/installation/index.md).
2. [Run a first CPU simulation](getting-started/first-simulation.md).
3. Learn [container construction](data-model/construction.md)
   and [state access and ownership](data-model/ownership.md).
4. Choose a [scientific process](processes/index.md)
   and a related [example](../examples/index.md).

## Choose a path

| Task | Read next |
| --- | --- |
| Configure particle, gas, and environment state | [Data model](data-model/index.md) |
| Simulate growth, collisions, sources, or losses | [Processes](processes/index.md) |
| Use explicit Warp transfers and device execution | [GPU selection](gpu/backend-selection.md), then [resident checkpoints](gpu/resident/checkpoints.md) |
| Capture a resident loop on native CUDA | [Graph-capture runbook](gpu/resident/graph-capture.md) |
| Upgrade legacy scripts | [v0.3.0 migration](migration/v0.3.0/index.md) |

CPU process examples are the first-use path. Direct Warp examples require
explicit transfers; resident execution has additional lifecycle rules. Native
CUDA capture has no CPU fallback. See each guide's supported scope before
combining execution paths.
