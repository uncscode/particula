# GPU execution

Start with the [prepared simulation workflow](prepared.md): construct a native
CPU aerosol, upload once, evolve supported processes on CUDA, and download a
detached snapshot for analysis. Warp is a declared Particula dependency;
CUDA requires an available NVIDIA GPU and compatible driver. Device selection
is explicit. Direct kernels remain available for manual resource control.

| Task | Guide |
| --- | --- |
| Run native aerosol processes on CUDA | [Prepared simulations](prepared.md) |
| Choose the supported execution path | [Execution](execution.md), [support matrix](support.md), [backend selection](backend-selection.md) |
| Move state between CPU and Warp | [Transfers](transfers.md) |
| Manage fixed-capacity particle storage | [Slots](slots.md), [capacity exhaustion](slot-exhaustion.md) |
| Configure direct condensation | [Condensation](direct/condensation.md), [configuration](direct/condensation-configuration.md) |
| Configure direct coagulation | [Coagulation](direct/coagulation.md), [configuration](direct/coagulation-configuration.md) |
| Apply direct dilution or wall loss | [Dilution and wall loss](direct/dilution-and-wall-loss.md) |
| Inspect, finalize, or restart a resident session | [Checkpoints](resident/checkpoints.md) |
| Capture a stable resident loop on native CUDA | [Graph capture](resident/graph-capture.md) |
| Diagnose rejected calls | [Troubleshooting](troubleshooting.md) |

Use the [GPU examples](../../examples/gpu/index.md) for runnable entry points.
CUDA capture has no CPU fallback. [Qualification gates](../../contribute/validation/gpu/graph-capture-qualification.md)
remain separate from implementation support and do not provide measured
performance recommendations.
