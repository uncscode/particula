# Working with a GPU

<a id="turn-a-result-into-a-next-experiment"></a>

<a id="choose-a-route-through-the-studies"></a>

<a id="gpu-examples"></a>

Start with a native CPU aerosol and prepare supported processes on the GPU. Use the direct and resident tutorials for low-level interfaces. Native CUDA publication runs require a compatible NVIDIA device.

Use the prepared simulation to carry a native CPU aerosol onto the GPU and bring a detached result back. Direct kernels expose the state and scratch arrays for a particular physical operation. Resident simulations keep a session across steps and compare ordinary execution with graph capture. For the timing of this choice on an RTX 5060, see the [prepared scaling study](../../benchmarks/prepared-scaling/gpu_scaling_results.py).

<!-- STUDY_LIST_START -->
- [Build on the CPU, evolve on CUDA, analyze on the CPU](gpu_prepared_quick_start.py)
- [Direct kernels](direct-kernels/index.md)
- [Resident simulations](resident/index.md)
<!-- STUDY_LIST_END -->
