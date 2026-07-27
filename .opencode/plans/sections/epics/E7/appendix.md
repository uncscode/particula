# Appendix

## Scope Authority

- GitHub issue #1451: Backend Selection and GPU-Resident Simulation.
- `docs/Features/Roadmap/data-oriented-gpu.md:1461-1593`: Epic G tracks,
  integration boundaries, and exit bar.

## Architecture and Implementation References

- `particula/runnable.py:36-218`: CPU runnable and sequence contracts.
- `particula/dynamics/particle_process.py:458-819`: CPU process runnables.
- `particula/gpu/warp_types.py:24-184`: fixed-shape multi-box Warp state.
- `particula/gpu/conversion.py:120-317,422-666`: explicit upload, restore,
  synchronization, and particle-only context prior art.
- `particula/gpu/kernels/__init__.py:1-64`: deliberate direct-step exports.
- `particula/gpu/kernels/thermodynamics.py:318-377`: on-device vapor-pressure
  refresh needed after temperature changes.
- `docs/Examples/gpu_complete_process_sequence.py:380-494`: one-upload,
  five-process, one-restore illustrative sequence.

## Validation References

- `particula/gpu/tests/process_sequence_test.py:808-872`: no-intermediate-
  restore guard.
- `particula/gpu/tests/process_sequence_test.py:1651-1677`: five-process
  direct composition fixture.
- `particula/gpu/tests/gpu_complete_process_sequence_example_test.py:494-801`:
  ordering, transfer, and failure-propagation regression patterns.
- `particula/tests/runnable_test.py`: CPU scheduling assertions.
- `particula/gpu/tests/kernel_exports_test.py`: public-export boundaries.

## Rejected Approaches

- Treating the illustrative complete-process example as a production scheduler.
- Passing Warp state through the existing `Aerosol`-typed runnable unchanged.
- Per-step host orchestration that downloads state or reseeds random streams.
- Silent fallback after a GPU adapter error.
- Dynamic resizing/compaction in place of shipped fixed-slot contracts.
- Broad export of concrete scratch and configuration internals.
- Pulling graph capture/performance or autodiff into this epic.
