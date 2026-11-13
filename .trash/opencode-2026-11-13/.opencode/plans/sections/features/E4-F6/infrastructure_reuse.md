# Infrastructure Reuse

- `condensation_step_gpu` and its validation/buffer setup in
  `particula/gpu/kernels/condensation.py:1708-2470` are the production surface
  under qualification; extend behavior delivered by E4-F1 through E4-F5.
- Shape/device validators in `particula/gpu/kernels/condensation.py:270-384`
  provide the fail-before-mutation contract.
- `device` and `warp_devices` patterns in
  `particula/gpu/kernels/tests/_condensation_test_support.py:16-86` and
  `particula/gpu/tests/cuda_availability.py:14-36` preserve mandatory Warp CPU
  plus optional CUDA execution.
- Deterministic fp64 fixtures and the independent NumPy reference in
  `_condensation_test_support.py:89-149,558-663` should be extended, not
  replaced with a reference that duplicates GPU vectorization assumptions.
- Existing parity and fixed-loop evidence at
  `_condensation_test_support.py:2637-2730,2891-3049` supplies conventions for
  explicit tolerances, deterministic repeats, and stable scratch identity.
- CPU conservation precedents live in
  `particula/integration_tests/condensation_particle_resolved_test.py:121-147`
  and `condensation_latent_heat_conservation_test.py:231-299`.
- The separate strict conservation assertion pattern at
  `particula/gpu/kernels/tests/coagulation_test.py:2562-2633` should be reused.
- `particula/gpu/tests/benchmark_test.py:549-581` demonstrates Warp capture
  mechanics, while `docs/Features/Roadmap/warp-autodiff-limitations.md:344-403`
  defines the bounded autodiff evidence and caveats.
- Support helpers remain private and runnable tests are re-exported from
  discoverable `*_test.py` modules per `.opencode/guides/testing_guide.md`.
