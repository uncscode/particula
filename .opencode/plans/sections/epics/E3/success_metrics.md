# Success Metrics

## Correctness Metrics

- Repeated `coagulation_step_gpu` calls with caller-supplied `rng_states` show
  advancing state and avoid repeated identical draw patterns caused by reset.
- Mixed NPF/droplet coagulation tests expose acceptance and conservation
  behavior without flaky exact stochastic equality.
- Deterministic latent-heat CPU baseline conserves expected mass/energy
  quantities within documented tolerances.

## API and Documentation Metrics

- Supported import paths for low-level condensation and coagulation kernels are
  documented and tested.
- A direct GPU quick-start runs on Warp CPU and documents CUDA-if-available
  behavior.
- `CondensationLatentHeat` has a runnable docs example using public CPU APIs.
- Roadmap documentation captures the one-thread-per-box decision and sampling
  limitations or follow-up scope.

## Validation Metrics

- Warp-dependent tests use `pytest.importorskip("warp")` or equivalent optional
  dependency handling.
- Device-parametrized tests include CPU and CUDA-if-available devices without
  requiring CUDA.
- Stochastic tolerance policy is referenced by new coagulation tests.
- Coverage thresholds remain unchanged.

## Program Metrics

- All seven child feature tracks are created with clear dependencies.
- No maintenance or research tracks are required for Epic C.
- No hidden CPU/GPU transfers, high-level backend selection, or new GPU physics
  are introduced while closing the epic.
