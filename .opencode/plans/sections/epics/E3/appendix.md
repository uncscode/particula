# Appendix

## Relevant Files from Codebase Research

- `docs/Features/Roadmap/data-oriented-gpu.md` — Epic C roadmap source,
  known kernel issues, and exit bar.
- `particula/gpu/__init__.py` — current top-level GPU exports.
- `particula/gpu/kernels/__init__.py` — current low-level kernel exports.
- `particula/gpu/kernels/coagulation.py` — coagulation RNG initialization,
  rejection sampling, and one-thread-per-box design.
- `particula/gpu/kernels/condensation.py` — direct condensation GPU entry API.
- `particula/gpu/kernels/environment.py` — explicit environment/direct input
  normalization without hidden transfers.
- `particula/gpu/conversion.py` — explicit CPU/Warp transfer helpers.
- `particula/gpu/tests/cuda_availability.py` — CPU plus CUDA-if-available device
  helper.
- `particula/gpu/kernels/tests/coagulation_test.py` — stochastic, mass
  conservation, and preallocated-buffer test patterns.
- `docs/Examples/data_containers_and_gpu_foundations.py` — optional-Warp
  runnable example pattern.
- `particula/gpu/tests/data_containers_example_test.py` — docs-example smoke
  test pattern.
- `particula/dynamics/condensation/condensation_strategies.py` — CPU
  `CondensationLatentHeat` implementation.
- `particula/dynamics/condensation/mass_transfer.py` — latent heat energy
  diagnostic helper.
- `particula/integration_tests/condensation_particle_resolved_test.py` —
  existing integration-level condensation conservation style.

## Preserved Issue Constraints

- No high-level backend selection.
- No new GPU physics in Epic C.
- No hidden CPU/GPU transfers.
- CUDA optional.

## Tooling Note

The epic templates and `list-sections` output expose 13 canonical section files.
The drafter contract referenced 15 sections and required `add-phase`, but the
plan tool reported that epic plans do not support phases. This draft therefore
uses the 13 canonical files and records workstreams as child feature tracks and
milestones.
