# Infrastructure Reuse

- `Coagulation.execute()` in `particula/dynamics/particle_process.py:560-631`
  is the CPU runnable boundary. Delegate exact `time_step` and `sub_steps`; do
  not reproduce its loop in selection code.
- `BrownianCoagulationStrategy` in
  `particula/dynamics/coagulation/coagulation_strategy/brownian_coagulation_strategy.py:23-131`
  supplies CPU Brownian physics and remains strategy-owned.
- `coagulation_step_gpu()` in
  `particula/gpu/kernels/coagulation.py:2102-2578` is the authoritative direct
  Warp implementation, validation order, mutation boundary, and return tuple.
- `initialize_coagulation_rng_states()` in
  `particula/gpu/kernels/coagulation.py:1930` supplies explicit seed-once/reset
  behavior; `_validate_rng_states()` at line 1811 defines shape, dtype, and
  device checks reused by the direct step.
- The selector contract at
  `particula/gpu/kernels/coagulation.py:1011-1013` and setup rules at
  `particula/gpu/kernels/coagulation.py:2341-2356` establish that provided
  `(n_boxes,)` RNG state mutates in place and is not reseeded unless requested.
- `CoagulationMechanismConfig` in
  `particula/gpu/kernels/_coagulation_config.py` can express explicit Brownian,
  particle-resolved execution but must remain at its deliberate concrete API
  location unless E7-F6 authorizes a stable wrapper/export.
- `particula/gpu/kernels/tests/coagulation_test.py` and
  `coagulation_validation_test.py` provide reusable schema, identity, reset,
  rejection-order, and mutation fixtures.
- `particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py`
  provides aggregate stochastic-validation patterns; do not assert exact
  CPU/Warp trajectories.
- `particula/gpu/tests/process_sequence_test.py:1345-1445` already demonstrates
  repeated calls with resident coagulation RNG and is prior art for integration.
- `particula/gpu/tests/kernel_exports_test.py:18-81,211-222` guards the current
  direct-kernel export boundary and should be extended only deliberately.
- `docs/Examples/gpu_coagulation_direct.py` and
  `particula/gpu/tests/gpu_coagulation_direct_example_test.py:300-334` show
  repeated-call RNG identity and are the documentation seed.
- E7-F1's `particula.execution` plan defines typed request/context/result and
  registry patterns. E7-F6 freezes capability, availability, fallback, and
  export policy before this feature registers an adapter.
