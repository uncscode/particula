# Dependencies

## Internal Dependencies

- **E3-F6:** Runnable CPU `CondensationLatentHeat` example and guidance. E3-F7
  can proceed with code-level APIs, but documentation should cross-link to E3-F6
  when that example is available.
- **Existing particle-resolved condensation integration test:**
  `particula/integration_tests/condensation_particle_resolved_test.py` is the
  fixture and assertion style dependency.
- **CPU latent-heat strategy:** `CondensationLatentHeat` and
  `last_latent_heat_energy` in `condensation_strategies.py`.
- **Latent heat strategy:** `ConstantLatentHeat` in
  `particula/gas/latent_heat_strategies.py`.
- **Runnable wrapper:** `MassCondensation.execute()` in
  `particula/dynamics/particle_process.py`.

## External Dependencies

- No new third-party dependencies are expected.
- Existing NumPy and pytest capabilities are sufficient.

## Ordering Dependencies

1. Build the CPU fixture before tightening assertions.
2. Prove conservation and energy bookkeeping before documenting the baseline as
   Epic D reference evidence.
3. Do not depend on GPU availability, Warp device fixtures, or CUDA markers.
