# Infrastructure Reuse

- `condensation_step_gpu` in
  `particula/gpu/kernels/condensation.py:1` is the bounded low-level entry point;
  call it without changing its two-item return or caller-owned sidecars.
- `particula/gpu/kernels/tests/_condensation_test_support.py:316` contains the
  physical-tolerance conservation assertion pattern and independent oracle
  helpers. Reuse formulas and fixture conventions, but keep walkthrough
  expected-state construction independent from Warp arrays.
- `particula/gpu/kernels/tests/condensation_test.py:170` exposes the discoverable
  fixed-four-substep CPU-oracle/Warp regression matrix and energy cases.
- `docs/Examples/gpu_direct_kernels_quick_start.py:218-327` demonstrates explicit
  conversion, stable fp64 scratch buffers, `latent_heat`, `energy_transfer`,
  synchronization/restoration, and caller ownership. Follow its no-hidden-
  transfer pattern without turning this feature into a second quick start.
- `docs/Features/condensation_strategy_system.md:676-717` defines the latent-
  heat formula, signed energy identity, P1/P2 evidence boundary, and non-claims.
- `docs/Features/data-containers-and-gpu-foundations.md:589-606` already states
  that parity, conservation, and energy/bookkeeping are distinct evidence
  classes and supplies focused commands.
- `docs/Features/Roadmap/condensation-stiffness-study.md:181-198` records the
  fixed-four fp64 production-hook boundary and current P1-P4 evidence.
- `docs/Features/Roadmap/data-oriented-gpu.md:999-1029` is authoritative for the
  Epic D carry-forward and the minimum deferred-capability list.
- `docs/Features/Roadmap/data-oriented-gpu.md:1036`, `:1084`, `:1218`, and
  `:1371` define downstream Epic F-I ownership lanes.
- `particula/gpu/tests/gpu_direct_kernels_example_test.py` provides the
  repository pattern for importing and testing runnable documentation examples
  without requiring CUDA.
- `particula/tests/condensation_latent_heat_docs_test.py` provides the existing
  link/content regression pattern for condensation documentation.
