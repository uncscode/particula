# Implementation Tasks

1. [x] **Issue #1302:** In `particula/gpu/kernels/condensation.py`, extend the
    `condensation_step_gpu()` preflight for partitioning shape/device/binary value
    and every reduction, scale, and accumulator buffer; snapshot all mutable
    buffers in `particula/gpu/kernels/tests/_condensation_test_support.py` to
    prove rejection is pre-mutation. Conversion coverage in
    `particula/gpu/tests/conversion_test.py` validates binary partitioning on
    CPU↔Warp restoration.
2. [x] **Issue #1302:** Add a private Warp launch in `condensation.py` that
    zeros raw transfer for a disabled species or inactive particle before
    application. Preserve `gas.concentration` and the existing enabled-entry
    clamp/accumulator behavior.
3. [x] **Issue #1303:** Add private fp64 evaporation-bound, ordered
    positive/negative reduction, uptake-scale, and finalize/apply launches plus
    a direct-test-only helper. It resolves only P2 sidecars, applies finalized
    particle transfer, and leaves gas read-only.
4. [x] **Issue #1303:** Add an independent NumPy oracle and direct-helper
    atomic-preflight regressions, including multi-box/species isolation,
    mixed-sign funding, supplied-sidecar identity, and unchanged gas.
5. [x] **Issue #1303:** Prove `condensation_step_gpu()` remains P1-only: its
    four-substep launch trace excludes every P2 kernel and sentinel P2
    sidecars remain untouched.
6. Call the limiting/apply launches from each of the four
   `condensation_step_gpu()` substeps and refresh E4-F1/F2/F4 dependent state
   from the newly updated gas.
7. Route the finalized whole-call transfer, not the raw request, to the return
   accumulator and E4-F4 latent-energy accumulator.
8. Add public-production and end-to-end cases to
   `particula/gpu/kernels/tests/condensation_test.py` and the same-change
   conservation regression to
   `particula/integration_tests/condensation_particle_resolved_test.py`.
9. Run focused NumPy-reference and Warp CPU suites, plus CUDA parity when
   available with a clean skip otherwise.
10. Update roadmap wording only after the production and conservation gates pass.
