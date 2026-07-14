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
3. Add private fp64 launches for per-particle evaporation bounds and separate
   positive/negative `(n_boxes, n_species)` transfer reductions, using the
   caller-owned fixed-shape scratch established by E4-F3.
4. Add the gas-limited positive-scale launch using current gas inventory plus
   permitted evaporation; keep launches 2--4 and their orchestration to roughly
   100 production LOC before tests.
5. Add the finalized-transfer apply launch so particle gain and gas loss are
   opposite aggregates, then assert finite nonnegative postconditions in the
   kernel tests.
6. Call the limiting/apply launches from each of the four
   `condensation_step_gpu()` substeps and refresh E4-F1/F2/F4 dependent state
   from the newly updated gas.
7. Route the finalized whole-call transfer, not the raw request, to the return
   accumulator and E4-F4 latent-energy accumulator.
8. Add per-launch and end-to-end cases to
   `particula/gpu/kernels/tests/condensation_test.py` and the same-change
   conservation regression to
   `particula/integration_tests/condensation_particle_resolved_test.py`.
9. Run focused NumPy-reference and Warp CPU suites, plus CUDA parity when
   available with a clean skip otherwise.
10. Update roadmap wording only after the production and conservation gates pass.
