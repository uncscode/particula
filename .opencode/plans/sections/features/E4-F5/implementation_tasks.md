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
6. [x] **Issue #1304:** Run P2 finalization/application, finalized-total
   accumulation, and deterministic weighted gas coupling in every fixed public
   substep; later transfer proposals, not vapor-pressure refresh, read coupled
   gas.
7. [x] **Issue #1304:** Perform aggregate primary-state/metadata/ownership
   preflight and resolve scratch once per successful call; preserve raw-work and
   later-substep partial-failure semantics.
8. [x] **Issue #1304:** Route only finalized whole-call transfer to the returned
   total and E4-F4 energy output, retaining caller-buffer identity.
9. [x] **Issue #1304:** Add focused Warp wrapper/support regressions for oracle
   coupling, sequence/order, atomic preflight, fresh-proposal failure, and
   scratch reuse.
10. Run P4 production-hook/integration and backend-parity evidence.
11. Update roadmap wording only after remaining production gates pass.
