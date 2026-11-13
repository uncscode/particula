# Dependencies

## Upstream

- **E7-F1:** typed backend-selection, capability matrix, CPU adapter, ownership.
- **E7-F2:** condensation adapter, thermodynamic sidecars, parity/tolerances.
- **E7-F3:** Brownian coagulation adapter, output and persistent RNG resources.
- **E7-F4:** resident lifecycle, sidecar registry, checkpoint/finalize boundary.
- **E7-F5:** canonical process graph, state refresh ordering, diagnostic hooks.
- **E7-F6:** availability/error taxonomy, explicit fallback, export/stability policy.
- **E7-F7:** prescribed multi-box transport, mixing, volume, and ledger contracts.
- **E7-F8:** stable per-box/process streams, reset, checkpoint/restart semantics.
- Shipped E2 and E6 fixed-shape containers, conversions, direct process kernels,
  activation/exhaustion, nucleation, integrated fixtures, and documentation.
- NumPy/CPU references and Warp. CUDA hardware is optional evidence only.

All E7-F1 through E7-F8 dependency gates must be shipped before E7-F9 can assert
closeout. E7-F9 may prepare fixtures earlier but must not freeze evidence against
draft or conflicting upstream contracts.

## Downstream

- Epic G/issue #1451 closure and user adoption of the public execution system.
- Epic H graph-capture/performance work depends on the stable resident loop and
  validation baseline but is not implemented here.
- Epic I autodiff/optimization depends on stable state semantics but is not
  implemented here.

## Phase Ordering

P1 diagnostics precedes diagnostic use in P3-P6. P2 freezes checkpoint evidence
before P5 restart coverage. P3 proves the base full loop before P4/P5 extend it.
P4 and P5 may proceed in parallel after P3. P6 consumes the proven workflow.
P7 is the final documentation/closeout phase and runs only after P1-P6 pass.
