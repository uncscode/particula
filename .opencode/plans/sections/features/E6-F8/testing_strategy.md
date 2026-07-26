# Testing Strategy

Every production phase includes co-located fast tests in the same change. Test
files use `*_test.py`; the configured threshold remains at least 80% and must
never be lowered. Scientific expectations use hand calculations or an
independent E6-F7 float64 oracle, never the production GPU helper itself.

## Per-Phase Approach

- **P1:** `particula/gpu/kernels/tests/nucleation_test.py` covers config and
   sidecar shape/dtype/device/overlap rules, scientific validation order,
   invalid counts, no-op gates, and byte-for-byte preflight snapshots. This
   suite shipped with #1438; it is Warp-guarded, uses Warp CPU fixtures, and
   verifies that valid preflight, gates, and rejections do not mutate caller
   state or stale sidecars. P1 has no rate or output-write assertions because
   those operations are deferred.
- **P2:** Shipped co-located fixtures compare survival-included `J=A*C` and
   `J=K*C^2`, potential/accepted demand, each limiting species and lowest-index
   ties, planned removal, and diagnostics with an independent float64 oracle.
   They cover one/many boxes and species, gate precedence, zero capacity and
   empty boxes, ULP-safe admission correction, and snapshots proving that P2
   leaves particle/gas state and P3-owned sidecars unchanged.
- **P3 (#1440):** Shipped co-located tests cover all-free, sparse/mixed,
  exact-capacity, and over-capacity layouts; zero boxes and zero capacity;
  retained full integer counts; ascending E6-F5 free indices; selected prefixes;
  and `-1` tails. They independently calculate layouts, verify the supplied
  sidecars reach E6-F5, reject malformed/aliased/wrong-device sidecars and
  invalid conversion products without writes, and cover the inclusive int32
  maximum plus E6-F5 slot-validation preservation. Warp CPU is the baseline;
  CUDA rows skip cleanly when unavailable.
- **P4 (#1441):** Shipped co-located Warp tests use an independent NumPy policy
  oracle for required release, resampling precedence, scaling fallback, final
  demand/counts, and ascending `-1`-tailed prefixes. They cover free/zero-demand,
  resampling-only, scaling-only, mixed-box, boundary, and no-policy cases;
  exact-bool and P4/nested schema/identity/alias validation; stale P2/P3
  handoffs; insufficient scratch; and invalid final demand products. Expected
  all-box rejections snapshot particles, gas, P2/P3/P4, and nested scratch;
  separate coverage documents the entered-primitive failure boundary without
  claiming cross-primitive rollback. Warp CPU is the baseline and CUDA skips
  cleanly when unavailable.
- **P5 (#1442):** Shipped entry-point tests cover supplied-buffer/container
  identity, nominal and multi-box final commits, repeated current-gas calls,
  direct/environment inputs, P4 resampling/scaling integration, no-work paths,
  malformed or rebound P5 handoffs, and all-box precommit atomicity. Export
  tests verify lazy resolution, while boundary guards cover no hidden host
  conversion or CPU fallback.
- **P6 (#1443):** Shipped 718-line
  `particula/gpu/kernels/tests/nucleation_parity_test.py` uses independent
  NumPy float64 P2/P3/P4/P5 expectations (without production planning or
  orchestration helpers). Warp CPU coverage includes activation/kinetic modes,
  multiple species/boxes, inventory limits, sparse/full slots, resampling,
  scaling, exact no-ops, repeated current-gas calls, preflight preservation, and
  zero-box/zero-capacity boundaries. CUDA is optional and skips cleanly when
  unavailable.
- **P7 (#1444):** `particula/tests/nucleation_docs_test.py` keeps publication,
  links, import-boundary, equation, exclusion, and command assertions free of
  Warp execution. `particula/gpu/tests/gpu_direct_nucleation_example_test.py`
  defers Warp imports, cleanly skips without Warp, executes the direct example
  in process for identity/schema/one-slot/gas-depletion and unscaled inventory
  checks, and separately validates its documented `python -Werror` subprocess
  command. The existing kernel and parity suites remain the physics evidence.

## Required Invariants

- Without scaling, per-box/species represented particle plus gas mass is
  conserved. Scaling separately verifies that existing particle inventory is
  multiplied by `s`, gas is unchanged by P4, and P5 represented source inventory
  equals its gas removal; final inventory is `s * initial_particle + initial_gas`.
  Aggregate-only checks are insufficient; mass/gas checks use
  `rtol=1e-12`, `atol=1e-30`.
- Potential/admitted events and deterministic outputs match the CPU oracle at
  recorded float64 tolerances; gas remains finite and nonnegative.
- Zero time, coefficient, precursor, survival, and unsatisfied configured gates
  are exact no-ops with exact zero diagnostics where specified by E6-F7.
- Rejected calls preserve particles, gas, volume, diagnostics, requests,
  scratch/work buffers, shapes, dtypes, devices, and identities.
- Fixed capacity never causes silent truncation. Supplied sidecars are returned
  or retained by identity, and no test permits a hidden host fallback.

## P7 validation

Run `pytest particula/tests/nucleation_docs_test.py -q -Werror`,
`pytest particula/gpu/tests/gpu_direct_nucleation_example_test.py -q -Werror`,
the focused nucleation kernel/parity suites, and `mkdocs build --strict`. Warp
CPU is the baseline and CUDA remains an optional clean-skip row.
