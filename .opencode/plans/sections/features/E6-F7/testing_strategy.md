# Testing Strategy

Every phase ships implementation and co-located fast tests. Files use the
`*_test.py` convention, the configured coverage threshold remains at least 80%,
and scientific expectations come from hand calculations or an independent
NumPy oracle rather than production helpers.

## Per-Phase Approach

- **P1 (shipped):**
  `particula/dynamics/nucleation/tests/nucleation_strategies_test.py` verifies
  `J=A*C` and `J=K*C^2`, SI conversion, linear/quadratic scaling, inclusive
  bounds, exact zero and saturation gates, strict scalar and record validation,
  overflow ordering, frozen records, abstract-interface behavior, and P4
  export-boundary compatibility.
- **P2 (shipped):**
  `particula/dynamics/nucleation/tests/particle_source_test.py` covers
  one/multi-box and multi-species shared admission, per-box limiting species,
  deterministic equal-ratio ties, exact depletion, zero rate/duration/
  inventory and zero-box inputs, provisional demand, diagnostics, record
  immutability/nonaliasing, package-export absence, schema/physical/overflow
  rejections, bounded-correction failure, and gas nonmutation.
- **P3 (shipped):** `particula/dynamics/nucleation/tests/particle_source_test.py`
  covers capacity, zero/no-op rows, scaling, atomic rejection, immutable P2 and
  finalized records, and absent package exports. It verifies private staging,
  fixed-slot activation/exhaustion behavior, scaled gas treatment, and
  per-box/species conservation without exposing a new public transaction API.
- **P4 (shipped):**
  `nucleation_builders_test.py`, `nucleation_factories_test.py`, and export/
  provenance cases in `nucleation_strategies_test.py` cover exact conversions,
  defaults, provenance, immutable input ownership, strict mapping schemas,
  atomic failed-configuration recovery, source-config validation, fresh factory
  isolation, approved imports, and absent P2/P3 exports.
- **P5 (shipped):** `particula/dynamics/tests/nucleation_runnable_test.py`
  verifies public exports and construction, immutable commit controls, topology
  and duration/substep validation, backing-container identity, gas-only
  nonmutation, equal sequential current-gas feedback, zero-write paths,
  runnable sequencing, and P1/P2/P3 failure boundaries including per-substep
  (not whole-call) atomicity.
- **P6 (shipped):** Test-local independent NumPy `float64` P2/P3 oracles and
  snapshots in `particula/dynamics/nucleation/tests/particle_source_test.py`
  cover multi-box/multi-species potential, admission, representation, limiting
  lanes, exact depletion, repeated current-gas coupling, capacity policies,
  and P2/P3/no-viable-policy failure atomicity. The self-contained
  `particula/integration_tests/nucleation_process_test.py` exercises the same
  concrete P2/P3 boundary over a bounded integration matrix without importing
  unit-test helpers. `particula/dynamics/tests/nucleation_runnable_test.py`
  adds single-box P5 live-gas progression and backing-identity regressions;
  it makes no multi-box runnable claim.
- **P7 (shipped):** `particula/tests/nucleation_docs_test.py` validates public
  imports, example identity/transfer/conservation, navigation, theory and
  deferred-scope statements without hardware. Run the published script, focused
  nucleation suites, and `mkdocs build --strict`.

## Required Invariants

- Without representation scaling, post particle represented mass plus post gas
  mass equals the pre total. With scale `s`, compare final represented totals
  against `s * pre_total` and independently require unchanged intensive
  particle-plus-gas concentration plus the source transfer balance (target
  `rtol=1e-12`, `atol=1e-30`; adjustments require written justification).
- Gas remains finite/nonnegative; admitted events never exceed potential events
  or participating species inventory.
- Zero time, coefficient, precursor, survival, and unsatisfied configured gate
  are exact no-ops with zero diagnostics.
- Rejected calls preserve particles, gas, diagnostics, work buffers, shapes,
  dtypes, and identities.
- Particle mass added equals gas mass removed independently for every species;
  aggregate-only checks are insufficient.

Focused suites are deterministic and fast. E6-F7 makes no stochastic CPU/GPU
sequence or performance claim.
