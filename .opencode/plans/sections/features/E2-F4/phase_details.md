# Phase Details

- [x] **E2-F4-P1:** Audit gas schema authority and round-trip expectations with unit tests
  - Issue: `#1197` | Size: S | Status: Landed
  - Goal: Capture the current CPU/GPU gas schema contract in tests before
    changing behavior.
  - Files: `particula/gpu/tests/conversion_test.py`.
  - Implemented: Added focused regression tests that lock current
    `GasData`/`WarpGasData` shapes, dtypes, bool↔`int32` `partitioning`,
    explicit and implicit `vapor_pressure` handling, placeholder-name restore
    behavior, name-length mismatch failures, and GPU-only `vapor_pressure`
    loss on restore.
  - Notes: No production semantics or broader documentation changed in this
    phase.

- [x] **E2-F4-P2:** Make name and partitioning conversion semantics explicit with tests
  - Issue: `#1198` | Size: S | Status: Landed
  - Goal: Ensure `from_warp_gas_data()` behavior for missing names and
    `partitioning` int32-to-bool conversion is intentional and non-surprising.
  - Files: `particula/gpu/conversion.py`,
    `particula/gpu/tests/conversion_test.py`, and nearby wording in
    `particula/gpu/warp_types.py`.
  - Implemented:
    - `from_warp_gas_data()` now documents that caller-supplied ordered names
      are preferred because `WarpGasData` does not store strings.
    - Omitted names, including explicit `name=None`, restore as placeholder
      names `species_0..n-1`.
    - Wrong-length and empty provided name lists fail with explicit
      actual/expected count messaging.
    - `_restore_partitioning_bool()` now acts as the explicit restore gate and
      rejects non-binary `partitioning` values before CPU bool conversion.
  - Tests: Coverage now pins supplied names, placeholder fallback,
    `name=None`, invalid name counts, valid binary `partitioning` restore,
    invalid non-binary failures, vapor-pressure drop behavior, multi-box shape
    preservation, and retry-safe correction paths.

- [ ] **E2-F4-P3:** Clarify vapor-pressure ownership and GPU transfer behavior with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Make `vapor_pressure` treatment explicit as GPU transient input,
    returned sidecar, or documented loss according to the chosen ownership
    decision.
  - Files: `particula/gpu/conversion.py`,
    `particula/gpu/tests/conversion_test.py`,
    `particula/gpu/kernels/condensation.py` if validation messages need
    alignment.
  - Tests: Supplied vapor pressure transfers to Warp with exact shape; invalid
    shapes raise `ValueError`; missing vapor pressure behavior is explicit;
    round-trip preservation or intentional discard is tested.

- [ ] **E2-F4-P4:** Update migration documentation for gas round-trip semantics
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Update development docs with field authority, data loss boundaries,
    placeholder-name behavior, and vapor-pressure ownership for migration
    users.
  - Files: `docs/Features/particle-data-migration.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, relevant code docstrings.
  - Tests: Documentation examples or doctest-like snippets are kept consistent
    with conversion tests; fast test suite still passes.

## Phase Ordering Notes

- P1 should capture the current contract before P2 or P3 changes semantics.
- P2 proceeded from the P1 audit and is now landed; P3 should continue to use
  the explicit P2 name and `partitioning` restore contract as its baseline
  while vapor-pressure ownership remains open.
- P4 is the publication gate and should follow the tested decisions from P2 and P3
  so downstream kernel and docs tracks cite the same gas round-trip behavior.
