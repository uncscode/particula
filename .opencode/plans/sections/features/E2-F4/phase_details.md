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

- [x] **E2-F4-P3:** Clarify vapor-pressure ownership and GPU transfer behavior with tests
  - Issue: `#1199` | Size: S | Status: Landed
  - Goal: Make `vapor_pressure` treatment explicit at the transfer boundary
    without expanding CPU `GasData` ownership.
  - Files: `particula/gpu/conversion.py`, `particula/gpu/warp_types.py`, and
    `particula/gpu/tests/conversion_test.py`.
  - Implemented:
    - `to_warp_gas_data()` now explicitly documents optional caller-supplied
      `vapor_pressure`, required `(n_boxes, n_species)` shape validation, and
      zero-filled GPU allocation when omitted.
    - `from_warp_gas_data()` now explicitly documents intentionally lossy CPU
      restore for GPU-only `vapor_pressure` and the need to read/save it before
      restore.
    - `WarpGasData` keeps `vapor_pressure` documented as GPU-only helper state.
    - No kernel compatibility edits were required.
  - Tests: Coverage now directly asserts valid explicit vapor-pressure
    transfer, omitted zero-fill behavior, invalid-shape `ValueError` cases for
    `(n_species,)`, swapped axes, and mismatched species counts, plus sidecar
    preservation across restore.

- [x] **E2-F4-P4:** Update migration documentation for gas round-trip semantics
  - Issue: `#1200` | Size: XS | Status: Landed
  - Goal: Publish the final tested migration-facing contract without changing
    gas runtime behavior.
  - Files: `docs/Features/particle-data-migration.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, `particula/gas/gas_data.py`.
  - Implemented:
    - Added the migration guide authority table for `name`, `molar_mass`,
      `concentration`, `partitioning`, and `vapor_pressure`.
    - Added explicit round-trip guidance for caller-supplied names,
      placeholder fallback, `bool → int32 → bool` `partitioning`, and
      `(n_boxes, n_species)` vapor-pressure sidecars.
    - Revised roadmap wording so the gas split is described as an intentional,
      tested contract rather than unresolved schema drift.
    - Updated `GasData` docstrings only where wording needed to match the final
      published contract.
    - Kept the phase documentation-sized with no runtime logic changes.
  - Tests: Used `particula/gpu/tests/conversion_test.py` as the contract
    authority and kept documentation wording aligned with its existing
    regression coverage.

## Phase Ordering Notes

- P1 should capture the current contract before P2 or P3 changes semantics.
- P2 proceeded from the P1 audit and is now landed; P3 is now landed as the
  explicit vapor-pressure contract layer built on top of the P2 name and
  `partitioning` restore baseline.
- P4 served as the publication gate after P2 and P3 so downstream kernel and
  docs tracks can now cite the same gas round-trip behavior.
