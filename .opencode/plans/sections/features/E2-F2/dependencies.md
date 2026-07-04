# Dependencies

## Internal Dependencies

- **E2-F1/T1:** Required schema foundation. Implementation should align field
  names, positive/nonnegative semantics, and package location with T1 decisions.
- `particula/gas/gas_data.py`: primary container pattern for dataclass shape
  validation, dtype coercion, properties, and `copy()`.
- `particula/particles/particle_data.py`: multi-box container precedent.
- `particula/gas/tests/gas_data_test.py`: test layout and validation style.
- `particula/util/validate_inputs.py`: shared value-validation vocabulary.
- `docs/Features/particle-data-migration.md` and
  `docs/Features/Roadmap/data-oriented-gpu.md`: documentation targets.

## Sibling Feature Dependencies

- GPU mirror/conversion features depend on this CPU schema but are not required
  to complete this feature.
- Kernel and process migration tracks should not be blocked from planning, but
  they should not change this feature's acceptance criteria.
- E2-F3 and E2-F5 should treat `E2-F2-P2` as the implementation handoff because
  public exports and `copy()` semantics are part of the CPU contract they mirror.
- E2-F8 and E2-F9 should consume `E2-F2-P3` documentation wording so support
  boundaries and user guides do not get ahead of the shipped CPU API.

## External Dependencies

- NumPy for array coercion and dtype handling.
- Pytest for tests.
- Ruff for formatting and lint validation.

## Dependency Risks

- If E2-F1 chooses different humidity/saturation naming or bounds, P1 must
  adopt those decisions before tests are finalized.
- Existing scalar dynamics APIs may tempt scope creep; keep those migrations in
  downstream sibling features.

## Sequencing Notes

- P1 establishes field names and validation rules.
- P2 should follow P1 without re-opening schema ownership unless E2-F1 changes.
- P3 is documentation-only and should trail the accepted P1/P2 API so downstream
  GPU and support-boundary tracks cite one stable environment contract.
