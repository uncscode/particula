# Infrastructure Reuse

## Container Patterns

- Reuse the mutable dataclass pattern from `particula/gas/gas_data.py` and
  `particula/particles/particle_data.py`:
  - arrays coerced in `__post_init__`;
  - explicit shape checks with clear `ValueError` messages;
  - `n_boxes` property;
  - deep-copy `copy()` method.
- Keep single-box data represented as arrays with shape `(1,)`, not scalars.
- Use `np.float64` for all numeric state.

## Validation Patterns

- Mirror current manual validation in `GasData` for dimensionality and shape.
- Use the repository's value semantics from `validate_inputs` as guidance:
  temperature should be finite and physically positive in Kelvin; pressure and
  humidity/saturation fields should be finite and nonnegative, with upper
  bounds only when the field represents relative humidity rather than a general
  saturation ratio.

## Test Patterns

- Follow `particula/gas/tests/gas_data_test.py` structure with grouped classes
  for valid instantiation, validation errors, properties, and copy behavior.
- Use `pytest.raises(..., match=...)` for focused error-message assertions.
- Keep tests fast, deterministic, and local to the gas module.

## Documentation Patterns

- Update `docs/Features/particle-data-migration.md` to keep the data/behavior
  split guidance current.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` from planned gap toward
  implemented CPU baseline while preserving downstream GPU migration notes.
