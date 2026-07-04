# Scope

## In Scope

- Add a CPU `EnvironmentData` dataclass for per-box thermodynamic state.
- Store and validate `temperature`, `pressure`, and humidity/saturation state
  as one-dimensional `np.float64` arrays shaped `(n_boxes,)`.
- Coerce array-like inputs to `np.float64` in `__post_init__` following
  `GasData` and `ParticleData` conventions.
- Provide `n_boxes` and `copy()` behavior consistent with existing containers.
- Export the container from the gas package if implementation location is
  `particula/gas/environment_data.py`.
- Add single-box, multi-box, invalid-shape, invalid-value, dtype, and copy
  independence tests in module-local `tests/` files.
- Update feature documentation to explain how processes should read and mutate
  environment state.

## Out of Scope

- Adding `temperature`, `pressure`, or humidity fields to `GasData`.
- Implementing `WarpEnvironmentData` or CPU/GPU transfer helpers; those belong
  to downstream sibling features.
- Migrating condensation, coagulation, parcel, or wall-loss process signatures
  from scalar `temperature`/`pressure` to `EnvironmentData`.
- Changing numerical kernels or validating multi-box physics behavior.
- Introducing a standalone testing-only phase; tests ship with the phases that
  add or document behavior.

## Assumptions

- E2-F1 establishes naming and schema conventions used by this feature.
- Humidity/saturation state is per-box, not per-species, unless E2-F1 specifies
  otherwise before implementation starts.
- Existing scalar process APIs remain supported until sibling migration tracks
  wire in `EnvironmentData` explicitly.
