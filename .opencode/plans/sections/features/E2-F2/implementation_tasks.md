# Implementation Tasks

## E2-F2-P1: Fields and Validation

- Create `particula/gas/environment_data.py` with a module docstring describing
  per-box thermodynamic environment state.
- Define `EnvironmentData` as a mutable dataclass.
- Add fields for `temperature`, `pressure`, and `saturation_ratio` using
  `NDArray[np.float64]` type hints.
- Coerce inputs with `np.asarray(..., dtype=np.float64)` in `__post_init__`.
- Validate one-dimensional `(n_boxes,)` shape for `temperature` and `pressure`.
- Validate two-dimensional `(n_boxes, n_species)` shape for
  `saturation_ratio`, including finite nonnegative values that may exceed `1.0`.
- Add focused `ValueError` messages for dimensionality, length, and invalid
  numeric values.
- Add tests for valid single-box input and invalid shapes/values.

## E2-F2-P2: Container API and Exports

- Add `n_boxes` property based on the per-box arrays.
- Add `copy()` returning independent arrays.
- Export `EnvironmentData` in `particula/gas/__init__.py`.
- Add multi-box, dtype coercion, copy independence, and export smoke tests.
- Run `ruff` formatting/checks on touched Python files and scoped pytest for
  gas tests.

## E2-F2-P3: Documentation

- Update data-container migration docs to include `EnvironmentData` as the
  owner of per-box thermodynamic state, explicitly excluding simulation volume.
- Update the GPU roadmap docs to note the CPU baseline exists and that GPU
  mirror/conversion work remains downstream.
- Document current read/mutate boundaries: existing processes still consume
  scalar temperature and pressure until explicit migration features wire in the
  new container.
- Validate documentation references if the repository's docs validation tooling
  is available.
