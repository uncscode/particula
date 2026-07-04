# Implementation Tasks

## E2-F2-P1: Fields and Validation

- Done in issue #1188: created `particula/gas/environment_data.py` with a
  module docstring describing per-box thermodynamic environment state.
- Done in issue #1188: defined `EnvironmentData` as a mutable dataclass with
  `temperature`, `pressure`, and `saturation_ratio` typed as
  `NDArray[np.float64]`.
- Done in issue #1188: coerced constructor inputs with
  `np.asarray(..., dtype=np.float64)` via a dedicated helper in `__post_init__`.
- Done in issue #1188: validated one-dimensional `(n_boxes,)` shape for
  `temperature` and `pressure`.
- Done in issue #1188: validated two-dimensional `(n_boxes, n_species)` shape
  for `saturation_ratio`, including finite nonnegative values that may exceed
  `1.0`.
- Done in issue #1188: added focused `ValueError` messages for dimensionality,
  box-count mismatch, non-finite values, and invalid physical bounds.
- Done in issue #1188: added tests for valid single-box and multi-box input,
  list/tuple coercion, direct-module import, and invalid shapes/values.

## E2-F2-P2: Container API and Exports

- Add `n_boxes` property based on the per-box arrays.
- Add `copy()` returning independent arrays.
- Export `EnvironmentData` in `particula/gas/__init__.py`.
- Reuse the already-shipped multi-box and dtype coercion coverage; add only the
  new copy/export-specific tests needed for this phase.
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
