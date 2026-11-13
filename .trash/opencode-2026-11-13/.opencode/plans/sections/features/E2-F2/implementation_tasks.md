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

- Done in issue #1189: added `EnvironmentData.n_boxes` based on the validated
  `temperature` axis.
- Done in issue #1189: added `EnvironmentData.copy()` returning independent
  arrays.
- Done in issue #1189: exported `EnvironmentData` in
  `particula/gas/__init__.py`.
- Done in issue #1189: extended
  `particula/gas/tests/environment_data_test.py` with `n_boxes`, copy
  independence, mutation-isolation, retained multi-box/dtype coverage, and
  package-export smoke tests.

## E2-F2-P3: Documentation

- Done in issue #1191: updated migration/docs wording to present
  `EnvironmentData` as the shipped CPU owner of per-box thermodynamic state,
  explicitly excluding simulation volume.
- Done in issue #1191: updated roadmap language so remaining work is scoped to
  `WarpEnvironmentData`, CPU↔GPU conversion, and downstream migration
  boundaries.
- Done in issue #1191: documented current read/mutate boundaries: existing
  processes still consume scalar temperature and pressure until explicit
  migration features wire in the new container.
- Done in issue #1191: performed a lightweight documentation accuracy review
  for the touched roadmap and implementation-task files.
