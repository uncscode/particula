# Implementation Tasks

## E2-F4-P1: Audit Current Contract

- Inspect `GasData`, `WarpGasData`, and gas conversion helpers after `E2-F1`
  lands.
- Add tests that lock down current field ownership and round-trip outcomes.
- Ensure tests include names, molar mass, concentration, partitioning, vapor
  pressure, and invalid vapor-pressure shapes.
- Update docstrings only where they are clearly stale.

## E2-F4-P2: Name and Partitioning Semantics

- Decide the missing-name contract for `from_warp_gas_data()`:
  - require explicit names,
  - preserve current placeholders with a clear warning/docstring, or
  - support metadata sidecar restoration.
- Implement the selected behavior in `particula/gpu/conversion.py`.
- Add tests for supplied names, missing names, mismatched name counts, and
  placeholder/error behavior.
- Add tests that GPU `int32` partitioning returns CPU boolean partitioning and
  invalid or non-binary values behave according to the chosen contract.

## E2-F4-P3: Vapor-Pressure Semantics

- Decide whether vapor pressure is intentionally discarded on CPU return,
  returned through a sidecar/helper, or added to a separate environment-derived
  transfer structure.
- Keep `to_warp_gas_data()` shape validation explicit for
  `(n_boxes, n_species)` buffers.
- Replace surprising zero defaults with an explicit behavior if the audit finds
  that silent defaults are unsafe.
- Add tests for supplied, missing, invalid-shape, and round-trip vapor-pressure
  behavior.

## E2-F4-P4: Documentation

- Update migration docs with a field-by-field CPU/GPU authority table.
- Update roadmap notes so known schema drift is described as intentional or
  resolved.
- Include migration guidance for users with name-keyed downstream logic.
- Include guidance for computing and passing vapor pressure to GPU kernels.
