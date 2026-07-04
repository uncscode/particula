# Overview

Feature `E2-F3` implements the Warp-side representation of the environment
container introduced by `E2-F2`. The shipped work now includes issue `#1192`
(`E2-F3-P1`), which added `WarpEnvironmentData` plus focused schema tests, and
issue `#1193` (`E2-F3-P2`), which added the explicit CPU-to-Warp helper
`to_warp_environment_data` with targeted conversion coverage. Reverse
conversion, package exports, CUDA-parametrized transfer coverage, and broader
transfer documentation remain deferred to later phases of the feature.

## Goals

- Add `WarpEnvironmentData` as the GPU mirror of the CPU `EnvironmentData`
  schema, preserving `(n_boxes,)` temperature and pressure arrays plus
  `(n_boxes, n_species)` saturation-ratio arrays.
- Prove the struct schema on the Warp CPU backend with deterministic
  `warp_types_test.py` coverage for shapes, dtypes, field access, and value
  round trips.
- Add `to_warp_environment_data(data, device="cuda", copy=True)` in
  `particula/gpu/conversion.py` using the same explicit field-by-field transfer
  pattern already used for particle and gas helpers.
- Cover CPU transfer values, shapes, dtypes, invalid-device handling,
  Warp-unavailable behavior, and `copy=True` / `copy=False` semantics in
  `particula/gpu/tests/conversion_test.py`.
- Keep the shipped scope intentionally narrow: no Warp-to-CPU helper, package
  export changes, or CUDA-parametrized transfer coverage were added.

## Motivation

The data-oriented GPU roadmap requires per-box environmental state to travel
alongside gas and particle data before kernels can migrate from scalar
temperature and pressure inputs to batched environment arrays. The landed P1
and P2 work establishes both the Warp-side schema boundary and the first
explicit environment transfer entry point that later `E2-F3` phases and
downstream kernel migration tracks (`E2-F5+`) can build on.

## Parent and sibling context

- Parent epic: `E2` (Data-model and numerical foundations v2).
- Upstream dependency: `E2-F2`, which defines the CPU
  `EnvironmentData` schema and validation boundary.
- Related completed tracks: `E2-F1` established CPU data-container patterns;
  `E2-F2` provides the environment container this feature mirrors.
- Downstream tracks can now consume `to_warp_environment_data` rather than
  performing ad-hoc Warp transfers inside kernels, while reverse conversion and
  public package-export work remain for later phases.
