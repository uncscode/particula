# Overview

Feature `E2-F3` implements the Warp-side representation of the environment
container introduced by `E2-F2`. The shipped work for issue `#1192`
(`E2-F3-P1`) adds `WarpEnvironmentData` plus focused schema tests, while
conversion helpers, package exports, and broader transfer documentation remain
deferred to later phases of the feature.

## Goals

- Add `WarpEnvironmentData` as the GPU mirror of the CPU `EnvironmentData`
  schema, preserving `(n_boxes,)` temperature and pressure arrays plus
  `(n_boxes, n_species)` saturation-ratio arrays.
- Prove the struct schema on the Warp CPU backend with deterministic
  `warp_types_test.py` coverage for shapes, dtypes, field access, and value
  round trips.
- Keep this shipped slice intentionally narrow: no conversion helpers, package
  export changes, or CUDA-parametrized transfer coverage were added.

## Motivation

The data-oriented GPU roadmap requires per-box environmental state to travel
alongside gas and particle data before kernels can migrate from scalar
temperature and pressure inputs to batched environment arrays. The landed P1
work establishes the Warp-side schema boundary that later `E2-F3` phases and
downstream kernel migration tracks (`E2-F5+`) can build on.

## Parent and sibling context

- Parent epic: `E2` (Data-model and numerical foundations v2).
- Upstream dependency: `E2-F2`, which defines the CPU
  `EnvironmentData` schema and validation boundary.
- Related completed tracks: `E2-F1` established CPU data-container patterns;
  `E2-F2` provides the environment container this feature mirrors.
- Downstream tracks will eventually consume explicit conversion helpers rather
  than performing ad-hoc Warp transfers inside kernels, but those helpers were
  not part of issue `#1192`.
