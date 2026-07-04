# Overview

Feature `E2-F3` implements the Warp-side representation and explicit transfer
helpers for the environment container introduced by `E2-F2`. The shipped work
now includes issue `#1192` (`E2-F3-P1`), which added `WarpEnvironmentData`
plus focused schema tests, issue `#1193` (`E2-F3-P2`), which added the
explicit CPU-to-Warp helper `to_warp_environment_data`, issue `#1194`
(`E2-F3-P3`), which completed the public round-trip surface with
`from_warp_environment_data`, `particula.gpu` exports, round-trip/sync/error
tests, and issue `#1195` (`E2-F3-P4`), which added one device-aware
environment round-trip parity test plus final documentation updates for the
explicit transfer boundary and shape rules.

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
- Add `from_warp_environment_data(gpu_data, sync=True)` so CPU environment
  containers can be reconstructed explicitly from `WarpEnvironmentData`.
- Export `WarpEnvironmentData`, `to_warp_environment_data`, and
  `from_warp_environment_data` through `particula.gpu`.
- Cover round-trip behavior, `sync=True` / `sync=False` usage, and malformed
  schema failures in `particula/gpu/tests/conversion_test.py`.
- Keep the remaining scope intentionally narrow: the shipped work stops at the
  helper/documentation boundary and does not add higher-level runtime
  integration.

## Motivation

The data-oriented GPU roadmap requires per-box environmental state to travel
alongside gas and particle data before kernels can migrate from scalar
temperature and pressure inputs to batched environment arrays. The landed P1,
P2, and P3 work establishes the Warp-side schema boundary, both explicit
transfer directions, and the public helper surface that downstream kernel
migration tracks (`E2-F5+`) can build on.

## Parent and sibling context

- Parent epic: `E2` (Data-model and numerical foundations v2).
- Upstream dependency: `E2-F2`, which defines the CPU
  `EnvironmentData` schema and validation boundary.
- Related completed tracks: `E2-F1` established CPU data-container patterns;
  `E2-F2` provides the environment container this feature mirrors.
- Downstream tracks can now consume the shipped
  `to_warp_environment_data`/`from_warp_environment_data` pair rather than
  performing ad-hoc Warp transfers inside kernels.
- Remaining future work is limited to higher-level runtime integration on top
  of the now-shipped helper surface.
