# Documentation Updates

## Required updates

- `particula/gpu/warp_types.py`: update the module docstring so it covers
  environment containers alongside particle and gas containers.
- `particula/gpu/warp_types.py`: add the `WarpEnvironmentData` class docstring
  describing `temperature`, `pressure`, and `saturation_ratio` plus their
  `(n_boxes,)` / `(n_boxes, n_species)` shape semantics.
- `particula/gpu/conversion.py`: add the `to_warp_environment_data` docstring
  describing the explicit CPU-to-Warp transfer boundary, `device`, `copy`, and
  the shared `RuntimeError` failure modes.
- `particula/gpu/conversion.py`: document
  `from_warp_environment_data(gpu_data, sync=True)` and its manual-sync caveat
  for `sync=False`.
- `particula/gas/environment_data.py`: remove the outdated note that the
  environment container does not yet have CPU↔GPU helpers.
- Repository docs should name the shipped helper surface in `particula.gpu` and
  describe the now-supported environment CPU↔GPU round trip.

## Content to include

- The shipped work now includes helper-level code documentation for both
  `to_warp_environment_data` and `from_warp_environment_data`.
- Repository docs were updated in `readme.md`, `docs/index.md`,
  `docs/Features/particle-data-migration.md`, `docs/Features/Roadmap/data-oriented-gpu.md`,
  and `AGENTS.md` to reflect the shipped environment round-trip helpers.
- Environment string metadata, if any is added later, remains CPU-only unless a
  separate GPU representation is designed.
- Future docs can build on the now-stable field list without revisiting names or
  shape conventions.

## Documentation examples

Examples can now import `WarpEnvironmentData`, `to_warp_environment_data`, and
`from_warp_environment_data` directly from `particula.gpu`.
