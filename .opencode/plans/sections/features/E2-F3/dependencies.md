# Dependencies

## Internal dependencies

- `E2-F2` / `T2`: required. It defines the CPU `EnvironmentData` schema,
  validation rules, and public import path.
- `E2-F1`: provides the broader data-container conventions used by this track.
- Existing GPU transfer modules:
  - `particula/gpu/warp_types.py`
  - `particula/gpu/conversion.py`
  - `particula/gpu/__init__.py`
- Existing Warp tests under `particula/gpu/tests/`.

## External dependencies

- `warp-lang` / `warp`: optional runtime dependency used by GPU code and tests.
- CUDA runtime/driver: optional; only required for CUDA-parametrized coverage.
- `numpy`: required for CPU arrays and assertion helpers.
- `pytest`: required for test gating and parametrization.

## Dependency constraints

- CPU tests must not require CUDA.
- Importing `particula.gpu` without Warp installed must preserve existing
  optional-dependency behavior.
- The feature must not add a hard dependency on Warp to non-GPU modules.

## Downstream consumers

- Later environment-aware GPU kernel migration tracks.
- Documentation and examples that demonstrate explicit transfer boundaries.
