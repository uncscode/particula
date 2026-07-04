# Dependencies

## Internal dependencies

- `E2-F2`: required. It defines the CPU `EnvironmentData` schema,
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

## Sequencing notes

- `E2-F3-P1` should wait for `E2-F2-P1` so the Warp struct mirrors the accepted
  `temperature` / `pressure` / `saturation_ratio` field list instead of a draft.
- `E2-F3-P2` and `E2-F3-P3` should stage behind `E2-F2-P2` because the CPU
  import path, `n_boxes`, and copy semantics are part of the contract these
  helpers mirror.
- `E2-F3-P4` should follow the tested `P2`/`P3` helper names so roadmap and
  migration docs point to one stable transfer surface.
- E2-F5 should consume the helper names and shape guarantees published by
  `E2-F3-P2`/`P3`; this feature should not wait on E2-F5, which avoids a
  reverse dependency cycle.

## Downstream consumers

- E2-F5 reuses the environment transfer contract when normalizing per-box kernel
  inputs.
- E2-F9 documents the final helper names, import paths, and shape semantics only
  after `E2-F3-P4` publishes them.
- Later environment-aware GPU kernel migration tracks.
- Documentation and examples that demonstrate explicit transfer boundaries.
