# Scope

## In scope

- Add `WarpEnvironmentData` to `particula/gpu/warp_types.py`.
- Mirror all numeric fields from the `E2-F2` CPU `EnvironmentData` schema as
  one-dimensional Warp arrays shaped `(n_boxes,)`.
- Add CPU-to-Warp conversion helper with explicit `device` and `copy` controls.
- Add Warp-to-CPU conversion helper with explicit `sync` control.
- Update GPU package exports in `particula/gpu/__init__.py`.
- Add Warp CPU tests for struct creation, conversion, copy behavior, sync
  behavior, and round-trip equality.
- Add optional CUDA-parametrized tests using the existing CUDA availability
  helper; tests must skip cleanly when CUDA is unavailable.
- Update feature and theory documentation with transfer semantics and the
  no-hidden-transfer rule.

## Out of scope

- Do not redesign the CPU `EnvironmentData` schema except for small alignment
  fixes needed to make GPU transfer unambiguous.
- Do not migrate existing condensation or coagulation kernels from scalar
  temperature/pressure arguments in this track.
- Do not introduce implicit transfers inside runnable objects or kernel launch
  wrappers.
- Do not store string metadata or Python-only objects in `WarpEnvironmentData`.
- Do not change precision defaults; use `float64` unless a later precision
  decision changes repository policy.

## Done signal

Environment data round trips pass on the Warp CPU backend, CUDA coverage is
automatically active when available, and code/documentation make transfer points
explicit.
