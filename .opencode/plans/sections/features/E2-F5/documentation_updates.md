# Documentation Updates

## Required Updates

- Update kernel docstrings to describe the shipped scalar/direct-array/
  explicit-environment normalization contract.
- Document accepted call forms for condensation and coagulation GPU APIs:
  scalar temperature/pressure, direct `(n_boxes,)` Warp arrays, hybrid
  scalar-plus-array direct inputs, and `WarpEnvironmentData`.
- Document stable early `ValueError` behavior for mixed direct-plus-environment,
  missing direct inputs, shape mismatch, and device mismatch failures.
- Note that temperature and pressure remain environment state, not `GasData`
  fields.

## Files Updated in This Phase

- `particula/gpu/kernels/environment.py` private helper docstrings.
- Docstrings in `particula/gpu/kernels/condensation.py` and
  `particula/gpu/kernels/coagulation.py`.

## Shipped P3 Documentation Outcome

- Issue #1205 did not require further docstring edits because the public
  `condensation_step_gpu(...)` docstring already described the shipped
  scalar/direct-array/hybrid/explicit-environment call forms and fail-fast
  validation behavior.
- P3 documentation work was therefore limited to confirming that code-local
  docs already matched implementation while adding regression coverage for the
  documented contract.

## Shipped P4 Documentation Outcome

- Issue #1206 added the deferred developer-facing roadmap note in
  `docs/Features/Roadmap/data-oriented-gpu.md` rather than changing public
  kernel docstrings.
- The roadmap now states that `coagulation_step_gpu(...)` accepts scalar direct
  `temperature`/`pressure`, direct Warp arrays shaped `(n_boxes,)`, hybrid
  scalar-plus-array direct inputs, or keyword-only
  `environment=WarpEnvironmentData(...)`.
- The same roadmap note now captures the downstream handoff rule for future
  GPU kernels: keep temperature and pressure as validated environment-owned
  state and do not migrate them into `GasData`.

## Deferred Documentation

- Broader user-facing GPU docs remain deferred until later phases need to
  expose more than the code-local entry-point contract and shipped roadmap
  guidance.

## Developer Notes

- Explain scalar compatibility as a migration aid for existing callers.
- Explain that normalization is private to `particula.gpu.kernels` and does not
  expand the public conversion API in this phase.
- Note that P4 documentation scope was intentionally limited to the roadmap
  handoff note because the implementation contract and kernel docstrings were
  already current.
