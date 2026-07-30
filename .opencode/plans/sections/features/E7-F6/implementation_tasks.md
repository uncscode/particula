# Implementation Tasks

## Execution Layer

- [x] Add the typed hierarchy and stable reason fields in `particula/execution/errors.py`.
- [x] Map E7-F1 matrix failures to unsupported process/capability errors without
  string parsing in the concrete availability resolver.
- [x] Add a lazy availability-provider protocol and CPU/Warp providers in
  `particula/execution/availability.py`.
- [x] Validate provider recognition, structural declarations, runtime, device,
  then state in the documented deterministic order.
- [x] Add direct-only `FallbackPolicy` with default-deny `RAISE` and explicit
  `CPU` option in `particula/execution/fallback.py`.
- [x] Resolve explicit fallback only at CPU-authoritative `PRE_UPLOAD` or
  caller-asserted `RESTORED` boundaries, without performing restoration.
- [x] Record requested backend, selected backend, and capability reason in
  fallback resolution/dispatch metadata without changing native result metadata.
- [x] Propagate exceptions raised after adapter invocation begins without retry.

## Public API and Compatibility

- [ ] Define exact `particula.execution.__all__` and approved top-level re-exports.
- [ ] Keep concrete adapters, registries, probes, sidecars, and configs non-public.
- [ ] Add experimental/stability wording to `particula.gpu` module docs without import-time warnings.
- [ ] Document compatibility and deprecation rules; do not remove existing direct imports.

## Tooling / Tests

- [x] Add co-located P1 unit tests in `particula/execution/tests/errors_test.py`
  for the root, all concrete errors, hierarchy, exact rendering, and invalid types.
- [x] Add a P1 subprocess import test with Warp and `particula.gpu` blocked to
  prove the direct taxonomy module remains neutral.
- [x] Add co-located P2 availability contract tests in
  `particula/execution/tests/availability_test.py`, including subprocess import
    neutrality and no execution-seam access.
- [x] Add co-located P3 fallback contract tests in
  `particula/execution/tests/fallback_test.py`, including default-deny identity
  re-raise, accepted-reason selection, authority rejection, import neutrality,
  metadata preservation, and no-retry failure propagation.
- [ ] Add exact export allow/deny tests modeled on `kernel_exports_test.py`.
- [ ] Add conversion, synchronization, adapter-call, and mutation spies for rejected requests.
- [ ] Run focused tests, Ruff, mypy for the execution package, and `mkdocs build --strict`.
