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

- [x] Define exact `particula.execution.__all__` and approved top-level re-exports.
- [x] Keep concrete adapters, registries, probes, sidecars, and configs non-public.
- [x] Document supported experimental `particula.gpu` APIs without import-time
  warnings or removal of existing direct imports.
- [x] Add `docs/Features/backend_selection.md` and cross-links that distinguish
  the frozen value API from concrete-only availability/fallback and resident
  mechanics, without representing the guide as an automatic runtime path.

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
- [x] Add exact export allow/deny tests modeled on `kernel_exports_test.py`.
- [x] Add P6 documentation regressions in
  `particula/tests/execution_selection_docs_test.py` for the ordered public
  surface, reason table, resolver/no-movement rules, executable public-only
  selection fence, guarded resident pseudocode, and Markdown links.
- [x] Add P5 integration spies in
  `particula/execution/tests/fallback_integration_test.py` for rejected P2
  requests and explicit P3 CPU fallback: conversion, transfer, synchronization,
  kernel, mutation, checkpoint/restore, CPU/GPU adapter calls, state identity,
  provenance, and terminal adapter-error propagation.
- [x] Run focused execution/documentation tests and `mkdocs build --strict`.
