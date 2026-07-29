# Testing Strategy

Every phase ships tests in the same PR under `particula/execution/tests/` using
the `*_test.py` suffix. Repository and changed-module coverage thresholds are
never lowered; changed modules target at least 80%.

- **P1 (completed, #1492):**
  `particula/execution/tests/process_graph_test.py` covers frozen declarations
  and enums; exact tuple/frozenset contracts; all ten catalogue rows; duplicate
  and unknown IDs; requirements; allowed/disallowed, malformed, duplicate, and
  missing-endpoint dependencies; deterministic normalization and cycle text;
  retry after rejection; and a guarded import proving no Warp/GPU backend load.
  The focused warning-clean and changed-module-coverage commands are recorded
  in the implementation specification.
- **P2 (completed, #1493):** `process_graph_test.py` covers lexical canonical
  topology, endpoint validation, and effective-cycle rejection.
  `scheduler_test.py` covers immutable/exact-type records, P1-first validation,
  selection and predecessor closure, derived freshness edges, both reviewed
  nucleation/condensation directions, registration-order invariance, and guarded
  imports proving no optional backend/lifecycle/resource/adapter load. Both
  modules are declaration-only and perform no launch or mutation.
- **P3:** Adapter contract tests spy on exact direct calls and assert state,
  sidecar, output, and RNG identity; cover no-op/rejection/failure paths and no
  conversion, synchronization, fallback, or private-kernel bypass.
- **P4 (completed, #1495):**
  `particula/execution/tests/state_updates_test.py` uses lazy Warp fixtures to
  cover exact carrier/session/registry/graph/node binding; canonical node roles;
  schema, device, contiguity, identity and nonempty byte-range alias rejection;
  finite positive environment and finite nonnegative gas values; ordered
  in-place copies; protected-field preservation; rejected-call write freedom;
  and empty-box/zero-species write-free no-ops. Import isolation preserves the
  ten-name `particula.execution.__all__`; tests make no scheduler, refresh,
  transfer, fallback, or lifecycle claim.
- **P5 (completed, #1496):**
  `particula/execution/tests/thermodynamic_updates_test.py` uses lazy Warp
  resident fixtures and resolver-produced graph/schedule records. It covers
  successful environment/gas reporting and cursor order; vapor then saturation
  writer/callback ordering and stale-writer elision; condensation-to-diagnostics
  gas visibility; one- and multi-box/species SI saturation references; and
  write-free empty dimensions. Rejection coverage includes invalid callback and
  out-of-order reporting. Separate vapor-writer, saturation-writer, and
  callback failure rows assert callback suppression, unchanged cursor, and the
  defined partial-failure stale-marker state. The tests retain direct-module
  scope and make no lifecycle, full-scheduler, transfer, or fallback claim.
- **P6 (completed, #1497):** `scheduler_test.py` retains the declaration-only
  scheduler blocked-import regression and adds lazy-Warp complete-loop coverage:
  closed diagnostics/output rejection and empty no-ops; exact resolved order;
  one-token lifecycle; consumer-window freshness/current snapshots; stable
  identities; read-only abort and writer/diagnostics-failure faulting without
  rollback claims; no transfer/sync/checkpoint/fallback; deterministic loops,
  conservation/loss, isolation, persistent RNG identity, Warp CPU, and
  skip-clean CUDA. Combined diagnostics/scheduler coverage is at least 80%.
- **P7:** Validate import/export wording, links, examples, and strict MkDocs.

Deterministic fields use explicit `rtol`/`atol` recorded by the owning process.
Stochastic coagulation/wall-loss checks use persistent stream contracts,
invariants, or aggregate statistics—not exact CPU/CUDA trajectory equality.
Focused commands include `pytest particula/execution/tests -q -Werror`, existing
GPU process-sequence/export tests, Ruff, mypy, and `mkdocs build --strict`.
