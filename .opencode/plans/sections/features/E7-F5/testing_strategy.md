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
- **P5:** Call-order tests prove temperature updates precede on-device vapor
  pressure and saturation refresh, and refresh precedes condensation. Cover
  unchanged state, multi-species/multi-box state, and no host evaluation.
- **P6:** Integration-test repeated complete timesteps on Warp CPU using the
  E6-F9 fixtures. Assert one setup, no intermediate bulk transfer or sync,
  stable identities, one lifecycle increment, current gas visibility,
  particle-plus-gas conservation, deterministic call order, and session faulting.
  Independent boxes match equivalent one-box runs when transport is absent.
  CUDA rows are optional and skip cleanly.
- **P7:** Validate import/export wording, links, examples, and strict MkDocs.

Deterministic fields use explicit `rtol`/`atol` recorded by the owning process.
Stochastic coagulation/wall-loss checks use persistent stream contracts,
invariants, or aggregate statistics—not exact CPU/CUDA trajectory equality.
Focused commands include `pytest particula/execution/tests -q -Werror`, existing
GPU process-sequence/export tests, Ruff, mypy, and `mkdocs build --strict`.
