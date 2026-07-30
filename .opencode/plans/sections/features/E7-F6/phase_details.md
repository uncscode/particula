# Phase Details

- [x] **E7-F6-P1:** Define execution capability error taxonomy with unit tests
  - Issue: #1500 | Size: S | Status: Implemented (`d1a000769`)
  - Goal: Added stable typed exceptions with deterministic messages and structured context.
  - Files: `particula/execution/errors.py`, `particula/execution/tests/errors_test.py`
  - Tests: Exact direct-module exports/reason codes, root and concrete fields,
    hierarchy, deterministic messages, invalid types, chaining, and subprocess
    import neutrality with Warp and `particula.gpu` blocked.
  - Scope retained: no package/top-level exports, availability resolution,
    fallback selection, GPU behavior, or user documentation changed.

- [x] **E7-F6-P2:** Resolve backend availability before execution with unit tests
  - Issue: #1501 | Size: S | Status: Implemented
  - Goal: Resolve valid P1 request/matrix metadata fail-closed before adapter
    mutation in recognition, declarations, runtime, device, and state order.
  - Files: `particula/execution/availability.py`,
    `particula/execution/tests/availability_test.py`
  - Tests: Dependency-free provider/import fakes cover canonical CPU,
    opaque-native lazy Warp, typed failure mapping, malformed injected seams,
    exact short-circuit ordering, immutable request-only decisions, and no
    adapter/transfer/synchronization/mutation/launch access.
  - Scope retained: no adapter selection, fallback, package/top-level export,
    transfer, synchronization, or execution-state mutation.

- [x] **E7-F6-P3:** Implement opt-in CPU fallback boundary with unit tests
  - Issue: #1502 | Size: S | Status: Implemented
  - Goal: Provide a default-deny, direct-import-only CPU fallback boundary that
    dispatches only for explicit eligible capability errors and exact
    CPU-authoritative state at pre-upload or caller-asserted restored boundaries.
  - Files: `particula/execution/fallback.py`,
    `particula/execution/tests/fallback_test.py`,
    `docs/Features/data-containers-and-gpu-foundations.md`
  - Tests: All five eligible reasons, default identical re-raise, one lookup and
    one dispatch, authority/boundary rejection before lookup, provenance
    metadata with unchanged native metadata, import neutrality, and propagation
    of adapter/result failures without retry or recovery.

- [ ] **E7-F6-P4:** Freeze public exports and experimental API policy with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Publish only errors and user-facing policy types and mark low-level GPU APIs experimental without breaking imports.
  - Files: `particula/execution/__init__.py`, `particula/__init__.py`, `particula/gpu/__init__.py`
  - Tests: Exact `__all__`, allowed imports, forbidden internals, CPU-only import subprocess.

- [x] **E7-F6-P5:** Add no-silent-fallback integration regressions
  - Issue: #1504 | Size: S | Status: Implemented
  - Goal: Prove unavailable-device rejection reaches no execution/movement seam,
    and explicit CPU fallback preserves provenance/state without retry.
  - Files: `particula/execution/tests/fallback_integration_test.py`
  - Tests: Local provider phase logs and P1 snapshots; CPU/GPU adapter and
    conversion/transfer/synchronization/kernel/mutation/checkpoint/restore
    spies; explicit pre-upload CPU fallback provenance/state preservation; and
    unchanged propagation of a terminal CPU adapter sentinel.

- [ ] **E7-F6-P6:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document errors, fallback decision table, imports, stability levels, and migration expectations.
  - Files: `docs/Features/backend_selection.md`, roadmap/API references, plan status
  - Tests: `mkdocs build --strict` and documentation import/snippet validation.
