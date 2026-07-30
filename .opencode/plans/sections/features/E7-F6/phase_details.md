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

- [ ] **E7-F6-P2:** Resolve backend availability before execution with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Resolve runtime, device, process, and capability support before adapter mutation.
  - Files: `particula/execution/availability.py`, E7-F1 context/capability modules
  - Tests: Missing Warp, unavailable CUDA, unknown device, unsupported process, validation order.

- [ ] **E7-F6-P3:** Implement opt-in CPU fallback boundary with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Permit fallback only when explicitly requested and CPU state is authoritative or explicitly restored.
  - Files: `particula/execution/fallback.py`, E7-F1 request/result/context modules
  - Tests: Default rejection, pre-upload fallback, requested/selected backend metadata, resident-state rejection.

- [ ] **E7-F6-P4:** Freeze public exports and experimental API policy with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Publish only errors and user-facing policy types and mark low-level GPU APIs experimental without breaking imports.
  - Files: `particula/execution/__init__.py`, `particula/__init__.py`, `particula/gpu/__init__.py`
  - Tests: Exact `__all__`, allowed imports, forbidden internals, CPU-only import subprocess.

- [ ] **E7-F6-P5:** Add no-silent-fallback integration regressions
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Prove capability and runtime failures never trigger conversion, synchronization, adapter retry, or mutation.
  - Files: `particula/execution/tests/fallback_integration_test.py`
  - Tests: Transfer/sync spies, sentinel adapter failures, state snapshots, explicit-boundary success.

- [ ] **E7-F6-P6:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document errors, fallback decision table, imports, stability levels, and migration expectations.
  - Files: `docs/Features/backend_selection.md`, roadmap/API references, plan status
  - Tests: `mkdocs build --strict` and documentation import/snippet validation.
